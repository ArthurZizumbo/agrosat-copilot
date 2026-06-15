"""RLS / multi-tenant isolation tests (US-045 AC-3).

The real :func:`ml.agent.db.session_scoped_conn` must, for every tool query,
prime the per-session hook with ``set_config('app.current_session', $1, true)``
bound to the active session id (the parametrised ``SET LOCAL`` equivalent). These
tests exercise the *real* helper, replacing only the pool with an in-memory fake
so the actual SQL it emits is captured and asserted. Two different session ids
must each set their own value -- they never leak across connections.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import UUID

import pytest

import ml.agent.db as db_mod

from .conftest import SESSION_A, SESSION_B, FakeConn


class _FakePool:
    """asyncpg ``Pool`` double handing out a single :class:`FakeConn`.

    ``session_scoped_conn`` calls ``acquire`` / ``release`` and opens a
    transaction on the connection; this fake supplies both, plus a transaction
    context manager that is a no-op.
    """

    def __init__(self, conn: FakeConn) -> None:
        self._conn = conn
        self.released: list[FakeConn] = []

    async def acquire(self) -> FakeConn:
        return self._conn

    async def release(self, conn: FakeConn) -> None:
        self.released.append(conn)


def _install_fake_pool(monkeypatch: pytest.MonkeyPatch, conn: FakeConn) -> _FakePool:
    """Patch ``get_pool`` and the connection transaction so the real helper runs.

    Args:
        monkeypatch: pytest patcher.
        conn: the :class:`FakeConn` the fake pool will hand out.

    Returns:
        The installed :class:`_FakePool` (to inspect ``released``).
    """
    pool = _FakePool(conn)

    async def _get_pool() -> _FakePool:
        return pool

    @asynccontextmanager
    async def _txn():
        yield

    # The real helper wraps the body in ``async with conn.transaction()``; give
    # the fake connection a no-op transaction context manager.
    monkeypatch.setattr(conn, "transaction", _txn, raising=False)
    monkeypatch.setattr(db_mod, "get_pool", _get_pool)
    return pool


async def test_session_scoped_conn_emits_set_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The helper issues the parametrised ``set_config`` with the session id."""
    conn = FakeConn()
    pool = _install_fake_pool(monkeypatch, conn)

    async with db_mod.session_scoped_conn(SESSION_A) as scoped:
        assert scoped is conn

    set_calls = conn.set_config_calls()
    assert len(set_calls) == 1
    sql, args = set_calls[0]
    assert "set_config('app.current_session', $1, true)" in sql
    # Bound (not string-interpolated) and equal to the session id as text.
    assert args == (str(SESSION_A),)
    # The connection is returned to the pool afterwards.
    assert pool.released == [conn]


async def test_two_sessions_do_not_mix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two distinct sessions each set their own ``app.current_session`` value."""
    conn = FakeConn()
    _install_fake_pool(monkeypatch, conn)

    async with db_mod.session_scoped_conn(SESSION_A):
        pass
    async with db_mod.session_scoped_conn(SESSION_B):
        pass

    bound_values = [args[0] for _sql, args in conn.set_config_calls()]
    assert bound_values == [str(SESSION_A), str(SESSION_B)]
    assert str(SESSION_A) != str(SESSION_B)


async def test_set_config_runs_before_yield(monkeypatch: pytest.MonkeyPatch) -> None:
    """RLS priming happens before the body runs any tool query."""
    conn = FakeConn(fetch_rows=[])
    _install_fake_pool(monkeypatch, conn)

    async with db_mod.session_scoped_conn(SESSION_A) as scoped:
        await scoped.fetch("SELECT id FROM parcels WHERE session_id = $1", SESSION_A)

    # The first recorded call must be the RLS set_config, not the SELECT.
    first_sql = conn.calls[0][0]
    assert "set_config" in first_sql


def test_to_asyncpg_dsn_strips_driver() -> None:
    """The DSN helper removes the SQLAlchemy ``+asyncpg`` driver marker."""
    url = "postgresql+asyncpg://u:p@h:5432/db"
    assert db_mod.to_asyncpg_dsn(url) == "postgresql://u:p@h:5432/db"
    # Idempotent on an already-plain DSN.
    assert db_mod.to_asyncpg_dsn("postgresql://u:p@h/db") == "postgresql://u:p@h/db"


def test_session_ids_are_distinct_uuids() -> None:
    """Fixture sanity: the two test sessions are different UUIDs."""
    assert isinstance(SESSION_A, UUID)
    assert SESSION_A != SESSION_B
