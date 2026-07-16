"""Shared fixtures and test doubles for the agent-tool suite (US-045).

The agent tools talk to PostgreSQL through :mod:`ml.agent.db` (asyncpg) and to a
few CPU-light ML artifacts. No test here touches a real database, GEE, TiTiler or
an LLM: every external boundary is replaced by a deterministic in-memory double.

Why we cannot use the real settings/pool:

- ``ml.agent.db.get_pool`` calls ``backend.app.core.config.get_settings`` which
  loads the developer's ``.env.local``. That file legitimately carries keys the
  ``Settings`` model declares as ``extra="forbid"`` for *other* tooling, so
  constructing it raises ``ValidationError`` in this environment. Tests therefore
  build a lightweight settings stub and never call ``get_settings``.
- ``session_scoped_conn`` acquires from a live pool. Tools that touch the DB are
  tested by monkeypatching the tool module's ``session_scoped_conn`` symbol with
  :func:`fake_session_scoped_conn`, which yields a :class:`FakeConn` capturing the
  SQL and returning scripted rows.

The doubles mimic the small slice of the asyncpg API the tools actually use:
``execute`` / ``fetch`` / ``fetchrow`` / ``fetchval`` and ``dict``-style record
access (``record["column"]``).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

import pytest

from ml.agent.context import ToolContext

# A second, distinct session id used by the cross-session isolation tests.
SESSION_A = UUID("11111111-1111-1111-1111-111111111111")
SESSION_B = UUID("22222222-2222-2222-2222-222222222222")


class FakeRecord(dict):
    """asyncpg ``Record`` stand-in: a dict supporting ``record["column"]``.

    asyncpg records expose mapping access and ``.get``; a plain ``dict`` already
    provides both, so this subclass exists only to document intent in fixtures.
    """


class FakeConn:
    """Minimal asyncpg ``Connection`` double that records executed SQL.

    Every ``execute`` / ``fetch`` / ``fetchrow`` / ``fetchval`` call appends a
    ``(sql, args)`` tuple to :attr:`calls`, so tests can assert *which* SQL ran
    (e.g. the RLS ``set_config``) and with which bound parameters. Return values
    are scripted per method via the constructor.

    Attributes:
        calls: Ordered log of ``(sql, args)`` for every query issued.
        fetch_rows: Rows returned by :meth:`fetch`.
        fetchrow_row: Row returned by :meth:`fetchrow`.
        fetchval_value: Scalar returned by :meth:`fetchval`.
    """

    def __init__(
        self,
        *,
        fetch_rows: list[FakeRecord] | None = None,
        fetchrow_row: FakeRecord | None = None,
        fetchval_value: Any = None,
    ) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetch_rows = fetch_rows if fetch_rows is not None else []
        self.fetchrow_row = fetchrow_row
        self.fetchval_value = fetchval_value

    async def execute(self, sql: str, *args: Any) -> str:
        """Record a statement (e.g. the RLS ``set_config``) and return a tag."""
        self.calls.append((sql, args))
        return "SELECT 1"

    async def fetch(self, sql: str, *args: Any) -> list[FakeRecord]:
        """Record a query and return the scripted row list."""
        self.calls.append((sql, args))
        return self.fetch_rows

    async def fetchrow(self, sql: str, *args: Any) -> FakeRecord | None:
        """Record a query and return the scripted single row."""
        self.calls.append((sql, args))
        return self.fetchrow_row

    async def fetchval(self, sql: str, *args: Any) -> Any:
        """Record a query and return the scripted scalar value."""
        self.calls.append((sql, args))
        return self.fetchval_value

    def set_config_calls(self) -> list[tuple[str, tuple[Any, ...]]]:
        """Return only the ``set_config`` calls (the RLS priming statements)."""
        return [c for c in self.calls if "set_config" in c[0]]


def fake_session_scoped_conn(conn: FakeConn):
    """Build a drop-in replacement for :func:`ml.agent.db.session_scoped_conn`.

    The returned async context manager mirrors the real helper's contract: it
    primes the per-session RLS hook on ``conn`` by issuing the exact same
    ``set_config('app.current_session', $1, true)`` statement (bound to the
    string form of ``session_id``) and then yields ``conn``. Because the call is
    recorded by :class:`FakeConn`, the RLS-isolation tests can assert the session
    id actually reached the connection.

    Args:
        conn: The :class:`FakeConn` to yield and prime.

    Returns:
        An async-context-manager factory ``(session_id) -> conn``.
    """

    @asynccontextmanager
    async def _scoped(session_id: UUID):
        await conn.execute(
            "SELECT set_config('app.current_session', $1, true)",
            str(session_id),
        )
        yield conn

    return _scoped


class FakeSettings:
    """Lightweight settings stub for :class:`ToolContext`.

    Only the attributes the tools read are provided. ``database_url`` matches the
    dev DSN form so ``to_asyncpg_dsn`` round-trips; ``titiler_host_port`` drives
    the ``get_tiles`` URL builder.
    """

    database_url = "postgresql+asyncpg://agrosat:agrosat@localhost:5432/agrosat"
    titiler_host_port = 8001
    # CDSE credentials default to empty so ``search_stac`` degrades to pgstac
    # unless a test overrides them (no live CDSE call is ever made).
    cdse_client_id = ""
    cdse_client_secret = ""
    cdse_token_url = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    )


@pytest.fixture
def fake_settings() -> FakeSettings:
    """Return a fresh settings stub (never the real ``.env.local`` one)."""
    return FakeSettings()


@pytest.fixture
def make_ctx(fake_settings: FakeSettings):
    """Return a factory building a :class:`ToolContext` for tests.

    The ``pool`` slot is left ``None`` because every DB-touching tool is tested by
    monkeypatching its ``session_scoped_conn`` symbol (the pool is never acquired
    directly).

    Returns:
        A callable ``(session_id=SESSION_A, defer=None, crop_model=None) ->
        ToolContext``. ``crop_model`` mirrors the model the user pinned in the UI,
        which ``classify.run`` enforces over the reasoner's argument.
    """

    def _make(session_id: UUID = SESSION_A, defer=None, crop_model=None) -> ToolContext:
        return ToolContext(
            pool=None,  # type: ignore[arg-type]
            settings=fake_settings,  # type: ignore[arg-type]
            session_id=session_id,
            defer=defer,
            crop_model=crop_model,
        )

    return _make
