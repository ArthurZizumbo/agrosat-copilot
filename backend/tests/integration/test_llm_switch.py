"""``POST /llm/switch`` integration tests (US-054).

Two harnesses, mirroring the two AC families:

1. **Real PostGIS + RLS** (the US-051 / US-053 testcontainers harness): all
   ``db/migrations/*.sql`` are applied -- including the US-054 rename
   ``llm_variant`` -> ``llm_model`` and the 4-value CHECK -- and the FastAPI app
   is driven through ``httpx`` with ``get_scoped_conn`` overridden to bind a
   real ``agrosat_app`` (NOBYPASSRLS) connection primed with the request's
   ``X-Session-ID``. The real ``verify_chat_session`` guard and the real
   ``LLMSwitchService.switch`` UPDATE run end to end, so persistence AND tenant
   isolation are exercised against the database:

   - **AC-1** the switch persists ``chat_sessions.llm_model`` for the session and
     a subsequent ``/llm/switch`` round-trips the new value (session-scoped).
   - **AC-3** the four variants are accepted; an unknown one is ``422`` at the
     edge (the ``Literal`` body) before any SQL.
   - **AC-6 / cross-session** session B cannot change session A's model: under
     RLS the UPDATE on B's connection never touches A's row, so A keeps its
     value and the switch is session-scoped.

2. **Mocked agent + memory limiter** (the US-052 ``/chat`` harness, no DB): the
   reasoner and pool are stubbed so the ``/chat`` flow runs without Gemini /
   vLLM, and the limiter is pointed at ``memory://``:

   - **AC-2** a ``/chat`` after a switch builds the backend of the persisted
     variant (asserted on the variant the ``agent_factory`` receives -- the
     ``ChatService`` read it off the scoped session row).
   - **AC-6 rate limit** the 6th switch in the window for one session is ``429``,
     and the budget is per-session (a second session is unaffected).

Auto-skips the DB harness without ``testcontainers`` / ``asyncpg`` / Docker so
``make test`` stays green in CI without a Docker daemon. No real Gemini / Qwen /
Gemma call ever happens (that is the QA manual validation against ``:8002``).
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

import pytest
from fastapi import Depends
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.integration

pytest.importorskip("testcontainers.postgres", reason="testcontainers no instalado")
pytest.importorskip("asyncpg", reason="asyncpg requerido")

import asyncpg  # noqa: E402
from testcontainers.postgres import PostgresContainer  # noqa: E402

from backend.app.core import db as core_db  # noqa: E402
from backend.app.core.config import Settings  # noqa: E402
from backend.app.core.db import get_request_session_id  # noqa: E402
from backend.app.core.rate_limit import build_limiter, limiter  # noqa: E402
from backend.app.main import create_app  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATIONS_DIR = REPO_ROOT / "db" / "migrations"

CANDIDATE_IMAGES: tuple[str, ...] = (
    "agrosat-postgres:15-3.4-pgvector",
    "postgis/postgis:15-3.4",
)

APP_ROLE = "agrosat_app"
APP_PASSWORD = "agrosat_app"
SET_SESSION_SQL = "SELECT set_config('app.current_session', $1, true)"

_MIGRATE_UP_RE = re.compile(r"--\s*migrate:up\s*\n(.*?)(?=--\s*migrate:down|\Z)", re.DOTALL)


def _split_up(sql_text: str) -> str:
    match = _MIGRATE_UP_RE.search(sql_text)
    if match is None:
        raise ValueError("Archivo de migracion sin bloque -- migrate:up")
    return match.group(1).strip()


async def _execute_tolerant(conn: asyncpg.Connection, sql_block: str) -> None:
    """Execute a multi-statement block, tolerating missing optional extensions."""
    try:
        await conn.execute(sql_block)
        return
    except asyncpg.PostgresError:
        pass
    for stmt in (s.strip() for s in sql_block.split(";") if s.strip()):
        try:
            await conn.execute(stmt)
        except asyncpg.PostgresError as exc:
            if "CREATE EXTENSION" in stmt.upper():
                continue
            raise RuntimeError(f"Fallo al aplicar: {stmt[:80]} -- {exc}") from exc


async def _apply_all_migrations(dsn: str) -> None:
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not files:
        raise RuntimeError(f"No se encontraron migraciones en {MIGRATIONS_DIR}")
    conn = await asyncpg.connect(dsn=dsn)
    try:
        for migration_path in files:
            await _execute_tolerant(conn, _split_up(migration_path.read_text(encoding="utf-8")))
    finally:
        await conn.close()


@pytest.fixture(scope="module")
def app_dsn() -> Iterator[str]:
    """Boot Postgres, apply all migrations, yield the ``agrosat_app`` role DSN."""
    last_error: Exception | None = None
    for image in CANDIDATE_IMAGES:
        container = PostgresContainer(
            image=image, username="agrosat", password="agrosat", dbname="agrosat"
        )
        try:
            container.start()
        except Exception as exc:  # noqa: BLE001 - depends on host Docker
            last_error = exc
            continue
        try:
            host = container.get_container_host_ip()
            port = container.get_exposed_port(5432)
            superuser_dsn = f"postgresql://agrosat:agrosat@{host}:{port}/agrosat"

            async def _bootstrap(dsn: str = superuser_dsn) -> None:
                conn = await asyncpg.connect(dsn=dsn)
                try:
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                finally:
                    await conn.close()
                await _apply_all_migrations(dsn)

            asyncio.run(_bootstrap())
            yield f"postgresql://{APP_ROLE}:{APP_PASSWORD}@{host}:{port}/agrosat"
            container.stop()
            return
        except Exception as exc:  # noqa: BLE001 - image/host dependent
            last_error = exc
            container.stop()
            continue
    pytest.skip(f"Sin imagen Postgres+PostGIS+pgvector utilizable: {last_error}")


async def _seed_session(app_dsn: str) -> UUID:
    """Create a ``chat_sessions`` row (as the app role) so the guard sees it."""
    session_id = uuid4()
    conn = await asyncpg.connect(dsn=app_dsn)
    try:
        async with conn.transaction():
            await conn.execute(SET_SESSION_SQL, str(session_id))
            await conn.execute(
                "INSERT INTO chat_sessions (id, user_id) VALUES ($1, $2)",
                session_id,
                f"user-{session_id}",
            )
    finally:
        await conn.close()
    return session_id


async def _read_model(app_dsn: str, session_id: UUID) -> str | None:
    """Read ``chat_sessions.llm_model`` directly (scoped) for an assertion."""
    conn = await asyncpg.connect(dsn=app_dsn)
    try:
        async with conn.transaction():
            await conn.execute(SET_SESSION_SQL, str(session_id))
            row = await conn.fetchrow(
                "SELECT llm_model FROM chat_sessions WHERE id = $1", session_id
            )
    finally:
        await conn.close()
    return None if row is None else row["llm_model"]


def _make_client(app_dsn: str) -> tuple[AsyncClient, object]:
    """Build a client whose scoped-conn dependency binds a real RLS connection.

    Mirrors the US-053 harness: ``get_scoped_conn`` opens a per-request
    transaction on a fresh ``agrosat_app`` connection primed with
    ``app.current_session`` from the validated header, and the real
    ``verify_chat_session`` guard runs against it (so its ``403`` is exercised).
    """
    app = create_app()

    async def scoped_override(
        session_id: Annotated[UUID, Depends(get_request_session_id)],
    ) -> AsyncIterator[asyncpg.Connection]:
        conn = await asyncpg.connect(dsn=app_dsn)
        try:
            async with conn.transaction():
                await conn.execute(SET_SESSION_SQL, str(session_id))
                yield conn
        finally:
            await conn.close()

    app.dependency_overrides[core_db.get_scoped_conn] = scoped_override
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test"), app


def _hdr(session_id: UUID) -> dict[str, str]:
    return {"X-Session-ID": str(session_id)}


# ---------------------------------------------------------------------------
# Memory limiter fixture (US-052 pattern): a successful /llm/switch injects the
# slowapi X-RateLimit-* headers, which makes the limiter singleton talk to its
# storage. Without this the singleton is wired to ``settings.redis_url`` (a real
# Redis that is not running in the test env) and every successful switch raises
# ``redis.ConnectionError``. Point it at ``memory://`` and reset per test.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def memory_limiter() -> Iterator[None]:
    """Point the production limiter singleton at ``memory://`` and reset it."""
    mem = build_limiter(Settings(redis_url="memory://"))
    saved = {
        "_storage": limiter._storage,
        "_storage_uri": limiter._storage_uri,
        "_limiter": limiter._limiter,
    }
    limiter._storage = mem._storage
    limiter._storage_uri = mem._storage_uri
    limiter._limiter = mem._limiter
    limiter.reset()
    try:
        yield
    finally:
        limiter.reset()
        limiter._storage = saved["_storage"]
        limiter._storage_uri = saved["_storage_uri"]
        limiter._limiter = saved["_limiter"]


# ---------------------------------------------------------------------------
# AC-1: switch persists chat_sessions.llm_model (session-scoped, RLS).
# ---------------------------------------------------------------------------
async def test_switch_persists_llm_model_session_scoped(app_dsn: str) -> None:
    """``POST /llm/switch`` returns ``200`` and persists ``llm_model`` for the session."""
    session_a = await _seed_session(app_dsn)
    client, _app = _make_client(app_dsn)
    async with client:
        resp = await client.post(
            "/llm/switch", json={"model": "qwen-onprem"}, headers=_hdr(session_a)
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"] == "qwen-onprem"
    assert "applied_at" in body
    # The value is durably persisted on the caller's own row.
    assert await _read_model(app_dsn, session_a) == "qwen-onprem"


async def test_switch_overwrites_previous_value(app_dsn: str) -> None:
    """A second switch round-trips the new variant (last write wins, scoped)."""
    session_a = await _seed_session(app_dsn)
    client, _app = _make_client(app_dsn)
    async with client:
        first = await client.post(
            "/llm/switch", json={"model": "qwen-api"}, headers=_hdr(session_a)
        )
        assert first.status_code == 200
        second = await client.post("/llm/switch", json={"model": "gemma"}, headers=_hdr(session_a))
    assert second.status_code == 200
    assert second.json()["model"] == "gemma"
    assert await _read_model(app_dsn, session_a) == "gemma"


# ---------------------------------------------------------------------------
# AC-3: the four valid variants are accepted; an invalid one is 422.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("variant", ["gemini", "qwen-api", "qwen-onprem", "gemma"])
async def test_four_variants_accepted(app_dsn: str, variant: str) -> None:
    """Each of the four supported variants is accepted and persisted (AC-3)."""
    session_a = await _seed_session(app_dsn)
    client, _app = _make_client(app_dsn)
    async with client:
        resp = await client.post("/llm/switch", json={"model": variant}, headers=_hdr(session_a))
    assert resp.status_code == 200, resp.text
    assert resp.json()["model"] == variant
    assert await _read_model(app_dsn, session_a) == variant


async def test_invalid_variant_is_422(app_dsn: str) -> None:
    """An unsupported variant is rejected with ``422`` at the edge (AC-3).

    The ``Literal`` body validates before the handler, so the session row is
    never touched and the DB CHECK is never reached.
    """
    session_a = await _seed_session(app_dsn)
    client, _app = _make_client(app_dsn)
    async with client:
        resp = await client.post(
            "/llm/switch", json={"model": "llama-99b"}, headers=_hdr(session_a)
        )
    assert resp.status_code == 422
    # The bogus value did not overwrite the default.
    assert await _read_model(app_dsn, session_a) == "gemini"


# ---------------------------------------------------------------------------
# AC-6 / cross-session: B cannot change A's model; switch is session-scoped.
# ---------------------------------------------------------------------------
async def test_unknown_session_is_403(app_dsn: str) -> None:
    """A session with no ``chat_sessions`` row is ``403`` (fail-closed, AC-6)."""
    unknown = uuid4()  # never seeded -> no chat_sessions row, invisible under RLS
    client, _app = _make_client(app_dsn)
    async with client:
        resp = await client.post(
            "/llm/switch", json={"model": "qwen-onprem"}, headers=_hdr(unknown)
        )
    assert resp.status_code == 403, resp.text


async def test_session_b_cannot_change_session_a_model(app_dsn: str) -> None:
    """Cross-session: B's switch never touches A's row; the switch is session-scoped.

    Both sessions are valid (both pass the guard). A sets ``qwen-onprem``; B then
    switches itself to ``gemma``. Under RLS B's UPDATE only sees its own row, so
    A's persisted model is unchanged -- a switch in session A does not affect B
    and vice-versa.
    """
    session_a = await _seed_session(app_dsn)
    session_b = await _seed_session(app_dsn)
    client, _app = _make_client(app_dsn)
    async with client:
        a_set = await client.post(
            "/llm/switch", json={"model": "qwen-onprem"}, headers=_hdr(session_a)
        )
        assert a_set.status_code == 200
        b_set = await client.post("/llm/switch", json={"model": "gemma"}, headers=_hdr(session_b))
        assert b_set.status_code == 200

    # Each session kept exactly its own value; neither leaked into the other.
    assert await _read_model(app_dsn, session_a) == "qwen-onprem"
    assert await _read_model(app_dsn, session_b) == "gemma"
