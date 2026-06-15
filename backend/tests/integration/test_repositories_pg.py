"""Integration tests for repositories + agent adapters against real PostGIS.

Spins up an ephemeral PostGIS+pgvector container, applies every ``migrate:up``
block in ``db/migrations`` (in filename order) and exercises the session / AOI /
parcel / chat-message repositories plus the SQL agent adapters end to end.

Auto-skips when Docker or pgvector is unavailable.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATIONS_DIR = REPO_ROOT / "db" / "migrations"

_SQUARE_WKT = "POLYGON((11.10 43.30,11.11 43.30,11.11 43.31,11.10 43.31,11.10 43.30))"


def _migrate_up_blocks() -> list[str]:
    blocks: list[str] = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        text = path.read_text(encoding="utf-8")
        up = text.split("-- migrate:up", 1)[1].split("-- migrate:down", 1)[0]
        blocks.append(up.strip())
    return blocks


async def _apply_migrations(dsn: str) -> None:
    import asyncpg

    conn = await asyncpg.connect(dsn=dsn)
    try:
        for block in _migrate_up_blocks():
            for stmt in (s.strip() for s in block.split(";") if s.strip()):
                try:
                    await conn.execute(stmt)
                except Exception as exc:
                    if "CREATE EXTENSION" in stmt.upper():
                        continue
                    raise RuntimeError(f"failed: {stmt[:80]} ({exc})") from exc
    finally:
        await conn.close()


@pytest.fixture
def pg_dsn():  # type: ignore[no-untyped-def]
    import asyncio

    testcontainers = pytest.importorskip("testcontainers.postgres")
    pytest.importorskip("asyncpg")
    try:
        container = testcontainers.PostgresContainer(
            image="postgis/postgis:15-3.4",
            username="agrosat",
            password="agrosat",
            dbname="agrosat",
        )
        container.start()
    except Exception as exc:  # noqa: BLE001 - any Docker error -> skip
        pytest.skip(f"Docker no disponible: {exc}")
    host = container.get_container_host_ip()
    port = container.get_exposed_port(5432)
    asyncpg_dsn = f"postgresql://agrosat:agrosat@{host}:{port}/agrosat"
    sqlalchemy_dsn = f"postgresql+asyncpg://agrosat:agrosat@{host}:{port}/agrosat"
    try:
        asyncio.run(_apply_migrations(asyncpg_dsn))
    except RuntimeError as exc:
        container.stop()
        pytest.skip(f"migracion fallo (pgvector ausente?): {exc}")
    yield sqlalchemy_dsn
    container.stop()


@pytest.mark.asyncio
async def test_session_aoi_message_roundtrip(pg_dsn: str) -> None:
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from backend.app.services.agent_adapters import SqlChatMemory, SqlParcelReader
    from backend.app.services.aoi_service import AoiService
    from backend.app.services.session_service import SessionService
    from ml.agent.ports import ChatTurn

    engine = create_async_engine(pg_dsn)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as db:
            sessions = SessionService(db)
            created = await sessions.create(user_id="u@x.dev", llm_variant="gemini")
            sid = created.id
            assert isinstance(sid, uuid.UUID)

            # Ownership guard.
            owned = await sessions.get_owned_or_none(session_id=sid, user_id="u@x.dev")
            assert owned is not None
            not_owned = await sessions.get_owned_or_none(session_id=sid, user_id="other@x.dev")
            assert not_owned is None

            # AOI create + list (GeoJSON roundtrip + area).
            aoi_service = AoiService(db)
            geometry = {
                "type": "Polygon",
                "coordinates": [
                    [
                        [11.10, 43.30],
                        [11.11, 43.30],
                        [11.11, 43.31],
                        [11.10, 43.31],
                        [11.10, 43.30],
                    ]
                ],
            }
            view = await aoi_service.create(session_id=sid, geometry=geometry, label="field")
            assert view.area_ha is not None and view.area_ha > 0
            listed = await aoi_service.list_for_session(session_id=sid)
            assert len(listed) == 1
            assert listed[0].geometry["type"] == "Polygon"

        # Chat memory adapter.
        memory = SqlChatMemory(factory)
        await memory.append_turn(session_id=str(sid), turn=ChatTurn(role="user", content="hola"))
        await memory.append_turn(session_id=str(sid), turn=ChatTurn(role="assistant", content="hi"))
        history = await memory.load_history(session_id=str(sid))
        assert [t.role for t in history] == ["user", "assistant"]

        # Parcel reader on an empty parcel set returns [].
        reader = SqlParcelReader(factory)
        parcels = await reader.list_parcels_in_aoi(session_id=str(sid))
        assert parcels == []
    finally:
        await engine.dispose()
