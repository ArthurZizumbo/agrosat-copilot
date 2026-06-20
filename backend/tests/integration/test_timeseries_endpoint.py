"""``/aois/{id}/timeseries`` endpoint tests (US-053 AC-2).

Real integration test against a PostGIS + pgvector container (same harness as
``test_aois_crud.py``): all migrations applied, the app driven over
``ASGITransport`` with ``get_scoped_conn`` overridden to bind a real
``agrosat_app`` RLS connection primed from ``X-Session-ID``.

Seeds session A with an AOI, a parcel linked to it, and a ``features_parcels`` row
carrying a real ``peak_doy``/``peak_value`` (the only anchor that pairs a stored
date with a measured value -- see ``ml/agent/tools/timeseries.py``). Asserts:

- ``index=NDVI`` -> exactly one point (the in-window peak), value round-trips.
- ``index=NDWI`` -> empty series (no temporal anchor persisted; honest, not faked).
- session B requesting A's AOI -> ``404`` (RLS hides it).
- ``index=BOGUS`` -> ``422`` (the index ``Literal`` rejects it).

Auto-skip without Docker/testcontainers (mirrors the RLS suite).
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
from backend.app.core.db import get_request_session_id  # noqa: E402
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
POLY_WKT = "POLYGON((11.0 43.0, 11.1 43.0, 11.1 43.1, 11.0 43.1, 11.0 43.0))"

# Year + day-of-year of the seeded NDVI peak; the default endpoint window
# (2017..2100) contains it.
_PEAK_YEAR = 2019
_PEAK_DOY = 196  # mid-July
_PEAK_VALUE = 0.82


def _split_up(sql_text: str) -> str:
    match = _MIGRATE_UP_RE.search(sql_text)
    if match is None:
        raise ValueError("Archivo de migracion sin bloque -- migrate:up")
    return match.group(1).strip()


async def _execute_tolerant(conn: asyncpg.Connection, sql_block: str) -> None:
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
    conn = await asyncpg.connect(dsn=dsn)
    try:
        for migration_path in files:
            await _execute_tolerant(conn, _split_up(migration_path.read_text(encoding="utf-8")))
    finally:
        await conn.close()


@pytest.fixture(scope="module")
def app_dsn() -> Iterator[str]:
    """Boot Postgres, apply migrations, yield the ``agrosat_app`` DSN."""
    last_error: Exception | None = None
    for image in CANDIDATE_IMAGES:
        container = PostgresContainer(
            image=image, username="agrosat", password="agrosat", dbname="agrosat"
        )
        try:
            container.start()
        except Exception as exc:  # noqa: BLE001
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
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            container.stop()
            continue
    pytest.skip(f"Sin imagen Postgres+PostGIS+pgvector utilizable: {last_error}")


async def _seed_session_with_parcel(app_dsn: str) -> tuple[UUID, int]:
    """Seed a session + AOI + parcel + feature row with a real NDVI peak.

    Returns:
        ``(session_id, aoi_id)``.
    """
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
            aoi_id = await conn.fetchval(
                """
                INSERT INTO aois (session_id, geom, label)
                VALUES ($1, ST_GeomFromText($2, 4326), 'ts-aoi') RETURNING id
                """,
                session_id,
                POLY_WKT,
            )
            parcel_id = await conn.fetchval(
                """
                INSERT INTO parcels (session_id, aoi_id, geom, year)
                VALUES ($1, $2, ST_GeomFromText($3, 4326), $4) RETURNING id
                """,
                session_id,
                aoi_id,
                POLY_WKT,
                _PEAK_YEAR,
            )
            await conn.execute(
                """
                INSERT INTO features_parcels (parcel_id, year, peak_doy, peak_value)
                VALUES ($1, $2, $3, $4)
                """,
                parcel_id,
                _PEAK_YEAR,
                _PEAK_DOY,
                _PEAK_VALUE,
            )
        return session_id, int(aoi_id)
    finally:
        await conn.close()


def _make_client(app_dsn: str) -> tuple[AsyncClient, object]:
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


async def test_ndvi_series_has_peak_point(app_dsn: str) -> None:
    """NDVI returns exactly the stored in-window peak (date + value round-trip)."""
    session_a, aoi_id = await _seed_session_with_parcel(app_dsn)
    client, _app = _make_client(app_dsn)
    async with client:
        resp = await client.get(
            f"/aois/{aoi_id}/timeseries", params={"index": "NDVI"}, headers=_hdr(session_a)
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["aoi_id"] == aoi_id
    assert body["index"] == "NDVI"
    assert len(body["dates"]) == 1
    assert len(body["values"]) == 1
    assert body["dates"][0] == "2019-07-15"  # 2019 doy 196
    assert abs(body["values"][0] - _PEAK_VALUE) < 1e-5


async def test_ndwi_series_is_empty_honest(app_dsn: str) -> None:
    """NDWI degrades to an empty series (no temporal anchor persisted)."""
    session_a, aoi_id = await _seed_session_with_parcel(app_dsn)
    client, _app = _make_client(app_dsn)
    async with client:
        resp = await client.get(
            f"/aois/{aoi_id}/timeseries", params={"index": "NDWI"}, headers=_hdr(session_a)
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["index"] == "NDWI"
    assert body["dates"] == []
    assert body["values"] == []


async def test_foreign_aoi_is_404(app_dsn: str) -> None:
    """Session B requesting A's AOI timeseries is ``404`` (RLS hides the AOI)."""
    _session_a, aoi_id = await _seed_session_with_parcel(app_dsn)
    session_b, _ = await _seed_session_with_parcel(app_dsn)
    client, _app = _make_client(app_dsn)
    async with client:
        resp = await client.get(
            f"/aois/{aoi_id}/timeseries", params={"index": "NDVI"}, headers=_hdr(session_b)
        )
    assert resp.status_code == 404


async def test_invalid_index_is_422(app_dsn: str) -> None:
    """An index outside NDVI/NDWI/NDMI is rejected with ``422``."""
    session_a, aoi_id = await _seed_session_with_parcel(app_dsn)
    client, _app = _make_client(app_dsn)
    async with client:
        resp = await client.get(
            f"/aois/{aoi_id}/timeseries", params={"index": "BOGUS"}, headers=_hdr(session_a)
        )
    assert resp.status_code == 422
