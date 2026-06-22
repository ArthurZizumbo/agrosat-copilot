"""``GET /parcels`` bbox-clipped GeoJSON + cross-session isolation tests.

Real integration tests over a throwaway PostGIS + pgvector container (the US-051
RLS harness), mirroring ``test_aois_endpoint.py``: every ``db/migrations/*.sql``
is applied, the FastAPI app is driven through ``httpx.AsyncClient`` +
``ASGITransport``, and ``get_scoped_conn`` is overridden to bind a connection
from the **non-superuser** ``agrosat_app`` role (``NOBYPASSRLS``) primed with the
request's ``X-Session-ID`` -- byte-identical to the production RLS hook. The
auth-guard ``verify_chat_session`` runs unchanged, so both the database (RLS) AND
the application guard are exercised end to end.

What is pinned here:

- **Happy path**: the session's parcels overlapping the bbox for ``year`` come
  back as a GeoJSON ``FeatureCollection`` whose features expose ``parcel_id``
  (the row id) plus ``crop_class`` / ``confidence`` / ``area_ha``.
- **Spatial / year filters**: a parcel outside the bbox (or a different year) is
  excluded; ``limit`` caps the page.
- **422**: a malformed ``bbox`` query string is rejected at the edge.
- **403 (unknown session)** and **foreign isolation (empty under RLS)**: a parcel
  owned by session A is invisible to session B, and an unseeded session is
  ``403`` before any handler runs.

Auto-skips without ``testcontainers`` / ``asyncpg`` / Docker so ``make test`` is
green in CI without a Docker daemon.
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

# A 0.1deg square inside the test bbox (lon ~7.0..7.9, lat ~48.3..48.8).
_PARCEL_INSIDE = "POLYGON((7.10 48.40, 7.20 48.40, 7.20 48.50, 7.10 48.50, 7.10 48.40))"
# Far away (different continent) -> excluded by the bbox && filter.
_PARCEL_OUTSIDE = "POLYGON((11.00 43.00, 11.10 43.00, 11.10 43.10, 11.00 43.10, 11.00 43.00))"
_BBOX = "7.0,48.3,7.9,48.8"


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


async def _insert_parcel(
    app_dsn: str,
    session_id: UUID,
    *,
    wkt: str,
    year: int,
    crop_class: str | None = None,
    confidence: float | None = None,
    area_ha: float | None = None,
) -> int:
    """Insert a parcel for ``session_id`` (RLS-primed) and return its id."""
    conn = await asyncpg.connect(dsn=app_dsn)
    try:
        async with conn.transaction():
            await conn.execute(SET_SESSION_SQL, str(session_id))
            row = await conn.fetchrow(
                """
                INSERT INTO parcels (session_id, geom, crop_class, confidence, area_ha, year)
                VALUES (
                    current_setting('app.current_session')::uuid,
                    ST_SetSRID(ST_GeomFromText($1), 4326),
                    $2, $3, $4, $5
                )
                RETURNING id
                """,
                wkt,
                crop_class,
                confidence,
                area_ha,
                year,
            )
        return int(row["id"])
    finally:
        await conn.close()


def _make_client(app_dsn: str) -> tuple[AsyncClient, object]:
    """Build a client whose scoped-conn dependency binds a real RLS connection."""
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
# Happy path + filters
# ---------------------------------------------------------------------------
async def test_returns_feature_collection_with_parcel_properties(app_dsn: str) -> None:
    """The bbox query returns a FeatureCollection with ``parcel_id`` properties."""
    session_a = await _seed_session(app_dsn)
    pid = await _insert_parcel(
        app_dsn,
        session_a,
        wkt=_PARCEL_INSIDE,
        year=2019,
        crop_class="wheat",
        confidence=0.91,
        area_ha=12.5,
    )
    client, _app = _make_client(app_dsn)
    async with client:
        resp = await client.get(
            "/parcels", params={"bbox": _BBOX, "year": 2019}, headers=_hdr(session_a)
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["type"] == "FeatureCollection"
    feature = next(f for f in body["features"] if f["properties"]["parcel_id"] == pid)
    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "Polygon"
    props = feature["properties"]
    assert props["crop_class"] == "wheat"
    assert props["confidence"] == pytest.approx(0.91, abs=1e-4)
    assert props["area_ha"] == pytest.approx(12.5, abs=1e-4)
    # The contract exposes ``parcel_id``, never a raw ``id`` on the feature.
    assert "id" not in feature


async def test_bbox_and_year_filters_exclude_non_matching(app_dsn: str) -> None:
    """Parcels outside the bbox or with another year are excluded."""
    session_a = await _seed_session(app_dsn)
    inside = await _insert_parcel(app_dsn, session_a, wkt=_PARCEL_INSIDE, year=2019)
    outside = await _insert_parcel(app_dsn, session_a, wkt=_PARCEL_OUTSIDE, year=2019)
    other_year = await _insert_parcel(app_dsn, session_a, wkt=_PARCEL_INSIDE, year=2018)
    client, _app = _make_client(app_dsn)
    async with client:
        resp = await client.get(
            "/parcels", params={"bbox": _BBOX, "year": 2019}, headers=_hdr(session_a)
        )
    assert resp.status_code == 200
    ids = {f["properties"]["parcel_id"] for f in resp.json()["features"]}
    assert inside in ids
    assert outside not in ids
    assert other_year not in ids


async def test_limit_caps_the_page(app_dsn: str) -> None:
    """``limit`` bounds the number of features returned."""
    session_a = await _seed_session(app_dsn)
    for _ in range(3):
        await _insert_parcel(app_dsn, session_a, wkt=_PARCEL_INSIDE, year=2019)
    client, _app = _make_client(app_dsn)
    async with client:
        resp = await client.get(
            "/parcels", params={"bbox": _BBOX, "year": 2019, "limit": 2}, headers=_hdr(session_a)
        )
    assert resp.status_code == 200
    assert len(resp.json()["features"]) == 2


async def test_malformed_bbox_is_422(app_dsn: str) -> None:
    """A bbox without four numeric components is ``422`` at the edge."""
    session_a = await _seed_session(app_dsn)
    client, _app = _make_client(app_dsn)
    async with client:
        too_few = await client.get(
            "/parcels", params={"bbox": "7.0,48.3,7.9"}, headers=_hdr(session_a)
        )
        non_numeric = await client.get(
            "/parcels", params={"bbox": "a,b,c,d"}, headers=_hdr(session_a)
        )
    assert too_few.status_code == 422
    assert non_numeric.status_code == 422


# ---------------------------------------------------------------------------
# Cross-session isolation
# ---------------------------------------------------------------------------
async def test_unknown_session_is_403(app_dsn: str) -> None:
    """A session with no ``chat_sessions`` row is ``403`` before any handler runs."""
    unknown = uuid4()
    client, _app = _make_client(app_dsn)
    async with client:
        resp = await client.get("/parcels", params={"bbox": _BBOX}, headers=_hdr(unknown))
    assert resp.status_code == 403, resp.text


async def test_foreign_session_cannot_see_parcels(app_dsn: str) -> None:
    """A valid session B never sees session A's parcels (RLS-filtered)."""
    session_a = await _seed_session(app_dsn)
    session_b = await _seed_session(app_dsn)
    await _insert_parcel(app_dsn, session_a, wkt=_PARCEL_INSIDE, year=2019)
    client, _app = _make_client(app_dsn)
    async with client:
        resp = await client.get(
            "/parcels", params={"bbox": _BBOX, "year": 2019}, headers=_hdr(session_b)
        )
    assert resp.status_code == 200
    assert resp.json()["features"] == []


async def test_malformed_session_header_is_400(app_dsn: str) -> None:
    """A malformed ``X-Session-ID`` is ``400`` upstream (not ``403``/``500``)."""
    client, _app = _make_client(app_dsn)
    async with client:
        resp = await client.get(
            "/parcels", params={"bbox": _BBOX}, headers={"X-Session-ID": "not-a-uuid"}
        )
    assert resp.status_code == 400
