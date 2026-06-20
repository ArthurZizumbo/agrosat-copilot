"""``/aois`` CRUD + cross-session authorisation tests (US-053).

Real integration tests over a throwaway PostGIS + pgvector container (the US-051
RLS harness): every ``db/migrations/*.sql`` is applied, the FastAPI app is driven
through ``httpx.AsyncClient`` + ``ASGITransport``, and ``get_scoped_conn`` is
overridden to bind a connection from the **non-superuser** ``agrosat_app`` role
(``NOBYPASSRLS``) primed with the request's ``X-Session-ID`` -- byte-identical to
the production RLS hook. The auth-guard ``verify_session`` runs unchanged, so both
the database (RLS) AND the application guard are exercised end to end.

Two distinct cross-tenant defences are pinned here:

- **403 (unknown session)**: a session with NO ``chat_sessions`` row never passes
  ``verify_session`` -- the guard ``SELECT 1 FROM chat_sessions`` returns zero rows
  under RLS, so the request fails closed with ``403`` before any handler runs.
- **404 / empty (foreign AOI)**: a *valid* session B cannot see or delete session
  A's AOI -- RLS hides the row, so ``GET /aois`` is empty and ``GET/DELETE`` of A's
  id is ``404`` (no foreign-existence leak), and B's failed delete leaves A's AOI
  intact.

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

_POLYGON = {
    "type": "Polygon",
    "coordinates": [[[11.0, 43.0], [11.1, 43.0], [11.1, 43.1], [11.0, 43.1], [11.0, 43.0]]],
}
_POLYGON_B = {
    "type": "Polygon",
    "coordinates": [[[12.0, 44.0], [12.1, 44.0], [12.1, 44.1], [12.0, 44.1], [12.0, 44.0]]],
}


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


def _make_client(app_dsn: str) -> tuple[AsyncClient, object]:
    """Build a client whose scoped-conn dependency binds a real RLS connection.

    ``get_scoped_conn`` is overridden to open a per-request transaction on a fresh
    ``agrosat_app`` connection and prime ``app.current_session`` from the validated
    ``X-Session-ID`` (via the real ``get_request_session_id`` dependency, so a bad
    header still yields ``400``). ``verify_session`` runs unchanged against the
    scoped connection -- the guard is exercised end to end (incl. its ``403``).
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
# CRUD happy path
# ---------------------------------------------------------------------------
async def test_post_creates_aoi_201_with_geometry_and_area(app_dsn: str) -> None:
    """``POST /aois`` returns ``201`` with the geometry round-tripped + area_ha."""
    session_a = await _seed_session(app_dsn)
    client, _app = _make_client(app_dsn)
    async with client:
        created = await client.post(
            "/aois",
            json={"geometry": _POLYGON, "label": "field-a"},
            headers=_hdr(session_a),
        )
    assert created.status_code == 201, created.text
    feature = created.json()
    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "Polygon"
    assert feature["properties"]["label"] == "field-a"
    # area_ha is server-computed (geodesic); the small polygon has a positive area.
    assert feature["properties"]["area_ha"] > 0
    assert isinstance(feature["id"], int)


async def test_get_lists_only_session_aois(app_dsn: str) -> None:
    """``GET /aois`` returns exactly the session's own AOIs (RLS-filtered)."""
    session_a = await _seed_session(app_dsn)
    client, _app = _make_client(app_dsn)
    async with client:
        c1 = await client.post("/aois", json={"geometry": _POLYGON}, headers=_hdr(session_a))
        c2 = await client.post("/aois", json={"geometry": _POLYGON_B}, headers=_hdr(session_a))
        assert c1.status_code == 201 and c2.status_code == 201
        ids = {c1.json()["id"], c2.json()["id"]}

        listed = await client.get("/aois", headers=_hdr(session_a))
    assert listed.status_code == 200
    body = listed.json()
    assert body["type"] == "FeatureCollection"
    assert {f["id"] for f in body["features"]} == ids


async def test_get_by_id_and_delete_204(app_dsn: str) -> None:
    """``GET /aois/{id}`` fetches the AOI; ``DELETE`` returns ``204`` then ``404``."""
    session_a = await _seed_session(app_dsn)
    client, _app = _make_client(app_dsn)
    async with client:
        created = await client.post("/aois", json={"geometry": _POLYGON}, headers=_hdr(session_a))
        aoi_id = created.json()["id"]

        got = await client.get(f"/aois/{aoi_id}", headers=_hdr(session_a))
        assert got.status_code == 200
        assert got.json()["id"] == aoi_id

        deleted = await client.delete(f"/aois/{aoi_id}", headers=_hdr(session_a))
        assert deleted.status_code == 204
        assert deleted.content == b""

        gone = await client.get(f"/aois/{aoi_id}", headers=_hdr(session_a))
        assert gone.status_code == 404


async def test_create_rejects_non_polygon_422(app_dsn: str) -> None:
    """A non-Polygon geometry is ``422`` at the edge (before PostGIS)."""
    session_a = await _seed_session(app_dsn)
    client, _app = _make_client(app_dsn)
    multi = {"type": "MultiPolygon", "coordinates": [_POLYGON["coordinates"]]}
    async with client:
        resp = await client.post("/aois", json={"geometry": multi}, headers=_hdr(session_a))
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Cross-session: 403 (unknown session) vs 404/empty (foreign AOI under RLS)
# ---------------------------------------------------------------------------
async def test_unknown_session_is_403_on_every_verb(app_dsn: str) -> None:
    """A session with no ``chat_sessions`` row is ``403`` on POST/GET/DELETE.

    ``verify_session``'s ``SELECT 1 FROM chat_sessions`` returns zero rows under
    RLS for an unseeded session id, so the guard fails closed (``403``) before any
    AOI handler runs -- never a ``200``/``404`` leak.
    """
    unknown = uuid4()  # never seeded -> no chat_sessions row
    client, _app = _make_client(app_dsn)
    async with client:
        posted = await client.post("/aois", json={"geometry": _POLYGON}, headers=_hdr(unknown))
        listed = await client.get("/aois", headers=_hdr(unknown))
        got = await client.get("/aois/1", headers=_hdr(unknown))
        deleted = await client.delete("/aois/1", headers=_hdr(unknown))
    assert posted.status_code == 403, posted.text
    assert listed.status_code == 403
    assert got.status_code == 403
    assert deleted.status_code == 403


async def test_foreign_session_cannot_see_or_delete_aoi(app_dsn: str) -> None:
    """A valid session B is isolated from A's AOI by RLS (empty list, 404, no delete).

    Both sessions exist (both pass the ``403`` guard), so this exercises the *data*
    isolation layer: B's ``GET /aois`` is empty, B's ``GET/DELETE`` of A's id is
    ``404`` (RLS hides the row), and B's failed delete leaves A's AOI intact.
    """
    session_a = await _seed_session(app_dsn)
    session_b = await _seed_session(app_dsn)
    client, _app = _make_client(app_dsn)
    async with client:
        created = await client.post(
            "/aois", json={"geometry": _POLYGON, "label": "owned-by-a"}, headers=_hdr(session_a)
        )
        assert created.status_code == 201
        aoi_id = created.json()["id"]

        # B is a valid session (no 403) but sees none of A's AOIs.
        b_list = await client.get("/aois", headers=_hdr(session_b))
        assert b_list.status_code == 200
        assert b_list.json()["features"] == []

        # B cannot fetch or delete A's AOI (RLS -> 404, not 403, not 200).
        b_get = await client.get(f"/aois/{aoi_id}", headers=_hdr(session_b))
        assert b_get.status_code == 404
        b_del = await client.delete(f"/aois/{aoi_id}", headers=_hdr(session_b))
        assert b_del.status_code == 404

        # A's AOI survived B's failed delete attempt.
        still = await client.get(f"/aois/{aoi_id}", headers=_hdr(session_a))
        assert still.status_code == 200
        assert still.json()["id"] == aoi_id


async def test_malformed_session_header_is_400(app_dsn: str) -> None:
    """A malformed ``X-Session-ID`` is ``400`` upstream (not ``403``/``500``)."""
    client, _app = _make_client(app_dsn)
    async with client:
        resp = await client.get("/aois", headers={"X-Session-ID": "not-a-uuid"})
    assert resp.status_code == 400
