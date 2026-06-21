"""Cross-session isolation integration tests for the US-051 RLS migration.

These are **real** integration tests against PostgreSQL (the whole point of
US-051): no mocks. A throwaway PostGIS + pgvector container is started, the four
existing migrations plus the new ``20260620000418_rls_multi_tenant.sql`` are
applied as the container superuser (the migration role that *bypasses* RLS),
and then a **second** connection is opened as the non-superuser application role
``agrosat_app`` to assert that the ``tenant_isolation`` policies actually
enforce.

Risk #1 of the plan (documented in ``docs/us-handoff/us-051.md``): a *superuser*
bypasses RLS **always**, even under ``FORCE ROW LEVEL SECURITY``. If the
assertions ran as the migration superuser they would be a false green. Hence
every isolation assertion below runs over ``_app_conn`` (role ``agrosat_app``,
``NOSUPERUSER NOBYPASSRLS``), and the suite first proves the role really lacks
those attributes (:func:`test_app_role_is_not_superuser_and_not_bypassrls`).

The session key is the runtime setting ``app.current_session``, primed via the
contract shared with ``ml.agent.db`` / ``backend.app.core.db``::

    SELECT set_config('app.current_session', $1, true)   -- SET LOCAL semantics

Auto-skip: if ``testcontainers`` / ``asyncpg`` are missing, or Docker is not
running, or no PostGIS+pgvector image is available, the module skips cleanly so
``make test`` does not break in CI without Docker (mirrors
``test_seed_smoke.py``).
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID, uuid4

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("testcontainers.postgres", reason="testcontainers no instalado")
pytest.importorskip("asyncpg", reason="asyncpg requerido")

import asyncpg  # noqa: E402
from testcontainers.postgres import PostgresContainer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATIONS_DIR = REPO_ROOT / "db" / "migrations"
RLS_MIGRATION = MIGRATIONS_DIR / "20260620000418_rls_multi_tenant.sql"

# Candidate images in preference order; first that boots wins. The pgvector image
# is preferred because features_parcels.alphaearth_embedding is VECTOR(64).
CANDIDATE_IMAGES: tuple[str, ...] = (
    "agrosat-postgres:15-3.4-pgvector",
    "postgis/postgis:15-3.4",
)

# Application role created by the RLS migration (NON-superuser, NOBYPASSRLS).
APP_ROLE = "agrosat_app"
APP_PASSWORD = "agrosat_app"  # dev-only; matches the migration's dev password.
SET_SESSION_SQL = "SELECT set_config('app.current_session', $1, true)"

_MIGRATE_UP_RE = re.compile(r"--\s*migrate:up\s*\n(.*?)(?=--\s*migrate:down|\Z)", re.DOTALL)
_MIGRATE_DOWN_RE = re.compile(r"--\s*migrate:down\s*\n(.*?)\Z", re.DOTALL)

# A WKT polygon (Tuscany-ish) reused for aois/parcels geometries.
POLY_WKT = "POLYGON((11.0 43.0, 11.1 43.0, 11.1 43.1, 11.0 43.1, 11.0 43.0))"


def _split_up(sql_text: str) -> str:
    """Return the ``migrate:up`` block of a dbmate migration file."""
    match = _MIGRATE_UP_RE.search(sql_text)
    if match is None:
        raise ValueError("Archivo de migracion sin bloque -- migrate:up")
    return match.group(1).strip()


def _split_down(sql_text: str) -> str:
    """Return the ``migrate:down`` block of a dbmate migration file."""
    match = _MIGRATE_DOWN_RE.search(sql_text)
    if match is None:
        raise ValueError("Archivo de migracion sin bloque -- migrate:down")
    return match.group(1).strip()


async def _apply_all_migrations(dsn: str) -> None:
    """Apply every ``db/migrations/*.sql`` migrate:up block in lexical order.

    Runs as the container superuser. Each ``CREATE EXTENSION`` is tolerated if
    the optional extension is absent in the base image (e.g. postgis_topology,
    pg_stat_statements), mirroring ``test_seed_smoke.py``.
    """
    files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not files:
        raise RuntimeError(f"No se encontraron migraciones en {MIGRATIONS_DIR}")
    conn = await asyncpg.connect(dsn=dsn)
    try:
        for migration_path in files:
            up_block = _split_up(migration_path.read_text(encoding="utf-8"))
            await _execute_tolerant(conn, up_block)
    finally:
        await conn.close()


async def _execute_tolerant(conn: asyncpg.Connection, sql_block: str) -> None:
    """Execute a multi-statement SQL block, tolerating missing optional extensions.

    ``DO $$ ... $$`` blocks (the role-creation guard) contain semicolons, so the
    block is executed whole first; only if that fails do we fall back to a
    naive split that still honours the optional-extension tolerance.
    """
    try:
        await conn.execute(sql_block)
        return
    except asyncpg.PostgresError:
        pass
    statements = [s.strip() for s in sql_block.split(";") if s.strip()]
    for stmt in statements:
        try:
            await conn.execute(stmt)
        except asyncpg.PostgresError as exc:
            if "CREATE EXTENSION" in stmt.upper():
                continue
            raise RuntimeError(f"Fallo al aplicar: {stmt[:80]} -- {exc}") from exc


async def _seed_two_sessions(superuser_dsn: str) -> tuple[UUID, UUID, int, int]:
    """Seed two isolated tenants A and B as superuser (bypasses RLS to set up).

    Returns:
        ``(session_a, session_b, parcel_a_id, parcel_b_id)``.
    """
    session_a = uuid4()
    session_b = uuid4()
    conn = await asyncpg.connect(dsn=superuser_dsn)
    try:
        for sid in (session_a, session_b):
            await conn.execute(
                "INSERT INTO chat_sessions (id, user_id) VALUES ($1, $2)",
                sid,
                f"user-{sid}",
            )
            await conn.execute(
                """
                INSERT INTO aois (session_id, geom, label)
                VALUES ($1, ST_GeomFromText($2, 4326), $3)
                """,
                sid,
                POLY_WKT,
                f"aoi-{sid}",
            )
        parcel_a_id = await conn.fetchval(
            """
            INSERT INTO parcels (session_id, geom, year)
            VALUES ($1, ST_GeomFromText($2, 4326), 2024) RETURNING id
            """,
            session_a,
            POLY_WKT,
        )
        parcel_b_id = await conn.fetchval(
            """
            INSERT INTO parcels (session_id, geom, year)
            VALUES ($1, ST_GeomFromText($2, 4326), 2024) RETURNING id
            """,
            session_b,
            POLY_WKT,
        )
        for pid in (parcel_a_id, parcel_b_id):
            await conn.execute(
                "INSERT INTO features_parcels (parcel_id, year) VALUES ($1, 2024)",
                pid,
            )
    finally:
        await conn.close()
    return session_a, session_b, parcel_a_id, parcel_b_id


async def _scoped(conn: asyncpg.Connection, session_id: UUID) -> asyncpg.transaction.Transaction:
    """Open a transaction on ``conn`` and prime ``app.current_session``.

    Returns the open transaction so the caller can roll it back. ``SET LOCAL``
    semantics require the setting and the queries to share one transaction.
    """
    tx = conn.transaction()
    await tx.start()
    await conn.execute(SET_SESSION_SQL, str(session_id))
    return tx


@pytest.fixture(scope="module")
def pg_dsns() -> Iterator[tuple[str, str]]:
    """Boot Postgres, apply all migrations (incl. RLS), yield (superuser, app) DSNs.

    - ``superuser_dsn``: the container's own role (BYPASSRLS) — used only for
      seeding/inspection setup.
    - ``app_dsn``: role ``agrosat_app`` (NOSUPERUSER, NOBYPASSRLS) — used for
      every isolation assertion. This is the load-bearing distinction (plan
      risk #1).
    """
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

            # Bind loop vars as defaults so the closure does not capture them by
            # late reference (ruff B023); both are consumed in this iteration.
            async def _bootstrap(dsn: str = superuser_dsn, img: str = image) -> None:
                conn = await asyncpg.connect(dsn=dsn)
                try:
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                except asyncpg.PostgresError as exc:
                    raise RuntimeError(f"pgvector ausente en {img}: {exc}") from exc
                finally:
                    await conn.close()
                await _apply_all_migrations(dsn)

            asyncio.run(_bootstrap())

            app_dsn = f"postgresql://{APP_ROLE}:{APP_PASSWORD}@{host}:{port}/agrosat"
            yield superuser_dsn, app_dsn
            container.stop()
            return
        except Exception as exc:  # noqa: BLE001 - image/host dependent
            last_error = exc
            container.stop()
            continue

    pytest.skip(f"Sin imagen Postgres+PostGIS+pgvector utilizable: {last_error}")


@pytest.fixture(scope="module")
def seeded(pg_dsns: tuple[str, str]) -> tuple[UUID, UUID, int, int]:
    """Seed two tenants once for the module and return their identifiers."""
    superuser_dsn, _ = pg_dsns
    return asyncio.run(_seed_two_sessions(superuser_dsn))


def test_app_role_is_not_superuser_and_not_bypassrls(pg_dsns: tuple[str, str]) -> None:
    """Guard: the test role must be NON-superuser + NOBYPASSRLS, else false green.

    A superuser bypasses RLS even under FORCE; if ``agrosat_app`` had either
    attribute the isolation assertions below would be meaningless.
    """
    superuser_dsn, _ = pg_dsns

    async def _check() -> tuple[bool, bool]:
        conn = await asyncpg.connect(dsn=superuser_dsn)
        try:
            row = await conn.fetchrow(
                "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = $1",
                APP_ROLE,
            )
        finally:
            await conn.close()
        assert row is not None, f"rol {APP_ROLE} no existe tras la migracion"
        return row["rolsuper"], row["rolbypassrls"]

    rolsuper, rolbypassrls = asyncio.run(_check())
    assert rolsuper is False, "agrosat_app NO debe ser superuser (bypassaria RLS)"
    assert rolbypassrls is False, "agrosat_app NO debe tener BYPASSRLS"


def test_select_isolation_session_a_sees_only_its_rows(
    pg_dsns: tuple[str, str], seeded: tuple[UUID, UUID, int, int]
) -> None:
    """With session A primed, A sees only its own chat_session/aoi/parcel rows."""
    _, app_dsn = pg_dsns
    session_a, _session_b, parcel_a_id, parcel_b_id = seeded

    async def _run() -> None:
        conn = await asyncpg.connect(dsn=app_dsn)
        try:
            tx = await _scoped(conn, session_a)
            try:
                chat_ids = {r["id"] for r in await conn.fetch("SELECT id FROM chat_sessions")}
                assert chat_ids == {session_a}, f"A vio chat_sessions ajenas: {chat_ids}"

                aoi_owners = {
                    r["session_id"] for r in await conn.fetch("SELECT session_id FROM aois")
                }
                assert aoi_owners == {session_a}, f"A vio aois ajenas: {aoi_owners}"

                parcel_ids = {r["id"] for r in await conn.fetch("SELECT id FROM parcels")}
                assert parcel_a_id in parcel_ids
                assert parcel_b_id not in parcel_ids, "A vio la parcela de B (RLS rota)"

                # features_parcels has no session_id: isolated via EXISTS on parcels.
                feat_parcels = {
                    r["parcel_id"]
                    for r in await conn.fetch("SELECT parcel_id FROM features_parcels")
                }
                assert parcel_a_id in feat_parcels
                assert parcel_b_id not in feat_parcels, "A vio features de B (subquery rota)"
            finally:
                await tx.rollback()
        finally:
            await conn.close()

    asyncio.run(_run())


def test_no_session_set_is_fail_closed(
    pg_dsns: tuple[str, str], seeded: tuple[UUID, UUID, int, int]
) -> None:
    """Without app.current_session, every multi-tenant table returns 0 rows."""
    _, app_dsn = pg_dsns

    async def _run() -> None:
        conn = await asyncpg.connect(dsn=app_dsn)
        try:
            # No set_config -> current_setting(..., true) is NULL -> 0 rows.
            for table in ("chat_sessions", "aois", "parcels", "features_parcels"):
                count = await conn.fetchval(f"SELECT count(*) FROM {table}")
                assert count == 0, f"{table} debio fallar-cerrado sin sesion, vi {count}"
        finally:
            await conn.close()

    asyncio.run(_run())


def test_update_across_tenant_affects_zero_rows(
    pg_dsns: tuple[str, str], seeded: tuple[UUID, UUID, int, int]
) -> None:
    """A's UPDATE targeting B's parcel/aoi affects 0 rows (B's data untouched)."""
    _, app_dsn = pg_dsns
    session_a, session_b, _parcel_a_id, parcel_b_id = seeded

    async def _run() -> None:
        conn = await asyncpg.connect(dsn=app_dsn)
        try:
            tx = await _scoped(conn, session_a)
            try:
                # UPDATE by explicit id of B's parcel: RLS USING hides it -> 0 rows.
                status_parcel = await conn.execute(
                    "UPDATE parcels SET crop_class = 'hacked' WHERE id = $1", parcel_b_id
                )
                assert status_parcel.endswith(" 0"), f"UPDATE cross-tenant afecto: {status_parcel}"

                status_aoi = await conn.execute(
                    "UPDATE aois SET label = 'hacked' WHERE session_id = $1", session_b
                )
                assert status_aoi.endswith(" 0"), f"UPDATE aoi cross-tenant afecto: {status_aoi}"
            finally:
                await tx.rollback()
        finally:
            await conn.close()

    asyncio.run(_run())


def test_delete_across_tenant_affects_zero_rows(
    pg_dsns: tuple[str, str], seeded: tuple[UUID, UUID, int, int]
) -> None:
    """A's DELETE targeting B's parcel affects 0 rows; B's parcel still present."""
    _, app_dsn = pg_dsns
    superuser_dsn, app_dsn = pg_dsns
    session_a, _session_b, _parcel_a_id, parcel_b_id = seeded

    async def _run() -> None:
        app_conn = await asyncpg.connect(dsn=app_dsn)
        try:
            tx = await _scoped(app_conn, session_a)
            try:
                status = await app_conn.execute("DELETE FROM parcels WHERE id = $1", parcel_b_id)
                assert status.endswith(" 0"), f"DELETE cross-tenant afecto: {status}"
            finally:
                await tx.rollback()
        finally:
            await app_conn.close()

        # Confirm B's parcel survived (verified as superuser to bypass RLS).
        su_conn = await asyncpg.connect(dsn=superuser_dsn)
        try:
            still_there = await su_conn.fetchval(
                "SELECT count(*) FROM parcels WHERE id = $1", parcel_b_id
            )
        finally:
            await su_conn.close()
        assert still_there == 1, "La parcela de B fue borrada por A (RLS rota)"

    asyncio.run(_run())


def test_insert_with_foreign_session_id_blocked_by_with_check(
    pg_dsns: tuple[str, str], seeded: tuple[UUID, UUID, int, int]
) -> None:
    """A inserting an aoi/parcel with B's session_id is blocked by WITH CHECK."""
    _, app_dsn = pg_dsns
    session_a, session_b, _, _ = seeded

    async def _run() -> None:
        conn = await asyncpg.connect(dsn=app_dsn)
        try:
            tx = await _scoped(conn, session_a)
            try:
                with pytest.raises(asyncpg.PostgresError) as exc_info:
                    await conn.execute(
                        """
                        INSERT INTO aois (session_id, geom, label)
                        VALUES ($1, ST_GeomFromText($2, 4326), 'intruder')
                        """,
                        session_b,
                        POLY_WKT,
                    )
                # row-level security violation maps to SQLSTATE 42501.
                assert "row-level security" in str(exc_info.value).lower() or (
                    getattr(exc_info.value, "sqlstate", None) == "42501"
                ), f"Esperaba violacion RLS WITH CHECK, vi: {exc_info.value!r}"
            finally:
                await tx.rollback()
        finally:
            await conn.close()

    asyncio.run(_run())


def test_insert_own_session_id_allowed(
    pg_dsns: tuple[str, str], seeded: tuple[UUID, UUID, int, int]
) -> None:
    """A inserting an aoi with its OWN session_id passes WITH CHECK and is visible."""
    _, app_dsn = pg_dsns
    session_a, _, _, _ = seeded

    async def _run() -> None:
        conn = await asyncpg.connect(dsn=app_dsn)
        try:
            tx = await _scoped(conn, session_a)
            try:
                new_id = await conn.fetchval(
                    """
                    INSERT INTO aois (session_id, geom, label)
                    VALUES ($1, ST_GeomFromText($2, 4326), 'own-row') RETURNING id
                    """,
                    session_a,
                    POLY_WKT,
                )
                assert new_id is not None
                seen = await conn.fetchval("SELECT count(*) FROM aois WHERE id = $1", new_id)
                assert seen == 1, "A no ve su propia fila recien insertada"
            finally:
                # Roll back so the module's seeded state stays deterministic.
                await tx.rollback()
        finally:
            await conn.close()

    asyncio.run(_run())


def test_features_parcels_with_check_blocks_foreign_parent(
    pg_dsns: tuple[str, str], seeded: tuple[UUID, UUID, int, int]
) -> None:
    """A cannot attach a feature row to B's parcel (EXISTS subquery WITH CHECK)."""
    _, app_dsn = pg_dsns
    session_a, _, _, parcel_b_id = seeded

    async def _run() -> None:
        conn = await asyncpg.connect(dsn=app_dsn)
        try:
            tx = await _scoped(conn, session_a)
            try:
                with pytest.raises(asyncpg.PostgresError) as exc_info:
                    await conn.execute(
                        "INSERT INTO features_parcels (parcel_id, year) VALUES ($1, 2025)",
                        parcel_b_id,
                    )
                assert "row-level security" in str(exc_info.value).lower() or (
                    getattr(exc_info.value, "sqlstate", None) == "42501"
                ), f"Esperaba bloqueo WITH CHECK por parcela ajena, vi: {exc_info.value!r}"
            finally:
                await tx.rollback()
        finally:
            await conn.close()

    asyncio.run(_run())


def test_migrate_down_removes_policies_and_disables_rls(pg_dsns: tuple[str, str]) -> None:
    """``migrate:down`` reverts cleanly: policies gone, RLS disabled, role dropped.

    Runs last in the module (lexically after the isolation tests it depends on).
    After rollback the suite re-applies ``migrate:up`` so the module teardown
    leaves no surprises for other tests sharing the container.
    """
    superuser_dsn, _ = pg_dsns
    down_block = _split_down(RLS_MIGRATION.read_text(encoding="utf-8"))
    up_block = _split_up(RLS_MIGRATION.read_text(encoding="utf-8"))
    tables = ("chat_sessions", "aois", "parcels", "features_parcels")

    async def _run() -> None:
        conn = await asyncpg.connect(dsn=superuser_dsn)
        try:
            # Sanity: policies + RLS present before rollback.
            policies_before = await conn.fetchval(
                "SELECT count(*) FROM pg_policies "
                "WHERE policyname = 'tenant_isolation' AND tablename = ANY($1::text[])",
                list(tables),
            )
            assert policies_before == 4, f"Esperaba 4 policies pre-down, vi {policies_before}"

            await _execute_tolerant(conn, down_block)

            policies_after = await conn.fetchval(
                "SELECT count(*) FROM pg_policies "
                "WHERE policyname = 'tenant_isolation' AND tablename = ANY($1::text[])",
                list(tables),
            )
            assert policies_after == 0, f"down dejo policies: {policies_after}"

            rls_enabled = await conn.fetch(
                "SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class "
                "WHERE relname = ANY($1::text[])",
                list(tables),
            )
            for row in rls_enabled:
                assert row["relrowsecurity"] is False, f"{row['relname']} sigue con RLS ENABLED"
                assert row["relforcerowsecurity"] is False, f"{row['relname']} sigue con FORCE"

            role_left = await conn.fetchval(
                "SELECT count(*) FROM pg_roles WHERE rolname = $1", APP_ROLE
            )
            assert role_left == 0, "down no elimino el rol agrosat_app"

            # Re-apply up so the container is left in the migrated state.
            await _execute_tolerant(conn, up_block)
            policies_reapplied = await conn.fetchval(
                "SELECT count(*) FROM pg_policies "
                "WHERE policyname = 'tenant_isolation' AND tablename = ANY($1::text[])",
                list(tables),
            )
            assert policies_reapplied == 4, "round-trip up->down->up no restauro las policies"
        finally:
            await conn.close()

    asyncio.run(_run())
