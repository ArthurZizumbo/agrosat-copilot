"""Integration smoke test for ``scripts/seed.py`` (US-001).

Spins up an ephemeral PostGIS container, manually applies the
``-- migrate:up`` block of the initial migration and runs ``scripts/seed.py``
twice in a row to validate success + idempotency.

It is skipped automatically if ``testcontainers`` is not available or if
Docker is not running on the host.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
SEED_SCRIPT = REPO_ROOT / "scripts" / "seed.py"
MIGRATION_SQL = REPO_ROOT / "db" / "migrations" / "20260511213942_initial_schema.sql"
DEMO_LABEL = "Demo parcel - Tuscany"


def _extract_migrate_up(sql_text: str) -> str:
    """Extract the block between ``-- migrate:up`` and ``-- migrate:down``."""
    after_up = sql_text.split("-- migrate:up", 1)[1]
    up_block = after_up.split("-- migrate:down", 1)[0]
    return up_block.strip()


async def _apply_migration(dsn: str, up_sql: str) -> None:
    """Apply the initial migration to the container via asyncpg."""
    import asyncpg  # type: ignore[import-not-found]

    conn = await asyncpg.connect(dsn=dsn)
    try:
        # postgis_topology and pg_stat_statements may not be in the base image;
        # we run each CREATE EXTENSION in a tolerant way.
        statements = [s.strip() for s in up_sql.split(";") if s.strip()]
        for stmt in statements:
            try:
                await conn.execute(stmt)
            except Exception as exc:
                if "CREATE EXTENSION" in stmt.upper():
                    # Optional extensions (postgis_topology, pg_stat_statements,
                    # vector) may be missing in the postgis/postgis:15-3.4 image.
                    print(f"skip optional stmt ({exc}): {stmt[:60]}")
                    continue
                raise
    finally:
        await conn.close()


async def _count_demo_aoi(dsn: str) -> int:
    """Count AOIs with the demo label."""
    import asyncpg  # type: ignore[import-not-found]

    conn = await asyncpg.connect(dsn=dsn)
    try:
        return await conn.fetchval("SELECT COUNT(*) FROM aois WHERE label = $1", DEMO_LABEL)
    finally:
        await conn.close()


def test_seed_smoke_idempotent() -> None:
    """Run seed.py twice and validate success, demo row and idempotency."""
    testcontainers = pytest.importorskip("testcontainers.postgres")
    PostgresContainer = testcontainers.PostgresContainer

    pytest.importorskip("asyncpg")

    container = PostgresContainer(
        image="postgis/postgis:15-3.4",
        username="agrosat",
        password="agrosat",
        dbname="agrosat",
    )

    try:
        container.start()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Docker no disponible para testcontainers: {exc}")

    try:
        # URL for asyncpg (without +psycopg2 suffix)
        host = container.get_container_host_ip()
        port = container.get_exposed_port(5432)
        dsn = f"postgresql://agrosat:agrosat@{host}:{port}/agrosat"

        up_sql = _extract_migrate_up(MIGRATION_SQL.read_text(encoding="utf-8"))
        asyncio.run(_apply_migration(dsn, up_sql))

        env = {**os.environ, "DATABASE_URL": dsn}

        # First run: must insert demo
        first = subprocess.run(
            [sys.executable, str(SEED_SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
            timeout=60,
            cwd=str(REPO_ROOT),
        )
        assert first.returncode == 0, f"seed.py exit={first.returncode} stderr={first.stderr}"
        assert "seeded session_id=" in first.stdout or "already seeded, skipping" in first.stdout, (
            f"stdout inesperado: {first.stdout!r}"
        )

        count = asyncio.run(_count_demo_aoi(dsn))
        assert count == 1, f"esperaba 1 AOI demo, encontre {count}"

        # Second run: must be idempotent
        second = subprocess.run(
            [sys.executable, str(SEED_SCRIPT)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
            timeout=60,
            cwd=str(REPO_ROOT),
        )
        assert second.returncode == 0, (
            f"seed.py (2nd) exit={second.returncode} stderr={second.stderr}"
        )
        assert "already seeded, skipping" in second.stdout, (
            f"esperaba mensaje idempotente, stdout={second.stdout!r}"
        )

        count_after = asyncio.run(_count_demo_aoi(dsn))
        assert count_after == 1, f"idempotencia rota: {count_after} filas tras segundo run"
    finally:
        container.stop()
