"""Smoke test de integracion para ``scripts/seed.py`` (US-001).

Levanta un contenedor PostGIS efimero, aplica manualmente el bloque
``-- migrate:up`` de la migracion inicial y ejecuta ``scripts/seed.py``
dos veces consecutivas para validar exito + idempotencia.

Se omite automaticamente si ``testcontainers`` no esta disponible o si
Docker no esta corriendo en el host.
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
    """Extrae el bloque entre ``-- migrate:up`` y ``-- migrate:down``."""
    after_up = sql_text.split("-- migrate:up", 1)[1]
    up_block = after_up.split("-- migrate:down", 1)[0]
    return up_block.strip()


async def _apply_migration(dsn: str, up_sql: str) -> None:
    """Aplica la migracion inicial al contenedor via asyncpg."""
    import asyncpg  # type: ignore[import-not-found]

    conn = await asyncpg.connect(dsn=dsn)
    try:
        # postgis_topology y pg_stat_statements pueden no estar en la imagen base;
        # ejecutamos cada CREATE EXTENSION de forma tolerante.
        statements = [s.strip() for s in up_sql.split(";") if s.strip()]
        for stmt in statements:
            try:
                await conn.execute(stmt)
            except Exception as exc:
                if "CREATE EXTENSION" in stmt.upper():
                    # Extensiones opcionales (postgis_topology, pg_stat_statements,
                    # vector) pueden faltar en la imagen postgis/postgis:15-3.4.
                    print(f"skip optional stmt ({exc}): {stmt[:60]}")
                    continue
                raise
    finally:
        await conn.close()


async def _count_demo_aoi(dsn: str) -> int:
    """Cuenta AOIs con el label demo."""
    import asyncpg  # type: ignore[import-not-found]

    conn = await asyncpg.connect(dsn=dsn)
    try:
        return await conn.fetchval("SELECT COUNT(*) FROM aois WHERE label = $1", DEMO_LABEL)
    finally:
        await conn.close()


def test_seed_smoke_idempotent() -> None:
    """Ejecuta seed.py dos veces y valida exito, fila demo y idempotencia."""
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
        # URL para asyncpg (sin sufijo +psycopg2)
        host = container.get_container_host_ip()
        port = container.get_exposed_port(5432)
        dsn = f"postgresql://agrosat:agrosat@{host}:{port}/agrosat"

        up_sql = _extract_migrate_up(MIGRATION_SQL.read_text(encoding="utf-8"))
        asyncio.run(_apply_migration(dsn, up_sql))

        env = {**os.environ, "DATABASE_URL": dsn}

        # Primer run: debe insertar demo
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

        # Segundo run: debe ser idempotente
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
