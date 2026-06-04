"""Idempotent seed of demo data for AgroSatCopilot.

Inserts a demo chat session and a parcel (AOI) in Tuscany, Italy,
on the initial schema (``chat_sessions`` + ``aois``). Intended to be
run via ``make db-seed`` after ``dbmate up`` in a freshly-cloned ``dev``
environment.

Usage:
    poetry run python scripts/seed.py

Relevant environment variables:
    DATABASE_URL: Postgres URL. Accepts the ``postgresql+asyncpg://`` prefix
        used by SQLAlchemy; it is normalized to ``postgresql://`` for asyncpg.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Final

import asyncpg
import structlog

logger = structlog.get_logger(__name__)

DEFAULT_DATABASE_URL: Final[str] = "postgresql://agrosat:agrosat@localhost:5432/agrosat"
DEMO_USER_ID: Final[str] = "demo@agrosat.dev"
DEMO_LLM_VARIANT: Final[str] = "gemini"
DEMO_AOI_LABEL: Final[str] = "Demo parcel - Tuscany"
DEMO_AOI_WKT: Final[str] = (
    "POLYGON((11.10 43.30, 11.11 43.30, 11.11 43.31, 11.10 43.31, 11.10 43.30))"
)
DEMO_AOI_SRID: Final[int] = 4326
DEMO_AOI_AREA_HA: Final[float] = 1.0


def _resolve_database_url() -> str:
    """Resolves the Postgres URL normalized for asyncpg.

    Returns:
        URL with ``postgresql://`` scheme (without the SQLAlchemy driver suffix).
    """
    raw_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    if raw_url.startswith("postgresql+asyncpg://"):
        return raw_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    return raw_url


async def _aoi_exists(conn: asyncpg.Connection, label: str) -> bool:
    """Checks whether an AOI with the given label already exists."""
    row = await conn.fetchrow("SELECT 1 FROM aois WHERE label = $1 LIMIT 1", label)
    return row is not None


async def _insert_demo(conn: asyncpg.Connection) -> tuple[str, int]:
    """Inserts the demo session and its AOI within a transaction.

    Returns:
        Tuple ``(session_id, aoi_id)`` just created.
    """
    async with conn.transaction():
        session_id: str = await conn.fetchval(
            """
            INSERT INTO chat_sessions (user_id, llm_variant)
            VALUES ($1, $2)
            RETURNING id
            """,
            DEMO_USER_ID,
            DEMO_LLM_VARIANT,
        )
        aoi_id: int = await conn.fetchval(
            """
            INSERT INTO aois (session_id, geom, label, area_ha)
            VALUES (
                $1,
                ST_GeomFromText($2, $3),
                $4,
                $5
            )
            RETURNING id
            """,
            session_id,
            DEMO_AOI_WKT,
            DEMO_AOI_SRID,
            DEMO_AOI_LABEL,
            DEMO_AOI_AREA_HA,
        )
    return session_id, aoi_id


async def main() -> int:
    """Async entry point of the seed.

    Connects to Postgres, checks idempotency by ``label`` and, if it does not
    exist, creates a demo ``chat_sessions`` + ``aois``. Prints the result to stdout.

    Returns:
        Exit code: ``0`` success, ``1`` connection or execution error.
    """
    dsn = _resolve_database_url()
    logger.info("seed.connect", dsn_host=dsn.split("@")[-1])

    try:
        conn = await asyncpg.connect(dsn=dsn)
    except (OSError, asyncpg.PostgresError) as exc:
        logger.error("seed.connect_failed", error=str(exc))
        print(f"ERROR: no se pudo conectar a Postgres ({exc})", file=sys.stderr)
        return 1

    try:
        if await _aoi_exists(conn, DEMO_AOI_LABEL):
            logger.info("seed.skip", label=DEMO_AOI_LABEL)
            print("already seeded, skipping")
            return 0

        session_id, aoi_id = await _insert_demo(conn)
    except asyncpg.PostgresError as exc:
        logger.error("seed.insert_failed", error=str(exc))
        print(f"ERROR: fallo al insertar datos demo ({exc})", file=sys.stderr)
        return 1
    finally:
        await conn.close()

    logger.info("seed.done", session_id=str(session_id), aoi_id=aoi_id)
    print(f"seeded session_id={session_id}, aoi_id={aoi_id}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
