"""Seed idempotente de datos demo para AgroSatCopilot.

Inserta una sesion de chat demo y una parcela (AOI) en la Toscana, Italia,
sobre el esquema inicial (``chat_sessions`` + ``aois``). Pensado para
ejecutarse via ``make db-seed`` tras ``dbmate up`` en un entorno ``dev``
recien clonado.

Uso:
    poetry run python scripts/seed.py

Variables de entorno relevantes:
    DATABASE_URL: URL de Postgres. Acepta el prefijo ``postgresql+asyncpg://``
        usado por SQLAlchemy; se normaliza a ``postgresql://`` para asyncpg.
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
    """Resuelve la URL de Postgres normalizada para asyncpg.

    Returns:
        URL con esquema ``postgresql://`` (sin el sufijo de driver SQLAlchemy).
    """
    raw_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    if raw_url.startswith("postgresql+asyncpg://"):
        return raw_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    return raw_url


async def _aoi_exists(conn: asyncpg.Connection, label: str) -> bool:
    """Verifica si ya existe un AOI con la etiqueta dada."""
    row = await conn.fetchrow("SELECT 1 FROM aois WHERE label = $1 LIMIT 1", label)
    return row is not None


async def _insert_demo(conn: asyncpg.Connection) -> tuple[str, int]:
    """Inserta la sesion demo y su AOI dentro de una transaccion.

    Returns:
        Tupla ``(session_id, aoi_id)`` recien creados.
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
    """Punto de entrada async del seed.

    Conecta a Postgres, verifica idempotencia por ``label`` y, si no existe,
    crea una ``chat_sessions`` + ``aois`` demo. Imprime el resultado en stdout.

    Returns:
        Codigo de salida: ``0`` exito, ``1`` error de conexion o ejecucion.
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
