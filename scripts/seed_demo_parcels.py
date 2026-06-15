"""Seed demo parcels and features inside the demo AOI (Tuscany).

Complements ``scripts/seed.py`` (which only creates ``chat_sessions`` + the demo
``aois`` row) so the conversational vertical slice is *runnable* end-to-end
without the full ML ingestion pipeline: the vision tools (``classify_parcel``,
``compute_ndvi``) need ``parcels`` + ``features_parcels`` rows to return real
findings. ``classify_parcel`` falls back to the stored ``crop_class`` when no
XGBoost model is registered, so a live demo works from this seed alone.

Idempotent: skips if the demo session already has parcels. Does NOT touch
``seed.py`` nor its idempotency contract (``DEMO_AOI_LABEL`` skip key).

Run: ``poetry run python scripts/seed_demo_parcels.py`` (Postgres must be up and
migrations applied; run ``make db-seed`` first to create the demo AOI).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Final

import asyncpg
import structlog

logger = structlog.get_logger(__name__)

DEFAULT_DATABASE_URL: Final[str] = "postgresql://agrosat:agrosat@localhost:5432/agrosat"
DEMO_AOI_LABEL: Final[str] = "Demo parcel - Tuscany"
DEMO_YEAR: Final[int] = 2019


def _rect_wkt(min_x: float, min_y: float, max_x: float, max_y: float) -> str:
    """Build a closed rectangular POLYGON WKT from a bbox."""
    return (
        f"POLYGON(({min_x} {min_y}, {max_x} {min_y}, {max_x} {max_y}, "
        f"{min_x} {max_y}, {min_x} {min_y}))"
    )


# Four parcels tiling the demo AOI bbox (lon 11.10..11.11, lat 43.30..43.31).
# crop_class values are real PASTIS-R names so the agent cites plausible crops.
_DEMO_PARCELS: Final[tuple[dict[str, object], ...]] = (
    {
        "bbox": (11.100, 43.300, 11.105, 43.305),
        "crop_class": "Vineyard",
        "confidence": 0.93,
        "area_ha": 12.4,
        "ndvi": {"mean": 0.58, "min": 0.31, "max": 0.72, "std": 0.11},
        "pheno": {"sog_doy": 95, "peak_doy": 196, "peak_value": 0.72, "senescence_doy": 280},
        "ndvi_auc": 118.5,
    },
    {
        "bbox": (11.105, 43.300, 11.110, 43.305),
        "crop_class": "Soft winter wheat",
        "confidence": 0.88,
        "area_ha": 9.1,
        "ndvi": {"mean": 0.64, "min": 0.22, "max": 0.91, "std": 0.20},
        "pheno": {"sog_doy": 30, "peak_doy": 120, "peak_value": 0.91, "senescence_doy": 175},
        "ndvi_auc": 140.2,
    },
    {
        "bbox": (11.100, 43.305, 11.105, 43.310),
        "crop_class": "Meadow",
        "confidence": 0.79,
        "area_ha": 6.7,
        "ndvi": {"mean": 0.61, "min": 0.40, "max": 0.78, "std": 0.09},
        "pheno": {"sog_doy": 60, "peak_doy": 150, "peak_value": 0.78, "senescence_doy": 300},
        "ndvi_auc": 132.0,
    },
    {
        "bbox": (11.105, 43.305, 11.110, 43.310),
        "crop_class": "Sunflower",
        "confidence": 0.85,
        "area_ha": 8.3,
        "ndvi": {"mean": 0.55, "min": 0.18, "max": 0.86, "std": 0.22},
        "pheno": {"sog_doy": 110, "peak_doy": 205, "peak_value": 0.86, "senescence_doy": 255},
        "ndvi_auc": 101.7,
    },
)


def _resolve_database_url() -> str:
    """Return the Postgres URL normalized for asyncpg (drops the driver suffix)."""
    raw_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    if raw_url.startswith("postgresql+asyncpg://"):
        return raw_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    return raw_url


def _placeholder_embedding() -> str:
    """Return a 64-dim pgvector literal.

    A neutral placeholder: real embeddings are populated by the AlphaEarth
    ingestion US. ``classify_parcel`` uses the stored ``crop_class`` fallback
    when no model is registered, so this value only matters once a model loads.
    """
    return "[" + ",".join(["0.0"] * 64) + "]"


async def _parcels_exist(conn: asyncpg.Connection, session_id: str) -> bool:
    """Whether the demo session already has parcels (idempotency guard)."""
    row = await conn.fetchrow("SELECT 1 FROM parcels WHERE session_id = $1 LIMIT 1", session_id)
    return row is not None


async def _insert_demo_parcels(conn: asyncpg.Connection, session_id: str, aoi_id: int) -> int:
    """Insert the demo parcels and their features. Returns the count inserted."""
    embedding = _placeholder_embedding()
    inserted = 0
    async with conn.transaction():
        for parcel in _DEMO_PARCELS:
            bbox = parcel["bbox"]
            assert isinstance(bbox, tuple)
            parcel_id: int = await conn.fetchval(
                """
                INSERT INTO parcels
                    (session_id, aoi_id, geom, crop_class, confidence, area_ha, year)
                VALUES ($1, $2, ST_GeomFromText($3, 4326), $4, $5, $6, $7)
                RETURNING id
                """,
                session_id,
                aoi_id,
                _rect_wkt(*bbox),
                parcel["crop_class"],
                parcel["confidence"],
                parcel["area_ha"],
                DEMO_YEAR,
            )
            pheno = parcel["pheno"]
            assert isinstance(pheno, dict)
            await conn.execute(
                """
                INSERT INTO features_parcels
                    (parcel_id, year, alphaearth_embedding, ndvi_stats, phenology,
                     sog_doy, peak_doy, peak_value, senescence_doy, ndvi_auc)
                VALUES ($1, $2, $3::vector, $4::jsonb, $5::jsonb, $6, $7, $8, $9, $10)
                """,
                parcel_id,
                DEMO_YEAR,
                embedding,
                _json(parcel["ndvi"]),
                _json(parcel["pheno"]),
                pheno["sog_doy"],
                pheno["peak_doy"],
                pheno["peak_value"],
                pheno["senescence_doy"],
                parcel["ndvi_auc"],
            )
            inserted += 1
    return inserted


def _json(value: object) -> str:
    """Serialize a value to a JSON string for a ``::jsonb`` cast."""
    return json.dumps(value)


async def main() -> int:
    """Seed demo parcels/features for the Tuscany demo AOI. Idempotent."""
    structlog.configure(processors=[structlog.processors.JSONRenderer()])
    url = _resolve_database_url()
    conn = await asyncpg.connect(url)
    try:
        aoi = await conn.fetchrow(
            "SELECT id, session_id FROM aois WHERE label = $1 LIMIT 1", DEMO_AOI_LABEL
        )
        if aoi is None:
            logger.error("seed_parcels.no_aoi", label=DEMO_AOI_LABEL)
            sys.stderr.write("Demo AOI not found. Run `make db-seed` first to create it.\n")
            return 1
        session_id = str(aoi["session_id"])
        aoi_id = int(aoi["id"])
        if await _parcels_exist(conn, session_id):
            logger.info("seed_parcels.skip", session_id=session_id)
            return 0
        count = await _insert_demo_parcels(conn, session_id, aoi_id)
        logger.info("seed_parcels.done", session_id=session_id, parcels=count)
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
