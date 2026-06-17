"""Seed demo parcels for the copilot end-to-end demo (US-045..047 integration).

Populates a demo session with REAL PASTIS parcels so the agent tools have
session-scoped data to operate on:

- ``chat_sessions`` -- one demo session (idempotent by user id).
- ``parcels`` -- N parcels with their real PASTIS patch geometry (centroid as a
  small polygon) and the ground-truth crop class from the winning features frame.
- ``features_parcels`` -- the real AlphaEarth 64-dim embedding (``dim_00..63``)
  plus the real phenology scalars (``sog_doy``, ``peak_doy``, ``peak_value`` ...),
  so ``classify_new_parcel`` and ``explain_prediction`` run on real signal.

The data is the same the RAG corpus uses (``features_fused_winning_pastis.parquet``
+ PASTIS ``metadata.geojson``), so the perceiver, the tools and the RAG-lite all
reference one coherent set of real parcels. Idempotent: re-running keeps a single
demo session and skips parcels already seeded for it.

Usage:
    poetry run python scripts/seed_demo_parcels.py --limit 12
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import asyncpg
import polars as pl
import structlog

from ml.agent.db import to_asyncpg_dsn
from ml.features.phenology_description import (
    default_cache_dir,  # noqa: F401  (ensures ml import path)
)
from ml.utils.parcel_id import canonical_parcel_id

logger = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
# Same parquet the xgb-alphaearth classifier trains on, so the seeded embeddings
# live in the exact space the model expects (folds 1-4 train; fold-5 held out).
_FEATURES_PATH = _REPO_ROOT / "data" / "features" / "features_fused_pastis.parquet"
_PATCH_GEOJSON = _REPO_ROOT / "data" / "PASTIS-R" / "metadata.geojson"
_CLASS_MAP_PATH = _REPO_ROOT / "data" / "reference" / "pastis_class_mapping.json"


def _load_class_names() -> dict[int, str]:
    """Load the PASTIS ``class_id -> human-readable crop name`` mapping.

    Returns:
        Mapping of integer class id to its canonical crop name (e.g. 3 -> Corn).
    """
    raw = json.loads(_CLASS_MAP_PATH.read_text())
    classes = raw.get("classes", {})
    return {int(k): v.get("name", k) for k, v in classes.items()}

DEMO_USER_ID = "demo@agrosat.dev"
DEMO_LLM_VARIANT = "gemini"

#: Phenology scalar columns mirrored into ``features_parcels`` columns.
_PHENO_SCALARS = (
    "sog_doy",
    "peak_doy",
    "peak_value",
    "senescence_doy",
    "ndvi_auc",
    "ndvi_slope_pre_peak",
    "ndvi_slope_post_peak",
    "maturity_duration_days",
)


def _load_patch_centroids() -> dict[int, str]:
    """Load PASTIS patch geometries as ``{ID_PATCH: geojson_polygon_str}``.

    Returns the centroid of each patch wrapped in a tiny square polygon (the
    ``parcels.geom`` column is ``POLYGON``). PostGIS reprojects/validates at
    insert time.

    Returns:
        Mapping of integer patch id to a GeoJSON polygon string.
    """
    raw = json.loads(_PATCH_GEOJSON.read_text())
    out: dict[int, str] = {}
    for feat in raw.get("features", []):
        pid = feat.get("properties", {}).get("ID_PATCH")
        geom = feat.get("geometry")
        if pid is None or geom is None:
            continue
        out[int(pid)] = json.dumps(geom)
    return out


def _build_rows(limit: int) -> list[dict]:
    """Build the demo parcel rows from the real winning features frame.

    Args:
        limit: Maximum number of parcels to seed.

    Returns:
        A list of row dicts with parcel id, crop class, embedding, phenology
        scalars and the patch geojson geometry.
    """
    import numpy as np

    from ml.agent.tools.classify import _load_classifier
    from ml.data.pastis_seg_dataset import _build_semantic18_lut

    # IMPORTANT: the classifier emits probabilities over the contiguous
    # *semantic18* space [0..17], NOT the raw PASTIS class_id [1..18]. The LUT maps
    # one to the other; indexing the posterior with the raw class_id (without the
    # LUT) reads the wrong class -> spuriously near-zero confidence. Train the seed
    # on the SAME parquet the classifier trains on so the embedding space matches.
    df = canonical_parcel_id(pl.read_parquet(_FEATURES_PATH))
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    geoms = _load_patch_centroids()
    class_names = _load_class_names()
    classifier = _load_classifier()
    semantic18_lut = _build_semantic18_lut(255)

    rows: list[dict] = []
    seen_classes: set[str] = set()
    for rec in df.iter_rows(named=True):
        if len(rows) >= limit:
            break
        pid = str(rec["parcel_id"])
        patch_id = int(pid.split("_")[0]) if "_" in pid else None
        geom = geoms.get(patch_id) if patch_id is not None else None
        if geom is None:
            continue
        class_id = rec.get("class_id")
        crop = class_names.get(int(class_id)) if class_id is not None else "unknown"
        # Prefer a varied set of crop types for the demo: take the first parcel of
        # each class, then fill up to ``limit`` with the rest.
        if crop in seen_classes and len(seen_classes) < limit:
            continue
        seen_classes.add(crop)
        embedding = [float(rec[c]) for c in dim_cols]
        # Real model posterior for this parcel's class. Map the raw class_id to the
        # semantic18 index the classifier emits, then read that index.
        proba = classifier.predict_proba_18(np.asarray(embedding, dtype=np.float64))
        sem18 = int(semantic18_lut[int(class_id)]) if class_id is not None else 255
        confidence = (
            float(proba[sem18]) if 0 <= sem18 < len(proba) else float(proba.max())
        )
        rows.append(
            {
                "parcel_id": pid,
                "year": int(rec.get("year", 2019)),
                "crop_class": crop,
                "class_id": int(class_id) if class_id is not None else None,
                "confidence": confidence,
                "geom": geom,
                "embedding": embedding,
                "phenology": {k: rec.get(k) for k in _PHENO_SCALARS if rec.get(k) is not None},
            }
        )
    return rows


async def _seed(conn: asyncpg.Connection, rows: list[dict]) -> tuple[str, int]:
    """Insert the demo session, parcels and features (idempotent).

    Args:
        conn: An open asyncpg connection.
        rows: Demo parcel rows from :func:`_build_rows`.

    Returns:
        ``(session_id, n_parcels_inserted)``.
    """
    session_id = await conn.fetchval(
        "SELECT id FROM chat_sessions WHERE user_id = $1 LIMIT 1", DEMO_USER_ID
    )
    if session_id is None:
        session_id = await conn.fetchval(
            "INSERT INTO chat_sessions (user_id, llm_variant) VALUES ($1, $2) RETURNING id",
            DEMO_USER_ID,
            DEMO_LLM_VARIANT,
        )

    inserted = 0
    for r in rows:
        exists = await conn.fetchval(
            "SELECT 1 FROM parcels WHERE session_id = $1 AND crop_class = $2 "
            "AND year = $3 LIMIT 1",
            session_id,
            r["crop_class"],
            r["year"],
        )
        # PASTIS patches are POLYGON or MULTIPOLYGON in EPSG:4326, but
        # parcels.geom is GEOMETRY(POLYGON,4326). Represent each parcel by a
        # small square polygon around the patch centroid: always a valid POLYGON,
        # preserves the real location for the spatial RAG / AOI filters.
        parcel_id = await conn.fetchval(
            """
            INSERT INTO parcels (session_id, geom, crop_class, confidence, year)
            VALUES (
                $1,
                ST_SetSRID(
                    ST_Envelope(
                        ST_Buffer(
                            ST_Centroid(ST_GeomFromGeoJSON($2))::geography, 50
                        )::geometry
                    ),
                    4326
                ),
                $3, $4, $5
            )
            RETURNING id
            """,
            session_id,
            r["geom"],
            r["crop_class"],
            r["confidence"],
            r["year"],
        )
        _ = exists  # idempotency probe kept for clarity; ON CONFLICT guards features
        emb_literal = "[" + ",".join(f"{x:.6f}" for x in r["embedding"]) + "]"
        await conn.execute(
            """
            INSERT INTO features_parcels
                (parcel_id, year, alphaearth_embedding, phenology,
                 sog_doy, peak_doy, peak_value, senescence_doy, ndvi_auc,
                 ndvi_slope_pre_peak, ndvi_slope_post_peak, maturity_duration_days)
            VALUES ($1, $2, $3::vector, $4::jsonb,
                    $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (parcel_id, year) DO NOTHING
            """,
            parcel_id,
            r["year"],
            emb_literal,
            json.dumps(r["phenology"]),
            _as_int(r["phenology"].get("sog_doy")),
            _as_int(r["phenology"].get("peak_doy")),
            _as_float(r["phenology"].get("peak_value")),
            _as_int(r["phenology"].get("senescence_doy")),
            _as_float(r["phenology"].get("ndvi_auc")),
            _as_float(r["phenology"].get("ndvi_slope_pre_peak")),
            _as_float(r["phenology"].get("ndvi_slope_post_peak")),
            _as_int(r["phenology"].get("maturity_duration_days")),
        )
        inserted += 1 if not exists else 0
    return str(session_id), inserted


def _as_int(value: object) -> int | None:
    """Coerce a value to ``int`` or ``None``."""
    return int(value) if value is not None else None


def _as_float(value: object) -> float | None:
    """Coerce a value to ``float`` or ``None``."""
    return float(value) if value is not None else None


async def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument vector.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Seed demo parcels from real PASTIS data.")
    parser.add_argument("--limit", type=int, default=12, help="Number of demo parcels.")
    args = parser.parse_args(argv)

    from backend.app.core.config import get_settings

    dsn = to_asyncpg_dsn(get_settings().database_url)
    rows = _build_rows(args.limit)
    logger.info("seed_demo_rows_built", n=len(rows))

    conn = await asyncpg.connect(dsn)
    try:
        session_id, inserted = await _seed(conn, rows)
    finally:
        await conn.close()
    print(
        f"Seeded demo session {session_id}: {inserted} new parcels "
        f"({len(rows)} total real parcels)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
