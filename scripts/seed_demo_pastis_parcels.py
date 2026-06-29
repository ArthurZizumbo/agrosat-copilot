"""Re-seed the demo session with REAL PASTIS-R fold-5 parcels (Voting-3 backed).

Replaces the demo session's placeholder parcels with real PASTIS-R fold-5 parcels
so the OOF-backed copilot tools work on coherent real data:

- the **Voting-3 perceiver** scores them from the real fold-5 OOF (the EPIC 12
  deployment champion, france-10 F1 0.9069),
- **compare_models** returns the real per-member predictions (the three Voting-3
  members + FarSLIP), and
- the Spatial-RAG corpus (the same PASTIS-R region) has real neighbours,

all bridged by ``parcels.canonical_parcel_id`` (the US-079 migration). The numeric
cast of ``parcels.id`` never matches a ``"{patch}_{local}"`` OOF key, which is why
the previous placeholder parcels always degraded.

Selection: one parcel per well-resolved ``france-9`` class (for demo variety) that
is present in the Voting-3 members' AND FarSLIP's fold-5 OOF. The geometry is the
real PASTIS patch centroid (reprojected 2154 -> 4326), so the parcels sit in
Normandy with the corpus. ``crop_class`` / ``confidence`` are the real Voting-3
``france-9`` argmax. No synthetic values.

Idempotent: deletes the target session's parcels (CASCADE features) and re-inserts.

Run: ``poetry run python scripts/seed_demo_pastis_parcels.py`` (Postgres up,
migrations applied, ``dvc pull ml/eval/oof data/PASTIS-R`` for the OOF + geometry).
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Final

import asyncpg
import polars as pl
import structlog

logger = structlog.get_logger(__name__)

DEFAULT_DATABASE_URL: Final[str] = "postgresql://agrosat:agrosat@localhost:5432/agrosat"
DEMO_USER: Final[str] = "demo@agrosat.dev"
DEMO_YEAR: Final[int] = 2019
PASTIS_SRID: Final[int] = 2154
#: FarSLIP member whose fold-5 OOF must also cover a parcel (so compare_models can
#: surface FarSLIP next to the three Voting-3 members).
FARSLIP_MEMBER: Final[str] = "farslip-ft18"
#: Half-width (degrees) of the small square footprint built around each real
#: parcel centroid (~0.6 km), enough for the AOI tools without overlapping.
FOOTPRINT_DEG: Final[float] = 0.006


def _resolve_database_url() -> str:
    """Return the Postgres URL normalized for asyncpg (drops the driver suffix)."""
    raw_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    if raw_url.startswith("postgresql+asyncpg://"):
        return raw_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    return raw_url


def _load_patch_geometries() -> dict[int, str]:
    """Load PASTIS patch geometries as ``{ID_PATCH: geojson_str}`` (EPSG:2154).

    Mirrors ``scripts/ingest_rag_documents._load_patch_geometries`` (inlined to
    avoid a cross-script package import). PostGIS reprojects each geometry to 4326
    at insert time.

    Returns:
        Mapping of integer patch id to its GeoJSON geometry string.

    Raises:
        FileNotFoundError: if the PASTIS metadata geojson is absent.
    """
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data" / "PASTIS-R" / "metadata.geojson"
    if not path.exists():
        raise FileNotFoundError(
            f"PASTIS patch geometries not found: {path}. Run `dvc pull data/PASTIS-R`."
        )
    with path.open(encoding="utf-8") as handle:
        collection = json.load(handle)
    geom_by_patch: dict[int, str] = {}
    for feature in collection.get("features", []):
        patch_id = feature.get("properties", {}).get("ID_PATCH")
        geometry = feature.get("geometry")
        if patch_id is not None and geometry is not None:
            geom_by_patch[int(patch_id)] = json.dumps(geometry)
    return geom_by_patch


def _select_parcels(n_per_class_cap: int = 1, max_parcels: int = 12) -> list[dict]:
    """Pick real fold-5 parcels: one per france-9 class, Voting-3 + FarSLIP backed.

    Args:
        n_per_class_cap: How many parcels to keep per resolved class (1 = variety).
        max_parcels: Hard cap on the number of parcels selected.

    Returns:
        Per-parcel dicts ``{canonical_id, patch_id, crop_class, confidence}`` with
        the real Voting-3 ``france-9`` argmax label and probability.
    """
    from ml.agent.tools import classify
    from ml.eval.class_remap import get_label_space, restrict_posterior

    voting = classify._load_voting_three()
    label_space = get_label_space("france-9")

    farslip_path = classify._OOF_DIR / f"oof_parcel_{FARSLIP_MEMBER}_fold5.parquet"
    farslip_ids: set[str] = set()
    if farslip_path.exists():
        from ml.utils.parcel_id import canonical_parcel_id

        farslip_ids = set(
            canonical_parcel_id(pl.read_parquet(farslip_path), col="canonical_parcel_id")[
                "canonical_parcel_id"
            ].to_list()
        )
    else:
        logger.warning("seed_pastis_no_farslip_oof", path=str(farslip_path))

    selected: list[dict] = []
    per_class: dict[str, int] = {}
    for canonical_id in sorted(voting.member_probs_by_id):
        if farslip_ids and canonical_id not in farslip_ids:
            continue
        proba = voting.posterior_for_parcel(canonical_id)
        if proba is None:
            continue
        restricted = restrict_posterior(proba, label_space)
        if not restricted:
            continue
        cid = max(restricted, key=lambda k: restricted[k])
        name = label_space.class_names.get(cid, str(cid))
        if per_class.get(name, 0) >= n_per_class_cap:
            continue
        per_class[name] = per_class.get(name, 0) + 1
        selected.append(
            {
                "canonical_id": canonical_id,
                "patch_id": int(canonical_id.split("_", 1)[0]),
                "crop_class": name,
                "confidence": float(restricted[cid]),
            }
        )
        if len(selected) >= max_parcels:
            break
    logger.info("seed_pastis_selected", n=len(selected), classes=sorted(per_class))
    return selected


async def _replace_session_parcels(
    conn: asyncpg.Connection, session_id: str, parcels: list[dict], patch_geoms: dict[int, str]
) -> int:
    """Delete the session's parcels and insert the selected PASTIS-R parcels.

    Args:
        conn: Live asyncpg connection (RLS primed for ``session_id``).
        session_id: Target demo session.
        parcels: Selected parcel dicts from :func:`_select_parcels`.
        patch_geoms: ``{patch_id: geojson_str}`` PASTIS patch geometries (EPSG:2154).

    Returns:
        The number of parcels inserted.
    """
    insert_sql = """
        INSERT INTO parcels
            (session_id, canonical_parcel_id, geom, crop_class, confidence, year)
        VALUES (
            $1, $2,
            -- real PASTIS patch centroid (2154 -> 4326), expanded to a small square
            -- so the AOI tools have a simple Polygon at the real Normandy location.
            ST_Expand(
                ST_Centroid(ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON($3), $4), 4326)),
                $5
            ),
            $6, $7, $8
        )
    """
    inserted = 0
    async with conn.transaction():
        await conn.execute("DELETE FROM parcels WHERE session_id = $1", session_id)
        for parcel in parcels:
            geom_geojson = patch_geoms.get(parcel["patch_id"])
            if geom_geojson is None:
                logger.warning("seed_pastis_skip_no_geom", **parcel)
                continue
            await conn.execute(
                insert_sql,
                session_id,
                parcel["canonical_id"],
                geom_geojson,
                PASTIS_SRID,
                FOOTPRINT_DEG,
                parcel["crop_class"],
                parcel["confidence"],
                DEMO_YEAR,
            )
            inserted += 1
    return inserted


async def main() -> int:
    """Re-seed the demo session with real PASTIS-R Voting-3 parcels. Idempotent."""
    structlog.configure(processors=[structlog.processors.JSONRenderer()])

    parcels = _select_parcels()
    if not parcels:
        sys.stderr.write("No PASTIS-R parcels selected (check OOF / dvc pull).\n")
        return 1

    patch_geoms = _load_patch_geometries()

    conn = await asyncpg.connect(_resolve_database_url())
    try:
        session_id = await conn.fetchval(
            "SELECT cs.id FROM chat_sessions cs "
            "LEFT JOIN parcels p ON p.session_id = cs.id "
            "WHERE cs.user_id = $1 "
            "GROUP BY cs.id ORDER BY count(p.id) DESC, cs.id LIMIT 1",
            DEMO_USER,
        )
        if session_id is None:
            sys.stderr.write(f"No demo session for {DEMO_USER}. Run `make db-seed` first.\n")
            return 1
        # Prime RLS so the INSERT WITH CHECK (session_id = app.current_session) passes.
        await conn.execute("SELECT set_config('app.current_session', $1, false)", str(session_id))
        count = await _replace_session_parcels(conn, str(session_id), parcels, patch_geoms)
        logger.info("seed_pastis_done", session_id=str(session_id), parcels=count)
        print(f"re-seeded {count} real PASTIS-R parcels into demo session {session_id}")
        return 0
    finally:
        await conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
