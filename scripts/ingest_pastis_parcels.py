"""Ingest a real PASTIS parcel mosaic into ``parcels`` + ``features_parcels``.

Why this exists
---------------
The operational ``parcels`` table is multi-tenant (one row per ``session_id``);
it is NOT a global cadastre, so out of the box it only holds the four hand-seeded
Tuscany demo parcels (``scripts/seed_demo_parcels.py``). That makes ``list_parcels``
and the map look empty for any real exploration. This loader populates the table
with a contiguous block of REAL PASTIS parcels (Brittany, France) so the map and
``list_parcels`` show a believable universe.

Data sources (already materialised under ``data/``)
---------------------------------------------------
- ``data/features/features_fused_pastis.parquet`` -- one row per PASTIS parcel
  (``patch_id``/``instance_id``), carrying the real ``class_name`` label, the
  parcel ``area_m2`` and the 64-dim AlphaEarth embedding (``dim_00..dim_63``) the
  tabular classifier was trained on. So a parcel ingested here is classifiable /
  explainable instantly and consistently with the model.
- ``data/PASTIS-R/metadata.geojson`` -- one MultiPolygon per 1.28 km patch tile,
  in EPSG:2154 (Lambert-93, metres). PASTIS ships no per-parcel vector geometry,
  so each parcel's polygon is APPROXIMATED as a cell of a regular grid that tiles
  its real patch footprint. The location (patch) is real; the boundary is a grid
  approximation (true boundaries would require vectorising the instance rasters).

Geometry is written by letting PostGIS reproject the Lambert-93 WKT to EPSG:4326
(``ST_Transform(ST_GeomFromText(wkt, 2154), 4326)``), so no ``pyproj`` is needed.

Idempotency
-----------
Parcels are tagged with a dedicated "catalog" AOI row (``aoi_id``) per session.
A re-run deletes the session's parcels carrying that ``aoi_id`` (features cascade)
and re-inserts, so the loader is safe to run repeatedly and never touches the
Tuscany demo parcels (which carry the demo AOI id).

Run
---
``poetry run python scripts/ingest_pastis_parcels.py`` (Postgres up + migrations
applied). Tunables via env: ``PARCELS_SESSION_ID`` (default: the demo session),
``PARCELS_MAX`` (default 600), ``PARCELS_YEAR`` (default 2019).
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from typing import Final

import asyncpg
import polars as pl
import structlog

logger = structlog.get_logger(__name__)

DEFAULT_DATABASE_URL: Final[str] = "postgresql://agrosat:agrosat@localhost:55432/agrosat"
#: Demo session seeded by ``scripts/seed.py``; default target so the existing
#: ``agrosat-session-id`` cookie workaround sees the ingested parcels.
DEFAULT_SESSION_ID: Final[str] = "f5849b8e-cd2c-4453-86bc-a01754a5ec19"
#: Year tag stored on parcels/features. Matches the year the frontend sends on
#: POST /chat (so the perceiver's spatial-anchored embedding lookup hits).
DEFAULT_YEAR: Final[int] = 2019
#: Cap on ingested parcels (map render + listing stay light).
DEFAULT_MAX_PARCELS: Final[int] = 600

#: Idempotency marker: the AOI row whose id tags every parcel this loader writes.
CATALOG_AOI_LABEL: Final[str] = "PASTIS catalog - France"

PARQUET_PATH: Final[str] = "data/features/features_fused_pastis.parquet"
GEOJSON_PATH: Final[str] = "data/PASTIS-R/metadata.geojson"

#: Fraction of a grid cell kept as a gap on each side, so adjacent parcels read
#: as separate fields rather than one solid block.
_CELL_INSET: Final[float] = 0.08


def _resolve_database_url() -> str:
    """Return the Postgres URL normalised for asyncpg (drops the driver suffix)."""
    raw_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    if raw_url.startswith("postgresql+asyncpg://"):
        return raw_url.replace("postgresql+asyncpg://", "postgresql://", 1)
    return raw_url


def _accumulate_coords(node: object, xs: list[float], ys: list[float]) -> None:
    """Recursively collect x/y positions from a nested GeoJSON coordinate array."""
    if (
        isinstance(node, list)
        and len(node) >= 2
        and isinstance(node[0], (int, float))
        and isinstance(node[1], (int, float))
    ):
        xs.append(float(node[0]))
        ys.append(float(node[1]))
    elif isinstance(node, list):
        for child in node:
            _accumulate_coords(child, xs, ys)


def _patch_bboxes() -> dict[int, tuple[float, float, float, float]]:
    """Map ``ID_PATCH`` -> Lambert-93 bbox ``(minx, miny, maxx, maxy)``.

    The bbox of each patch's MultiPolygon footprint is enough: parcels are placed
    as a grid inside it.
    """
    with open(GEOJSON_PATH, encoding="utf-8") as handle:
        collection = json.load(handle)

    bboxes: dict[int, tuple[float, float, float, float]] = {}
    for feature in collection["features"]:
        patch_id = int(feature["properties"]["ID_PATCH"])
        xs: list[float] = []
        ys: list[float] = []
        _accumulate_coords(feature["geometry"]["coordinates"], xs, ys)
        if xs and ys:
            bboxes[patch_id] = (min(xs), min(ys), max(xs), max(ys))
    return bboxes


def _select_contiguous_patches(
    parcels_per_patch: dict[int, int],
    bboxes: dict[int, tuple[float, float, float, float]],
    max_parcels: int,
) -> list[int]:
    """Pick a spatially compact block of patches totalling ~``max_parcels``.

    Starts from the densest patch and greedily adds the nearest patches by
    centroid distance, so the ingested parcels form one contiguous mosaic the
    user can pan to (rather than dots scattered across Brittany).
    """
    candidates = {p: n for p, n in parcels_per_patch.items() if p in bboxes and n > 0}
    if not candidates:
        return []

    def _centroid(patch_id: int) -> tuple[float, float]:
        minx, miny, maxx, maxy = bboxes[patch_id]
        return ((minx + maxx) / 2.0, (miny + maxy) / 2.0)

    seed = max(candidates, key=lambda p: candidates[p])
    seed_cx, seed_cy = _centroid(seed)
    ordered = sorted(
        candidates,
        key=lambda p: math.dist(_centroid(p), (seed_cx, seed_cy)),
    )

    selected: list[int] = []
    total = 0
    for patch_id in ordered:
        selected.append(patch_id)
        total += candidates[patch_id]
        if total >= max_parcels:
            break
    return selected


def _cell_wkt(
    bbox: tuple[float, float, float, float], cols: int, rows: int, index: int
) -> str:
    """Build the Lambert-93 POLYGON WKT for parcel ``index`` in a ``cols``x``rows`` grid."""
    minx, miny, maxx, maxy = bbox
    cell_w = (maxx - minx) / cols
    cell_h = (maxy - miny) / rows
    col = index % cols
    row = index // cols
    inset_x = cell_w * _CELL_INSET
    inset_y = cell_h * _CELL_INSET
    x0 = minx + col * cell_w + inset_x
    x1 = minx + (col + 1) * cell_w - inset_x
    y0 = miny + row * cell_h + inset_y
    y1 = miny + (row + 1) * cell_h - inset_y
    return (
        f"POLYGON(({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))"
    )


def _load_parcels(
    max_parcels: int,
) -> tuple[list[dict[str, object]], int, tuple[float, float, float, float]]:
    """Resolve which parcels to ingest and assign each a grid-cell geometry.

    Returns the per-parcel records (with a Lambert-93 ``wkt`` and the AlphaEarth
    embedding rendered as a pgvector literal), the number of distinct patches the
    mosaic spans, and the Lambert-93 union bbox of the selected patches (used to
    build the catalog AOI the frontend flies to).
    """
    bboxes = _patch_bboxes()

    dim_cols = [f"dim_{i:02d}" for i in range(64)]
    frame = (
        pl.scan_parquet(PARQUET_PATH)
        .select(["patch_id", "class_name", "area_m2", *dim_cols])
        .collect()
    )

    counts = frame.group_by("patch_id").len().sort("len", descending=True)
    parcels_per_patch = {
        int(pid): int(n) for pid, n in zip(counts["patch_id"], counts["len"], strict=True)
    }
    selected = _select_contiguous_patches(parcels_per_patch, bboxes, max_parcels)
    selected_set = set(selected)

    sel_boxes = [bboxes[p] for p in selected_set]
    union_bbox = (
        (
            min(b[0] for b in sel_boxes),
            min(b[1] for b in sel_boxes),
            max(b[2] for b in sel_boxes),
            max(b[3] for b in sel_boxes),
        )
        if sel_boxes
        else (0.0, 0.0, 0.0, 0.0)
    )

    records: list[dict[str, object]] = []
    subset = frame.filter(pl.col("patch_id").is_in(list(selected_set)))
    for patch_id in selected:
        patch_df = subset.filter(pl.col("patch_id") == patch_id)
        n = patch_df.height
        if n == 0:
            continue
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        bbox = bboxes[patch_id]
        class_names = patch_df["class_name"].to_list()
        areas = patch_df["area_m2"].to_list()
        embeddings = patch_df.select(dim_cols).to_numpy()
        for index in range(n):
            embedding = "[" + ",".join(repr(float(v)) for v in embeddings[index]) + "]"
            records.append(
                {
                    "wkt": _cell_wkt(bbox, cols, rows, index),
                    "crop_class": str(class_names[index]),
                    "area_ha": float(areas[index]) / 10_000.0,
                    "embedding": embedding,
                }
            )
            if len(records) >= max_parcels:
                return records, len(selected_set), union_bbox
    return records, len(selected_set), union_bbox


async def _ensure_catalog_aoi(
    conn: asyncpg.Connection,
    session_id: str,
    bbox: tuple[float, float, float, float],
) -> int:
    """Return the catalog AOI id for the session, creating it if needed.

    The AOI geometry is the Lambert-93 union bbox of the mosaic, reprojected to
    EPSG:4326 (so the frontend can fly to it). The row's id is the idempotency tag
    every ingested parcel carries in ``parcels.aoi_id``.
    """
    existing = await conn.fetchval(
        "SELECT id FROM aois WHERE session_id = $1 AND label = $2 LIMIT 1",
        session_id,
        CATALOG_AOI_LABEL,
    )
    if existing is not None:
        return int(existing)

    minx, miny, maxx, maxy = bbox
    rect_wkt = (
        f"POLYGON(({minx} {miny}, {maxx} {miny}, {maxx} {maxy}, "
        f"{minx} {maxy}, {minx} {miny}))"
    )
    aoi_id: int = await conn.fetchval(
        # SRID 2154 (Lambert-93) is the fixed PASTIS source projection, not input.
        """
        INSERT INTO aois (session_id, label, geom)
        VALUES ($1, $2, ST_Transform(ST_GeomFromText($3, 2154), 4326))
        RETURNING id
        """,
        session_id,
        CATALOG_AOI_LABEL,
        rect_wkt,
    )
    return int(aoi_id)


async def _ingest(conn: asyncpg.Connection, session_id: str, year: int, max_parcels: int) -> int:
    """Replace the session's catalog parcels with a fresh PASTIS mosaic."""
    records, n_patches, union_bbox = _load_parcels(max_parcels)
    if not records:
        logger.error("ingest_pastis.no_records")
        return 0

    async with conn.transaction():
        # Scope writes to the tenant so the FORCE-RLS policies pass even when the
        # connecting role is not a BYPASSRLS superuser.
        await conn.execute("SELECT set_config('app.current_session', $1, true)", session_id)

        aoi_id = await _ensure_catalog_aoi(conn, session_id, union_bbox)
        await conn.execute(
            "DELETE FROM parcels WHERE session_id = $1 AND aoi_id = $2", session_id, aoi_id
        )

        for record in records:
            parcel_id: int = await conn.fetchval(
                # SRID 2154 (Lambert-93) is the fixed PASTIS source projection.
                """
                INSERT INTO parcels (session_id, aoi_id, geom, crop_class, area_ha, year)
                VALUES (
                    $1, $2,
                    ST_Transform(ST_GeomFromText($3, 2154), 4326),
                    $4, $5, $6
                )
                RETURNING id
                """,
                session_id,
                aoi_id,
                record["wkt"],
                record["crop_class"],
                record["area_ha"],
                year,
            )
            await conn.execute(
                """
                INSERT INTO features_parcels (parcel_id, year, alphaearth_embedding)
                VALUES ($1, $2, $3::vector)
                """,
                parcel_id,
                year,
                record["embedding"],
            )

    logger.info(
        "ingest_pastis.done",
        session_id=session_id,
        parcels=len(records),
        patches=n_patches,
        year=year,
    )
    return len(records)


async def main() -> int:
    """Ingest the PASTIS parcel mosaic for the target session. Idempotent."""
    structlog.configure(processors=[structlog.processors.JSONRenderer()])
    session_id = os.environ.get("PARCELS_SESSION_ID", DEFAULT_SESSION_ID)
    year = int(os.environ.get("PARCELS_YEAR", str(DEFAULT_YEAR)))
    max_parcels = int(os.environ.get("PARCELS_MAX", str(DEFAULT_MAX_PARCELS)))

    conn = await asyncpg.connect(_resolve_database_url())
    try:
        session_exists = await conn.fetchval(
            "SELECT 1 FROM chat_sessions WHERE id = $1", session_id
        )
        if session_exists is None:
            logger.error("ingest_pastis.no_session", session_id=session_id)
            sys.stderr.write(
                f"Session {session_id} not found. Run `make db-seed` first.\n"
            )
            return 1
        count = await _ingest(conn, session_id, year, max_parcels)
        return 0 if count > 0 else 1
    finally:
        await conn.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
