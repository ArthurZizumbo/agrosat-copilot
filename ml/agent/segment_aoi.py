"""Live per-cell crop segmentation of a drawn AOI (honest, anywhere).

The conversational ``classify``/perceiver path reduces an AOI to a SINGLE mean
AlphaEarth embedding -> one dominant crop. That is correct but cannot paint
parcels. This module produces the per-cell crop map the UI needs: it samples the
64-dim AlphaEarth embedding on a regular grid INSIDE the drawn polygon, runs the
SAME deployed XGBoost-AlphaEarth classifier per cell, and vectorizes the cell
class map into merged polygons (one per contiguous same-crop region) in
EPSG:4326. It is the deployed model applied pixel-wise -- no segmentation model
is trained or invoked; the resolution is the grid step (coarse but honest).

Works for ANY drawn AOI (not just the benchmarked PASTIS patches): the embeddings
are sampled live from Earth Engine. It is best-effort -- any failure (EE missing,
no credentials, no coverage, classifier artifacts absent) returns an empty list
so the caller (the perceiver) degrades to the AOI-level estimate.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise
from math import cos, radians
from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

__all__ = ["segment_aoi_live"]

#: Grid cells per axis. ~22x22 = 484 sample points keeps the live GEE sampling to
#: a single batch (~500) so the perceiver latency stays bounded; the cell size
#: adapts to the AOI extent (a 1.5 km AOI -> ~67 m cells, a 0.4 km AOI -> ~20 m).
_GRID_N: int = 22

#: Hard cap on emitted segments (largest first) so a noisy classification cannot
#: flood the map / LLM context with hundreds of tiny polygons.
_MAX_SEGMENTS: int = 60


def _meters_per_degree(lat_deg: float) -> tuple[float, float]:
    """Return approximate (metres/deg lon, metres/deg lat) at ``lat_deg``."""
    m_per_deg_lat = 110_540.0
    m_per_deg_lon = 111_320.0 * cos(radians(lat_deg))
    return m_per_deg_lon, m_per_deg_lat


def _polygon_area_ha(ring: list[list[float]], lat_deg: float) -> float:
    """Approximate the area (hectares) of a lon/lat ring via local equirectangular.

    Args:
        ring: Closed exterior ring as ``[[lon, lat], ...]``.
        lat_deg: Reference latitude for the metres-per-degree scaling.

    Returns:
        The shoelace area scaled to hectares (approximate, fine for a label).
    """
    m_lon, m_lat = _meters_per_degree(lat_deg)
    pts = [(x * m_lon, y * m_lat) for x, y in ring]
    area2 = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in pairwise(pts))
    return abs(area2) * 0.5 / 10_000.0


def segment_aoi_live(
    geometry: dict[str, Any],
    year: int,
    *,
    project: str | None = None,
    service_account_json: Path | None = None,
    grid_n: int = _GRID_N,
    scale: int = 10,
) -> list[dict[str, Any]]:
    """Segment a drawn AOI into per-crop polygons via live AlphaEarth + XGBoost.

    Pipeline (all best-effort, never raises):

    1. Build a ``grid_n x grid_n`` grid of cell centres clipped to the polygon.
    2. Sample the 64-dim annual AlphaEarth embedding at each centre from Earth
       Engine (``sample_alphaearth_at_coords``, one batched server call).
    3. Classify every cell with the cached XGBoost-AlphaEarth classifier
       (``classify._load_classifier``) -> a ``(grid_n, grid_n)`` semantic18 class
       grid (cells without coverage stay nodata).
    4. Vectorize the class grid (``rasterio.features.shapes``) into merged
       polygons in EPSG:4326, one per contiguous same-crop region.

    Args:
        geometry: GeoJSON geometry (``{"type": "Polygon", "coordinates": ...}``)
            in EPSG:4326 delimiting the AOI.
        year: Campaign year of the annual embedding (2017-2025).
        project: GCP project id with the EE quota.
        service_account_json: Optional service-account key; falls back to ADC.
        grid_n: Grid cells per axis.
        scale: Sampling resolution in metres (AlphaEarth native = 10).

    Returns:
        A list of segment dicts ``{crop_class, confidence, area_ha, geometry}``
        (largest first), or an empty list on any failure / no coverage.
    """
    try:
        import polars as pl
        import rasterio.features
        from rasterio.transform import from_bounds
        from shapely.geometry import Point, shape

        from ml.agent.tools.classify import _load_classifier
        from ml.data.pastis_filter import SEMANTIC18_CLASS_NAMES
        from ml.ingest.gee_sampler import (
            ALPHAEARTH_DIM_COLS,
            init_ee,
            sample_alphaearth_at_coords,
        )

        poly = shape(geometry)
        if poly.is_empty:
            return []
        minx, miny, maxx, maxy = (float(v) for v in poly.bounds)
        if maxx <= minx or maxy <= miny:
            return []

        nx = ny = max(4, int(grid_n))
        cell_w = (maxx - minx) / nx
        cell_h = (maxy - miny) / ny

        px_ids: list[str] = []
        lons: list[float] = []
        lats: list[float] = []
        for iy in range(ny):
            lat = maxy - (iy + 0.5) * cell_h  # row 0 at the top (maxy)
            for ix in range(nx):
                lon = minx + (ix + 0.5) * cell_w
                if poly.contains(Point(lon, lat)):
                    px_ids.append(f"{iy}_{ix}")
                    lons.append(lon)
                    lats.append(lat)
        if not px_ids:
            return []

        # Initialise EE (idempotent) before the point sampler (which assumes it).
        init_ee(service_account_json=service_account_json, project=project)
        coords = pl.DataFrame({"px_id": px_ids, "lon": lons, "lat": lats})
        cache_key = f"seg_{minx:.4f}_{miny:.4f}_{maxx:.4f}_{maxy:.4f}"
        sampled = sample_alphaearth_at_coords(
            coords, year=year, cache_key=cache_key, scale=scale
        )
        if sampled.is_empty():
            return []

        dim_cols = sorted(ALPHAEARTH_DIM_COLS)  # dim_00..dim_63, classifier order
        sampled = sampled.drop_nulls(subset=dim_cols)
        if sampled.is_empty():
            return []

        classifier = _load_classifier()
        embeddings = sampled.select(dim_cols).to_numpy().astype(np.float64)
        embeddings = np.where(np.isfinite(embeddings), embeddings, 0.0)
        estimator: Any = classifier.estimator
        proba_local = np.asarray(estimator.predict_proba(embeddings), dtype=np.float64)
        full = np.zeros((embeddings.shape[0], 18), dtype=np.float64)
        for col, gid in enumerate(classifier.global_classes):
            gid_int = int(gid)
            if 0 <= gid_int < 18:
                full[:, gid_int] = proba_local[:, col]
        pred = full.argmax(axis=1).astype(np.int16)
        conf = full.max(axis=1)

        class_grid = np.full((ny, nx), -1, dtype=np.int16)
        conf_sum: dict[int, float] = defaultdict(float)
        conf_cnt: dict[int, int] = defaultdict(int)
        for row_px, cid, cval in zip(
            sampled.get_column("px_id").to_list(), pred, conf, strict=True
        ):
            iy_str, ix_str = str(row_px).split("_")
            iy, ix = int(iy_str), int(ix_str)
            class_grid[iy, ix] = cid
            conf_sum[int(cid)] += float(cval)
            conf_cnt[int(cid)] += 1

        transform = from_bounds(minx, miny, maxx, maxy, nx, ny)
        mask = class_grid >= 0
        mid_lat = (miny + maxy) / 2.0

        segments: list[dict[str, Any]] = []
        for geom, value in rasterio.features.shapes(
            class_grid, mask=mask, transform=transform
        ):
            cid = int(value)
            if cid < 0 or cid >= len(SEMANTIC18_CLASS_NAMES):
                continue
            rings = [
                [[round(x, 6), round(y, 6)] for x, y in ring]
                for ring in geom["coordinates"]
            ]
            class_conf = conf_sum[cid] / conf_cnt[cid] if conf_cnt[cid] else None
            segments.append(
                {
                    "crop_class": SEMANTIC18_CLASS_NAMES[cid],
                    "confidence": (
                        round(float(class_conf), 3) if class_conf is not None else None
                    ),
                    "area_ha": round(_polygon_area_ha(rings[0], mid_lat), 2),
                    "geometry": {"type": "Polygon", "coordinates": rings},
                }
            )

        segments.sort(key=lambda s: s["area_ha"] or 0.0, reverse=True)
        logger.info(
            "segment_aoi_done",
            year=int(year),
            n_points=len(px_ids),
            n_segments=len(segments),
        )
        return segments[:_MAX_SEGMENTS]
    except Exception as exc:  # noqa: BLE001 - best-effort overlay; degrade to AOI estimate
        logger.warning("segment_aoi_failed", year=int(year), error=str(exc))
        return []
