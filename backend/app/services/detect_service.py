"""Detect service: on-demand crop map of an AOI from the trained classifier.

This realises option B of the segmentation discussion. Instead of displaying a
pre-loaded parcel catalogue, it produces *what the model detects* in the zone the
user drew:

1. tile the AOI bbox into a grid of cells and keep the cells inside the polygon;
2. sample each cell's mean AlphaEarth embedding live from Earth Engine, in ONE
   ``reduceRegions`` call (:func:`ml.ingest.gee_sampler.sample_alphaearth_cells`);
3. classify each cell with the SAME ``xgb-alphaearth`` model the ``/chat`` reasoner
   serves (:func:`ml.agent.tools.classify._load_classifier`), restricted to the
   active label-space (``france-9``);
4. merge adjacent same-crop cells into polygons (Shapely ``unary_union``) and
   return them as a GeoJSON ``FeatureCollection``.

It is a semantic crop map (per-cell classification), NOT the heavy temporal
instance segmentation (U-TAE / TSViT) -- that is the deferred Full GPU path
(US-056, ADR-009 / ADR-012). This runs CPU-light + on-demand for any AOI.

Heavy work (the Earth Engine round-trip + the XGBoost forward passes) runs in a
worker thread so the request event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING

import numpy as np
import structlog
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

from backend.app.core.config import Settings, get_settings
from backend.app.models.detect import (
    DetectionFeature,
    DetectionFeatureCollection,
    DetectionProperties,
)
from ml.agent.schemas import GeoJSONGeometry

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

logger = structlog.get_logger(__name__)

__all__ = ["DetectService"]

#: Number of cells per side of the grid tiling the AOI bbox. 16 -> up to 256
#: cells before the in-polygon filter; a balance of detail vs Earth Engine cost.
_GRID_SIDE: int = 16
#: Hard cap on cells actually sampled (defence against a huge AOI bbox).
_MAX_CELLS: int = 400
#: Crop classes that are not a real detection -- never emitted as a region.
_NON_CROP: frozenset[str] = frozenset({"unresolved", "needs_gee_sampling"})

#: Guard so Earth Engine is initialised at most once per process (ADC-based).
_EE_INITIALISED: bool = False


class DetectService:
    """Produce an on-demand crop map (detected crop polygons) for an AOI."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialise with typed settings (injected, never read from os.environ)."""
        self._settings = settings or get_settings()

    async def crop_map(
        self, aoi: GeoJSONGeometry, year: int, *, grid_side: int = _GRID_SIDE
    ) -> DetectionFeatureCollection:
        """Detect crops over ``aoi`` and return them as merged GeoJSON polygons.

        Args:
            aoi: AOI polygon to analyse (EPSG:4326).
            year: Campaign year of the AlphaEarth annual embedding.
            grid_side: Cells per side of the grid tiling the AOI bbox.

        Returns:
            A :class:`DetectionFeatureCollection`; empty when the AOI has no
            AlphaEarth coverage or the model resolves no crop in it.
        """
        polygon = shape({"type": aoi.type, "coordinates": aoi.coordinates})
        cells = self._build_cells(polygon, grid_side)
        if not cells:
            return DetectionFeatureCollection(features=[], count=0)

        bboxes = [c.bounds for c in cells]
        project = self._settings.gee_project_id or None
        embeddings = await asyncio.to_thread(_sample_cells, bboxes, year, project)

        regions = await asyncio.to_thread(_classify_and_merge, cells, embeddings)
        features = [
            DetectionFeature(
                geometry=mapping(geom),
                properties=DetectionProperties(
                    crop_class=crop, confidence=round(conf, 4), n_cells=n
                ),
            )
            for crop, geom, conf, n in regions
        ]
        logger.info(
            "detect_crop_map",
            year=year,
            n_cells=len(cells),
            n_regions=len(features),
        )
        return DetectionFeatureCollection(features=features, count=len(features))

    @staticmethod
    def _build_cells(polygon: BaseGeometry, grid_side: int) -> list[BaseGeometry]:
        """Tile the polygon's bbox into a grid and keep the cells inside it.

        A cell is kept when its centroid falls within the AOI polygon, so the
        detection follows the drawn shape rather than its rectangular bounds.
        """
        minx, miny, maxx, maxy = polygon.bounds
        if maxx <= minx or maxy <= miny:
            return []
        side = max(1, min(grid_side, math.isqrt(_MAX_CELLS)))
        cell_w = (maxx - minx) / side
        cell_h = (maxy - miny) / side
        cells: list[BaseGeometry] = []
        for row in range(side):
            for col in range(side):
                x0 = minx + col * cell_w
                y0 = miny + row * cell_h
                cell = box(x0, y0, x0 + cell_w, y0 + cell_h)
                if polygon.contains(cell.centroid):
                    cells.append(cell)
        return cells[:_MAX_CELLS]


def _sample_cells(
    bboxes: list[tuple[float, float, float, float]], year: int, project: str | None
) -> list[list[float] | None]:
    """Blocking Earth Engine sampling (runs in a worker thread)."""
    global _EE_INITIALISED
    from ml.ingest.gee_sampler import init_ee, sample_alphaearth_cells

    if not _EE_INITIALISED:
        init_ee(project=project)
        _EE_INITIALISED = True
    return sample_alphaearth_cells(bboxes, year)


def _classify_and_merge(
    cells: list[BaseGeometry], embeddings: list[list[float] | None]
) -> list[tuple[str, BaseGeometry, float, int]]:
    """Classify each cell and merge same-crop cells into polygons (worker thread).

    Returns a list of ``(crop_class, merged_geometry, mean_confidence, n_cells)``.
    Cells without an embedding or that resolve to no real crop are dropped.
    """
    from ml.agent.tools.classify import _build_result, _load_classifier
    from ml.eval.class_remap import get_label_space

    classifier = _load_classifier()
    label_space = get_label_space("france-9")

    grouped: dict[str, list[tuple[BaseGeometry, float]]] = {}
    for cell, embedding in zip(cells, embeddings, strict=True):
        if embedding is None:
            continue
        proba = classifier.predict_proba_18(np.asarray(embedding, dtype=np.float64))
        result = _build_result(
            proba, classifier.class_names, restrict=True, label_space=label_space
        )
        if result.crop_class in _NON_CROP or result.confidence <= 0.0:
            continue
        grouped.setdefault(result.crop_class, []).append((cell, result.confidence))

    regions: list[tuple[str, BaseGeometry, float, int]] = []
    for crop, members in grouped.items():
        merged = unary_union([cell for cell, _ in members])
        mean_conf = float(np.mean([conf for _, conf in members]))
        regions.append((crop, merged, mean_conf, len(members)))
    return regions
