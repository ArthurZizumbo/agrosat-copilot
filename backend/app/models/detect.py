"""Pydantic response models for the ``/detect`` crop-map endpoint.

The endpoint runs the trained ``xgb-alphaearth`` classifier over a grid of cells
tiling the requested AOI and merges same-crop cells into polygons -- i.e. it
returns *what the model detected* in the zone, as GeoJSON, not a pre-loaded
catalogue. These models shape that GeoJSON ``FeatureCollection`` so the router
returns typed objects (never a raw dict assembled in the handler).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

__all__ = [
    "DetectionFeature",
    "DetectionFeatureCollection",
    "DetectionProperties",
]

_FORBID = ConfigDict(extra="forbid")


class DetectionProperties(BaseModel):
    """Per-polygon attributes of a detected crop region.

    Attributes:
        crop_class: Crop the model assigned to the merged region (resolved
            label-space, e.g. ``france-9``).
        confidence: Mean top-class probability over the cells of the region.
        n_cells: Number of grid cells merged into the region (a size/density cue).
    """

    model_config = _FORBID

    crop_class: str
    confidence: float
    n_cells: int


class DetectionFeature(BaseModel):
    """A GeoJSON ``Feature``: one detected crop region.

    Attributes:
        geometry: GeoJSON geometry (``Polygon`` / ``MultiPolygon``) in EPSG:4326,
            the union of the region's grid cells.
        properties: The region's crop / confidence / size attributes.
    """

    model_config = _FORBID

    type: Literal["Feature"] = "Feature"
    geometry: dict
    properties: DetectionProperties


class DetectionFeatureCollection(BaseModel):
    """The ``/detect`` response: every crop region the model found in the AOI.

    Attributes:
        features: One :class:`DetectionFeature` per merged crop region.
        count: Number of regions returned.
    """

    model_config = _FORBID

    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[DetectionFeature]
    count: int
