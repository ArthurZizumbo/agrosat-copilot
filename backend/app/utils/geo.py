"""Geospatial helpers: GeoJSON <-> shapely and area in hectares.

Shared so the AOI service stays thin (DRY, root rule). All polygons are WGS84
(EPSG:4326). Area is computed via an equal-area reprojection (no external CRS
deps) using a spherical approximation accurate enough for AOI-scale fields.
"""

from __future__ import annotations

import json
import math
from typing import Any

from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

_EARTH_RADIUS_M = 6_371_008.8  # mean Earth radius (IUGG)


def geojson_to_shape(geometry: dict[str, Any]) -> BaseGeometry:
    """Build a shapely geometry from a GeoJSON geometry mapping.

    Raises:
        ValueError: if the geometry is not a valid Polygon.
    """
    geom = shape(geometry)
    if geom.geom_type != "Polygon":
        raise ValueError(f"expected a Polygon geometry, got {geom.geom_type!r}")
    if not geom.is_valid:
        raise ValueError("polygon geometry is invalid (self-intersection?)")
    return geom


def shape_to_geojson(geom: BaseGeometry) -> dict[str, Any]:
    """Serialize a shapely geometry to a GeoJSON geometry mapping."""
    return dict(mapping(geom))


def geometry_to_geojson_str(geometry: dict[str, Any]) -> str:
    """Serialize a GeoJSON geometry mapping to a compact JSON string."""
    return json.dumps(geometry, separators=(",", ":"))


def polygon_area_ha(geom: BaseGeometry) -> float:
    """Approximate area of a WGS84 polygon in hectares.

    Projects coordinates to a local equirectangular plane centred on the polygon
    centroid (metres) and applies the shoelace formula. Accurate to well under
    1% at field scale, with no pyproj dependency. Subtracts interior rings.
    """
    lon0 = geom.centroid.x
    cos_lat0 = math.cos(math.radians(geom.centroid.y))

    def _ring_area_m2(coords: list[tuple[float, float]]) -> float:
        pts = [
            (
                math.radians(lon - lon0) * _EARTH_RADIUS_M * cos_lat0,
                math.radians(lat) * _EARTH_RADIUS_M,
            )
            for lon, lat in coords
        ]
        total = 0.0
        for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1], strict=True):
            total += x1 * y2 - x2 * y1
        return abs(total) / 2.0

    polygon = geom  # type: ignore[assignment]
    area = _ring_area_m2(list(polygon.exterior.coords))  # type: ignore[attr-defined]
    for interior in polygon.interiors:  # type: ignore[attr-defined]
        area -= _ring_area_m2(list(interior.coords))
    return area / 10_000.0
