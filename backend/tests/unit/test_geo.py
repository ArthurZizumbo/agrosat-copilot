"""Unit tests for geo helpers (GeoJSON parsing + area)."""

from __future__ import annotations

import pytest

from backend.app.utils.geo import (
    geojson_to_shape,
    geometry_to_geojson_str,
    polygon_area_ha,
)

# ~1 km x ~1 km square near the equator-ish latitude used by the demo (Tuscany).
_SQUARE = {
    "type": "Polygon",
    "coordinates": [
        [
            [11.10, 43.30],
            [11.11, 43.30],
            [11.11, 43.31],
            [11.10, 43.31],
            [11.10, 43.30],
        ]
    ],
}


def test_geojson_to_shape_accepts_polygon() -> None:
    geom = geojson_to_shape(_SQUARE)
    assert geom.geom_type == "Polygon"


def test_geojson_to_shape_rejects_non_polygon() -> None:
    with pytest.raises(ValueError, match="Polygon"):
        geojson_to_shape({"type": "Point", "coordinates": [0, 0]})


def test_polygon_area_ha_is_reasonable() -> None:
    """0.01deg x 0.01deg at lat 43 is roughly 90 ha; allow a wide tolerance."""
    area = polygon_area_ha(geojson_to_shape(_SQUARE))
    assert 70.0 < area < 110.0


def test_geometry_to_geojson_str_roundtrips() -> None:
    import json

    s = geometry_to_geojson_str(_SQUARE)
    assert json.loads(s) == _SQUARE
