"""Unit tests for the US-053 geospatial Pydantic models (no DB, no network).

Cover the request/response contracts in :mod:`backend.app.models.geo`:

- ``AoiCreate`` accepts a Polygon and rejects non-Polygon geometries (so a
  ``MultiPolygon`` is a clean ``422`` before PostGIS, not a mid-request DB error)
  and rejects an unknown ``session_id`` key (anti-spoofing via ``extra=forbid``).
- ``AoiFeature`` / ``AoiFeatureCollection`` build a valid GeoJSON Feature(s).
- ``TimeSeriesResponse`` enforces aligned ``dates``/``values`` and the
  ``NDVI``/``NDWI``/``NDMI`` index literal.
- ``StacSearchQuery`` validates the bbox shape/ranges and the limit bound.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from backend.app.models.geo import (
    AoiCreate,
    AoiFeature,
    AoiFeatureCollection,
    AoiProperties,
    StacSearchQuery,
    TimeSeriesResponse,
)

_POLYGON = {
    "type": "Polygon",
    "coordinates": [[[11.0, 43.0], [11.1, 43.0], [11.1, 43.1], [11.0, 43.1], [11.0, 43.0]]],
}


def test_aoi_create_accepts_polygon() -> None:
    """A valid Polygon geometry with an optional label is accepted."""
    body = AoiCreate(geometry=_POLYGON, label="field-1")
    assert body.geometry.type == "Polygon"
    assert body.label == "field-1"


def test_aoi_create_rejects_multipolygon() -> None:
    """A non-Polygon geometry is rejected (the column is GEOMETRY(POLYGON,4326))."""
    multi = {"type": "MultiPolygon", "coordinates": [_POLYGON["coordinates"]]}
    with pytest.raises(ValidationError, match="must be a Polygon"):
        AoiCreate(geometry=multi)


def test_aoi_create_rejects_unknown_session_id_key() -> None:
    """A spoofed ``session_id`` key is rejected by ``extra='forbid'``."""
    with pytest.raises(ValidationError):
        AoiCreate(geometry=_POLYGON, session_id="00000000-0000-0000-0000-000000000000")  # type: ignore[call-arg]


def test_aoi_feature_is_geojson_feature() -> None:
    """``AoiFeature`` serialises as a GeoJSON ``Feature`` with typed properties."""
    feature = AoiFeature(
        id=7,
        geometry=_POLYGON,
        properties=AoiProperties(label="x", area_ha=12.5, created_at=datetime(2026, 1, 1)),
    )
    dumped = feature.model_dump()
    assert dumped["type"] == "Feature"
    assert dumped["id"] == 7
    assert dumped["geometry"]["type"] == "Polygon"
    assert dumped["properties"]["area_ha"] == 12.5


def test_aoi_feature_collection_wraps_features() -> None:
    """``AoiFeatureCollection`` is a GeoJSON ``FeatureCollection``."""
    feature = AoiFeature(
        id=1,
        geometry=_POLYGON,
        properties=AoiProperties(created_at=datetime(2026, 1, 1)),
    )
    collection = AoiFeatureCollection(features=[feature])
    dumped = collection.model_dump()
    assert dumped["type"] == "FeatureCollection"
    assert len(dumped["features"]) == 1


def test_timeseries_response_requires_aligned_dates_values() -> None:
    """Mismatched ``dates``/``values`` lengths are rejected."""
    with pytest.raises(ValidationError, match="values length"):
        TimeSeriesResponse(aoi_id=1, index="NDVI", dates=[date(2019, 7, 1)], values=[])


def test_timeseries_response_accepts_empty_series() -> None:
    """An empty (honest) NDWI/NDMI series is valid."""
    resp = TimeSeriesResponse(aoi_id=1, index="NDWI", dates=[], values=[])
    assert resp.dates == []
    assert resp.values == []


def test_timeseries_response_rejects_unknown_index() -> None:
    """An index outside the NDVI/NDWI/NDMI literal is rejected."""
    with pytest.raises(ValidationError):
        TimeSeriesResponse(aoi_id=1, index="EVI", dates=[], values=[])  # type: ignore[arg-type]


def test_stac_query_defaults() -> None:
    """An empty STAC query keeps optional filters unset and the default limit."""
    query = StacSearchQuery()
    assert query.bbox is None
    assert query.datetime is None
    assert query.collections is None
    assert query.limit == 10


def test_stac_query_rejects_bad_bbox_length() -> None:
    """A bbox without exactly 4 elements is rejected."""
    with pytest.raises(ValidationError, match="4 elements"):
        StacSearchQuery(bbox=[1.0, 2.0, 3.0])


def test_stac_query_rejects_inverted_bbox() -> None:
    """A bbox whose min edge exceeds its max edge is rejected."""
    with pytest.raises(ValidationError, match="min edge"):
        StacSearchQuery(bbox=[10.0, 43.0, 9.0, 44.0])


def test_stac_query_rejects_out_of_range_limit() -> None:
    """A limit outside [1, 1000] is rejected."""
    with pytest.raises(ValidationError, match="limit"):
        StacSearchQuery(limit=0)
