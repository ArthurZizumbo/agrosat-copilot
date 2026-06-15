"""Schema contract tests for the nine tool input/output models (US-045).

Each ``*Input`` model must accept well-formed LLM arguments and reject malformed
ones with a :class:`pydantic.ValidationError` (the strict, ``extra="forbid"``
config). These tests also pin the shared value objects (``GeoJSONGeometry``,
``BBox``, ``ParcelRef``, ``AoiRef``) and guard the ``GeoJSONGeometry``
construction path that geometry-bearing tools depend on.
"""

from __future__ import annotations

from datetime import date
from uuid import UUID

import pytest
from pydantic import ValidationError

from ml.agent.schemas import (
    AddAoiInput,
    AoiRef,
    AoiStatsInput,
    BBox,
    ClassifyParcelInput,
    CompareModelsInput,
    ExplainPredictionInput,
    GeoJSONGeometry,
    GetTilesInput,
    ListParcelsInput,
    ParcelRef,
    ParcelTimeseriesInput,
    SearchStacInput,
)

_SESSION = UUID("11111111-1111-1111-1111-111111111111")
_POLYGON = {"type": "Polygon", "coordinates": [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]]}


# ---------------------------------------------------------------------------
# Shared value objects
# ---------------------------------------------------------------------------
def test_geojson_geometry_constructs() -> None:
    """A valid polygon geometry constructs and keeps its fields.

    Regression guard: the private ``_ALLOWED_TYPES`` set must not break model
    construction (a Pydantic v2 private-attribute pitfall).
    """
    geom = GeoJSONGeometry(**_POLYGON)
    assert geom.type == "Polygon"
    assert geom.coordinates == _POLYGON["coordinates"]


def test_geojson_geometry_rejects_unknown_type() -> None:
    """An unsupported OGC geometry type is rejected."""
    with pytest.raises(ValidationError):
        GeoJSONGeometry(type="Tesseract", coordinates=[[0, 0]])


def test_geojson_geometry_rejects_empty_coordinates() -> None:
    """An empty coordinate array is rejected (a geometry needs vertices)."""
    with pytest.raises(ValidationError):
        GeoJSONGeometry(type="Polygon", coordinates=[])


def test_bbox_valid_and_out_of_range() -> None:
    """A bbox accepts in-range degrees and rejects out-of-range ones."""
    bbox = BBox(minx=-3.7, miny=40.0, maxx=-3.6, maxy=40.1)
    assert bbox.maxx == pytest.approx(-3.6)
    with pytest.raises(ValidationError):
        BBox(minx=-200.0, miny=0.0, maxx=10.0, maxy=10.0)
    with pytest.raises(ValidationError):
        BBox(minx=0.0, miny=-91.0, maxx=10.0, maxy=10.0)


def test_parcel_ref_optionals_default_none() -> None:
    """``ParcelRef`` allows missing crop/confidence (unlabelled parcels)."""
    ref = ParcelRef(parcel_id=42)
    assert ref.parcel_id == 42
    assert ref.crop_class is None
    assert ref.confidence is None


def test_aoi_ref_roundtrips() -> None:
    """``AoiRef`` (also the ``add_aoi`` output) carries id/label/area."""
    ref = AoiRef(aoi_id=7, label="Demo", area_ha=12.5)
    assert (ref.aoi_id, ref.label, ref.area_ha) == (7, "Demo", 12.5)


# ---------------------------------------------------------------------------
# Per-tool input validation: valid acceptance + invalid rejection
# ---------------------------------------------------------------------------
def test_list_parcels_input_valid_and_strict() -> None:
    """``ListParcelsInput`` accepts an optional AOI and forbids extras."""
    assert ListParcelsInput(session_id=_SESSION).aoi is None
    assert ListParcelsInput(session_id=_SESSION, aoi=_POLYGON).aoi is not None
    with pytest.raises(ValidationError):
        ListParcelsInput(session_id="not-a-uuid")
    with pytest.raises(ValidationError):
        ListParcelsInput(session_id=_SESSION, unexpected="x")


def test_parcel_timeseries_input_window_and_index() -> None:
    """``ParcelTimeseriesInput`` validates the index literal and date window."""
    ok = ParcelTimeseriesInput(
        session_id=_SESSION,
        parcel_id=1,
        start=date(2019, 1, 1),
        end=date(2019, 12, 31),
        index="ndvi",
    )
    assert ok.index == "ndvi"
    # Index outside the literal set is rejected.
    with pytest.raises(ValidationError):
        ParcelTimeseriesInput(
            session_id=_SESSION,
            parcel_id=1,
            start=date(2019, 1, 1),
            end=date(2019, 12, 31),
            index="savi",
        )
    # end before start is rejected by the custom validator.
    with pytest.raises(ValidationError):
        ParcelTimeseriesInput(
            session_id=_SESSION,
            parcel_id=1,
            start=date(2019, 12, 31),
            end=date(2019, 1, 1),
            index="evi",
        )


def test_aoi_stats_input_year_range() -> None:
    """``AoiStatsInput`` requires an AOI and a year within AlphaEarth coverage."""
    assert AoiStatsInput(session_id=_SESSION, aoi=_POLYGON, year=2019).year == 2019
    with pytest.raises(ValidationError):
        AoiStatsInput(session_id=_SESSION, aoi=_POLYGON, year=1990)
    with pytest.raises(ValidationError):
        AoiStatsInput(session_id=_SESSION, year=2019)  # missing AOI


def test_search_stac_input_cloud_cover_default_and_bounds() -> None:
    """``SearchStacInput`` defaults cloud cover to 20 and bounds it to [0, 100]."""
    bbox = {"minx": -3.7, "miny": 40.0, "maxx": -3.6, "maxy": 40.1}
    ok = SearchStacInput(bbox=bbox, datetime_range="2019-01-01/2019-12-31")
    assert ok.cloud_cover_max == pytest.approx(20.0)
    with pytest.raises(ValidationError):
        SearchStacInput(bbox=bbox, datetime_range="2019-01-01/2019-12-31", cloud_cover_max=150.0)
    with pytest.raises(ValidationError):
        SearchStacInput(bbox=bbox, datetime_range="   ")  # empty interval


def test_get_tiles_input_index_literal() -> None:
    """``GetTilesInput`` requires a non-empty scene id and a known product."""
    assert GetTilesInput(scene_id="S2_X", index="rgb").index == "rgb"
    with pytest.raises(ValidationError):
        GetTilesInput(scene_id="", index="ndvi")
    with pytest.raises(ValidationError):
        GetTilesInput(scene_id="S2_X", index="thermal")


def test_classify_parcel_input_defaults_year() -> None:
    """``ClassifyParcelInput`` defaults the campaign year to 2019."""
    inp = ClassifyParcelInput(session_id=_SESSION, aoi=_POLYGON)
    assert inp.year == 2019
    with pytest.raises(ValidationError):
        ClassifyParcelInput(session_id=_SESSION, aoi=_POLYGON, year=1800)


def test_add_aoi_input_requires_name() -> None:
    """``AddAoiInput`` requires a non-blank name."""
    assert AddAoiInput(session_id=_SESSION, aoi=_POLYGON, name="Field 1").name == "Field 1"
    with pytest.raises(ValidationError):
        AddAoiInput(session_id=_SESSION, aoi=_POLYGON, name="   ")


def test_compare_models_input_requires_two_unique() -> None:
    """``CompareModelsInput`` needs at least two distinct model names."""
    ok = CompareModelsInput(session_id=_SESSION, parcel_id=1, models=["utae", "tsvit-pheno"])
    assert len(ok.models) == 2
    with pytest.raises(ValidationError):
        CompareModelsInput(session_id=_SESSION, parcel_id=1, models=["utae"])
    with pytest.raises(ValidationError):
        CompareModelsInput(session_id=_SESSION, parcel_id=1, models=["utae", "utae"])


def test_explain_prediction_input_minimal() -> None:
    """``ExplainPredictionInput`` carries only session + parcel id."""
    inp = ExplainPredictionInput(session_id=_SESSION, parcel_id=99)
    assert inp.parcel_id == 99
    with pytest.raises(ValidationError):
        ExplainPredictionInput(session_id=_SESSION)  # missing parcel id
