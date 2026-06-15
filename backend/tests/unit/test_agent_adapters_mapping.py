"""Unit tests for the pure mapping helpers of the SQL agent adapters."""

from __future__ import annotations

from backend.app.models.feature import FeatureParcel
from backend.app.services.agent_adapters import _feature_to_record, _to_embedding


def test_to_embedding_none() -> None:
    assert _to_embedding(None) is None


def test_to_embedding_iterable() -> None:
    assert _to_embedding([1, 2, 3]) == [1.0, 2.0, 3.0]


def test_to_embedding_non_iterable() -> None:
    assert _to_embedding(42) is None


def test_feature_to_record_maps_all_fields() -> None:
    row = FeatureParcel(
        parcel_id=10,
        year=2024,
        alphaearth_embedding=[0.1, 0.2],
        ndvi_stats={"mean": 0.5},
        phenology={"sog": 90.0},
        ndvi_auc=12.3,
        peak_value=0.8,
    )
    record = _feature_to_record(row)
    assert record.parcel_id == 10
    assert record.year == 2024
    assert record.alphaearth_embedding == [0.1, 0.2]
    assert record.ndvi_stats == {"mean": 0.5}
    assert record.phenology == {"sog": 90.0}
    assert record.ndvi_auc == 12.3
    assert record.peak_value == 0.8


def test_feature_to_record_handles_empty_jsonb() -> None:
    row = FeatureParcel(parcel_id=1, year=2024)
    record = _feature_to_record(row)
    assert record.ndvi_stats == {}
    assert record.phenology == {}
    assert record.alphaearth_embedding is None
