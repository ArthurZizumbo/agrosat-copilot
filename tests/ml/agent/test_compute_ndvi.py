"""Tests for the compute_ndvi FunctionTool (reads stored features only)."""

from __future__ import annotations

import pytest

from ml.agent.ports import FeatureRecord, ParcelRecord
from ml.agent.tools.compute_ndvi import ComputeNdviInput, compute_ndvi
from tests.ml.agent.fakes import FakeParcelReader


async def test_compute_ndvi_reads_stats() -> None:
    parcels = [ParcelRecord(id=10, aoi_id=1, crop_class="Meadow", area_ha=4.2, year=2023)]
    features = {
        10: FeatureRecord(
            parcel_id=10,
            year=2023,
            ndvi_stats={"mean": 0.72, "max": 0.91},
            phenology={"sos": 90.0},
            ndvi_auc=120.5,
            peak_value=0.93,
        )
    }
    reader = FakeParcelReader(parcels, features)

    out = await compute_ndvi(
        ComputeNdviInput(session_id="s1", aoi_id=1, year=2023), parcels=reader
    )

    assert len(out.findings) == 1
    f = out.findings[0]
    assert f.ndvi_mean == pytest.approx(0.72)
    assert f.metrics["ndvi_auc"] == pytest.approx(120.5)
    assert f.metrics["peak_value"] == pytest.approx(0.93)
    assert f.metrics["sos"] == pytest.approx(90.0)
    assert f.citation.source == "ndvi_stats:features"
    assert f.citation.parcel_id == 10


async def test_compute_ndvi_skips_parcels_without_features() -> None:
    parcels = [
        ParcelRecord(id=10, aoi_id=1, year=2023),
        ParcelRecord(id=11, aoi_id=1, year=2023),
    ]
    features = {10: FeatureRecord(parcel_id=10, year=2023, ndvi_stats={"mean": 0.5})}
    reader = FakeParcelReader(parcels, features)

    out = await compute_ndvi(
        ComputeNdviInput(session_id="s1", aoi_id=1, year=2023), parcels=reader
    )

    assert len(out.findings) == 1
    assert out.findings[0].parcel_id == 10
