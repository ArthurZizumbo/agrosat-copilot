"""Tests for the classify_parcel FunctionTool (model path + honest fallback)."""

from __future__ import annotations

import pytest

from ml.agent.ports import FeatureRecord, ParcelRecord
from ml.agent.tools.classify_parcel import (
    ClassifyParcelInput,
    classify_parcel,
)
from tests.ml.agent.fakes import FakeBundle, FakeParcelReader, FakeXGBModel


@pytest.fixture(autouse=True)
def _reset_model_cache() -> None:
    """Clear the module-level model caches between tests (no shared state)."""
    from ml.agent.tools import classify_parcel as mod

    mod._MODEL_CACHE.clear()
    mod._REGISTRY_TRIED.clear()


@pytest.fixture
def parcels() -> list[ParcelRecord]:
    return [
        ParcelRecord(id=10, aoi_id=1, crop_class="Meadow", confidence=0.5, area_ha=4.2, year=2023),
        ParcelRecord(id=11, aoi_id=1, crop_class="Wheat", confidence=0.6, area_ha=2.1, year=2023),
    ]


@pytest.fixture
def features() -> dict[int, FeatureRecord]:
    emb = [0.1] * 64
    return {
        10: FeatureRecord(parcel_id=10, year=2023, alphaearth_embedding=emb),
        11: FeatureRecord(parcel_id=11, year=2023, alphaearth_embedding=emb),
    }


async def test_classify_registry_unavailable_falls_back(parcels, features) -> None:
    """A real (unreachable) registry URI degrades to the stored fallback."""
    reader = FakeParcelReader(parcels, features)

    out = await classify_parcel(
        ClassifyParcelInput(session_id="s1", aoi_id=1, year=2023),
        parcels=reader,
        model_uri="models:/does-not-exist@none",
    )

    assert out.used_model is False
    assert len(out.findings) == 2
    assert all(f.citation.source == "stored:crop_class" for f in out.findings)


async def test_classify_model_path_via_cache(monkeypatch, parcels, features) -> None:
    """Force the model path by seeding the module cache, assert citation source."""
    from ml.agent.tools import classify_parcel as mod

    model = FakeXGBModel(classes=(0, 1), forced_index=1)
    bundle = FakeBundle(model=model, label_classes=(0, 1), n_features=64)
    monkeypatch.setattr(mod, "_load_classifier", lambda uri: bundle)

    reader = FakeParcelReader(parcels, features)
    out = await classify_parcel(
        ClassifyParcelInput(session_id="s1", aoi_id=1, year=2023),
        parcels=reader,
    )

    assert out.used_model is True
    assert len(out.findings) == 2
    for f in out.findings:
        assert f.citation.source == "XGBoost+AlphaEarth"
        assert f.confidence == pytest.approx(0.9)
        # Class id 1 maps to a readable PASTIS-R name (not "class_1").
        assert f.crop_class is not None


async def test_classify_fallback_stored(monkeypatch, parcels) -> None:
    """No model -> uses stored crop_class and marks source stored:crop_class."""
    from ml.agent.tools import classify_parcel as mod

    monkeypatch.setattr(mod, "_load_classifier", lambda uri: None)
    reader = FakeParcelReader(parcels, features={})

    out = await classify_parcel(
        ClassifyParcelInput(session_id="s1", aoi_id=1, year=2023),
        parcels=reader,
    )

    assert out.used_model is False
    assert {f.crop_class for f in out.findings} == {"Meadow", "Wheat"}
    for f in out.findings:
        assert f.citation.source == "stored:crop_class"
        assert f.citation.parcel_id == f.parcel_id


async def test_classify_feature_mismatch_falls_back(monkeypatch, parcels) -> None:
    """A wrong-length embedding does not crash; finding keeps stored class."""
    from ml.agent.tools import classify_parcel as mod

    model = FakeXGBModel(classes=(0, 1), forced_index=1)
    bundle = FakeBundle(model=model, label_classes=(0, 1), n_features=64)
    monkeypatch.setattr(mod, "_load_classifier", lambda uri: bundle)

    bad_features = {
        10: FeatureRecord(parcel_id=10, year=2023, alphaearth_embedding=[0.1] * 8),
    }
    reader = FakeParcelReader([parcels[0]], bad_features)
    out = await classify_parcel(
        ClassifyParcelInput(session_id="s1", aoi_id=1, year=2023),
        parcels=reader,
    )

    assert out.used_model is False
    assert out.findings[0].citation.source == "stored:crop_class"
