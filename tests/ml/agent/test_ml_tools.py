"""ML / perceiver tool tests (US-045 AC-4, AC-7).

Covers ``classify_new_parcel``, ``explain_prediction`` and ``compare_models``.

- ``classify`` is driven by a real posterior row taken from the
  ``xgb-alphaearth`` OOF parquet (real probabilities, never random); the DB
  embedding fetch is mocked. The no-embedding path is also exercised.
- ``explain`` mocks ``session_scoped_conn`` (DB) and the phenology descriptor
  (no Gemini / litellm network), asserting the structured phenology text and the
  vigor mapping from the real scalar landmarks.
- ``compare`` reads the *real* per-parcel OOF parquets shipped in
  ``ml/eval/oof/`` as deterministic fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import ml.agent.tools.classify as classify_mod
import ml.agent.tools.compare as compare_mod
import ml.agent.tools.explain as explain_mod
from ml.agent.schemas import (
    ClassificationResult,
    ClassifyParcelInput,
    CompareModelsInput,
    ExplainPredictionInput,
    Explanation,
    ModelComparison,
)
from ml.utils.parcel_id import canonical_parcel_id
from ml.utils.parcel_reconcile import PROB_COLUMNS

from .conftest import SESSION_A

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OOF_DIR = _REPO_ROOT / "ml" / "eval" / "oof"
# A parcel present in the real OOF parquets (composite canonical id).
_REAL_PARCEL = "10003_1103071"
_POLYGON = {"type": "Polygon", "coordinates": [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]]}


def _real_xgb_posterior() -> np.ndarray:
    """Return a real 18-class posterior row from the xgb-alphaearth OOF parquet.

    Sourcing the vector from the actual model output keeps the classify test
    grounded in real probabilities (per the US-045 "no random" rule).
    """
    path = _OOF_DIR / f"oof_parcel_xgb-alphaearth_fold{5}.parquet"
    if not path.exists():
        pytest.skip(f"OOF fixture not found: {path} (run `dvc pull ml/eval/oof`).")
    frame = canonical_parcel_id(pl.read_parquet(path), col="canonical_parcel_id")
    row = frame.filter(pl.col("canonical_parcel_id") == _REAL_PARCEL)
    if row.height == 0:
        pytest.skip(f"parcel {_REAL_PARCEL} absent from xgb OOF fixture.")
    return row.select(PROB_COLUMNS).to_numpy().astype(np.float64)[0]


class _FakeClassifier:
    """``_XgbAlphaEarthClassifier`` stand-in returning a fixed real posterior."""

    def __init__(self, proba: np.ndarray, class_names: dict[int, str]) -> None:
        self._proba = proba
        self.class_names = class_names

    def predict_proba_18(self, embedding: np.ndarray) -> np.ndarray:
        return self._proba


# ---------------------------------------------------------------------------
# classify_new_parcel
# ---------------------------------------------------------------------------
async def test_classify_uses_real_posterior(monkeypatch, make_ctx) -> None:
    """``classify_new_parcel`` returns the argmax of a real model posterior."""
    proba = _real_xgb_posterior()
    expected_idx = int(np.argmax(proba))
    class_names = {i: f"class_{i}" for i in range(18)}

    async def _fake_fetch_embedding(ctx, year):
        return np.linspace(0.0, 1.0, 64, dtype=np.float64)

    monkeypatch.setattr(classify_mod, "_fetch_parcel_embedding", _fake_fetch_embedding)
    monkeypatch.setattr(
        classify_mod, "_load_classifier", lambda: _FakeClassifier(proba, class_names)
    )

    out = await classify_mod.run(
        ClassifyParcelInput(session_id=SESSION_A, aoi=_POLYGON, year=2019), make_ctx()
    )

    assert isinstance(out, ClassificationResult)
    assert out.crop_class == f"class_{expected_idx}"
    assert out.confidence == pytest.approx(float(proba[expected_idx]))
    # Full 18-class posterior surfaced and summing to ~1 (post-softmax).
    assert len(out.class_probabilities) == 18
    assert sum(out.class_probabilities.values()) == pytest.approx(1.0, abs=1e-6)


async def test_classify_needs_gee_when_no_embedding(monkeypatch, make_ctx) -> None:
    """With no persisted embedding the tool returns the controlled sentinel."""

    async def _no_embedding(ctx, year):
        return None

    monkeypatch.setattr(classify_mod, "_fetch_parcel_embedding", _no_embedding)
    # The classifier must NOT be loaded down this path.
    monkeypatch.setattr(
        classify_mod,
        "_load_classifier",
        lambda: pytest.fail("classifier should not load without an embedding"),
    )

    out = await classify_mod.run(
        ClassifyParcelInput(session_id=SESSION_A, aoi=_POLYGON, year=2019), make_ctx()
    )

    assert out.crop_class == "needs_gee_sampling"
    assert 0.0 < out.confidence < 1.0  # uniform prior, not a fabricated certainty


# ---------------------------------------------------------------------------
# explain_prediction
# ---------------------------------------------------------------------------
async def test_explain_with_mocked_descriptor(monkeypatch, make_ctx) -> None:
    """``explain_prediction`` builds phenology text + uses the mocked descriptor.

    The FFT harmonics drive the reconstructed curve; the descriptor is mocked so
    no Gemini/litellm call happens. We assert the structured phenology text,
    vigor, and that the mocked description is surfaced.
    """
    # ``_reconstruct_curve`` requires the full amp/phase set for k=0..3
    # (DEFAULT_FFT_HARMONICS=3); supply all eight so the real inverse-FFT yields
    # a finite non-zero NDVI curve and the descriptor path runs.
    phenology_json = {
        "NDVI_fft_amp_0": 0.55,
        "NDVI_fft_amp_1": 0.20,
        "NDVI_fft_amp_2": 0.05,
        "NDVI_fft_amp_3": 0.02,
        "NDVI_fft_phase_0": 0.0,
        "NDVI_fft_phase_1": 0.10,
        "NDVI_fft_phase_2": 0.30,
        "NDVI_fft_phase_3": 0.50,
    }
    record = {
        "crop_class": "wheat",
        "confidence": 0.88,
        "phenology": json.dumps(phenology_json),
        "sog_doy": 95,
        "peak_doy": 185,
        "peak_value": 0.82,
        "senescence_doy": 265,
        "ndvi_auc": 120.5,
        "maturity_duration_days": 170,
    }

    async def _fake_fetch(ctx, parcel_id):
        return dict(record)

    # ``explain`` imports ``session_scoped_conn`` lazily inside ``_fetch_parcel``;
    # patch the fetch helper itself (the DB boundary) instead.
    monkeypatch.setattr(explain_mod, "_fetch_parcel", _fake_fetch)

    captured = {}

    def _fake_descriptor(curve, *, parcel_id=None, crop_type_hint=None, temperature=0.0):
        captured["called"] = True
        captured["len"] = int(np.asarray(curve).size)
        captured["hint"] = crop_type_hint
        return "Descripcion fenologica generada por el descriptor (mock)."

    # Patch where the tool imports it (lazy import inside run()).
    monkeypatch.setattr(
        "ml.features.phenology_description.generate_phenology_description",
        _fake_descriptor,
    )

    out = await explain_mod.run(
        ExplainPredictionInput(session_id=SESSION_A, parcel_id=11), make_ctx()
    )

    assert isinstance(out, Explanation)
    assert out.parcel_id == 11
    assert out.crop_class == "wheat"
    assert out.confidence == pytest.approx(0.88)
    assert out.vigor == "high"  # peak_value 0.82 >= 0.7
    assert out.phenology_text  # non-empty structured block
    assert "dia 95" in out.phenology_text  # real SOG landmark surfaced
    assert captured.get("called") is True
    assert out.description == "Descripcion fenologica generada por el descriptor (mock)."


async def test_explain_falls_back_without_fft(monkeypatch, make_ctx) -> None:
    """Without FFT harmonics the description falls back to the structured text.

    No LLM call must happen in this path; the description equals the phenology
    text block built from the real scalar landmarks.
    """
    record = {
        "crop_class": "maize",
        "confidence": 0.5,
        "phenology": None,  # no harmonics
        "sog_doy": 100,
        "peak_doy": 190,
        "peak_value": 0.35,  # low vigor
        "senescence_doy": 260,
        "ndvi_auc": 80.0,
        "maturity_duration_days": None,
    }

    async def _fake_fetch(ctx, parcel_id):
        return dict(record)

    monkeypatch.setattr(explain_mod, "_fetch_parcel", _fake_fetch)
    monkeypatch.setattr(
        "ml.features.phenology_description.generate_phenology_description",
        lambda *a, **k: pytest.fail("descriptor must not be called without a curve"),
    )

    out = await explain_mod.run(
        ExplainPredictionInput(session_id=SESSION_A, parcel_id=12), make_ctx()
    )

    assert out.vigor == "low"  # 0.35 < 0.4
    assert out.description == out.phenology_text
    assert out.phenology_text.startswith("Fenologia:")


# ---------------------------------------------------------------------------
# compare_models (deferred) -- real OOF parquet fixtures
# ---------------------------------------------------------------------------
def _oof_present(*models: str) -> bool:
    """True if every model's per-parcel OOF parquet exists locally."""
    return all((_OOF_DIR / f"oof_parcel_{m}_fold5.parquet").exists() for m in models)


def test_predict_for_parcel_reads_real_oof() -> None:
    """``_predict_for_parcel`` returns the real argmax for a known parcel.

    utae and xgb-alphaearth both classify parcel ``10003_1103071`` as class 1 in
    the shipped OOF parquets -- a deterministic, real-data assertion.
    """
    if not _oof_present("utae", "xgb-alphaearth"):
        pytest.skip("OOF parquet fixtures missing (run `dvc pull ml/eval/oof`).")
    assert compare_mod._predict_for_parcel("utae", _REAL_PARCEL) == 1
    assert compare_mod._predict_for_parcel("xgb-alphaearth", _REAL_PARCEL) == 1


def test_predict_for_parcel_missing_model_returns_none() -> None:
    """A model with no OOF parquet is reported as ``None`` (never fabricated)."""
    assert compare_mod._predict_for_parcel("does-not-exist", _REAL_PARCEL) is None


async def test_compare_models_full_agreement(monkeypatch, make_ctx) -> None:
    """Two agreeing models over the real parcel yield ``agreement == 1.0``.

    The DB integer parcel id is bridged to the real composite canonical id so the
    tool reads the genuine OOF rows (no mock of the probabilities themselves).
    """
    if not _oof_present("utae", "xgb-alphaearth"):
        pytest.skip("OOF parquet fixtures missing (run `dvc pull ml/eval/oof`).")

    # The DB int parcel id has no composite OOF row, so redirect the per-model
    # lookup to the real composite parcel id. ``_predict_for_parcel`` still reads
    # the genuine OOF parquet and takes the real argmax -- no faked probabilities.
    real_predict = compare_mod._predict_for_parcel
    monkeypatch.setattr(
        compare_mod,
        "_predict_for_parcel",
        lambda model, canonical_id: real_predict(model, _REAL_PARCEL),
    )

    out = await compare_mod.run(
        CompareModelsInput(
            session_id=SESSION_A, parcel_id=10003, models=["utae", "xgb-alphaearth"]
        ),
        make_ctx(),
    )

    assert isinstance(out, ModelComparison)
    assert set(out.predictions) == {"utae", "xgb-alphaearth"}
    # Both predict the same class -> identical labels, perfect agreement.
    assert len(set(out.predictions.values())) == 1
    assert out.agreement == pytest.approx(1.0)


async def test_compare_models_empty_when_no_match(make_ctx) -> None:
    """A plain int parcel id matches no composite OOF row -> empty, no crash."""
    if not _oof_present("utae", "tsvit-pheno"):
        pytest.skip("OOF parquet fixtures missing (run `dvc pull ml/eval/oof`).")

    out = await compare_mod.run(
        CompareModelsInput(
            session_id=SESSION_A, parcel_id=999_999, models=["utae", "tsvit-pheno"]
        ),
        make_ctx(),
    )

    assert out.predictions == {}
    assert out.agreement == 0.0


async def test_compare_models_enqueues_when_defer_wired(monkeypatch, make_ctx) -> None:
    """When a defer hook is wired the job is enqueued and a typed result returned."""
    if not _oof_present("utae", "xgb-alphaearth"):
        pytest.skip("OOF parquet fixtures missing (run `dvc pull ml/eval/oof`).")

    enqueued = {}

    async def _defer(job_name, payload):
        enqueued["job"] = job_name
        enqueued["payload"] = payload
        return "handle-123"

    out = await compare_mod.run(
        CompareModelsInput(
            session_id=SESSION_A, parcel_id=999_999, models=["utae", "xgb-alphaearth"]
        ),
        make_ctx(defer=_defer),
    )

    assert enqueued["job"] == "compare_models"
    assert enqueued["payload"]["parcel_id"] == 999_999
    assert isinstance(out, ModelComparison)  # inline result still returned
