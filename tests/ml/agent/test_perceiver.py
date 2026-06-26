"""Perceiver-layer tests (US-046 AC-2, "Be My Eyes" text-only contract).

The perceiver is the eyes of the agent: it composes the real phenology descriptor
(``explain_prediction``) and the XGBoost-AlphaEarth posterior (``classify``) into a
structured TEXT observation. These tests assert it:

- builds a :class:`PerceiverObservation` with a non-empty ``phenology_text`` from a
  stored parcel (``observe``), reusing the explanation fields verbatim;
- renders ``to_prompt_block`` as Spanish text carrying the crop and phenology, and
  NEVER leaks logits/tensors into that block;
- never runs an LLM (no Gemini/litellm client is touched -- the descriptor is
  mocked, exactly like the US-045 ``explain`` test);
- degrades to a degenerate ``{crop_class: 1.0}`` posterior when the session has no
  persisted AlphaEarth embedding (no invented alternatives);
- observes a fresh AOI through ``classify`` and surfaces the controlled
  ``needs_gee_sampling`` result verbatim (no hallucinated crop).

Every external boundary is mocked: the DB (via ``explain._fetch_parcel`` and
``classify._fetch_parcel_embedding``), the classifier (via ``classify._load_classifier``)
and the phenology descriptor. Reuses :func:`make_ctx` from the US-045 conftest.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

import ml.agent.perceiver as perceiver_mod
import ml.agent.tools.classify as classify_mod
import ml.agent.tools.explain as explain_mod
from ml.agent.perceiver import PerceiverLayer, PerceiverObservation

_POLYGON = {
    "type": "Polygon",
    "coordinates": [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]],
}


class _FakeClassifier:
    """``_XgbAlphaEarthClassifier`` stand-in returning a fixed posterior."""

    def __init__(self, proba: np.ndarray, class_names: dict[int, str]) -> None:
        self._proba = proba
        self.class_names = class_names

    def predict_proba_18(self, embedding: np.ndarray) -> np.ndarray:
        return self._proba


def _stored_parcel_record() -> dict:
    """A realistic ``features_parcels`` row with high-vigor phenology landmarks."""
    return {
        "crop_class": "wheat",
        "confidence": 0.88,
        "phenology": None,  # no FFT harmonics -> descriptor not invoked
        "sog_doy": 95,
        "peak_doy": 185,
        "peak_value": 0.82,  # >= 0.7 -> high vigor
        "senescence_doy": 265,
        "ndvi_auc": 120.5,
        "maturity_duration_days": 170,
    }


# ---------------------------------------------------------------------------
# observe(parcel_id): stored parcel
# ---------------------------------------------------------------------------
async def test_observe_builds_text_observation(monkeypatch, make_ctx) -> None:
    """``observe`` composes explanation + posterior into a TEXT observation.

    The phenology text is non-empty and grounded in the real scalar landmarks; the
    crop class and confidence come straight from the explanation; the posterior is
    the classifier output. No LLM runs (no FFT curve -> descriptor never called).
    """
    record = _stored_parcel_record()

    async def _fake_fetch_parcel(ctx, parcel_id):
        return dict(record)

    # 18-class posterior with a clear argmax matching the stored crop.
    proba = np.full(18, 0.01, dtype=np.float64)
    proba[3] = 0.83
    proba /= proba.sum()
    class_names = {i: f"class_{i}" for i in range(18)}
    class_names[3] = "wheat"

    async def _fake_fetch_embedding(ctx, year, aoi=None):
        return np.linspace(0.0, 1.0, 64, dtype=np.float64)

    # Patch the DB boundary inside ``explain`` and the classifier+embedding inside
    # ``classify``. The descriptor must never be reached (no FFT harmonics here);
    # guard it so a regression that calls Gemini/litellm fails loudly.
    monkeypatch.setattr(explain_mod, "_fetch_parcel", _fake_fetch_parcel)
    monkeypatch.setattr(classify_mod, "_fetch_parcel_embedding", _fake_fetch_embedding)
    monkeypatch.setattr(
        classify_mod, "_load_classifier", lambda: _FakeClassifier(proba, class_names)
    )
    # The champion (Stacking-5) is unavailable in this unit (no OOF artifacts):
    # force the clean degradation to the ``xgb-alphaearth`` member so the test
    # exercises the tabular posterior path. ``_load_stacking_five`` raising
    # FileNotFoundError is exactly the "DVC not pulled" degradation contract.
    def _no_stacking() -> object:
        raise FileNotFoundError("Stacking-5 OOF not available in this unit test")

    monkeypatch.setattr(classify_mod, "_load_stacking_five", _no_stacking)
    monkeypatch.setattr(
        "ml.features.phenology_description.generate_phenology_description",
        lambda *a, **k: pytest.fail("perceiver must not call the LLM descriptor here"),
    )

    obs = await PerceiverLayer(make_ctx()).observe(parcel_id=11)

    assert isinstance(obs, PerceiverObservation)
    assert obs.parcel_id == 11
    assert obs.crop_class == "wheat"
    assert obs.confidence == pytest.approx(0.88)
    assert obs.vigor == "high"  # peak_value 0.82
    # Non-empty structured phenology text grounded in the real SOG landmark.
    assert obs.phenology_text
    assert "dia 95" in obs.phenology_text
    # Posterior restricted to the nine well-resolved france-9 classes (the
    # agent+app directive), renormalized to sum to ~1. The argmax mass (id 3,
    # "Winter barley") is a france-9 class so it survives the restriction.
    assert len(obs.class_probabilities) == 9
    assert sum(obs.class_probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert set(obs.class_probabilities) <= {
        "Meadow",
        "Soft winter wheat",
        "Corn",
        "Winter barley",
        "Winter rapeseed",
        "Sunflower",
        "Grapevine",
        "Beet",
        "Soybeans",
    }


async def test_to_prompt_block_is_text_without_logits(monkeypatch, make_ctx) -> None:
    """``to_prompt_block`` renders Spanish text with crop + phenology, no tensors.

    The block must be a non-empty ``str`` mentioning the crop class and the
    phenology text, and must NOT leak any tensor/logit artefact.
    """
    record = _stored_parcel_record()

    async def _fake_fetch_parcel(ctx, parcel_id):
        return dict(record)

    async def _no_embedding(ctx, year, aoi=None):
        return None

    monkeypatch.setattr(explain_mod, "_fetch_parcel", _fake_fetch_parcel)
    # No embedding -> degenerate posterior path (also proves no classifier needed).
    monkeypatch.setattr(classify_mod, "_fetch_parcel_embedding", _no_embedding)
    monkeypatch.setattr(
        classify_mod,
        "_load_classifier",
        lambda: pytest.fail("classifier must not load without an embedding"),
    )

    obs = await PerceiverLayer(make_ctx()).observe(parcel_id=7)
    block = obs.to_prompt_block()

    assert isinstance(block, str)
    assert block.strip()  # non-empty
    assert "wheat" in block  # crop class present
    assert "Fenologia" in block or "fenolog" in block.lower()  # phenology present
    assert "Vigor" in block
    # Be My Eyes contract: no tensor/array leakage in the reasoner-facing text.
    # (The block's own header reads "sin logits", so the bare word "logit" is
    # legitimate; we forbid actual tensor/array repr artefacts instead.)
    for forbidden in ("tensor(", "array(", "dtype", "ndarray", "predict_proba"):
        assert forbidden not in block


async def test_observe_degenerate_posterior_without_embedding(
    monkeypatch, make_ctx
) -> None:
    """No persisted embedding -> posterior collapses to ``{crop_class: 1.0}``.

    The perceiver must not invent alternative classes; the degenerate posterior
    places all mass on the stored crop class.
    """
    record = _stored_parcel_record()

    async def _fake_fetch_parcel(ctx, parcel_id):
        return dict(record)

    async def _no_embedding(ctx, year, aoi=None):
        return None

    monkeypatch.setattr(explain_mod, "_fetch_parcel", _fake_fetch_parcel)
    monkeypatch.setattr(classify_mod, "_fetch_parcel_embedding", _no_embedding)
    monkeypatch.setattr(
        classify_mod,
        "_load_classifier",
        lambda: pytest.fail("classifier must not load without an embedding"),
    )

    obs = await PerceiverLayer(make_ctx()).observe(parcel_id=5)

    assert obs.class_probabilities == {"wheat": 1.0}


# ---------------------------------------------------------------------------
# observe_aoi(aoi, year): fresh AOI
# ---------------------------------------------------------------------------
async def test_observe_aoi_propagates_needs_gee_sampling(monkeypatch, make_ctx) -> None:
    """A fresh AOI with no embedding surfaces ``needs_gee_sampling`` verbatim.

    ``classify.run`` returns the controlled sentinel; the perceiver must not
    fabricate a crop. The observation is AOI-level (``parcel_id == -1``).
    """

    async def _no_embedding(ctx, year, aoi=None):
        return None

    monkeypatch.setattr(classify_mod, "_fetch_parcel_embedding", _no_embedding)
    monkeypatch.setattr(
        classify_mod,
        "_load_classifier",
        lambda: pytest.fail("classifier must not load without an embedding"),
    )

    obs = await PerceiverLayer(make_ctx()).observe_aoi(
        perceiver_mod.GeoJSONGeometry(**_POLYGON), year=2019
    )

    assert obs.parcel_id == -1
    assert obs.crop_class == "needs_gee_sampling"
    assert obs.class_probabilities == {"needs_gee_sampling": 1.0}
    assert obs.to_prompt_block().strip()  # still renders a coherent block


async def test_observe_aoi_uses_classifier_posterior(monkeypatch, make_ctx) -> None:
    """An AOI with a persisted embedding yields the real classifier posterior.

    The AOI observation derives crop/confidence/vigor from ``classify.run`` (the
    XGBoost-AlphaEarth member); no stored phenology row exists, so the phenology
    text is the documented classifier-derived note.
    """
    proba = np.full(18, 0.02, dtype=np.float64)
    proba[7] = 0.66
    proba /= proba.sum()
    class_names = {i: f"class_{i}" for i in range(18)}
    class_names[7] = "maize"

    async def _fake_fetch_embedding(ctx, year, aoi=None):
        return np.linspace(0.0, 1.0, 64, dtype=np.float64)

    async def _no_canonical(ctx, aoi):
        return None  # AOI resolves to no fold-5 parcel -> degrade to xgb cleanly

    monkeypatch.setattr(classify_mod, "_fetch_parcel_embedding", _fake_fetch_embedding)
    monkeypatch.setattr(classify_mod, "_resolve_canonical_parcel_id", _no_canonical)
    monkeypatch.setattr(
        classify_mod, "_load_classifier", lambda: _FakeClassifier(proba, class_names)
    )

    obs = await PerceiverLayer(make_ctx()).observe_aoi(
        perceiver_mod.GeoJSONGeometry(**_POLYGON), year=2019
    )

    # ``observe_aoi`` now serves the champion restricted to france-9 (use_stacking
    # ON); here the parcel is not in the OOF universe so it degrades to the
    # ``xgb-alphaearth`` member, then restricts to the nine resolved classes.
    assert obs.parcel_id == -1
    assert obs.crop_class == "Grapevine"  # id 7 is france-9; argmax after restrict
    # Confidence is the renormalized mass over the nine france-9 classes (>= the
    # raw 0.66 since the dropped classes' mass is removed).
    assert obs.confidence >= float(proba[7])
    assert len(obs.class_probabilities) == 9
    assert set(obs.class_probabilities) <= {
        "Meadow",
        "Soft winter wheat",
        "Corn",
        "Winter barley",
        "Winter rapeseed",
        "Sunflower",
        "Grapevine",
        "Beet",
        "Soybeans",
    }


def test_perceiver_observation_is_strict_extra_forbid() -> None:
    """The observation model rejects extra keys and non-coerced confidence (strict)."""
    with pytest.raises(ValidationError):
        PerceiverObservation(
            parcel_id=1,
            crop_class="wheat",
            confidence="0.8",  # strict mode: must be a real float, not a string
            phenology_text="x",
            vigor="high",
            class_probabilities={"wheat": 1.0},
            description="d",
        )
    with pytest.raises(ValidationError):
        PerceiverObservation(
            parcel_id=1,
            crop_class="wheat",
            confidence=0.8,
            phenology_text="x",
            vigor="high",
            class_probabilities={"wheat": 1.0},
            description="d",
            extra_key="boom",  # extra="forbid"
        )
