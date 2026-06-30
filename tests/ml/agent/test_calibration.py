"""Tests for the honest confidence calibration module (US-081 AC11).

Pure, network-free: every case is a hand-built ``(18,)`` posterior or a
:class:`ClassificationResult`, so the raw-vs-restricted split is checked exactly
against known arithmetic. No Vertex AI / vLLM / DB is touched.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ml.agent.calibration import (
    ConfidenceReport,
    calibrate_from_posterior,
    calibrate_from_result,
)
from ml.agent.schemas import ClassificationResult
from ml.eval.class_remap import FRANCE_12, get_label_space

# france-12 dropped (out-of-vocabulary) ids: 9, 11, 12, 13, 16, 17.
# Kept (resolved) ids: 0,1,2,3,4,5,6,7,8,10,14,15.


def _posterior(mass: dict[int, float]) -> np.ndarray:
    """Build a normalized ``(18,)`` posterior from ``{class_id: mass}``."""
    arr = np.zeros(18, dtype=np.float64)
    for cid, m in mass.items():
        arr[cid] = m
    total = arr.sum()
    return arr / total if total > 0 else arr


def test_in_vocabulary_top_is_not_inflated() -> None:
    """A confident in-vocabulary argmax is reported as honest (not inflated)."""
    # Corn (id 2) dominates; all mass on resolved classes.
    proba = _posterior({2: 0.9, 6: 0.05, 14: 0.05})
    report = calibrate_from_posterior(proba, label_space=FRANCE_12)

    assert report.crop_class == "Corn"
    assert report.raw_top_class == "Corn"
    assert report.unresolved_candidate is None
    assert report.resolved_mass == pytest.approx(1.0)
    assert report.dropped_mass == pytest.approx(0.0)
    # Raw and restricted agree because no mass was dropped.
    assert report.restricted_confidence == pytest.approx(report.raw_confidence)
    assert report.is_inflated is False
    assert report.confidence_gap == pytest.approx(0.0)


def test_out_of_vocabulary_argmax_flags_inflated() -> None:
    """When the RAW argmax is out-of-vocabulary the headline is flagged inflated."""
    # Sorghum (id 17, dropped) is the RAW top with 0.55; the resolved mass (0.45)
    # is split so Corn (id 2) wins the restricted headline.
    proba = _posterior({17: 0.55, 2: 0.30, 6: 0.15})
    report = calibrate_from_posterior(proba, label_space="france-12")

    # Raw top is the out-of-vocabulary crop.
    assert report.raw_top_class == "Sorghum"
    assert report.unresolved_candidate == "Sorghum"
    # Restricted headline is a renormalization artifact (Corn at 0.30/0.45).
    assert report.crop_class == "Corn"
    assert report.resolved_mass == pytest.approx(0.45)
    assert report.dropped_mass == pytest.approx(0.55)
    assert report.restricted_confidence == pytest.approx(0.30 / 0.45)
    assert report.raw_confidence == pytest.approx(0.55)
    # Out-of-vocabulary raw argmax -> always inflated (hedge cue).
    assert report.is_inflated is True


def test_low_resolved_mass_flags_inflated_even_when_raw_in_vocab() -> None:
    """A low resolved mass inflates the headline even if the raw top is in vocab.

    The RAW argmax is Corn (in vocab) but most of the mass is spread across dropped
    classes, so the resolved mass is below the floor -> inflated.
    """
    # Corn (id 2) is the raw top at 0.30 but dropped ids hold 0.55 total.
    proba = _posterior({2: 0.30, 6: 0.15, 11: 0.20, 12: 0.20, 13: 0.15})
    report = calibrate_from_posterior(proba, label_space=FRANCE_12, resolved_mass_floor=0.5)

    assert report.raw_top_class == "Corn"
    assert report.unresolved_candidate is None  # raw top IS in vocabulary
    assert report.resolved_mass == pytest.approx(0.45)
    # Resolved mass 0.45 < floor 0.5 -> flagged inflated despite an in-vocab top.
    assert report.is_inflated is True
    # The restricted confidence (0.30/0.45) is higher than the raw (0.30): inflated.
    assert report.confidence_gap > 0.0


def test_zero_resolved_mass_reports_empty_headline() -> None:
    """All mass on dropped classes -> empty restricted headline, zero confidence."""
    proba = _posterior({17: 0.5, 12: 0.3, 13: 0.2})
    report = calibrate_from_posterior(proba, label_space=FRANCE_12)

    assert report.crop_class == ""
    assert report.restricted_confidence == pytest.approx(0.0)
    assert report.resolved_mass == pytest.approx(0.0)
    assert report.unresolved_candidate == "Sorghum"  # raw top id 17
    assert report.is_inflated is True


def test_floor_is_configurable() -> None:
    """A lower floor stops flagging a borderline resolved mass as inflated."""
    proba = _posterior({2: 0.30, 6: 0.15, 11: 0.30, 12: 0.25})  # resolved mass 0.45
    high_floor = calibrate_from_posterior(proba, label_space=FRANCE_12, resolved_mass_floor=0.5)
    low_floor = calibrate_from_posterior(proba, label_space=FRANCE_12, resolved_mass_floor=0.3)

    assert high_floor.is_inflated is True  # 0.45 < 0.5
    assert low_floor.is_inflated is False  # 0.45 >= 0.3 and raw top in vocab
    assert low_floor.resolved_mass_floor == pytest.approx(0.3)


def test_default_label_space_is_france12() -> None:
    """``None`` label-space resolves to the configured default (france-12)."""
    proba = _posterior({17: 0.6, 2: 0.4})
    report = calibrate_from_posterior(proba, label_space=None)
    # france-12 drops Sorghum, so the raw top is out-of-vocabulary.
    assert report.unresolved_candidate == "Sorghum"


def test_rejects_wrong_length_posterior() -> None:
    """A non-18-length posterior fails fast."""
    with pytest.raises(ValueError, match="semantic18"):
        calibrate_from_posterior(np.ones(9, dtype=np.float64))


def test_calibrate_from_result_in_vocabulary() -> None:
    """An in-vocabulary result reports resolved_mass 1.0 and not inflated."""
    result = ClassificationResult(
        crop_class="Corn",
        confidence=0.91,
        class_probabilities={"Corn": 0.91, "Sunflower": 0.05, "Soybeans": 0.04},
        out_of_vocabulary_classes=list(FRANCE_12.dropped_class_names.values()),
        unresolved_candidate=None,
    )
    report = calibrate_from_result(result, label_space="france-12")

    assert report.crop_class == "Corn"
    assert report.restricted_confidence == pytest.approx(0.91)
    assert report.resolved_mass == pytest.approx(1.0)
    assert report.is_inflated is False


def test_calibrate_from_result_out_of_vocabulary_is_inflated() -> None:
    """A result carrying an unresolved_candidate is flagged inflated (lossy path)."""
    result = ClassificationResult(
        crop_class="Corn",
        confidence=0.67,
        class_probabilities={"Corn": 0.67, "Sunflower": 0.20, "Soybeans": 0.13},
        out_of_vocabulary_classes=list(FRANCE_12.dropped_class_names.values()),
        unresolved_candidate="Sorghum",
    )
    report = calibrate_from_result(result, label_space="france-12")

    assert report.unresolved_candidate == "Sorghum"
    assert report.raw_top_class == "Sorghum"
    assert report.is_inflated is True
    # On the lossy path the raw mass split is unknown -> reported as NaN, not faked.
    assert math.isnan(report.resolved_mass)
    assert math.isnan(report.dropped_mass)


def test_calibrate_from_result_confidence_falls_back_to_max_prob() -> None:
    """A zero ``confidence`` falls back to the max class probability."""
    result = ClassificationResult(
        crop_class="Beet",
        confidence=0.0,
        class_probabilities={"Beet": 0.55, "Corn": 0.45},
        unresolved_candidate=None,
    )
    report = calibrate_from_result(result)
    assert report.restricted_confidence == pytest.approx(0.55)


def test_report_is_frozen_dataclass() -> None:
    """The report is immutable (auditable snapshot)."""
    report = calibrate_from_posterior(_posterior({2: 1.0}), label_space=FRANCE_12)
    assert isinstance(report, ConfidenceReport)
    with pytest.raises((AttributeError, TypeError)):
        report.crop_class = "Tampered"  # type: ignore[misc]


def test_posterior_and_default_space_consistency() -> None:
    """The default france-12 space exposes the expected six dropped crops."""
    space = get_label_space("france-12")
    assert set(space.dropped_class_names.values()) == {
        "Winter triticale",
        "Fruits, vegetables, flowers",
        "Potatoes",
        "Leguminous fodder",
        "Mixed cereal",
        "Sorghum",
    }
