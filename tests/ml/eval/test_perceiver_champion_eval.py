"""Tests for the parametrized perceiver champion-vs-baseline eval (US-081 AC1).

The eval is CPU-only and consumes the cached fold-5 OOF via the
:mod:`ml.agent.tools.classify` loaders. These tests replace those loaders +
the PASTIS-R ground-truth reconstructor with deterministic in-memory doubles
(cero red, cero DVC, cero raster), so they prove the PARAMETRIZATION contract:

- ``--label-space`` selects which semantic18 ids are scorable (``france-9`` vs
  ``france-12``): a parcel whose GT is one of the three NEW france-12 crops
  (Spring barley / Winter durum wheat / Orchard) is counted under ``france-12``
  but excluded under ``france-9``;
- ``--champion`` selects which model is the NEW perceiver path (``voting3`` /
  ``stacking5``) and the xgb baseline is read from the SAME cached member rows;
- the summary carries the right keys and the v2 report bundles both spaces.

The numeric uplift over the REAL OOF is produced by running the module against
the pulled DVC data (``reports/agent_bench/perceiver_champion_eval_v2.json``);
here the lens is purely the parametrization, kept synthetic on purpose.
"""

from __future__ import annotations

import numpy as np
import pytest

import ml.eval.perceiver_champion_eval as pce
from ml.eval.class_remap import get_label_space

# semantic18 ids: 0 Meadow, 2 Corn (france-9), 5 Spring barley, 15 Orchard (new in
# france-12). A confident one-hot posterior on the GT id makes the argmax exact.
_F9_ID = 2  # Corn (resolved by both france-9 and france-12)
_NEW_ID = 5  # Spring barley (only resolved by france-12)


class _FakeVoting:
    """Stand-in for ``_VotingThree`` exposing ``member_probs_by_id``.

    Each parcel carries a ``(3, 18)`` member tensor; the champion posterior is a
    one-hot on the GT id (perfect), while the xgb member row (index 2) is one-hot
    on a WRONG id, so the champion fixes every parcel vs the baseline.
    """

    def __init__(self, ids_to_gt: dict[str, int]) -> None:
        self.member_probs_by_id: dict[str, np.ndarray] = {}
        for cid, gt in ids_to_gt.items():
            rows = np.zeros((3, 18), dtype=np.float64)
            rows[0, gt] = 1.0  # tsvit-pheno member: correct
            rows[1, gt] = 1.0  # utae member: correct
            wrong = (gt + 1) % 18
            rows[2, wrong] = 1.0  # xgb-alphaearth member (baseline): wrong
            self.member_probs_by_id[cid] = rows

    def posterior_for_parcel(self, cid: str) -> np.ndarray | None:
        rows = self.member_probs_by_id.get(cid)
        if rows is None:
            return None
        return rows[0]  # the champion serves the (correct) tsvit row


@pytest.fixture
def patched_classify(monkeypatch: pytest.MonkeyPatch):
    """Patch the classify loaders + GT reconstructor with synthetic doubles."""
    import polars as pl

    from ml.agent.tools import classify as cls

    ids_to_gt = {
        "patchA_1": _F9_ID,  # Corn (in both spaces)
        "patchA_2": _F9_ID,
        "patchB_1": _NEW_ID,  # Spring barley (only france-12)
        "patchB_2": _NEW_ID,
    }
    fake = _FakeVoting(ids_to_gt)
    monkeypatch.setattr(cls, "_load_voting_three", lambda: fake)

    def _fake_gt(canonical_ids: list[str]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "canonical_parcel_id": list(ids_to_gt.keys()),
                "label": list(ids_to_gt.values()),
            }
        )

    monkeypatch.setattr(cls, "_build_parcel_ground_truth", _fake_gt)
    return ids_to_gt


def test_voting3_france9_excludes_new_crops(patched_classify) -> None:
    """Under france-9 the two Spring-barley parcels are NOT scorable."""
    summary = pce.evaluate(champion="voting3", label_space_name="france-9")
    assert summary["label_space"] == "france-9"
    assert summary["champion"] == "voting3"
    # Only the two Corn parcels fall inside france-9.
    assert summary["n_parcels"] == 2
    # Champion is one-hot-correct, baseline one-hot-wrong -> perfect uplift.
    assert summary["champion_voting3"]["accuracy"] == 1.0
    assert summary["baseline_xgb"]["accuracy"] == 0.0
    assert summary["net_fixed"] == 2


def test_voting3_france12_includes_new_crops(patched_classify) -> None:
    """Under france-12 the Spring-barley parcels ARE scored (expanded scope)."""
    summary = pce.evaluate(champion="voting3", label_space_name="france-12")
    assert summary["label_space"] == "france-12"
    # All four parcels (Corn + Spring barley) are in the twelve-class space.
    assert summary["n_parcels"] == 4
    assert summary["champion_voting3"]["accuracy"] == 1.0
    assert summary["net_fixed"] == 4


def test_default_label_space_resolves(patched_classify) -> None:
    """A None label-space resolves to DEFAULT_LABEL_SPACE (france-12)."""
    summary = pce.evaluate(champion="voting3", label_space_name=None)
    assert summary["label_space"] == get_label_space(None).name


def test_unknown_champion_raises() -> None:
    """An unsupported champion id fails fast."""
    with pytest.raises(ValueError, match="unknown champion"):
        pce.evaluate(champion="nope", label_space_name="france-9")


def test_v2_report_bundles_both_spaces(monkeypatch: pytest.MonkeyPatch) -> None:
    """The v2 report bundles previous/v2 champions over france-9 + france-12."""

    def _fake_evaluate(*, champion: str, label_space_name: str | None, out_path=None):
        key = f"champion_{champion}"
        return {
            "label_space": label_space_name,
            "champion": champion,
            "n_parcels": 10,
            "n_classes": 12 if label_space_name == "france-12" else 9,
            "baseline_xgb": {"accuracy": 0.80, "macro_f1": 0.70},
            key: {"accuracy": 0.95, "macro_f1": 0.90},
            "delta_accuracy": 0.15,
            "delta_macro_f1": 0.20,
            "agreement": 0.88,
            "champion_fixes": 5,
            "champion_breaks": 1,
            "net_fixed": 4,
        }

    monkeypatch.setattr(pce, "evaluate", _fake_evaluate)
    report = pce.evaluate_v2_report()
    assert set(report) >= {
        "previous_champion_france9",
        "v2_champion_france9",
        "v2_champion_france12",
        "deltas",
    }
    assert report["v2_champion_france12"]["label_space"] == "france-12"
    assert "v2_vs_v1_france9_macro_f1" in report["deltas"]
