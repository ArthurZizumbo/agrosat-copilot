"""Tests for the US-080 refinement eval (:mod:`ml.eval.farslip_refine_eval`).

Pure core driven by injected Voting-3 posteriors, ground truth and a fake FarSLIP
scorer -- no FarSLIP model, no chips, no MLflow.
"""

from __future__ import annotations

from ml.eval.farslip_refine_eval import f1_macro, run_refine_eval


def test_f1_macro_perfect_and_partial() -> None:
    assert f1_macro(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    assert f1_macro([], []) == 0.0
    partial = f1_macro(["a", "a", "b"], ["a", "b", "b"], labels=["a", "b"])
    assert 0.0 < partial < 1.0


def test_run_refine_eval_flips_uncertain_parcel_and_lifts_f1() -> None:
    # p1: Voting wrongly prefers 'b' by a thin margin (0.1) -> fires; FarSLIP favours
    # the correct 'a' and flips it. p2: confident & correct -> never touched.
    voting = {"p1": {"a": 0.45, "b": 0.55}, "p2": {"a": 0.9, "b": 0.1}}
    gt = {"p1": "a", "p2": "a"}

    def scorer(canonical_id: str):
        return {"a": 1.0, "b": 0.0} if canonical_id == "p1" else {"a": 0.0, "b": 1.0}

    report = run_refine_eval(voting, gt, scorer, alpha=0.6, margin_tau=0.15)

    assert report["n_parcels"] == 2
    assert report["n_fired"] == 1  # only p1 (low margin)
    assert report["n_changed"] == 1  # p1 flipped b -> a
    assert report["f1_after"] >= report["f1_before"]
    assert report["delta_f1"] > 0.0
    assert report["delta_f1_fired"] > 0.0


def test_run_refine_eval_without_scores_is_baseline() -> None:
    voting = {"p1": {"a": 0.4, "b": 0.6}}
    gt = {"p1": "a"}

    def scorer(canonical_id: str):
        return None  # FarSLIP unavailable for this parcel

    report = run_refine_eval(voting, gt, scorer)
    assert report["n_fired"] == 0
    assert report["delta_f1"] == 0.0
