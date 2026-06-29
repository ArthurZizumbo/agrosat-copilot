"""Tests for the conditional Voting-3 refinement (US-080, :mod:`ml.agent.refine`).

Pure-logic tests (no torch / FarSLIP / DB): the refiner only fires on uncertain or
open-set parcels and never degrades the easy case, and the convex blend stays a
valid distribution.
"""

from __future__ import annotations

import pytest

from ml.agent.refine import (
    apply_refinement,
    refine_posterior,
    should_refine,
    top1_top2_margin,
)


def test_top1_top2_margin() -> None:
    assert top1_top2_margin({"a": 0.6, "b": 0.3, "c": 0.1}) == pytest.approx(0.3)
    assert top1_top2_margin({"a": 1.0}) == 1.0  # single class -> max margin
    assert top1_top2_margin({}) == 0.0


def test_should_refine_open_set_always_fires() -> None:
    fire, reason = should_refine({"a": 0.99, "b": 0.01}, open_set=True)
    assert fire is True and reason == "open_set"


def test_should_refine_on_member_disagreement() -> None:
    fire, reason = should_refine(
        {"a": 0.9, "b": 0.1},  # confident posterior...
        member_predictions={"m1": "a", "m2": "b", "m3": "a"},  # ...but members disagree
    )
    assert fire is True and reason == "disagreement"


def test_should_refine_on_low_margin() -> None:
    fire, reason = should_refine({"a": 0.52, "b": 0.48}, margin_tau=0.15)
    assert fire is True and reason == "margin"


def test_should_not_refine_easy_case() -> None:
    fire, reason = should_refine(
        {"a": 0.9, "b": 0.1},
        member_predictions={"m1": "a", "m2": "a", "m3": "a"},  # agree
        margin_tau=0.15,
    )
    assert fire is False and reason == "high_margin"


def test_refine_posterior_alpha_zero_is_identity() -> None:
    post = {"a": 0.6, "b": 0.4}
    assert refine_posterior(post, {"a": 0.0, "b": 1.0}, alpha=0.0) == post


def test_refine_posterior_no_overlap_is_identity() -> None:
    post = {"a": 0.6, "b": 0.4}
    # FarSLIP scored only classes outside the posterior -> nothing to blend.
    assert refine_posterior(post, {"z": 1.0}, alpha=0.5) == post


def test_refine_posterior_blends_and_renormalizes() -> None:
    refined = refine_posterior({"a": 0.5, "b": 0.5}, {"a": 1.0, "b": 0.0}, alpha=0.4)
    assert sum(refined.values()) == pytest.approx(1.0)
    # FarSLIP favours 'a' -> 'a' gets more mass than 'b'.
    assert refined["a"] > refined["b"]
    assert refined["a"] == pytest.approx(0.7)


def test_apply_refinement_can_flip_the_decision() -> None:
    # Voting marginally prefers 'b' (margin 0.2), but the members disagree, so the
    # refinement fires; FarSLIP strongly favours 'a' and flips the decision.
    result = apply_refinement(
        {"a": 0.4, "b": 0.6},
        {"a": 1.0, "b": 0.0},
        member_predictions={"m1": "a", "m2": "b"},
        alpha=0.6,
    )
    assert result.refined is True
    assert result.reason == "disagreement"
    assert result.top_class_before == "b"
    assert result.top_class_after == "a"
    assert sum(result.posterior.values()) == pytest.approx(1.0)


def test_apply_refinement_easy_case_is_untouched() -> None:
    post = {"a": 0.9, "b": 0.1}
    result = apply_refinement(
        post,
        {"a": 0.0, "b": 1.0},  # FarSLIP disagrees, but the easy case must NOT flip
        member_predictions={"m1": "a", "m2": "a"},
        margin_tau=0.15,
    )
    assert result.refined is False
    assert result.reason == "high_margin"
    assert result.posterior == post
    assert result.top_class_after == "a"


def test_apply_refinement_without_scores_reports_no_scores() -> None:
    post = {"a": 0.5, "b": 0.5}
    result = apply_refinement(post, None, open_set=True)
    assert result.refined is False
    assert result.reason == "no_scores"
    assert result.posterior == post
