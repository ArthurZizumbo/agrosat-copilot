"""Evaluate the US-080 FarSLIP refinement: F1-macro Voting-3 vs Voting-3+refine.

Measures, on the real PASTIS-R fold-5 OOF, how much the conditional FarSLIP
second stage (:mod:`ml.agent.refine`) moves the F1-macro of the deployment champion
-- globally and on the subset of parcels where the refinement actually fired
(AC5/AC6). REAL VALUES ONLY: the FarSLIP scoring is injected, so the pure
computation is unit-tested offline; the live run needs the FarSLIP model + the
per-parcel chips (the documented blocker) and reports the delta as measured,
positive or not.

The core (:func:`f1_macro`, :func:`run_refine_eval`) is pure -- it consumes the
Voting-3 posteriors, the ground truth and a per-parcel FarSLIP scorer -- so a test
drives it with fakes. :func:`main` wires the REAL inputs (the cached Voting-3 OOF,
the reconstructed GT, the FarSLIP zero-shot head over the chips) and logs to MLflow.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import structlog

logger = structlog.get_logger(__name__)

__all__ = ["RefineEvalReport", "f1_macro", "run_refine_eval"]

#: A per-parcel FarSLIP scorer: ``canonical_id -> {class_name: score}`` (or ``None``
#: when the chip / FarSLIP signal is unavailable for that parcel).
FarSLIPScorer = Callable[[str], dict[str, float] | None]


def f1_macro(y_true: list[str], y_pred: list[str], *, labels: list[str] | None = None) -> float:
    """Compute the macro-averaged F1 over ``labels`` (pure, no sklearn needed).

    Args:
        y_true: Ground-truth class names, aligned with ``y_pred``.
        y_pred: Predicted class names.
        labels: Class set to average over; inferred from ``y_true`` when ``None``.

    Returns:
        The unweighted mean per-class F1 in ``[0, 1]`` (``0.0`` for empty input).
    """
    if not y_true:
        return 0.0
    classes = labels if labels is not None else sorted(set(y_true))
    f1s: list[float] = []
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == cls and p != cls)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp) / denom if denom > 0 else 0.0)
    return float(sum(f1s) / len(f1s)) if f1s else 0.0


class RefineEvalReport(dict):
    """Plain dict report (keys documented in :func:`run_refine_eval`)."""


def run_refine_eval(
    voting_posteriors: Mapping[str, dict[str, float]],
    ground_truth: Mapping[str, str],
    scorer: FarSLIPScorer,
    *,
    member_predictions: Mapping[str, dict[str, str]] | None = None,
    alpha: float = 0.4,
    margin_tau: float = 0.15,
) -> RefineEvalReport:
    """Compare Voting-3 vs Voting-3+refine F1-macro over the labelled parcels.

    For every parcel with a ground-truth label, the Voting-3 argmax is the baseline
    prediction; the gated FarSLIP refinement (:func:`ml.agent.refine.apply_refinement`)
    may re-rank it. F1-macro is computed before and after, globally and on the
    "fired" subset (the parcels where the refinement actually engaged).

    Args:
        voting_posteriors: ``canonical_id -> {class_name: probability}`` Voting-3
            posterior (restricted to the active label-space).
        ground_truth: ``canonical_id -> true class name``.
        scorer: Per-parcel FarSLIP scorer (injected; a fake in tests).
        member_predictions: Optional ``canonical_id -> {member: argmax class}`` for
            the disagreement trigger.
        alpha: Convex weight of the FarSLIP signal when the refinement fires.
        margin_tau: Uncertainty margin threshold for the trigger.

    Returns:
        A :class:`RefineEvalReport` with ``f1_before`` / ``f1_after`` / ``delta_f1``
        (global), ``f1_before_fired`` / ``f1_after_fired`` / ``delta_f1_fired``
        (the fired subset), ``n_parcels`` / ``n_fired`` / ``n_changed``.
    """
    from ml.agent.refine import apply_refinement

    labels = sorted({cls for post in voting_posteriors.values() for cls in post})
    y_true: list[str] = []
    y_before: list[str] = []
    y_after: list[str] = []
    fired_idx: list[int] = []
    n_changed = 0

    for canonical_id, truth in ground_truth.items():
        posterior = voting_posteriors.get(canonical_id)
        if not posterior:
            continue
        result = apply_refinement(
            dict(posterior),
            scorer(canonical_id),
            member_predictions=member_predictions.get(canonical_id) if member_predictions else None,
            alpha=alpha,
            margin_tau=margin_tau,
        )
        y_true.append(truth)
        y_before.append(result.top_class_before)
        y_after.append(result.top_class_after)
        if result.refined:
            fired_idx.append(len(y_true) - 1)
            if result.top_class_after != result.top_class_before:
                n_changed += 1

    f1_before = f1_macro(y_true, y_before, labels=labels)
    f1_after = f1_macro(y_true, y_after, labels=labels)
    fired_true = [y_true[i] for i in fired_idx]
    fired_before = [y_before[i] for i in fired_idx]
    fired_after = [y_after[i] for i in fired_idx]
    f1_before_fired = f1_macro(fired_true, fired_before, labels=labels)
    f1_after_fired = f1_macro(fired_true, fired_after, labels=labels)

    report = RefineEvalReport(
        n_parcels=len(y_true),
        n_fired=len(fired_idx),
        n_changed=n_changed,
        f1_before=round(f1_before, 4),
        f1_after=round(f1_after, 4),
        delta_f1=round(f1_after - f1_before, 4),
        f1_before_fired=round(f1_before_fired, 4),
        f1_after_fired=round(f1_after_fired, 4),
        delta_f1_fired=round(f1_after_fired - f1_before_fired, 4),
    )
    logger.info("farslip_refine_eval_done", **report)
    return report
