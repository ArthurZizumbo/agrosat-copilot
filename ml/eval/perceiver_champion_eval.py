"""Perceiver champion-vs-baseline evaluation (US-046 / US-049 re-wiring check).

Quantifies the impact of re-wiring the agent perceiver from the ``xgb-alphaearth``
baseline to the Stacking-5 champion (EPIC 6 / US-043 winner) restricted to the
nine well-resolved ``france-9`` classes. Over the real fold-5 OOF universe (the
parcels held out from every base member, leak-free), it compares, per parcel:

* ``xgb-alphaearth`` argmax (the OLD perceiver path),
* ``Stacking-5`` argmax (the NEW perceiver path), both restricted to ``france-9``,

against the per-parcel semantic18 ground truth reconstructed from PASTIS-R. It
reports the france-9 accuracy and macro-F1 of each, plus the agreement and the
net parcels the champion fixes vs breaks. This is the project-grounded evidence
that the re-wiring improves the agent's perception (the US-049 system-eval uses a
stub classifier by design, so it cannot show this difference).

The evaluation reuses the cached loaders of :mod:`ml.agent.tools.classify`
(``_load_stacking_five``, ``_load_classifier``) and the ground-truth reconstructor
(``_build_parcel_ground_truth``) -- it never re-implements the model logic. It is
CPU-only (no GPU, no raster): the dense members are consumed through their
pre-materialised OOF, exactly like the deployed perceiver.

Run:
    poetry run python -m ml.eval.perceiver_champion_eval \
        --out reports/agent_bench/perceiver_champion_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from ml.eval.class_remap import get_label_space, restrict_posterior

logger = structlog.get_logger(__name__)

#: france-9 label-space (the nine classes the champion resolves over).
_LABEL_SPACE_NAME = "france-9"


def _restricted_argmax(proba18: np.ndarray, label_space: Any) -> int | None:
    """Argmax semantic18 class id after restricting an 18-vector to france-9.

    Args:
        proba18: A ``(18,)`` post-softmax posterior over the semantic18 space.
        label_space: The active france-9 :class:`~ml.eval.class_remap.LabelSpace`.

    Returns:
        The semantic18 id of the top france-9 class, or ``None`` when no mass
        landed on the resolved classes (an honest abstention).
    """
    restricted = restrict_posterior(proba18, label_space)
    if not restricted or max(restricted.values()) <= 0.0:
        return None
    return max(restricted, key=lambda cid: restricted[cid])


def _macro_f1(y_true: list[int], y_pred: list[int | None]) -> float:
    """Macro-F1 over the union of observed france-9 classes (None preds count as wrong).

    Args:
        y_true: Ground-truth semantic18 ids.
        y_pred: Predicted semantic18 ids (``None`` for abstentions, scored as miss).

    Returns:
        Unweighted mean per-class F1 over the classes present in ``y_true``.
    """
    classes = sorted(set(y_true))
    f1s: list[float] = []
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == cls and p != cls)
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def evaluate(out_path: Path | None = None) -> dict[str, Any]:
    """Compare champion vs baseline perception over the fold-5 OOF universe.

    Loads the Stacking-5 meta and the XGBoost-AlphaEarth classifier (cached), scores
    every fold-5 parcel both ways restricted to france-9, and contrasts them against
    the PASTIS-R ground truth.

    Args:
        out_path: Optional path to dump the JSON summary.

    Returns:
        A summary dict with per-model france-9 accuracy/macro-F1, agreement, and the
        net parcels the champion fixes vs breaks relative to the baseline.

    Raises:
        FileNotFoundError: if the fold-5 OOF parquets or PASTIS-R ground truth are
            unavailable (run ``dvc pull ml/eval/oof`` / ``dvc pull data/PASTIS-R``).
    """
    from ml.agent.tools import classify as cls

    label_space = get_label_space(_LABEL_SPACE_NAME)

    stacking = cls._load_stacking_five()

    # Ground truth for every parcel the Stacking-5 meta can score (its joined OOF).
    canonical_ids = list(stacking.meta_features_by_id.keys())
    gt_frame = cls._build_parcel_ground_truth(canonical_ids)
    gt_by_id = dict(
        zip(
            gt_frame.get_column("canonical_parcel_id").to_list(),
            gt_frame.get_column("label").to_list(),
            strict=True,
        )
    )

    kept = set(label_space.kept_class_ids)
    y_true: list[int] = []
    pred_champion: list[int | None] = []
    pred_baseline: list[int | None] = []

    for cid, label in gt_by_id.items():
        if int(label) not in kept:
            continue  # GT outside france-9: not scorable in this label-space.
        champ_proba = stacking.posterior_for_parcel(cid)
        if champ_proba is None:
            continue
        y_true.append(int(label))
        pred_champion.append(_restricted_argmax(champ_proba, label_space))
        # Baseline: the xgb-alphaearth posterior needs the parcel embedding, which
        # is not in the OOF dump. The OOF parquet of xgb-alphaearth IS its fold-5
        # posterior, so reuse it directly as the baseline prediction (same source
        # the perceiver's degraded path would produce).
        base_proba = stacking.meta_features_by_id[cid]
        # meta-features are the 5 members x 18 probs concatenated; the
        # xgb-alphaearth block is identified by its column order in _STACKING_MEMBERS.
        xgb_index = cls._STACKING_MEMBERS.index("xgb-alphaearth")
        base_block = base_proba[xgb_index * 18 : (xgb_index + 1) * 18]
        pred_baseline.append(_restricted_argmax(np.asarray(base_block), label_space))

    n = len(y_true)
    if n == 0:
        raise ValueError("no france-9 parcels with both GT and a champion posterior.")

    champ_acc = sum(1 for t, p in zip(y_true, pred_champion, strict=True) if t == p) / n
    base_acc = sum(1 for t, p in zip(y_true, pred_baseline, strict=True) if t == p) / n
    agreement = sum(
        1 for a, b in zip(pred_champion, pred_baseline, strict=True) if a == b
    ) / n
    champion_fixes = sum(
        1
        for t, c, b in zip(y_true, pred_champion, pred_baseline, strict=True)
        if c == t and b != t
    )
    champion_breaks = sum(
        1
        for t, c, b in zip(y_true, pred_champion, pred_baseline, strict=True)
        if c != t and b == t
    )

    summary = {
        "label_space": _LABEL_SPACE_NAME,
        "n_parcels": n,
        "baseline_xgb": {
            "accuracy": round(base_acc, 4),
            "macro_f1": round(_macro_f1(y_true, pred_baseline), 4),
        },
        "champion_stacking5": {
            "accuracy": round(champ_acc, 4),
            "macro_f1": round(_macro_f1(y_true, pred_champion), 4),
        },
        "delta_accuracy": round(champ_acc - base_acc, 4),
        "agreement": round(agreement, 4),
        "champion_fixes": champion_fixes,
        "champion_breaks": champion_breaks,
        "net_fixed": champion_fixes - champion_breaks,
    }
    logger.info("perceiver_champion_eval_done", **{
        k: v for k, v in summary.items() if not isinstance(v, dict)
    })

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("perceiver_champion_eval_written", path=str(out_path))
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the perceiver champion-vs-baseline evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/agent_bench/perceiver_champion_eval.json"),
        help="Path to dump the JSON summary.",
    )
    args = parser.parse_args(argv)
    summary = evaluate(out_path=args.out)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
