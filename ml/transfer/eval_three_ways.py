"""The three label-space VIAS of the honest Italian TL re-evaluation (US-082).

Arthur's decision: re-evaluate the complete-dataset transfer along THREE
explicit vias and compare them, instead of a single global macro-F1 that hides
which classes actually rescue with real support. The three vias share the SAME
dense predictions and ground truth (the held-out Italian fold-5); they differ
only in the LABEL SPACE the pixels are scored against:

- **VIA A -- native 39 HCAT leaves** (Arthur: "no reagrupar clases"). The fine
  Italian label space scored as-is. This is the honest per-class verdict: which
  of the 39 leaves clears F1 >= 0.6 / >= 0.8 with the full extraction. Maps to
  the FINE granularity of :mod:`ml.eval.transfer_italia_eval`.

- **VIA B -- mapped to the champion's input label space** (the conserved crosswalk
  to PASTIS-18 / france-12 that the French members already know). The Italian
  leaves collapse to their PASTIS parent via
  :meth:`ItaliaLabelSpace.coarse_of`, so a model that only speaks the coarse
  PASTIS taxonomy is scorable. This isolates how much of the transfer is "the
  champion already knew this crop" vs genuinely new. Maps to the COARSE
  granularity.

- **VIA C -- full procedure replicated end-to-end** on the new 1,438-patch
  dataset, exactly as PASTIS-France was done: AlphaEarth extraction -> per-member
  training (xgb / TSViT / U-TAE) -> fold-5 OOF -> Voting-3 -> dense eval. VIA C is
  the ORCHESTRATION (run on the H100 by the operator); this module scores its
  OUTPUT (the Voting-3 dense predictions) at BOTH the native (A) and the
  PASTIS-mapped (B) label spaces, and ties the three into one comparison table.

The comparison is reported as a flat table so the notebook and the handoff can
show "via -> macro-F1 -> n classes >= 0.6 -> n classes >= 0.8" side by side, with
the per-class breakdown preserved (no class silently dropped).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import structlog

from ml.eval.transfer_italia_eval import (
    DenseEvalResult,
    best_subset_over_threshold,
    build_coarse_label_space,
    evaluate_dense_predictions,
    per_class_f1,
)
from ml.transfer.italia_label_space import ItaliaLabelSpace

logger = structlog.get_logger(__name__)

__all__ = [
    "ThreeWayComparison",
    "ViaResult",
    "compare_three_ways",
    "count_classes_over",
]

#: The honest per-class F1 gates of US-082 (KPI-2): how many classes rescue at
#: each floor once the full extraction gives every class its real support.
_F1_GATES: tuple[float, ...] = (0.6, 0.8)


def count_classes_over(per_class: list[dict[str, object]], threshold: float) -> int:
    """Count classes whose per-class F1 clears ``threshold`` (background excluded).

    Args:
        per_class: The ``DenseEvalResult.per_class`` rows (each carries ``"f1"``).
        threshold: The F1 floor.

    Returns:
        Number of classes with ``f1 >= threshold``.
    """
    return sum(1 for row in per_class if float(cast("float", row["f1"])) >= threshold)


@dataclass
class ViaResult:
    """One via's scored result plus its per-class gate counts.

    Attributes:
        via: ``"A"`` (native 39), ``"B"`` (PASTIS-mapped), or ``"C"`` (full
            procedure, scored at the native granularity).
        label_space_name: Human label of the scored space (e.g. ``"italia-39"``).
        macro_f1: The macro-F1 over the scored classes.
        n_classes_scored: Number of classes with support in this via.
        classes_over: ``{gate: n_classes >= gate}`` for the US-082 F1 gates.
        best_subset: The largest top-n subset whose macro-F1 stays >= 0.6
            (mirror of the france-10 deployment subset).
        per_class: The full per-class breakdown (preserved, nothing dropped).
    """

    via: str
    label_space_name: str
    macro_f1: float
    n_classes_scored: int
    classes_over: dict[str, int]
    best_subset: dict[str, object]
    per_class: list[dict[str, object]]

    def row(self) -> dict[str, object]:
        """Return the flat comparison-table row (no per-class list)."""
        return {
            "via": self.via,
            "label_space": self.label_space_name,
            "macro_f1": round(self.macro_f1, 4),
            "n_classes_scored": self.n_classes_scored,
            "n_classes_ge_0.6": self.classes_over.get("0.6", 0),
            "n_classes_ge_0.8": self.classes_over.get("0.8", 0),
            "deploy_subset_n": int(cast("int", self.best_subset.get("n_classes", 0))),
            "deploy_subset_macro_f1": self.best_subset.get("macro_f1", 0.0),
        }


@dataclass
class ThreeWayComparison:
    """The A/B/C comparison of one model's Italian transfer (US-082).

    Attributes:
        model_name: The scored model/combiner (e.g. ``"voting-3-italia-full"``).
        via_a: Native 39-leaf result.
        via_b: PASTIS-mapped (crosswalk) result.
        via_c: Full-procedure result scored at the native granularity (the
            end-to-end pipeline output); ``None`` if VIA C was not run yet.
    """

    model_name: str
    via_a: ViaResult
    via_b: ViaResult
    via_c: ViaResult | None = None

    def table(self) -> list[dict[str, object]]:
        """Return the flat comparison table (one row per available via)."""
        rows = [self.via_a.row(), self.via_b.row()]
        if self.via_c is not None:
            rows.append(self.via_c.row())
        return rows


def _via_from_fine(via: str, label_space_name: str, result: DenseEvalResult) -> ViaResult:
    """Build a :class:`ViaResult` from a fine-granularity dense result (VIA A / C)."""
    return ViaResult(
        via=via,
        label_space_name=label_space_name,
        macro_f1=result.fine_f1_macro,
        n_classes_scored=len(result.per_class),
        classes_over={str(g): count_classes_over(result.per_class, g) for g in _F1_GATES},
        best_subset=best_subset_over_threshold(result, threshold=0.6),
        per_class=result.per_class,
    )


def _coarse_per_class(
    result_preds: dict[int, np.ndarray],
    result_masks: dict[int, np.ndarray],
    label_space: ItaliaLabelSpace,
) -> tuple[float, list[dict[str, object]]]:
    """Score the predictions at the COARSE (PASTIS-mapped) space -> (macro_f1, rows).

    VIA B: collapse both prediction and target to the conserved PASTIS bucket via
    the crosswalk LUT, then compute the coarse macro-F1 and per-coarse-class F1.
    """
    from ml.eval.dense_metrics import DenseConfusionAccumulator

    lut, coarse_names = build_coarse_label_space(label_space)
    ids = sorted(set(result_preds) & set(result_masks))
    preds = np.concatenate([result_preds[i].reshape(-1) for i in ids]).astype(np.int64)
    target = np.concatenate([result_masks[i].reshape(-1) for i in ids]).astype(np.int64)
    n_coarse = len(coarse_names)
    acc = DenseConfusionAccumulator(n_coarse, ignore_index=0)
    acc.update(lut[preds], lut[target])
    compute = acc.compute()
    per_f1 = per_class_f1(acc.confusion_matrix(), ignore_index=0)
    rows = [
        {"leaf": coarse_names.get(cid, str(cid)), "f1": round(float(f1), 4), "is_new": False}
        for cid, f1 in sorted(per_f1.items())
    ]
    return float(compute["f1_macro"]), rows


def compare_three_ways(
    model_name: str,
    preds_by_patch: dict[int, np.ndarray],
    masks_by_patch: dict[int, np.ndarray],
    *,
    label_space: ItaliaLabelSpace,
    is_full_procedure: bool = False,
) -> ThreeWayComparison:
    """Score one model's dense predictions along the three label-space vias.

    Args:
        model_name: The scored model/combiner name (for the table rows).
        preds_by_patch: ``{patch_id: (H, W)}`` predicted fine class maps.
        masks_by_patch: ``{patch_id: (H, W)}`` ground-truth fine class masks.
        label_space: The Italian fine label space (39 HCAT leaves + background).
        is_full_procedure: When ``True`` the scored predictions come from the
            VIA C end-to-end pipeline (extraction -> train -> OOF -> vote), so the
            native-granularity result is ALSO emitted as VIA C. When ``False``
            (e.g. a zero-shot or single-member input) only VIA A / VIA B are
            populated.

    Returns:
        A :class:`ThreeWayComparison` carrying VIA A (native 39), VIA B
        (PASTIS-mapped) and, when ``is_full_procedure``, VIA C.
    """
    fine = evaluate_dense_predictions(
        model_name, preds_by_patch, masks_by_patch, label_space=label_space
    )
    via_a = _via_from_fine("A", f"italia-{label_space.num_classes - 1}", fine)

    coarse_macro, coarse_rows = _coarse_per_class(preds_by_patch, masks_by_patch, label_space)
    via_b = ViaResult(
        via="B",
        label_space_name="pastis-crosswalk",
        macro_f1=coarse_macro,
        n_classes_scored=len(coarse_rows),
        classes_over={str(g): count_classes_over(coarse_rows, g) for g in _F1_GATES},
        best_subset={"n_classes": len(coarse_rows), "macro_f1": round(coarse_macro, 4)},
        per_class=coarse_rows,
    )

    via_c = (
        _via_from_fine("C", f"italia-{label_space.num_classes - 1}-fullproc", fine)
        if is_full_procedure
        else None
    )

    comparison = ThreeWayComparison(model_name=model_name, via_a=via_a, via_b=via_b, via_c=via_c)
    logger.info(
        "italia_three_way_compared",
        model=model_name,
        via_a_macro_f1=round(via_a.macro_f1, 4),
        via_b_macro_f1=round(via_b.macro_f1, 4),
        via_c_macro_f1=round(via_c.macro_f1, 4) if via_c else None,
        via_a_ge_0_6=via_a.classes_over.get("0.6", 0),
    )
    return comparison
