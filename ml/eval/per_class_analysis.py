"""Per-class cardinality analysis OVER the final ensemble (Stacking-5 / Blending-5).

This module answers the question the user posed: *how many crop classes does the
**final ensemble** predict well, and which should be dropped?* It is deliberately
applied to the **ensemble** predictions (e.g. Stacking-5 or Blending-5), NOT to
each FarSLIP member in isolation.

It exposes three pure functions (no GPU, no PASTIS I/O) that operate on already
materialized predictions in the contiguous 18-class space ``[0..17]`` (NOT the
raw PASTIS ``1..18`` ids):

* :func:`per_class_report` — one row per class with precision / recall / F1 /
  IoU (and optional Average Precision) computed from a unified 18x18 confusion
  matrix, reusing :class:`ml.eval.dense_metrics.DenseConfusionAccumulator` (DRY).
* :func:`cardinality_cutoff_curve` — the top-K curve: keeping the K classes with
  the best F1, recompute the macro-F1 over *those K classes only*. This traces
  "up to how many classes the ensemble predicts well".
* :func:`recommend_classes_to_drop` — crosses the per-class F1 with the support
  bands of :func:`ml.utils.class_distribution.class_distribution_report` and
  flags classes that are simultaneously low-F1 and low-support.

Anti-leakage (R-LEAK, rule #1)
------------------------------
The cardinality cut-off (which classes to keep / drop) MUST be **decided** with
the F1 measured on the OOF sub-folds (folds 1-4), never on the held-out fold-5,
to avoid cherry-picking the report against the very split it is reported on. The
caller therefore:

1. computes :func:`per_class_report` / :func:`cardinality_cutoff_curve` /
   :func:`recommend_classes_to_drop` with the **OOF** ``y_true`` / ``y_pred``
   when *deciding* the cut, and
2. reports :func:`per_class_report` on fold-5 **once**, with the decision frozen.

These functions are agnostic to which split they receive; the discipline lives
in the caller. The docstrings restate it so the convention is not lost.

Contiguous index convention
---------------------------
The class axis is the contiguous ``[0..17]`` training space. Contiguous index
``c`` corresponds to PASTIS-R ``class_id = c + 1``, so the readable name is
``PASTIS_R_CLASSES[c + 1]`` (``c=0`` -> "Meadow", ..., ``c=17`` -> "Sorghum").
Background (0) and Void (19) never appear in this space.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import structlog

from ml.eval.dense_metrics import DenseConfusionAccumulator
from ml.ingest.pastis_loader import PASTIS_R_CLASSES
from ml.utils.class_distribution import SupportBand

logger = structlog.get_logger(__name__)

__all__ = [
    "cardinality_cutoff_curve",
    "per_class_report",
    "recommend_classes_to_drop",
]

#: Number of contiguous agronomic classes in the semantic18 space.
_NUM_CLASSES: int = 18

#: Default ignore label (Background/Void) excluded from every computation.
_IGNORE_INDEX: int = 255


def _contiguous_class_name(contiguous_id: int) -> str:
    """Translate a contiguous ``[0..17]`` index into its readable PASTIS-R name.

    Args:
        contiguous_id: Class index in the contiguous training space ``[0..17]``.

    Returns:
        The readable name from :data:`PASTIS_R_CLASSES` at ``contiguous_id + 1``
        (the raw PASTIS id), or ``"class_<id>"`` if it is out of range.
    """
    raw_id = int(contiguous_id) + 1
    return PASTIS_R_CLASSES.get(raw_id, f"class_{int(contiguous_id)}")


def _per_class_from_cm(cm: np.ndarray) -> dict[str, np.ndarray]:
    """Derive per-class support / precision / recall / F1 / IoU from a CM.

    The confusion matrix follows the convention rows = ground truth, columns =
    prediction. Classes absent from the ground truth (``support == 0``) get
    ``nan`` for precision / recall / F1 / IoU so they can be excluded from any
    macro average and surfaced as ``null`` in the report.

    Args:
        cm: Dense ``(C, C)`` confusion matrix (``int`` or ``float``).

    Returns:
        Dict of ``(C,)`` numpy arrays with keys ``support`` (``int64``),
        ``precision``, ``recall``, ``f1`` and ``iou`` (``float64`` with ``nan``
        for absent classes).
    """
    cm_f = cm.astype(np.float64)
    tp = np.diag(cm_f)
    support = cm_f.sum(axis=1)  # real pixels/parcels per class (row sum)
    predicted = cm_f.sum(axis=0)  # times each class was predicted (column sum)
    fp = predicted - tp
    fn = support - tp
    union = support + predicted - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0.0, tp / predicted, np.nan)
        recall = np.where(support > 0.0, tp / support, np.nan)
        denom_f1 = 2.0 * tp + fp + fn
        f1 = np.where(denom_f1 > 0.0, 2.0 * tp / denom_f1, np.nan)
        iou = np.where(union > 0.0, tp / union, np.nan)

    # A class absent from the ground truth (support==0) is "not predicted by the
    # data": its precision/recall/F1/IoU are undefined -> nan (reported null).
    absent = support <= 0.0
    precision = np.where(absent, np.nan, precision)
    recall = np.where(absent, np.nan, recall)
    f1 = np.where(absent, np.nan, f1)
    iou = np.where(absent, np.nan, iou)

    return {
        "support": support.astype(np.int64),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def _average_precision_per_class(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    num_classes: int,
    ignore_index: int,
) -> np.ndarray:
    """One-vs-rest Average Precision per class from soft probabilities.

    Computes ``sklearn.metrics.average_precision_score`` in one-vs-rest mode for
    each class, using only the samples whose ground truth is valid (not
    ``ignore_index`` and inside ``[0, num_classes)``). A class with no positive
    sample in ``y_true`` gets ``nan`` (AP is undefined without positives).

    Args:
        y_true: Flat ground-truth labels ``(N,)`` in the contiguous space.
        proba: Soft probabilities ``(N, num_classes)`` (post-softmax, rows sum
            to ~1).
        num_classes: Number of contiguous classes.
        ignore_index: Label to exclude from the computation.

    Returns:
        ``(num_classes,)`` float64 array of AP values (``nan`` where undefined).
    """
    from sklearn.metrics import average_precision_score

    valid = (y_true != ignore_index) & (y_true >= 0) & (y_true < num_classes)
    y_valid = y_true[valid]
    p_valid = proba[valid]

    ap = np.full(num_classes, np.nan, dtype=np.float64)
    for c in range(num_classes):
        positives = y_valid == c
        n_pos = int(positives.sum())
        if n_pos == 0 or n_pos == y_valid.size:
            # No positives (or all positives) -> AP is undefined / degenerate.
            continue
        ap[c] = float(average_precision_score(positives.astype(np.int64), p_valid[:, c]))
    return ap


def per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None = None,
    *,
    class_names: dict[int, str] = PASTIS_R_CLASSES,
    num_classes: int = _NUM_CLASSES,
    ignore_index: int = _IGNORE_INDEX,
) -> pl.DataFrame:
    """Per-class precision / recall / F1 / IoU report for the **ensemble**.

    Builds a single ``num_classes x num_classes`` confusion matrix with
    :class:`ml.eval.dense_metrics.DenseConfusionAccumulator` (DRY: the same
    accumulator EPIC 5/6 uses) and derives the per-class metrics from it. This is
    meant to be fed the predictions of the FINAL ENSEMBLE (Stacking-5 /
    Blending-5), not those of an individual member.

    Anti-leakage (R-LEAK): when this report is used to *decide* a cardinality
    cut, the caller must pass the OOF sub-fold (folds 1-4) ``y_true`` / ``y_pred``
    so the cut is not cherry-picked against fold-5. The final fold-5 report is
    produced once, after the cut is frozen.

    Args:
        y_true: Ground-truth labels (any shape) in the contiguous ``[0..17]``
            space; ``ignore_index`` pixels/parcels are dropped.
        y_pred: Hard predicted labels, same number of elements as ``y_true``.
        proba: Optional soft probabilities ``(N, num_classes)`` (post-softmax).
            When provided, a one-vs-rest ``ap`` (Average Precision) column is
            added. ``N`` must equal the number of elements in ``y_true``.
        class_names: Map ``{raw_class_id: name}`` (defaults to
            :data:`PASTIS_R_CLASSES`). The contiguous index ``c`` is looked up at
            ``c + 1`` (raw PASTIS id).
        num_classes: Number of contiguous classes (18 for semantic PASTIS-R).
        ignore_index: Label to ignore (Background/Void).

    Returns:
        Polars DataFrame with one row per contiguous class, columns
        ``class_id`` (contiguous ``[0..num_classes)``), ``name``, ``support``,
        ``precision``, ``recall``, ``f1``, ``iou`` (and ``ap`` if ``proba`` was
        given). Absent classes have ``support == 0`` and ``null`` metrics. Sorted
        by ``f1`` descending (``null`` F1 last).

    Raises:
        ValueError: if ``y_true`` / ``y_pred`` differ in element count, or if
            ``proba`` is given with a mismatched number of rows or columns.
    """
    true = np.asarray(y_true).reshape(-1).astype(np.int64)
    pred = np.asarray(y_pred).reshape(-1).astype(np.int64)
    if true.shape != pred.shape:
        raise ValueError(
            f"`y_true` and `y_pred` must have the same number of elements; "
            f"got {true.size} vs {pred.size}."
        )

    accumulator = DenseConfusionAccumulator(num_classes, ignore_index=ignore_index)
    accumulator.update(pred, true)
    cm = accumulator.confusion_matrix()
    per_class = _per_class_from_cm(cm)

    columns: dict[str, object] = {
        "class_id": list(range(num_classes)),
        "name": [
            class_names.get(c + 1, f"class_{c}") for c in range(num_classes)
        ],
        "support": per_class["support"].tolist(),
        "precision": [
            None if np.isnan(v) else float(v) for v in per_class["precision"]
        ],
        "recall": [None if np.isnan(v) else float(v) for v in per_class["recall"]],
        "f1": [None if np.isnan(v) else float(v) for v in per_class["f1"]],
        "iou": [None if np.isnan(v) else float(v) for v in per_class["iou"]],
    }

    if proba is not None:
        proba_arr = np.asarray(proba, dtype=np.float64)
        if proba_arr.ndim != 2 or proba_arr.shape[1] != num_classes:
            raise ValueError(
                f"`proba` must have shape (N, {num_classes}); "
                f"got {proba_arr.shape}."
            )
        if proba_arr.shape[0] != true.size:
            raise ValueError(
                f"`proba` rows ({proba_arr.shape[0]}) must match `y_true` "
                f"elements ({true.size})."
            )
        ap = _average_precision_per_class(
            true, proba_arr, num_classes=num_classes, ignore_index=ignore_index
        )
        columns["ap"] = [None if np.isnan(v) else float(v) for v in ap]

    report = pl.DataFrame(columns)
    logger.info(
        "per_class_report",
        num_classes=num_classes,
        n_present=int((per_class["support"] > 0).sum()),
        with_ap=proba is not None,
    )
    # Sort by F1 descending; absent/undefined-F1 classes (null) go last.
    return report.sort("f1", descending=True, nulls_last=True)


def cardinality_cutoff_curve(
    report_df: pl.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int = _NUM_CLASSES,
    ignore_index: int = _IGNORE_INDEX,
) -> pl.DataFrame:
    """Top-K cardinality curve: macro-F1 over the K best-F1 classes only.

    For ``K = 1 .. n_present`` (number of classes with support in
    ``report_df``), keep the K classes with the highest per-class F1 and
    recompute the **macro-F1 over those K retained classes only**. The semantics
    is "up to how many classes does the ensemble predict well": as K grows the
    retained set adds progressively worse classes, so ``macro_f1_topk`` typically
    decreases (it is monotone non-increasing by construction, since the K-th best
    F1 is <= the mean of the top K-1 only when... see note). The curve makes the
    cardinality/quality trade-off explicit.

    Note on monotonicity: ``macro_f1_topk`` is the mean of the K largest
    per-class F1 values, so it is **monotone non-increasing in K** — every step
    adds a class whose F1 is <= the current mean, which can only keep the mean
    equal or pull it down. ``cumulative_support_share`` is monotone
    non-decreasing in K.

    The per-class F1 used to RANK and to AVERAGE comes straight from
    ``report_df`` (so the caller controls the split: pass the OOF report to
    *decide* the cut, anti-leakage R-LEAK). ``y_true`` / ``y_pred`` are used only
    to compute ``cumulative_support_share`` (the fraction of valid samples that
    the retained classes cover in the same split).

    Args:
        report_df: Output of :func:`per_class_report` (must have ``class_id``,
            ``f1``, ``support`` columns). Classes with ``null`` F1 are treated as
            absent and never retained.
        y_true: Ground-truth labels (same split that produced ``report_df``);
            used only for the support share denominator.
        y_pred: Hard predictions (unused for the curve values, accepted for
            signature symmetry and future per-K recomputations).
        num_classes: Number of contiguous classes.
        ignore_index: Label to ignore when counting the support denominator.

    Returns:
        Polars DataFrame with one row per ``k`` (1-indexed) and columns:
        ``k`` (number of retained classes), ``kept_class_ids`` (list of the
        contiguous ids retained at this K, best-F1 first), ``macro_f1_topk``
        (mean F1 over the K retained classes) and ``cumulative_support_share``
        (share of valid ground-truth samples belonging to the retained classes).
        Empty if no class has a defined F1.
    """
    del y_pred  # Accepted for symmetry; the curve is driven by report_df F1.

    present = (
        report_df.filter(pl.col("f1").is_not_null())
        .sort("f1", descending=True, nulls_last=True)
    )
    n_present = present.height
    if n_present == 0:
        return pl.DataFrame(
            schema={
                "k": pl.Int64,
                "kept_class_ids": pl.List(pl.Int64),
                "macro_f1_topk": pl.Float64,
                "cumulative_support_share": pl.Float64,
            }
        )

    ranked_ids = [int(c) for c in present["class_id"].to_list()]
    ranked_f1 = [float(v) for v in present["f1"].to_list()]

    # Support denominator: total valid samples in this split.
    true = np.asarray(y_true).reshape(-1).astype(np.int64)
    valid = (true != ignore_index) & (true >= 0) & (true < num_classes)
    true_valid = true[valid]
    total_valid = int(true_valid.size)
    per_class_support = {
        c: int((true_valid == c).sum()) for c in ranked_ids
    }

    rows: list[dict[str, object]] = []
    cumulative_support = 0
    for k in range(1, n_present + 1):
        kept = ranked_ids[:k]
        macro_f1_topk = float(np.mean(ranked_f1[:k]))
        cumulative_support += per_class_support[ranked_ids[k - 1]]
        support_share = (
            0.0 if total_valid == 0 else cumulative_support / total_valid
        )
        rows.append(
            {
                "k": k,
                "kept_class_ids": kept,
                "macro_f1_topk": macro_f1_topk,
                "cumulative_support_share": support_share,
            }
        )

    logger.info(
        "cardinality_cutoff_curve",
        n_present=n_present,
        best_f1=round(ranked_f1[0], 4),
        full_macro_f1=round(float(np.mean(ranked_f1)), 4),
    )
    return pl.DataFrame(rows)


def recommend_classes_to_drop(
    report_df: pl.DataFrame,
    dist_report: pl.DataFrame,
    *,
    f1_threshold: float = 0.30,
    support_bands: tuple[SupportBand, ...] = ("low", "very_low"),
) -> pl.DataFrame:
    """Flag classes to drop: low F1 AND low support (crossed criterion).

    Crosses the per-class F1 of the **ensemble** (:func:`per_class_report`) with
    the support bands of
    :func:`ml.utils.class_distribution.class_distribution_report`. A class is
    marked for dropping only when it is **simultaneously** below the F1 threshold
    AND in one of the low support bands. Requiring both avoids dropping a rare
    class the ensemble still predicts well, or a frequent class that merely has a
    bad F1 (which deserves modelling attention, not removal).

    Anti-leakage (R-LEAK): the F1 used here must be the OOF sub-fold F1 (folds
    1-4), so the decision of which classes to drop is taken without ever looking
    at fold-5. The caller passes the OOF ``report_df``; fold-5 is reported once,
    after the drop list is frozen.

    The join is on the RAW PASTIS ``class_id`` (1..18): ``report_df`` uses the
    contiguous id ``c``, so it is translated to ``c + 1`` to match the raw
    ``class_id`` of ``dist_report``.

    Args:
        report_df: Output of :func:`per_class_report` (contiguous ``class_id``,
            ``f1``, ``support``).
        dist_report: Output of
            :func:`ml.utils.class_distribution.class_distribution_report` (raw
            ``class_id``, ``support_band``, ``n_parcels``, ``share``).
        f1_threshold: Classes with ``f1 < f1_threshold`` are candidates. A
            ``null`` F1 (class absent from the report split) also qualifies as
            below threshold.
        support_bands: Support bands considered "low" for the drop criterion.

    Returns:
        Polars DataFrame with one row per class evaluated, columns:
        ``class_id`` (raw PASTIS id), ``contiguous_id``, ``name``, ``f1``,
        ``support`` (from the report), ``support_band`` and ``n_parcels`` (from
        the distribution), ``below_f1`` (bool), ``low_support`` (bool) and
        ``drop`` (bool = ``below_f1 AND low_support``). Sorted with the dropped
        classes first, then by ``f1`` ascending.
    """
    bands = list(support_bands)

    report = report_df.select(
        (pl.col("class_id") + 1).alias("class_id"),  # contiguous -> raw PASTIS id
        pl.col("class_id").alias("contiguous_id"),
        "name",
        "f1",
        "support",
    )
    dist = dist_report.select(
        pl.col("class_id").cast(pl.Int64),
        "support_band",
        "n_parcels",
    )

    crossed = report.join(dist, on="class_id", how="left").with_columns(
        # A null F1 (absent class) is treated as below the threshold.
        (pl.col("f1").is_null() | (pl.col("f1") < f1_threshold)).alias("below_f1"),
        pl.col("support_band").is_in(bands).fill_null(False).alias("low_support"),
    )
    crossed = crossed.with_columns(
        (pl.col("below_f1") & pl.col("low_support")).alias("drop")
    )

    logger.info(
        "recommend_classes_to_drop",
        f1_threshold=f1_threshold,
        support_bands=bands,
        n_drop=int(crossed.filter(pl.col("drop")).height),
        n_total=crossed.height,
    )
    # Dropped classes first, then ascending F1 (worst-defined first; null last).
    return crossed.sort(
        ["drop", "f1"], descending=[True, False], nulls_last=True
    )
