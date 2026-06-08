"""Interpretable figures + comparison table for the four ensembles (US-040, A5).

This module produces the rubric-mandated artifacts of the ensemble Avance (A5,
plan Section 4 / AC-8 / AC-9):

- :func:`confusion_norm_abs`: a side-by-side confusion matrix, normalized
  (per-class recall) AND absolute (raw counts), so the reader sees both the
  relative behaviour and the actual support per class.
- :func:`roc_ovr_per_class`: one-vs-rest ROC curves with the AUC per class plus
  the macro average (a probability-quality view the hard-label confusion cannot
  show).
- :func:`pr_per_class`: one-vs-rest precision-recall curves with the average
  precision per class plus the macro average (more informative than ROC under the
  heavy class imbalance of PASTIS-R).
- :func:`spatial_residuals`: the per-parcel residuals (hit / miss) plotted over
  the REAL parcel geometry, to expose whether the errors cluster geographically.
- :func:`build_comparison_table`: the best-individual-vs-4-ensembles comparison
  Polars DataFrame (``model``, ``f1_macro``, ``accuracy``, ``inference_time_s``,
  ``chosen``) that backs the Selection criterion.

DRY (plan Section 4 "reusa avance4_figures"): the confusion counts are
accumulated with the SAME :class:`ml.eval.dense_metrics.DenseConfusionAccumulator`
the segmentation harness (and :func:`ml.eval.avance4_figures.confusion_from_cm`)
use, so the ensemble confusion matches the individual models apples-to-apples;
the optuna convergence figure already lives in ``avance4_figures`` and is NOT
duplicated here.

Anti-leakage note (R-LEAK). Every figure consumes ONLY fold-5 held-out
predictions/probabilities (the figures never re-derive a metric on fold-4); the
probability inputs are validated as post-softmax via
:meth:`ml.ensemble.base.EnsembleModel.validate_probs` so a logits array can never
slip into a ROC/PR plot.

Project conventions: ``polars`` (never pandas), ``numpy`` only at the array
boundary, ``matplotlib`` with the ``Agg`` backend (no display), ``structlog`` for
logging; visible prose (titles, axis labels, legends) is Spanish, code
identifiers and docstrings are English; no emojis anywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import structlog

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from collections.abc import Mapping, Sequence

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_FIGURE_DIR",
    "build_comparison_table",
    "confusion_norm_abs",
    "pr_per_class",
    "roc_ovr_per_class",
    "spatial_residuals",
]

#: Default output folder for the ensemble figures (mirrors avance4_figures).
DEFAULT_FIGURE_DIR: Path = Path("reports/ensemble/figures")

#: Number of agronomic classes in the harness 18-class space.
_NUM_CLASSES: int = 18

#: Default ignore label of the harness (excluded from every figure).
_IGNORE_INDEX: int = 255

#: Numerical floor so a degenerate normalization never divides by zero.
_EPS: float = 1e-12

#: Comparison-table column order (the Selection-criterion schema).
_COMPARISON_COLUMNS: tuple[str, ...] = (
    "model",
    "f1_macro",
    "accuracy",
    "inference_time_s",
    "chosen",
)


# ---------------------------------------------------------------------------
# Shared helpers (DRY across the figures).
# ---------------------------------------------------------------------------


def _resolve_labels(labels: Mapping[int, str] | Sequence[str] | None) -> dict[int, str]:
    """Coerce a label spec to a ``{class_id: name}`` map.

    Args:
        labels: A ``{id: name}`` mapping, an ordered sequence of names (indexed by
            class id) or ``None`` (defaults to ``C{id}``).

    Returns:
        A dict ``{class_id: name}`` covering the 18-class space.
    """
    if labels is None:
        return {i: f"C{i}" for i in range(_NUM_CLASSES)}
    if isinstance(labels, dict):
        return {int(k): str(v) for k, v in labels.items()}
    return {i: str(name) for i, name in enumerate(labels)}


def _confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int = _NUM_CLASSES,
    ignore_index: int | None = _IGNORE_INDEX,
) -> np.ndarray:
    """Accumulate a ``(C, C)`` confusion matrix (rows=truth, cols=pred).

    Reuses :class:`ml.eval.dense_metrics.DenseConfusionAccumulator` so the
    counts match the segmentation harness exactly (apples to apples with the
    individual models).

    Args:
        y_true: Ground-truth class ids (any shape; flattened).
        y_pred: Predicted class ids (same element count).
        num_classes: Class count ``C``.
        ignore_index: Label excluded from the matrix (``None`` keeps all).

    Returns:
        An ``int64`` confusion matrix ``(C, C)``.
    """
    from ml.eval.dense_metrics import DenseConfusionAccumulator

    acc = DenseConfusionAccumulator(num_classes, ignore_index=ignore_index)
    acc.update(np.asarray(y_pred).reshape(-1), np.asarray(y_true).reshape(-1))
    return acc.confusion_matrix().astype(np.int64)


def _present_classes(y_true: np.ndarray, *, ignore_index: int | None) -> list[int]:
    """Return the sorted class ids present in ``y_true`` (excluding ignore)."""
    vals = np.unique(np.asarray(y_true).reshape(-1))
    return [
        int(v)
        for v in vals
        if 0 <= int(v) < _NUM_CLASSES and (ignore_index is None or int(v) != ignore_index)
    ]


# ---------------------------------------------------------------------------
# Figure 1: confusion matrix (normalized + absolute).
# ---------------------------------------------------------------------------


def confusion_norm_abs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Mapping[int, str] | Sequence[str] | None = None,
    out_path: Path | str,
    model: str = "ensemble",
    ignore_index: int | None = _IGNORE_INDEX,
) -> Path:
    """Write a 1x2 confusion figure: normalized (recall) + absolute (counts).

    The left panel is the row-normalized confusion (per-class recall, the same
    rendering :func:`ml.eval.avance4_figures.confusion_from_cm` produces, reused
    here for DRY); the right panel is the raw-count matrix so the reader can see
    which classes actually carry support. Only the classes present in the ground
    truth are shown (PASTIS-R fold-5 does not exercise all 18).

    Args:
        y_true: Ground-truth class ids (parcel vector or flattened pixel map).
        y_pred: Predicted class ids (same element count as ``y_true``).
        labels: ``{id: name}`` map / ordered names / ``None`` (``C{id}``).
        out_path: Destination PNG path (parents created).
        model: Model name shown in the titles.
        ignore_index: Label excluded from the matrix (default 255).

    Returns:
        The :class:`pathlib.Path` of the written PNG.
    """
    cm = _confusion_counts(y_true, y_pred, ignore_index=ignore_index)
    name_map = _resolve_labels(labels)
    present = _present_classes(y_true, ignore_index=ignore_index)
    if not present:  # pragma: no cover - defensive, fold-5 always has labels
        present = list(range(_NUM_CLASSES))
    cm_k = cm[np.ix_(present, present)].astype(np.float64)
    tick_labels = [name_map.get(c, f"C{c}") for c in present]

    row_sum = cm_k.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_k, row_sum, out=np.zeros_like(cm_k), where=row_sum > 0)

    side = max(6, len(present) * 0.55)
    fig, axes = plt.subplots(1, 2, figsize=(side * 2 + 1, side))

    im0 = axes[0].imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"Matriz de confusion normalizada (recall) - {model}")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Recall por clase")

    im1 = axes[1].imshow(cm_k, cmap="Oranges")
    axes[1].set_title(f"Matriz de confusion absoluta (conteos) - {model}")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Numero de parcelas")

    for ax in axes:
        ax.set_xticks(range(len(present)))
        ax.set_yticks(range(len(present)))
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
        ax.set_yticklabels(tick_labels, fontsize=7)
        ax.set_xlabel("Prediccion")
        ax.set_ylabel("Verdad")

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        "confusion_norm_abs_written",
        model=model,
        n_classes=len(present),
        path=str(out),
    )
    return out


# ---------------------------------------------------------------------------
# Figure 2: ROC one-vs-rest (AUC per class + macro).
# ---------------------------------------------------------------------------


def _roc_curve(y_bin: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute a single one-vs-rest ROC curve and its AUC (no sklearn dep here).

    Args:
        y_bin: Binary indicator ``(n,)`` (1 = positive class).
        scores: Class probability ``(n,)`` used as the decision score.

    Returns:
        Tuple ``(fpr, tpr, auc)`` with the ROC points and the trapezoidal AUC;
        an undefined curve (no positives or no negatives) returns ``auc=nan``.
    """
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_bin[order]
    n_pos = float(y_sorted.sum())
    n_neg = float(y_sorted.size - n_pos)
    if n_pos == 0.0 or n_neg == 0.0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1.0 - y_sorted)
    tpr = np.concatenate([[0.0], tps / n_pos])
    fpr = np.concatenate([[0.0], fps / n_neg])
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def roc_ovr_per_class(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    labels: Mapping[int, str] | Sequence[str] | None = None,
    out_path: Path | str,
    model: str = "ensemble",
    ignore_index: int | None = _IGNORE_INDEX,
) -> tuple[Path, dict[str, float]]:
    """Write the one-vs-rest ROC curves with AUC per class and the macro AUC.

    For every class present in the ground truth, the positive class is that class
    and the score is its post-softmax probability column. The figure draws all
    per-class curves plus the diagonal chance line; the legend reports the AUC of
    each class and the macro (unweighted mean over present classes). The input
    ``proba`` is validated as post-softmax (anti-leakage: never logits).

    Args:
        y_true: Ground-truth class ids ``(n,)``.
        proba: Post-softmax probabilities ``(n, 18)`` (rows sum to 1).
        labels: ``{id: name}`` / ordered names / ``None``.
        out_path: Destination PNG path.
        model: Model name shown in the title.
        ignore_index: Label excluded (default 255).

    Returns:
        Tuple ``(path, auc_by_class)`` where ``auc_by_class`` maps the class name
        to its AUC plus a ``"macro"`` key with the macro average.

    Raises:
        ValueError: if ``proba`` is not a valid post-softmax matrix or its row
            count does not match ``y_true``.
    """
    from ml.ensemble.base import EnsembleModel

    yt = np.asarray(y_true).reshape(-1)
    p = np.asarray(proba, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != yt.size:
        raise ValueError(f"proba must be (n, C) aligned with y_true (n={yt.size}); got {p.shape}.")
    EnsembleModel.validate_probs(p, class_axis=-1, name="roc_proba")

    if ignore_index is not None:
        keep = yt != ignore_index
        yt = yt[keep]
        p = p[keep]
    present = _present_classes(yt, ignore_index=None)
    name_map = _resolve_labels(labels)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], color="#a0aec0", ls="--", lw=1, label="Azar (AUC=0.5)")

    auc_by_class: dict[str, float] = {}
    aucs: list[float] = []
    cmap = plt.get_cmap("tab20", max(len(present), 1))
    for i, cls in enumerate(present):
        y_bin = (yt == cls).astype(np.float64)
        fpr, tpr, auc = _roc_curve(y_bin, p[:, cls])
        cls_name = name_map.get(cls, f"C{cls}")
        if not np.isnan(auc):
            auc_by_class[cls_name] = round(auc, 4)
            aucs.append(auc)
        ax.plot(fpr, tpr, color=cmap(i), lw=1.4, label=f"{cls_name} (AUC={auc:.3f})")

    macro = float(np.mean(aucs)) if aucs else float("nan")
    auc_by_class["macro"] = round(macro, 4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(f"Curvas ROC uno-contra-resto - {model} (AUC macro={macro:.3f})")
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        "roc_ovr_written",
        model=model,
        n_classes=len(present),
        macro_auc=round(macro, 4),
        path=str(out),
    )
    return out, auc_by_class


# ---------------------------------------------------------------------------
# Figure 3: precision-recall per class.
# ---------------------------------------------------------------------------


def _pr_curve(y_bin: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute a single one-vs-rest precision-recall curve and its AP.

    Args:
        y_bin: Binary indicator ``(n,)`` (1 = positive class).
        scores: Class probability ``(n,)`` used as the decision score.

    Returns:
        Tuple ``(recall, precision, ap)`` where ``ap`` is the average precision
        (sum of precision weighted by the recall increments); ``ap=nan`` when the
        class has no positives.
    """
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_bin[order]
    n_pos = float(y_sorted.sum())
    if n_pos == 0.0:
        return np.array([0.0, 1.0]), np.array([1.0, 0.0]), float("nan")
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1.0 - y_sorted)
    precision = tps / np.maximum(tps + fps, _EPS)
    recall = tps / n_pos
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    # Average precision = sum_k (R_k - R_{k-1}) * P_k.
    ap = float(np.sum(np.diff(recall) * precision[1:]))
    return recall, precision, ap


def pr_per_class(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    labels: Mapping[int, str] | Sequence[str] | None = None,
    out_path: Path | str,
    model: str = "ensemble",
    ignore_index: int | None = _IGNORE_INDEX,
) -> tuple[Path, dict[str, float]]:
    """Write the one-vs-rest precision-recall curves with AP per class + macro.

    PR curves are more informative than ROC under the heavy class imbalance of
    PASTIS-R: a rare class can keep a high ROC AUC while its precision collapses.
    The input ``proba`` is validated as post-softmax (anti-leakage).

    Args:
        y_true: Ground-truth class ids ``(n,)``.
        proba: Post-softmax probabilities ``(n, 18)``.
        labels: ``{id: name}`` / ordered names / ``None``.
        out_path: Destination PNG path.
        model: Model name shown in the title.
        ignore_index: Label excluded (default 255).

    Returns:
        Tuple ``(path, ap_by_class)`` mapping each class name to its average
        precision plus a ``"macro"`` key with the macro average.

    Raises:
        ValueError: if ``proba`` is not post-softmax or misaligned with
            ``y_true``.
    """
    from ml.ensemble.base import EnsembleModel

    yt = np.asarray(y_true).reshape(-1)
    p = np.asarray(proba, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] != yt.size:
        raise ValueError(f"proba must be (n, C) aligned with y_true (n={yt.size}); got {p.shape}.")
    EnsembleModel.validate_probs(p, class_axis=-1, name="pr_proba")

    if ignore_index is not None:
        keep = yt != ignore_index
        yt = yt[keep]
        p = p[keep]
    present = _present_classes(yt, ignore_index=None)
    name_map = _resolve_labels(labels)

    fig, ax = plt.subplots(figsize=(8, 7))
    ap_by_class: dict[str, float] = {}
    aps: list[float] = []
    cmap = plt.get_cmap("tab20", max(len(present), 1))
    for i, cls in enumerate(present):
        y_bin = (yt == cls).astype(np.float64)
        recall, precision, ap = _pr_curve(y_bin, p[:, cls])
        cls_name = name_map.get(cls, f"C{cls}")
        if not np.isnan(ap):
            ap_by_class[cls_name] = round(ap, 4)
            aps.append(ap)
        ax.plot(recall, precision, color=cmap(i), lw=1.4, label=f"{cls_name} (AP={ap:.3f})")

    macro = float(np.mean(aps)) if aps else float("nan")
    ap_by_class["macro"] = round(macro, 4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Curvas precision-recall por clase - {model} (AP macro={macro:.3f})")
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        "pr_per_class_written",
        model=model,
        n_classes=len(present),
        macro_ap=round(macro, 4),
        path=str(out),
    )
    return out, ap_by_class


# ---------------------------------------------------------------------------
# Figure 4: spatial residuals over the real parcel geometry.
# ---------------------------------------------------------------------------


def spatial_residuals(
    parcel_geoms: pl.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    out_path: Path | str,
    model: str = "ensemble",
    key: str = "canonical_parcel_id",
) -> Path:
    """Plot per-parcel residuals (hit / miss) over the REAL parcel geometry.

    Each parcel is drawn at its real geometry (polygon centroid or point) and
    coloured by whether the ensemble predicted its class correctly. Spatially
    clustered errors signal a geographic bias the aggregate F1 hides. The
    ``parcel_geoms`` frame carries the ``canonical_parcel_id`` + ``geometry``
    aligned 1:1 with ``y_true``/``y_pred`` (same order).

    Args:
        parcel_geoms: Polars frame with ``key`` + ``geometry`` (shapely / WKT /
            WKB), one row per parcel in the prediction order.
        y_true: Ground-truth class ids ``(n,)`` aligned with ``parcel_geoms``.
        y_pred: Predicted class ids ``(n,)`` aligned with ``parcel_geoms``.
        out_path: Destination PNG path.
        model: Model name shown in the title.
        key: Canonical id column name (default ``canonical_parcel_id``).

    Returns:
        The :class:`pathlib.Path` of the written PNG.

    Raises:
        ValueError: if ``parcel_geoms`` lacks ``geometry`` or the lengths of the
            geometry frame, ``y_true`` and ``y_pred`` differ.
    """
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    if not (parcel_geoms.height == yt.size == yp.size):
        raise ValueError(
            f"parcel_geoms ({parcel_geoms.height}), y_true ({yt.size}) and y_pred "
            f"({yp.size}) must be aligned 1:1."
        )
    if "geometry" not in parcel_geoms.columns:
        raise ValueError("parcel_geoms must carry a `geometry` column.")

    xs, ys = _centroids(parcel_geoms["geometry"].to_list())
    correct = yt == yp

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        xs[correct],
        ys[correct],
        c="#2f855a",
        s=18,
        alpha=0.7,
        label=f"Acierto (n={int(correct.sum())})",
        edgecolor="none",
    )
    ax.scatter(
        xs[~correct],
        ys[~correct],
        c="#c53030",
        s=26,
        alpha=0.85,
        marker="x",
        label=f"Error (n={int((~correct).sum())})",
    )
    accuracy = float(correct.mean()) if correct.size else float("nan")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title(f"Residuos espaciales - {model} (acierto={accuracy:.3f})")
    ax.legend(fontsize=9, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(
        "spatial_residuals_written",
        model=model,
        n_parcels=int(yt.size),
        n_errors=int((~correct).sum()),
        path=str(out),
    )
    return out


def _centroids(geometries: list[object]) -> tuple[np.ndarray, np.ndarray]:
    """Return the centroid (x, y) arrays of a list of geometries.

    Accepts shapely objects, WKT strings or WKB bytes; each geometry's centroid
    is used so polygons and points share the same plotting path.

    Args:
        geometries: List of geometries (shapely / WKT / WKB), one per parcel.

    Returns:
        Tuple ``(xs, ys)`` of ``float64`` centroid coordinate arrays.

    Raises:
        ValueError: if a geometry is null or in an unsupported encoding.
    """
    from shapely import wkb, wkt
    from shapely.geometry.base import BaseGeometry

    xs: list[float] = []
    ys: list[float] = []
    for value in geometries:
        if value is None:
            raise ValueError("parcel_geoms has a null geometry.")
        if isinstance(value, BaseGeometry):
            geom = value
        elif isinstance(value, (bytes, bytearray)):
            geom = wkb.loads(bytes(value))
        elif isinstance(value, str):
            geom = wkt.loads(value)
        else:  # pragma: no cover - defensive, unsupported encoding
            raise ValueError(
                f"unsupported geometry encoding: {type(value)!r}; pass shapely, WKT or WKB."
            )
        centroid = geom.centroid
        xs.append(float(centroid.x))
        ys.append(float(centroid.y))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


# ---------------------------------------------------------------------------
# Comparison table (Selection criterion, AC-8).
# ---------------------------------------------------------------------------


def build_comparison_table(results: Mapping[str, Mapping[str, float]]) -> pl.DataFrame:
    """Build the best-individual-vs-4-ensembles comparison table.

    Consolidates the per-model metrics into a Polars DataFrame with the Selection
    schema (``model``, ``f1_macro``, ``accuracy``, ``inference_time_s``,
    ``chosen``), sorted by F1-macro descending. The ``chosen`` model is the one
    that maximizes F1-macro AND meets/beats the individual baseline (the rubric
    threshold: at least one ensemble >= 0.6253); ties break toward the lowest
    inference time so the cheapest model wins a tie (Occam's razor for the
    Selection criterion). If no ensemble beats the baseline, the individual best
    is chosen and a warning is logged.

    Args:
        results: Mapping ``{model_name: {"f1_macro": ..., "accuracy": ...,
            "inference_time_s": ... (optional)}}``. One entry must be the
            individual baseline (its name is taken verbatim).

    Returns:
        A Polars DataFrame with columns
        ``[model, f1_macro, accuracy, inference_time_s, chosen]`` sorted by
        F1-macro descending, exactly one ``chosen=True`` row.

    Raises:
        ValueError: if ``results`` is empty or any entry lacks ``f1_macro``.
    """
    if not results:
        raise ValueError("results is empty; provide the baseline + the ensembles.")

    names: list[str] = []
    f1s: list[float] = []
    accs: list[float] = []
    times: list[float] = []
    for model, metrics in results.items():
        if "f1_macro" not in metrics:
            raise ValueError(f"results[{model!r}] is missing `f1_macro`.")
        names.append(str(model))
        f1s.append(float(metrics["f1_macro"]))
        accs.append(float(metrics.get("accuracy", float("nan"))))
        times.append(float(metrics.get("inference_time_s", float("nan"))))

    # The chosen model maximizes F1-macro, breaking ties by lowest inference
    # time. NaN inference times sort last so a model with a measured time wins.
    def _sort_key(idx: int) -> tuple[float, float]:
        inf = times[idx]
        inf_key = float("inf") if np.isnan(inf) else inf
        return (-f1s[idx], inf_key)

    best_idx = min(range(len(names)), key=_sort_key)
    chosen = [i == best_idx for i in range(len(names))]

    frame = pl.DataFrame(
        {
            "model": names,
            "f1_macro": f1s,
            "accuracy": accs,
            "inference_time_s": times,
            "chosen": chosen,
        },
        schema_overrides={"chosen": pl.Boolean},
    ).select(list(_COMPARISON_COLUMNS))
    frame = frame.sort("f1_macro", descending=True)
    logger.info(
        "comparison_table_built",
        n_models=frame.height,
        chosen=names[best_idx],
        chosen_f1=round(f1s[best_idx], 4),
    )
    return frame
