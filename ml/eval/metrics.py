"""Suite de metricas de evaluacion para el baseline tabular (Avance 3).

Modulo canonico de la fase **Evaluation** del CRISP-ML(Q) para el modelo de
referencia (EPIC 4 / US-019..US-022). Expone una API Polars-friendly que
calcula las metricas de clasificacion multiclase usadas en todo el proyecto:

- **F1-macro** (metrica principal — penaliza el desbalance de clases).
- **F1-weighted** (ponderada por soporte, lectura optimista).
- **accuracy** (fraccion de aciertos, sensible al desbalance).
- **Cohen kappa** (acuerdo corregido por azar).
- **mIoU** (Jaccard macro — comparable con la segmentacion densa de EPIC 5).
- **classification report** por clase (precision/recall/F1/support).
- **matriz de confusion** como :class:`polars.DataFrame` legible.

Decisiones de diseno
--------------------
- Polars in / Polars out: las entradas pueden ser :class:`polars.Series`,
  ``np.ndarray`` o ``list``; la conversion a numpy se hace en
  :func:`_as_int_array` (regla ``ml/CLAUDE.md NEVER pandas``).
- F1-macro es la metrica principal: con clases agricolas desbalanceadas
  (Meadow 31k vs Beet 871 en PASTIS-R) la accuracy enmascara el fallo en
  clases minoritarias.
- ``zero_division=0`` en todas las metricas: una clase ausente en las
  predicciones contribuye 0 (no NaN), de modo que el promedio macro es
  estable y comparable entre folds.

Referencias
-----------
- Cohen, J. (1960). *A coefficient of agreement for nominal scales*.
  Educational and Psychological Measurement 20(1), 37-46.
- Sokolova, M., Lapalme, G. (2009). *A systematic analysis of performance
  measures for classification tasks*. Information Processing & Management
  45(4), 427-437 — justifica F1-macro en datos desbalanceados.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
import structlog
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    precision_recall_fscore_support,
)

logger = structlog.get_logger(__name__)

__all__ = [
    "classification_report_df",
    "compute_classification_metrics",
    "confusion_matrix_df",
    "summarize_cv_metrics",
]

# Tipo de entrada aceptado para etiquetas: Polars, numpy o lista de enteros.
LabelInput = "pl.Series | np.ndarray | Sequence[int]"


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------


def _as_int_array(values: pl.Series | np.ndarray | Sequence[int]) -> np.ndarray:
    """Convierte etiquetas (Polars / numpy / lista) a ``np.ndarray`` int64.

    Args:
        values: Etiquetas como :class:`polars.Series`, ``np.ndarray`` o
            secuencia de enteros.

    Returns:
        Vector ``np.ndarray`` shape ``(n,)`` de dtype int64.
    """
    if isinstance(values, pl.Series):
        return values.to_numpy().astype(np.int64, copy=False)
    return np.asarray(values, dtype=np.int64)


def _resolved_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int] | None,
) -> list[int]:
    """Resuelve el conjunto ordenado de etiquetas para metricas consistentes.

    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Etiquetas predichas.
        labels: Lista explicita de etiquetas. Si ``None`` se infiere de la
            union de ``y_true`` y ``y_pred``.

    Returns:
        Lista ordenada de etiquetas enteras.
    """
    if labels is not None:
        return [int(x) for x in labels]
    union = np.unique(np.concatenate([y_true, y_pred]))
    return [int(x) for x in union]


# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------


def compute_classification_metrics(
    y_true: pl.Series | np.ndarray | Sequence[int],
    y_pred: pl.Series | np.ndarray | Sequence[int],
    *,
    labels: Sequence[int] | None = None,
) -> dict[str, float]:
    """Calcula la suite de metricas escalares de clasificacion multiclase.

    Args:
        y_true: Etiquetas verdaderas (Polars / numpy / lista).
        y_pred: Etiquetas predichas (mismo largo que ``y_true``).
        labels: Conjunto explicito de etiquetas a considerar. Si ``None`` se
            infiere de la union observada (garantiza promedios estables).

    Returns:
        Diccionario con keys ``f1_macro`` (principal), ``f1_weighted``,
        ``accuracy``, ``cohen_kappa``, ``miou``, ``n_samples``, ``n_classes``.
        Si ``y_true`` esta vacio, todas las metricas son ``0.0``.

    Raises:
        ValueError: Si ``y_true`` y ``y_pred`` tienen distinto largo.
    """
    y_t = _as_int_array(y_true)
    y_p = _as_int_array(y_pred)
    if y_t.shape != y_p.shape:
        raise ValueError(
            f"y_true ({y_t.shape}) y y_pred ({y_p.shape}) deben tener el mismo shape."
        )
    if y_t.size == 0:
        return {
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "accuracy": 0.0,
            "cohen_kappa": 0.0,
            "miou": 0.0,
            "n_samples": 0.0,
            "n_classes": 0.0,
        }

    label_list = _resolved_labels(y_t, y_p, labels)
    metrics = {
        "f1_macro": float(
            f1_score(y_t, y_p, labels=label_list, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_t, y_p, labels=label_list, average="weighted", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_t, y_p)),
        "cohen_kappa": float(cohen_kappa_score(y_t, y_p)),
        "miou": float(
            jaccard_score(y_t, y_p, labels=label_list, average="macro", zero_division=0)
        ),
        "n_samples": float(y_t.size),
        "n_classes": float(len(label_list)),
    }
    logger.info(
        "classification_metrics_computed",
        f1_macro=round(metrics["f1_macro"], 4),
        accuracy=round(metrics["accuracy"], 4),
        n_samples=int(metrics["n_samples"]),
        n_classes=int(metrics["n_classes"]),
    )
    return metrics


def classification_report_df(
    y_true: pl.Series | np.ndarray | Sequence[int],
    y_pred: pl.Series | np.ndarray | Sequence[int],
    *,
    labels: Sequence[int] | None = None,
    class_names: dict[int, str] | None = None,
) -> pl.DataFrame:
    """Genera el classification report como :class:`polars.DataFrame`.

    Equivalente Polars de :func:`sklearn.metrics.classification_report`: una
    fila por clase con precision, recall, F1 y support, mas las filas de
    resumen ``macro avg`` y ``weighted avg``.

    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Etiquetas predichas.
        labels: Conjunto explicito de etiquetas. Si ``None`` se infiere.
        class_names: Mapeo opcional ``{class_id: nombre}`` para la columna
            ``class_name`` (default ``str(class_id)``).

    Returns:
        DataFrame Polars con columnas ``class_id, class_name, precision,
        recall, f1, support``. Las filas resumen tienen ``class_id = -1``
        (macro) y ``class_id = -2`` (weighted).
    """
    schema: dict[str, Any] = {
        "class_id": pl.Int64,
        "class_name": pl.Utf8,
        "precision": pl.Float64,
        "recall": pl.Float64,
        "f1": pl.Float64,
        "support": pl.Int64,
    }
    y_t = _as_int_array(y_true)
    y_p = _as_int_array(y_pred)
    if y_t.size == 0:
        return pl.DataFrame(schema=schema)

    label_list = _resolved_labels(y_t, y_p, labels)
    names = class_names or {}

    precision, recall, f1, support = precision_recall_fscore_support(
        y_t, y_p, labels=label_list, average=None, zero_division=0
    )
    rows: list[dict[str, Any]] = []
    for idx, cls in enumerate(label_list):
        rows.append(
            {
                "class_id": int(cls),
                "class_name": names.get(int(cls), str(cls)),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
        )

    macro = precision_recall_fscore_support(
        y_t, y_p, labels=label_list, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_t, y_p, labels=label_list, average="weighted", zero_division=0
    )
    total_support = int(support.sum())
    rows.append(
        {
            "class_id": -1,
            "class_name": "macro avg",
            "precision": float(macro[0]),
            "recall": float(macro[1]),
            "f1": float(macro[2]),
            "support": total_support,
        }
    )
    rows.append(
        {
            "class_id": -2,
            "class_name": "weighted avg",
            "precision": float(weighted[0]),
            "recall": float(weighted[1]),
            "f1": float(weighted[2]),
            "support": total_support,
        }
    )
    return pl.DataFrame(rows, schema=schema)


def confusion_matrix_df(
    y_true: pl.Series | np.ndarray | Sequence[int],
    y_pred: pl.Series | np.ndarray | Sequence[int],
    *,
    labels: Sequence[int] | None = None,
    normalize: bool = False,
) -> tuple[pl.DataFrame, np.ndarray, list[int]]:
    """Calcula la matriz de confusion como DataFrame Polars + matriz numpy.

    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Etiquetas predichas.
        labels: Conjunto explicito de etiquetas. Si ``None`` se infiere.
        normalize: Si ``True`` normaliza por fila (recall por clase).

    Returns:
        Tupla ``(cm_df, cm_matrix, label_list)`` donde ``cm_df`` tiene una
        columna ``true_class`` mas una columna por clase predicha,
        ``cm_matrix`` es la matriz ``np.ndarray`` cuadrada y ``label_list``
        las etiquetas en orden de fila/columna.
    """
    y_t = _as_int_array(y_true)
    y_p = _as_int_array(y_pred)
    if y_t.size == 0:
        return pl.DataFrame({"true_class": []}), np.zeros((0, 0)), []

    label_list = _resolved_labels(y_t, y_p, labels)
    cm = confusion_matrix(y_t, y_p, labels=label_list).astype(np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    data: dict[str, list[Any]] = {"true_class": [str(c) for c in label_list]}
    for col_idx, cls in enumerate(label_list):
        data[str(cls)] = cm[:, col_idx].tolist()
    cm_df = pl.DataFrame(data)
    return cm_df, cm, label_list


def summarize_cv_metrics(fold_metrics: Sequence[dict[str, float]]) -> pl.DataFrame:
    """Resume metricas por fold en media +/- desviacion estandar.

    Args:
        fold_metrics: Lista de diccionarios devueltos por
            :func:`compute_classification_metrics`, uno por fold.

    Returns:
        DataFrame Polars con columnas ``metric, mean, std, min, max,
        n_folds``. Vacio si ``fold_metrics`` esta vacio.
    """
    schema: dict[str, Any] = {
        "metric": pl.Utf8,
        "mean": pl.Float64,
        "std": pl.Float64,
        "min": pl.Float64,
        "max": pl.Float64,
        "n_folds": pl.Int64,
    }
    if not fold_metrics:
        return pl.DataFrame(schema=schema)

    metric_keys = sorted({k for fm in fold_metrics for k in fm})
    rows: list[dict[str, Any]] = []
    for key in metric_keys:
        values = np.asarray(
            [float(fm.get(key, np.nan)) for fm in fold_metrics], dtype=np.float64
        )
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            rows.append(
                {
                    "metric": key,
                    "mean": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                    "n_folds": 0,
                }
            )
            continue
        rows.append(
            {
                "metric": key,
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "n_folds": int(finite.size),
            }
        )
    return pl.DataFrame(rows, schema=schema)
