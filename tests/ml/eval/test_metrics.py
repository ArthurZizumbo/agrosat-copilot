"""Tests de la suite de metricas ``ml.eval.metrics`` (Avance 3)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ml.eval.metrics import (
    classification_report_df,
    compute_classification_metrics,
    confusion_matrix_df,
    summarize_cv_metrics,
)


def test_perfect_prediction_gives_unit_metrics() -> None:
    """Una prediccion perfecta produce F1-macro y accuracy iguales a 1.0."""
    y = [0, 1, 2, 0, 1, 2]
    metrics = compute_classification_metrics(y, y)
    assert metrics["f1_macro"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["cohen_kappa"] == pytest.approx(1.0)
    assert metrics["miou"] == pytest.approx(1.0)
    assert metrics["n_classes"] == 3.0


def test_metrics_accept_polars_series() -> None:
    """Las metricas aceptan ``pl.Series`` sin conversion manual."""
    y_true = pl.Series("y", [1, 1, 2, 2])
    y_pred = pl.Series("p", [1, 2, 2, 2])
    metrics = compute_classification_metrics(y_true, y_pred)
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert metrics["n_samples"] == 4.0


def test_empty_input_returns_zero_metrics() -> None:
    """Una entrada vacia devuelve todas las metricas en 0.0 sin lanzar."""
    metrics = compute_classification_metrics([], [])
    assert metrics["f1_macro"] == 0.0
    assert metrics["n_samples"] == 0.0


def test_mismatched_shapes_raise() -> None:
    """``y_true`` y ``y_pred`` de distinto largo lanzan ``ValueError``."""
    with pytest.raises(ValueError, match="mismo shape"):
        compute_classification_metrics([0, 1, 2], [0, 1])


def test_f1_macro_penalizes_minority_class_failure() -> None:
    """El F1-macro cae cuando se falla por completo la clase minoritaria."""
    # 8 muestras clase 0 (todas acertadas) + 2 clase 1 (todas falladas).
    y_true = [0] * 8 + [1] * 2
    y_pred = [0] * 10
    metrics = compute_classification_metrics(y_true, y_pred)
    # accuracy alta (0.8) pero F1-macro bajo: clase 1 contribuye 0.
    assert metrics["accuracy"] == pytest.approx(0.8)
    assert metrics["f1_macro"] < 0.5


def test_classification_report_has_summary_rows() -> None:
    """El report incluye filas resumen macro avg (-1) y weighted avg (-2)."""
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 2, 2]
    report = classification_report_df(y_true, y_pred, class_names={0: "a", 1: "b", 2: "c"})
    assert report.height == 5  # 3 clases + 2 resumenes
    summary_ids = set(report.get_column("class_id").to_list())
    assert -1 in summary_ids
    assert -2 in summary_ids
    macro = report.filter(pl.col("class_id") == -1)
    assert macro.get_column("class_name").item() == "macro avg"


def test_confusion_matrix_shape_and_diagonal() -> None:
    """La matriz de confusion es cuadrada y la diagonal cuenta los aciertos."""
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 1, 2, 2]
    cm_df, cm_matrix, labels = confusion_matrix_df(y_true, y_pred)
    assert cm_matrix.shape == (3, 3)
    assert labels == [0, 1, 2]
    # Prediccion perfecta: solo la diagonal tiene masa.
    assert np.array_equal(cm_matrix, np.diag([2, 2, 2]).astype(float))
    assert "true_class" in cm_df.columns


def test_confusion_matrix_normalize_rows_sum_to_one() -> None:
    """Con ``normalize=True`` las filas de la matriz suman 1 (recall)."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    _, cm_matrix, _ = confusion_matrix_df(y_true, y_pred, normalize=True)
    row_sums = cm_matrix.sum(axis=1)
    assert np.allclose(row_sums, 1.0)


def test_summarize_cv_metrics_aggregates_mean_std() -> None:
    """El resumen de CV calcula media y desviacion por metrica."""
    fold_metrics = [
        {"f1_macro": 0.6, "accuracy": 0.7},
        {"f1_macro": 0.8, "accuracy": 0.9},
    ]
    summary = summarize_cv_metrics(fold_metrics)
    f1_row = summary.filter(pl.col("metric") == "f1_macro")
    assert f1_row.get_column("mean").item() == pytest.approx(0.7)
    assert f1_row.get_column("n_folds").item() == 2


def test_summarize_cv_metrics_empty_returns_empty_frame() -> None:
    """Sin folds, el resumen devuelve un DataFrame vacio con esquema valido."""
    summary = summarize_cv_metrics([])
    assert summary.height == 0
    assert "metric" in summary.columns
