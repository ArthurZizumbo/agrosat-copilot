"""Tests de los plots de diagnostico ``ml.eval.plots`` (Avance 3).

Las funciones de plot se validan por contrato (devuelven una
:class:`matplotlib.figure.Figure` con ejes) y por degradacion segura ante
entradas vacias; no se compara pixel a pixel.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure

from ml.eval.plots import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_learning_curve,
    plot_validation_curve,
)


def test_class_distribution_returns_figure() -> None:
    """``plot_class_distribution`` devuelve una figura con al menos un eje."""
    y = [0, 0, 0, 1, 1, 2]
    fig = plot_class_distribution(y, class_names={0: "a", 1: "b", 2: "c"})
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_class_distribution_empty_is_safe() -> None:
    """Sin etiquetas, devuelve una figura placeholder sin lanzar."""
    fig = plot_class_distribution([])
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_confusion_matrix_plot_returns_figure() -> None:
    """``plot_confusion_matrix`` rinde un heatmap como figura."""
    cm = np.array([[5.0, 1.0], [2.0, 4.0]])
    fig = plot_confusion_matrix(cm, [0, 1])
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_confusion_matrix_empty_is_safe() -> None:
    """Una matriz vacia produce una figura placeholder."""
    fig = plot_confusion_matrix(np.zeros((0, 0)), [])
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_learning_curve_returns_figure() -> None:
    """``plot_learning_curve`` traza train vs val frente al tamano."""
    sizes = [10, 20, 30]
    train = np.array([[0.9, 0.92], [0.91, 0.93], [0.92, 0.94]])
    val = np.array([[0.6, 0.62], [0.65, 0.66], [0.7, 0.71]])
    fig = plot_learning_curve(sizes, train, val)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_validation_curve_returns_figure() -> None:
    """``plot_validation_curve`` traza score frente a un hiperparametro."""
    params = [10, 50, 100]
    train = np.array([[0.8, 0.82], [0.9, 0.91], [0.95, 0.96]])
    val = np.array([[0.55, 0.57], [0.62, 0.63], [0.64, 0.65]])
    fig = plot_validation_curve(params, train, val, param_name="n_estimators")
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_feature_importance_plot_returns_figure() -> None:
    """``plot_feature_importance`` rinde barras horizontales como figura."""
    imp = pl.DataFrame(
        {
            "feature": ["dim_00", "dim_01", "dim_02"],
            "importance": [0.5, 0.3, 0.2],
            "rank": [1, 2, 3],
        }
    )
    fig = plot_feature_importance(imp, top_k=3)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_feature_importance_empty_is_safe() -> None:
    """Sin datos de importancia, devuelve una figura placeholder."""
    empty = pl.DataFrame({"feature": [], "importance": []})
    fig = plot_feature_importance(empty)
    assert isinstance(fig, Figure)
    plt.close(fig)
