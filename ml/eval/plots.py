"""Plots de diagnostico para el baseline tabular (Avance 3, fase Evaluation).

Catalogo de visualizaciones interpretables para el modelo de referencia:

- :func:`plot_class_distribution` — barras de soporte por clase (desbalance).
- :func:`plot_confusion_matrix` — heatmap de la matriz de confusion.
- :func:`plot_learning_curve` — error train vs val frente al tamano de muestra.
- :func:`plot_validation_curve` — metrica frente a un hiperparametro.
- :func:`plot_feature_importance` — barras de importancia (RF Gini / XGB gain).
- :func:`plot_shap_summary` — beeswarm SHAP del modelo arbol.
- :func:`plot_shap_dependency` — dependencia SHAP de un feature concreto.

Todas las funciones devuelven una :class:`matplotlib.figure.Figure` lista para
``display(fig)`` en el notebook (el caller hace ``plt.close(fig)`` despues).
Ninguna llama a ``plt.show()``; el backend lo decide el notebook.

Decisiones de diseno
--------------------
- Polars in: las entradas tabulares son :class:`polars.DataFrame`; la
  conversion a numpy / pandas se hace solo en el borde de las librerias que
  lo exigen (matplotlib, shap).
- Sin estado global: cada funcion crea su propia figura.
- Degradacion segura: si una entrada esta vacia se devuelve una figura con
  un mensaje, de modo que el notebook nunca rompe.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import structlog
from matplotlib.figure import Figure

logger = structlog.get_logger(__name__)

__all__ = [
    "plot_class_distribution",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "plot_learning_curve",
    "plot_shap_dependency",
    "plot_shap_summary",
    "plot_validation_curve",
]


def _empty_figure(message: str) -> Figure:
    """Crea una figura placeholder con un mensaje centrado.

    Args:
        message: Texto a mostrar (modo degradado).

    Returns:
        Figura matplotlib con el mensaje.
    """
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11)
    ax.axis("off")
    return fig


def plot_class_distribution(
    y: pl.Series | np.ndarray | Sequence[int],
    *,
    class_names: dict[int, str] | None = None,
    title: str = "Distribucion de clases",
) -> Figure:
    """Grafica el numero de muestras por clase (diagnostico de desbalance).

    Args:
        y: Etiquetas (Polars / numpy / lista).
        class_names: Mapeo opcional ``{class_id: nombre}`` para el eje x.
        title: Titulo del grafico.

    Returns:
        Figura con barras ordenadas de mayor a menor soporte.
    """
    arr = (
        y.to_numpy()
        if isinstance(y, pl.Series)
        else np.asarray(y, dtype=np.int64)
    )
    if arr.size == 0:
        return _empty_figure("Sin etiquetas para graficar la distribucion de clases")

    classes, counts = np.unique(arr, return_counts=True)
    order = np.argsort(-counts)
    classes = classes[order]
    counts = counts[order]
    names = class_names or {}
    labels = [names.get(int(c), str(c)) for c in classes]

    fig, ax = plt.subplots(figsize=(max(7, len(classes) * 0.55), 4.5))
    bars = ax.bar(range(len(classes)), counts, color="steelblue", alpha=0.85)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Numero de muestras")
    ax.set_xlabel("Clase")
    ax.set_title(title)
    imbalance = float(counts.max() / max(counts.min(), 1))
    ax.text(
        0.97,
        0.95,
        f"Ratio max/min = {imbalance:.1f}x",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "gray"},
    )
    for rect, cnt in zip(bars, counts, strict=True):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height(),
            str(int(cnt)),
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Sequence[int | str],
    *,
    normalize: bool = False,
    title: str = "Matriz de confusion",
) -> Figure:
    """Dibuja la matriz de confusion como heatmap anotado.

    Args:
        cm: Matriz cuadrada ``np.ndarray`` (filas = verdad, columnas =
            prediccion).
        labels: Etiquetas de fila/columna en orden.
        normalize: Si ``True``, la anotacion se muestra con 2 decimales
            (asume que ``cm`` ya viene normalizada por fila).
        title: Titulo del grafico.

    Returns:
        Figura con el heatmap.
    """
    cm = np.asarray(cm, dtype=np.float64)
    if cm.size == 0:
        return _empty_figure("Matriz de confusion vacia")

    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(max(5, n * 0.55), max(4, n * 0.5)))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([str(x) for x in labels], rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels([str(x) for x in labels], fontsize=8)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Verdad")
    ax.set_title(title)

    threshold = cm.max() / 2.0 if cm.max() > 0 else 0.5
    fmt = "{:.2f}" if normalize else "{:.0f}"
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                fmt.format(cm[i, j]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_learning_curve(
    train_sizes: Sequence[float] | np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    *,
    metric_name: str = "F1-macro",
    title: str = "Curva de aprendizaje",
) -> Figure:
    """Grafica la curva de aprendizaje (score train vs val por tamano).

    Args:
        train_sizes: Numero absoluto de muestras de train por punto.
        train_scores: Matriz ``(n_points, n_folds)`` de scores en train.
        val_scores: Matriz ``(n_points, n_folds)`` de scores en validacion.
        metric_name: Nombre de la metrica para el eje y.
        title: Titulo del grafico.

    Returns:
        Figura con las dos curvas y su banda de desviacion estandar.
    """
    sizes = np.asarray(train_sizes, dtype=np.float64)
    train_scores = np.asarray(train_scores, dtype=np.float64)
    val_scores = np.asarray(val_scores, dtype=np.float64)
    if sizes.size == 0:
        return _empty_figure("Sin datos para la curva de aprendizaje")

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, train_mean, "o-", color="darkorange", label=f"Train {metric_name}")
    ax.fill_between(
        sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="darkorange"
    )
    ax.plot(sizes, val_mean, "o-", color="steelblue", label=f"Validacion {metric_name}")
    ax.fill_between(
        sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="steelblue"
    )
    ax.set_xlabel("Numero de muestras de entrenamiento")
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_validation_curve(
    param_values: Sequence[float] | np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    *,
    param_name: str = "n_estimators",
    metric_name: str = "F1-macro",
    log_x: bool = False,
    title: str = "Curva de validacion",
) -> Figure:
    """Grafica la curva de validacion (score frente a un hiperparametro).

    Args:
        param_values: Valores del hiperparametro evaluados.
        train_scores: Matriz ``(n_values, n_folds)`` de scores en train.
        val_scores: Matriz ``(n_values, n_folds)`` de scores en validacion.
        param_name: Nombre del hiperparametro para el eje x.
        metric_name: Nombre de la metrica para el eje y.
        log_x: Si ``True``, eje x logaritmico.
        title: Titulo del grafico.

    Returns:
        Figura con las dos curvas y su banda de desviacion estandar.
    """
    values = np.asarray(param_values, dtype=np.float64)
    train_scores = np.asarray(train_scores, dtype=np.float64)
    val_scores = np.asarray(val_scores, dtype=np.float64)
    if values.size == 0:
        return _empty_figure("Sin datos para la curva de validacion")

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(values, train_mean, "o-", color="darkorange", label=f"Train {metric_name}")
    ax.fill_between(
        values, train_mean - train_std, train_mean + train_std, alpha=0.15, color="darkorange"
    )
    ax.plot(values, val_mean, "o-", color="steelblue", label=f"Validacion {metric_name}")
    ax.fill_between(
        values, val_mean - val_std, val_mean + val_std, alpha=0.15, color="steelblue"
    )
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(param_name)
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_feature_importance(
    importance_df: pl.DataFrame,
    *,
    top_k: int = 20,
    feature_col: str = "feature",
    importance_col: str = "importance",
    title: str = "Importancia de caracteristicas",
) -> Figure:
    """Grafica las ``top_k`` features mas importantes en barras horizontales.

    Args:
        importance_df: DataFrame Polars con columnas de feature e importancia
            (formato de :func:`ml.features.selection.compute_feature_importance`).
        top_k: Numero de features a mostrar.
        feature_col: Nombre de la columna de nombres de feature.
        importance_col: Nombre de la columna de importancia.
        title: Titulo del grafico.

    Returns:
        Figura con barras horizontales ordenadas (la mas importante arriba).
    """
    if importance_df.height == 0 or feature_col not in importance_df.columns:
        return _empty_figure("Sin datos de importancia de features")

    top = importance_df.sort(importance_col, descending=True).head(top_k)
    features = top.get_column(feature_col).to_list()
    values = top.get_column(importance_col).to_numpy()

    fig, ax = plt.subplots(figsize=(8, max(4, len(features) * 0.32)))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, values, color="seagreen", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Importancia")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


def plot_shap_summary(
    shap_values: Any,
    features: np.ndarray,
    feature_names: Sequence[str],
    *,
    max_display: int = 15,
    title: str = "SHAP summary",
) -> Figure:
    """Genera el beeswarm SHAP (impacto de cada feature en la prediccion).

    Args:
        shap_values: Salida de ``shap.TreeExplainer.shap_values`` o un objeto
            :class:`shap.Explanation`. Para multiclase puede ser una lista
            de matrices; se promedia el ``abs`` sobre clases.
        features: Matriz ``(n_samples, n_features)`` de valores de feature.
        feature_names: Nombres de las features en orden de columna.
        max_display: Numero maximo de features a mostrar.
        title: Titulo del grafico.

    Returns:
        Figura con el beeswarm SHAP.
    """
    import shap  # type: ignore[import-untyped]

    matrix = np.asarray(features, dtype=np.float64)
    if matrix.size == 0:
        return _empty_figure("Sin datos para el SHAP summary")

    values = _reduce_shap_values(shap_values)
    fig = plt.figure(figsize=(8, max(4, min(max_display, len(feature_names)) * 0.4)))
    try:
        shap.summary_plot(
            values,
            matrix,
            feature_names=list(feature_names),
            max_display=max_display,
            show=False,
            plot_size=None,
        )
    except Exception as exc:  # noqa: BLE001
        plt.close(fig)
        logger.warning("shap_summary_failed", error=str(exc))
        return _empty_figure(f"SHAP summary no disponible: {exc}")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_shap_dependency(
    feature_name: str,
    shap_values: Any,
    features: np.ndarray,
    feature_names: Sequence[str],
    *,
    title: str | None = None,
) -> Figure:
    """Genera el grafico de dependencia SHAP de un feature concreto.

    Args:
        feature_name: Nombre del feature a graficar.
        shap_values: Salida de ``shap.TreeExplainer`` (matriz o lista).
        features: Matriz ``(n_samples, n_features)`` de valores de feature.
        feature_names: Nombres de las features en orden de columna.
        title: Titulo del grafico. Si ``None`` se construye automaticamente.

    Returns:
        Figura con el scatter de dependencia SHAP.
    """
    import shap  # type: ignore[import-untyped]

    matrix = np.asarray(features, dtype=np.float64)
    names = list(feature_names)
    if matrix.size == 0 or feature_name not in names:
        return _empty_figure(f"Feature {feature_name!r} no disponible para dependencia SHAP")

    values = _reduce_shap_values(shap_values)
    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        shap.dependence_plot(
            feature_name,
            values,
            matrix,
            feature_names=names,
            interaction_index=None,
            ax=ax,
            show=False,
        )
    except Exception as exc:  # noqa: BLE001
        plt.close(fig)
        logger.warning("shap_dependency_failed", feature=feature_name, error=str(exc))
        return _empty_figure(f"Dependencia SHAP no disponible: {exc}")
    ax.set_title(title or f"Dependencia SHAP - {feature_name}")
    fig.tight_layout()
    return fig


def _reduce_shap_values(shap_values: Any) -> np.ndarray:
    """Reduce la salida SHAP a una matriz 2D ``(n_samples, n_features)``.

    SHAP devuelve listas (una matriz por clase) para clasificacion
    multiclase, o tensores ``(n_samples, n_features, n_classes)`` en
    versiones recientes. Para los plots agregados promediamos el valor
    absoluto sobre el eje de clases.

    Args:
        shap_values: Salida cruda de un explainer SHAP.

    Returns:
        Matriz ``np.ndarray`` 2D.
    """
    if isinstance(shap_values, list):
        stacked = np.stack([np.abs(np.asarray(v)) for v in shap_values], axis=0)
        return np.asarray(stacked.mean(axis=0), dtype=np.float64)
    arr = np.asarray(getattr(shap_values, "values", shap_values), dtype=np.float64)
    if arr.ndim == 3:
        return np.asarray(np.abs(arr).mean(axis=2), dtype=np.float64)
    return arr
