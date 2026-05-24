"""Plots reutilizables para el notebook 05 del reencuadre fenologico.

Funciones de visualizacion para los analisis de US-022b-C/D. Cada funcion
devuelve un ``matplotlib.figure.Figure`` para que el notebook decida si lo
muestra con ``display(fig)`` y/o lo persiste con ``fig.savefig(...)``.

Patron canonico (consistente con :mod:`ml.eval.learning_curves` y
:mod:`ml.eval.interpretability`):

- Acepta inputs tipados (``FeatureAblationResult`` / ``TemporalModelResult`` /
  ndarrays / DataFrames Polars), nunca paths.
- Devuelve la figura, nunca la persiste ni la cierra.
- Sin ``plt.show()`` ni side-effects globales (matplotlib backend lo configura
  el caller).

Plots provistos:

- :func:`plot_ablation_bars` — F1-macro por conjunto de features.
- :func:`plot_model_comparison_bars` — F1-macro de varios modelos vs baseline.
- :func:`plot_class_support_bars` — distribucion de clases con umbral.
- :func:`plot_per_class_f1` — F1 por clase del mejor modelo (highlight clases debiles).
- :func:`plot_umap_clusters` — UMAP 2D coloreado por cluster KMeans.
- :func:`plot_cluster_ndvi_curves` — curva NDVI media por cluster.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

if TYPE_CHECKING:
    from ml.eval.feature_ablation import FeatureAblationResult


__all__ = [
    "plot_ablation_bars",
    "plot_class_support_bars",
    "plot_cluster_ndvi_curves",
    "plot_model_comparison_bars",
    "plot_per_class_f1",
    "plot_umap_clusters",
]


# ---------------------------------------------------------------------------
# 1. Ablation de features (F1-macro por conjunto, mismo modelo).
# ---------------------------------------------------------------------------


def plot_ablation_bars(
    results: Sequence[FeatureAblationResult],
    *,
    metric: str = "f1_macro",
    title: str | None = None,
    baseline_value: float | None = None,
    figsize: tuple[float, float] = (8.0, 4.0),
) -> matplotlib.figure.Figure:
    """Bar plot horizontal de F1-macro por conjunto de features.

    Cada barra es un par ``(feature_set, model)``. Si hay varios modelos en
    ``results``, los grupos por modelo se separan con colores distintos.

    Args:
        results: Resultados de :func:`run_feature_ablation`.
        metric: ``"f1_macro"``, ``"f1_weighted"`` o ``"miou"``.
        title: Titulo opcional; si ``None`` se genera automaticamente.
        baseline_value: Linea vertical de referencia (ej. F1-macro del
            baseline tabular cerrado). ``None`` la omite.
        figsize: Tupla ``(ancho, alto)`` en pulgadas.

    Returns:
        Figura matplotlib lista para ``display(fig)`` o ``fig.savefig(...)``.
    """
    if not results:
        raise ValueError("`results` esta vacio.")
    if metric not in {"f1_macro", "f1_weighted", "miou"}:
        raise ValueError(f"metric={metric!r} no soportada.")

    # Agrupa por modelo manteniendo el orden estable de aparicion.
    by_model: dict[str, list[tuple[str, float]]] = {}
    for r in results:
        by_model.setdefault(r.model_kind, []).append(
            (r.feature_set, float(getattr(r, metric)))
        )

    fig, ax = plt.subplots(figsize=figsize, dpi=110)

    if len(by_model) == 1:
        # Un solo modelo: barras horizontales simples.
        model_kind, items = next(iter(by_model.items()))
        # Filtra NaN (entrenamientos que fallaron por cobertura nula del set).
        items_clean = [(fs, v) for fs, v in items if v == v]
        items_sorted = sorted(items_clean, key=lambda kv: kv[1], reverse=False)
        labels = [s for s, _ in items_sorted]
        values = [v for _, v in items_sorted]
        if not labels:
            ax.text(
                0.5,
                0.5,
                "Sin metricas validas para graficar.",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="#888",
            )
            ax.set_axis_off()
            fig.tight_layout()
            return fig
        bars = ax.barh(labels, values, color="#4C72B0", edgecolor="white")
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                value + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha="left",
                fontsize=9,
            )
        ax.set_xlabel(metric.replace("_", "-"))
        ax.set_title(
            title or f"{metric.replace('_', '-')} por conjunto de features ({model_kind})"
        )
    else:
        # Varios modelos: barras agrupadas verticalmente.
        feature_sets: list[str] = []
        for items in by_model.values():
            for fs, _ in items:
                if fs not in feature_sets:
                    feature_sets.append(fs)
        n_models = len(by_model)
        width = 0.8 / n_models
        x_positions = np.arange(len(feature_sets))
        palette = plt.get_cmap("tab10").colors  # type: ignore[attr-defined]
        for idx, (model_kind, items) in enumerate(by_model.items()):
            lookup = dict(items)
            values = [lookup.get(fs, np.nan) for fs in feature_sets]
            offsets = x_positions + (idx - (n_models - 1) / 2) * width
            ax.bar(
                offsets,
                values,
                width=width,
                label=model_kind,
                color=palette[idx % len(palette)],
                edgecolor="white",
            )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(feature_sets, rotation=20, ha="right")
        ax.set_ylabel(metric.replace("_", "-"))
        ax.set_title(title or f"{metric.replace('_', '-')} por conjunto x modelo")
        ax.legend(loc="best", frameon=False)

    if baseline_value is not None:
        ax.axvline(baseline_value, color="#888", linestyle="--", linewidth=1)
        ax.text(
            baseline_value,
            ax.get_ylim()[1] * 0.95 if len(by_model) == 1 else 0,
            f"  baseline {baseline_value:.3f}",
            color="#666",
            fontsize=8,
            rotation=90 if len(by_model) > 1 else 0,
            va="top",
            ha="left",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Comparativa de modelos (XGBoost vs TempCNN vs InceptionTime sobre el
#    mismo conjunto de features).
# ---------------------------------------------------------------------------


def plot_model_comparison_bars(
    metric_by_model: Mapping[str, float],
    *,
    baseline_label: str = "baseline 0.32",
    baseline_value: float = 0.32,
    title: str = "Comparativa de modelos sobre el conjunto ganador",
    metric_name: str = "F1-macro",
    figsize: tuple[float, float] = (6.5, 4.0),
) -> matplotlib.figure.Figure:
    """Bar plot vertical comparando varios modelos contra una linea baseline.

    Args:
        metric_by_model: Mapping ``{nombre_modelo: F1_macro}``. Tipico:
            ``{"xgboost": 0.34, "tempcnn": 0.41, "inceptiontime": 0.39}``.
        baseline_label: Etiqueta de la linea baseline en la leyenda.
        baseline_value: Valor de referencia (linea horizontal).
        title: Titulo del plot.
        metric_name: Nombre legible de la metrica.
        figsize: Tupla ``(ancho, alto)``.

    Returns:
        Figura matplotlib.
    """
    if not metric_by_model:
        raise ValueError("`metric_by_model` esta vacio.")

    items = list(metric_by_model.items())
    items_sorted = sorted(items, key=lambda kv: kv[1], reverse=True)
    labels = [k for k, _ in items_sorted]
    values = [v for _, v in items_sorted]

    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    colors = ["#55A868" if v >= baseline_value else "#C44E52" for v in values]
    bars = ax.bar(labels, values, color=colors, edgecolor="white")
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.005,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.axhline(
        baseline_value,
        color="#444",
        linestyle="--",
        linewidth=1.2,
        label=baseline_label,
    )
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.set_ylim(0, max(max(values), baseline_value) * 1.20)
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Soporte por clase (desbalance ~31x).
# ---------------------------------------------------------------------------


def plot_class_support_bars(
    class_counts: pl.DataFrame,
    *,
    class_col: str = "class_id",
    count_col: str = "len",
    weak_threshold: int = 1000,
    title: str = "Numero de parcelas por clase (resaltadas clases con soporte debil)",
    figsize: tuple[float, float] = (8.0, 4.5),
) -> matplotlib.figure.Figure:
    """Bar plot horizontal con soporte por clase.

    Las clases con soporte ``< weak_threshold`` se colorean diferente para
    resaltar el desbalance.

    Args:
        class_counts: DataFrame con columnas ``class_col`` y ``count_col``,
            tipico output de ``df.group_by("class_id").len()``.
        class_col: Nombre de la columna con el id de clase.
        count_col: Nombre de la columna con el conteo.
        weak_threshold: Umbral debajo del cual la barra se marca como debil.
        title: Titulo.
        figsize: Tupla ``(ancho, alto)``.

    Returns:
        Figura matplotlib.
    """
    if class_col not in class_counts.columns or count_col not in class_counts.columns:
        raise ValueError(
            f"`class_counts` debe contener `{class_col}` y `{count_col}`."
        )

    ordered = class_counts.sort(count_col, descending=False)
    labels = [str(v) for v in ordered[class_col].to_list()]
    values = ordered[count_col].to_list()
    colors = ["#C44E52" if v < weak_threshold else "#4C72B0" for v in values]

    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    ax.barh(labels, values, color=colors, edgecolor="white")
    for idx, value in enumerate(values):
        ax.text(value, idx, f" {value:,}", va="center", ha="left", fontsize=8)
    ax.set_xlabel("Numero de parcelas")
    ax.set_ylabel(class_col)
    ax.set_title(title)
    ax.set_xscale("log")
    ax.axvline(weak_threshold, color="#888", linestyle="--", linewidth=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. F1 por clase del mejor modelo (highlight de clases debiles).
# ---------------------------------------------------------------------------


def plot_per_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_labels: Sequence[int] | None = None,
    class_names: Mapping[int, str] | None = None,
    title: str = "F1 por clase",
    weak_threshold: float = 0.10,
    figsize: tuple[float, float] = (8.0, 4.0),
) -> matplotlib.figure.Figure:
    """Bar plot horizontal de F1 por clase con highlight de clases debiles.

    Args:
        y_true: Etiquetas verdaderas (1D).
        y_pred: Etiquetas predichas (1D).
        class_labels: Lista de ids de clase a reportar (orden de la grafica).
            ``None`` infiere de la union de ``y_true`` y ``y_pred``.
        class_names: Mapping opcional ``{class_id: nombre legible}``.
        title: Titulo del plot.
        weak_threshold: F1 debajo del cual la barra se colorea como debil.
        figsize: Tupla ``(ancho, alto)``.

    Returns:
        Figura matplotlib.
    """
    from sklearn.metrics import f1_score

    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()
    if y_true_arr.size == 0 or y_pred_arr.size == 0:
        raise ValueError("`y_true` y `y_pred` no pueden estar vacios.")
    if y_true_arr.size != y_pred_arr.size:
        raise ValueError(
            f"Shape mismatch: y_true.shape={y_true_arr.shape} vs y_pred.shape={y_pred_arr.shape}."
        )

    if class_labels is None:
        class_labels = sorted({*y_true_arr.tolist(), *y_pred_arr.tolist()})
    labels_list = list(class_labels)
    per_class = f1_score(
        y_true_arr,
        y_pred_arr,
        labels=labels_list,
        average=None,
        zero_division=0,
    )
    ordered = sorted(zip(labels_list, per_class, strict=True), key=lambda kv: kv[1])
    y_labels = [
        (class_names.get(cid, str(cid)) if class_names else str(cid))
        for cid, _ in ordered
    ]
    values = [float(v) for _, v in ordered]
    colors = ["#C44E52" if v < weak_threshold else "#55A868" for v in values]

    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    ax.barh(y_labels, values, color=colors, edgecolor="white")
    for idx, value in enumerate(values):
        ax.text(value + 0.01, idx, f"{value:.2f}", va="center", ha="left", fontsize=8)
    ax.axvline(weak_threshold, color="#888", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("F1")
    ax.set_ylabel("clase")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. UMAP 2D coloreado por cluster KMeans (no por class_id, para validar
#    estructura sin coordenadas).
# ---------------------------------------------------------------------------


def plot_umap_clusters(
    embedding: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    title: str = "UMAP de la firma fenologica sin coordenadas, coloreado por cluster",
    figsize: tuple[float, float] = (7.0, 5.0),
) -> matplotlib.figure.Figure:
    """Scatter UMAP 2D coloreado por cluster KMeans.

    Args:
        embedding: Array ``(N, 2)`` con la proyeccion UMAP.
        cluster_labels: Array ``(N,)`` con la asignacion KMeans.
        title: Titulo.
        figsize: Tupla ``(ancho, alto)``.

    Returns:
        Figura matplotlib.
    """
    emb = np.asarray(embedding)
    labels = np.asarray(cluster_labels).ravel()
    if emb.ndim != 2 or emb.shape[1] != 2:
        raise ValueError(f"`embedding` debe ser (N, 2), got {emb.shape}.")
    if emb.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Shape mismatch: embedding={emb.shape}, labels={labels.shape}."
        )

    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    n_clusters = int(labels.max()) + 1 if labels.size > 0 else 1
    palette = plt.get_cmap("tab10").colors  # type: ignore[attr-defined]
    for cid in range(n_clusters):
        mask = labels == cid
        if not np.any(mask):
            continue
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=8,
            alpha=0.6,
            color=palette[cid % len(palette)],
            label=f"cluster {cid}",
            edgecolors="none",
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Curva NDVI sintetica media por cluster (interpretabilidad agronomica).
# ---------------------------------------------------------------------------


def plot_cluster_ndvi_curves(
    df: pl.DataFrame,
    cluster_labels: np.ndarray,
    *,
    fft_cols: Sequence[str] | None = None,
    sequence_length: int = 72,
    title: str = "Curva NDVI media reconstruida por cluster",
    figsize: tuple[float, float] = (8.0, 4.0),
) -> matplotlib.figure.Figure:
    """Reconstruye la curva NDVI media por cluster a partir de las FFT cols.

    Para cada cluster KMeans, promedia los coeficientes FFT NDVI de las
    parcelas del cluster, reconstruye la serie diaria por IDFT parcial y
    grafica la curva sobre el dia del anio.

    Args:
        df: DataFrame Polars con las columnas FFT NDVI presentes.
        cluster_labels: Array ``(N,)`` con la asignacion KMeans (mismo orden
            que ``df``).
        fft_cols: Lista de columnas FFT a reconstruir; ``None`` autodetecta
            ``NDVI_fft_amp_k`` y ``NDVI_fft_phase_k``.
        sequence_length: Longitud temporal reconstruida.
        title: Titulo.
        figsize: Tupla ``(ancho, alto)``.

    Returns:
        Figura matplotlib.
    """
    if df.height != cluster_labels.shape[0]:
        raise ValueError(
            f"`df.height`={df.height} debe igualar `cluster_labels`={cluster_labels.shape[0]}."
        )

    if fft_cols is None:
        cols = [
            c
            for c in df.columns
            if c.startswith("NDVI_fft_amp") or c.startswith("NDVI_fft_phase")
        ]
        fft_cols = tuple(sorted(cols))

    amp_cols = [c for c in fft_cols if "_fft_amp_" in c]
    phase_cols = [c for c in fft_cols if "_fft_phase_" in c]

    if not amp_cols or not phase_cols:
        # Fallback: sin FFT, grafica vacia con mensaje (notebook seguira
        # sin romperse en CI con dataset reducido).
        fig, ax = plt.subplots(figsize=figsize, dpi=110)
        ax.text(
            0.5,
            0.5,
            "No hay columnas FFT NDVI en el DataFrame.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="#888",
        )
        ax.set_axis_off()
        ax.set_title(title)
        fig.tight_layout()
        return fig

    n_harmonics = min(len(amp_cols), len(phase_cols))
    amp_cols_sorted = sorted(amp_cols, key=lambda c: int(c.rsplit("_", 1)[-1]))[:n_harmonics]
    phase_cols_sorted = sorted(phase_cols, key=lambda c: int(c.rsplit("_", 1)[-1]))[:n_harmonics]

    amps = df.select(amp_cols_sorted).to_numpy().astype(np.float64)
    phases = df.select(phase_cols_sorted).to_numpy().astype(np.float64)
    amps = np.where(np.isfinite(amps), amps, 0.0)
    phases = np.where(np.isfinite(phases), phases, 0.0)

    t = np.linspace(0.0, 1.0, sequence_length, endpoint=False)
    # Reconstruye: y(t) = sum_k amp_k * cos(2*pi*k*t + phase_k); k = 1..K.
    k_indices = np.arange(1, n_harmonics + 1).reshape(1, -1)
    # series shape (N, T)
    series = np.zeros((df.height, sequence_length), dtype=np.float64)
    for ti, tv in enumerate(t):
        arg = 2 * np.pi * k_indices * tv + phases
        series[:, ti] = (amps * np.cos(arg)).sum(axis=1)

    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    palette = plt.get_cmap("tab10").colors  # type: ignore[attr-defined]
    n_clusters = int(cluster_labels.max()) + 1 if cluster_labels.size > 0 else 1
    doy = t * 365.0
    for cid in range(n_clusters):
        mask = cluster_labels == cid
        if not np.any(mask):
            continue
        mean_curve = series[mask].mean(axis=0)
        ax.plot(
            doy,
            mean_curve,
            color=palette[cid % len(palette)],
            label=f"cluster {cid} (n={int(mask.sum())})",
            linewidth=1.5,
        )
    ax.set_xlabel("Dia del anio (DOY)")
    ax.set_ylabel("NDVI reconstruido (FFT)")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig
