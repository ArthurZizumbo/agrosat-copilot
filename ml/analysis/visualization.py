"""Utilidades de visualización para EDA Sentinel-2.

- `stretch_2_98`: stretch percentil 2-98 por banda para visualización RGB.
- `folium_rois`: mapa folium con las 4 ROIs desde `config/rois.yaml`.
- `plot_band_grid`: grid de histogramas por banda.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


def stretch_2_98(arr: np.ndarray) -> np.ndarray:
    """Aplica stretch percentil 2-98 banda por banda.

    Acepta arrays 2D `(H, W)` o 3D `(C, H, W)` / `(H, W, C)`.

    Args:
        arr: Array numérico a estirar.

    Returns:
        Array `float32` con valores en [0, 1].
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        lo, hi = np.percentile(arr, [2.0, 98.0])
        out = (arr - lo) / max(hi - lo, 1e-6)
        return np.clip(out, 0.0, 1.0)
    if arr.ndim == 3:
        # Detectar layout (C, H, W) vs (H, W, C) por el tamaño del eje
        if arr.shape[0] <= 12 and arr.shape[0] < arr.shape[-1]:
            channels = arr.shape[0]
            out = np.empty_like(arr, dtype=np.float32)
            for c in range(channels):
                lo, hi = np.percentile(arr[c], [2.0, 98.0])
                out[c] = np.clip((arr[c] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            return out
        else:
            out = np.empty_like(arr, dtype=np.float32)
            for c in range(arr.shape[-1]):
                lo, hi = np.percentile(arr[..., c], [2.0, 98.0])
                out[..., c] = np.clip((arr[..., c] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            return out
    raise ValueError(f"Array shape no soportado: {arr.shape}")


def folium_rois(rois_yaml: Path, output_path: Path) -> Any:
    """Genera un mapa folium con todas las ROIs declaradas en YAML.

    Args:
        rois_yaml: Ruta al `config/rois.yaml`.
        output_path: Ruta para guardar el HTML final.

    Returns:
        Objeto `folium.Map` ya guardado en disco.
    """
    import folium  # type: ignore[import-untyped]
    import yaml

    with rois_yaml.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    rois = cfg.get("rois", [])
    if not rois:
        m = folium.Map(location=[42.0, 10.0], zoom_start=5)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(output_path))
        return m

    # Centrar en bbox global aproximado
    lats: list[float] = []
    lons: list[float] = []
    for r in rois:
        bbox = r.get("bbox", [])
        if len(bbox) == 4:
            lons.extend([bbox[0], bbox[2]])
            lats.extend([bbox[1], bbox[3]])
    if lats and lons:
        center = [float(np.mean(lats)), float(np.mean(lons))]
    else:
        center = [42.0, 10.0]

    m = folium.Map(location=center, zoom_start=5, tiles="OpenStreetMap")
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, roi in enumerate(rois):
        bbox = roi.get("bbox")
        name = roi.get("name", f"roi_{i}")
        color = palette[i % len(palette)]
        if bbox and len(bbox) == 4:
            min_lon, min_lat, max_lon, max_lat = bbox
            folium.Rectangle(
                bounds=[[min_lat, min_lon], [max_lat, max_lon]],
                color=color,
                weight=2,
                fill=True,
                fill_opacity=0.15,
                popup=folium.Popup(
                    f"<b>{name}</b><br>region: {roi.get('region', 'n/a')}<br>"
                    f"crops: {', '.join(roi.get('crops', []))}",
                    max_width=300,
                ),
            ).add_to(m)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    return m


def plot_band_grid(
    df: pl.DataFrame,
    output_path: Path,
    dpi: int = 200,
    band_col: str = "band",
    value_col: str = "value",
    bands: list[str] | None = None,
) -> None:
    """Genera un grid 2x5 de histogramas, uno por banda Sentinel-2.

    Args:
        df: DataFrame long-format con columnas `band_col, value_col`.
        output_path: Ruta al PNG de salida.
        dpi: Resolución (default 200).
        band_col: Nombre columna banda.
        value_col: Nombre columna valor.
        bands: Subset de bandas (default 10 PASTIS).
    """
    import matplotlib.pyplot as plt

    if bands is None:
        bands = [
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B11",
            "B12",
        ]

    fig, axes = plt.subplots(2, 5, figsize=(18, 7), dpi=dpi)
    axes = axes.flatten()
    for i, band in enumerate(bands):
        ax = axes[i]
        vals = (
            df.filter(pl.col(band_col) == band)
            .select(value_col)
            .to_series()
            .drop_nulls()
            .to_numpy()
        )
        if vals.size:
            ax.hist(vals, bins=80, color="#1f77b4", alpha=0.85)
            ax.set_title(band, fontsize=11)
            ax.set_yscale("log")
        else:
            ax.set_visible(False)
    for j in range(len(bands), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Distribuciones por banda Sentinel-2 (log y)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _categorical_palette(n: int) -> list[tuple[float, float, float, float]]:
    """Paleta tab20 ciclica con `n` colores RGBA."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    return [cmap(i % 20) for i in range(n)]


def tsne_scatter(
    emb_2d: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: Path,
    dpi: int = 200,
) -> Any:
    """Scatter 2D coloreado por clase con leyenda agrupada.

    Args:
        emb_2d: Array shape `(n, 2)` con coords t-SNE.
        labels: Array shape `(n,)` con etiquetas categoricas.
        title: Titulo del plot.
        out_path: Ruta de salida PNG.
        dpi: Resolucion.

    Returns:
        Figura matplotlib (ya guardada en disco si out_path se provee).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7), dpi=dpi)
    if emb_2d.size == 0 or len(labels) == 0:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
    else:
        uniq = list(dict.fromkeys(labels.tolist()))
        colors = _categorical_palette(len(uniq))
        for cls, color in zip(uniq, colors, strict=True):
            mask = labels == cls
            ax.scatter(
                emb_2d[mask, 0],
                emb_2d[mask, 1],
                s=4,
                alpha=0.6,
                c=[color],
                label=str(cls),
            )
        ax.legend(loc="best", fontsize=8, markerscale=2)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig


def umap_scatter(
    emb_2d: np.ndarray,
    labels: np.ndarray,
    title: str,
    out_path: Path,
    dpi: int = 200,
) -> Any:
    """Scatter 2D UMAP coloreado por clase.

    Mismos parametros que `tsne_scatter`. Estilo identico para comparabilidad.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7), dpi=dpi)
    if emb_2d.size == 0 or len(labels) == 0:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
    else:
        uniq = list(dict.fromkeys(labels.tolist()))
        colors = _categorical_palette(len(uniq))
        for cls, color in zip(uniq, colors, strict=True):
            mask = labels == cls
            ax.scatter(
                emb_2d[mask, 0],
                emb_2d[mask, 1],
                s=4,
                alpha=0.6,
                c=[color],
                label=str(cls),
            )
        ax.legend(loc="best", fontsize=8, markerscale=2)
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig


def correlation_heatmap(
    corr_df: pl.DataFrame,
    out_path: Path,
    threshold: float = 0.7,
    dpi: int = 200,
) -> Any:
    """Heatmap de la matriz de correlacion 64x64.

    Acepta el output long-format de `correlation_matrix()` y reconstruye la
    matriz simetrica para graficar.

    Args:
        corr_df: DataFrame long con columnas `dim_i, dim_j, pearson|spearman`.
        out_path: Ruta PNG.
        threshold: Linea de referencia (anota celdas con |corr| > threshold).
        dpi: Resolucion.

    Returns:
        Figura matplotlib.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 10), dpi=dpi)
    if corr_df.is_empty():
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return fig

    method = "pearson" if "pearson" in corr_df.columns else "spearman"
    dims = sorted(set(corr_df["dim_i"].to_list()) | set(corr_df["dim_j"].to_list()))
    idx = {d: i for i, d in enumerate(dims)}
    n = len(dims)
    mat = np.full((n, n), np.nan, dtype=np.float64)
    for row in corr_df.iter_rows(named=True):
        i = idx[row["dim_i"]]
        j = idx[row["dim_j"]]
        val = row[method]
        if val is None:
            continue
        mat[i, j] = float(val)
        mat[j, i] = float(val)

    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(0, n, 4))
    ax.set_xticklabels([dims[i] for i in range(0, n, 4)], rotation=90, fontsize=7)
    ax.set_yticks(range(0, n, 4))
    ax.set_yticklabels([dims[i] for i in range(0, n, 4)], fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, label=method)
    redundant = int(((np.abs(mat) > threshold) & (np.abs(mat) < 0.999)).sum() // 2)
    ax.set_title(f"Matriz correlacion 64x64 ({method}) — {redundant} pares |r|>{threshold}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig


def qq_grid(
    stats_df: pl.DataFrame,
    out_path: Path,
    ncols: int = 8,
    dpi: int = 200,
) -> Any:
    """Grid de QQ plots aproximados por dimension.

    Como el QQ exacto requiere los valores raw, este grid grafica curva normal
    de comparacion `(z_score, value_aprox)` derivada de los estadisticos
    (mean, std). Para visualizacion exacta usar `scipy.stats.probplot` en el
    notebook con los arrays raw.

    Args:
        stats_df: Output de `qq_test_dims()` con columnas `dim, mean, std,
            skewness, kurtosis, shapiro_pvalue`.
        out_path: Ruta PNG.
        ncols: Columnas del grid (default 8 -> 64 dims = 8x8).
        dpi: Resolucion.

    Returns:
        Figura matplotlib.
    """
    import matplotlib.pyplot as plt

    n = stats_df.height
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return fig

    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 1.8 * nrows), dpi=dpi)
    axes_flat = np.asarray(axes).flatten()
    z = np.linspace(-3, 3, 50)
    for i, row in enumerate(stats_df.iter_rows(named=True)):
        ax = axes_flat[i]
        mean = row.get("mean", 0.0) or 0.0
        std = row.get("std", 1.0) or 1.0
        pval = row.get("shapiro_pvalue", float("nan"))
        ax.plot(z, z * std + mean, lw=0.8, color="#1f77b4")
        ax.plot(z, z, lw=0.5, color="black", linestyle="--", alpha=0.5)
        ax.set_title(f"{row['dim']} p={pval:.2g}", fontsize=6)
        ax.tick_params(axis="both", labelsize=5)
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("QQ plots aproximados vs N(0,1) por dimension AlphaEarth", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig


def cross_region_scatter(
    consistency_df: pl.DataFrame,
    out_path: Path,
    annotate_top_k: int = 10,
    dpi: int = 200,
) -> Any:
    """Scatter `importance_italia` vs `importance_francia` con linea identidad.

    Args:
        consistency_df: Output de `cross_region_consistency()`.
        out_path: Ruta PNG.
        annotate_top_k: Cuantas dims top etiquetar con su nombre.
        dpi: Resolucion.

    Returns:
        Figura matplotlib.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    if consistency_df.is_empty():
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return fig

    x = consistency_df["importance_italia"].to_numpy()
    y = consistency_df["importance_francia"].to_numpy()
    dims = consistency_df["dim"].to_list()
    consistent = consistency_df["consistente_top10"].to_list()

    colors = ["#d62728" if c else "#1f77b4" for c in consistent]
    ax.scatter(x, y, c=colors, s=40, alpha=0.75, edgecolors="black", linewidths=0.3)
    max_val = float(np.nanmax([x.max() if x.size else 0, y.max() if y.size else 0]))
    ax.plot([0, max_val], [0, max_val], "k--", lw=0.8, alpha=0.6, label="Identidad")

    # Anotar top-K por suma de importance
    sorted_idx = np.argsort(-(x + y))[:annotate_top_k]
    for i in sorted_idx:
        ax.annotate(
            dims[i],
            (x[i], y[i]),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )

    ax.set_xlabel("Importance Italia (Dynamic World)")
    ax.set_ylabel("Importance Francia (PASTIS-R)")
    ax.set_title("Consistencia cross-region de top dimensiones AlphaEarth")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig
