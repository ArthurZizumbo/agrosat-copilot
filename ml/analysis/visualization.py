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
            axes = (1, 2)
            channels = arr.shape[0]
            out = np.empty_like(arr, dtype=np.float32)
            for c in range(channels):
                lo, hi = np.percentile(arr[c], [2.0, 98.0])
                out[c] = np.clip((arr[c] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            _ = axes
            return out
        else:
            out = np.empty_like(arr, dtype=np.float32)
            for c in range(arr.shape[-1]):
                lo, hi = np.percentile(arr[..., c], [2.0, 98.0])
                out[..., c] = np.clip(
                    (arr[..., c] - lo) / max(hi - lo, 1e-6), 0.0, 1.0
                )
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
