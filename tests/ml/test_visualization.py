"""Tests para `ml.analysis.visualization`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ml.analysis.visualization import folium_rois, plot_band_grid, stretch_2_98


def test_stretch_2_98_2d_in_unit_range() -> None:
    """Stretch sobre 2D devuelve valores en [0, 1]."""
    rng = np.random.default_rng(42)
    arr = rng.normal(loc=1000, scale=200, size=(64, 64)).astype(np.float32)
    out = stretch_2_98(arr)
    assert out.shape == arr.shape
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_stretch_2_98_3d_channels_first() -> None:
    """Stretch sobre (C, H, W) preserva shape y rango."""
    rng = np.random.default_rng(42)
    arr = rng.normal(loc=1000, scale=200, size=(3, 32, 32)).astype(np.float32)
    out = stretch_2_98(arr)
    assert out.shape == arr.shape
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_stretch_2_98_invalid_shape() -> None:
    """4D no soportado -> ValueError."""
    arr = np.zeros((2, 2, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        stretch_2_98(arr)


def test_plot_band_grid_creates_png(tmp_path: Path) -> None:
    """plot_band_grid debe generar un archivo PNG en disco."""
    rng = np.random.default_rng(42)
    rows = []
    for band in ["B02", "B03", "B04"]:
        for v in rng.normal(loc=1000, scale=100, size=200):
            rows.append({"band": band, "value": float(v)})
    df = pl.DataFrame(rows)
    out_path = tmp_path / "grid.png"
    plot_band_grid(df, output_path=out_path, dpi=100, bands=["B02", "B03", "B04"])
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_folium_rois_saves_html(tmp_path: Path) -> None:
    """folium_rois debe generar HTML válido desde un YAML mínimo."""
    yaml_path = tmp_path / "rois.yaml"
    yaml_path.write_text(
        """rois:
  - name: test_roi
    region: Italy
    bbox: [9.0, 44.5, 12.0, 46.0]
    crs: EPSG:4326
    crops: [maiz]
""",
        encoding="utf-8",
    )
    out_path = tmp_path / "map.html"
    folium_rois(yaml_path, out_path)
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content or "<html" in content
