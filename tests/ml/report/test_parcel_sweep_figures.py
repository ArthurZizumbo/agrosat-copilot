"""Tests for the per-parcel sweep figures (US-036-b, notebook 06b).

Covers ``ml/report/parcel_sweep_figures.py``: the two PNGs are written from a
tiny synthetic sweep CSV, the headline factor uses the real max, and the guards
(missing file, missing column) fire. No real PASTIS, no training.
"""

from __future__ import annotations

import polars as pl
import pytest

from ml.report.parcel_sweep_figures import render_parcel_sweep_figures


def _write_sweep(path) -> None:
    pl.DataFrame(
        {
            "n_classes": [4, 6, 8, 10, 12],
            "macro_f1": [0.7025, 0.4579, 0.4075, 0.3589, 0.3328],
            "macro_iou": [0.5547, 0.312, 0.2657, 0.2288, 0.2071],
        }
    ).write_csv(path)


def test_renders_both_pngs(tmp_path) -> None:
    """Both figures are written and non-empty."""
    csv = tmp_path / "parcel_sweep.csv"
    _write_sweep(csv)
    out = render_parcel_sweep_figures(csv, tmp_path / "figs")
    assert set(out) == {"curve", "parcel_vs_patch"}
    for p in out.values():
        assert p.is_file()
        assert p.stat().st_size > 0


def test_out_dir_created(tmp_path) -> None:
    """A non-existent out_dir is created."""
    csv = tmp_path / "parcel_sweep.csv"
    _write_sweep(csv)
    target = tmp_path / "nested" / "figs"
    assert not target.exists()
    render_parcel_sweep_figures(csv, target)
    assert target.is_dir()


def test_missing_file_raises(tmp_path) -> None:
    """A non-existent CSV raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        render_parcel_sweep_figures(tmp_path / "nope.csv", tmp_path / "figs")


def test_missing_column_raises(tmp_path) -> None:
    """A CSV lacking macro_iou is rejected."""
    csv = tmp_path / "bad.csv"
    pl.DataFrame({"n_classes": [4], "macro_f1": [0.7]}).write_csv(csv)
    with pytest.raises(ValueError, match="missing column"):
        render_parcel_sweep_figures(csv, tmp_path / "figs")
