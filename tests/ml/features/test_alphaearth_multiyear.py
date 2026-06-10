"""Tests for the multi-year AlphaEarth averaging (US-042 E-b).

Covers ``ml/features/alphaearth_multiyear.py``: the per-dimension mean over years
(inner join on ``parcel_id``), the single-year degenerate case, the
missing-column guard, and the disk wrapper's graceful skip of a missing year.

In-memory synthetic frames (no GEE, no DVC). Conventions: Polars, no emojis.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from ml.features.alphaearth_multiyear import (
    ALPHAEARTH_DIM,
    alphaearth_dim_columns,
    average_alphaearth_years,
    build_averaged_alphaearth,
)


def _synthetic_year(parcel_ids: list[str], fill: float) -> pl.DataFrame:
    """Build a synthetic AlphaEarth year frame (all dims = ``fill``)."""
    data: dict[str, object] = {"parcel_id": parcel_ids, "year": [2019] * len(parcel_ids)}
    for c in alphaearth_dim_columns():
        data[c] = [fill] * len(parcel_ids)
    return pl.DataFrame(data)


def test_dim_columns_count() -> None:
    """There are exactly 64 embedding columns named dim_00..dim_63."""
    cols = alphaearth_dim_columns()
    assert len(cols) == ALPHAEARTH_DIM == 64
    assert cols[0] == "dim_00"
    assert cols[-1] == "dim_63"


def test_average_two_years_is_per_dim_mean() -> None:
    """Averaging two years yields the per-dimension mean on the shared parcels."""
    y18 = _synthetic_year(["10000_1", "10000_2"], fill=0.2)
    y19 = _synthetic_year(["10000_1", "10000_2"], fill=0.6)
    out = average_alphaearth_years([y18, y19])
    assert out.height == 2
    assert out["n_years"].unique().to_list() == [2]
    for c in alphaearth_dim_columns():
        assert out[c].to_list() == pytest.approx([0.4, 0.4])


def test_average_inner_joins_on_parcel_id() -> None:
    """Only parcels present in EVERY year survive the average (inner join)."""
    y18 = _synthetic_year(["10000_1", "10000_2", "10000_3"], fill=0.2)
    y19 = _synthetic_year(["10000_1", "10000_2"], fill=0.6)
    out = average_alphaearth_years([y18, y19])
    assert sorted(out["parcel_id"].to_list()) == ["10000_1", "10000_2"]


def test_single_year_is_passthrough() -> None:
    """A single frame returns its embeddings unchanged with n_years=1 (fallback)."""
    y19 = _synthetic_year(["10000_1"], fill=0.5)
    out = average_alphaearth_years([y19])
    assert out["n_years"].to_list() == [1]
    assert out["dim_00"].to_list() == [0.5]


def test_missing_columns_raises() -> None:
    """A frame without the embedding columns is rejected."""
    bad = pl.DataFrame({"parcel_id": ["10000_1"], "year": [2019]})
    with pytest.raises(ValueError, match="missing columns"):
        average_alphaearth_years([bad])


def test_empty_frames_raises() -> None:
    """No frames is an error."""
    with pytest.raises(ValueError, match="at least one frame"):
        average_alphaearth_years([])


def test_build_skips_missing_year_file(tmp_path: Path) -> None:
    """The disk wrapper averages only the existing years (graceful fallback)."""
    y19 = _synthetic_year(["10000_1"], fill=0.5)
    p19 = tmp_path / "ae_2019.parquet"
    y19.write_parquet(p19)
    out_path = tmp_path / "ae_mean.parquet"
    result = build_averaged_alphaearth(
        [tmp_path / "ae_2018_missing.parquet", p19], out_path=out_path
    )
    assert result == out_path
    df = pl.read_parquet(out_path)
    assert df["n_years"].to_list() == [1]  # only 2019 existed
    assert df["dim_00"].to_list() == [0.5]


def test_build_all_missing_raises(tmp_path: Path) -> None:
    """If NO year file exists the builder raises (no silent empty output)."""
    with pytest.raises(FileNotFoundError, match="none of the AlphaEarth"):
        build_averaged_alphaearth(
            [tmp_path / "a.parquet", tmp_path / "b.parquet"],
            out_path=tmp_path / "out.parquet",
        )
