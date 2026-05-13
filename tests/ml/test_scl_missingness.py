"""Tests para `ml.analysis.scl_missingness`."""

from __future__ import annotations

import polars as pl
import pytest

from ml.analysis.scl_missingness import (
    SCL_CLASSES,
    pct_invalid_total,
    pct_missing_by_scl,
)


@pytest.fixture
def synthetic_scl_df() -> pl.DataFrame:
    """DataFrame sintético: 3 ROIs x 3 seasons x distribución SCL conocida."""
    rows = []
    rois = ["pianura_padana", "toscana", "apulia"]
    seasons = ["spring", "summer", "autumn"]
    # Distribución arbitraria pero verificable
    distribution = {4: 600, 5: 100, 6: 50, 8: 100, 9: 80, 3: 40, 0: 30}
    for roi in rois:
        for season in seasons:
            for code, count in distribution.items():
                rows.extend([{"roi": roi, "season": season, "scl": code}] * count)
    return pl.DataFrame(rows)


def test_pct_missing_by_scl_sums_to_100(synthetic_scl_df: pl.DataFrame) -> None:
    """Para cada (roi, season) la suma de pct debe ser 100 ± 0.01."""
    out = pct_missing_by_scl(synthetic_scl_df)
    sums = out.group_by(["roi", "season"]).agg(pl.col("pct").sum().alias("total_pct"))
    for r in sums.iter_rows(named=True):
        assert abs(r["total_pct"] - 100.0) < 0.01


def test_pct_missing_by_scl_has_class_names(synthetic_scl_df: pl.DataFrame) -> None:
    """Debe traducir scl_class a scl_name usando SCL_CLASSES."""
    out = pct_missing_by_scl(synthetic_scl_df)
    for r in out.iter_rows(named=True):
        expected_name = SCL_CLASSES.get(r["scl_class"], "unknown")
        assert r["scl_name"] == expected_name


def test_pct_missing_by_scl_missing_column_raises() -> None:
    """Falta de columna SCL -> ValueError."""
    bad = pl.DataFrame({"roi": ["a"], "season": ["x"]})
    with pytest.raises(ValueError):
        pct_missing_by_scl(bad)


def test_pct_invalid_total_columns(synthetic_scl_df: pl.DataFrame) -> None:
    """pct_invalid_total reporta pct_invalid, pct_cloud, pct_shadow."""
    out = pct_invalid_total(synthetic_scl_df)
    expected = {"roi", "season", "pct_invalid", "pct_cloud", "pct_shadow"}
    assert expected.issubset(set(out.columns))
    for r in out.iter_rows(named=True):
        assert 0.0 <= r["pct_invalid"] <= 100.0
        assert 0.0 <= r["pct_cloud"] <= 100.0
        assert 0.0 <= r["pct_shadow"] <= 100.0
