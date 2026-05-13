"""Tests para `ml.analysis.distributions`."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ml.analysis.distributions import recommend_transform, shapiro_test_bands


@pytest.fixture
def mixed_distributions_df() -> pl.DataFrame:
    """3 bandas: una normal positiva, una skewed positiva, una con negativos."""
    rng = np.random.default_rng(42)
    n = 1000
    normal_pos = rng.normal(loc=1000, scale=100, size=n).clip(min=1)
    skewed_pos = rng.exponential(scale=500, size=n) + 1.0
    with_neg = rng.normal(loc=0.0, scale=50, size=n)
    rows = []
    for v in normal_pos:
        rows.append({"band": "B02", "value": float(v)})
    for v in skewed_pos:
        rows.append({"band": "B04", "value": float(v)})
    for v in with_neg:
        rows.append({"band": "B08", "value": float(v)})
    return pl.DataFrame(rows)


def test_shapiro_test_bands_returns_one_row_per_band(
    mixed_distributions_df: pl.DataFrame,
) -> None:
    """Espera 3 filas (3 bandas) con columnas estándar."""
    out = shapiro_test_bands(mixed_distributions_df, subsample_n=500)
    assert out.height == 3
    expected = {"band", "n_test", "shapiro_stat", "shapiro_pvalue", "normal_at_alpha"}
    assert expected.issubset(set(out.columns))


def test_recommend_transform_assigns_correctly(
    mixed_distributions_df: pl.DataFrame,
) -> None:
    """Recomendación consistente con signo + normalidad.

    Si Shapiro confirma normalidad la recomendación es `none` (no
    transformar). De lo contrario, `box-cox` para valores positivos y
    `yeo-johnson` cuando existen negativos.
    """
    norm = shapiro_test_bands(mixed_distributions_df, subsample_n=500)
    rec = recommend_transform(mixed_distributions_df, normality_df=norm)
    rec_dict = {r["band"]: r["recommended_transform"] for r in rec.iter_rows(named=True)}
    rec_meta = {r["band"]: r for r in rec.iter_rows(named=True)}

    # B04 es exponencial (no normal, todos positivos) -> box-cox
    assert rec_dict["B04"] == "box-cox"
    # B02 / B08 son normales: la recomendación correcta es "none"
    # pero si Shapiro rechazara, B02 sería box-cox y B08 yeo-johnson.
    for band in ("B02", "B08"):
        tname = rec_dict[band]
        if not rec_meta[band]["normal"]:
            if rec_meta[band]["all_positive"]:
                assert tname == "box-cox"
            else:
                assert tname == "yeo-johnson"
        else:
            assert tname == "none"
