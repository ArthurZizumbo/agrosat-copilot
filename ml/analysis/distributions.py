"""Normality tests and per-band transformation recommendations.

`shapiro_test_bands`: Shapiro-Wilk with subsample (scipy limit: 5000).
`recommend_transform`: Box-Cox if all values are positive, Yeo-Johnson if there
are negatives (PowerTransformer admits any sign).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from scipy import stats

TransformName = Literal["none", "box-cox", "yeo-johnson"]


def shapiro_test_bands(
    df: pl.DataFrame,
    subsample_n: int = 5000,
    seed: int = 42,
    band_col: str = "band",
    value_col: str = "value",
    alpha: float = 0.01,
) -> pl.DataFrame:
    """Shapiro-Wilk normality test per band.

    Subsamples each band to `subsample_n` to respect the scipy limit.

    Args:
        df: Long-format DataFrame.
        subsample_n: Subsample size per band (max 5000 for scipy).
        seed: Seed for a reproducible subsample.
        band_col: Band column name.
        value_col: Value column name.
        alpha: Significance level (default 0.01).

    Returns:
        DataFrame with columns `band, n_test, shapiro_stat, shapiro_pvalue, normal_at_alpha`.
    """
    n = min(subsample_n, 5000)
    rng = np.random.default_rng(seed)
    rows = []
    for band, group in df.group_by(band_col):
        band_name = band[0] if isinstance(band, tuple) else band
        vals = group.select(value_col).to_series().drop_nulls().to_numpy()
        if vals.size < 3:
            continue
        if vals.size > n:
            idx = rng.choice(vals.size, size=n, replace=False)
            sample = vals[idx]
        else:
            sample = vals
        try:
            stat, p = stats.shapiro(sample)
        except Exception:  # noqa: BLE001
            stat, p = float("nan"), float("nan")
        rows.append(
            {
                "band": band_name,
                "n_test": int(sample.size),
                "shapiro_stat": float(stat),
                "shapiro_pvalue": float(p),
                "normal_at_alpha": bool(p > alpha) if not np.isnan(p) else False,
            }
        )
    return pl.DataFrame(rows).sort("band")


def recommend_transform(
    df: pl.DataFrame,
    band_col: str = "band",
    value_col: str = "value",
    normality_df: pl.DataFrame | None = None,
    alpha: float = 0.01,
) -> pl.DataFrame:
    """Recommend a transformation per band based on sign and normality.

    Rules:
    - If the band already passes Shapiro at `alpha`: `none`.
    - If all values are strictly positive: `box-cox`.
    - If there are values <= 0: `yeo-johnson` (admits any sign).

    Args:
        df: Long-format DataFrame.
        band_col: Band column name.
        value_col: Value column name.
        normality_df: Optional result of `shapiro_test_bands`.
        alpha: Significance level.

    Returns:
        DataFrame with columns `band, min_value, all_positive, normal, recommended_transform`.
    """
    summary = df.group_by(band_col).agg(
        pl.col(value_col).min().alias("min_value"),
    )

    rows: list[dict[str, object]] = []
    for r in summary.iter_rows(named=True):
        band = r[band_col]
        min_v = r["min_value"]
        all_pos = min_v is not None and min_v > 0
        is_normal = False
        if normality_df is not None:
            match = normality_df.filter(pl.col("band") == band)
            if match.height > 0:
                pval = match["shapiro_pvalue"][0]
                is_normal = (pval is not None) and (pval > alpha)
        if is_normal:
            tname: TransformName = "none"
        elif all_pos:
            tname = "box-cox"
        else:
            tname = "yeo-johnson"
        rows.append(
            {
                "band": band,
                "min_value": float(min_v) if min_v is not None else None,
                "all_positive": bool(all_pos),
                "normal": bool(is_normal),
                "recommended_transform": tname,
            }
        )
    return pl.DataFrame(rows).sort("band")
