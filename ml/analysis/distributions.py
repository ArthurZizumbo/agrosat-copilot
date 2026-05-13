"""Tests de normalidad y recomendaciones de transformación por banda.

`shapiro_test_bands`: Shapiro-Wilk con subsample (límite scipy: 5000).
`recommend_transform`: Box-Cox si todos los valores son positivos,
Yeo-Johnson si hay negativos (PowerTransformer admite cualquier signo).
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
    """Test de normalidad Shapiro-Wilk por banda.

    Subsamplea cada banda a `subsample_n` para respetar el límite scipy.

    Args:
        df: DataFrame long-format.
        subsample_n: Tamaño de subsample por banda (max 5000 para scipy).
        seed: Semilla para subsample reproducible.
        band_col: Nombre columna banda.
        value_col: Nombre columna valor.
        alpha: Nivel de significancia (default 0.01).

    Returns:
        DataFrame con columnas `band, n_test, shapiro_stat, shapiro_pvalue, normal_at_alpha`.
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
    """Recomienda transformación por banda según signo y normalidad.

    Reglas:
    - Si la banda ya pasa Shapiro a `alpha`: `none`.
    - Si todos los valores son estrictamente positivos: `box-cox`.
    - Si hay valores <= 0: `yeo-johnson` (admite cualquier signo).

    Args:
        df: DataFrame long-format.
        band_col: Nombre columna banda.
        value_col: Nombre columna valor.
        normality_df: Resultado opcional de `shapiro_test_bands`.
        alpha: Nivel de significancia.

    Returns:
        DataFrame con columnas `band, min_value, all_positive, normal, recommended_transform`.
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
