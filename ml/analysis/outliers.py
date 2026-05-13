"""Detección de outliers univariados por banda.

Dos métodos:
- IQR clásico (Tukey): outlier si valor < Q1 - k*IQR o > Q3 + k*IQR.
- IsolationForest multivariado sobre pivot ancho banda x píxel.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest


def detect_outliers_iqr(
    df: pl.DataFrame,
    band: str | None = None,
    k: float = 1.5,
    band_col: str = "band",
    value_col: str = "value",
) -> pl.DataFrame:
    """Detecta outliers por banda usando regla IQR (Tukey).

    Args:
        df: DataFrame long-format.
        band: Banda específica a evaluar. Si None, evalúa todas.
        k: Multiplicador IQR (1.5 ligero, 3.0 agresivo).
        band_col: Nombre columna banda.
        value_col: Nombre columna valor.

    Returns:
        DataFrame con columnas `band, n, n_outliers, pct_outliers, q1, q3, lower, upper`.
    """
    if band is not None:
        df = df.filter(pl.col(band_col) == band)

    stats = df.group_by(band_col).agg(
        pl.col(value_col).quantile(0.25).alias("q1"),
        pl.col(value_col).quantile(0.75).alias("q3"),
        pl.len().alias("n"),
    )
    stats = stats.with_columns(
        (pl.col("q3") - pl.col("q1")).alias("iqr"),
    ).with_columns(
        (pl.col("q1") - k * pl.col("iqr")).alias("lower"),
        (pl.col("q3") + k * pl.col("iqr")).alias("upper"),
    )

    joined = df.join(stats, on=band_col)
    out_counts = (
        joined.with_columns(
            (
                (pl.col(value_col) < pl.col("lower"))
                | (pl.col(value_col) > pl.col("upper"))
            ).alias("is_outlier")
        )
        .group_by(band_col)
        .agg(
            pl.col("is_outlier").sum().alias("n_outliers"),
        )
    )

    result = stats.join(out_counts, on=band_col).with_columns(
        (pl.col("n_outliers") / pl.col("n") * 100.0).alias("pct_outliers")
    )
    return result.select(
        [
            pl.col(band_col).alias("band"),
            "n",
            "n_outliers",
            "pct_outliers",
            "q1",
            "q3",
            "lower",
            "upper",
        ]
    ).sort("band")


def detect_outliers_isoforest(
    df: pl.DataFrame,
    contamination: float = 0.05,
    seed: int = 42,
    band_col: str = "band",
    value_col: str = "value",
    pixel_id_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Detecta outliers multivariados con Isolation Forest.

    Pivotea el DataFrame long-format a wide (1 columna por banda) y entrena
    `IsolationForest(contamination=contamination)`. Reporta porcentaje de
    outliers detectados global y la importancia (no nativa) aproximada
    como pct outliers por banda en el subset detectado.

    Args:
        df: DataFrame long-format con columnas `band`, `value`, y un identificador
            de píxel (por defecto `patch_id, t, y, x`).
        contamination: Proporción esperada de outliers en [0, 0.5).
        seed: Semilla para reproducibilidad.
        band_col: Nombre columna banda.
        value_col: Nombre columna valor.
        pixel_id_cols: Columnas que identifican cada píxel-fecha unívocamente.

    Returns:
        DataFrame con columnas `band, n, n_outliers, pct_outliers,
        contamination_target`. Si no hay datos suficientes, retorna DataFrame vacío.
    """
    pixel_id_cols = pixel_id_cols or [
        c for c in ("patch_id", "t", "y", "x") if c in df.columns
    ]
    if not pixel_id_cols:
        return pl.DataFrame()

    wide = df.pivot(
        on=band_col,
        index=pixel_id_cols,
        values=value_col,
        aggregate_function="mean",
    )
    band_cols = [c for c in wide.columns if c not in pixel_id_cols]
    if not band_cols:
        return pl.DataFrame()

    arr = wide.select(band_cols).drop_nulls().to_numpy()
    if arr.shape[0] < 10:
        return pl.DataFrame()

    iso = IsolationForest(
        contamination=contamination, random_state=seed, n_jobs=-1
    )
    preds = iso.fit_predict(arr)
    is_out = preds == -1

    # Atribuir outliers por banda: pct píxeles fuera del p1/p99 por banda
    rows = []
    for i, band in enumerate(band_cols):
        col_vals = arr[:, i]
        out_vals = col_vals[is_out]
        rows.append(
            {
                "band": band,
                "n": int(col_vals.size),
                "n_outliers": int(out_vals.size),
                "pct_outliers": (
                    float(out_vals.size) / col_vals.size * 100.0
                    if col_vals.size
                    else 0.0
                ),
                "contamination_target": contamination * 100.0,
            }
        )
    # Ajustar n_outliers por banda como aporte (no por banda real)
    total_out = int(is_out.sum())
    for r in rows:
        r["n_outliers"] = total_out
        r["pct_outliers"] = float(total_out) / arr.shape[0] * 100.0 if arr.shape[0] else 0.0
        band_pos = band_cols.index(r["band"])
        r["band_mean_outliers"] = (
            float(np.mean(arr[is_out, band_pos])) if total_out else 0.0
        )
    return pl.DataFrame(rows).sort("band")
