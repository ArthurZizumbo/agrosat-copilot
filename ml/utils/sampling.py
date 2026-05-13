"""Utilidades de muestreo estratificado con Polars 1.x.

Proporciona muestreo proporcional sobre DataFrames Polars, útil para
EDA cuando hay clases o regiones desbalanceadas y se quiere preservar la
distribución original sin saturar memoria.
"""

from __future__ import annotations

import math

import polars as pl


def stratified_sample(
    df: pl.DataFrame,
    by: list[str],
    n: int,
    seed: int = 42,
) -> pl.DataFrame:
    """Muestreo estratificado proporcional con Polars.

    Recorre los grupos definidos por `by` y toma de cada uno una fracción
    proporcional al tamaño relativo del grupo, garantizando que la suma
    total de filas sea cercana a `n` (puede variar +/- 1 por estrato por
    redondeo y por estratos con menos filas que la cuota asignada).

    Args:
        df: DataFrame de entrada con las columnas categóricas en `by`.
        by: Lista de columnas para estratificar (ej. ["roi", "class_id"]).
        n: Tamaño total objetivo del sample.
        seed: Semilla para reproducibilidad.

    Returns:
        DataFrame con aproximadamente `n` filas preservando proporciones
        relativas de los estratos.

    Raises:
        ValueError: Si `by` es vacío, `n` es no positivo, o si `df` no
            contiene alguna columna de `by`.
    """
    if not by:
        raise ValueError("`by` no puede ser vacío.")
    if n <= 0:
        raise ValueError(f"`n` debe ser positivo, se recibió {n}.")
    missing = [c for c in by if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas no encontradas en df: {missing}")
    if df.is_empty():
        return df.clear()

    total = df.height
    counts = df.group_by(by).agg(pl.len().alias("__count__"))

    # Orden estable de los estratos para reproducibilidad determinística.
    counts = counts.sort(by)

    fractions: list[pl.DataFrame] = []
    for row in counts.iter_rows(named=True):
        count_g = int(row["__count__"])
        # cuota proporcional, al menos 1 si el grupo tiene filas
        quota = max(1, math.floor(count_g / total * n))
        quota = min(quota, count_g)

        filter_expr = pl.lit(True)
        for col in by:
            filter_expr = filter_expr & (pl.col(col) == row[col])
        group_df = df.filter(filter_expr)
        sample_g = group_df.sample(n=quota, seed=seed, shuffle=True)
        fractions.append(sample_g)

    if not fractions:
        return df.clear()
    return pl.concat(fractions, how="vertical_relaxed")
