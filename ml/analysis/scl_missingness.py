"""Análisis de faltantes en Sentinel-2 vía la capa SCL.

Sentinel-2 L2A entrega la capa SCL (Scene Classification Layer) con 12
clases que indican cobertura nubosa, sombra, saturación, etc. Estas son
las "máscaras de faltantes" para análisis univariado de bandas reflectivas.
"""

from __future__ import annotations

import polars as pl

SCL_CLASSES: dict[int, str] = {
    0: "no_data",
    1: "saturated_defective",
    2: "dark_area",
    3: "cloud_shadow",
    4: "vegetation",
    5: "bare_soil",
    6: "water",
    7: "unclassified",
    8: "medium_cloud",
    9: "high_cloud",
    10: "thin_cirrus",
    11: "snow_ice",
}
"""Mapeo `scl_code -> nombre legible` de las 12 clases SCL Sentinel-2 L2A."""

_INVALID_CODES: set[int] = {0, 1, 3, 8, 9, 10}
"""Códigos que se consideran 'faltantes/inválidos' para reflectancia útil."""


def pct_missing_by_scl(
    df: pl.DataFrame,
    group_by: list[str] | None = None,
    scl_col: str = "scl",
) -> pl.DataFrame:
    """Calcula porcentaje por clase SCL agrupando por ROI y temporada.

    Args:
        df: DataFrame con columna `scl_col` (int) y columnas de agrupación.
        group_by: Lista de columnas (default `["roi", "season"]`).
        scl_col: Nombre de la columna SCL.

    Returns:
        DataFrame con columnas `group_by + [scl_class, scl_name, n, pct]`,
        donde para cada grupo la suma de `pct` por todas las clases SCL es 100.
    """
    group_by = group_by or ["roi", "season"]
    if scl_col not in df.columns:
        raise ValueError(f"Columna SCL `{scl_col}` no encontrada en df.")

    counts = df.group_by([*group_by, scl_col]).agg(pl.len().alias("n"))
    totals = df.group_by(group_by).agg(pl.len().alias("total"))
    joined = counts.join(totals, on=group_by).with_columns(
        (pl.col("n") / pl.col("total") * 100.0).alias("pct"),
    )

    name_map = SCL_CLASSES
    joined = joined.with_columns(
        pl.col(scl_col)
        .cast(pl.Int64)
        .replace_strict(name_map, default="unknown")
        .alias("scl_name"),
        pl.col(scl_col).alias("scl_class"),
    )

    return joined.select([*group_by, "scl_class", "scl_name", "n", "pct"]).sort(
        [*group_by, "scl_class"]
    )


def pct_invalid_total(
    df: pl.DataFrame,
    group_by: list[str] | None = None,
    scl_col: str = "scl",
) -> pl.DataFrame:
    """Calcula pct agregado de píxeles "inválidos" (nubes/sombra/saturado).

    Args:
        df: DataFrame con columna SCL.
        group_by: Columnas de agrupación.
        scl_col: Nombre columna SCL.

    Returns:
        DataFrame `group_by + [pct_invalid, pct_cloud, pct_shadow]`.
    """
    group_by = group_by or ["roi", "season"]
    return (
        df.with_columns(
            pl.col(scl_col).is_in(list(_INVALID_CODES)).alias("__invalid__"),
            pl.col(scl_col).is_in([8, 9, 10]).alias("__cloud__"),
            (pl.col(scl_col) == 3).alias("__shadow__"),
        )
        .group_by(group_by)
        .agg(
            (pl.col("__invalid__").cast(pl.Float64).mean() * 100.0).alias("pct_invalid"),
            (pl.col("__cloud__").cast(pl.Float64).mean() * 100.0).alias("pct_cloud"),
            (pl.col("__shadow__").cast(pl.Float64).mean() * 100.0).alias("pct_shadow"),
        )
        .sort(group_by)
    )
