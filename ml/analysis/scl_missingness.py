"""Missing-data analysis in Sentinel-2 via the SCL layer.

Sentinel-2 L2A delivers the SCL (Scene Classification Layer) with 12 classes
that indicate cloud cover, shadow, saturation, etc. These are the
"missing-data masks" for univariate analysis of reflective bands.
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
"""Mapping `scl_code -> readable name` of the 12 Sentinel-2 L2A SCL classes."""

_INVALID_CODES: set[int] = {0, 1, 3, 8, 9, 10}
"""Codes considered 'missing/invalid' for usable reflectance."""


def pct_missing_by_scl(
    df: pl.DataFrame,
    group_by: list[str] | None = None,
    scl_col: str = "scl",
) -> pl.DataFrame:
    """Compute the percentage per SCL class grouping by ROI and season.

    Args:
        df: DataFrame with column `scl_col` (int) and grouping columns.
        group_by: List of columns (default `["roi", "season"]`).
        scl_col: Name of the SCL column.

    Returns:
        DataFrame with columns `group_by + [scl_class, scl_name, n, pct]`,
        where for each group the sum of `pct` over all SCL classes is 100.
    """
    group_by = group_by or ["roi", "season"]
    if scl_col not in df.columns:
        raise ValueError(f"SCL column `{scl_col}` not found in df.")

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
    """Compute the aggregate pct of "invalid" pixels (clouds/shadow/saturated).

    Args:
        df: DataFrame with an SCL column.
        group_by: Grouping columns.
        scl_col: Name of the SCL column.

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
