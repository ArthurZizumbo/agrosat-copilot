"""Stratified sampling utilities with Polars 1.x.

Provides proportional sampling over Polars DataFrames, useful for
EDA when there are imbalanced classes or regions and the original
distribution should be preserved without saturating memory.
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
    """Proportional stratified sampling with Polars.

    Iterates over the groups defined by `by` and takes from each one a fraction
    proportional to the relative size of the group, guaranteeing that the total
    sum of rows is close to `n` (it may vary +/- 1 per stratum due to
    rounding and strata with fewer rows than the assigned quota).

    Args:
        df: Input DataFrame with the categorical columns in `by`.
        by: List of columns to stratify by (e.g. ["roi", "class_id"]).
        n: Target total sample size.
        seed: Seed for reproducibility.

    Returns:
        DataFrame with approximately `n` rows preserving the relative
        proportions of the strata.

    Raises:
        ValueError: If `by` is empty, `n` is non-positive, or if `df` does
            not contain some column of `by`.
    """
    if not by:
        raise ValueError("`by` cannot be empty.")
    if n <= 0:
        raise ValueError(f"`n` must be positive, got {n}.")
    missing = [c for c in by if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in df: {missing}")
    if df.is_empty():
        return df.clear()

    total = df.height
    counts = df.group_by(by).agg(pl.len().alias("__count__"))

    # Stable order of the strata for deterministic reproducibility.
    counts = counts.sort(by)

    fractions: list[pl.DataFrame] = []
    for row in counts.iter_rows(named=True):
        count_g = int(row["__count__"])
        # proportional quota, at least 1 if the group has rows
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
