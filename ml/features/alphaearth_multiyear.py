"""Multi-year AlphaEarth embedding averaging for the E-b ensemble (US-042).

AlphaEarth (``SATELLITE_EMBEDDING/V1/ANNUAL`` v1.1, 64-dim) is an ANNUAL
embedding. PASTIS-R spans the 2018-2019 agricultural seasons, so a single-year
embedding can miss inter-annual signal. This module averages the per-parcel
embeddings of two years (default 2018 + 2019) into a single 64-dim vector per
parcel, the tabular feature space fed to the XGBoost-AlphaEarth base learner of
the E-b stacking ensemble.

The two yearly parquets share the canonical ``parcel_id`` (``"{patch}_{local}"``,
the SAME key as the PASTIS-R OOF sidecars), so the average is an inner join on
``parcel_id`` followed by the per-dimension mean. If a year is missing, the
caller falls back to the single available year (documented in the US-042 plan).

Conventions: Polars (never pandas), structlog, type hints, English docstrings,
Spanish prose, no emojis. Real AlphaEarth data only.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl
import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "ALPHAEARTH_DIM",
    "alphaearth_dim_columns",
    "average_alphaearth_years",
    "build_averaged_alphaearth",
]

#: AlphaEarth Satellite Embedding V1 Annual dimensionality.
ALPHAEARTH_DIM: int = 64


def alphaearth_dim_columns() -> list[str]:
    """Return the ordered embedding columns ``["dim_00", ..., "dim_63"]``."""
    return [f"dim_{i:02d}" for i in range(ALPHAEARTH_DIM)]


def average_alphaearth_years(
    frames: Sequence[pl.DataFrame],
    *,
    id_col: str = "parcel_id",
) -> pl.DataFrame:
    """Average the per-parcel AlphaEarth embeddings of several years.

    Inner-joins the yearly frames on ``id_col`` (so only parcels present in EVERY
    year survive) and returns the per-dimension mean over the years. The output
    carries ``id_col`` + ``dim_00..dim_63`` (and a ``n_years`` column documenting
    how many years were averaged). A single-frame input returns its embedding
    columns unchanged (the degenerate "fallback to one year" case).

    Args:
        frames: yearly AlphaEarth frames, each with ``id_col`` + ``dim_00..dim_63``.
        id_col: parcel id column (canonical ``"{patch}_{local}"``).

    Returns:
        A Polars DataFrame ``id_col`` + ``dim_00..dim_63`` (mean) + ``n_years``.

    Raises:
        ValueError: if ``frames`` is empty or a frame lacks the embedding columns.
    """
    if not frames:
        raise ValueError("average_alphaearth_years needs at least one frame.")
    dims = alphaearth_dim_columns()
    for i, fr in enumerate(frames):
        missing = [c for c in (id_col, *dims) if c not in fr.columns]
        if missing:
            raise ValueError(f"frame {i} is missing columns: {missing}.")

    selected = [fr.select([id_col, *dims]) for fr in frames]
    if len(selected) == 1:
        return selected[0].with_columns(pl.lit(1).alias("n_years"))

    # Inner-join the years on parcel_id, suffixing each year's dims, then mean.
    merged = selected[0]
    for j, fr in enumerate(selected[1:], start=1):
        merged = merged.join(fr, on=id_col, how="inner", suffix=f"_y{j}")
    n_years = len(selected)
    mean_exprs = []
    for d in dims:
        cols = [d] + [f"{d}_y{j}" for j in range(1, n_years)]
        mean_exprs.append(
            (sum(pl.col(c) for c in cols) / float(n_years)).alias(d)
        )
    out = merged.select([id_col, *mean_exprs]).with_columns(
        pl.lit(n_years).alias("n_years")
    )
    logger.info(
        "alphaearth_years_averaged",
        n_years=n_years,
        n_parcels=out.height,
        n_dims=len(dims),
    )
    return out


def build_averaged_alphaearth(
    year_paths: Sequence[Path | str],
    *,
    out_path: Path | str,
    id_col: str = "parcel_id",
) -> Path:
    """Read yearly AlphaEarth parquets, average them, and write the result.

    Convenience wrapper over :func:`average_alphaearth_years` for the US-042
    pipeline: reads each existing yearly parquet, averages the per-parcel
    embeddings, and writes ``out_path``. Missing year files are skipped with a
    warning (so 2018+2019 degrades gracefully to whichever exists -- the plan's
    "fallback to 2019" path).

    Args:
        year_paths: yearly AlphaEarth parquet paths (e.g. 2018 + 2019).
        out_path: destination parquet for the averaged embeddings.
        id_col: parcel id column.

    Returns:
        The written ``out_path``.

    Raises:
        FileNotFoundError: if NONE of the year paths exist.
    """
    frames: list[pl.DataFrame] = []
    used: list[str] = []
    for p in year_paths:
        path = Path(p)
        if not path.is_file():
            logger.warning("alphaearth_year_missing", path=str(path))
            continue
        frames.append(pl.read_parquet(path))
        used.append(str(path))
    if not frames:
        raise FileNotFoundError(
            f"none of the AlphaEarth year parquets exist: {list(year_paths)}."
        )
    averaged = average_alphaearth_years(frames, id_col=id_col)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    averaged.write_parquet(out)
    logger.info(
        "alphaearth_averaged_written",
        out=str(out),
        n_years=len(frames),
        years_used=used,
        n_parcels=averaged.height,
    )
    return out
