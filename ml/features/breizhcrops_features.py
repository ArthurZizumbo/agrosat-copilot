"""Adapter of raw BreizhCrops series to the project's 185-feature vector.

BreizhCrops delivers Sentinel-2 time series in **raw bands** and in *long*
format (one row per ``(parcel_id, t, band)``), whereas the project's feature
pipeline (``ml.features.temporal_features``) operates over an
``xarray.DataArray`` with dims ``(time, band)`` whose band labels are the
**17 spectral indices** already computed, not the raw bands.

This module bridges that gap identically to how PASTIS-R reaches its 185
features, reusing exactly the same canonical components:

1. :func:`ml.features.spectral_indices.compute_index` for the 17 indices.
2. :func:`ml.features.temporal_features.extract_temporal_features` for the
   153 statistics + 24 FFT columns + 8 phenological ones.

This way the feature space of BreizhCrops and PASTIS-R is the same (same column
names, same semantics), enabling a *direct tabular* transfer: train XGBoost on
PASTIS-R and predict over BreizhCrops without retraining.

Reflectance scale
-----------------
The BreizhCrops L2A bands arrive as DN (digital numbers, range ~0-10000), just
like raw PASTIS-R. ``compute_index`` expects reflectance in [0, 1] (see its
docstring), so we divide by ``REFLECTANCE_SCALE`` (10000) before computing
indices. It is the same contract that the Avance 1 EDA documented for
Sentinel-2 DN.

Output
------
:func:`build_breizhcrops_features` returns a ``pl.DataFrame`` with one row per
parcel and the same ~185 feature columns as the US-018 subset of PASTIS-R, plus
``parcel_id``, ``year``, ``class_id`` and ``class_name``.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl
import structlog
import xarray as xr

from ml.features.spectral_indices import compute_index
from ml.features.temporal_features import DEFAULT_INDICES, extract_temporal_features
from ml.ingest.pastis_loader import PASTIS_S2_BANDS

logger = structlog.get_logger(__name__)

__all__ = [
    "REFLECTANCE_SCALE",
    "build_breizhcrops_features",
    "pixel_series_to_index_dataarray",
]

#: Factor to convert Sentinel-2 DN (0-10000) to reflectance [0, 1].
#: Contract of ``compute_index`` (see EDA Avance 1, global conclusions).
REFLECTANCE_SCALE: Final[float] = 10_000.0

#: Raw band used as a stable integer id hash when the ``parcel_id``
#: from BreizhCrops is not convertible to int (does not apply here: the ids are
#: numeric, but the defense is kept).
_INT_PARCEL_FALLBACK: Final[int] = -1


def pixel_series_to_index_dataarray(
    parcel_long: pl.DataFrame,
    *,
    parcel_id_int: int,
    year: int,
    bands: tuple[str, ...] = tuple(PASTIS_S2_BANDS),
    indices: tuple[str, ...] = DEFAULT_INDICES,
    reflectance_scale: float = REFLECTANCE_SCALE,
) -> xr.DataArray | None:
    """Convert the long-format of ONE parcel to an indices ``DataArray``.

    Pivots the raw bands to a ``(time, raw_band)`` matrix, scales them to
    reflectance, computes the 17 canonical spectral indices and returns an
    ``xarray.DataArray`` with dims ``(time, band=indices)`` and the
    ``parcel_id`` / ``year`` attrs that :func:`extract_temporal_features`
    requires.

    Args:
        parcel_long: Sub-DataFrame of a single parcel with columns
            ``t, date, band, value`` (format of
            :func:`ml.ingest.breizhcrops_loader.breizhcrops_pixel_series`).
        parcel_id_int: Integer id of the parcel (DataArray attrs; must be
            ``int`` because ``extract_temporal_features`` does ``int(...)``).
        year: Year of the agricultural cycle (DataArray attrs).
        bands: Canonical order of raw Sentinel-2 bands expected in
            ``parcel_long`` (default :data:`PASTIS_S2_BANDS`).
        indices: Spectral indices to compute (default the 17 canonical ones).
        reflectance_scale: DN -> reflectance divisor (default 10000).

    Returns:
        ``xarray.DataArray`` dims ``(time, band)`` with band labels = the
        names of the indices, or ``None`` if the parcel does not have at least
        2 valid temporal steps (insufficient for FFT / phenology).
    """
    # Pivot long -> wide (time x raw_band). One row per t, one col per band.
    wide = (
        parcel_long.select("t", "date", "band", "value")
        .pivot(on="band", index=["t", "date"], values="value", aggregate_function="first")
        .sort("t")
    )
    if wide.height < 2:
        return None

    missing = [b for b in bands if b not in wide.columns]
    if missing:
        # Without some raw band we cannot compute the full set of indices.
        logger.warning(
            "breizhcrops_parcel_missing_bands",
            parcel_id=parcel_id_int,
            missing=missing,
        )
        return None

    # Matrix (time, raw_band) scaled to reflectance.
    raw = wide.select(list(bands)).to_numpy().astype(np.float64)
    refl = raw / reflectance_scale

    # Temporal axis datetime64[ns] from the YYYYMMDD integer.
    date_ints = wide.get_column("date").to_numpy().astype(np.int64)
    times = _yyyymmdd_to_datetime64(date_ints)

    da_bands = xr.DataArray(
        refl.astype(np.float32),
        dims=("time", "band"),
        coords={"time": times, "band": list(bands)},
    )

    # Compute each index over the (time,) series and stack them to (time, n_idx).
    index_stack = np.empty((da_bands.sizes["time"], len(indices)), dtype=np.float32)
    for col, name in enumerate(indices):
        index_stack[:, col] = compute_index(da_bands, name).values

    da_idx = xr.DataArray(
        index_stack,
        dims=("time", "band"),
        coords={"time": times, "band": list(indices)},
    )
    da_idx.attrs["parcel_id"] = int(parcel_id_int)
    da_idx.attrs["year"] = int(year)
    return da_idx


def build_breizhcrops_features(
    pixel_series: pl.DataFrame,
    *,
    indices: tuple[str, ...] = DEFAULT_INDICES,
    reflectance_scale: float = REFLECTANCE_SCALE,
) -> pl.DataFrame:
    """Build the 185-feature vector per parcel from the long-format.

    Iterates per parcel, reuses :func:`pixel_series_to_index_dataarray` +
    :func:`extract_temporal_features` (the same pipeline as PASTIS-R) and
    concatenates the results appending ``class_id`` and ``class_name``.

    Args:
        pixel_series: Long-format DataFrame from
            :func:`ml.ingest.breizhcrops_loader.breizhcrops_pixel_series`
            (columns ``parcel_id, t, date, doy, band, value, class_id,
            class_name``). May contain several parcels.
        indices: Spectral indices to compute (default the 17 canonical ones,
            which produce the same 185 columns as PASTIS-R).
        reflectance_scale: DN -> reflectance divisor (default 10000).

    Returns:
        ``pl.DataFrame`` with one row per parcel and columns ``parcel_id``
        (Utf8), ``year``, ``class_id``, ``class_name`` + ~185 features.
        Empty (with minimal schema) if no parcel produces features.
    """
    if pixel_series.height == 0:
        return pl.DataFrame(
            schema={
                "parcel_id": pl.Utf8,
                "year": pl.Int64,
                "class_id": pl.Int16,
                "class_name": pl.Utf8,
            }
        )

    # Map parcel_id (str) -> stable integer id and class/year metadata.
    parcel_meta = (
        pixel_series.select("parcel_id", "class_id", "class_name")
        .unique(subset=["parcel_id"], keep="first")
        .with_row_index(name="parcel_id_int")
    )
    meta_map = {
        row["parcel_id"]: (
            int(row["parcel_id_int"]),
            int(row["class_id"]),
            str(row["class_name"]),
        )
        for row in parcel_meta.iter_rows(named=True)
    }

    feature_frames: list[pl.DataFrame] = []
    parcel_ids = list(meta_map.keys())
    n_total = len(parcel_ids)
    n_skipped = 0

    for str_pid in parcel_ids:
        pid_int, class_id, class_name = meta_map[str_pid]
        parcel_long = pixel_series.filter(pl.col("parcel_id") == str_pid)
        # The year is derived from the first valid date of the series.
        year = _year_from_first_date(parcel_long)
        da_idx = pixel_series_to_index_dataarray(
            parcel_long,
            parcel_id_int=pid_int,
            year=year,
            indices=indices,
            reflectance_scale=reflectance_scale,
        )
        if da_idx is None:
            n_skipped += 1
            continue

        feats = extract_temporal_features(da_idx, indices=indices)
        feats = feats.with_columns(
            pl.lit(str_pid).alias("parcel_id"),
            pl.lit(class_id).cast(pl.Int16).alias("class_id"),
            pl.lit(class_name).alias("class_name"),
        )
        feature_frames.append(feats)

    if not feature_frames:
        return pl.DataFrame(
            schema={
                "parcel_id": pl.Utf8,
                "year": pl.Int64,
                "class_id": pl.Int16,
                "class_name": pl.Utf8,
            }
        )

    result = pl.concat(feature_frames, how="vertical_relaxed")
    meta_cols = ("parcel_id", "year", "class_id", "class_name")
    n_features = len([c for c in result.columns if c not in meta_cols])
    logger.info(
        "breizhcrops_features_built",
        n_parcels_in=n_total,
        n_parcels_out=result.height,
        n_skipped=n_skipped,
        n_features=n_features,
    )
    return result


# ---------------------------------------------------------------------------
# Private helpers.
# ---------------------------------------------------------------------------


def _yyyymmdd_to_datetime64(date_ints: np.ndarray) -> np.ndarray:
    """Convert ``YYYYMMDD`` integers to ``datetime64[ns]``.

    Args:
        date_ints: Array of ``YYYYMMDD`` integers (the ``date`` field of the
            BreizhCrops long-format).

    Returns:
        ``datetime64[ns]`` array of the same length. Invalid dates (0) are
        mapped to the first day of the valid range to avoid breaking the axis.
    """
    out = np.empty(date_ints.size, dtype="datetime64[ns]")
    for i, d in enumerate(date_ints):
        if d <= 0:
            out[i] = np.datetime64("NaT")
            continue
        year = d // 10000
        month = (d // 100) % 100
        day = d % 100
        out[i] = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}", "ns")
    # Replace NaT with the minimum valid value (defense; BreizhCrops has no NaT).
    if np.isnat(out).any():
        valid = out[~np.isnat(out)]
        fill = valid.min() if valid.size else np.datetime64("2017-01-01", "ns")
        out[np.isnat(out)] = fill
    return out


def _year_from_first_date(parcel_long: pl.DataFrame) -> int:
    """Derive the cycle year from the parcel's first valid date."""
    dates = parcel_long.get_column("date").to_numpy()
    valid = dates[dates > 0]
    if valid.size == 0:
        return 2017
    return int(valid.min() // 10000)
