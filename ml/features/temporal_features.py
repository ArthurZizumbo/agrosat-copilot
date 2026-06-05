"""Temporal aggregation of per-parcel multispectral series (US-015).

This module extracts descriptive, spectral (FFT) and phenological features
from a multiband time series of a single parcel, returning a
:class:`polars.DataFrame` ready to load into the PostgreSQL
``features_parcels`` table (see ``ml/features/persist_features.py``).

The output consists of approximately 187 columns per (parcel_id, year):

- 153 statistics: 9 stats (``mean``, ``std``, ``min``, ``max``,
  ``p05``, ``p25``, ``p50``, ``p75``, ``p95``) for each of the 17 canonical
  spectral indices.
- 24 FFT columns: 4 amplitudes and 4 phases (DC + 3 harmonics) for each of
  the 3 key indices (NDVI, NDWI, EVI).
- 8 NDVI-derived phenology columns: ``sog_doy``, ``peak_doy``,
  ``peak_value``, ``senescence_doy``, ``ndvi_auc``, ``ndvi_slope_pre_peak``,
  ``ndvi_slope_post_peak``, ``maturity_duration_days``.
- 2 index columns: ``parcel_id``, ``year``.

Pre-conditions
--------------
Sentinel-2 has an irregular revisit (~5 days with cloud gaps). Before
applying FFT the series is linearly interpolated to a daily grid. The
descriptive statistics are computed on the original samples (without
imputation). Phenology is computed on the daily interpolated NDVI curve.

References
----------
- White et al. 1997 — SOG (start of greenness) threshold NDVI 0.3 for the
  start of the growth phase. DOI 10.1029/97GB00993.
- Reed et al. 2003 — NDVI AUC as a proxy for gross primary productivity
  (GPP). DOI 10.1016/S0034-4257(03)00018-1.
- Jönsson & Eklundh 2002 — TIMESAT, phenology metrics (peak, slopes,
  amplitude). DOI 10.1016/S0098-3004(02)00040-X.
- Eklundh & Jönsson 2017 — TIMESAT 3.3 software for temporal vegetation
  analysis. ISBN 978-91-87983-19-0.

Implementation notes
--------------------
- Polars LazyFrame with ``.collect(engine="streaming")`` (Polars 1.x) to
  scale to ~30k Italy parcels without exhausting memory. The legacy
  ``streaming=True`` signature is discouraged by upstream.
- Pure function with no side-effects: two consecutive calls return
  byte-for-byte identical DataFrames (determinism verified in tests).
- Graceful phenology: if NDVI never crosses the SOG threshold, the phenology
  columns are returned as ``None`` (NULL in Postgres) and no exception is
  raised.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl
import structlog
import xarray as xr

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Canonical spectral indices of the project (must match
#: :data:`ml.features.spectral_indices.INDEX_NAMES`).
DEFAULT_INDICES: Final[tuple[str, ...]] = (
    "NDVI",
    "NDWI",
    "EVI",
    "NDMI",
    "NBR",
    "MSAVI2",
    "NDRE",
    "MCARI",
    "CCCI",
    "GCVI",
    "PSRI",
    "NDCI",
    "FAPAR",
    "LAI",
    "RENDVI",
    "SAVI",
    "TSAVI",
)

#: Subset to which FFT is applied by default (US-015 AC-2).
DEFAULT_FFT_INDICES: Final[tuple[str, ...]] = ("NDVI", "NDWI", "EVI")

#: Suffixes of statistics generated per index.
_STAT_SUFFIXES: Final[tuple[str, ...]] = (
    "mean",
    "std",
    "min",
    "max",
    "p05",
    "p25",
    "p50",
    "p75",
    "p95",
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_temporal_features(
    parcel_timeseries: xr.DataArray,
    *,
    indices: tuple[str, ...] = DEFAULT_INDICES,
    fft_indices: tuple[str, ...] = DEFAULT_FFT_INDICES,
    n_fft_harmonics: int = 3,
    sog_threshold: float = 0.3,
    maturity_pct: float = 0.8,
) -> pl.DataFrame:
    """Extract temporal features per ``(parcel_id, year)`` from an xarray series.

    Args:
        parcel_timeseries: DataArray with dims ``(time, band)`` and attrs
            ``{"parcel_id": int, "year": int}``. The ``time`` coord must be
            ``datetime64``; the ``band`` coord must contain the names of the
            indices listed in ``indices``.
        indices: spectral indices to aggregate statistically (default: the
            17 canonical project indices).
        fft_indices: subset of ``indices`` to which FFT decomposition is
            applied (default: ``("NDVI", "NDWI", "EVI")``).
        n_fft_harmonics: number of FFT harmonics to extract **in addition** to
            the DC component (default 3 → 4 amplitudes and 4 phases per index).
            The agronomic justification for 3 harmonics is in
            ``docs/spectral_indices.md`` §"Temporal aggregation".
        sog_threshold: NDVI threshold for start of greenness (default 0.3,
            White et al. 1997).
        maturity_pct: fraction of the NDVI peak that defines the maturity
            period (default 0.8 → days with NDVI ≥ 0.8 * peak).

    Returns:
        :class:`polars.DataFrame` with one row per ``(parcel_id, year)`` and
        ~187 columns (see the module docstring).

    Raises:
        ValueError: if ``parcel_timeseries`` lacks ``attrs["parcel_id"]`` or
            ``attrs["year"]``, if ``time`` is not ``datetime64``, or if
            ``indices`` contains a name absent from ``coord band``.
    """
    _validate_input(parcel_timeseries, indices=indices, fft_indices=fft_indices)

    parcel_id = int(parcel_timeseries.attrs["parcel_id"])
    year = int(parcel_timeseries.attrs["year"])

    lf = _xr_to_lazy(parcel_timeseries, indices=indices, parcel_id=parcel_id, year=year)

    stats_df = _aggregate_stats(lf, indices=indices).collect(engine="streaming")

    # Per-index curves already interpolated to a daily grid (shared by FFT
    # and phenology).
    daily_curves = _interpolate_daily(parcel_timeseries, indices=indices)

    fft_df = _fft_harmonics(
        daily_curves,
        fft_indices=fft_indices,
        n_harmonics=n_fft_harmonics,
        parcel_id=parcel_id,
        year=year,
    )

    pheno_df = _phenology_frame(
        daily_curves["NDVI"] if "NDVI" in daily_curves else None,
        parcel_id=parcel_id,
        year=year,
        sog_threshold=sog_threshold,
        maturity_pct=maturity_pct,
    )

    result = stats_df.join(fft_df, on=["parcel_id", "year"], how="inner").join(
        pheno_df, on=["parcel_id", "year"], how="inner"
    )

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_input(
    da: xr.DataArray,
    *,
    indices: tuple[str, ...],
    fft_indices: tuple[str, ...],
) -> None:
    """Validate attrs, dims and required bands."""
    for attr in ("parcel_id", "year"):
        if attr not in da.attrs:
            raise ValueError(f"parcel_timeseries.attrs missing required key '{attr}'")

    if "time" not in da.dims:
        raise ValueError("parcel_timeseries must have a 'time' dimension")
    if "band" not in da.dims:
        raise ValueError("parcel_timeseries must have a 'band' dimension")

    time_values = da.coords["time"].values
    if not np.issubdtype(time_values.dtype, np.datetime64):
        raise ValueError(
            f"parcel_timeseries.coords['time'] must be datetime64; got {time_values.dtype}"
        )

    available_bands = set(da.coords["band"].values.tolist())
    missing = [idx for idx in indices if idx not in available_bands]
    if missing:
        raise ValueError(
            f"indices {missing} not present in parcel_timeseries.coords['band'] "
            f"(available: {sorted(available_bands)})"
        )

    fft_missing = [idx for idx in fft_indices if idx not in indices]
    if fft_missing:
        raise ValueError(
            f"fft_indices {fft_missing} must be a subset of indices "
            f"(current indices: {list(indices)})"
        )


def _xr_to_lazy(
    da: xr.DataArray,
    *,
    indices: tuple[str, ...],
    parcel_id: int,
    year: int,
) -> pl.LazyFrame:
    """Convert the DataArray to a long-format LazyFrame for aggregation."""
    subset = da.sel(band=list(indices))
    values = np.asarray(subset.values, dtype=np.float64)  # shape (T, B)
    times = np.asarray(subset.coords["time"].values)
    bands = subset.coords["band"].values.tolist()

    n_times, n_bands = values.shape
    time_col = np.repeat(times, n_bands)
    band_col = np.tile(np.asarray(bands), n_times)
    value_col = values.reshape(-1)

    long_df = pl.DataFrame(
        {
            "parcel_id": np.full(value_col.size, parcel_id, dtype=np.int64),
            "year": np.full(value_col.size, year, dtype=np.int32),
            "time": time_col,
            "band": band_col,
            "value": value_col,
        }
    )
    return long_df.lazy()


def _aggregate_stats(
    lf: pl.LazyFrame,
    *,
    indices: tuple[str, ...],
) -> pl.LazyFrame:
    """Generate 9 stats per index grouping by (parcel_id, year, band) and pivot.

    Output: LazyFrame with columns ``parcel_id, year`` + 9*len(indices) cols
    ``{idx}_{stat}``.
    """
    valid = lf.filter(pl.col("value").is_not_nan() & pl.col("value").is_not_null())

    aggregated = valid.group_by(["parcel_id", "year", "band"]).agg(
        [
            pl.col("value").mean().alias("mean"),
            pl.col("value").std(ddof=0).alias("std"),
            pl.col("value").min().alias("min"),
            pl.col("value").max().alias("max"),
            pl.col("value").quantile(0.05, interpolation="linear").alias("p05"),
            pl.col("value").quantile(0.25, interpolation="linear").alias("p25"),
            pl.col("value").quantile(0.50, interpolation="linear").alias("p50"),
            pl.col("value").quantile(0.75, interpolation="linear").alias("p75"),
            pl.col("value").quantile(0.95, interpolation="linear").alias("p95"),
        ]
    )

    aggregated_df = aggregated.collect(engine="streaming")

    # Manual pivot to wide to guarantee deterministic names.
    pivoted: dict[str, list[object]] = {"parcel_id": [], "year": []}
    for idx in indices:
        for suffix in _STAT_SUFFIXES:
            pivoted[f"{idx}_{suffix}"] = []

    # We expect a single group (parcel_id, year) by input contract.
    grouped = aggregated_df.group_by(["parcel_id", "year"], maintain_order=True)
    for (pid, yr), subdf in grouped:
        pivoted["parcel_id"].append(pid)
        pivoted["year"].append(yr)
        band_rows = {row["band"]: row for row in subdf.to_dicts()}
        for idx in indices:
            row = band_rows.get(idx)
            for suffix in _STAT_SUFFIXES:
                pivoted[f"{idx}_{suffix}"].append(
                    float(row[suffix]) if row is not None and row[suffix] is not None else None
                )

    schema: dict[str, pl.DataType] = {
        "parcel_id": pl.Int64(),
        "year": pl.Int32(),
    }
    for idx in indices:
        for suffix in _STAT_SUFFIXES:
            schema[f"{idx}_{suffix}"] = pl.Float64()

    return pl.DataFrame(pivoted, schema=schema).lazy()


def _interpolate_daily(
    da: xr.DataArray,
    *,
    indices: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Linearly interpolate each index to a daily grid of the year.

    Args:
        da: DataArray with dims ``(time, band)``.
        indices: list of indices to interpolate (subset of ``coord band``).

    Returns:
        Mapping ``index_name -> np.ndarray`` with daily values over the
        window ``[t_min, t_max]`` in 1-day steps. If the series has fewer
        than 2 valid points, returns an empty array for that index.
    """
    times = np.asarray(da.coords["time"].values, dtype="datetime64[ns]")
    t_min = times.min()
    t_max = times.max()

    daily_axis = np.arange(t_min, t_max + np.timedelta64(1, "D"), np.timedelta64(1, "D"))
    if daily_axis.size < 2:
        return {idx: np.empty(0, dtype=np.float64) for idx in indices}

    # We convert timestamps to float (days since the start) for np.interp.
    x_known = (times - t_min) / np.timedelta64(1, "D")
    x_query = (daily_axis - t_min) / np.timedelta64(1, "D")

    curves: dict[str, np.ndarray] = {}
    for idx in indices:
        try:
            series = np.asarray(da.sel(band=idx).values, dtype=np.float64)
        except KeyError:
            continue
        mask = np.isfinite(series)
        if mask.sum() < 2:
            curves[idx] = np.empty(0, dtype=np.float64)
            continue
        order = np.argsort(x_known[mask])
        xk = x_known[mask][order]
        yk = series[mask][order]
        # np.interp assumes increasing xp; it is already sorted.
        curves[idx] = np.interp(x_query, xk, yk)

    return curves


def _fft_harmonics(
    daily_curves: dict[str, np.ndarray],
    *,
    fft_indices: tuple[str, ...],
    n_harmonics: int,
    parcel_id: int,
    year: int,
) -> pl.DataFrame:
    """Compute amplitude and phase of the first ``n_harmonics`` FFT harmonics.

    For a real series ``x[n]`` of length ``N``, ``np.fft.rfft`` is applied:

    - DC component (``k=0``): amplitude = ``|X[0]| / N`` (equal to the mean).
    - Harmonics ``k >= 1``: amplitude = ``|X[k]| * 2 / N`` (single-sided
      normalization), phase = ``angle(X[k])`` in radians in ``(-π, π]``.

    Args:
        daily_curves: output of :func:`_interpolate_daily`.
        fft_indices: indices over which to apply FFT.
        n_harmonics: number of harmonics besides the DC.
        parcel_id: parcel identifier.
        year: year of the agricultural cycle.

    Returns:
        :class:`polars.DataFrame` with a single row and columns
        ``parcel_id, year`` + ``{idx}_fft_amp_{k}`` and ``{idx}_fft_phase_{k}``
        for ``k`` ∈ ``[0, n_harmonics]``.
    """
    row: dict[str, object] = {"parcel_id": parcel_id, "year": year}
    n_components = n_harmonics + 1  # includes DC

    for idx in fft_indices:
        curve = daily_curves.get(idx, np.empty(0, dtype=np.float64))
        amps, phases = _compute_rfft_components(curve, n_components=n_components)
        for k in range(n_components):
            amp_k = amps[k]
            phase_k = phases[k]
            row[f"{idx}_fft_amp_{k}"] = float(amp_k) if amp_k is not None else None
            row[f"{idx}_fft_phase_{k}"] = float(phase_k) if phase_k is not None else None

    schema: dict[str, pl.DataType] = {
        "parcel_id": pl.Int64(),
        "year": pl.Int32(),
    }
    for idx in fft_indices:
        for k in range(n_components):
            schema[f"{idx}_fft_amp_{k}"] = pl.Float64()
            schema[f"{idx}_fft_phase_{k}"] = pl.Float64()

    return pl.DataFrame([row], schema=schema)


def _compute_rfft_components(
    curve: np.ndarray,
    *,
    n_components: int,
) -> tuple[list[float | None], list[float | None]]:
    """Apply ``np.fft.rfft`` and return lists of amplitudes and phases."""
    if curve.size == 0:
        return ([None] * n_components, [None] * n_components)

    n = curve.size
    spectrum = np.fft.rfft(curve)
    n_available = spectrum.size

    amps: list[float | None] = []
    phases: list[float | None] = []
    for k in range(n_components):
        if k >= n_available:
            amps.append(None)
            phases.append(None)
            continue
        magnitude = np.abs(spectrum[k])
        # DC normalized by N; rest single-sided (x 2 / N).
        amp = float(magnitude / n) if k == 0 else float(magnitude * 2.0 / n)
        amps.append(amp)
        # The DC phase lacks physical interpretation: 0.0 is reported if the
        # signal is non-zero.
        phases.append(0.0 if k == 0 else float(np.angle(spectrum[k])))

    return amps, phases


def _phenology_frame(
    ndvi_daily: np.ndarray | None,
    *,
    parcel_id: int,
    year: int,
    sog_threshold: float,
    maturity_pct: float,
) -> pl.DataFrame:
    """Build the 1-row DataFrame with the 8 phenology metrics."""
    metrics = _detect_phenology(
        ndvi_daily if ndvi_daily is not None else np.empty(0, dtype=np.float64),
        sog_threshold=sog_threshold,
    )
    slopes = _phenology_slopes(
        ndvi_daily if ndvi_daily is not None else np.empty(0, dtype=np.float64),
        metrics=metrics,
        maturity_pct=maturity_pct,
    )

    row: dict[str, object] = {
        "parcel_id": parcel_id,
        "year": year,
        "sog_doy": metrics["sog_doy"],
        "peak_doy": metrics["peak_doy"],
        "peak_value": metrics["peak_value"],
        "senescence_doy": metrics["senescence_doy"],
        "ndvi_auc": metrics["ndvi_auc"],
        "ndvi_slope_pre_peak": slopes["slope_pre"],
        "ndvi_slope_post_peak": slopes["slope_post"],
        "maturity_duration_days": slopes["maturity_duration_days"],
    }

    schema: dict[str, pl.DataType] = {
        "parcel_id": pl.Int64(),
        "year": pl.Int32(),
        "sog_doy": pl.Int32(),
        "peak_doy": pl.Int32(),
        "peak_value": pl.Float64(),
        "senescence_doy": pl.Int32(),
        "ndvi_auc": pl.Float64(),
        "ndvi_slope_pre_peak": pl.Float64(),
        "ndvi_slope_post_peak": pl.Float64(),
        "maturity_duration_days": pl.Int32(),
    }
    return pl.DataFrame([row], schema=schema)


def _detect_phenology(
    ndvi_daily: np.ndarray,
    *,
    sog_threshold: float,
) -> dict[str, float | int | None]:
    """Detect SOG, peak, senescence and AUC over a daily NDVI curve.

    Implements the fixed-threshold criterion (White et al. 1997): SOG is the
    first day of the year on which NDVI crosses ``sog_threshold`` upward;
    senescence is the first day after the peak on which NDVI falls below the
    same threshold. AUC is the trapezoidal integral of the curve (Reed et al.
    2003).

    Returns:
        ``{"sog_doy", "peak_doy", "peak_value", "senescence_doy", "ndvi_auc"}``
        with ``None`` for metrics that do not apply (e.g. NDVI never crosses
        the threshold or the peak is at the edge of the cycle).
    """
    null_result: dict[str, float | int | None] = {
        "sog_doy": None,
        "peak_doy": None,
        "peak_value": None,
        "senescence_doy": None,
        "ndvi_auc": None,
    }
    if ndvi_daily.size == 0:
        return null_result

    peak_idx = int(np.argmax(ndvi_daily))
    peak_value = float(ndvi_daily[peak_idx])

    # SOG: first ascending crossing of the threshold before (or at) the peak.
    sog_doy: int | None = None
    for i in range(1, peak_idx + 1):
        if ndvi_daily[i - 1] < sog_threshold <= ndvi_daily[i]:
            sog_doy = i + 1  # DOY 1-based
            break
    # Peak at the initial edge case: if NDVI[0] already exceeds the threshold.
    if sog_doy is None and peak_idx == 0 and peak_value >= sog_threshold:
        sog_doy = 1

    # Senescence: first descending crossing of the threshold after the peak.
    senescence_doy: int | None = None
    for i in range(peak_idx + 1, ndvi_daily.size):
        if ndvi_daily[i - 1] >= sog_threshold > ndvi_daily[i]:
            senescence_doy = i + 1
            break

    if peak_value < sog_threshold:
        # Curve below the threshold across its entire extent: graceful None.
        return null_result

    ndvi_auc = float(np.trapezoid(ndvi_daily, dx=1.0))

    return {
        "sog_doy": sog_doy,
        "peak_doy": peak_idx + 1,
        "peak_value": peak_value,
        "senescence_doy": senescence_doy,
        "ndvi_auc": ndvi_auc,
    }


def _phenology_slopes(
    ndvi_daily: np.ndarray,
    *,
    metrics: dict[str, float | int | None],
    maturity_pct: float,
) -> dict[str, float | int | None]:
    """Compute pre/post-peak slopes and maturity duration.

    - ``slope_pre``: linear regression slope in the window
      ``[sog_doy, peak_doy]`` (NDVI/day).
    - ``slope_post``: slope in ``[peak_doy, senescence_doy]``.
    - ``maturity_duration_days``: number of consecutive days around the peak
      where NDVI ≥ ``maturity_pct * peak_value`` (TIMESAT-like, Jönsson &
      Eklundh 2002).
    """
    null_result: dict[str, float | int | None] = {
        "slope_pre": None,
        "slope_post": None,
        "maturity_duration_days": None,
    }
    if ndvi_daily.size == 0 or metrics["peak_doy"] is None:
        return null_result

    peak_idx = int(metrics["peak_doy"]) - 1  # type: ignore[arg-type]
    peak_value = float(metrics["peak_value"])  # type: ignore[arg-type]

    slope_pre: float | None = None
    if metrics["sog_doy"] is not None:
        sog_idx = int(metrics["sog_doy"]) - 1  # type: ignore[arg-type]
        if peak_idx - sog_idx >= 1:
            x = np.arange(sog_idx, peak_idx + 1, dtype=np.float64)
            y = ndvi_daily[sog_idx : peak_idx + 1]
            slope_pre = float(np.polyfit(x, y, 1)[0])

    slope_post: float | None = None
    if metrics["senescence_doy"] is not None:
        sen_idx = int(metrics["senescence_doy"]) - 1  # type: ignore[arg-type]
        if sen_idx - peak_idx >= 1:
            x = np.arange(peak_idx, sen_idx + 1, dtype=np.float64)
            y = ndvi_daily[peak_idx : sen_idx + 1]
            slope_post = float(np.polyfit(x, y, 1)[0])

    threshold = maturity_pct * peak_value
    above = ndvi_daily >= threshold
    # We search for the contiguous run that contains the peak.
    maturity_days: int | None = None
    if above[peak_idx]:
        left = peak_idx
        while left > 0 and above[left - 1]:
            left -= 1
        right = peak_idx
        while right < ndvi_daily.size - 1 and above[right + 1]:
            right += 1
        maturity_days = int(right - left + 1)

    return {
        "slope_pre": slope_pre,
        "slope_post": slope_post,
        "maturity_duration_days": maturity_days,
    }


__all__ = [
    "DEFAULT_FFT_INDICES",
    "DEFAULT_INDICES",
    "extract_temporal_features",
]
