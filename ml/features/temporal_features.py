"""Agregación temporal de series multiespectrales por parcela (US-015).

Este módulo extrae features descriptivos, espectrales (FFT) y fenológicos a
partir de una serie temporal multibanda de una parcela individual, devolviendo
un :class:`polars.DataFrame` listo para cargar a la tabla ``features_parcels``
de PostgreSQL (ver ``ml/features/persist_features.py``).

La salida consta de aproximadamente 187 columnas por (parcel_id, year):

- 153 estadísticos: 9 stats (``mean``, ``std``, ``min``, ``max``,
  ``p05``, ``p25``, ``p50``, ``p75``, ``p95``) por cada uno de los 17
  índices espectrales canónicos.
- 24 columnas FFT: 4 amplitudes y 4 fases (DC + 3 armónicos) por cada
  uno de los 3 índices clave (NDVI, NDWI, EVI).
- 8 columnas fenológicas derivadas de NDVI: ``sog_doy``, ``peak_doy``,
  ``peak_value``, ``senescence_doy``, ``ndvi_auc``, ``ndvi_slope_pre_peak``,
  ``ndvi_slope_post_peak``, ``maturity_duration_days``.
- 2 columnas índice: ``parcel_id``, ``year``.

Pre-condiciones
---------------
Sentinel-2 tiene revisita irregular (~5 días con huecos por nubes). Antes
de aplicar FFT la serie se interpola linealmente a una rejilla diaria. Los
estadísticos descriptivos se calculan sobre las muestras originales (sin
imputar). La fenología se calcula sobre la curva NDVI interpolada diaria.

Referencias
-----------
- White et al. 1997 — umbral SOG (start of greenness) NDVI 0.3 para inicio
  de fase de crecimiento. DOI 10.1029/97GB00993.
- Reed et al. 2003 — AUC NDVI como proxy de productividad primaria bruta
  (GPP). DOI 10.1016/S0034-4257(03)00018-1.
- Jönsson & Eklundh 2002 — TIMESAT, métricas fenológicas (peak, slopes,
  amplitude). DOI 10.1016/S0098-3004(02)00040-X.
- Eklundh & Jönsson 2017 — TIMESAT 3.3 software para análisis temporal de
  vegetación. ISBN 978-91-87983-19-0.

Notas de implementación
-----------------------
- Polars LazyFrame con ``.collect(engine="streaming")`` (Polars 1.x) para
  escalar a ~30k parcelas Italia sin desbordar memoria. La firma legacy
  ``streaming=True`` queda desaconsejada por upstream.
- Función pura sin side-effects: dos llamadas consecutivas devuelven
  DataFrames idénticos byte-a-byte (determinismo verificado en test).
- Fenología graceful: si NDVI nunca cruza el umbral SOG, las columnas
  fenológicas se devuelven como ``None`` (NULL en Postgres) y no se lanza
  excepción.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl
import structlog
import xarray as xr

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constantes públicas
# ---------------------------------------------------------------------------

#: Índices espectrales canónicos del proyecto (debe coincidir con
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

#: Subset al que se aplica FFT por defecto (US-015 AC-2).
DEFAULT_FFT_INDICES: Final[tuple[str, ...]] = ("NDVI", "NDWI", "EVI")

#: Sufijos de estadísticos generados por índice.
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
# API pública
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
    """Extrae features temporales por ``(parcel_id, year)`` desde una serie xarray.

    Args:
        parcel_timeseries: DataArray con dims ``(time, band)`` y attrs
            ``{"parcel_id": int, "year": int}``. La coord ``time`` debe ser
            ``datetime64``; la coord ``band`` debe contener los nombres de los
            índices listados en ``indices``.
        indices: índices espectrales a agregar estadísticamente (default: los
            17 canónicos del proyecto).
        fft_indices: subconjunto de ``indices`` al que se aplica descomposición
            FFT (default: ``("NDVI", "NDWI", "EVI")``).
        n_fft_harmonics: número de armónicos FFT a extraer **además** del
            componente DC (default 3 → 4 amplitudes y 4 fases por índice). La
            justificación agronómica de 3 armónicos está en
            ``docs/spectral_indices.md`` §"Temporal aggregation".
        sog_threshold: umbral NDVI para start of greenness (default 0.3, White
            et al. 1997).
        maturity_pct: fracción del NDVI pico que define el período de madurez
            (default 0.8 → días con NDVI ≥ 0.8 * pico).

    Returns:
        :class:`polars.DataFrame` con una fila por ``(parcel_id, year)`` y
        ~187 columnas (ver docstring del módulo).

    Raises:
        ValueError: si ``parcel_timeseries`` no tiene ``attrs["parcel_id"]`` o
            ``attrs["year"]``, si ``time`` no es ``datetime64``, o si
            ``indices`` contiene un nombre ausente en ``coord band``.
    """
    _validate_input(parcel_timeseries, indices=indices, fft_indices=fft_indices)

    parcel_id = int(parcel_timeseries.attrs["parcel_id"])
    year = int(parcel_timeseries.attrs["year"])

    lf = _xr_to_lazy(parcel_timeseries, indices=indices, parcel_id=parcel_id, year=year)

    stats_df = _aggregate_stats(lf, indices=indices).collect(engine="streaming")

    # Curvas por índice ya interpoladas a rejilla diaria (compartidas por FFT
    # y fenología).
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
# Helpers privados
# ---------------------------------------------------------------------------


def _validate_input(
    da: xr.DataArray,
    *,
    indices: tuple[str, ...],
    fft_indices: tuple[str, ...],
) -> None:
    """Valida attrs, dims y bandas requeridas."""
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
    """Convierte el DataArray a un LazyFrame en formato long para agregar."""
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
    """Genera 9 stats por índice agrupando por (parcel_id, year, band) y pivota.

    Output: LazyFrame con columnas ``parcel_id, year`` + 9*len(indices) cols
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

    # Pivot manual a wide para garantizar nombres deterministas.
    pivoted: dict[str, list[object]] = {"parcel_id": [], "year": []}
    for idx in indices:
        for suffix in _STAT_SUFFIXES:
            pivoted[f"{idx}_{suffix}"] = []

    # Esperamos un solo grupo (parcel_id, year) por contrato de entrada.
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
    """Interpola linealmente cada índice a una rejilla diaria del año.

    Args:
        da: DataArray con dims ``(time, band)``.
        indices: lista de índices a interpolar (subset de ``coord band``).

    Returns:
        Mapping ``index_name -> np.ndarray`` con valores diarios sobre la
        ventana ``[t_min, t_max]`` en pasos de 1 día. Si la serie tiene
        menos de 2 puntos válidos, devuelve un array vacío para ese índice.
    """
    times = np.asarray(da.coords["time"].values, dtype="datetime64[ns]")
    t_min = times.min()
    t_max = times.max()

    daily_axis = np.arange(t_min, t_max + np.timedelta64(1, "D"), np.timedelta64(1, "D"))
    if daily_axis.size < 2:
        return {idx: np.empty(0, dtype=np.float64) for idx in indices}

    # Convertimos timestamps a float (días desde el inicio) para np.interp.
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
        # np.interp asume xp creciente; ya está ordenado.
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
    """Calcula amplitud y fase de los primeros ``n_harmonics`` armónicos FFT.

    Para una serie real ``x[n]`` de longitud ``N``, se aplica ``np.fft.rfft``:

    - Componente DC (``k=0``): amplitud = ``|X[0]| / N`` (igual a la media).
    - Armónicos ``k >= 1``: amplitud = ``|X[k]| * 2 / N`` (normalización
      single-sided), fase = ``angle(X[k])`` en radianes en ``(-π, π]``.

    Args:
        daily_curves: salida de :func:`_interpolate_daily`.
        fft_indices: índices sobre los que aplicar FFT.
        n_harmonics: número de armónicos además del DC.
        parcel_id: identificador de la parcela.
        year: año del ciclo agrícola.

    Returns:
        :class:`polars.DataFrame` con una sola fila y columnas
        ``parcel_id, year`` + ``{idx}_fft_amp_{k}`` y ``{idx}_fft_phase_{k}``
        para ``k`` ∈ ``[0, n_harmonics]``.
    """
    row: dict[str, object] = {"parcel_id": parcel_id, "year": year}
    n_components = n_harmonics + 1  # incluye DC

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
    """Aplica ``np.fft.rfft`` y devuelve listas de amplitudes y fases."""
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
        # DC normalizado por N; resto single-sided (x 2 / N).
        amp = float(magnitude / n) if k == 0 else float(magnitude * 2.0 / n)
        amps.append(amp)
        # Fase del DC carece de interpretación física: se reporta 0.0 si la
        # señal no es nula.
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
    """Construye el DataFrame de 1 fila con las 8 métricas fenológicas."""
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
    """Detecta SOG, peak, senescencia y AUC sobre una curva NDVI diaria.

    Implementa el criterio de umbral fijo (White et al. 1997): SOG es el
    primer día del año en que NDVI cruza ``sog_threshold`` ascendente;
    senescencia es el primer día tras el peak en que NDVI cae bajo el mismo
    umbral. AUC es la integral trapezoidal de la curva (Reed et al. 2003).

    Returns:
        ``{"sog_doy", "peak_doy", "peak_value", "senescence_doy", "ndvi_auc"}``
        con ``None`` para métricas que no aplican (ej. NDVI nunca cruza el
        umbral o peak en el extremo del ciclo).
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

    # SOG: primer cruce ascendente del umbral antes (o en) el peak.
    sog_doy: int | None = None
    for i in range(1, peak_idx + 1):
        if ndvi_daily[i - 1] < sog_threshold <= ndvi_daily[i]:
            sog_doy = i + 1  # DOY 1-based
            break
    # Caso peak en el borde inicial: si NDVI[0] ya supera el umbral.
    if sog_doy is None and peak_idx == 0 and peak_value >= sog_threshold:
        sog_doy = 1

    # Senescencia: primer cruce descendente del umbral tras el peak.
    senescence_doy: int | None = None
    for i in range(peak_idx + 1, ndvi_daily.size):
        if ndvi_daily[i - 1] >= sog_threshold > ndvi_daily[i]:
            senescence_doy = i + 1
            break

    if peak_value < sog_threshold:
        # Curva por debajo del umbral en toda su extensión: graceful None.
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
    """Calcula pendientes pre/post peak y duración de madurez.

    - ``slope_pre``: pendiente de regresión lineal en la ventana
      ``[sog_doy, peak_doy]`` (NDVI/día).
    - ``slope_post``: pendiente en ``[peak_doy, senescence_doy]``.
    - ``maturity_duration_days``: número de días consecutivos en torno al
      peak donde NDVI ≥ ``maturity_pct * peak_value`` (TIMESAT-like, Jönsson
      & Eklundh 2002).
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
    # Buscamos la racha contigua que contiene al peak.
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
