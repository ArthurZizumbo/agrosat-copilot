"""Analisis bivariado, multivariado y temporal para US-012 (EDA).

Provee 7 funciones reutilizables sobre DataFrames Polars con bandas Sentinel-2,
indices espectrales derivados y series temporales por parcela:

- `compute_indices_subset`: anade columnas con el subset core de 6 indices
  espectrales (NDVI, NDWI, NDMI, EVI, SAVI, NDRE).
- `correlation_pair`: matriz de correlacion long-format entre dos subsets de
  columnas (Pearson o Spearman).
- `vif_table`: Variance Inflation Factor por columna con statsmodels.
- `phenology_peaks`: detecta pico NDVI por parcela y devuelve mes, doy y year.
- `acf_pacf_per_parcel`: ACF y PACF de la serie NDVI por parcela tras
  resampleo mensual con interpolacion lineal.
- `dtw_cluster_temporal`: clustering DTW (`tslearn.TimeSeriesKMeans`) con
  Sakoe-Chiba band sobre series NDVI z-normalizadas.
- `era5_ndvi_anomaly`: cruza precipitacion anual ERA5 con NDVI maximo anual,
  marcando years secos vs normales por percentil.

Nota Polars: el adapter a pandas se usa unicamente como borde tecnico para
`statsmodels.tsa.stattools` (ACF/PACF) y `variance_inflation_factor` cuando
esas libs no aceptan Polars directamente. Toda la persistencia y agregacion
se mantiene en Polars.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import polars as pl

SPECTRAL_INDICES_CORE: dict[str, str] = {
    "NDVI": "(B08-B04)/(B08+B04)",
    "NDWI": "(B03-B08)/(B03+B08)",
    "NDMI": "(B08-B11)/(B08+B11)",
    "EVI": "2.5*(B08-B04)/(B08+6*B04-7.5*B02+1)",
    "SAVI": "1.5*(B08-B04)/(B08+B04+0.5)",
    "NDRE": "(B08-B05)/(B08+B05)",
}
"""Subset core de 6 indices espectrales US-012.

La biblioteca completa (17 indices) se entrega en US-014 con `spyndex`.
Las formulas usan bandas Sentinel-2 con escala original (sin dividir por 10000);
internamente la implementacion castea a Float64 y aplica un epsilon para evitar
divisiones por cero.
"""

_EPS: float = 1e-6
_DEFAULT_REQUIRED_BANDS: tuple[str, ...] = ("B02", "B03", "B04", "B05", "B08", "B11")


def _safe_div(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    """Division segura con epsilon en el denominador (vectorizada Polars)."""
    return num / pl.when(den.abs() < _EPS).then(_EPS).otherwise(den)


def compute_indices_subset(
    df_bands: pl.DataFrame,
    indices: list[str] | None = None,
    scale: float = 1.0,
    clip_negative: bool = False,
    mask_invalid_band_range: tuple[float, float] | None = None,
    clip_evi_range: tuple[float, float] | None = None,
) -> pl.DataFrame:
    """Calcula el subset core de 6 indices espectrales vectorizado en Polars.

    Las bandas deben estar en columnas separadas (formato wide) con los
    nombres canonicos PASTIS-R: `B02, B03, B04, B05, B08, B11`. Si recibes
    un DataFrame long-format de `pastis_to_polars`, pivota primero por
    `(patch_id, t, y, x)` antes de pasar a esta funcion.

    Args:
        df_bands: DataFrame Polars con las bandas Sentinel-2 como columnas
            float (o casteables a float). Como minimo `B02, B03, B04, B05,
            B08, B11`.
        indices: Subset opcional de indices a computar; default todos los 6.
        scale: Factor multiplicativo aplicado a cada banda antes del computo
            (default 1.0, sin escala). Para Sentinel-2 L2A en DN crudo
            (rango 0-10000) usar 1e-4 para llevar a reflectancia [0, 1] y
            evitar overflow en EVI por denominador cercano a cero.
        clip_negative: Si True clipa valores negativos a 0 antes del computo
            (artefactos BOA pueden producir DN negativo, lo que rompe los
            indices normalizados al hacer a + b cercano a 0). Aplica antes
            del computo de indices. Recomendado usar `mask_invalid_band_range`
            en su lugar para preservar variabilidad real.
        mask_invalid_band_range: Tupla `(min, max)` post-escala. Timesteps con
            cualquier banda requerida fuera de este rango se filtran (drop)
            antes de calcular indices. Ejemplo: `(0.0, 1.5)` con `scale=1e-4`
            descarta DN negativos (artefactos BOA) y reflectancias > 1.5
            (nubes saturadas). Mutuamente excluyente con `clip_negative`.
        clip_evi_range: Tupla `(min, max)` opcional aplicada a `EVI` despues
            del computo. Util porque la formula EVI puede producir outliers
            por geometria del denominador aun con bandas validas. Default
            no clipea.

    Returns:
        DataFrame original con columnas adicionales `NDVI, NDWI, NDMI, EVI,
        SAVI, NDRE` (o el subset solicitado). Si `mask_invalid_band_range` se
        pasa, el DataFrame retornado puede tener menos filas que la entrada
        (los timesteps invalidos quedan filtrados). Si `df_bands` esta vacio
        o le faltan bandas requeridas para los indices pedidos, devuelve el
        df original sin alterar.
    """
    if clip_negative and mask_invalid_band_range is not None:
        raise ValueError("clip_negative y mask_invalid_band_range son mutuamente excluyentes")
    requested = indices or list(SPECTRAL_INDICES_CORE.keys())
    unknown = [i for i in requested if i not in SPECTRAL_INDICES_CORE]
    if unknown:
        raise ValueError(f"Indices no soportados: {unknown}")

    if df_bands.is_empty():
        # Apendear columnas Float64 vacias para preservar contrato downstream
        new_cols = [pl.lit(None, dtype=pl.Float64).alias(name) for name in requested]
        return df_bands.with_columns(new_cols) if new_cols else df_bands

    missing = [b for b in _DEFAULT_REQUIRED_BANDS if b not in df_bands.columns]
    if missing:
        # No podemos computar; devolvemos el df original sin cambios.
        return df_bands

    df = df_bands
    if mask_invalid_band_range is not None:
        lo, hi = mask_invalid_band_range
        # Escalamos las bandas para evaluar el rango en la misma unidad que `scale`
        band_scaled = [
            (pl.col(b).cast(pl.Float64) * scale).alias(f"__scaled_{b}")
            for b in _DEFAULT_REQUIRED_BANDS
        ]
        df = df.with_columns(band_scaled)
        keep = pl.lit(True)
        for b in _DEFAULT_REQUIRED_BANDS:
            keep = keep & pl.col(f"__scaled_{b}").is_between(lo, hi, closed="both")
        df = df.filter(keep).drop([f"__scaled_{b}" for b in _DEFAULT_REQUIRED_BANDS])
        if df.is_empty():
            new_cols = [pl.lit(None, dtype=pl.Float64).alias(name) for name in requested]
            return df.with_columns(new_cols) if new_cols else df

    def _cast_band(name: str) -> pl.Expr:
        expr = pl.col(name).cast(pl.Float64)
        if scale != 1.0:
            expr = expr * scale
        if clip_negative:
            expr = pl.when(expr < 0).then(0.0).otherwise(expr)
        return expr

    casts = {b: _cast_band(b) for b in _DEFAULT_REQUIRED_BANDS}

    exprs: list[pl.Expr] = []
    if "NDVI" in requested:
        num = casts["B08"] - casts["B04"]
        den = casts["B08"] + casts["B04"]
        exprs.append(_safe_div(num, den).alias("NDVI"))
    if "NDWI" in requested:
        num = casts["B03"] - casts["B08"]
        den = casts["B03"] + casts["B08"]
        exprs.append(_safe_div(num, den).alias("NDWI"))
    if "NDMI" in requested:
        num = casts["B08"] - casts["B11"]
        den = casts["B08"] + casts["B11"]
        exprs.append(_safe_div(num, den).alias("NDMI"))
    if "EVI" in requested:
        num = casts["B08"] - casts["B04"]
        den = casts["B08"] + 6.0 * casts["B04"] - 7.5 * casts["B02"] + 1.0
        exprs.append((2.5 * _safe_div(num, den)).alias("EVI"))
    if "SAVI" in requested:
        num = casts["B08"] - casts["B04"]
        den = casts["B08"] + casts["B04"] + 0.5
        exprs.append((1.5 * _safe_div(num, den)).alias("SAVI"))
    if "NDRE" in requested:
        num = casts["B08"] - casts["B05"]
        den = casts["B08"] + casts["B05"]
        exprs.append(_safe_div(num, den).alias("NDRE"))

    df_out = df.with_columns(exprs)
    if clip_evi_range is not None and "EVI" in requested:
        lo, hi = clip_evi_range
        df_out = df_out.with_columns(pl.col("EVI").clip(lo, hi).alias("EVI"))
    return df_out


def correlation_pair(
    df: pl.DataFrame,
    cols_a: list[str],
    cols_b: list[str],
    method: Literal["pearson", "spearman"] = "pearson",
) -> pl.DataFrame:
    """Matriz de correlacion long-format entre dos subconjuntos de columnas.

    Args:
        df: DataFrame Polars con las columnas en `cols_a` y `cols_b`.
        cols_a: Columnas del primer subset (filas del heatmap).
        cols_b: Columnas del segundo subset (columnas del heatmap).
        method: `pearson` o `spearman`.

    Returns:
        DataFrame Polars con columnas `feature_a, feature_b, corr, abs_corr`,
        ordenado por `abs_corr` desc. Si `df` esta vacio o falta alguna
        columna devuelve DataFrame con esquema correcto pero sin filas.
    """
    schema = {
        "feature_a": pl.Utf8,
        "feature_b": pl.Utf8,
        "corr": pl.Float64,
        "abs_corr": pl.Float64,
    }
    if df.is_empty() or not cols_a or not cols_b:
        return pl.DataFrame(schema=schema)
    missing = [c for c in (cols_a + cols_b) if c not in df.columns]
    if missing:
        return pl.DataFrame(schema=schema)

    all_cols = list(dict.fromkeys(cols_a + cols_b))
    arr = df.select(all_cols).cast(pl.Float64, strict=False).to_numpy()
    mask = ~np.isnan(arr).any(axis=1)
    arr_v = arr[mask]
    if arr_v.shape[0] < 2:
        return pl.DataFrame(schema=schema)

    if method == "spearman":
        from scipy.stats import rankdata

        arr_v = np.apply_along_axis(rankdata, 0, arr_v)
    corr_full = np.corrcoef(arr_v, rowvar=False)

    idx = {c: i for i, c in enumerate(all_cols)}
    rows: list[dict[str, Any]] = []
    for ca in cols_a:
        for cb in cols_b:
            val = corr_full[idx[ca], idx[cb]]
            v = float(val) if np.isfinite(val) else float("nan")
            rows.append(
                {
                    "feature_a": ca,
                    "feature_b": cb,
                    "corr": v,
                    "abs_corr": abs(v) if np.isfinite(v) else float("nan"),
                }
            )
    out = pl.DataFrame(rows, schema=schema)
    return out.sort("abs_corr", descending=True, nulls_last=True)


def vif_table(
    df: pl.DataFrame,
    cols: list[str],
    drop_na: bool = True,
    near_perfect_corr_threshold: float = 0.99,
) -> pl.DataFrame:
    """Variance Inflation Factor por columna usando statsmodels.

    Pre-filtra columnas con correlacion absoluta mayor a `near_perfect_corr_threshold`
    para evitar matrices singulares (la primera de cada par se conserva, la
    segunda se descarta y se documenta como `dropped_near_perfect_corr`).

    Args:
        df: DataFrame Polars con las columnas numericas.
        cols: Lista de columnas sobre las que calcular VIF.
        drop_na: Si elimina filas con NaN en cualquier `cols` antes del VIF.
        near_perfect_corr_threshold: Umbral de `|corr|` a partir del cual se
            considera redundancia casi perfecta (default 0.99).

    Returns:
        DataFrame Polars con columnas `feature, vif, status`
        (`status in {"ok", "warning", "drop", "dropped_near_perfect_corr"}`),
        ordenado por VIF desc. Si statsmodels no esta instalado o `cols` esta
        vacio devuelve DataFrame vacio con esquema valido.
    """
    schema = {"feature": pl.Utf8, "vif": pl.Float64, "status": pl.Utf8}
    if df.is_empty() or not cols:
        return pl.DataFrame(schema=schema)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return pl.DataFrame(schema=schema)

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:  # pragma: no cover - statsmodels esta en grupo ml
        return pl.DataFrame(schema=schema)

    sub = df.select(cols).cast(pl.Float64, strict=False)
    if drop_na:
        sub = sub.drop_nulls()
    arr = sub.to_numpy()
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return pl.DataFrame(schema=schema)

    # Pre-filtro de pares casi perfectamente correlacionados para evitar
    # matrices singulares al invertir.
    corr_mat = np.corrcoef(arr, rowvar=False)
    n = arr.shape[1]
    to_drop: set[int] = set()
    drop_reasons: dict[str, str] = {}
    for i in range(n):
        if i in to_drop:
            continue
        for j in range(i + 1, n):
            if j in to_drop:
                continue
            val = corr_mat[i, j]
            if np.isfinite(val) and abs(val) >= near_perfect_corr_threshold:
                to_drop.add(j)
                drop_reasons[cols[j]] = (
                    f"|corr|={abs(float(val)):.3f} con {cols[i]} >= {near_perfect_corr_threshold}"
                )

    keep_idx = [i for i in range(n) if i not in to_drop]
    keep_cols = [cols[i] for i in keep_idx]
    arr_keep = arr[:, keep_idx]

    rows: list[dict[str, Any]] = []
    if arr_keep.shape[1] >= 2:
        for k, name in enumerate(keep_cols):
            try:
                v = float(variance_inflation_factor(arr_keep, k))
            except Exception:  # noqa: BLE001
                v = float("inf")
            if not np.isfinite(v):
                status = "drop"
            elif v >= 10.0:
                status = "drop"
            elif v >= 5.0:
                status = "warning"
            else:
                status = "ok"
            rows.append({"feature": name, "vif": v, "status": status})

    for name in cols:
        if name in drop_reasons:
            rows.append(
                {
                    "feature": name,
                    "vif": float("inf"),
                    "status": "dropped_near_perfect_corr",
                }
            )

    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema).sort("vif", descending=True, nulls_last=True)


def phenology_peaks(
    df_ts: pl.DataFrame,
    parcel_col: str = "parcel_id",
    date_col: str = "date",
    ndvi_col: str = "ndvi",
    class_col: str = "class_name",
) -> pl.DataFrame:
    """Detecta el pico NDVI por parcela.

    La columna `date` puede ser tipo Date / Datetime o entero `YYYYMMDD`. La
    funcion la normaliza a `pl.Date` antes de extraer mes / doy / year.

    Args:
        df_ts: DataFrame Polars con series NDVI por parcela. Columnas
            requeridas: `parcel_col, date_col, ndvi_col, class_col`.
        parcel_col: Nombre columna identificador de parcela.
        date_col: Nombre columna fecha (Date / Datetime / Int).
        ndvi_col: Nombre columna NDVI.
        class_col: Nombre columna clase (se conserva en el output).

    Returns:
        DataFrame Polars con columnas `parcel_id, class_name, peak_ndvi_value,
        peak_ndvi_month, peak_ndvi_doy, peak_ndvi_year`. Vacio (con esquema)
        si el input esta vacio o si faltan columnas requeridas.
    """
    schema = {
        "parcel_id": pl.Utf8,
        "class_name": pl.Utf8,
        "peak_ndvi_value": pl.Float64,
        "peak_ndvi_month": pl.Int64,
        "peak_ndvi_doy": pl.Int64,
        "peak_ndvi_year": pl.Int64,
    }
    if df_ts.is_empty():
        return pl.DataFrame(schema=schema)
    required = {parcel_col, date_col, ndvi_col, class_col}
    if not required.issubset(set(df_ts.columns)):
        return pl.DataFrame(schema=schema)

    df = df_ts.select([parcel_col, date_col, ndvi_col, class_col]).drop_nulls(
        subset=[parcel_col, date_col, ndvi_col]
    )
    if df.is_empty():
        return pl.DataFrame(schema=schema)

    # Normaliza date a pl.Date sin importar el dtype original.
    date_dtype = df.schema[date_col]
    if date_dtype == pl.Date:
        df = df.with_columns(pl.col(date_col).alias("_d"))
    elif date_dtype in (pl.Datetime, pl.Datetime("us"), pl.Datetime("ms"), pl.Datetime("ns")):
        df = df.with_columns(pl.col(date_col).cast(pl.Date).alias("_d"))
    elif date_dtype == pl.Utf8:
        # Intento parsear formato ISO; si falla cae a YYYYMMDD
        try:
            df = df.with_columns(pl.col(date_col).str.to_date(strict=False).alias("_d"))
        except Exception:  # noqa: BLE001
            df = df.with_columns(pl.col(date_col).str.to_date("%Y%m%d", strict=False).alias("_d"))
    else:
        # Asumimos entero YYYYMMDD (formato PASTIS dates-S2).
        df = df.with_columns(
            pl.col(date_col).cast(pl.Utf8).str.to_date("%Y%m%d", strict=False).alias("_d")
        )

    df = df.drop_nulls(subset=["_d"])
    if df.is_empty():
        return pl.DataFrame(schema=schema)

    # Pico por parcela
    idx_max = (
        df.group_by(parcel_col)
        .agg(pl.col(ndvi_col).arg_max().alias("__idx"))
        .with_columns(pl.col("__idx").cast(pl.Int64))
    )
    df_with_pos = df.with_columns(pl.int_range(0, pl.len()).over(parcel_col).alias("__pos"))
    df_peak = df_with_pos.join(
        idx_max,
        left_on=[parcel_col, "__pos"],
        right_on=[parcel_col, "__idx"],
        how="inner",
    )

    out = df_peak.select(
        [
            pl.col(parcel_col).cast(pl.Utf8).alias("parcel_id"),
            pl.col(class_col).cast(pl.Utf8).alias("class_name"),
            pl.col(ndvi_col).cast(pl.Float64).alias("peak_ndvi_value"),
            pl.col("_d").dt.month().cast(pl.Int64).alias("peak_ndvi_month"),
            pl.col("_d").dt.ordinal_day().cast(pl.Int64).alias("peak_ndvi_doy"),
            pl.col("_d").dt.year().cast(pl.Int64).alias("peak_ndvi_year"),
        ]
    )
    return out.unique(subset=["parcel_id"], keep="first").cast(schema)


def _resample_monthly_pandas(
    parcel_id: str,
    dates: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Resamplea mensualmente con interpolacion lineal usando pandas como adapter.

    `pandas.resample("MS").mean()` es el camino mas corto: las fechas Sentinel-2
    son irregulares (median ~5 dias por filtrado de nubes) y necesitamos paso
    uniforme antes de ACF/PACF.

    Args:
        parcel_id: Identificador de la parcela (solo para logging).
        dates: Array `datetime64[D]` ordenado ascendente.
        values: Array de NDVI alineado con `dates`.

    Returns:
        Tupla `(months_ts, ndvi_monthly)` con paso mensual (`MS` = month start).
    """
    import pandas as pd

    _ = parcel_id
    if dates.size == 0 or values.size == 0:
        return np.array([], dtype="datetime64[ns]"), np.array([], dtype=np.float64)
    ser = pd.Series(values, index=pd.to_datetime(dates)).sort_index()
    ser = ser[~ser.index.duplicated(keep="first")]
    monthly = ser.resample("MS").mean().interpolate(method="linear", limit_direction="both")
    return monthly.index.to_numpy(), monthly.to_numpy(dtype=np.float64)


def acf_pacf_per_parcel(
    df_ts: pl.DataFrame,
    max_lag: int = 6,
    parcel_col: str = "parcel_id",
    date_col: str = "date",
    series_col: str = "ndvi",
    class_col: str = "class_name",
) -> pl.DataFrame:
    """ACF y PACF de la serie NDVI por parcela tras resampleo mensual.

    Pre-filtra clases (`class_id`) fuera de `[1, 18]` cuando exista la columna
    `class_id`, para excluir background (0) y void (19) en PASTIS-R. Antes
    del calculo, cada serie se resamplea con paso mensual y se interpola
    linealmente (Sentinel-2 PASTIS tiene median ~5 dias entre acquisitions
    pero irregular por filtrado de nubes).

    Nota Polars: `pandas` se usa como adapter borde porque
    `statsmodels.tsa.stattools.acf/pacf` no acepta Polars y porque
    `pl.DataFrame.upsample` no soporta interpolacion linear out-of-the-box
    sobre series con valores reales arbitrarios.

    Args:
        df_ts: DataFrame Polars long con columnas requeridas
            `parcel_col, date_col, series_col, class_col`. Opcional `class_id`
            para filtrar `[1, 18]`.
        max_lag: Numero de lags a computar (default 6, justificado por
            cobertura PASTIS ~14 meses).
        parcel_col: Columna identificador de parcela.
        date_col: Columna fecha (Date / Datetime / Int YYYYMMDD).
        series_col: Columna NDVI.
        class_col: Columna clase (se preserva en output).

    Returns:
        DataFrame Polars long con columnas
        `parcel_id, class_name, lag, acf, pacf`. ACF y PACF estan acotados
        en `[-1, 1]`. `acf[0] = 1.0` siempre.
    """
    schema = {
        "parcel_id": pl.Utf8,
        "class_name": pl.Utf8,
        "lag": pl.Int64,
        "acf": pl.Float64,
        "pacf": pl.Float64,
    }
    if df_ts.is_empty():
        return pl.DataFrame(schema=schema)
    required = {parcel_col, date_col, series_col, class_col}
    if not required.issubset(set(df_ts.columns)):
        return pl.DataFrame(schema=schema)

    try:
        from statsmodels.tsa.stattools import acf, pacf
    except ImportError:  # pragma: no cover
        return pl.DataFrame(schema=schema)

    df = df_ts.clone()
    if "class_id" in df.columns:
        df = df.filter(pl.col("class_id").is_between(1, 18))
    df = df.select([parcel_col, date_col, series_col, class_col]).drop_nulls(
        subset=[parcel_col, date_col, series_col]
    )
    if df.is_empty():
        return pl.DataFrame(schema=schema)

    # Normaliza date a pl.Date para conversion limpia a numpy datetime64
    date_dtype = df.schema[date_col]
    if date_dtype == pl.Date:
        df = df.with_columns(pl.col(date_col).alias("_d"))
    elif date_dtype in (pl.Datetime, pl.Datetime("us"), pl.Datetime("ms"), pl.Datetime("ns")):
        df = df.with_columns(pl.col(date_col).cast(pl.Date).alias("_d"))
    elif date_dtype == pl.Utf8:
        try:
            df = df.with_columns(pl.col(date_col).str.to_date(strict=False).alias("_d"))
        except Exception:  # noqa: BLE001
            df = df.with_columns(pl.col(date_col).str.to_date("%Y%m%d", strict=False).alias("_d"))
    else:
        df = df.with_columns(
            pl.col(date_col).cast(pl.Utf8).str.to_date("%Y%m%d", strict=False).alias("_d")
        )
    df = df.drop_nulls(subset=["_d"]).sort([parcel_col, "_d"])
    if df.is_empty():
        return pl.DataFrame(schema=schema)

    rows: list[dict[str, Any]] = []
    for pid, sub in df.group_by(parcel_col, maintain_order=True):
        parcel_id = str(pid[0]) if isinstance(pid, tuple) else str(pid)
        sub_sorted = sub.sort("_d")
        dates = sub_sorted["_d"].to_numpy()
        vals = sub_sorted[series_col].cast(pl.Float64).to_numpy()
        cls_series = sub_sorted[class_col].to_list()
        class_name = str(cls_series[0]) if cls_series else "unknown"

        _, monthly = _resample_monthly_pandas(parcel_id, dates, vals)
        if monthly.size < 3:
            continue
        # Cap max_lag al tamano efectivo de la serie - 1
        eff_lag = max(1, min(max_lag, monthly.size - 1))
        try:
            acf_vals = acf(monthly, nlags=eff_lag, fft=False, missing="drop")
        except Exception:  # noqa: BLE001, S112
            # statsmodels puede fallar con series degeneradas (varianza cero);
            # saltamos la parcela sin registrarla.
            continue
        try:
            pacf_vals = pacf(monthly, nlags=eff_lag, method="ywm")
        except Exception:  # noqa: BLE001
            pacf_vals = np.full(eff_lag + 1, np.nan, dtype=np.float64)

        for lag_i in range(eff_lag + 1):
            a = float(acf_vals[lag_i]) if lag_i < acf_vals.size else float("nan")
            p = float(pacf_vals[lag_i]) if lag_i < pacf_vals.size else float("nan")
            # Clip a [-1, 1] por seguridad numerica
            if np.isfinite(a):
                a = float(max(-1.0, min(1.0, a)))
            if np.isfinite(p):
                p = float(max(-1.0, min(1.0, p)))
            rows.append(
                {
                    "parcel_id": parcel_id,
                    "class_name": class_name,
                    "lag": int(lag_i),
                    "acf": a,
                    "pacf": p,
                }
            )

    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema)


def dtw_cluster_temporal(
    df_ts: pl.DataFrame,
    n_clusters: int = 4,
    parcel_col: str = "parcel_id",
    date_col: str = "date",
    series_col: str = "ndvi",
    class_col: str = "class_name",
    sakoe_chiba_radius: int = 3,
    seed: int = 42,
) -> tuple[pl.DataFrame, Any]:
    """Clustering DTW con `tslearn.TimeSeriesKMeans` y Sakoe-Chiba band.

    Pre-filtra `class_id in [1, 18]` cuando exista la columna. Cada serie
    NDVI se resamplea mensualmente, se interpola y se z-normaliza por parcela
    antes del fit DTW. La banda de Sakoe-Chiba (`sakoe_chiba_radius`) acota
    el coste DTW a O(T*radius) en lugar de O(T^2).

    Args:
        df_ts: DataFrame Polars long con series NDVI por parcela. Columnas
            requeridas: `parcel_col, date_col, series_col, class_col`. Opcional
            `class_id` para filtrar `[1, 18]`.
        n_clusters: Numero de clusters DTW (default 4).
        parcel_col: Columna identificador de parcela.
        date_col: Columna fecha.
        series_col: Columna NDVI.
        class_col: Columna clase (se preserva en output).
        sakoe_chiba_radius: Radio de la banda de Sakoe-Chiba (default 3).
        seed: Semilla para reproducibilidad.

    Returns:
        Tupla `(df_with_cluster, fitted_model)` donde `df_with_cluster` tiene
        columnas `parcel_id, class_name, cluster_id`, y `fitted_model` es el
        `TimeSeriesKMeans` ajustado (con `cluster_centers_` accesible). Si
        `tslearn` no esta instalado o no hay suficientes series, devuelve
        DataFrame vacio + `None`.
    """
    schema = {
        "parcel_id": pl.Utf8,
        "class_name": pl.Utf8,
        "cluster_id": pl.Int64,
    }
    empty = pl.DataFrame(schema=schema)
    if df_ts.is_empty():
        return empty, None
    required = {parcel_col, date_col, series_col, class_col}
    if not required.issubset(set(df_ts.columns)):
        return empty, None

    try:
        from tslearn.clustering import TimeSeriesKMeans
        from tslearn.utils import to_time_series_dataset
    except ImportError:  # pragma: no cover
        return empty, None

    # Compat tslearn 0.6.3 + scikit-learn >= 1.6: sklearn renombro el kwarg
    # `force_all_finite` a `ensure_all_finite`. tslearn aun lo invoca con el
    # nombre viejo. El shim se aplica una sola vez por proceso.
    try:
        import inspect

        import sklearn.utils.validation as _skv

        _check_array_sig = inspect.signature(_skv.check_array)
        if "force_all_finite" not in _check_array_sig.parameters:
            _orig_check_array = _skv.check_array

            def _check_array_compat(*args: Any, **kwargs: Any) -> Any:
                if "force_all_finite" in kwargs:
                    kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
                return _orig_check_array(*args, **kwargs)

            _skv.check_array = _check_array_compat
            try:  # tslearn.clustering.kmeans importa check_array por nombre
                import tslearn.clustering.kmeans as _tskm

                _tskm.check_array = _check_array_compat
            except Exception:  # noqa: BLE001, S110  # pragma: no cover
                pass
    except Exception:  # noqa: BLE001, S110  # pragma: no cover
        pass

    df = df_ts.clone()
    if "class_id" in df.columns:
        df = df.filter(pl.col("class_id").is_between(1, 18))
    df = df.select([parcel_col, date_col, series_col, class_col]).drop_nulls(
        subset=[parcel_col, date_col, series_col]
    )
    if df.is_empty():
        return empty, None

    # Normaliza date a pl.Date
    date_dtype = df.schema[date_col]
    if date_dtype == pl.Date:
        df = df.with_columns(pl.col(date_col).alias("_d"))
    elif date_dtype in (pl.Datetime, pl.Datetime("us"), pl.Datetime("ms"), pl.Datetime("ns")):
        df = df.with_columns(pl.col(date_col).cast(pl.Date).alias("_d"))
    elif date_dtype == pl.Utf8:
        try:
            df = df.with_columns(pl.col(date_col).str.to_date(strict=False).alias("_d"))
        except Exception:  # noqa: BLE001
            df = df.with_columns(pl.col(date_col).str.to_date("%Y%m%d", strict=False).alias("_d"))
    else:
        df = df.with_columns(
            pl.col(date_col).cast(pl.Utf8).str.to_date("%Y%m%d", strict=False).alias("_d")
        )
    df = df.drop_nulls(subset=["_d"]).sort([parcel_col, "_d"])
    if df.is_empty():
        return empty, None

    series_list: list[np.ndarray] = []
    parcel_ids: list[str] = []
    class_names: list[str] = []
    for pid, sub in df.group_by(parcel_col, maintain_order=True):
        parcel_id = str(pid[0]) if isinstance(pid, tuple) else str(pid)
        sub_sorted = sub.sort("_d")
        dates = sub_sorted["_d"].to_numpy()
        vals = sub_sorted[series_col].cast(pl.Float64).to_numpy()
        cls_series = sub_sorted[class_col].to_list()
        class_name = str(cls_series[0]) if cls_series else "unknown"

        _, monthly = _resample_monthly_pandas(parcel_id, dates, vals)
        if monthly.size < 3:
            continue
        # z-score por parcela
        mean = float(np.mean(monthly))
        std = float(np.std(monthly))
        if std < _EPS:
            continue
        z = (monthly - mean) / std
        series_list.append(z.astype(np.float64))
        parcel_ids.append(parcel_id)
        class_names.append(class_name)

    if len(series_list) < n_clusters:
        return empty, None

    X = to_time_series_dataset(series_list)
    model = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric="dtw",
        metric_params={"sakoe_chiba_radius": sakoe_chiba_radius},
        random_state=seed,
        n_init=2,
        max_iter=10,
    )
    labels = model.fit_predict(X)

    out = pl.DataFrame(
        {
            "parcel_id": parcel_ids,
            "class_name": class_names,
            "cluster_id": [int(x) for x in labels],
        },
        schema=schema,
    )
    return out, model


def era5_ndvi_anomaly(
    df_era5: pl.DataFrame,
    df_ndvi_annual: pl.DataFrame,
    dry_year_percentile: float = 0.25,
) -> pl.DataFrame:
    """Cruza precipitacion anual ERA5 con NDVI maximo anual y marca anos secos.

    Args:
        df_era5: DataFrame Polars con `year, roi_name, precip_mm`.
        df_ndvi_annual: DataFrame Polars con `year, roi_name, ndvi_max`.
        dry_year_percentile: Percentil para considerar un ano como "seco"
            (default 0.25 = inferior al cuartil 1 de precipitacion historica
            por ROI).

    Returns:
        DataFrame Polars con columnas `year, roi_name, precip_mm, ndvi_max,
        ndvi_anomaly_z, is_dry_year`. La anomalia z se calcula por ROI sobre
        `ndvi_max`. Si cualquiera de los inputs esta vacio devuelve DataFrame
        vacio con esquema valido.
    """
    schema = {
        "year": pl.Int64,
        "roi_name": pl.Utf8,
        "precip_mm": pl.Float64,
        "ndvi_max": pl.Float64,
        "ndvi_anomaly_z": pl.Float64,
        "is_dry_year": pl.Boolean,
    }
    if df_era5.is_empty() or df_ndvi_annual.is_empty():
        return pl.DataFrame(schema=schema)
    required_era5 = {"year", "roi_name", "precip_mm"}
    required_ndvi = {"year", "roi_name", "ndvi_max"}
    if not required_era5.issubset(set(df_era5.columns)):
        return pl.DataFrame(schema=schema)
    if not required_ndvi.issubset(set(df_ndvi_annual.columns)):
        return pl.DataFrame(schema=schema)

    merged = df_era5.join(df_ndvi_annual, on=["year", "roi_name"], how="inner")
    if merged.is_empty():
        return pl.DataFrame(schema=schema)

    mean_expr = pl.col("ndvi_max").mean().over("roi_name")
    std_expr = pl.col("ndvi_max").std().over("roi_name").fill_null(1.0)
    dry_thr = pl.col("precip_mm").quantile(dry_year_percentile).over("roi_name")
    merged = merged.with_columns(
        [
            ((pl.col("ndvi_max") - mean_expr) / std_expr).alias("ndvi_anomaly_z"),
            (pl.col("precip_mm") <= dry_thr).alias("is_dry_year"),
        ]
    )
    return merged.select(
        [
            pl.col("year").cast(pl.Int64),
            pl.col("roi_name").cast(pl.Utf8),
            pl.col("precip_mm").cast(pl.Float64),
            pl.col("ndvi_max").cast(pl.Float64),
            pl.col("ndvi_anomaly_z").cast(pl.Float64),
            pl.col("is_dry_year").cast(pl.Boolean),
        ]
    ).sort(["roi_name", "year"])
