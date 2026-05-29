"""Adaptador de series crudas BreizhCrops al vector de 185 features del proyecto.

BreizhCrops entrega series temporales Sentinel-2 en **bandas crudas** y en
formato *long* (una fila por ``(parcel_id, t, band)``), mientras que el
pipeline de features del proyecto (``ml.features.temporal_features``) opera
sobre un ``xarray.DataArray`` con dims ``(time, band)`` cuyos labels de banda
son los **17 indices espectrales** ya calculados, no las bandas crudas.

Este modulo cierra esa brecha de forma identica a como PASTIS-R llega a sus
185 features, reutilizando exactamente los mismos componentes canonicos:

1. :func:`ml.features.spectral_indices.compute_index` para los 17 indices.
2. :func:`ml.features.temporal_features.extract_temporal_features` para las
   153 estadisticas + 24 columnas FFT + 8 fenologicas.

De esta forma el espacio de features de BreizhCrops y PASTIS-R es el mismo
(mismos nombres de columna, misma semantica), habilitando un transfer
*tabular directo*: entrenar XGBoost en PASTIS-R y predecir sobre BreizhCrops
sin reentrenar.

Escala de reflectancia
----------------------
Las bandas BreizhCrops L2A llegan como DN (digital numbers, rango ~0-10000),
igual que PASTIS-R crudo. ``compute_index`` espera reflectancia en [0, 1]
(ver su docstring), por lo que dividimos por ``REFLECTANCE_SCALE`` (10000)
antes de calcular indices. Es el mismo contrato que el EDA del Avance 1
documento para Sentinel-2 DN.

Salida
------
:func:`build_breizhcrops_features` devuelve un ``pl.DataFrame`` con una fila
por parcela y las mismas ~185 columnas de feature que el subset US-018 de
PASTIS-R, mas ``parcel_id``, ``year``, ``class_id`` y ``class_name``.
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

#: Factor para convertir DN Sentinel-2 (0-10000) a reflectancia [0, 1].
#: Contrato de ``compute_index`` (ver EDA Avance 1, conclusiones globales).
REFLECTANCE_SCALE: Final[float] = 10_000.0

#: Banda cruda usada como hash de id entero estable cuando el ``parcel_id``
#: de BreizhCrops no es convertible a int (no aplica aqui: los ids son
#: numericos, pero se mantiene la defensa).
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
    """Convierte el long-format de UNA parcela a un ``DataArray`` de indices.

    Pivota las bandas crudas a una matriz ``(time, band_cruda)``, las escala a
    reflectancia, calcula los 17 indices espectrales canonicos y devuelve un
    ``xarray.DataArray`` con dims ``(time, band=indices)`` y los attrs
    ``parcel_id`` / ``year`` que :func:`extract_temporal_features` exige.

    Args:
        parcel_long: Sub-DataFrame de una sola parcela con columnas
            ``t, date, band, value`` (formato de
            :func:`ml.ingest.breizhcrops_loader.breizhcrops_pixel_series`).
        parcel_id_int: Id entero de la parcela (attrs del DataArray; debe ser
            ``int`` porque ``extract_temporal_features`` hace ``int(...)``).
        year: Anio del ciclo agricola (attrs del DataArray).
        bands: Orden canonico de bandas crudas Sentinel-2 esperado en
            ``parcel_long`` (default :data:`PASTIS_S2_BANDS`).
        indices: Indices espectrales a calcular (default los 17 canonicos).
        reflectance_scale: Divisor DN -> reflectancia (default 10000).

    Returns:
        ``xarray.DataArray`` dims ``(time, band)`` con labels de banda = los
        nombres de los indices, o ``None`` si la parcela no tiene al menos
        2 pasos temporales validos (insuficiente para FFT / fenologia).
    """
    # Pivot long -> wide (time x band_cruda). Una fila por t, una col por banda.
    wide = (
        parcel_long.select("t", "date", "band", "value")
        .pivot(on="band", index=["t", "date"], values="value", aggregate_function="first")
        .sort("t")
    )
    if wide.height < 2:
        return None

    missing = [b for b in bands if b not in wide.columns]
    if missing:
        # Sin alguna banda cruda no podemos calcular el set completo de indices.
        logger.warning(
            "breizhcrops_parcel_missing_bands",
            parcel_id=parcel_id_int,
            missing=missing,
        )
        return None

    # Matriz (time, band_cruda) escalada a reflectancia.
    raw = wide.select(list(bands)).to_numpy().astype(np.float64)
    refl = raw / reflectance_scale

    # Eje temporal datetime64[ns] desde el entero YYYYMMDD.
    date_ints = wide.get_column("date").to_numpy().astype(np.int64)
    times = _yyyymmdd_to_datetime64(date_ints)

    da_bands = xr.DataArray(
        refl.astype(np.float32),
        dims=("time", "band"),
        coords={"time": times, "band": list(bands)},
    )

    # Calcula cada indice sobre la serie (time,) y los apila a (time, n_idx).
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
    """Construye el vector de 185 features por parcela desde el long-format.

    Itera por parcela, reusa :func:`pixel_series_to_index_dataarray` +
    :func:`extract_temporal_features` (el mismo pipeline que PASTIS-R) y
    concatena los resultados anexando ``class_id`` y ``class_name``.

    Args:
        pixel_series: DataFrame long-format de
            :func:`ml.ingest.breizhcrops_loader.breizhcrops_pixel_series`
            (columnas ``parcel_id, t, date, doy, band, value, class_id,
            class_name``). Puede contener varias parcelas.
        indices: Indices espectrales a calcular (default los 17 canonicos,
            que producen las mismas 185 columnas que PASTIS-R).
        reflectance_scale: Divisor DN -> reflectancia (default 10000).

    Returns:
        ``pl.DataFrame`` con una fila por parcela y columnas ``parcel_id``
        (Utf8), ``year``, ``class_id``, ``class_name`` + ~185 features.
        Vacio (con esquema minimo) si ninguna parcela produce features.
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

    # Mapa parcel_id (str) -> id entero estable y metadata de clase/anio.
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
        # El anio se deriva de la primera fecha valida de la serie.
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
# Helpers privados.
# ---------------------------------------------------------------------------


def _yyyymmdd_to_datetime64(date_ints: np.ndarray) -> np.ndarray:
    """Convierte enteros ``YYYYMMDD`` a ``datetime64[ns]``.

    Args:
        date_ints: Array de enteros ``YYYYMMDD`` (campo ``date`` del
            long-format BreizhCrops).

    Returns:
        Array ``datetime64[ns]`` del mismo largo. Las fechas invalidas (0)
        se mapean al primer dia del rango valido para no romper el eje.
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
    # Sustituye NaT por el minimo valido (defensa; BreizhCrops no trae NaT).
    if np.isnat(out).any():
        valid = out[~np.isnat(out)]
        fill = valid.min() if valid.size else np.datetime64("2017-01-01", "ns")
        out[np.isnat(out)] = fill
    return out


def _year_from_first_date(parcel_long: pl.DataFrame) -> int:
    """Deriva el anio del ciclo de la primera fecha valida de la parcela."""
    dates = parcel_long.get_column("date").to_numpy()
    valid = dates[dates > 0]
    if valid.size == 0:
        return 2017
    return int(valid.min() // 10000)
