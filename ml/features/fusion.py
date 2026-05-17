"""Fusión multisensor a nivel parcela (US-016).

Este módulo construye el vector tabular fusionado que alimenta los baselines
RF/XGBoost (US-019/020/021) y los heads tabulares de las arquitecturas de
segmentación (EPIC 5). El vector concatena 6 bloques heterogéneos alineados
por ``(parcel_id, year)`` con un layout determinista, más un séptimo bloque
opcional (FarSLIP, 512-dim) que se incorpora vía ``LEFT JOIN`` cuando los
embeddings se entreguen en ``data/farslip/embeddings_italy.parquet``.

Layout de columnas (orden estable, downstream depende):

::

    parcel_id (i64) | year (i16) |
    ae_00 .. ae_63 (64)                                | bloque AlphaEarth
    {idx}_{stat} (17 * 5 = 85)                         | bloque índices x stats
    s1_vv_{stat} | s1_vh_{stat} (2 * 5 = 10)           | bloque Sentinel-1
    srtm_elev_mean | srtm_slope_mean | srtm_aspect_dominant (3) |
    era5_tmean_m01..m12 | era5_prec_m01..m12 (24)       |
    geom_area_ha | geom_perimeter_m | geom_elongation (3) |
    [farslip_000 .. farslip_511 (512)]                  | opcional

Decisiones técnicas (ver ``docs/us-planning/us-016.md`` §2):

- Polars 1.x ``LazyFrame`` con ``collect(engine="streaming")`` final.
- Stats temporales subset ``("mean", "std", "p25", "p50", "p95")`` (5 stats,
  no los 9 stats completos de US-015) para mantener 85 cols por bloque
  índices y privilegiar economía downstream.
- Bloque FarSLIP es opcional vía ``LEFT JOIN``. Si ``include_farslip=True``
  y ``farslip_path`` no existe, emite warning estructurado y omite el bloque
  sin fallar la build.
- Las columnas de geometría (``geom_area_ha``, ``geom_perimeter_m``,
  ``geom_elongation``) se calculan con ``GeoSeries.to_crs("EPSG:3857")`` para
  reportar unidades métricas reales (el cálculo Polsby-Popper de elongación
  es adimensional y, por construcción, ≥ 1).
- ``srtm_aspect_dominant`` se devuelve como string cardinal de 8 cuadrantes
  ``{N, NE, E, SE, S, SW, W, NW}``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

import geopandas as gpd
import numpy as np
import polars as pl
import structlog

from ml.features.spectral_indices import INDEX_NAMES

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Constantes públicas (downstream las importa para validar shape).
# ---------------------------------------------------------------------------

#: Subconjunto de estadísticos temporales aplicados al bloque índices y al
#: bloque Sentinel-1. Difiere del set de 9 stats de US-015 por economía.
FUSION_STATS: Final[tuple[str, ...]] = ("mean", "std", "p25", "p50", "p95")

#: Nombres canónicos de los 7 bloques del vector fusionado.
BLOCK_NAMES: Final[tuple[str, ...]] = (
    "alphaearth",
    "indices_stats",
    "sentinel1",
    "srtm",
    "era5_monthly",
    "geometry",
    "farslip",
)

#: Conteo esperado de columnas SIN el bloque FarSLIP (excluye `parcel_id`, `year`).
#: 64 (AE) + 85 (idx*stats) + 10 (S1) + 3 (SRTM) + 24 (ERA5) + 3 (geom) = 189.
EXPECTED_COL_COUNT_NO_FARSLIP: Final[int] = 189

#: Conteo esperado de columnas CON el bloque FarSLIP (excluye `parcel_id`, `year`).
#: 189 + 512 (FarSLIP) = 701.
EXPECTED_COL_COUNT_WITH_FARSLIP: Final[int] = EXPECTED_COL_COUNT_NO_FARSLIP + 512

#: Polarizaciones Sentinel-1 canónicas del bloque (orden fijo).
_S1_POLARIZATIONS: Final[tuple[str, ...]] = ("vv", "vh")

#: Nombres de columnas AlphaEarth (``ae_00 .. ae_63``).
AE_COLS: Final[tuple[str, ...]] = tuple(f"ae_{i:02d}" for i in range(64))

#: Default path para los embeddings FarSLIP (US-016b).
_DEFAULT_FARSLIP_PATH: Final[Path] = Path("data/farslip/embeddings_italy.parquet")


__all__ = [
    "AE_COLS",
    "BLOCK_NAMES",
    "EXPECTED_COL_COUNT_NO_FARSLIP",
    "EXPECTED_COL_COUNT_WITH_FARSLIP",
    "FUSION_STATS",
    "build_fused_features",
]


# ---------------------------------------------------------------------------
# API pública.
# ---------------------------------------------------------------------------


def build_fused_features(
    parcels: gpd.GeoDataFrame,
    year: int,
    *,
    blocks: tuple[str, ...] = BLOCK_NAMES,
    include_farslip: bool = False,
    farslip_path: str | Path | None = None,
    stats: tuple[str, ...] = FUSION_STATS,
    lazy: bool = True,
    ae_frame: pl.DataFrame | None = None,
    indices_frame: pl.DataFrame | None = None,
    s1_frame: pl.DataFrame | None = None,
    srtm_frame: pl.DataFrame | None = None,
    era5_frame: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Construye el vector de features fusionados por ``(parcel_id, year)``.

    Args:
        parcels: GeoDataFrame con columnas ``parcel_id`` (int), ``year`` (int)
            y ``geometry`` (POLYGON en EPSG:4326). Opcionalmente ``crop_class``
            y ``region``. El año debe coincidir con ``year``.
        year: Año de referencia para AlphaEarth / S1 / ERA5.
        blocks: Subconjunto de bloques a computar. Permite ablation.
            Default ``BLOCK_NAMES`` (los 7 bloques, FarSLIP solo si
            ``include_farslip=True``).
        include_farslip: Si ``True`` intenta unir el bloque FarSLIP. Si
            ``farslip_path`` no existe, emite warning y omite el bloque sin
            fallar la build.
        farslip_path: Ruta al parquet con embeddings FarSLIP. Default
            ``data/farslip/embeddings_italy.parquet``.
        stats: Stats temporales aplicados a índices y S1. Default
            :data:`FUSION_STATS`. Cambiar este parámetro rompe el contrato
            de 85 columnas del bloque índices — utilizar solo en ablation.
        lazy: Si ``True`` (default) los joins se hacen en ``LazyFrame`` con
            ``collect(engine="streaming")`` final. Si ``False`` se usa eager
            (útil para debugging).
        ae_frame: Inyección opcional del bloque AlphaEarth ya muestreado
            (testing y dependency injection desde script CLI / Dagster).
        indices_frame: Inyección opcional del bloque índices*stats.
        s1_frame: Inyección opcional del bloque Sentinel-1.
        srtm_frame: Inyección opcional del bloque SRTM.
        era5_frame: Inyección opcional del bloque ERA5 mensual.

    Returns:
        ``pl.DataFrame`` con shape ``(N, 2 + 189)`` o ``(N, 2 + 701)`` si
        FarSLIP fue incluido. Primera columna ``parcel_id`` (i64), segunda
        ``year`` (i16). El resto en el orden documentado en el módulo.

    Raises:
        ValueError: si ``parcels`` no contiene las columnas requeridas, si
            ``year`` no coincide con ``parcels['year']``, o si ``stats`` no
            es subset de los stats soportados.
        FileNotFoundError: si ``include_farslip=True`` y se pasó
            explícitamente un ``farslip_path`` que no existe (warning
            estructurado para el path por defecto, sin fallar).
    """
    _validate_parcels(parcels, year=year)
    _validate_stats(stats)

    selected_blocks = tuple(b for b in blocks if b in BLOCK_NAMES)
    block_frames: list[pl.LazyFrame] = []

    base = (
        pl.from_pandas(
            parcels[["parcel_id", "year"]].astype({"parcel_id": "int64", "year": "int16"})
        )
        .lazy()
    )

    if "alphaearth" in selected_blocks:
        block_frames.append(_build_ae_block(parcels, year=year, injected=ae_frame))
    if "indices_stats" in selected_blocks:
        block_frames.append(
            _build_indices_stats_block(parcels, year=year, stats=stats, injected=indices_frame)
        )
    if "sentinel1" in selected_blocks:
        block_frames.append(
            _build_s1_block(parcels, year=year, stats=stats, injected=s1_frame)
        )
    if "srtm" in selected_blocks:
        block_frames.append(_build_srtm_block(parcels, injected=srtm_frame))
    if "era5_monthly" in selected_blocks:
        block_frames.append(_build_era5_block(parcels, year=year, injected=era5_frame))
    if "geometry" in selected_blocks:
        block_frames.append(_build_geom_block(parcels))
    if "farslip" in selected_blocks and include_farslip:
        farslip_block = _build_farslip_block(parcels, farslip_path=farslip_path)
        if farslip_block is not None:
            block_frames.append(farslip_block)

    joined = base
    for block in block_frames:
        joined = joined.join(block, on=["parcel_id", "year"], how="left")

    result = joined.collect(engine="streaming") if lazy else joined.collect()
    # AC-12: orden estable por parcel_id para garantizar MD5 byte-exacto en
    # re-ejecuciones. El motor streaming de Polars no preserva orden tras los
    # joins lazy, por lo que el sort final es necesario para determinismo.
    return result.sort("parcel_id")


# ---------------------------------------------------------------------------
# Validadores internos.
# ---------------------------------------------------------------------------


def _validate_parcels(parcels: gpd.GeoDataFrame, *, year: int) -> None:
    """Valida que el GeoDataFrame de parcelas trae las columnas mínimas."""
    if not isinstance(parcels, gpd.GeoDataFrame):  # pragma: no cover - guard
        raise ValueError(
            f"`parcels` debe ser un geopandas.GeoDataFrame; recibido {type(parcels)!r}"
        )
    missing = [c for c in ("parcel_id", "year") if c not in parcels.columns]
    if missing:
        raise ValueError(
            f"`parcels` no contiene columnas requeridas: {missing}. "
            "Esperadas al menos: ['parcel_id', 'year', 'geometry']."
        )
    if parcels.geometry.name not in parcels.columns:
        raise ValueError(
            "`parcels` no contiene columna de geometría activa. "
            "Verifica que el GeoDataFrame tenga `geometry` o set_geometry()."
        )
    unique_years = set(int(y) for y in parcels["year"].unique().tolist())
    if unique_years and unique_years != {int(year)}:
        raise ValueError(
            f"`year={year}` no coincide con los valores únicos en parcels['year']="
            f"{sorted(unique_years)}. La fusión asume un único año por build."
        )


def _validate_stats(stats: tuple[str, ...]) -> None:
    """Valida que `stats` es un subset reconocido."""
    supported = {"mean", "std", "p25", "p50", "p75", "p95", "min", "max"}
    invalid = [s for s in stats if s not in supported]
    if invalid:
        raise ValueError(
            f"Stats no soportadas: {invalid}. Disponibles: {sorted(supported)}."
        )


# ---------------------------------------------------------------------------
# Helpers privados — un helper por bloque.
# ---------------------------------------------------------------------------


def _build_ae_block(
    parcels: gpd.GeoDataFrame,
    *,
    year: int,
    injected: pl.DataFrame | None,
) -> pl.LazyFrame:
    """Construye el bloque AlphaEarth (64 cols) por ``(parcel_id, year)``.

    Cuando ``injected`` es ``None`` y el helper GEE no devolvió datos, se
    rellenan los 64 dims con ``None`` (downstream debe imputar si lo requiere).
    """
    if injected is not None:
        df = injected
    else:
        df = _empty_ae_frame(parcels, year=year)

    expected_cols = {"parcel_id", "year", *AE_COLS}
    actual_cols = set(df.columns)
    missing_ae = expected_cols - actual_cols
    if missing_ae:
        # Inyección parcial: completamos con None preservando contrato.
        fill: dict[str, list[float | None]] = {
            c: [None] * df.height for c in sorted(missing_ae) if c in AE_COLS
        }
        if fill:
            df = df.with_columns(
                [pl.Series(name=k, values=v, dtype=pl.Float64) for k, v in fill.items()]
            )
    select_cols = ["parcel_id", "year", *AE_COLS]
    return df.select(select_cols).lazy()


def _empty_ae_frame(parcels: gpd.GeoDataFrame, *, year: int) -> pl.DataFrame:
    """Devuelve un frame AE con los 64 dims rellenos en ``None``."""
    pids = parcels["parcel_id"].astype("int64").tolist()
    n = len(pids)
    cols: dict[str, list[object]] = {
        "parcel_id": pids,
        "year": [int(year)] * n,
    }
    for c in AE_COLS:
        cols[c] = [None] * n
    schema: dict[str, pl.DataType] = {
        "parcel_id": pl.Int64(),
        "year": pl.Int16(),
    }
    for c in AE_COLS:
        schema[c] = pl.Float64()
    return pl.DataFrame(cols, schema=schema)


def _build_indices_stats_block(
    parcels: gpd.GeoDataFrame,
    *,
    year: int,
    stats: tuple[str, ...],
    injected: pl.DataFrame | None,
) -> pl.LazyFrame:
    """Construye el bloque ``{idx}_{stat}`` (5 stats x 17 índices = 85 cols).

    El orden de columnas es ``idx`` outer + ``stat`` inner (NDVI_mean,
    NDVI_std, ... NDVI_p95, NDWI_mean, ...). Cuando ``injected`` es ``None``
    el helper rellena con ``None`` preservando el contrato exacto.
    """
    expected_cols = tuple(
        f"{idx.lower()}_{stat}" for idx in INDEX_NAMES for stat in stats
    )
    if injected is not None:
        df = injected
    else:
        df = _empty_indices_frame(parcels, year=year, expected_cols=expected_cols)

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        df = df.with_columns(
            [pl.Series(name=c, values=[None] * df.height, dtype=pl.Float64) for c in missing]
        )
    select_cols = ["parcel_id", "year", *expected_cols]
    return df.select(select_cols).lazy()


def _empty_indices_frame(
    parcels: gpd.GeoDataFrame,
    *,
    year: int,
    expected_cols: tuple[str, ...],
) -> pl.DataFrame:
    """Frame índices*stats relleno con ``None``."""
    pids = parcels["parcel_id"].astype("int64").tolist()
    n = len(pids)
    cols: dict[str, list[object]] = {
        "parcel_id": pids,
        "year": [int(year)] * n,
    }
    for c in expected_cols:
        cols[c] = [None] * n
    schema: dict[str, pl.DataType] = {
        "parcel_id": pl.Int64(),
        "year": pl.Int16(),
    }
    for c in expected_cols:
        schema[c] = pl.Float64()
    return pl.DataFrame(cols, schema=schema)


def _build_s1_block(
    parcels: gpd.GeoDataFrame,
    *,
    year: int,
    stats: tuple[str, ...],
    injected: pl.DataFrame | None,
) -> pl.LazyFrame:
    """Bloque Sentinel-1 VV+VH x stats = 10 cols."""
    expected_cols = tuple(
        f"s1_{pol}_{stat}" for pol in _S1_POLARIZATIONS for stat in stats
    )
    if injected is not None:
        df = injected
    else:
        df = _empty_generic_frame(parcels, year=year, columns=expected_cols)

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        df = df.with_columns(
            [pl.Series(name=c, values=[None] * df.height, dtype=pl.Float64) for c in missing]
        )
    return df.select(["parcel_id", "year", *expected_cols]).lazy()


def _build_srtm_block(
    parcels: gpd.GeoDataFrame,
    *,
    injected: pl.DataFrame | None,
) -> pl.LazyFrame:
    """Bloque SRTM (elevación, slope, aspect dominante) = 3 cols.

    Las parcelas no tienen año asociado en SRTM (DEM estático); el helper
    sintetiza ``year`` desde la GDF para preservar el join.
    """
    expected_cols = ("srtm_elev_mean", "srtm_slope_mean", "srtm_aspect_dominant")
    year_val = int(parcels["year"].iloc[0]) if len(parcels) else 0
    if injected is not None:
        df = injected
    else:
        pids = parcels["parcel_id"].astype("int64").tolist()
        n = len(pids)
        df = pl.DataFrame(
            {
                "parcel_id": pids,
                "year": [year_val] * n,
                "srtm_elev_mean": [None] * n,
                "srtm_slope_mean": [None] * n,
                "srtm_aspect_dominant": [None] * n,
            },
            schema={
                "parcel_id": pl.Int64(),
                "year": pl.Int16(),
                "srtm_elev_mean": pl.Float64(),
                "srtm_slope_mean": pl.Float64(),
                "srtm_aspect_dominant": pl.Utf8(),
            },
        )

    # SRTM puede llegar sin year (DEM estático): completamos con el año base.
    if "year" not in df.columns:
        df = df.with_columns(pl.lit(year_val, dtype=pl.Int16).alias("year"))
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        for col in missing:
            dtype = pl.Utf8() if col.endswith("aspect_dominant") else pl.Float64()
            df = df.with_columns(pl.Series(name=col, values=[None] * df.height, dtype=dtype))
    return df.select(["parcel_id", "year", *expected_cols]).lazy()


def _build_era5_block(
    parcels: gpd.GeoDataFrame,
    *,
    year: int,
    injected: pl.DataFrame | None,
) -> pl.LazyFrame:
    """Bloque ERA5 mensual = 24 cols (tmean_m01..12 + prec_m01..12)."""
    expected_cols = tuple(
        [f"era5_tmean_m{m:02d}" for m in range(1, 13)]
        + [f"era5_prec_m{m:02d}" for m in range(1, 13)]
    )
    if injected is not None:
        df = injected
    else:
        df = _empty_generic_frame(parcels, year=year, columns=expected_cols)

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        df = df.with_columns(
            [pl.Series(name=c, values=[None] * df.height, dtype=pl.Float64) for c in missing]
        )
    return df.select(["parcel_id", "year", *expected_cols]).lazy()


def _build_geom_block(parcels: gpd.GeoDataFrame) -> pl.LazyFrame:
    """Bloque geometría = 3 cols derivadas de la geometría EPSG:4326.

    Reproyecta a EPSG:3857 (Web Mercator) para obtener áreas y perímetros
    en unidades métricas (aproximación adecuada para zonas templadas /
    Italia). ``geom_elongation = perimetro^2 / (4 * pi * area)`` es el
    inverso del Polsby-Popper compactness (1 = círculo perfecto, > 1
    formas alargadas).
    """
    if len(parcels) == 0:
        return pl.DataFrame(
            schema={
                "parcel_id": pl.Int64(),
                "year": pl.Int16(),
                "geom_area_ha": pl.Float64(),
                "geom_perimeter_m": pl.Float64(),
                "geom_elongation": pl.Float64(),
            }
        ).lazy()

    metric = parcels.to_crs("EPSG:3857")
    area_m2 = metric.geometry.area.astype("float64").to_numpy()
    perimeter_m = metric.geometry.length.astype("float64").to_numpy()
    # Evitar división por cero en geometrías degeneradas.
    safe_area = np.where(area_m2 > 0, area_m2, np.nan)
    elongation = (perimeter_m**2) / (4.0 * np.pi * safe_area)
    area_ha = area_m2 / 10_000.0

    df = pl.DataFrame(
        {
            "parcel_id": parcels["parcel_id"].astype("int64").to_numpy(),
            "year": parcels["year"].astype("int16").to_numpy(),
            "geom_area_ha": area_ha,
            "geom_perimeter_m": perimeter_m,
            "geom_elongation": elongation,
        },
        schema={
            "parcel_id": pl.Int64(),
            "year": pl.Int16(),
            "geom_area_ha": pl.Float64(),
            "geom_perimeter_m": pl.Float64(),
            "geom_elongation": pl.Float64(),
        },
    )
    return df.lazy()


def _build_farslip_block(
    parcels: gpd.GeoDataFrame,
    *,
    farslip_path: str | Path | None,
) -> pl.LazyFrame | None:
    """Lee y prepara el bloque FarSLIP (512 cols) para ``LEFT JOIN``.

    Si el path no existe:

    - Si el caller pasó un path explícito, lanza ``FileNotFoundError`` para
      no enmascarar errores de configuración.
    - Si se usó el default y no existe, emite warning estructurado y
      devuelve ``None`` (el bloque se omite sin fallar).
    """
    explicit_path = farslip_path is not None
    resolved = Path(farslip_path) if farslip_path is not None else _DEFAULT_FARSLIP_PATH

    if not resolved.exists():
        if explicit_path:
            raise FileNotFoundError(
                f"FarSLIP embeddings parquet no encontrado: {resolved}. "
                "Pasa `include_farslip=False` o verifica el path."
            )
        logger.warning(
            "farslip_block_skipped",
            reason="default_path_not_found",
            path=str(resolved),
            note="US-016b aún no ha entregado los embeddings",
        )
        return None

    df = pl.read_parquet(resolved)
    farslip_cols = tuple(f"farslip_{i:03d}" for i in range(512))
    missing = [c for c in farslip_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"FarSLIP parquet en {resolved} no trae las 512 columnas esperadas. "
            f"Faltan {len(missing)} cols (ej. {missing[:3]}...)."
        )
    if "parcel_id" not in df.columns:
        raise ValueError(
            f"FarSLIP parquet en {resolved} no contiene `parcel_id` para el join."
        )

    # Si el frame FarSLIP no trae `year`, inferimos el año desde parcels.
    if "year" not in df.columns:
        year_val = int(parcels["year"].iloc[0]) if len(parcels) else 0
        df = df.with_columns(pl.lit(year_val, dtype=pl.Int16).alias("year"))

    return df.select(
        [
            pl.col("parcel_id").cast(pl.Int64),
            pl.col("year").cast(pl.Int16),
            *[pl.col(c).cast(pl.Float32) for c in farslip_cols],
        ]
    ).lazy()


def _empty_generic_frame(
    parcels: gpd.GeoDataFrame,
    *,
    year: int,
    columns: Sequence[str],
) -> pl.DataFrame:
    """Frame genérico relleno con ``None`` para los nombres de columna dados."""
    pids = parcels["parcel_id"].astype("int64").tolist()
    n = len(pids)
    cols: dict[str, list[object]] = {
        "parcel_id": pids,
        "year": [int(year)] * n,
    }
    for c in columns:
        cols[c] = [None] * n
    schema: dict[str, pl.DataType] = {
        "parcel_id": pl.Int64(),
        "year": pl.Int16(),
    }
    for c in columns:
        schema[c] = pl.Float64()
    return pl.DataFrame(cols, schema=schema)
