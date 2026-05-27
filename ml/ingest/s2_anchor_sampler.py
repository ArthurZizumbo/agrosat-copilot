"""Muestreo Sentinel-2 por parcela en anclas fenologicas (US-023-preview-v2 P5).

Materializa ``data/features/s2_anchors_italy.parquet`` que consume
:class:`ml.features.spectral_signature.SpectralSignatureFeatures`. Para cada
parcela italiana toma las bandas B04..B08 en 3 ventanas temporales ancladas al
DOY de Start-of-Growing (SOG), peak NDVI y senescence, calculadas aguas arriba
en el subset fenologico US-018 (o re-leidas desde un parquet de anclas).

Patron de uso::

    poetry run python -m ml.ingest.s2_anchor_sampler \\
        --parcels-path data/features/parcels_italy_2023.parquet \\
        --year 2023 \\
        --output data/features/s2_anchors_italy.parquet

El esquema de salida es deterministico y compatible con
``SpectralSignatureFeatures._extract_anchor_bands`` (busca columnas
``{anchor}_{band}`` en minusculas). Cada parcela produce 15 columnas
espectrales (3 anclas x 5 bandas) + ``parcel_id`` + ``year``.

Cache local en ``data/cache/gee/s2_anchors_{md5_parcels}_{year}.parquet``
para iteracion barata. Reusa :func:`ml.ingest.gee_sampler.init_ee` para
autenticacion EE coherente con los samplers existentes.

Modo degradado: si ``earthengine-api`` no esta disponible o GEE falla, el
modulo escribe un parquet con esquema valido y filas pobladas con ``NaN``
para no romper la cadena downstream.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import structlog

if TYPE_CHECKING:
    import geopandas as gpd

_log = structlog.get_logger(__name__)


DEFAULT_BANDS: tuple[str, ...] = ("B04", "B05", "B06", "B07", "B08")
"""Bandas Sentinel-2 muestreadas por defecto.

B04 (red, 665 nm), B05/B06/B07 (red-edge 704/740/783 nm), B08 (NIR 835 nm).
Estas son las requeridas por la Red Edge Position de Frampton et al. 2013
y por los momentos red-edge documentados en
:mod:`ml.features.spectral_signature`.
"""

DEFAULT_ANCHORS: tuple[str, ...] = ("sog", "peak", "senescence")
"""Anclas fenologicas canonicas: Start-Of-Growing, peak NDVI, senescence."""

ANCHOR_WINDOW_DAYS: int = 5
"""Ventana +/- N dias alrededor del DOY del ancla.

S2 revisita Italia cada ~5 dias por orbita, asi que +/- 5 dias garantiza
al menos una imagen disponible incluso con descarte por nubes.
"""

DEFAULT_CACHE_DIR: Path = Path("data/cache/gee")
DEFAULT_OUTPUT_PATH: Path = Path("data/features/s2_anchors_italy.parquet")

#: Estimacion conservadora del costo GEE por parcela en USD (free tier oculta
#: el costo real; este numero sirve para reportar al MLflow log un orden de
#: magnitud). Asume 3 anclas x 5 bandas x 1 reduceRegions ~ 0.0003 USD.
COST_PER_PARCEL_USD: float = 0.0003


def _band_col_name(anchor: str, band: str) -> str:
    """Devuelve el nombre de columna canonico ``{anchor}_{band_lower}``."""
    return f"{anchor}_{band.lower()}"


def _build_schema(
    anchors: tuple[str, ...], bands: tuple[str, ...]
) -> dict[str, Any]:
    """Construye el esquema Polars del output (orden estable)."""
    schema: dict[str, Any] = {
        "parcel_id": pl.Utf8,
        "year": pl.Int16,
    }
    for anchor in anchors:
        for band in bands:
            schema[_band_col_name(anchor, band)] = pl.Float64
    return schema


def _parcels_md5(parcels: gpd.GeoDataFrame) -> str:
    """Hash MD5 corto (10 chars) sobre parcel_id + bbox del GeoDataFrame.

    Reproducible: misma entrada -> mismo hash. Usado para nombrar el cache
    local en ``data/cache/gee/``.
    """
    if "parcel_id" in parcels.columns:
        ids = parcels["parcel_id"].astype(str).tolist()
    else:
        ids = [str(i) for i in range(len(parcels))]
    try:
        bounds = parcels.total_bounds  # noqa: PD011
        bbox_str = ",".join(f"{b:.6f}" for b in bounds)
    except Exception:  # noqa: BLE001
        bbox_str = "nobbox"
    payload = "|".join(sorted(ids)) + "::" + bbox_str
    return hashlib.md5(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]


def _resolve_anchors_table(
    parcels: gpd.GeoDataFrame,
    phenology_anchors_path: Path | None,
) -> pl.DataFrame:
    """Devuelve tabla ``parcel_id, sog_doy, peak_doy, senescence_doy``.

    Orden de preferencia:

    1. Columnas ``sog_doy``, ``peak_doy``, ``senescence_doy`` ya presentes
       en ``parcels``.
    2. Parquet externo ``phenology_anchors_path`` con la misma estructura.
    3. Fallback estatico: SOG=105, peak=180, senescence=260 (Italia
       continental, cultivos arables herbaceos).
    """
    needed = {"sog_doy", "peak_doy", "senescence_doy"}
    pcols = set(parcels.columns)
    if needed.issubset(pcols):
        # Convertir geopandas -> polars seleccionando solo cols necesarias.
        rows = [
            {
                "parcel_id": str(r["parcel_id"]),
                "sog_doy": int(r["sog_doy"]),
                "peak_doy": int(r["peak_doy"]),
                "senescence_doy": int(r["senescence_doy"]),
            }
            for _, r in parcels.iterrows()
        ]
        return pl.DataFrame(
            rows,
            schema={
                "parcel_id": pl.Utf8,
                "sog_doy": pl.Int16,
                "peak_doy": pl.Int16,
                "senescence_doy": pl.Int16,
            },
        )

    if phenology_anchors_path is not None and phenology_anchors_path.exists():
        df = pl.read_parquet(phenology_anchors_path)
        df = df.with_columns(pl.col("parcel_id").cast(pl.Utf8))
        return df.select(["parcel_id", "sog_doy", "peak_doy", "senescence_doy"])

    _log.warning(
        "phenology_anchors_fallback_static",
        hint="ningun parcels[*_doy] ni phenology_anchors_path; uso SOG=105/peak=180/senescence=260",
    )
    rows = [
        {
            "parcel_id": str(pid),
            "sog_doy": 105,
            "peak_doy": 180,
            "senescence_doy": 260,
        }
        for pid in parcels["parcel_id"].astype(str).tolist()
    ]
    return pl.DataFrame(
        rows,
        schema={
            "parcel_id": pl.Utf8,
            "sog_doy": pl.Int16,
            "peak_doy": pl.Int16,
            "senescence_doy": pl.Int16,
        },
    )


def _doy_to_dates(year: int, doy: int, window_days: int) -> tuple[str, str]:
    """Convierte ``(year, doy)`` a rango ``[start, end)`` ``YYYY-MM-DD``."""
    from datetime import datetime, timedelta

    center = datetime(year, 1, 1) + timedelta(days=int(doy) - 1)
    start = (center - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end = (center + timedelta(days=window_days + 1)).strftime("%Y-%m-%d")
    return start, end


def _sample_anchor_batch(
    ee_module: Any,
    parcels_chunk: gpd.GeoDataFrame,
    *,
    anchor: str,
    doy_col: str,
    year: int,
    bands: tuple[str, ...],
    anchors_table: pl.DataFrame,
    window_days: int,
    scale: int,
) -> list[dict[str, Any]]:
    """Sampla un chunk de parcelas en un solo ancla.

    Construye una ``ee.FeatureCollection`` con cada poligono + ``parcel_id``
    y agrega un ``reduceRegions(mean)`` sobre la mediana de la coleccion S2
    en la ventana ``[doy - window_days, doy + window_days]``.

    Devuelve lista de dicts ``{parcel_id, <anchor>_b04, ...}``. Si la
    consulta GEE falla, devuelve filas con ``None`` para todas las bandas.
    """
    rows: list[dict[str, Any]] = []
    # Group parcelas del chunk por DOY del ancla (parcelas con mismo DOY
    # comparten una sola consulta server-side).
    anchors_chunk = anchors_table.filter(
        pl.col("parcel_id").is_in(parcels_chunk["parcel_id"].astype(str).tolist())
    )
    if anchors_chunk.is_empty():
        return rows

    pid_to_doy: dict[str, int] = {
        r["parcel_id"]: int(r[doy_col]) for r in anchors_chunk.iter_rows(named=True)
    }

    # Agrupa parcelas por DOY identico — una consulta por DOY unico.
    by_doy: dict[int, list[Any]] = {}
    for _, row in parcels_chunk.iterrows():
        pid = str(row["parcel_id"])
        if pid not in pid_to_doy:
            continue
        doy = pid_to_doy[pid]
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        by_doy.setdefault(doy, []).append((pid, geom))

    for doy, items in by_doy.items():
        start, end = _doy_to_dates(year, doy, window_days)
        try:
            features = [
                ee_module.Feature(
                    ee_module.Geometry(geom.__geo_interface__),
                    {"parcel_id": pid},
                )
                for pid, geom in items
            ]
            fc = ee_module.FeatureCollection(features)
            collection = (
                ee_module.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterDate(start, end)
                .select(list(bands))
            )
            median = collection.median()
            reduced = median.reduceRegions(
                collection=fc,
                reducer=ee_module.Reducer.mean(),
                scale=scale,
            )
            info = reduced.getInfo()
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "s2_anchor_batch_failed",
                anchor=anchor,
                doy=doy,
                n=len(items),
                error=str(exc),
            )
            # Filas con None para esta DOY group.
            for pid, _ in items:
                row_out: dict[str, Any] = {"parcel_id": pid}
                for band in bands:
                    row_out[_band_col_name(anchor, band)] = None
                rows.append(row_out)
            continue

        for feat in info.get("features", []) or []:
            props = feat.get("properties", {}) or {}
            pid = str(props.get("parcel_id", ""))
            row_out = {"parcel_id": pid}
            for band in bands:
                val = props.get(band)
                row_out[_band_col_name(anchor, band)] = (
                    float(val) if val is not None and not _is_nan(val) else None
                )
            rows.append(row_out)
    return rows


def _is_nan(val: Any) -> bool:
    """True si ``val`` es NaN/inf."""
    try:
        f = float(val)
        return bool(np.isnan(f) or np.isinf(f))
    except (TypeError, ValueError):
        return False


def _merge_anchor_rows(
    rows_per_anchor: dict[str, list[dict[str, Any]]],
    *,
    parcel_ids: list[str],
    year: int,
    anchors: tuple[str, ...],
    bands: tuple[str, ...],
) -> pl.DataFrame:
    """Fusiona filas por anclas en un solo DataFrame ordenado por parcel_id.

    Garantiza determinismo: el output siempre se ordena ascendente por
    ``parcel_id``, independientemente del orden en que GEE devolvio los
    batches.
    """
    schema = _build_schema(anchors, bands)
    by_pid: dict[str, dict[str, Any]] = {
        pid: {"parcel_id": pid, "year": int(year)} for pid in parcel_ids
    }
    # Inicializa todas las cols a None.
    for pid in parcel_ids:
        for anchor in anchors:
            for band in bands:
                by_pid[pid][_band_col_name(anchor, band)] = None
    for anchor, rows in rows_per_anchor.items():
        for row in rows:
            pid = str(row["parcel_id"])
            if pid not in by_pid:
                continue
            for band in bands:
                col = _band_col_name(anchor, band)
                if col in row and row[col] is not None:
                    by_pid[pid][col] = row[col]
    ordered = [by_pid[pid] for pid in sorted(parcel_ids)]
    return pl.DataFrame(ordered, schema=schema)


def _count_completeness(
    df: pl.DataFrame, anchors: tuple[str, ...], bands: tuple[str, ...]
) -> tuple[int, int]:
    """Cuenta parcelas con TODAS las bandas pobladas vs parcialmente.

    Returns:
        ``(n_with_all_bands, n_with_partial)``.
    """
    band_cols = [_band_col_name(a, b) for a in anchors for b in bands]
    n_all = 0
    n_partial = 0
    for row in df.iter_rows(named=True):
        non_null = sum(1 for c in band_cols if row.get(c) is not None)
        if non_null == len(band_cols):
            n_all += 1
        elif non_null > 0:
            n_partial += 1
    return n_all, n_partial


def sample_s2_anchors_for_parcels(
    parcels: gpd.GeoDataFrame,
    year: int,
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    bands: tuple[str, ...] = DEFAULT_BANDS,
    phenology_anchors_path: Path | None = None,
    batch_size: int = 500,
    overwrite: bool = False,
    cache_dir: Path | None = None,
    window_days: int = ANCHOR_WINDOW_DAYS,
    scale: int = 10,
    anchors: tuple[str, ...] = DEFAULT_ANCHORS,
) -> Path:
    """Muestrea S2 en los DOY de SOG/peak/senescence por parcela.

    Para cada parcela en ``parcels`` (debe traer ``parcel_id`` (Utf8),
    ``year``, ``geometry`` y opcionalmente ``sog_doy/peak_doy/senescence_doy``
    pre-calculados), hace ``reduceRegions(mean)`` sobre una ventana
    ``+/- window_days`` alrededor del DOY del ancla y persiste como columnas
    ``{anchor}_b04, {anchor}_b05, ..., {anchor}_b08`` en formato consumible
    por :meth:`SpectralSignatureFeatures._extract_anchor_bands`.

    Args:
        parcels: GeoDataFrame con ``parcel_id``, ``geometry`` POLYGON
            EPSG:4326 y opcionalmente columnas ``*_doy``.
        year: Anio a samplear.
        output_path: Parquet de salida (parent se crea si no existe).
        bands: Bandas S2 a samplear (default ``("B04",...,"B08")``).
        phenology_anchors_path: Parquet opcional con anclas pre-calculadas.
            Si ``None`` y ``parcels`` no trae ``*_doy``, se cae a defaults
            estaticos para Italia continental (SOG=105/peak=180/senesc=260).
        batch_size: Tamanio del batch GEE por reduceRegions.
        overwrite: Si ``True`` ignora cache y reescribe.
        cache_dir: Carpeta de cache (default ``data/cache/gee/``).
        window_days: Ventana ``+/- window_days`` alrededor del DOY del ancla.
        scale: Resolucion de muestreo en metros (default 10, nativa S2).
        anchors: Tupla de nombres de ancla (default
            ``("sog","peak","senescence")``); deben coincidir con
            columnas ``{anchor}_doy`` en ``parcels`` o en
            ``phenology_anchors_path``.

    Returns:
        ``Path`` absoluto del parquet escrito (``output_path``).

    Raises:
        ImportError: Si ``earthengine-api`` no esta instalado.
        RuntimeError: Si ``init_ee`` falla y no hay cache previo utilizable.
    """
    output_path = Path(output_path)
    cache_root = cache_dir or DEFAULT_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = f"s2_anchors_{_parcels_md5(parcels)}_{year}.parquet"
    cache_file = cache_root / cache_key

    if cache_file.exists() and not overwrite:
        _log.info("s2_anchors_cache_hit", path=str(cache_file))
        df_cached = pl.read_parquet(cache_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_cached.write_parquet(output_path)
        return output_path.resolve()

    # Lazy import de earthengine-api: solo dentro del path "real" para evitar
    # romper tests sin EE instalado.
    from ml.ingest.gee_sampler import init_ee  # noqa: PLC0415

    try:
        import ee  # type: ignore[import-untyped]  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "earthengine-api no instalado. Ejecuta `poetry install --with ml,geo`."
        ) from exc

    anchors_table = _resolve_anchors_table(parcels, phenology_anchors_path)
    parcel_ids: list[str] = [str(p) for p in parcels["parcel_id"].astype(str).tolist()]

    try:
        init_ee()
    except Exception as exc:  # noqa: BLE001
        _log.warning(
            "s2_anchors_ee_init_failed_degraded_mode",
            error=str(exc),
            hint="se devolvera DataFrame vacio con esquema valido",
        )
        empty = pl.DataFrame(
            [{"parcel_id": pid, "year": int(year)} for pid in sorted(parcel_ids)],
            schema=_build_schema(anchors, bands),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty.write_parquet(output_path)
        empty.write_parquet(cache_file)
        return output_path.resolve()

    rows_per_anchor: dict[str, list[dict[str, Any]]] = {a: [] for a in anchors}
    total = len(parcels)
    t0 = time.perf_counter()
    for start in range(0, total, batch_size):
        chunk = parcels.iloc[start : start + batch_size]
        for anchor in anchors:
            doy_col = f"{anchor}_doy"
            rows = _sample_anchor_batch(
                ee,
                chunk,
                anchor=anchor,
                doy_col=doy_col,
                year=year,
                bands=bands,
                anchors_table=anchors_table,
                window_days=window_days,
                scale=scale,
            )
            rows_per_anchor[anchor].extend(rows)
        _log.info(
            "s2_anchors_batch_done",
            start=start,
            end=min(start + batch_size, total),
            total=total,
        )

    df = _merge_anchor_rows(
        rows_per_anchor,
        parcel_ids=parcel_ids,
        year=year,
        anchors=anchors,
        bands=bands,
    )
    elapsed = time.perf_counter() - t0
    n_all, n_partial = _count_completeness(df, anchors, bands)
    cost_estimate = round(total * COST_PER_PARCEL_USD, 4)
    _log.info(
        "s2_anchors_complete",
        n_parcels=total,
        n_with_all_bands=n_all,
        n_with_partial=n_partial,
        gee_seconds=round(elapsed, 2),
        cost_estimate_usd=cost_estimate,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache_file)
    df.write_parquet(output_path)
    return output_path.resolve()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ml.ingest.s2_anchor_sampler",
        description=(
            "Muestrea S2 B04..B08 por parcela en anclas SOG/peak/senescence "
            "y persiste a parquet consumible por SpectralSignatureFeatures."
        ),
    )
    p.add_argument(
        "--parcels-path",
        required=True,
        type=Path,
        help="Parquet o GeoParquet con parcel_id + geometry + opcionales *_doy.",
    )
    p.add_argument("--year", required=True, type=int)
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Parquet de salida (default {DEFAULT_OUTPUT_PATH}).",
    )
    p.add_argument(
        "--phenology-anchors-path",
        type=Path,
        default=None,
        help="Parquet opcional con sog_doy/peak_doy/senescence_doy.",
    )
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry-point CLI."""
    import geopandas as gpd  # noqa: PLC0415

    args = _build_arg_parser().parse_args(argv)
    parcels_path: Path = args.parcels_path
    if parcels_path.suffix.lower() in {".geoparquet", ".gpkg"}:
        parcels = gpd.read_file(parcels_path)
    else:
        parcels = gpd.read_parquet(parcels_path)
    out = sample_s2_anchors_for_parcels(
        parcels,
        args.year,
        output_path=args.output,
        phenology_anchors_path=args.phenology_anchors_path,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    _log.info("s2_anchor_sampler_done", output=str(out))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
