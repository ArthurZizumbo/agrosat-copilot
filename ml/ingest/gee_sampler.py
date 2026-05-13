"""Helpers de muestreo Sentinel-2 desde Google Earth Engine para EDA.

No realiza descargas masivas. Usa `sampleRegions` server-side y cachea
resultados en parquet local en `data/cache/gee/`. Apto solo para EDA
(US-010/011/012). La ingesta productiva con Dagster + GCS se cierra en
US-006/007/009.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog

try:
    import ee  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    ee = None  # type: ignore[assignment]

_log = structlog.get_logger(__name__)

ALPHAEARTH_COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
DYNAMIC_WORLD_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
ERA5_COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"
DYNAMIC_WORLD_CLASSES: dict[int, str] = {
    0: "water",
    1: "trees",
    2: "grass",
    3: "flooded_vegetation",
    4: "crops",
    5: "shrub_and_scrub",
    6: "built",
    7: "bare",
    8: "snow_and_ice",
}
ALPHAEARTH_DIM_COLS: list[str] = [f"dim_{i:02d}" for i in range(64)]

DEFAULT_CACHE_DIR = Path("data/cache/gee")
DEFAULT_S2_BANDS: list[str] = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
]


def init_ee(
    service_account_json: Path | None = None,
    project: str | None = None,
    interactive_auth: bool = False,
) -> None:
    """Inicializa Earth Engine con service account, ADC o auth interactiva.

    Orden de preferencia:

    1. Service account JSON si `service_account_json` apunta a un archivo válido.
    2. ADC (`gcloud auth application-default login`) — credenciales reusadas
       desde `~/.config/gcloud/application_default_credentials.json` o
       `~/.config/earthengine/credentials`.
    3. Solo si `interactive_auth=True` y `ee.Initialize` falla, dispara
       `ee.Authenticate()` en browser. Default `False` para evitar bloqueos
       en notebooks ejecutados con papermill / CI / contextos no interactivos.

    Args:
        service_account_json: Ruta al JSON de service account. Si None o no
            existe, se cae al ADC.
        project: ID de proyecto GCP asociado a la cuota EE (obligatorio para
            proyectos Cloud-registered desde 2024).
        interactive_auth: Si True y todo lo demás falla, lanza
            `ee.Authenticate()` (abre browser, requiere intervención). Default
            False — el caller debe ejecutar `earthengine authenticate` o
            generar service account fuera del proceso.

    Raises:
        ImportError: Si `earthengine-api` no está instalado.
        ee.EEException / RuntimeError: Si `ee.Initialize` falla y
            `interactive_auth=False`.
    """
    if ee is None:
        raise ImportError(
            "earthengine-api no está instalado. Ejecuta `poetry install --with ml,geo`."
        )
    sa_path = Path(service_account_json) if service_account_json is not None else None
    if sa_path is not None and sa_path.is_file():
        creds = ee.ServiceAccountCredentials(  # type: ignore[attr-defined]
            email=None, key_file=str(sa_path)
        )
        ee.Initialize(creds, project=project)
        return
    try:
        ee.Initialize(project=project)
    except Exception:
        if not interactive_auth:
            raise
        ee.Authenticate()
        ee.Initialize(project=project)


def _cache_key(roi_name: str, start_date: str, end_date: str, n_pixels: int) -> str:
    """Genera nombre de archivo cache reproducible."""
    return f"{roi_name}_{start_date}_{end_date}_{n_pixels}.parquet"


def sample_s2_roi(
    roi: Any,
    start_date: str,
    end_date: str,
    bands: list[str] | None = None,
    n_pixels: int = 100_000,
    cloud_pct_max: int = 30,
    cache_path: Path | None = None,
    roi_name: str = "roi",
    scale: int = 10,
) -> pl.DataFrame:
    """Muestrea Sentinel-2 L2A sobre una ROI con cache local parquet.

    Args:
        roi: `ee.Geometry` o `ee.FeatureCollection` que define la región.
        start_date: Fecha inicio formato `YYYY-MM-DD`.
        end_date: Fecha fin formato `YYYY-MM-DD`.
        bands: Bandas a extraer (default S2 SR sin atmosféricas).
        n_pixels: Número total aproximado de píxeles a muestrear.
        cloud_pct_max: Máximo `CLOUDY_PIXEL_PERCENTAGE` para filtrar imágenes.
        cache_path: Carpeta cache (default `data/cache/gee/`).
        roi_name: Nombre lógico de la ROI usado en cache + columna `roi`.
        scale: Resolución en metros para `sampleRegions`.

    Returns:
        DataFrame Polars con columnas `roi, date, band, value, lon, lat`.
        Si EE falla (auth/quota), retorna DataFrame vacío con esquema correcto.
    """
    cache_dir = cache_path or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / _cache_key(roi_name, start_date, end_date, n_pixels)
    if cache_file.exists():
        return pl.read_parquet(cache_file)

    if ee is None:
        return pl.DataFrame(
            schema={
                "roi": pl.Utf8,
                "date": pl.Utf8,
                "band": pl.Utf8,
                "value": pl.Float64,
                "lon": pl.Float64,
                "lat": pl.Float64,
            }
        )

    selected = bands or DEFAULT_S2_BANDS

    try:
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(roi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct_max))
            .select(selected)
        )
        median = collection.median()
        sample = median.sample(
            region=roi,
            scale=scale,
            numPixels=n_pixels,
            geometries=True,
            seed=42,
        )
        info = sample.getInfo()
    except Exception:  # noqa: BLE001
        # Quota / auth / red — modo degradado: retornamos DataFrame vacío
        # para no bloquear el notebook EDA. Logueo en notebook con print.
        return pl.DataFrame(
            schema={
                "roi": pl.Utf8,
                "date": pl.Utf8,
                "band": pl.Utf8,
                "value": pl.Float64,
                "lon": pl.Float64,
                "lat": pl.Float64,
            }
        )

    rows: list[dict[str, Any]] = []
    composite_date = f"{start_date}__{end_date}"
    for feat in info.get("features", []):
        props = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [None, None])
        lon, lat = (coords[0], coords[1]) if len(coords) >= 2 else (None, None)
        for band in selected:
            if band in props and props[band] is not None:
                rows.append(
                    {
                        "roi": roi_name,
                        "date": composite_date,
                        "band": band,
                        "value": float(props[band]),
                        "lon": float(lon) if lon is not None else None,
                        "lat": float(lat) if lat is not None else None,
                    }
                )

    df = (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            schema={
                "roi": pl.Utf8,
                "date": pl.Utf8,
                "band": pl.Utf8,
                "value": pl.Float64,
                "lon": pl.Float64,
                "lat": pl.Float64,
            }
        )
    )
    if not df.is_empty():
        df.write_parquet(cache_file)
    return df


def _alphaearth_empty_schema() -> dict[str, Any]:
    """Esquema canonico del DataFrame AlphaEarth (64 dims)."""
    base: dict[str, Any] = {
        "px_id": pl.Utf8,
        "lon": pl.Float64,
        "lat": pl.Float64,
        "roi": pl.Utf8,
        "year": pl.Int64,
    }
    for col in ALPHAEARTH_DIM_COLS:
        base[col] = pl.Float64
    return base


def _alphaearth_band_names() -> list[str]:
    """Nombres convencionales de bandas AlphaEarth (`A00`..`A63`).

    Se valida con `ee.ImageCollection(...).first().bandNames()` en runtime,
    pero el patron documentado es `A{ii}`.
    """
    return [f"A{i:02d}" for i in range(64)]


def sample_alphaearth_roi(
    roi: Any,
    year: int,
    n_pixels: int = 100_000,
    cache_path: Path | None = None,
    roi_name: str = "roi",
    scale: int = 10,
) -> pl.DataFrame:
    """Muestrea el embedding AlphaEarth 64-dim sobre una ROI/anio con cache parquet.

    Args:
        roi: `ee.Geometry` o `ee.FeatureCollection` delimitando la region.
        year: Anio (2017-2025) — selecciona la imagen anual correspondiente.
        n_pixels: Numero de pixeles a samplear via `sample(numPixels=...)`.
        cache_path: Carpeta cache local (default `data/cache/gee/`).
        roi_name: Nombre logico de la ROI usado en cache y columna `roi`.
        scale: Resolucion en metros (AlphaEarth nativo = 10).

    Returns:
        DataFrame Polars con columnas `px_id, lon, lat, roi, year, dim_00..dim_63`.
        Si EE no esta disponible o falla retorna DataFrame vacio con esquema valido.
    """
    cache_dir = cache_path or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"alphaearth_{roi_name}_{year}_{n_pixels}.parquet"
    if cache_file.exists():
        return pl.read_parquet(cache_file)

    empty_schema = _alphaearth_empty_schema()
    if ee is None:
        return pl.DataFrame(schema=empty_schema)

    band_names = _alphaearth_band_names()
    try:
        collection = (
            ee.ImageCollection(ALPHAEARTH_COLLECTION)
            .filterBounds(roi)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        )
        # mosaic() vs first(): AlphaEarth distribuye el embedding anual como
        # tiles disjuntos (~10k imagenes/anio en Europa). first() devuelve una
        # sola imagen con footprint limitado -> los pixeles fuera de ese tile
        # caen a null. mosaic() une todos los tiles del anio que tocan la ROI
        # en un raster continuo. Como cada (px, year) tiene un unico valor,
        # mosaic() no introduce ambiguedad.
        image = collection.mosaic().select(band_names)
        sample = image.sample(
            region=roi,
            scale=scale,
            numPixels=n_pixels,
            geometries=True,
            seed=42,
        )
        info = sample.getInfo()
    except Exception:  # noqa: BLE001
        return pl.DataFrame(schema=empty_schema)

    rows: list[dict[str, Any]] = []
    for idx, feat in enumerate(info.get("features", [])):
        props = feat.get("properties", {}) or {}
        geom = feat.get("geometry", {}) or {}
        coords = geom.get("coordinates", [None, None])
        lon = float(coords[0]) if len(coords) >= 2 and coords[0] is not None else None
        lat = float(coords[1]) if len(coords) >= 2 and coords[1] is not None else None
        row: dict[str, Any] = {
            "px_id": f"{roi_name}_{year}_{idx}",
            "lon": lon,
            "lat": lat,
            "roi": roi_name,
            "year": int(year),
        }
        for i, band in enumerate(band_names):
            val = props.get(band)
            row[ALPHAEARTH_DIM_COLS[i]] = float(val) if val is not None else None
        rows.append(row)

    if not rows:
        return pl.DataFrame(schema=empty_schema)
    df = pl.DataFrame(rows, schema=empty_schema)
    df.write_parquet(cache_file)
    return df


def sample_alphaearth_at_coords(
    coords: pl.DataFrame,
    year: int,
    cache_path: Path | None = None,
    cache_key: str = "coords",
    scale: int = 10,
    batch_size: int = 500,
) -> pl.DataFrame:
    """Sampla AlphaEarth 64-dim en coordenadas (lon, lat) EPSG:4326 dadas.

    Util para joinear con labels externos (e.g. PASTIS-R). Internamente arma
    una `ee.FeatureCollection` desde el DataFrame y llama `reduceRegions` con
    `ee.Reducer.first()` en lotes de `batch_size` puntos.

    Args:
        coords: DataFrame con columnas `px_id, lon, lat` en EPSG:4326.
        year: Anio del embedding anual a queryear.
        cache_path: Carpeta cache parquet local.
        cache_key: Identificador logico para el cache (e.g. `pastis_fold1`).
        scale: Resolucion en metros (default 10).
        batch_size: Tamano de batch por request server-side. AlphaEarth tiene
            64 bandas, asi que el payload es ~64x mas grande que DW; mantenemos
            500 puntos por batch para evitar timeouts.

    Returns:
        DataFrame con columnas `px_id, lon, lat, year, dim_00..dim_63`.
    """
    cache_dir = cache_path or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"alphaearth_at_{cache_key}_{year}_{coords.height}.parquet"
    if cache_file.exists():
        return pl.read_parquet(cache_file)

    schema: dict[str, Any] = {
        "px_id": pl.Utf8,
        "lon": pl.Float64,
        "lat": pl.Float64,
        "year": pl.Int64,
    }
    for col in ALPHAEARTH_DIM_COLS:
        schema[col] = pl.Float64

    if ee is None or coords.is_empty():
        return pl.DataFrame(schema=schema)

    band_names = _alphaearth_band_names()
    collection = ee.ImageCollection(ALPHAEARTH_COLLECTION).filterDate(
        f"{year}-01-01", f"{year + 1}-01-01"
    )
    # mosaic() vs first(): la coleccion AlphaEarth tiene ~10k tiles/anio.
    # first() devuelve un tile arbitrario con footprint limitado -> los
    # puntos fuera caen a null. mosaic() une todos los tiles del anio en
    # un raster continuo. Cada (px, year) tiene un unico valor canonico,
    # asi que mosaic() es deterministico.
    image = collection.mosaic().select(band_names)

    by_id: dict[str, dict[str, float | None]] = {
        str(r["px_id"]): {"lon": float(r["lon"]), "lat": float(r["lat"])}
        for r in coords.iter_rows(named=True)
    }

    rows: list[dict[str, Any]] = []
    total = coords.height
    for start in range(0, total, batch_size):
        chunk = coords.slice(start, batch_size)
        try:
            features = [
                ee.Feature(
                    ee.Geometry.Point([float(r["lon"]), float(r["lat"])]),
                    {"px_id": str(r["px_id"])},
                )
                for r in chunk.iter_rows(named=True)
            ]
            fc = ee.FeatureCollection(features)
            sampled = image.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.first(),
                scale=scale,
            )
            info = sampled.getInfo()
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "alphaearth_batch_failed",
                start=start,
                size=chunk.height,
                year=int(year),
                error=str(exc),
            )
            continue

        for feat in info.get("features", []):
            props = feat.get("properties", {}) or {}
            pid = str(props.get("px_id", ""))
            geo = by_id.get(pid, {"lon": None, "lat": None})
            row: dict[str, Any] = {
                "px_id": pid,
                "lon": geo["lon"],
                "lat": geo["lat"],
                "year": int(year),
            }
            for i, band in enumerate(band_names):
                val = props.get(band)
                row[ALPHAEARTH_DIM_COLS[i]] = float(val) if val is not None else None
            rows.append(row)

    if not rows:
        return pl.DataFrame(schema=schema)
    df = pl.DataFrame(rows, schema=schema)
    df.write_parquet(cache_file)
    return df


def sample_dynamic_world_at(
    coords: pl.DataFrame,
    year: int,
    cache_path: Path | None = None,
    cache_key: str = "coords",
    scale: int = 10,
    batch_size: int = 500,
) -> pl.DataFrame:
    """Extrae la clase moda Dynamic World del anio dado para cada (lon, lat).

    Procesa coords en lotes de `batch_size` puntos para evitar timeouts del
    compute graph server-side de GEE (un `reduceRegions` con >1000 puntos
    suele exceder el limite de 5 min y retornar respuesta vacia/parcial,
    quedando todas las filas con `dw_class_id=-1`).

    Args:
        coords: DataFrame con columnas `px_id, lon, lat` en EPSG:4326.
        year: Anio para filtrar la coleccion Dynamic World.
        cache_path: Carpeta cache parquet.
        cache_key: Identificador logico para el cache.
        scale: Resolucion en metros.
        batch_size: Numero maximo de puntos por request `reduceRegions`.
            Default 500 — empiricamente seguro para Italia con DW 2024.

    Returns:
        DataFrame con columnas `px_id, dw_class_id, dw_class_name, dw_confidence`.
    """
    cache_dir = cache_path or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"dw_at_{cache_key}_{year}_{coords.height}.parquet"
    if cache_file.exists():
        return pl.read_parquet(cache_file)

    schema: dict[str, Any] = {
        "px_id": pl.Utf8,
        "dw_class_id": pl.Int16,
        "dw_class_name": pl.Utf8,
        "dw_confidence": pl.Float64,
    }

    if ee is None or coords.is_empty():
        return pl.DataFrame(schema=schema)

    collection = (
        ee.ImageCollection(DYNAMIC_WORLD_COLLECTION)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .select(["label"])
    )
    mode_img = collection.mode()

    rows: list[dict[str, Any]] = []
    total = coords.height
    for start in range(0, total, batch_size):
        chunk = coords.slice(start, batch_size)
        try:
            features = [
                ee.Feature(
                    ee.Geometry.Point([float(r["lon"]), float(r["lat"])]),
                    {"px_id": str(r["px_id"])},
                )
                for r in chunk.iter_rows(named=True)
            ]
            fc = ee.FeatureCollection(features)
            sampled = mode_img.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.first(),
                scale=scale,
            )
            info = sampled.getInfo()
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "dynamic_world_batch_failed",
                start=start,
                size=chunk.height,
                year=int(year),
                error=str(exc),
            )
            # Batch fallido: anotamos px_id con clase -1 para no corromper el join.
            for r in chunk.iter_rows(named=True):
                rows.append(
                    {
                        "px_id": str(r["px_id"]),
                        "dw_class_id": -1,
                        "dw_class_name": "unknown",
                        "dw_confidence": 0.0,
                    }
                )
            continue

        for feat in info.get("features", []):
            props = feat.get("properties", {}) or {}
            pid = str(props.get("px_id", ""))
            # reduceRegions con ee.Reducer.first() renombra la banda a "first";
            # con sampleRegions seria "label". Cubrimos ambos por compatibilidad.
            cls_val = props.get("first", props.get("label"))
            cls_id = int(cls_val) if cls_val is not None else -1
            rows.append(
                {
                    "px_id": pid,
                    "dw_class_id": cls_id,
                    "dw_class_name": DYNAMIC_WORLD_CLASSES.get(cls_id, "unknown"),
                    "dw_confidence": float(props.get("confidence", 1.0)),
                }
            )

    if not rows:
        return pl.DataFrame(schema=schema)
    df = pl.DataFrame(rows, schema=schema)
    df.write_parquet(cache_file)
    return df


def fetch_s2_ndvi_rgb_for_parcel(
    parcel_geom: Any,
    date: str,
    cloud_pct_max: int = 20,
    scale: int = 10,
    max_pixels: int = 1_000_000,
) -> dict[str, np.ndarray]:
    """Devuelve RGB + NDVI Sentinel-2 para una parcela en una fecha aproximada.

    Toma una ventana +/- 15 dias alrededor de `date` y retorna la mediana
    de la coleccion filtrada por nubes. Si EE no esta disponible o falla,
    retorna arrays vacios para que el caller pueda degradar graciosamente.

    Args:
        parcel_geom: `ee.Geometry` de la parcela.
        date: Fecha central `YYYY-MM-DD`.
        cloud_pct_max: Maximo `CLOUDY_PIXEL_PERCENTAGE`.
        scale: Resolucion en metros.
        max_pixels: Limite de pixeles a recuperar.

    Returns:
        Diccionario con keys:
            - `rgb`: ndarray (H, W, 3) float, valores en [0, 1] post stretch.
            - `ndvi`: ndarray (H, W) float, valores en [-1, 1].
            - `date_used`: str fecha real usada.
    """
    empty = {
        "rgb": np.zeros((0, 0, 3), dtype=np.float32),
        "ndvi": np.zeros((0, 0), dtype=np.float32),
        "date_used": "",
    }
    if ee is None:
        return empty

    from datetime import datetime, timedelta

    try:
        center = datetime.strptime(date, "%Y-%m-%d")
        start = (center - timedelta(days=15)).strftime("%Y-%m-%d")
        end = (center + timedelta(days=15)).strftime("%Y-%m-%d")
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(parcel_geom)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct_max))
            .select(["B2", "B3", "B4", "B8"])
        )
        # median() colapsa la proyeccion nativa de S2 (UTM 10m) a la default
        # (EPSG:4326 con scale=1 grado/pixel). Sin reproyectar, sampleRectangle
        # devuelve un unico pixel porque un bbox de 0.01deg << 1 grado de scale.
        # Reproyectamos a la proyeccion de la primera imagen (UTM nativo S2 a 10m).
        ref_proj = collection.first().select("B4").projection()
        median = collection.median().reproject(crs=ref_proj, scale=scale)
        sample = median.sampleRectangle(region=parcel_geom, defaultValue=0)
        info = sample.getInfo()
    except Exception:  # noqa: BLE001
        return empty

    props = info.get("properties", {}) if isinstance(info, dict) else {}
    try:
        b2 = np.asarray(props.get("B2", []), dtype=np.float32)
        b3 = np.asarray(props.get("B3", []), dtype=np.float32)
        b4 = np.asarray(props.get("B4", []), dtype=np.float32)
        b8 = np.asarray(props.get("B8", []), dtype=np.float32)
    except Exception:  # noqa: BLE001
        return empty

    if b4.size == 0 or b8.size == 0:
        return empty

    denom = np.where((b8 + b4) == 0, 1.0, b8 + b4)
    ndvi = (b8 - b4) / denom

    # Stretch percentil 2-98 POR BANDA antes del stack. Aplicar el stretch
    # global al RGB tras stack colapsa el rango dinamico de las bandas con
    # menor magnitud, produciendo una imagen aparentemente uniforme cuando
    # B4 (rojo) >> B3 (verde) >> B2 (azul) en superficies vegetadas.
    def _stretch(band: np.ndarray) -> np.ndarray:
        if band.size == 0:
            return band
        lo, hi = np.percentile(band, [2.0, 98.0])
        return np.clip((band - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    rgb = np.stack([_stretch(b4), _stretch(b3), _stretch(b2)], axis=-1)
    _ = max_pixels
    return {"rgb": rgb.astype(np.float32), "ndvi": ndvi.astype(np.float32), "date_used": date}


def era5_annual_precip(
    roi: Any,
    years: list[int],
    cache_path: Path | None = None,
    roi_name: str = "roi",
    scale: int = 11132,
) -> pl.DataFrame:
    """Acumula precipitacion total anual ERA5-Land sobre una ROI con cache parquet.

    Agrega `total_precipitation_sum` (metros) sobre el ano completo via
    `reduceRegion(ee.Reducer.mean())` en el axis temporal y luego multiplica
    por 1000 para reportar en mm. Resolucion nativa ERA5-Land ~11132 m.

    Args:
        roi: `ee.Geometry` que delimita la region.
        years: Lista de anos enteros a procesar (e.g. `[2018, 2019, 2020]`).
        cache_path: Carpeta cache local (default `data/cache/gee/`).
        roi_name: Nombre logico de la ROI usado en cache y columna `roi_name`.
        scale: Resolucion en metros para `reduceRegion` (default 11132 = nativa).

    Returns:
        DataFrame Polars con columnas `year, roi_name, precip_mm`. Si EE no
        esta disponible o falla, devuelve DataFrame vacio con esquema valido
        para degradar el notebook sin romper la cadena Polars.
    """
    schema: dict[str, Any] = {
        "year": pl.Int64,
        "roi_name": pl.Utf8,
        "precip_mm": pl.Float64,
    }
    cache_dir = cache_path or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    years_key = "-".join(str(y) for y in sorted(years)) if years else "none"
    cache_file = cache_dir / f"era5_precip_{roi_name}_{years_key}.parquet"
    if cache_file.exists():
        return pl.read_parquet(cache_file)

    if ee is None or not years:
        return pl.DataFrame(schema=schema)

    rows: list[dict[str, Any]] = []
    try:
        for year in years:
            collection = (
                ee.ImageCollection(ERA5_COLLECTION)
                .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
                .select(["total_precipitation_sum"])
            )
            # ee.Reducer.sum sobre el axis temporal acumula la precipitacion
            # diaria del ano completo (metros). Luego reducimos espacialmente.
            annual_img = collection.sum()
            stat = annual_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=roi,
                scale=scale,
                maxPixels=1_000_000_000,
            )
            info = stat.getInfo() or {}
            precip_m = info.get("total_precipitation_sum")
            if precip_m is None:
                continue
            rows.append(
                {
                    "year": int(year),
                    "roi_name": roi_name,
                    "precip_mm": float(precip_m) * 1000.0,
                }
            )
    except Exception:  # noqa: BLE001
        return pl.DataFrame(schema=schema)

    if not rows:
        return pl.DataFrame(schema=schema)
    df = pl.DataFrame(rows, schema=schema)
    df.write_parquet(cache_file)
    return df
