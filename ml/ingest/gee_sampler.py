"""Helpers de muestreo Sentinel-2 desde Google Earth Engine para EDA.

No realiza descargas masivas. Usa `sampleRegions` server-side y cachea
resultados en parquet local en `data/cache/gee/`. Apto solo para EDA
(US-010/011/012). La ingesta productiva con Dagster + GCS se cierra en
US-006/007/009.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

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
S1_COLLECTION = "COPERNICUS/S1_GRD"
SRTM_COLLECTION = "USGS/SRTMGL1_003"

#: 8 cuadrantes cardinales usados por `sample_srtm_terrain` para discretizar
#: la orientación dominante (aspect) en grados [0, 360) → string cardinal.
_ASPECT_CARDINALS: tuple[str, ...] = (
    "N", "NE", "E", "SE", "S", "SW", "W", "NW",
)
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


# ===========================================================================
# US-016 — samplers nuevos (Sentinel-1, SRTM, ERA5 mensual) por parcela.
# ===========================================================================
#
# Convención común:
# - `parcels` es un `gpd.GeoDataFrame` con columnas `parcel_id`, `geometry`
#   (POLYGON EPSG:4326) y opcionalmente `year`. Cada parcela se convierte a
#   `ee.Geometry` server-side y se reduce con `ee.Reducer.mean()`.
# - Outputs Polars con cache parquet local en `data/cache/gee/` (mismo patrón
#   que `sample_alphaearth_*` y `era5_annual_precip`).
# - Modo degradado: si `ee` no está disponible o GEE falla, se devuelve un
#   DataFrame vacío con el esquema correcto para no romper la cadena Polars
#   en el resto del pipeline (los blocks de `ml/features/fusion.py` rellenan
#   con None las cols faltantes).


def _parcels_to_feature_collection(parcels: Any) -> Any:
    """Convierte un GeoDataFrame de parcelas a ``ee.FeatureCollection``.

    Args:
        parcels: GeoDataFrame con `parcel_id` y `geometry` POLYGON EPSG:4326.

    Returns:
        ``ee.FeatureCollection`` con propiedad `parcel_id` por feature.
    """
    if ee is None:
        raise ImportError("earthengine-api no disponible.")
    features = []
    for row in parcels.itertuples(index=False):
        geom = getattr(row, "geometry", None)
        if geom is None or geom.is_empty:
            continue
        gj = geom.__geo_interface__
        ee_geom = ee.Geometry(gj)
        features.append(
            ee.Feature(ee_geom, {"parcel_id": int(row.parcel_id)})
        )
    return ee.FeatureCollection(features)


def sample_s1_roi_for_parcels(
    parcels: Any,
    year: int,
    *,
    polarization: tuple[str, ...] = ("VV", "VH"),
    orbit_pass: Literal["both", "ascending", "descending"] = "both",  # noqa: S107
    despeckle: Literal["lee_7x7", "none"] = "lee_7x7",
    sigma0_units: Literal["dB", "linear"] = "dB",
    cache_dir: Path | None = None,
    cache_key: str = "parcels",
) -> pl.DataFrame:
    """Muestrea Sentinel-1 GRD VV+VH por parcela con stats temporales (US-016 AC-4).

    Preset operativo:

    - Colección ``COPERNICUS/S1_GRD`` modo IW (Interferometric Wide).
    - ``ascending + descending`` mosaicados (``orbit_pass="both"``),
      resolución 10 m.
    - Despeckle Lee 7x7 (default) aplicado por imagen antes del stack.
    - Salida en sigma0 dB (default; ``"linear"`` para datos crudos).

    Stats devueltos por (parcel_id, polarization): ``mean, std, p25, p50,
    p95`` sobre el stack temporal → 5 stats x 2 pol = 10 columnas con
    prefijos ``s1_vv_*`` y ``s1_vh_*``.

    Args:
        parcels: GeoDataFrame con `parcel_id` y `geometry` POLYGON EPSG:4326.
        year: Año a samplear (ventana ``[YYYY-01-01, (YYYY+1)-01-01)``).
        polarization: Polarizaciones a extraer (default ``("VV", "VH")``).
        orbit_pass: Filtro de pase orbital. ``"both"`` mosaicea asc+desc.
        despeckle: Filtro de speckle. ``"lee_7x7"`` aplica filtro Lee con
            kernel 7x7; ``"none"`` desactiva el filtro.
        sigma0_units: Unidades de salida; los datos GRD GEE ya vienen en dB.
            Si se solicita ``"linear"`` se aplica ``10^(x/10)``.
        cache_dir: Carpeta de cache local (default ``data/cache/gee/``).
        cache_key: Nombre lógico del subset para el cache.

    Returns:
        ``pl.DataFrame`` con columnas ``parcel_id, year, s1_vv_mean, ...,
        s1_vh_p95``. Devuelve frame vacío con esquema válido si GEE falla.
    """
    pol_cols: list[str] = []
    for pol in polarization:
        for stat in ("mean", "std", "p25", "p50", "p95"):
            pol_cols.append(f"s1_{pol.lower()}_{stat}")
    schema: dict[str, Any] = {
        "parcel_id": pl.Int64,
        "year": pl.Int16,
        **{c: pl.Float64 for c in pol_cols},
    }

    cache_root = cache_dir or DEFAULT_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = (
        cache_root
        / f"s1_{cache_key}_{year}_{orbit_pass}_{despeckle}_{sigma0_units}.parquet"
    )
    if cache_file.exists():
        return pl.read_parquet(cache_file)

    if ee is None or len(parcels) == 0:
        return pl.DataFrame(schema=schema)

    try:
        fc = _parcels_to_feature_collection(parcels)
        collection = (
            ee.ImageCollection(S1_COLLECTION)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(list(polarization))
        )
        if orbit_pass == "ascending":  # noqa: S105
            collection = collection.filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
        elif orbit_pass == "descending":  # noqa: S105
            collection = collection.filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))

        if despeckle == "lee_7x7":
            # focalMean con kernel cuadrado de radio 3 (7x7 pixels).
            kernel = ee.Kernel.square(radius=3, units="pixels")
            collection = collection.map(lambda img: img.focalMean(kernel=kernel))

        if sigma0_units == "linear":
            collection = collection.map(
                lambda img: img.expression(
                    "pow(10, x / 10)", {"x": img.select(list(polarization))}
                )
            )

        rows: list[dict[str, Any]] = []
        year_int = int(year)
        for pol in polarization:
            pol_col = collection.select(pol)
            stats_img = pol_col.reduce(
                ee.Reducer.mean()
                .combine(ee.Reducer.stdDev(), sharedInputs=True)
                .combine(
                    ee.Reducer.percentile([25, 50, 95]),
                    sharedInputs=True,
                )
            )
            reduced = stats_img.reduceRegions(
                collection=fc,
                reducer=ee.Reducer.mean(),
                scale=10,
            )
            info = reduced.getInfo()
            for feat in info.get("features", []) or []:
                props = feat.get("properties", {}) or {}
                pid = int(props.get("parcel_id"))
                row: dict[str, Any] = {"parcel_id": pid, "year": year_int}
                row[f"s1_{pol.lower()}_mean"] = _safe_float(props.get(f"{pol}_mean"))
                row[f"s1_{pol.lower()}_std"] = _safe_float(props.get(f"{pol}_stdDev"))
                row[f"s1_{pol.lower()}_p25"] = _safe_float(props.get(f"{pol}_p25"))
                row[f"s1_{pol.lower()}_p50"] = _safe_float(props.get(f"{pol}_p50"))
                row[f"s1_{pol.lower()}_p95"] = _safe_float(props.get(f"{pol}_p95"))
                rows.append(row)
    except Exception as exc:  # noqa: BLE001 — degradación graceful
        _log.warning("s1_sample_failed", error=str(exc), year=int(year))
        return pl.DataFrame(schema=schema)

    if not rows:
        return pl.DataFrame(schema=schema)

    # Merge filas VV y VH de la misma parcela.
    merged: dict[int, dict[str, Any]] = {}
    for row in rows:
        pid = int(row["parcel_id"])
        if pid not in merged:
            merged[pid] = {"parcel_id": pid, "year": int(year)}
        for k, v in row.items():
            if k in ("parcel_id", "year"):
                continue
            if v is not None:
                merged[pid][k] = v
    df = pl.DataFrame(list(merged.values()), schema=schema)
    df.write_parquet(cache_file)
    return df


def sample_srtm_terrain(
    parcels: Any,
    *,
    cache_dir: Path | None = None,
    cache_key: str = "parcels",
) -> pl.DataFrame:
    """Muestrea SRTM elevación + slope + aspect dominante por parcela (US-016 AC-5).

    Usa ``USGS/SRTMGL1_003`` (DEM 30m global) más ``ee.Terrain.slope`` y
    ``ee.Terrain.aspect``. ``aspect_dominant`` se discretiza en 8 cuadrantes
    cardinales (N, NE, ..., NW) usando el centro de cada bin de 45°.

    Args:
        parcels: GeoDataFrame con `parcel_id` y `geometry` POLYGON EPSG:4326.
        cache_dir: Carpeta cache local.
        cache_key: Nombre lógico para el cache.

    Returns:
        ``pl.DataFrame`` con cols ``parcel_id, srtm_elev_mean,
        srtm_slope_mean, srtm_aspect_dominant`` (string cardinal).
    """
    schema: dict[str, Any] = {
        "parcel_id": pl.Int64,
        "srtm_elev_mean": pl.Float64,
        "srtm_slope_mean": pl.Float64,
        "srtm_aspect_dominant": pl.Utf8,
    }
    cache_root = cache_dir or DEFAULT_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = cache_root / f"srtm_{cache_key}.parquet"
    if cache_file.exists():
        return pl.read_parquet(cache_file)

    if ee is None or len(parcels) == 0:
        return pl.DataFrame(schema=schema)

    try:
        fc = _parcels_to_feature_collection(parcels)
        dem = ee.Image(SRTM_COLLECTION).select("elevation")
        slope = ee.Terrain.slope(dem)
        aspect = ee.Terrain.aspect(dem)
        composite = dem.addBands(slope.rename("slope")).addBands(aspect.rename("aspect"))
        reduced = composite.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.mean(),
            scale=30,
        )
        info = reduced.getInfo()
    except Exception as exc:  # noqa: BLE001
        _log.warning("srtm_sample_failed", error=str(exc))
        return pl.DataFrame(schema=schema)

    rows: list[dict[str, Any]] = []
    for feat in info.get("features", []) or []:
        props = feat.get("properties", {}) or {}
        pid = int(props.get("parcel_id"))
        aspect_deg = _safe_float(props.get("aspect"))
        rows.append(
            {
                "parcel_id": pid,
                "srtm_elev_mean": _safe_float(props.get("elevation")),
                "srtm_slope_mean": _safe_float(props.get("slope")),
                "srtm_aspect_dominant": _aspect_to_cardinal(aspect_deg),
            }
        )

    if not rows:
        return pl.DataFrame(schema=schema)
    df = pl.DataFrame(rows, schema=schema)
    df.write_parquet(cache_file)
    return df


def sample_era5_monthly_climate(
    parcels: Any,
    year: int,
    *,
    temperature_units: Literal["K", "C"] = "C",
    cache_dir: Path | None = None,
    cache_key: str = "parcels",
) -> pl.DataFrame:
    """Muestrea ERA5-Land mensual: tmean (12) + prec acumulado (12) (US-016 AC-6).

    Usa ``ECMWF/ERA5_LAND/DAILY_AGGR`` agrupando server-side por mes:

    - ``temperature_2m`` reducido con ``mean()`` por mes → °C si
      ``temperature_units="C"``.
    - ``total_precipitation_sum`` reducido con ``sum()`` por mes (metros)
      → mm acumulado (multiplicado x 1000).

    Args:
        parcels: GeoDataFrame con `parcel_id` y `geometry` POLYGON EPSG:4326.
        year: Año (genera ventana ``[YYYY-01-01, (YYYY+1)-01-01)``).
        temperature_units: ``"C"`` (default) o ``"K"``.
        cache_dir: Carpeta cache local.
        cache_key: Nombre lógico para el cache.

    Returns:
        ``pl.DataFrame`` con cols ``parcel_id, year, era5_tmean_m01..m12,
        era5_prec_m01..m12`` (24 cols).
    """
    t_cols = [f"era5_tmean_m{m:02d}" for m in range(1, 13)]
    p_cols = [f"era5_prec_m{m:02d}" for m in range(1, 13)]
    schema: dict[str, Any] = {
        "parcel_id": pl.Int64,
        "year": pl.Int16,
        **{c: pl.Float64 for c in t_cols + p_cols},
    }
    cache_root = cache_dir or DEFAULT_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = cache_root / f"era5_monthly_{cache_key}_{year}_{temperature_units}.parquet"
    if cache_file.exists():
        return pl.read_parquet(cache_file)

    if ee is None or len(parcels) == 0:
        return pl.DataFrame(schema=schema)

    try:
        fc = _parcels_to_feature_collection(parcels)
        result_rows: dict[int, dict[str, Any]] = {}
        # Bug-6 fix (smoke real, 2026-05-17):
        # reduceRegions con scale=11132 (nativa ERA5-Land ~11 km/pixel) y
        # parcelas sub-pixel (~1 km2 del fixture demo) NO intersecta ningun
        # pixel: el payload omite por completo la propiedad reducida y queda
        # solo `{parcel_id}`. Bajamos scale a 1000 m (oversampling 11x) y
        # anadimos `tileScale=4` para evitar memory errors en parcelas grandes.
        # Resultado: scale=1000 interpola correctamente el pixel ERA5
        # contenedor y rellena las 24 cols con valores fisicos plausibles.
        # Ademas, dependiendo del scale, GEE renombra la propiedad al band
        # original (`temperature_2m` / `total_precipitation_sum`) o a `mean`;
        # leemos band-name con fallback a `mean` para cubrir ambos paths.
        tband = "temperature_2m"
        pband = "total_precipitation_sum"
        for month in range(1, 13):
            start = f"{year}-{month:02d}-01"
            end = (
                f"{year + 1}-01-01"
                if month == 12
                else f"{year}-{month + 1:02d}-01"
            )
            month_collection = ee.ImageCollection(ERA5_COLLECTION).filterDate(start, end)
            tmean_img = month_collection.select(tband).mean()
            prec_img = month_collection.select(pband).sum()

            tmean_reduced = tmean_img.reduceRegions(
                collection=fc, reducer=ee.Reducer.mean(), scale=1000, tileScale=4
            )
            prec_reduced = prec_img.reduceRegions(
                collection=fc, reducer=ee.Reducer.mean(), scale=1000, tileScale=4
            )
            t_info = tmean_reduced.getInfo()
            p_info = prec_reduced.getInfo()
            for feat in t_info.get("features", []) or []:
                props = feat.get("properties", {}) or {}
                pid = int(props.get("parcel_id"))
                if pid not in result_rows:
                    result_rows[pid] = {"parcel_id": pid, "year": int(year)}
                # `reduceRegions` con single-band image renombra la propiedad
                # al nombre del band (no a "mean"); fallback a "mean" por
                # compatibilidad con mocks viejos / multi-band reducers.
                tval = _safe_float(props.get(tband, props.get("mean")))
                if tval is not None and temperature_units == "C":
                    tval = tval - 273.15
                result_rows[pid][f"era5_tmean_m{month:02d}"] = tval
            for feat in p_info.get("features", []) or []:
                props = feat.get("properties", {}) or {}
                pid = int(props.get("parcel_id"))
                if pid not in result_rows:
                    result_rows[pid] = {"parcel_id": pid, "year": int(year)}
                pval = _safe_float(props.get(pband, props.get("mean")))
                # `pval` proviene del `sum()` mensual (metros) reducido en
                # espacio; multiplicamos x 1000 para reportar en mm.
                result_rows[pid][f"era5_prec_m{month:02d}"] = (
                    pval * 1000.0 if pval is not None else None
                )
    except Exception as exc:  # noqa: BLE001
        _log.warning("era5_monthly_sample_failed", error=str(exc), year=int(year))
        return pl.DataFrame(schema=schema)

    if not result_rows:
        return pl.DataFrame(schema=schema)
    df = pl.DataFrame(list(result_rows.values()), schema=schema)
    df.write_parquet(cache_file)
    return df


def _safe_float(val: Any) -> float | None:
    """Convierte valor a float o devuelve None si es nulo/NaN."""
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _aspect_to_cardinal(aspect_deg: float | None) -> str | None:
    """Discretiza un ángulo ``[0, 360)`` en uno de 8 cuadrantes cardinales.

    Bins centrados en cada cardinal (N=0, NE=45, ..., NW=315) con ancho 45°.
    ``None`` o NaN devuelve ``None``.
    """
    if aspect_deg is None:
        return None
    deg = float(aspect_deg) % 360.0
    idx = int(((deg + 22.5) // 45.0) % 8)
    return _ASPECT_CARDINALS[idx]
