"""Helpers de muestreo Sentinel-2 desde Google Earth Engine para EDA.

No realiza descargas masivas. Usa `sampleRegions` server-side y cachea
resultados en parquet local en `data/cache/gee/`. Apto solo para EDA
(US-010/011/012). La ingesta productiva con Dagster + GCS se cierra en
US-006/007/009.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

try:
    import ee  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    ee = None  # type: ignore[assignment]

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


def init_ee(service_account_json: Path | None = None, project: str | None = None) -> None:
    """Inicializa Earth Engine con auth interactiva o service account.

    Args:
        service_account_json: Ruta al JSON de service account. Si None,
            usa `ee.Authenticate()` interactivo en la primera ejecución.
        project: ID de proyecto GCP asociado a la cuota EE.

    Raises:
        ImportError: Si `earthengine-api` no está instalado.
    """
    if ee is None:
        raise ImportError(
            "earthengine-api no está instalado. Ejecuta `poetry install --with ml,geo`."
        )
    if service_account_json is not None:
        creds = ee.ServiceAccountCredentials(  # type: ignore[attr-defined]
            email=None, key_file=str(service_account_json)
        )
        ee.Initialize(creds, project=project)
    else:
        try:
            ee.Initialize(project=project)
        except Exception:  # noqa: BLE001
            # ee.Initialize lanza ee.EEException o EEException si no hay
            # credenciales; capturamos genérico para que la auth interactiva
            # se dispare ante cualquier fallo de inicialización.
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

    df = pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={
            "roi": pl.Utf8,
            "date": pl.Utf8,
            "band": pl.Utf8,
            "value": pl.Float64,
            "lon": pl.Float64,
            "lat": pl.Float64,
        }
    )
    if not df.is_empty():
        df.write_parquet(cache_file)
    return df
