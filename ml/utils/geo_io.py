"""I/O geoespacial centralizado: lectura/escritura de COG/GeoTIFF (US-017+).

Mover aqui los helpers que tocan rasterio para que no se dupliquen en
`ml/farslip/`, `ml/ingest/`, `dagster_project/assets/`, etc.

API publica:
    - ``write_crop_tiff(arr, out_path)``: escribe ``(C,H,W)`` uint16 como
      GeoTIFF simple (transform identidad para fixtures sinteticos).
    - ``read_crop_tiff(path, n_expected_bands)``: lee TIFF y valida shape.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog

_log = structlog.get_logger(__name__)


def write_crop_tiff(arr: np.ndarray, out_path: Path) -> Path:
    """Escribe crop ``(C, H, W)`` uint16 como GeoTIFF sin georeferencia real.

    Para crops sinteticos / tests no necesitamos CRS. En produccion el
    builder GEE produce COGs georeferenciados; este helper aplica transform
    identidad (Affine.identity) para mantener compatibilidad con rasterio.
    Si ``rasterio`` no esta instalado, cae a ``.npy`` (path con sufijo
    cambiado) para tests CPU sin GDAL.

    Args:
        arr: array ``(C, H, W)`` dtype uint16.
        out_path: ruta destino del ``.tif``. Padre se crea automaticamente.

    Returns:
        Ruta efectivamente escrita (puede ser ``.npy`` si rasterio ausente).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import rasterio  # type: ignore[import-untyped]
        from rasterio.transform import from_origin  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover
        fallback = out_path.with_suffix(".npy")
        np.save(fallback, arr)
        return fallback
    c, h, w = arr.shape
    transform = from_origin(0, 0, 1, 1)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=c,
        dtype="uint16",
        transform=transform,
    ) as dst:
        dst.write(arr)
    return out_path


def read_crop_tiff(path: Path, n_expected_bands: int | None = None) -> np.ndarray:
    """Lee TIFF y devuelve array ``(C, H, W)``.

    Args:
        path: ruta al ``.tif``.
        n_expected_bands: si se pasa, valida que el archivo tenga ese numero
            de bandas; en caso contrario lanza ``ValueError``.

    Returns:
        ``np.ndarray`` shape ``(C, H, W)``.

    Raises:
        ImportError: si ``rasterio`` no esta instalado.
        FileNotFoundError: si ``path`` no existe.
        ValueError: si ``n_expected_bands`` no coincide.
    """
    try:
        import rasterio  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("rasterio requerido para read_crop_tiff") from exc
    if not path.exists():
        raise FileNotFoundError(f"TIFF ausente: {path}")
    with rasterio.open(path) as src:
        arr: np.ndarray = src.read()
    if n_expected_bands is not None and arr.shape[0] != n_expected_bands:
        raise ValueError(
            f"TIFF {path} tiene {arr.shape[0]} bandas; esperado {n_expected_bands}"
        )
    return np.asarray(arr)


__all__ = ["read_crop_tiff", "write_crop_tiff"]
