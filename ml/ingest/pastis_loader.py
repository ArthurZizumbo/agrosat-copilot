"""Carga de patches PASTIS-R desde disco a estructuras Numpy / Polars.

PASTIS-R (Sainte-Fare-Garnot et al. 2021) entrega Sentinel-2 multitemporal
como tensores `(T, 10, 128, 128)` int16 en `DATA_S2/S2_<patch_id>.npy`,
anotaciones panopticas `(3, 128, 128)` uint8 en `ANNOTATIONS/TARGET_<patch_id>.npy`,
metadatos por patch en `metadata.geojson` (EPSG:2154) y estadísticas
por fold en `NORM_S2_patch.json`.

Este módulo expone helpers ligeros para cargar 1 patch o iterar varios,
y para convertir un sample a `pl.DataFrame` long-format ergonómico para
EDA.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

PASTIS_S2_BANDS: list[str] = [
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
]
"""Orden canónico de las 10 bandas Sentinel-2 conservadas en PASTIS-R.

Las bandas atmosféricas B01, B09, B10 se excluyen en el dataset original.
"""

_DEFAULT_ROOT = Path("data/PASTIS-R")
_CLASS_MAP_PATH = Path("data/reference/pastis_class_mapping.json")


def _load_class_mapping(path: Path = _CLASS_MAP_PATH) -> dict[int, str]:
    """Carga la tabla de clases canónicas PASTIS-R (20 clases).

    Args:
        path: Ruta al `pastis_class_mapping.json`.

    Returns:
        Diccionario `{class_id: nombre}` con las 20 clases (0..19).
    """
    if not path.exists():
        # Mapeo fallback minimalista si el JSON no se encuentra (modo degradado)
        return {0: "Background", 19: "Void label"}
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    classes = raw.get("classes", {})
    return {int(k): v["name"] for k, v in classes.items()}


PASTIS_CLASS_MAP: dict[int, str] = _load_class_mapping()
"""Mapeo `class_id -> nombre legible` cargado desde `data/reference/pastis_class_mapping.json`."""


def load_pastis_patch(
    patch_id: str | int,
    root: Path | None = None,
    load_annotations: bool = True,
) -> dict[str, Any]:
    """Carga un patch PASTIS-R (S2 + anotaciones + fechas + fold).

    Args:
        patch_id: Identificador del patch (numérico o string sin extensión).
        root: Raíz del dataset (default `data/PASTIS-R/`).
        load_annotations: Si True intenta cargar `TARGET_<id>.npy`.

    Returns:
        Diccionario con keys:
            - `s2`: ndarray int16 shape (T, 10, 128, 128)
            - `semantic`: ndarray uint8 (128, 128) | None — canal 0 de TARGET
            - `instance`: ndarray uint8 (128, 128) | None — canal 1 de TARGET
            - `zone`: ndarray uint8 (128, 128) | None — canal 2 de TARGET
            - `dates_s2`: list[int] — fechas YYYYMMDD ordenadas
            - `patch_id`: str
            - `fold`: int (1..5) si disponible en metadata, sino None

    Raises:
        FileNotFoundError: Si no existe el archivo S2 del patch.
    """
    root = root or _DEFAULT_ROOT
    pid = str(patch_id)

    s2_path = root / "DATA_S2" / f"S2_{pid}.npy"
    if not s2_path.exists():
        raise FileNotFoundError(f"PASTIS S2 patch no encontrado: {s2_path}")
    s2 = np.load(s2_path)

    semantic = instance = zone = None
    if load_annotations:
        tgt_path = root / "ANNOTATIONS" / f"TARGET_{pid}.npy"
        if tgt_path.exists():
            tgt = np.load(tgt_path)
            # shape (3, 128, 128): semantic, instance, zone
            semantic = tgt[0]
            instance = tgt[1] if tgt.shape[0] > 1 else None
            zone = tgt[2] if tgt.shape[0] > 2 else None

    dates_s2: list[int] = []
    fold: int | None = None
    metadata_path = root / "metadata.geojson"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as fh:
            md = json.load(fh)
        for feat in md.get("features", []):
            props = feat.get("properties", {})
            if str(props.get("ID_PATCH")) == pid:
                dates_raw = props.get("dates-S2", {})
                if isinstance(dates_raw, dict):
                    dates_s2 = [
                        int(v)
                        for _, v in sorted(
                            dates_raw.items(), key=lambda kv: int(kv[0])
                        )
                    ]
                fold_val = props.get("Fold")
                if fold_val is not None:
                    fold = int(fold_val)
                break

    return {
        "s2": s2,
        "semantic": semantic,
        "instance": instance,
        "zone": zone,
        "dates_s2": dates_s2,
        "patch_id": pid,
        "fold": fold,
    }


def iter_pastis_patches(
    patch_ids: list[str | int],
    root: Path | None = None,
    load_annotations: bool = True,
) -> Iterator[dict[str, Any]]:
    """Itera sobre múltiples patches sin cargarlos todos a memoria.

    Args:
        patch_ids: Lista de identificadores.
        root: Raíz del dataset.
        load_annotations: Si carga TARGETs.

    Yields:
        Diccionario por patch con la misma estructura que `load_pastis_patch`.
    """
    for pid in patch_ids:
        try:
            yield load_pastis_patch(pid, root=root, load_annotations=load_annotations)
        except FileNotFoundError:
            continue


def pastis_to_polars(
    patch_ids: list[str | int],
    bands: list[str] | None = None,
    root: Path | None = None,
    include_labels: bool = True,
    include_dates: bool = True,
    pixel_stride: int = 1,
) -> pl.DataFrame:
    """Convierte una lista de patches PASTIS-R a un `pl.DataFrame` long-format.

    Columnas resultantes:
        `patch_id, t, date, y, x, band, value, class_id, class_name, fold`

    Para evitar OOM con N alto se admite `pixel_stride > 1` que muestrea
    1 de cada `stride` píxeles por banda y por t.

    Args:
        patch_ids: Identificadores de patches a cargar.
        bands: Subset de bandas a incluir (default todas PASTIS_S2_BANDS).
        root: Raíz del dataset.
        include_labels: Si añade columnas `class_id`, `class_name` desde TARGET[0].
        include_dates: Si añade columna `date` (YYYYMMDD) parseada del metadata.
        pixel_stride: Stride de submuestreo espacial (1 = todos los píxeles).

    Returns:
        DataFrame Polars long-format. Vacío si ningún patch se pudo cargar.
    """
    selected = bands or PASTIS_S2_BANDS
    band_idx = [PASTIS_S2_BANDS.index(b) for b in selected]
    frames: list[pl.DataFrame] = []

    for patch in iter_pastis_patches(patch_ids, root=root, load_annotations=include_labels):
        s2 = patch["s2"]
        T, _, H, W = s2.shape
        dates = patch.get("dates_s2") or [0] * T
        ys = np.arange(0, H, pixel_stride)
        xs = np.arange(0, W, pixel_stride)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        yy_flat = yy.ravel()
        xx_flat = xx.ravel()
        n_pix = yy_flat.size

        for t in range(T):
            for bi, bname in zip(band_idx, selected, strict=True):
                vals = s2[t, bi, ys[:, None], xs[None, :]].ravel().astype(np.int32)
                date_val = (
                    int(dates[t]) if include_dates and t < len(dates) else 0
                )
                cls_arr: np.ndarray | None = None
                if include_labels and patch.get("semantic") is not None:
                    cls_arr = patch["semantic"][ys[:, None], xs[None, :]].ravel()
                frame = pl.DataFrame(
                    {
                        "patch_id": [patch["patch_id"]] * n_pix,
                        "t": [t] * n_pix,
                        "date": [date_val] * n_pix,
                        "y": yy_flat,
                        "x": xx_flat,
                        "band": [bname] * n_pix,
                        "value": vals,
                        "class_id": (
                            cls_arr.astype(np.int16)
                            if cls_arr is not None
                            else np.zeros(n_pix, dtype=np.int16)
                        ),
                        "fold": [patch.get("fold") or 0] * n_pix,
                    }
                )
                frames.append(frame)

    if not frames:
        return pl.DataFrame(
            schema={
                "patch_id": pl.Utf8,
                "t": pl.Int64,
                "date": pl.Int64,
                "y": pl.Int64,
                "x": pl.Int64,
                "band": pl.Utf8,
                "value": pl.Int32,
                "class_id": pl.Int16,
                "fold": pl.Int64,
            }
        )

    df = pl.concat(frames, how="vertical_relaxed")
    if include_labels:
        name_map = PASTIS_CLASS_MAP
        df = df.with_columns(
            pl.col("class_id")
            .cast(pl.Int64)
            .replace_strict(name_map, default="unknown")
            .alias("class_name")
        )
    return df
