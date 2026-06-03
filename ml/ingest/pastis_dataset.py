"""Dataset PyTorch denso para segmentacion semantica sobre PASTIS-R (EPIC 5).

Este modulo es el **pipeline denso compartido** que reutilizan las 6 arquitecturas
de segmentacion del Avance 4 (U-Net, DeepLabv3+, SegFormer, U-TAE, TSViT, AnySat).
Construye tensores listos para entrenamiento a partir de los patches PASTIS-R
crudos cargados por :func:`ml.ingest.pastis_loader.load_pastis_patch`.

Convenciones del equipo (comparabilidad de los 6 modelos):

- ``num_classes = 20`` (0 = background ... 19 = void).
- ``ignore_index = 19`` (void) en loss y metricas.
- Resolucion ``256x256`` (resize bilinear imagen / nearest label).
- Reduccion temporal ``median`` para modelos 2D (U-Net/DeepLabv3+/SegFormer);
  modo ``none`` (serie recortada a ``fixed_t`` frames) para modelos temporales
  (U-TAE/TSViT/AnySat).

Normalizacion por banda con las estadisticas oficiales ``NORM_S2_patch.json``
promediadas sobre los folds de entrenamiento (sin leakage del fold de test).
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import structlog
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ml.ingest.pastis_loader import (
    PASTIS_S2_BANDS,
    pastis_patch_index,
)

logger = structlog.get_logger(__name__)

__all__ = [
    "PASTIS_IGNORE_INDEX",
    "PASTIS_NUM_CLASSES",
    "PASTIS_TARGET_SIZE",
    "PASTISDataset",
    "load_norm_stats",
    "pastis_fold_split",
]

PASTIS_NUM_CLASSES: int = 20
"""Numero de clases semanticas PASTIS-R (0 background, 1-18 cultivos, 19 void)."""

PASTIS_IGNORE_INDEX: int = 19
"""Clase ``void`` ignorada en loss y metricas (convencion compartida del equipo)."""

PASTIS_TARGET_SIZE: int = 256
"""Resolucion espacial objetivo tras resize (los patches nativos son 128x128)."""

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ROOT = _REPO_ROOT / "data" / "PASTIS-R"
_N_BANDS = len(PASTIS_S2_BANDS)

TemporalReduction = Literal["median", "mean", "none"]


def load_norm_stats(
    root: Path | None = None,
    folds: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> tuple[np.ndarray, np.ndarray]:
    """Carga y promedia las estadisticas de normalizacion por banda de PASTIS-R.

    Lee ``NORM_S2_patch.json`` (un dict ``{Fold_k: {mean: [10], std: [10]}}``) y
    promedia las medias y desviaciones de los folds indicados. Para evitar
    leakage espacial se deben pasar **solo los folds de entrenamiento**.

    Args:
        root: Raiz del dataset PASTIS-R (default ``data/PASTIS-R/``).
        folds: Folds de entrenamiento sobre los que promediar (1..5).

    Returns:
        Tupla ``(mean, std)`` de arrays ``float32`` de forma ``(10,)``. Si el
        archivo no existe devuelve ``mean=0``, ``std=1`` (modo degradado, no-op).
    """
    root = root or _DEFAULT_ROOT
    norm_path = root / "NORM_S2_patch.json"
    if not norm_path.exists():
        logger.warning("pastis_norm_missing", path=str(norm_path))
        return np.zeros(_N_BANDS, dtype=np.float32), np.ones(_N_BANDS, dtype=np.float32)

    with norm_path.open(encoding="utf-8") as fh:
        stats = json.load(fh)

    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    for fold in folds:
        entry = stats.get(f"Fold_{fold}")
        if entry is None:
            continue
        means.append(np.asarray(entry["mean"], dtype=np.float32))
        stds.append(np.asarray(entry["std"], dtype=np.float32))

    if not means:
        return np.zeros(_N_BANDS, dtype=np.float32), np.ones(_N_BANDS, dtype=np.float32)

    mean = np.mean(np.stack(means, axis=0), axis=0)
    std = np.mean(np.stack(stds, axis=0), axis=0)
    # Avoid division by zero in degenerate bands.
    std = np.where(std <= 0.0, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def pastis_fold_split(
    root: Path | None = None,
    train_folds: tuple[int, ...] = (1, 2, 3),
    val_folds: tuple[int, ...] = (4,),
    test_folds: tuple[int, ...] = (5,),
) -> dict[str, list[str]]:
    """Construye el split train/val/test usando los 5 folds oficiales de PASTIS-R.

    Los folds vienen predefinidos en ``metadata.geojson`` (campo ``Fold``) y son
    espacialmente disjuntos por diseno del dataset, por lo que evitan el leakage
    espacial que prohibe la regla ML del proyecto (sin random split).

    Args:
        root: Raiz del dataset PASTIS-R.
        train_folds: Folds asignados a entrenamiento.
        val_folds: Folds asignados a validacion.
        test_folds: Folds asignados a test.

    Returns:
        Diccionario ``{"train": [...], "val": [...], "test": [...]}`` con listas
        de ``patch_id`` (str). Listas vacias si ``metadata.geojson`` no existe.
    """
    root = root or _DEFAULT_ROOT
    index = pastis_patch_index(root / "metadata.geojson")
    split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    if index.is_empty():
        logger.warning("pastis_metadata_missing", root=str(root))
        return split

    fold_to_key = {f: "train" for f in train_folds}
    fold_to_key.update({f: "val" for f in val_folds})
    fold_to_key.update({f: "test" for f in test_folds})

    for row in index.iter_rows(named=True):
        key = fold_to_key.get(int(row["Fold"]))
        if key is not None:
            split[key].append(str(row["patch_id"]))
    return split


def _resize_spatial(x: torch.Tensor, size: int, *, label: bool) -> torch.Tensor:
    """Reescala el plano espacial ``(..., H, W)`` a ``(..., size, size)``.

    Args:
        x: Tensor ``(C, H, W)`` (imagen) o ``(H, W)`` (label).
        size: Lado objetivo.
        label: Si ``True`` usa interpolacion ``nearest`` (preserva ids de clase);
            si ``False`` usa ``bilinear`` (imagen continua).

    Returns:
        Tensor reescalado con el mismo numero de dimensiones que la entrada.
    """
    if label:
        grid = x.float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        out = F.interpolate(grid, size=(size, size), mode="nearest")
        return out.squeeze(0).squeeze(0).long()
    grid = x.unsqueeze(0)  # (1, C, H, W)
    out = F.interpolate(grid, size=(size, size), mode="bilinear", align_corners=False)
    return out.squeeze(0)


def _yyyymmdd_to_doy(date_int: int) -> int:
    """Convierte una fecha ``YYYYMMDD`` a dia-del-anio (1-366).

    Los modelos temporales (U-TAE, TSViT, AnySat) esperan posiciones temporales
    como dia-del-anio, no el entero ``YYYYMMDD`` crudo que distribuye PASTIS-R.

    Args:
        date_int: Fecha como entero ``YYYYMMDD`` (ej. ``20190101``).

    Returns:
        Dia del anio en ``[1, 366]``; ``0`` si la fecha es invalida o <= 0.
    """
    if date_int <= 0:
        return 0
    year, month, day = date_int // 10000, (date_int // 100) % 100, date_int % 100
    try:
        return datetime.date(year, month, day).timetuple().tm_yday
    except ValueError:
        return 0


def _select_frames(n_t: int, fixed_t: int) -> list[int]:
    """Devuelve indices temporales para recortar/padear la serie a ``fixed_t``.

    Args:
        n_t: Numero de frames disponibles en el patch.
        fixed_t: Numero objetivo de frames.

    Returns:
        Lista de longitud ``fixed_t`` con indices en ``[0, n_t)`` (espaciado
        uniforme si ``n_t >= fixed_t``; con repeticion del ultimo si ``n_t <
        fixed_t``).
    """
    if n_t >= fixed_t:
        return np.linspace(0, n_t - 1, fixed_t).round().astype(int).tolist()
    return list(range(n_t)) + [n_t - 1] * (fixed_t - n_t)


def _load_pastis_metadata_index(root: Path) -> dict[str, dict[str, Any]]:
    """Parsea ``metadata.geojson`` una sola vez y devuelve fechas y fold por patch.

    Evita re-parsear el geojson (~19 MB) en cada ``__getitem__``, que es
    prohibitivo leyendo desde un Drive montado. Se invoca solo cuando el dataset
    necesita las fechas (modo temporal) o el fold (``return_meta``).

    Args:
        root: Raiz del dataset PASTIS-R.

    Returns:
        Diccionario ``{patch_id: {"dates": [int], "fold": int | None}}``. Vacio
        si ``metadata.geojson`` no existe.
    """
    meta_path = root / "metadata.geojson"
    if not meta_path.exists():
        return {}
    with meta_path.open(encoding="utf-8") as fh:
        md = json.load(fh)
    index: dict[str, dict[str, Any]] = {}
    for feat in md.get("features", []):
        props = feat.get("properties", {}) or {}
        pid_raw = feat.get("id") or props.get("ID_PATCH")
        if pid_raw is None:
            continue
        dates_raw = props.get("dates-S2", {})
        dates = (
            [int(v) for _, v in sorted(dates_raw.items(), key=lambda kv: int(kv[0]))]
            if isinstance(dates_raw, dict)
            else []
        )
        fold = int(props["Fold"]) if props.get("Fold") is not None else None
        index[str(pid_raw)] = {"dates": dates, "fold": fold}
    return index


class PASTISDataset(Dataset):
    """Dataset denso PASTIS-R para segmentacion semantica multitemporal.

    Cada item es un diccionario de tensores listo para alimentar un modelo de
    segmentacion. El modo ``temporal_reduction`` determina la forma de la imagen:

    - ``"median"`` / ``"mean"``: composite temporal 2D ``image (10, S, S)``,
      apropiado para CNN 2D (U-Net, DeepLabv3+, SegFormer).
    - ``"none"``: serie recortada a ``fixed_t`` frames ``image (fixed_t, 10, S, S)``
      mas ``dates (fixed_t,)``, apropiado para modelos temporales (U-TAE, TSViT,
      AnySat).

    En todos los modos ``semantic`` es ``(S, S)`` ``long`` con ids 0..19.
    """

    def __init__(
        self,
        patch_ids: Sequence[str | int],
        root: Path | None = None,
        *,
        target_size: int = PASTIS_TARGET_SIZE,
        temporal_reduction: TemporalReduction = "median",
        fixed_t: int = 10,
        norm: tuple[np.ndarray, np.ndarray] | None = None,
        num_classes: int = PASTIS_NUM_CLASSES,
        ignore_index: int = PASTIS_IGNORE_INDEX,
        return_meta: bool = False,
    ) -> None:
        """Inicializa el dataset.

        Args:
            patch_ids: Lista de identificadores de patch (de
                :func:`pastis_fold_split`).
            root: Raiz del dataset PASTIS-R.
            target_size: Lado espacial objetivo tras resize.
            temporal_reduction: ``median``, ``mean`` o ``none``.
            fixed_t: Numero de frames cuando ``temporal_reduction="none"``.
            norm: Tupla ``(mean, std)`` precomputada; si ``None`` se cargan los
                stats de todos los folds (pasar los de train para evitar leakage).
            num_classes: Numero de clases (default 20).
            ignore_index: Clase a ignorar (default 19, void).
            return_meta: Si ``True`` incluye ``patch_id`` y ``fold`` en el item.
        """
        self.patch_ids = [str(p) for p in patch_ids]
        self.root = root or _DEFAULT_ROOT
        self.target_size = target_size
        self.temporal_reduction = temporal_reduction
        self.fixed_t = fixed_t
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.return_meta = return_meta
        mean, std = norm if norm is not None else load_norm_stats(self.root)
        # (10, 1, 1) for broadcasting over (T, 10, H, W) or (10, H, W).
        self._mean = mean.reshape(_N_BANDS, 1, 1)
        self._std = std.reshape(_N_BANDS, 1, 1)
        # Parse metadata.geojson only once (not in each __getitem__) when
        # needed: dates for the temporal mode, fold for return_meta.
        needs_meta = temporal_reduction == "none" or return_meta
        self._meta_index = _load_pastis_metadata_index(self.root) if needs_meta else {}

    def __len__(self) -> int:
        """Numero de patches en el dataset."""
        return len(self.patch_ids)

    def _normalize(self, s2: np.ndarray) -> np.ndarray:
        """Normaliza por banda ``(T, 10, H, W)`` con ``(mean, std)`` del init."""
        return (s2 - self._mean[None]) / self._std[None]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Carga, normaliza y reescala un patch a tensores de entrenamiento.

        Args:
            idx: Indice del patch en ``patch_ids``.

        Returns:
            Diccionario con ``image``, ``semantic`` y, opcionalmente, ``dates``
            (modo temporal) y ``patch_id``/``fold`` (si ``return_meta``).
        """
        pid = self.patch_ids[idx]
        # Direct load of the .npy files (1 file per patch), without re-parsing
        # the ~19 MB metadata.geojson in each item.
        s2 = np.load(self.root / "DATA_S2" / f"S2_{pid}.npy").astype(np.float32)
        s2 = self._normalize(s2)  # (T, 10, 128, 128)

        tgt_path = self.root / "ANNOTATIONS" / f"TARGET_{pid}.npy"
        if tgt_path.exists():
            semantic = np.load(tgt_path)[0]  # channel 0 = semantic label
        else:
            semantic = np.zeros(s2.shape[-2:], dtype=np.uint8)
        label = torch.from_numpy(semantic.astype(np.int64))
        label = _resize_spatial(label, self.target_size, label=True)

        item: dict[str, Any] = {"semantic": label}

        if self.temporal_reduction in ("median", "mean"):
            reducer = np.median if self.temporal_reduction == "median" else np.mean
            composite = reducer(s2, axis=0).astype(np.float32)  # (10, 128, 128)
            image = _resize_spatial(torch.from_numpy(composite), self.target_size, label=False)
            item["image"] = image  # (10, S, S)
        else:
            n_t = s2.shape[0]
            frames = _select_frames(n_t, self.fixed_t)
            series = torch.from_numpy(s2[frames])  # (fixed_t, 10, 128, 128) = (N, C, H, W)
            # The series is already in (N, C, H, W) format; the spatial plane
            # is rescaled directly (each frame as an item of the "batch").
            series = F.interpolate(
                series,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )
            item["image"] = series  # (fixed_t, 10, S, S)
            dates = self._meta_index.get(pid, {}).get("dates") or [0] * n_t
            sel_dates = [int(dates[i]) if i < len(dates) else 0 for i in frames]
            # Temporal models expect day-of-year, not the raw YYYYMMDD.
            sel_doy = [_yyyymmdd_to_doy(d) for d in sel_dates]
            item["dates"] = torch.tensor(sel_doy, dtype=torch.int64)

        if self.return_meta:
            item["patch_id"] = pid
            item["fold"] = self._meta_index.get(pid, {}).get("fold")
        return item
