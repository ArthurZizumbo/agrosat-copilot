"""DataLoader comun de segmentacion densa PASTIS-R (US-025, Tarea 1).

``PASTISSegmentationDataset`` es la pieza compartida que destraba a los tres
segmentadores de EPIC 5: expone los patches Sentinel-2 multitemporales como
tensores listos para PyTorch en dos modos intercambiables sin reescribir el
pipeline de datos:

- **Modo 2D** (``collapse_time="median"|"pick"``): colapsa el eje temporal a
  un unico frame ``(10, H, W)`` para los segmentadores CNN puros
  (DeepLabv3+ MobileNetV3, U-Net, SegFormer).
- **Modo temporal** (``collapse_time=None``): submuestrea ``n_timesteps``
  fechas equiespaciadas de forma determinista y entrega ``(T_sub, 10, H, W)``
  para los segmentadores temporales (TSViT, U-TAE).

La etiqueta ``y (H, W)`` int64 da la **clase semantica por pixel**, lo que
ademas habilita la rama fenologica-contrastiva de TSViT (Wen et al. 2025): el
modelo indexa el prototipo de la clase de cada pixel directamente con ``y``,
sin necesitar ``ParcelIDs``.

Decisiones de diseno verificadas (verdad-de-tierra PASTIS-R, 31-may-2026):

- ``DATA_S2/S2_<pid>.npy`` = ``(T, 10, 128, 128)`` int16, escala ``/10000``.
- ``ANNOTATIONS/TARGET_<pid>.npy`` = ``(3, 128, 128)`` uint8; canal 0 es la
  clase semantica (0=Background, 1..18 cultivos, 19=Void).
- El split por fold es **oficial** (campo ``Fold`` por ``ID_PATCH`` en
  ``metadata.geojson``), nunca aleatorio, para evitar leakage espacial.
- La normalizacion usa ``NORM_S2_patch.json`` por fold
  (``{"Fold_N": {"mean": [10], "std": [10]}}``) si esta disponible; si no, se
  cae a la escala simple ``/10000``.
- ``metadata.geojson`` (19 MB) se lee con ``json.load`` puro (~0.1 s), nunca
  con ``geopandas.read_file`` (parsea 2433 geometrias y cuelga el proceso).

Convencion del proyecto: ``torch``/``numpy`` solo en el borde del modelo;
logging via ``structlog``; sin pandas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import structlog
import torch
from torch.utils.data import Dataset

from ml.analysis.hcat_grouping import PASTIS_CLASS_TO_HCAT_L1, hcat_group_id_map
from ml.ingest.pastis_loader import load_pastis_patch

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ROOT = _REPO_ROOT / "data" / "PASTIS-R"

#: Escala de reflectancia PASTIS-R: los int16 estan en 0..10000.
_S2_SCALE = 10000.0

#: Numero de bandas Sentinel-2 conservadas en PASTIS-R.
_N_BANDS = 10

#: Clases no agronomicas (Background, Void) que se mapean a ``ignore_index``.
_BACKGROUND_ID = 0
_VOID_ID = 19

CollapseMode = Literal["median", "pick", None]
TargetMode = Literal["semantic18", "hcat6"]


def _build_semantic18_lut(ignore_index: int) -> np.ndarray:
    """Construye la LUT ``class_id PASTIS (0..19) -> etiqueta de entrenamiento``.

    Las 18 clases agronomicas (1..18) se remapean al rango contiguo ``[0..17]``;
    Background (0) y Void (19) se mapean a ``ignore_index`` para que la perdida
    los excluya.

    Args:
        ignore_index: Valor para los pixeles ignorados (Background/Void).

    Returns:
        Array int64 de longitud 20 indexable por ``class_id``.
    """
    lut = np.full(20, ignore_index, dtype=np.int64)
    for cid in range(1, 19):
        lut[cid] = cid - 1
    return lut


def _build_hcat6_lut(ignore_index: int) -> np.ndarray:
    """Construye la LUT ``class_id PASTIS (0..19) -> grupo HCAT [0..5]``.

    Reusa el mapeo 18->6 de :mod:`ml.analysis.hcat_grouping`. Los ids de grupo
    de ``hcat_group_id_map`` viven en ``[1, 6]`` (evitan colision con la clase 0
    del baseline); aqui se desplazan a ``[0, 5]`` contiguo para indexar logits
    de segmentacion. Background/Void y cualquier clase sin grupo van a
    ``ignore_index``.

    Args:
        ignore_index: Valor para los pixeles ignorados.

    Returns:
        Array int64 de longitud 20 indexable por ``class_id``.
    """
    name_to_id = hcat_group_id_map()  # nombre -> [1..6]
    lut = np.full(20, ignore_index, dtype=np.int64)
    for cid, group_name in PASTIS_CLASS_TO_HCAT_L1.items():
        lut[cid] = name_to_id[group_name] - 1  # [1..6] -> [0..5]
    return lut


def _load_fold_index(metadata_path: Path) -> dict[str, int]:
    """Lee ``metadata.geojson`` y devuelve ``{patch_id: fold}``.

    Usa ``json.load`` puro (no ``geopandas.read_file``, que cuelga parseando
    las 2433 geometrias del archivo de 19 MB).

    Args:
        metadata_path: Ruta al ``metadata.geojson`` de PASTIS-R.

    Returns:
        Diccionario ``{patch_id (str): fold (int)}``. Vacio si no existe.
    """
    if not metadata_path.exists():
        return {}
    with metadata_path.open(encoding="utf-8") as fh:
        gj = json.load(fh)
    out: dict[str, int] = {}
    for feat in gj.get("features", []):
        props = feat.get("properties", {}) or {}
        pid_raw = feat.get("id") or props.get("ID_PATCH")
        fold_val = props.get("Fold")
        if pid_raw is None or fold_val is None:
            continue
        out[str(pid_raw)] = int(fold_val)
    return out


def _load_fold_norm_stats(
    norm_path: Path,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Lee ``NORM_S2_patch.json`` y devuelve ``{fold: (mean[10], std[10])}``.

    Estructura del archivo: ``{"Fold_N": {"mean": [10], "std": [10]}}``.

    Args:
        norm_path: Ruta al ``NORM_S2_patch.json`` de PASTIS-R.

    Returns:
        Diccionario ``{fold (int): (mean float32[10], std float32[10])}``.
        Vacio si el archivo no existe.
    """
    if not norm_path.exists():
        return {}
    with norm_path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for key, stats in raw.items():
        # key tipo "Fold_3" -> 3
        try:
            fold = int(str(key).split("_")[-1])
        except (ValueError, IndexError):
            continue
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        out[fold] = (mean, std)
    return out


def _equispaced_indices(n_available: int, n_select: int) -> np.ndarray:
    """Selecciona ``n_select`` indices equiespaciados deterministas de ``[0, n)``.

    Si hay menos fechas que las pedidas, devuelve todas las disponibles. El
    muestreo es ``np.linspace`` redondeado a entero (determinista, sin RNG),
    de modo que siempre incluye la primera y la ultima fecha y cubre la
    estacion de forma uniforme.

    Args:
        n_available: Numero de fechas disponibles en el patch (T).
        n_select: Numero de fechas a conservar (``n_timesteps``).

    Returns:
        Array int de indices unicos ordenados ascendentemente.
    """
    if n_select >= n_available:
        return np.arange(n_available)
    idx = np.linspace(0, n_available - 1, num=n_select)
    return np.unique(np.round(idx).astype(int))


class PASTISSegmentationDataset(Dataset):
    """Dataset PyTorch de segmentacion densa sobre PASTIS-R.

    Cada item es ``(x, y)`` donde ``y (128, 128)`` int64 es la clase por pixel
    (lista para indexar prototipos fenologicos) y ``x`` es:

    - ``(10, 128, 128)`` float32 en modo 2D (``collapse_time`` distinto de
      ``None``): el eje temporal se colapsa por mediana o se elige el frame
      central.
    - ``(T_sub, 10, 128, 128)`` float32 en modo temporal
      (``collapse_time=None``): ``T_sub = min(n_timesteps, T)`` fechas
      equiespaciadas deterministas.

    El split por fold es oficial (campo ``Fold`` de ``metadata.geojson``); la
    normalizacion usa ``NORM_S2_patch.json`` por fold si existe, si no escala
    ``/10000``.

    Attributes:
        root: Raiz del dataset PASTIS-R.
        folds: Folds incluidos en este split.
        n_timesteps: Numero de fechas a conservar en modo temporal.
        collapse_time: ``"median"``/``"pick"`` (2D) o ``None`` (temporal).
        target: ``"semantic18"`` (18 clases) o ``"hcat6"`` (6 grupos HCAT).
        ignore_index: Etiqueta para Background/Void y clases sin grupo.
        patch_ids: Lista ordenada de ``patch_id`` incluidos en este split.
    """

    def __init__(
        self,
        root: Path = _DEFAULT_ROOT,
        folds: Sequence[int] = (1, 2, 3),
        n_timesteps: int = 10,
        collapse_time: CollapseMode = "median",
        target: TargetMode = "semantic18",
        ignore_index: int = 255,
        seed: int = 42,
    ) -> None:
        """Inicializa el dataset filtrando los patches por fold oficial.

        Args:
            root: Raiz del dataset PASTIS-R (``data/PASTIS-R/`` por defecto).
            folds: Folds oficiales a incluir (subconjunto de 1..5).
            n_timesteps: Fechas a conservar en modo temporal (submuestreo
                equiespaciado determinista).
            collapse_time: Modo de colapso temporal. ``"median"`` y ``"pick"``
                producen ``(10, H, W)``; ``None`` produce ``(T_sub, 10, H, W)``.
            target: ``"semantic18"`` mapea a ``[0..17]``; ``"hcat6"`` remapea a
                los 6 grupos HCAT Level-1 ``[0..5]``.
            ignore_index: Valor de etiqueta para Background/Void (y clases sin
                grupo HCAT). Default 255.
            seed: Semilla para reproducibilidad. El submuestreo temporal ya es
                determinista (equiespaciado); ``seed`` se conserva para futuras
                variantes estocasticas y queda registrado.

        Raises:
            ValueError: si ``collapse_time`` o ``target`` no son validos, o si
                ``n_timesteps`` no es positivo.
            FileNotFoundError: si ``root`` no contiene ``DATA_S2/``.
        """
        if collapse_time not in ("median", "pick", None):
            raise ValueError(
                f"collapse_time invalido: {collapse_time!r}; "
                "use 'median', 'pick' o None."
            )
        if target not in ("semantic18", "hcat6"):
            raise ValueError(
                f"target invalido: {target!r}; use 'semantic18' o 'hcat6'."
            )
        if n_timesteps <= 0:
            raise ValueError(f"n_timesteps debe ser positivo, recibido {n_timesteps}.")

        self.root = Path(root)
        self.folds = tuple(int(f) for f in folds)
        self.n_timesteps = int(n_timesteps)
        self.collapse_time = collapse_time
        self.target = target
        self.ignore_index = int(ignore_index)
        self.seed = int(seed)

        s2_dir = self.root / "DATA_S2"
        if not s2_dir.exists():
            raise FileNotFoundError(f"No existe el directorio S2: {s2_dir}")

        # LUT de remapeo de clases (precomputada una sola vez).
        self._label_lut: np.ndarray = (
            _build_semantic18_lut(self.ignore_index)
            if target == "semantic18"
            else _build_hcat6_lut(self.ignore_index)
        )
        self.num_classes: int = 18 if target == "semantic18" else 6

        # Indice de folds oficial y stats de normalizacion por fold.
        fold_index = _load_fold_index(self.root / "metadata.geojson")
        self._fold_of: dict[str, int] = fold_index
        self._norm_stats = _load_fold_norm_stats(self.root / "NORM_S2_patch.json")

        # patch_ids del split = los presentes en disco cuyo fold esta en `folds`.
        wanted = set(self.folds)
        available = {p.stem.split("_", 1)[1] for p in s2_dir.glob("S2_*.npy")}
        self.patch_ids: list[str] = sorted(
            (pid for pid, fold in fold_index.items() if fold in wanted and pid in available),
            key=lambda s: int(s) if s.isdigit() else s,
        )

        logger.info(
            "pastis_seg_dataset_init",
            n_patches=len(self.patch_ids),
            folds=self.folds,
            collapse_time=str(self.collapse_time),
            target=self.target,
            num_classes=self.num_classes,
            has_norm_stats=bool(self._norm_stats),
        )

    def __len__(self) -> int:
        """Numero de patches en este split."""
        return len(self.patch_ids)

    def _normalize(self, s2: np.ndarray, fold: int | None) -> np.ndarray:
        """Normaliza el tensor S2 ``(T, 10, H, W)`` segun el fold.

        Si hay stats del fold en ``NORM_S2_patch.json`` aplica estandarizacion
        ``(x/scale - mean) / std`` por banda; si no, escala simple ``/10000``.

        Args:
            s2: Tensor int16 ``(T, 10, H, W)``.
            fold: Fold del patch (para elegir las stats) o ``None``.

        Returns:
            Tensor float32 ``(T, 10, H, W)`` normalizado.
        """
        x = s2.astype(np.float32) / _S2_SCALE
        if fold is not None and fold in self._norm_stats:
            mean, std = self._norm_stats[fold]
            # mean/std en escala de reflectancia 0..10000 -> pasar a 0..1.
            mean = (mean / _S2_SCALE).reshape(1, _N_BANDS, 1, 1)
            std = (std / _S2_SCALE).reshape(1, _N_BANDS, 1, 1)
            x = (x - mean) / np.where(std == 0.0, 1.0, std)
        return x.astype(np.float32)

    def _collapse(self, x: np.ndarray) -> np.ndarray:
        """Aplica el modo temporal configurado al tensor ``(T, 10, H, W)``.

        Args:
            x: Tensor float32 normalizado ``(T, 10, H, W)``.

        Returns:
            ``(10, H, W)`` si ``collapse_time`` es ``"median"``/``"pick"``;
            ``(T_sub, 10, H, W)`` submuestreado si es ``None``.
        """
        n_t = x.shape[0]
        if self.collapse_time == "median":
            collapsed: np.ndarray = np.median(x, axis=0)
            return collapsed.astype(np.float32)
        if self.collapse_time == "pick":
            return np.asarray(x[n_t // 2], dtype=np.float32)
        # Modo temporal: submuestreo equiespaciado determinista.
        idx = _equispaced_indices(n_t, self.n_timesteps)
        return np.asarray(x[idx], dtype=np.float32)

    def _remap_labels(self, semantic: np.ndarray) -> np.ndarray:
        """Remapea la mascara de clase PASTIS ``(H, W)`` a etiquetas de entrenamiento.

        Args:
            semantic: Mascara uint8 ``(H, W)`` con ``class_id`` en ``0..19``.

        Returns:
            Mascara int64 ``(H, W)`` en ``[0..num_classes-1]`` union
            ``{ignore_index}``.
        """
        sem = np.clip(semantic.astype(np.int64), 0, 19)
        return self._label_lut[sem]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Carga y transforma el patch ``idx`` a ``(x, y)`` tensores.

        Args:
            idx: Indice en ``self.patch_ids``.

        Returns:
            Tupla ``(x, y)``:
                - ``x``: float32 ``(10, H, W)`` (2D) o ``(T_sub, 10, H, W)``
                  (temporal).
                - ``y``: int64 ``(H, W)`` con la clase por pixel.

        Raises:
            IndexError: si ``idx`` esta fuera de rango.
        """
        if idx < 0:
            idx += len(self.patch_ids)
        if not 0 <= idx < len(self.patch_ids):
            raise IndexError(f"idx fuera de rango: {idx}")

        pid = self.patch_ids[idx]
        patch = load_pastis_patch(pid, root=self.root, load_annotations=True)
        fold = self._fold_of.get(pid)

        s2 = patch["s2"]  # (T, 10, H, W) int16
        x_norm = self._normalize(s2, fold)
        x = self._collapse(x_norm)

        semantic = patch["semantic"]
        if semantic is None:
            # Sin anotacion: todo ignorado (no deberia ocurrir en folds 1-5).
            h, w = x.shape[-2:]
            y = np.full((h, w), self.ignore_index, dtype=np.int64)
        else:
            y = self._remap_labels(semantic)

        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(
            np.ascontiguousarray(y)
        )
