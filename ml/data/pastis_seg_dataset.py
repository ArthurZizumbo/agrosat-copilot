"""Common PASTIS-R dense segmentation DataLoader (US-025, Task 1).

``PASTISSegmentationDataset`` is the shared piece that unblocks the three EPIC 5
segmenters: it exposes the multitemporal Sentinel-2 patches as PyTorch-ready
tensors in two interchangeable modes without rewriting the data pipeline:

- **2D mode** (``collapse_time="median"|"pick"``): collapses the temporal axis
  to a single frame ``(10, H, W)`` for the pure CNN segmenters
  (DeepLabv3+ MobileNetV3, U-Net, SegFormer).
- **Temporal mode** (``collapse_time=None``): deterministically subsamples
  ``n_timesteps`` equispaced dates and delivers ``(T_sub, 10, H, W)`` for the
  temporal segmenters (TSViT, U-TAE).

The label ``y (H, W)`` int64 gives the **per-pixel semantic class**, which also
enables the TSViT phenology-contrastive branch (Wen et al. 2025): the model
indexes each pixel's class prototype directly with ``y``, without needing
``ParcelIDs``.

Verified design decisions (PASTIS-R ground truth, 31-may-2026):

- ``DATA_S2/S2_<pid>.npy`` = ``(T, 10, 128, 128)`` int16, scale ``/10000``.
- ``ANNOTATIONS/TARGET_<pid>.npy`` = ``(3, 128, 128)`` uint8; channel 0 is the
  semantic class (0=Background, 1..18 crops, 19=Void).
- The fold split is **official** (``Fold`` field per ``ID_PATCH`` in
  ``metadata.geojson``), never random, to avoid spatial leakage.
- Normalization uses ``NORM_S2_patch.json`` per fold
  (``{"Fold_N": {"mean": [10], "std": [10]}}``) if available; otherwise it falls
  back to the simple ``/10000`` scale.
- ``metadata.geojson`` (19 MB) is read with plain ``json.load`` (~0.1 s), never
  with ``geopandas.read_file`` (parses 2433 geometries and hangs the process).

Project convention: ``torch``/``numpy`` only at the model boundary; logging via
``structlog``; no pandas.
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

#: PASTIS-R reflectance scale: the int16 values are in 0..10000.
_S2_SCALE = 10000.0

#: Number of Sentinel-2 bands kept in PASTIS-R.
_N_BANDS = 10

#: Non-agronomic classes (Background, Void) mapped to ``ignore_index``.
_BACKGROUND_ID = 0
_VOID_ID = 19

CollapseMode = Literal["median", "pick", None]
TargetMode = Literal["semantic18", "hcat6"]


def _build_semantic18_lut(ignore_index: int) -> np.ndarray:
    """Builds the LUT ``class_id PASTIS (0..19) -> training label``.

    The 18 agronomic classes (1..18) are remapped to the contiguous range
    ``[0..17]``; Background (0) and Void (19) are mapped to ``ignore_index`` so
    that the loss excludes them.

    Args:
        ignore_index: Value for the ignored pixels (Background/Void).

    Returns:
        int64 array of length 20 indexable by ``class_id``.
    """
    lut = np.full(20, ignore_index, dtype=np.int64)
    for cid in range(1, 19):
        lut[cid] = cid - 1
    return lut


def _build_hcat6_lut(ignore_index: int) -> np.ndarray:
    """Builds the LUT ``class_id PASTIS (0..19) -> HCAT group [0..5]``.

    Reuses the 18->6 mapping from :mod:`ml.analysis.hcat_grouping`. The group ids
    from ``hcat_group_id_map`` live in ``[1, 6]`` (they avoid collision with the
    baseline's class 0); here they are shifted to a contiguous ``[0, 5]`` to index
    segmentation logits. Background/Void and any class without a group go to
    ``ignore_index``.

    Args:
        ignore_index: Value for the ignored pixels.

    Returns:
        int64 array of length 20 indexable by ``class_id``.
    """
    name_to_id = hcat_group_id_map()  # name -> [1..6]
    lut = np.full(20, ignore_index, dtype=np.int64)
    for cid, group_name in PASTIS_CLASS_TO_HCAT_L1.items():
        lut[cid] = name_to_id[group_name] - 1  # [1..6] -> [0..5]
    return lut


def _load_fold_index(metadata_path: Path) -> dict[str, int]:
    """Reads ``metadata.geojson`` and returns ``{patch_id: fold}``.

    Uses plain ``json.load`` (not ``geopandas.read_file``, which hangs parsing
    the 2433 geometries of the 19 MB file).

    Args:
        metadata_path: Path to the PASTIS-R ``metadata.geojson``.

    Returns:
        Dictionary ``{patch_id (str): fold (int)}``. Empty if it does not exist.
    """
    if not metadata_path.exists():
        return {}
    with metadata_path.open(encoding="utf-8") as fh:
        gj = json.load(fh)
    out: dict[str, int] = {}
    for feat in gj.get("features", []):
        props = feat.get("properties", {}) or {}
        # `ID_PATCH` is the real patch identifier (matches the name
        # `S2_<ID_PATCH>.npy`). It is prioritized over `feat["id"]` because the
        # official Zenodo metadata uses `feat["id"]` as a sequential index
        # (0, 1, 2, ...) that does NOT match the file names; only
        # some derived metadata put the patch_id in `feat["id"]`.
        pid_raw = props.get("ID_PATCH")
        if pid_raw is None:
            pid_raw = feat.get("id")
        fold_val = props.get("Fold")
        if pid_raw is None or fold_val is None:
            continue
        out[str(pid_raw)] = int(fold_val)
    return out


def _load_fold_norm_stats(
    norm_path: Path,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Reads ``NORM_S2_patch.json`` and returns ``{fold: (mean[10], std[10])}``.

    File structure: ``{"Fold_N": {"mean": [10], "std": [10]}}``.

    Args:
        norm_path: Path to the PASTIS-R ``NORM_S2_patch.json``.

    Returns:
        Dictionary ``{fold (int): (mean float32[10], std float32[10])}``.
        Empty if the file does not exist.
    """
    if not norm_path.exists():
        return {}
    with norm_path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for key, stats in raw.items():
        # key like "Fold_3" -> 3
        try:
            fold = int(str(key).split("_")[-1])
        except (ValueError, IndexError):
            continue
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        out[fold] = (mean, std)
    return out


def _equispaced_indices(n_available: int, n_select: int) -> np.ndarray:
    """Selects ``n_select`` deterministic equispaced indices from ``[0, n)``.

    If there are fewer dates than requested, returns all available ones. The
    sampling is ``np.linspace`` rounded to integer (deterministic, no RNG), so it
    always includes the first and last date and covers the season uniformly.

    Args:
        n_available: Number of dates available in the patch (T).
        n_select: Number of dates to keep (``n_timesteps``).

    Returns:
        int array of unique indices sorted ascending.
    """
    if n_select >= n_available:
        return np.arange(n_available)
    idx = np.linspace(0, n_available - 1, num=n_select)
    return np.unique(np.round(idx).astype(int))


class PASTISSegmentationDataset(Dataset):
    """PyTorch dense segmentation dataset over PASTIS-R.

    Each item is ``(x, y)`` where ``y (128, 128)`` int64 is the per-pixel class
    (ready to index phenology prototypes) and ``x`` is:

    - ``(10, 128, 128)`` float32 in 2D mode (``collapse_time`` other than
      ``None``): the temporal axis is collapsed by median or the central frame
      is picked.
    - ``(T_sub, 10, 128, 128)`` float32 in temporal mode
      (``collapse_time=None``): ``T_sub = min(n_timesteps, T)`` deterministic
      equispaced dates.

    The fold split is official (``Fold`` field of ``metadata.geojson``);
    normalization uses ``NORM_S2_patch.json`` per fold if it exists, otherwise it
    scales by ``/10000``.

    Attributes:
        root: Root of the PASTIS-R dataset.
        folds: Folds included in this split.
        n_timesteps: Number of dates to keep in temporal mode.
        collapse_time: ``"median"``/``"pick"`` (2D) or ``None`` (temporal).
        target: ``"semantic18"`` (18 classes) or ``"hcat6"`` (6 HCAT groups).
        ignore_index: Label for Background/Void and classes without a group.
        patch_ids: Sorted list of ``patch_id`` included in this split.
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
        """Initializes the dataset filtering the patches by official fold.

        Args:
            root: Root of the PASTIS-R dataset (``data/PASTIS-R/`` by default).
            folds: Official folds to include (subset of 1..5).
            n_timesteps: Dates to keep in temporal mode (deterministic equispaced
                subsampling).
            collapse_time: Temporal collapse mode. ``"median"`` and ``"pick"``
                produce ``(10, H, W)``; ``None`` produces ``(T_sub, 10, H, W)``.
            target: ``"semantic18"`` maps to ``[0..17]``; ``"hcat6"`` remaps to
                the 6 HCAT Level-1 groups ``[0..5]``.
            ignore_index: Label value for Background/Void (and classes without an
                HCAT group). Default 255.
            seed: Seed for reproducibility. The temporal subsampling is already
                deterministic (equispaced); ``seed`` is kept for future stochastic
                variants and is recorded.

        Raises:
            ValueError: if ``collapse_time`` or ``target`` are invalid, or if
                ``n_timesteps`` is not positive.
            FileNotFoundError: if ``root`` does not contain ``DATA_S2/``.
        """
        if collapse_time not in ("median", "pick", None):
            raise ValueError(
                f"invalid collapse_time: {collapse_time!r}; "
                "use 'median', 'pick' or None."
            )
        if target not in ("semantic18", "hcat6"):
            raise ValueError(
                f"invalid target: {target!r}; use 'semantic18' or 'hcat6'."
            )
        if n_timesteps <= 0:
            raise ValueError(f"n_timesteps must be positive, received {n_timesteps}.")

        self.root = Path(root)
        self.folds = tuple(int(f) for f in folds)
        self.n_timesteps = int(n_timesteps)
        self.collapse_time = collapse_time
        self.target = target
        self.ignore_index = int(ignore_index)
        self.seed = int(seed)

        s2_dir = self.root / "DATA_S2"
        if not s2_dir.exists():
            raise FileNotFoundError(f"S2 directory does not exist: {s2_dir}")

        # Class remapping LUT (precomputed only once).
        self._label_lut: np.ndarray = (
            _build_semantic18_lut(self.ignore_index)
            if target == "semantic18"
            else _build_hcat6_lut(self.ignore_index)
        )
        self.num_classes: int = 18 if target == "semantic18" else 6

        # Official fold index and per-fold normalization stats.
        fold_index = _load_fold_index(self.root / "metadata.geojson")
        self._fold_of: dict[str, int] = fold_index
        self._norm_stats = _load_fold_norm_stats(self.root / "NORM_S2_patch.json")

        # split patch_ids = those present on disk whose fold is in `folds`.
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
        """Number of patches in this split."""
        return len(self.patch_ids)

    def _normalize(self, s2: np.ndarray, fold: int | None) -> np.ndarray:
        """Normalizes the S2 tensor ``(T, 10, H, W)`` according to the fold.

        If there are fold stats in ``NORM_S2_patch.json`` it applies per-band
        standardization ``(x/scale - mean) / std``; otherwise the simple
        ``/10000`` scale.

        Args:
            s2: int16 tensor ``(T, 10, H, W)``.
            fold: Fold of the patch (to pick the stats) or ``None``.

        Returns:
            Normalized float32 tensor ``(T, 10, H, W)``.
        """
        x = s2.astype(np.float32) / _S2_SCALE
        if fold is not None and fold in self._norm_stats:
            mean, std = self._norm_stats[fold]
            # mean/std in reflectance scale 0..10000 -> convert to 0..1.
            mean = (mean / _S2_SCALE).reshape(1, _N_BANDS, 1, 1)
            std = (std / _S2_SCALE).reshape(1, _N_BANDS, 1, 1)
            x = (x - mean) / np.where(std == 0.0, 1.0, std)
        return x.astype(np.float32)

    def _collapse(self, x: np.ndarray) -> np.ndarray:
        """Applies the configured temporal mode to the tensor ``(T, 10, H, W)``.

        Args:
            x: Normalized float32 tensor ``(T, 10, H, W)``.

        Returns:
            ``(10, H, W)`` if ``collapse_time`` is ``"median"``/``"pick"``;
            subsampled ``(T_sub, 10, H, W)`` if it is ``None``.
        """
        n_t = x.shape[0]
        if self.collapse_time == "median":
            collapsed: np.ndarray = np.median(x, axis=0)
            return collapsed.astype(np.float32)
        if self.collapse_time == "pick":
            return np.asarray(x[n_t // 2], dtype=np.float32)
        # Temporal mode: deterministic equispaced subsampling.
        idx = _equispaced_indices(n_t, self.n_timesteps)
        return np.asarray(x[idx], dtype=np.float32)

    def _remap_labels(self, semantic: np.ndarray) -> np.ndarray:
        """Remaps the PASTIS class mask ``(H, W)`` to training labels.

        Args:
            semantic: uint8 mask ``(H, W)`` with ``class_id`` in ``0..19``.

        Returns:
            int64 mask ``(H, W)`` in ``[0..num_classes-1]`` union
            ``{ignore_index}``.
        """
        sem = np.clip(semantic.astype(np.int64), 0, 19)
        return self._label_lut[sem]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Loads and transforms patch ``idx`` into ``(x, y)`` tensors.

        Args:
            idx: Index into ``self.patch_ids``.

        Returns:
            Tuple ``(x, y)``:
                - ``x``: float32 ``(10, H, W)`` (2D) or ``(T_sub, 10, H, W)``
                  (temporal).
                - ``y``: int64 ``(H, W)`` with the per-pixel class.

        Raises:
            IndexError: if ``idx`` is out of range.
        """
        if idx < 0:
            idx += len(self.patch_ids)
        if not 0 <= idx < len(self.patch_ids):
            raise IndexError(f"idx out of range: {idx}")

        pid = self.patch_ids[idx]
        patch = load_pastis_patch(pid, root=self.root, load_annotations=True)
        fold = self._fold_of.get(pid)

        s2 = patch["s2"]  # (T, 10, H, W) int16
        x_norm = self._normalize(s2, fold)
        x = self._collapse(x_norm)

        semantic = patch["semantic"]
        if semantic is None:
            # No annotation: everything ignored (should not happen in folds 1-5).
            h, w = x.shape[-2:]
            y = np.full((h, w), self.ignore_index, dtype=np.int64)
        else:
            y = self._remap_labels(semantic)

        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(
            np.ascontiguousarray(y)
        )
