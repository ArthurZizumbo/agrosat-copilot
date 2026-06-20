"""Sen4AgriNet (S4A) netCDF -> dense (T, C, H, W) adapter (US-075).

``Sen4AgriNetDataset`` replicates the exact tensor contract of
:class:`ml.data.pastis_seg_dataset.PASTISSegmentationDataset` so the dense
TSViT / U-TAE segmenters trained on PASTIS-R (France) can be evaluated and
finetuned on Sen4AgriNet (Catalonia) without rewriting the dataloader:

    __getitem__(i) -> (x: torch.Tensor (T, 10, H, W) float32,
                       y: torch.Tensor (H, W)       int64)

The transformation is a verbatim-conceptual port of the official
``Orion-AI-Lab/S4A-Models`` ``utils/PAD_dataset.py`` pipeline:

1. **netCDF read** per band via :mod:`netCDF4` (each band is a group with its own
   ``time`` dim and native resolution: 10 m -> 366, 20 m -> 183, 60 m -> 61).
2. **10 PASTIS-equivalent bands** selected from the 13 S2 bands in the canonical
   PASTIS order ``[B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12]`` (B04 red idx 2,
   B08 NIR idx 6 -- load-bearing for the FR weights to apply; the order is taken
   from :data:`ml.ingest.pastis_loader.PASTIS_S2_BANDS`). The atmospheric 60 m
   bands B01/B09/B10 are dropped.
3. **Monthly binning** identical to the official pipeline: group each band's
   acquisitions into the 12 calendar months and take the per-month median, then
   linearly interpolate the empty months (extrapolating the ends). Result
   ``(12, 10, H, W)``.
4. **Upsample 20 m -> 10 m** with ``np.repeat`` (``expand_ratio = 366 / native``)
   so every band lands on the 366x366 reference grid (B02).
5. **Temporal subsample** to ``n_timesteps`` deterministic equispaced months
   (reusing :func:`ml.data.pastis_seg_dataset._equispaced_indices`), aligning
   with the FR checkpoint's ordinal temporal PE (TSViT-pheno T=10).
6. **Normalization** ``/10000`` -> ``[0, 1]`` (PASTIS ``NORMALIZATION_DIV``).
7. **Spatial tiling** 366 -> 128: the patch is cut into a grid of 128x128
   sub-patches (preserving native resolution, yielding more samples for the
   few-shot finetune), the U-TAE/TSViT input size.
8. **Label remap** FAO-ICC -> macro-HCAT (US-074 crosswalk) -> contiguous id of
   the shared ``hcat-macro`` label-space. Background (0) and any crop outside the
   nomenclature -> ``ignore_index`` (255), the homogeneous US-074 convention.

Project convention: ``torch``/``numpy`` only at the model boundary; ``structlog``
logging; no pandas; type hints everywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog
import torch
from torch.utils.data import Dataset

from ml.data.hcat_crosswalk import MACRO_HCAT_GROUPS
from ml.data.pastis_seg_dataset import _equispaced_indices
from ml.ingest.pastis_loader import PASTIS_S2_BANDS

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ROOT = _REPO_ROOT / "data" / "sen4agrinet"

#: PASTIS reflectance scale (S4A ``NORMALIZATION_DIV``).
_S2_SCALE = 10000.0
#: Number of Sentinel-2 bands kept (PASTIS-equivalent subset).
_N_BANDS = 10
#: S4A reference band (10 m, 366x366) every other band is upsampled to.
_REFERENCE_BAND = "B02"
#: Native side length of each S4A band group (10 m / 20 m / 60 m).
_BAND_NATIVE_SIDE: dict[str, int] = {
    "B01": 61,
    "B02": 366,
    "B03": 366,
    "B04": 366,
    "B05": 183,
    "B06": 183,
    "B07": 183,
    "B08": 366,
    "B09": 61,
    "B10": 61,
    "B11": 183,
    "B12": 183,
    "B8A": 183,
}
#: Number of calendar-month bins (monthly binning).
_N_MONTHS = 12
#: Homogeneous ignore index (US-074 convention) for background / out-of-nomenclature.
IGNORE_INDEX = 255

#: Macro-HCAT group name -> contiguous training id, taken from the US-074
#: vocabulary so the Sen4AgriNet target lives in the SAME label-space the France
#: model is projected to (apples-to-apples Delta mIoU). ``void`` is NOT a trained
#: class: background / out-of-nomenclature pixels go to ``ignore_index`` instead,
#: so it is excluded from the contiguous id range.
MACRO_GROUP_TO_ID: dict[str, int] = {
    name: i for i, name in enumerate(g for g in MACRO_HCAT_GROUPS if g != "void")
}
#: Number of trained macro classes (10 = the US-074 crop groups, void excluded).
N_MACRO_CLASSES: int = len(MACRO_GROUP_TO_ID)

#: Authoritative S4A FAO-ICC ``SELECTED_CLASSES`` (``PAD_dataset.py`` /
#: us-075.md §1.4) mapped to the US-074 macro-HCAT group. Every other FAO-ICC
#: code present in the labels (1987549 background + minor classes, verified by
#: sampling 25 real 31TCG patches) falls outside this nomenclature and is sent
#: to ``ignore_index`` -- never invented into a class.
FAO_ICC_TO_MACRO: dict[int, str] = {
    110: "cereals",  # Wheat
    120: "cereals",  # Maize
    140: "cereals",  # Sorghum
    150: "cereals",  # Barley
    160: "cereals",  # Rye
    170: "cereals",  # Oats
    330: "vineyard",  # Grapes
    435: "oilseed_industrial",  # Rapeseed
    438: "oilseed_industrial",  # Sunflower
    510: "potato",  # Potatoes
    770: "legumes_fodder",  # Peas
}


def build_fao_icc_lut(ignore_index: int = IGNORE_INDEX) -> dict[int, int]:
    """Build the ``FAO-ICC code -> contiguous macro id`` mapping.

    Composes :data:`FAO_ICC_TO_MACRO` (FAO-ICC -> macro name) with
    :data:`MACRO_GROUP_TO_ID` (macro name -> contiguous id). Codes absent from
    the selected nomenclature are intentionally omitted: the encoder defaults
    them to ``ignore_index`` at lookup time.

    Args:
        ignore_index: Value used for background / out-of-nomenclature pixels
            (kept for symmetry; the LUT itself only holds the known crop codes).

    Returns:
        Mapping ``{fao_icc_code: macro_id}`` for the selected crop classes only.
    """
    _ = ignore_index  # documented default applied by the caller
    return {
        code: MACRO_GROUP_TO_ID[macro] for code, macro in FAO_ICC_TO_MACRO.items()
    }


def _list_patches(root: Path) -> list[Path]:
    """Return every ``.nc`` patch under ``root`` (recursive), sorted by name.

    Args:
        root: Subset root (``data/sen4agrinet/``).

    Returns:
        Sorted list of patch paths. Empty if ``root`` has no ``.nc`` files.
    """
    return sorted(root.rglob("*.nc"))


def _read_band_monthly_median(
    nc: object, band: str, year: int
) -> np.ndarray:
    """Read one band group and reduce its time axis to 12 monthly medians.

    Replicates ``PAD_dataset.get_medians``: bin the acquisitions of the band into
    the 12 calendar months of ``year`` (median per month), then linearly
    interpolate the empty months (extrapolating the season ends). Bands at 20/60 m
    are upsampled to the 366x366 reference grid with ``np.repeat``.

    Args:
        nc: An open :class:`netCDF4.Dataset` for the patch.
        band: Band name (e.g. ``"B05"``).
        year: Patch acquisition year (from the global ``patch_year`` attribute).

    Returns:
        ``float32`` array ``(12, 366, 366)`` of monthly medians on the 10 m grid.

    Raises:
        KeyError: if the band group or its time variable is missing.
    """
    import netCDF4  # local import: heavy optional dependency

    nc_ds: netCDF4.Dataset = nc  # type: ignore[assignment]
    grp = nc_ds.groups[band]
    data = np.asarray(grp.variables[band][:], dtype=np.float32)  # (T, h, w)
    time_var = grp.variables["time"]
    # `time` is "seconds since <ref date>"; cftime decodes to real datetimes so we
    # can derive each acquisition's calendar month (1..12) deterministically.
    units = getattr(time_var, "units", f"seconds since {year}-01-01 00:00:00")
    dates = netCDF4.num2date(
        np.asarray(time_var[:]),
        units=units,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )
    months = np.array([d.month for d in np.atleast_1d(dates)], dtype=np.int64)

    h, w = data.shape[1], data.shape[2]
    monthly = np.full((_N_MONTHS, h, w), np.nan, dtype=np.float32)
    for m in range(1, _N_MONTHS + 1):
        sel = months == m
        if bool(sel.any()):
            monthly[m - 1] = np.nanmedian(data[sel], axis=0)

    monthly = _interpolate_months(monthly)

    side: int = _BAND_NATIVE_SIDE.get(band, int(h))
    ref_side: int = _BAND_NATIVE_SIDE[_REFERENCE_BAND]
    if side != ref_side:
        ratio = ref_side // side
        monthly = np.repeat(np.repeat(monthly, ratio, axis=1), ratio, axis=2)
        monthly = monthly[:, :ref_side, :ref_side]
    return monthly.astype(np.float32)


def _interpolate_months(monthly: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN months per pixel, extrapolating the ends.

    Mirrors ``interpolate_na('time_bins', method='linear', fill_value=
    'extrapolate')`` of the official pipeline, vectorized over pixels.

    Args:
        monthly: ``(12, H, W)`` array with NaN for months without data.

    Returns:
        ``(12, H, W)`` array with every month filled. If a pixel has no valid
        month at all it is left as 0.0 (no signal is honest, not invented).
    """
    n_t = monthly.shape[0]
    flat = monthly.reshape(n_t, -1)
    xs = np.arange(n_t, dtype=np.float32)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        valid = ~np.isnan(col)
        n_valid = int(valid.sum())
        if n_valid == n_t:
            continue
        if n_valid == 0:
            flat[:, j] = 0.0
            continue
        flat[:, j] = np.interp(xs, xs[valid], col[valid])
    return flat.reshape(n_t, *monthly.shape[1:])


def _tile_indices(side: int, tile: int) -> list[int]:
    """Return the top-left offsets that tile ``side`` into ``tile``-px windows.

    Covers the full extent with non-overlapping windows; the last window is
    shifted left so it stays inside the patch (so 366 -> [0,128,238] for tile 128,
    yielding 3x3 = 9 sub-patches with a small overlap at the edge).

    Args:
        side: Length of the axis (e.g. 366).
        tile: Sub-patch side (e.g. 128).

    Returns:
        Sorted unique list of valid top-left offsets.
    """
    if tile >= side:
        return [0]
    offs = list(range(0, side - tile + 1, tile))
    if offs[-1] != side - tile:
        offs.append(side - tile)
    return sorted(set(offs))


class Sen4AgriNetDataset(Dataset):
    """Dense Sen4AgriNet dataset emitting the PASTIS ``(x, y)`` contract.

    Each item is ``(x, y)`` where ``x`` is ``(T, 10, tile, tile)`` float32 in
    ``[0, ~1]`` (10 PASTIS-equivalent bands, monthly-binned and temporally
    subsampled) and ``y`` is ``(tile, tile)`` int64 with the per-pixel macro-HCAT
    class (``[0, N_MACRO_CLASSES)`` union ``{ignore_index}``).

    Spatial tiling turns each 366x366 patch into several 128x128 sub-patches, so
    ``len(dataset)`` is ``n_patches * tiles_per_patch``.

    Attributes:
        root: Subset root (``data/sen4agrinet/``).
        n_timesteps: Number of months kept after temporal subsampling.
        tile_size: Side of the emitted sub-patches (128).
        ignore_index: Label for background / out-of-nomenclature pixels.
        num_classes: Number of trained macro classes (:data:`N_MACRO_CLASSES`).
        countries: Country filter applied to the patches (``("ES",)`` etc.).
        items: Flat list of ``(patch_path, row_off, col_off)`` samples.
    """

    def __init__(
        self,
        root: Path = _DEFAULT_ROOT,
        *,
        n_timesteps: int = 10,
        tile_size: int = 128,
        ignore_index: int = IGNORE_INDEX,
        countries: Sequence[str] | None = None,
        precache_all: bool = False,
    ) -> None:
        """Index the subset and enumerate the spatial sub-patches.

        Args:
            root: Subset root holding the ``.nc`` patches.
            n_timesteps: Months to keep (deterministic equispaced subsampling);
                10 aligns with the TSViT-pheno France checkpoint.
            tile_size: Sub-patch side (128, the U-TAE/TSViT input size).
            ignore_index: Background / out-of-nomenclature label (default 255).
            countries: If set, keep only patches whose ``patch_country_code`` is
                in this set (e.g. ``("ES",)`` for Catalonia-only). ``None`` keeps
                all patches.
            precache_all: If ``True`` decode EVERY unique patch once at init into an
                in-memory dict so ``__getitem__`` is decode-free even under a
                shuffling DataLoader (the netCDF decode is the dominant cost). Bounded
                memory for the small US-075 subset (~30 patches); leave ``False`` for
                the streaming default that holds only one patch (tests, large sets).

        Raises:
            ValueError: if ``n_timesteps`` or ``tile_size`` is not positive.
            FileNotFoundError: if ``root`` does not exist.
        """
        if n_timesteps <= 0:
            raise ValueError(f"n_timesteps must be positive, received {n_timesteps}.")
        if tile_size <= 0:
            raise ValueError(f"tile_size must be positive, received {tile_size}.")

        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Sen4AgriNet subset root not found: {self.root}")

        self.n_timesteps = int(n_timesteps)
        self.tile_size = int(tile_size)
        self.ignore_index = int(ignore_index)
        self.num_classes = N_MACRO_CLASSES
        self.countries = tuple(countries) if countries is not None else None
        self._fao_lut = build_fao_icc_lut(self.ignore_index)
        # Single-patch decode cache: ``items`` is grouped by patch (all tiles of a
        # patch are consecutive), so caching the last decoded ``(x_full, y_full)``
        # lets the N tiles of a patch reuse ONE netCDF decode instead of N. The
        # netCDF read + monthly binning is the dominant cost (~seconds/patch); this
        # gives a ~N x speedup with O(1) memory (one patch held at a time).
        self._cache_key: Path | None = None
        self._cache_val: tuple[np.ndarray, np.ndarray] | None = None
        #: Full decode dict (only populated when ``precache_all=True``).
        self._full_cache: dict[Path, tuple[np.ndarray, np.ndarray]] | None = None

        patches = _list_patches(self.root)
        if self.countries is not None:
            patches = [p for p in patches if self._country_of(p) in self.countries]

        ref_side = _BAND_NATIVE_SIDE[_REFERENCE_BAND]
        offsets = _tile_indices(ref_side, self.tile_size)
        self.items: list[tuple[Path, int, int]] = [
            (p, r, c) for p in patches for r in offsets for c in offsets
        ]

        if precache_all:
            self._full_cache = {
                p: self._load_patch_tensors(p) for p in patches
            }

        logger.info(
            "sen4agrinet_dataset_init",
            n_patches=len(patches),
            n_items=len(self.items),
            tile_size=self.tile_size,
            n_timesteps=self.n_timesteps,
            num_classes=self.num_classes,
            countries=self.countries,
            precache_all=precache_all,
        )

    @staticmethod
    def _country_of(nc_path: Path) -> str:
        """Read the ``patch_country_code`` global attribute of a patch.

        Args:
            nc_path: Path to a ``.nc`` patch.

        Returns:
            The country code (``"ES"`` / ``"FR"``), or ``""`` if unreadable.
        """
        import netCDF4

        try:
            ds = netCDF4.Dataset(str(nc_path))
        except OSError:
            return ""
        try:
            return str(getattr(ds, "patch_country_code", ""))
        finally:
            ds.close()

    def __len__(self) -> int:
        """Number of (sub-patch) samples in the subset."""
        return len(self.items)

    def _load_patch_tensors(
        self, nc_path: Path
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load and transform a full patch into ``(x_full, y_full)`` arrays.

        Args:
            nc_path: Path to a ``.nc`` patch.

        Returns:
            Tuple ``(x_full, y_full)``:
                - ``x_full``: float32 ``(T, 10, 366, 366)`` in ``[0, ~1]``.
                - ``y_full``: int64 ``(366, 366)`` macro-HCAT labels.
        """
        import netCDF4

        ds = netCDF4.Dataset(str(nc_path))
        try:
            year = int(getattr(ds, "patch_year", 0))
            # (12, 10, 366, 366): monthly medians per PASTIS-ordered band.
            bands = [
                _read_band_monthly_median(ds, b, year) for b in PASTIS_S2_BANDS
            ]
            stacked = np.stack(bands, axis=1)  # (12, 10, H, W)
            labels = np.asarray(
                ds.groups["labels"].variables["labels"][:], dtype=np.int64
            )
        finally:
            ds.close()

        # Temporal subsample to n_timesteps (deterministic equispaced months).
        idx = _equispaced_indices(stacked.shape[0], self.n_timesteps)
        x_full = (stacked[idx] / _S2_SCALE).astype(np.float32)
        y_full = self._encode_labels(labels)
        return x_full, y_full

    def _encode_labels(self, labels: np.ndarray) -> np.ndarray:
        """Map FAO-ICC label codes to contiguous macro-HCAT ids.

        Args:
            labels: int64 ``(H, W)`` array of raw FAO-ICC crop codes.

        Returns:
            int64 ``(H, W)`` array in ``[0, num_classes)`` union
            ``{ignore_index}``. Background (0) and any code outside the selected
            nomenclature go to ``ignore_index``.
        """
        out = np.full(labels.shape, self.ignore_index, dtype=np.int64)
        for code, macro_id in self._fao_lut.items():
            out[labels == code] = macro_id
        return out

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load sub-patch ``idx`` as ``(x, y)`` tensors.

        Args:
            idx: Index into :attr:`items`.

        Returns:
            Tuple ``(x, y)``:
                - ``x``: float32 ``(T, 10, tile, tile)``.
                - ``y``: int64 ``(tile, tile)`` macro-HCAT labels.

        Raises:
            IndexError: if ``idx`` is out of range.
        """
        if idx < 0:
            idx += len(self.items)
        if not 0 <= idx < len(self.items):
            raise IndexError(f"idx out of range: {idx}")

        nc_path, row, col = self.items[idx]
        if self._full_cache is not None:
            x_full, y_full = self._full_cache[nc_path]
        elif self._cache_key == nc_path and self._cache_val is not None:
            x_full, y_full = self._cache_val
        else:
            x_full, y_full = self._load_patch_tensors(nc_path)
            self._cache_key = nc_path
            self._cache_val = (x_full, y_full)
        ts = self.tile_size
        x = x_full[:, :, row : row + ts, col : col + ts]
        y = y_full[row : row + ts, col : col + ts]
        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.ascontiguousarray(y)),
        )
