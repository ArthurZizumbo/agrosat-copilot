"""Pure class-space helpers for the segmentation re-score harness (US-030).

The harness re-evaluates six trained segmentation checkpoints apples-to-apples
in a single, contiguous **18-class** space at a fixed **128** resolution with a
unified ``ignore_index``. Two transformations make that possible and live here,
isolated from :mod:`ml.eval.dense_metrics` (the harness module) so they can be
unit-tested without loading any checkpoint:

- :func:`remap_20_to_18` collapses the PASTIS-R 20-class label space
  ``[0..19]`` (Background + 18 agronomic classes + Void) into the contiguous
  ``[0..17]`` space, sending Background and Void to ``ignore_index``.
- :func:`resample_mask_128_nearest` resamples a discrete class map to
  ``128 x 128`` using nearest-neighbour, so models trained at 256 (U-Net,
  AnySat, SegFormer) are accumulated at the same resolution as the 128-native
  models without inventing interpolated class ids.

Both operate on already-discrete maps (labels or post-``argmax`` predictions),
never on logits, model heads or ``state_dict`` keys: the U-TAE checkpoint keys
(``out_conv`` etc.) must stay intact, so the 20->18 mapping happens purely in
prediction space.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "HARNESS_IGNORE_INDEX",
    "HARNESS_NUM_CLASSES",
    "HARNESS_SIZE",
    "remap_20_to_18",
    "resample_mask_128_nearest",
]

#: Number of contiguous classes the harness accumulates over (1..18 -> 0..17).
HARNESS_NUM_CLASSES: int = 18
#: Unified ignore index for Background, Void and out-of-range pixels.
HARNESS_IGNORE_INDEX: int = 255
#: Target side length every mask is resampled to before accumulation.
HARNESS_SIZE: int = 128


def _to_numpy_int(x: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return a contiguous numpy integer array from a numpy/torch input.

    Args:
        x: Discrete class map as a numpy array or torch tensor.

    Returns:
        A ``numpy.ndarray`` with an integer dtype, detached from any autograd
        graph and moved to CPU when the input is a tensor.
    """
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int64)
    return arr


def remap_20_to_18(
    labels: np.ndarray | torch.Tensor,
    *,
    background_id: int = 0,
    void_id: int = 19,
    ignore_index: int = HARNESS_IGNORE_INDEX,
) -> np.ndarray:
    """Map a 20-class label/prediction map ``[0..19]`` to contiguous ``[0..17]``.

    The PASTIS-R 20-class convention reserves id ``0`` for Background and id
    ``19`` for Void; the 18 agronomic classes occupy ``1..18``. This helper
    reindexes those agronomic classes to ``0..17`` (a simple shift of ``-1``)
    and sends both Background and Void to ``ignore_index`` so they are excluded
    from the unified confusion matrix.

    It operates AFTER ``argmax`` on discrete class maps, never on logits, the
    model head or the ``state_dict`` (the U-TAE checkpoint keys must stay
    intact); the remap lives purely in prediction/label space.

    Args:
        labels: Integer class map of any shape with values in ``[0..19]``.
        background_id: Class id treated as background (mapped to ignore).
        void_id: Class id treated as void (mapped to ignore).
        ignore_index: Value assigned to background, void and any id outside
            the agronomic ``1..18`` range.

    Returns:
        A ``numpy.ndarray`` of dtype ``int64`` and the same shape as ``labels``,
        with agronomic classes in ``[0..17]`` and background/void set to
        ``ignore_index``.
    """
    arr = _to_numpy_int(labels)
    out = np.full(arr.shape, ignore_index, dtype=np.int64)
    # Agronomic classes 1..18 -> 0..17. Anything else (Background, Void,
    # out-of-range) stays at ignore_index by construction.
    agronomic = (arr >= 1) & (arr <= HARNESS_NUM_CLASSES)
    agronomic &= arr != background_id
    agronomic &= arr != void_id
    out[agronomic] = arr[agronomic] - 1
    return out


def resample_mask_128_nearest(
    mask: np.ndarray | torch.Tensor,
    *,
    size: int = HARNESS_SIZE,
) -> np.ndarray:
    """Resample a discrete class map to ``size`` x ``size`` using nearest-neighbour.

    Used for models trained at 256 (U-Net, AnySat, SegFormer) so every model is
    accumulated at the same ``size`` resolution as the 128-native models.
    Nearest-neighbour guarantees no new (interpolated) class ids are introduced:
    every output value already appears in the input.

    Args:
        mask: Discrete class map of shape ``(H, W)``.
        size: Target side length (default :data:`HARNESS_SIZE` = 128).

    Returns:
        A ``numpy.ndarray`` of dtype ``int64`` and shape ``(size, size)``,
        nearest-neighbour resampled. When the input is already ``(size, size)``
        the values are returned unchanged (only dtype is normalized).

    Raises:
        ValueError: if ``mask`` is not a 2D map.
    """
    arr = _to_numpy_int(mask)
    if arr.ndim != 2:
        raise ValueError(
            f"`mask` must be a 2D (H, W) class map; received shape {arr.shape}."
        )
    if arr.shape == (size, size):
        return arr.astype(np.int64, copy=True)

    # Nearest interpolation in float would be lossless for integers, but we keep
    # the values exact by interpolating on a float view and casting back. torch
    # interpolate needs a (N, C, H, W) tensor.
    tensor = torch.from_numpy(arr.astype(np.float32))[None, None, :, :]
    resampled = torch.nn.functional.interpolate(
        tensor, size=(size, size), mode="nearest"
    )
    return resampled[0, 0].round().to(torch.int64).numpy()
