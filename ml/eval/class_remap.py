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

US-031 adds the **probability-space** counterparts
(:func:`remap_probs_20_to_18`, :func:`resample_probs_128_bilinear`) used by the
softmax/OOF dump. The discrete helpers above MUST NOT be used on probability
tensors: ``remap_20_to_18`` shifts class ids and would silently corrupt the
class axis of a softmax, and ``resample_mask_128_nearest`` uses nearest
interpolation, which degrades a continuous distribution (it picks a single
neighbour's value instead of blending). The probability helpers instead DROP the
Background (0) and Void (19) channels, renormalize the remaining 18 to sum to 1,
and resample with bilinear interpolation followed by a renormalization. They
operate on POST-softmax tensors only, never on logits or ``state_dict`` keys.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "HARNESS_IGNORE_INDEX",
    "HARNESS_NUM_CLASSES",
    "HARNESS_SIZE",
    "remap_20_to_18",
    "remap_probs_20_to_18",
    "resample_mask_128_nearest",
    "resample_probs_128_bilinear",
]

#: Numerical floor used to avoid divide-by-zero when renormalizing a probability
#: map whose 18 kept channels sum to (near) zero for some pixel.
_PROB_RENORM_EPS: float = 1e-12

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


def _to_numpy_float(x: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return a contiguous float32 numpy array from a numpy/torch input.

    Args:
        x: Probability map as a numpy array or torch tensor.

    Returns:
        A ``numpy.ndarray`` of dtype ``float32``, detached from any autograd
        graph and moved to CPU when the input is a tensor.
    """
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return np.ascontiguousarray(arr)


def remap_probs_20_to_18(
    probs: np.ndarray | torch.Tensor,
    *,
    background_id: int = 0,
    void_id: int = 19,
) -> np.ndarray:
    """Map a 20-class probability map to the contiguous 18-class space.

    Probability analogue of :func:`remap_20_to_18` (which only handles discrete
    class maps). The 20-class checkpoints (U-Net, U-TAE, AnySat, SegFormer)
    output a softmax over the PASTIS-R convention ``[0..19]`` where id ``0`` is
    Background and id ``19`` is Void. This helper DROPS those two channels and
    RENORMALIZES the remaining 18 agronomic channels so they sum to 1 per pixel,
    yielding a distribution over the contiguous ``[0..17]`` space identical to
    the one the 18-native models (DeepLabv3+, TSViT-pheno) emit directly.

    The class axis is assumed to be the FIRST axis for a ``(20, H, W)`` input;
    for higher-rank inputs (e.g. a batched ``(N, 20, H, W)``) pass an array whose
    axis ``-3`` is the 20-class axis -- the function locates the class axis as
    the one of length 20. It operates on POST-softmax tensors only, never on
    logits or ``state_dict`` keys.

    Args:
        probs: Probability map with a class axis of length 20. Common shape is
            ``(20, H, W)``; ``(N, 20, H, W)`` is also accepted.
        background_id: Channel index treated as Background and dropped.
        void_id: Channel index treated as Void and dropped.

    Returns:
        A ``float32`` ``numpy.ndarray`` with the 20-class axis replaced by an
        18-class axis (Background/Void removed), renormalized so the kept
        channels sum to 1 along that axis. Shape mirrors the input with the class
        axis shrunk from 20 to 18.

    Raises:
        ValueError: if no axis of length 20 is found, if the input is ambiguous
            (more than one axis of length 20), or if ``background_id``/``void_id``
            are out of range or equal.
    """
    arr = _to_numpy_float(probs)
    class_axis = _find_class_axis(arr.shape, expected=20)
    if not 0 <= background_id < 20 or not 0 <= void_id < 20:
        raise ValueError(
            f"background_id={background_id} and void_id={void_id} must be in "
            "[0, 20)."
        )
    if background_id == void_id:
        raise ValueError(
            f"background_id and void_id must differ; both were {background_id}."
        )

    keep = [c for c in range(20) if c not in (background_id, void_id)]
    kept = np.take(arr, keep, axis=class_axis)
    denom = kept.sum(axis=class_axis, keepdims=True)
    denom = np.where(denom < _PROB_RENORM_EPS, 1.0, denom)
    out: np.ndarray = (kept / denom).astype(np.float32)
    return out


def resample_probs_128_bilinear(
    probs: np.ndarray | torch.Tensor,
    *,
    size: int = HARNESS_SIZE,
) -> np.ndarray:
    """Resample a probability map ``(C, H, W)`` to ``(C, size, size)`` bilinearly.

    Probability analogue of :func:`resample_mask_128_nearest`. Bilinear (not
    nearest) interpolation is the correct choice for a continuous distribution:
    nearest would copy a single neighbour's value and destroy the smooth class
    posterior. After interpolation the per-pixel distribution is RENORMALIZED so
    every output pixel still sums to 1 along the class axis (bilinear blending of
    rows/columns that each sum to 1 already preserves the sum, but the explicit
    renormalization guards against float drift).

    Args:
        probs: Probability map ``(C, H, W)`` (class-first). Values are assumed
            POST-softmax (non-negative, sum 1 over ``C``).
        size: Target side length (default :data:`HARNESS_SIZE` = 128).

    Returns:
        A ``float32`` ``numpy.ndarray`` of shape ``(C, size, size)`` whose
        per-pixel distribution sums to 1 along the class axis. When the input is
        already ``(C, size, size)`` the values are returned renormalized only
        (no interpolation).

    Raises:
        ValueError: if ``probs`` is not a 3D ``(C, H, W)`` map.
    """
    arr = _to_numpy_float(probs)
    if arr.ndim != 3:
        raise ValueError(
            f"`probs` must be a 3D (C, H, W) probability map; received shape "
            f"{arr.shape}."
        )

    if arr.shape[1:] != (size, size):
        tensor = torch.from_numpy(arr)[None, ...]  # (1, C, H, W)
        resampled = torch.nn.functional.interpolate(
            tensor, size=(size, size), mode="bilinear", align_corners=False
        )
        arr = resampled[0].numpy()

    denom = arr.sum(axis=0, keepdims=True)
    denom = np.where(denom < _PROB_RENORM_EPS, 1.0, denom)
    out: np.ndarray = (arr / denom).astype(np.float32)
    return out


def _find_class_axis(shape: tuple[int, ...], *, expected: int) -> int:
    """Locate the single axis of length ``expected`` in ``shape``.

    Args:
        shape: Array shape to inspect.
        expected: The class-axis length to find (e.g. 20).

    Returns:
        The index of the unique axis whose length equals ``expected``.

    Raises:
        ValueError: if zero or more than one axis matches ``expected``.
    """
    matches = [ax for ax, dim in enumerate(shape) if dim == expected]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"no axis of length {expected} found in shape {shape}; the class "
            "axis is required to remap probabilities."
        )
    raise ValueError(
        f"ambiguous class axis: multiple axes of length {expected} in shape "
        f"{shape}. Pass a tensor with a single 20-length axis."
    )
