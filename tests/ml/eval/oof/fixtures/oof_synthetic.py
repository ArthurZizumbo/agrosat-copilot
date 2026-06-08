"""Synthetic logits / softmax / ParcelID maps for the US-031 tests.

Every factory is seeded through :func:`numpy.random.default_rng`, so two calls
with the same ``seed`` return identical arrays. The maps are intentionally small
to keep the tests fast while still exercising the probability-space remap,
bilinear resample, and pixel->parcel reconciliation with fixed golden values.

No checkpoint is loaded anywhere: the synthetic logits stand in for a model's
raw head output so the softmax helpers and the reconciliation can be asserted in
closed form.
"""

from __future__ import annotations

import numpy as np

#: Default spatial side of the synthetic maps.
DEFAULT_SIZE: int = 16
#: Background id of the PASTIS-R 20-class convention.
BACKGROUND_ID: int = 0
#: Void id of the PASTIS-R 20-class convention.
VOID_ID: int = 19


def make_logits(
    *,
    num_classes: int = 20,
    size: int = DEFAULT_SIZE,
    scale: float = 5.0,
    seed: int = 0,
) -> np.ndarray:
    """Build a synthetic logits tensor ``(1, num_classes, size, size)``.

    The values span a wide range (``+/- scale``) so that, when fed through
    softmax, the resulting distribution is peaked but never degenerate, and so
    that a test can confirm the softmax maps out-of-``[0, 1]`` logits into
    ``[0, 1]`` probabilities.

    Args:
        num_classes: Number of class channels (20 for native PASTIS, 18 for the
            contiguous space).
        size: Spatial side length of the square map.
        scale: Half-width of the uniform logit range ``[-scale, scale]``.
        seed: Deterministic seed.

    Returns:
        A ``float32`` array ``(1, num_classes, size, size)``.
    """
    rng = np.random.default_rng(seed)
    logits = rng.uniform(-scale, scale, size=(1, num_classes, size, size))
    return logits.astype(np.float32)


def make_softmax(
    *,
    num_classes: int = 20,
    size: int = DEFAULT_SIZE,
    seed: int = 0,
) -> np.ndarray:
    """Build a valid post-softmax map ``(num_classes, size, size)``.

    Applies a numerically-stable softmax over the class axis of synthetic
    logits, so ``out.sum(0) ~ 1`` and ``out >= 0`` hold exactly (float32).

    Args:
        num_classes: Number of class channels.
        size: Spatial side length.
        seed: Deterministic seed.

    Returns:
        A ``float32`` array ``(num_classes, size, size)`` summing to 1 over axis
        0 per pixel.
    """
    logits = make_logits(num_classes=num_classes, size=size, seed=seed)[0]
    shifted = logits - logits.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    probs: np.ndarray = exp / exp.sum(axis=0, keepdims=True)
    return probs.astype(np.float32)


def make_parcel_grid(
    *,
    size: int = 4,
) -> tuple[np.ndarray, dict[int, int]]:
    """Build a small ParcelID grid with known per-parcel pixel counts.

    The grid splits a ``size x size`` map vertically into two parcels plus a
    Background border, so the per-parcel support is exactly known:

    - left half (columns ``[0, size//2)``): parcel id ``101``.
    - right half except the last column: parcel id ``202``.
    - last column: Background (id ``0``), excluded from any parcel.

    Args:
        size: Side length of the square grid (must be >= 3).

    Returns:
        Tuple ``(parcel_ids, expected_counts)`` where ``parcel_ids`` is an
        ``int64`` ``(size, size)`` map and ``expected_counts`` maps each parcel
        id to its pixel count.
    """
    if size < 3:
        raise ValueError(f"size must be >= 3 to fit two parcels and a border, got {size}.")
    grid = np.zeros((size, size), dtype=np.int64)
    half = size // 2
    grid[:, :half] = 101
    grid[:, half : size - 1] = 202
    # last column stays Background (0)
    counts = {
        101: int((grid == 101).sum()),
        202: int((grid == 202).sum()),
    }
    return grid, counts


def make_constant_class_probs(
    parcel_ids: np.ndarray,
    *,
    class_of_parcel: dict[int, int],
    num_classes: int = 18,
) -> np.ndarray:
    """Build a softmax that is a one-hot of a known class per parcel.

    Every pixel of parcel ``p`` gets probability 1 on class ``class_of_parcel[p]``
    and 0 elsewhere, so the parcel-level mean is exactly that one-hot and the
    predicted class is deterministic. Background pixels (id ``0``) get a uniform
    distribution (they are dropped by the reconciler anyway).

    Args:
        parcel_ids: ``(H, W)`` int map of local ParcelIDs.
        class_of_parcel: Mapping ``parcel_id -> class index`` in ``[0, num_classes)``.
        num_classes: Number of contiguous classes (default 18).

    Returns:
        A ``float32`` array ``(num_classes, H, W)`` summing to 1 over axis 0.
    """
    h, w = parcel_ids.shape
    probs = np.full((num_classes, h, w), 1.0 / num_classes, dtype=np.float32)
    for pid, cls in class_of_parcel.items():
        mask = parcel_ids == pid
        probs[:, mask] = 0.0
        probs[cls, mask] = 1.0
    return probs
