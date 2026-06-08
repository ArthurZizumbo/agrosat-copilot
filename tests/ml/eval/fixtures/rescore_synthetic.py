"""Synthetic class maps for the re-score harness tests (US-030).

Provides deterministic prediction/target maps in both the contiguous 18-class
space and the PASTIS-R 20-class space, so the helpers in
:mod:`ml.eval.class_remap` (and, later, the harness consolidation) can be
exercised with fixed golden values and no checkpoint loading.

Everything is seeded through :func:`numpy.random.default_rng`, so two calls with
the same ``seed`` return identical arrays. The maps are intentionally small
(default ``32 x 32``) to keep the tests fast while still covering streaming vs
one-shot accumulation.
"""

from __future__ import annotations

import numpy as np

#: Default spatial side of the synthetic maps.
DEFAULT_SIZE: int = 32
#: Background id of the PASTIS-R 20-class convention.
BACKGROUND_ID: int = 0
#: Void id of the PASTIS-R 20-class convention.
VOID_ID: int = 19


def make_18class_pair(
    *,
    size: int = DEFAULT_SIZE,
    ignore_index: int = 255,
    ignore_frac: float = 0.1,
    error_frac: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a (preds, target) pair already in the contiguous 18-class space.

    The target holds agronomic ids in ``[0..17]`` plus a fraction of
    ``ignore_index`` pixels (the unified void). The prediction equals the target
    except on a controlled fraction of pixels (``error_frac``) where it is
    shifted to a different valid class, allowing closed-form metric assertions.

    Args:
        size: Side length of the square maps.
        ignore_index: Value used for ignored/void pixels in the target.
        ignore_frac: Fraction of pixels set to ``ignore_index`` in the target.
        error_frac: Fraction of non-ignored pixels where the prediction differs
            from the target (``0.0`` means a perfect prediction).
        seed: Deterministic seed.

    Returns:
        Tuple ``(preds, target)`` of ``int64`` arrays with shape ``(size, size)``.
    """
    rng = np.random.default_rng(seed)
    target = rng.integers(0, 18, size=(size, size)).astype(np.int64)

    n = size * size
    flat_idx = rng.permutation(n)
    n_ignore = round(ignore_frac * n)
    target_flat = target.reshape(-1)
    target_flat[flat_idx[:n_ignore]] = ignore_index

    preds_flat = target_flat.copy()
    valid_idx = flat_idx[n_ignore:]
    n_err = round(error_frac * valid_idx.size)
    for pos in valid_idx[:n_err]:
        # Shift to a guaranteed-different valid class in [0..17].
        preds_flat[pos] = (int(target_flat[pos]) + 1) % 18

    return preds_flat.reshape(size, size), target_flat.reshape(size, size)


def make_20class_pair(
    *,
    size: int = DEFAULT_SIZE,
    background_frac: float = 0.1,
    void_frac: float = 0.1,
    error_frac: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a (preds, target) pair in the PASTIS-R 20-class space ``[0..19]``.

    The target contains agronomic ids ``1..18`` plus controlled fractions of
    Background (``0``) and Void (``19``) pixels, mirroring the raw output of a
    20-class checkpoint before :func:`ml.eval.class_remap.remap_20_to_18`.

    Args:
        size: Side length of the square maps.
        background_frac: Fraction of pixels set to Background (id ``0``).
        void_frac: Fraction of pixels set to Void (id ``19``).
        error_frac: Fraction of agronomic pixels where the prediction differs
            from the target.
        seed: Deterministic seed.

    Returns:
        Tuple ``(preds, target)`` of ``int64`` arrays with shape ``(size, size)``
        and values in ``[0..19]``.
    """
    rng = np.random.default_rng(seed)
    # Agronomic base in 1..18.
    target = rng.integers(1, 19, size=(size, size)).astype(np.int64)

    n = size * size
    flat_idx = rng.permutation(n)
    n_bg = round(background_frac * n)
    n_void = round(void_frac * n)
    target_flat = target.reshape(-1)
    target_flat[flat_idx[:n_bg]] = BACKGROUND_ID
    target_flat[flat_idx[n_bg : n_bg + n_void]] = VOID_ID

    preds_flat = target_flat.copy()
    agronomic_positions = flat_idx[n_bg + n_void :]
    n_err = round(error_frac * agronomic_positions.size)
    for pos in agronomic_positions[:n_err]:
        # Keep the error inside the agronomic range 1..18.
        preds_flat[pos] = (int(target_flat[pos]) % 18) + 1

    return preds_flat.reshape(size, size), target_flat.reshape(size, size)


def make_label_map_256(
    *,
    size: int = 256,
    n_classes: int = 18,
    block: int = 16,
    seed: int = 0,
) -> np.ndarray:
    """Build a blocky 256-resolution label map for resampling tests.

    The map is piecewise-constant over ``block x block`` tiles so that a
    nearest-neighbour downsample to a multiple-of-``block`` size is exact and
    introduces no new class ids.

    Args:
        size: Side length of the square map (default 256).
        n_classes: Number of distinct class ids drawn (values in ``[0..n_classes)``).
        block: Side of each constant tile; ``size`` should be a multiple of it.
        seed: Deterministic seed.

    Returns:
        An ``int64`` array of shape ``(size, size)`` with blocky class regions.
    """
    rng = np.random.default_rng(seed)
    n_tiles = size // block
    tiles = rng.integers(0, n_classes, size=(n_tiles, n_tiles)).astype(np.int64)
    return np.kron(tiles, np.ones((block, block), dtype=np.int64))
