"""Golden-value tests for the probability-space class helpers (US-031).

Covers :func:`ml.eval.class_remap.remap_probs_20_to_18` and
:func:`ml.eval.class_remap.resample_probs_128_bilinear`: the new helpers that map
a 20-class softmax to the contiguous 18-class space and resample probabilities to
128 with bilinear interpolation. These operate purely on synthetic post-softmax
tensors (no checkpoint), so the asserts are closed-form golden values.

The discrete helpers (``remap_20_to_18`` / ``resample_mask_128_nearest``) are
intentionally NOT exercised here for probabilities: a regression test confirms
they remain available and unchanged for US-030.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.eval.class_remap import (
    HARNESS_SIZE,
    remap_20_to_18,
    remap_probs_20_to_18,
    resample_mask_128_nearest,
    resample_probs_128_bilinear,
)
from tests.ml.eval.oof.fixtures.oof_synthetic import make_logits, make_softmax

_SUM_TOL = 1e-5


def test_make_softmax_sums_to_one() -> None:
    """The synthetic softmax fixture is a valid distribution (sum 1, >= 0)."""
    probs = make_softmax(num_classes=20, size=8, seed=1)
    assert probs.shape == (20, 8, 8)
    np.testing.assert_allclose(probs.sum(axis=0), 1.0, atol=_SUM_TOL)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_softmax_maps_logits_into_unit_interval() -> None:
    """Softmax over out-of-[0,1] logits yields probabilities in [0, 1]."""
    logits = make_logits(num_classes=18, size=8, scale=7.0, seed=2)[0]
    # Confirm the raw logits are NOT already a distribution.
    assert logits.min() < 0.0
    assert logits.max() > 1.0
    shifted = logits - logits.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=0, keepdims=True)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
    np.testing.assert_allclose(probs.sum(axis=0), 1.0, atol=_SUM_TOL)


def test_remap_probs_20_to_18_drops_channels_and_renorms() -> None:
    """remap_probs_20_to_18 drops channels 0/19 and renormalizes to sum 1."""
    probs20 = make_softmax(num_classes=20, size=8, seed=3)
    out = remap_probs_20_to_18(probs20)

    assert out.shape == (18, 8, 8)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out.sum(axis=0), 1.0, atol=_SUM_TOL)
    assert out.min() >= 0.0

    # Golden: the kept 18 channels (1..18 of the original) renormalized by the
    # mass excluding Background (0) and Void (19).
    kept = probs20[1:19]
    expected = kept / kept.sum(axis=0, keepdims=True)
    np.testing.assert_allclose(out, expected.astype(np.float32), atol=_SUM_TOL)


def test_remap_probs_preserves_relative_order() -> None:
    """The argmax of the remapped 18 channels matches the argmax over 1..18."""
    probs20 = make_softmax(num_classes=20, size=6, seed=4)
    out = remap_probs_20_to_18(probs20)
    # argmax over the 18 agronomic channels (offset by 1 in the 20-space).
    expected_argmax = probs20[1:19].argmax(axis=0)
    np.testing.assert_array_equal(out.argmax(axis=0), expected_argmax)


def test_remap_probs_custom_drop_ids() -> None:
    """Custom background/void ids drop exactly those two channels."""
    probs20 = make_softmax(num_classes=20, size=4, seed=5)
    out = remap_probs_20_to_18(probs20, background_id=5, void_id=10)
    assert out.shape == (18, 4, 4)
    kept_idx = [c for c in range(20) if c not in (5, 10)]
    expected = probs20[kept_idx]
    expected = expected / expected.sum(axis=0, keepdims=True)
    np.testing.assert_allclose(out, expected.astype(np.float32), atol=_SUM_TOL)


def test_remap_probs_invalid_ids_raise() -> None:
    """Equal or out-of-range drop ids raise ValueError."""
    probs20 = make_softmax(num_classes=20, size=4, seed=6)
    with pytest.raises(ValueError):
        remap_probs_20_to_18(probs20, background_id=3, void_id=3)
    with pytest.raises(ValueError):
        remap_probs_20_to_18(probs20, void_id=99)


def test_remap_probs_no_20_axis_raises() -> None:
    """An input without a 20-length axis raises (cannot locate the class axis)."""
    probs18 = make_softmax(num_classes=18, size=4, seed=7)
    with pytest.raises(ValueError):
        remap_probs_20_to_18(probs18)


def test_resample_probs_bilinear_shape_and_sum() -> None:
    """Bilinear resample 256->128 keeps shape (18,128,128) and sum 1."""
    probs = make_softmax(num_classes=18, size=256, seed=8)
    out = resample_probs_128_bilinear(probs)
    assert out.shape == (18, HARNESS_SIZE, HARNESS_SIZE)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out.sum(axis=0), 1.0, atol=1e-4)
    assert out.min() >= 0.0


def test_resample_probs_identity_when_already_128() -> None:
    """A (C,128,128) input is returned renormalized, same shape, no interpolation."""
    probs = make_softmax(num_classes=18, size=128, seed=9)
    out = resample_probs_128_bilinear(probs)
    assert out.shape == (18, 128, 128)
    np.testing.assert_allclose(out.sum(axis=0), 1.0, atol=_SUM_TOL)
    # Already a normalized distribution -> values unchanged within float tol.
    np.testing.assert_allclose(out, probs, atol=1e-5)


def test_resample_probs_custom_size() -> None:
    """A custom target size is honored."""
    probs = make_softmax(num_classes=18, size=64, seed=10)
    out = resample_probs_128_bilinear(probs, size=32)
    assert out.shape == (18, 32, 32)
    np.testing.assert_allclose(out.sum(axis=0), 1.0, atol=1e-4)


def test_resample_probs_rejects_non_3d() -> None:
    """A non-3D input is rejected."""
    bad = make_softmax(num_classes=18, size=8, seed=11)[0]  # (8, 8)
    with pytest.raises(ValueError):
        resample_probs_128_bilinear(bad)


def test_discrete_helpers_unchanged_for_us030() -> None:
    """Discrete remap/resample still behave as US-030 expects (regression)."""
    # remap_20_to_18: agronomic 1..18 -> 0..17, background/void -> ignore.
    labels = np.array([[0, 1, 18, 19]], dtype=np.int64)
    out = remap_20_to_18(labels)
    np.testing.assert_array_equal(out, np.array([[255, 0, 17, 255]]))
    # resample_mask_128_nearest: discrete, introduces no new ids.
    mask = np.zeros((256, 256), dtype=np.int64)
    mask[128:, :] = 7
    res = resample_mask_128_nearest(mask)
    assert res.shape == (128, 128)
    assert set(np.unique(res).tolist()) <= {0, 7}
