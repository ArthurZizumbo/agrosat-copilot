"""Tests for the DirPA logit adjuster (ml.losses.dirpa)."""

from __future__ import annotations

import pytest
import torch

from ml.losses.dirpa import DirPALogitAdjuster, apply_dirpa


def test_tau_zero_is_noop() -> None:
    """tau=0 must return the logits untouched (plain training)."""
    z = torch.randn(2, 10, 4, 4)
    assert torch.equal(apply_dirpa(z, alpha=1.0, tau=0.0), z)


def test_shift_is_per_class_broadcast() -> None:
    """The adjustment is one K-vector broadcast over batch and spatial dims."""
    z = torch.randn(3, 10, 5, 5)
    out = apply_dirpa(z, alpha=0.5, tau=1.0)
    diff = out - z
    ref = diff[0, :, 0, 0]
    # Same per-class shift at any (batch, h, w) location.
    assert torch.allclose(diff[2, :, 4, 3], ref, atol=1e-5)
    assert out.shape == z.shape


def test_tabular_logits_supported() -> None:
    """A (B, K) tabular head is adjusted along the class axis too."""
    z = torch.randn(8, 10)
    assert apply_dirpa(z, alpha=5.0, tau=0.5).shape == z.shape


def test_reproducible_with_generator() -> None:
    """Same seed -> same sampled prior -> identical shift."""
    z = torch.randn(2, 7, 3, 3)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    a = apply_dirpa(z, alpha=0.5, tau=1.0, generator=g1)
    b = apply_dirpa(z, alpha=0.5, tau=1.0, generator=g2)
    assert torch.allclose(a, b)


def test_adjuster_eval_is_passthrough() -> None:
    """In eval (training=False) the adjuster returns logits unchanged."""
    z = torch.randn(2, 10, 4, 4)
    adj = DirPALogitAdjuster(alpha=0.5, tau=1.0)
    assert torch.equal(adj(z, training=False), z)
    assert not torch.equal(adj(z, training=True), z)


def test_adjuster_enabled_flag() -> None:
    """enabled reflects tau > 0."""
    assert DirPALogitAdjuster(alpha=1.0, tau=0.5).enabled
    assert not DirPALogitAdjuster(alpha=1.0, tau=0.0).enabled


@pytest.mark.parametrize(
    ("alpha", "tau"),
    [(0.0, 1.0), (-1.0, 1.0), (1.0, -0.5)],
)
def test_invalid_hyperparameters_raise(alpha: float, tau: float) -> None:
    """alpha<=0 or tau<0 must fail loudly, not silently."""
    z = torch.randn(2, 5)
    with pytest.raises(ValueError):
        apply_dirpa(z, alpha=alpha, tau=tau)
