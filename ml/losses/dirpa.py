"""Dirichlet Prior Augmentation (DirPA) for prior-shift-robust training.

DirPA (Reuss et al., 2026, arXiv:2511.16218 / 2603.12905) tackles *prior shift*:
the train label distribution rarely matches the deployment one, so a model trained
on the (imbalanced or artificially balanced) train prior learns a biased decision
boundary that hurts minority classes at test time.

Instead of correcting probabilities post-hoc at inference, DirPA augments training:
at every step it samples a pseudo-prior from a symmetric Dirichlet (encoding "no
knowledge of the test prior") and shifts the logits by its log before the softmax.
The model thus sees many class-frequency scenarios and learns a representation that
generalizes across priors -- a dynamic feature regularizer.

Method (Eq. 4 of the paper, verbatim):

    pi_tilde^(s) ~ Dir(alpha * 1)            # symmetric Dirichlet, per step s
    z'_i <- z_i + tau * log(pi_tilde^(s))    # element-wise logit adjustment
    p_hat_i = softmax(z'_i)                  # then standard CE / focal loss

``alpha`` is the concentration: ``alpha < 1`` samples highly skewed (long-tail)
priors, ``alpha > 1`` samples near-uniform ones. ``tau`` scales the shift. At
INFERENCE nothing is applied -- the trained weights already carry the robustness,
so any downstream ensemble (e.g. the Voting-3) recombines unchanged.

This module is architecture-agnostic: it only transforms a logits tensor, so it
drops into the existing ``CrossEntropyLoss`` path of ``ml/train/train_segmentation``
for any dense (B, K, H, W) or tabular (B, K) head.
"""

from __future__ import annotations

import torch

__all__ = ["DirPALogitAdjuster", "apply_dirpa"]

#: Floor added inside the log to keep ``log(pi_tilde)`` finite when a sampled
#: prior component underflows to ~0 (Dir with alpha < 1 can draw near-zero mass).
_LOG_EPS: float = 1e-8


def apply_dirpa(
    logits: torch.Tensor,
    *,
    alpha: float,
    tau: float,
    class_dim: int = 1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return logits shifted by one symmetric-Dirichlet pseudo-prior (Eq. 4).

    A single prior ``pi_tilde ~ Dir(alpha * 1_K)`` is drawn for the call and the
    same shift is broadcast over every non-class dimension (batch, spatial), which
    matches the paper: the augmentation perturbs the *class* prior, not per-pixel.

    Args:
        logits: Raw model logits with ``K`` classes along ``class_dim``
            (e.g. ``(B, K, H, W)`` dense or ``(B, K)`` tabular).
        alpha: Dirichlet concentration. ``< 1`` skewed/long-tail priors, ``> 1``
            near-uniform. Must be ``> 0``.
        tau: Non-negative scale of the logit shift. ``0`` disables the adjustment.
        class_dim: Axis indexing the ``K`` classes.
        generator: Optional RNG for reproducible sampling (e.g. per-seed A/B runs).

    Returns:
        The adjusted logits ``z' = z + tau * log(pi_tilde)``, same shape/dtype as
        ``logits``. When ``tau == 0`` the input is returned unchanged.

    Raises:
        ValueError: If ``alpha <= 0`` or ``tau < 0``.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if tau < 0:
        raise ValueError(f"tau must be >= 0, got {tau}")
    if tau == 0:
        return logits

    k = logits.shape[class_dim]
    # Gamma(alpha, 1) per class, normalized -> a Dirichlet(alpha * 1_K) sample.
    conc = torch.full((k,), float(alpha), device=logits.device, dtype=torch.float32)
    gamma = torch._standard_gamma(conc, generator=generator)
    pi_tilde = gamma / gamma.sum()
    log_prior = torch.log(pi_tilde + _LOG_EPS).to(logits.dtype)

    # Broadcast the K-vector over all non-class axes.
    shape = [1] * logits.ndim
    shape[class_dim] = k
    return logits + tau * log_prior.view(shape)


class DirPALogitAdjuster:
    """Stateful, toggleable DirPA wrapper for a training loop.

    Holds the hyper-parameters and only perturbs logits while ``training`` is
    ``True``; in eval it is a pass-through, so the same call site is safe in both
    phases. Sampling a fresh prior per ``__call__`` reproduces the paper's
    per-step augmentation.

    Attributes:
        alpha: Dirichlet concentration.
        tau: Logit-shift scale (``0`` => no-op, i.e. plain training).
        class_dim: Class axis of the logits.
    """

    def __init__(self, *, alpha: float, tau: float, class_dim: int = 1) -> None:
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        if tau < 0:
            raise ValueError(f"tau must be >= 0, got {tau}")
        self.alpha = alpha
        self.tau = tau
        self.class_dim = class_dim

    @property
    def enabled(self) -> bool:
        """Whether the adjuster actually shifts logits (``tau > 0``)."""
        return self.tau > 0

    def __call__(
        self, logits: torch.Tensor, *, training: bool, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Adjust ``logits`` during training; pass through in eval.

        Args:
            logits: Raw model logits, ``K`` classes on ``class_dim``.
            training: When ``False`` the logits are returned unchanged (DirPA is a
                train-only augmentation; inference uses the learned weights as-is).
            generator: Optional RNG for reproducible per-step sampling.

        Returns:
            Adjusted logits in training, the input otherwise.
        """
        if not training or self.tau == 0:
            return logits
        return apply_dirpa(
            logits,
            alpha=self.alpha,
            tau=self.tau,
            class_dim=self.class_dim,
            generator=generator,
        )
