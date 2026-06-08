"""FarSLIP faithful losses — MPCL (L_loc) + InfoNCE global (L_glo).

Faithful reimplementation of the Li et al. 2025 region-category objective
(arXiv:2511.14901, equations 1-4, transcribed in
``docs/us-planning/us-036-a-v2-faithful.md`` Section 0).

This module is PURE logic (CPU, no GPU/dataset/network). It supersedes the
single-positive ``RegionCategoryAlignmentLoss`` of ``ml/farslip/distill.py``
(which uses ``F.cross_entropy`` = exactly 1 positive). The trainer integration
(T4) decides how to consume these classes; this file does not import or modify
``distill.py``.

Classes:
    - :class:`MultiPositiveRegionCategoryLoss` (``L_loc``, eq. 3-4): for a batch
      of ``R`` regions each carrying a ``category_id``, the positive set
      ``P(i)`` of anchor ``i`` is **all** the regions of the batch that share its
      category (mutual positives, SupCon-style). Computes
      ``L_loc = 1/2 (L_{R->C} + L_{C->R})``. When every category appears exactly
      once (``|P(i)|=1``) the loss reduces NUMERICALLY to
      ``F.cross_entropy(logits, targets)`` (proven by a golden test) -- this is
      why the v1 single-positive loss is a particular impoverished case.
    - :class:`GlobalImageTextLoss` (``L_glo``, eq. 1-2): symmetric CLIP InfoNCE
      between image CLS and caption CLS.
    - :func:`combine_losses`: helper for ``L_total = L_glo + lambda_loc * L_loc``.

Numerical stability is handled via ``logsumexp``; empty positive sets and
extreme temperatures never produce ``NaN`` (anchors with no positive contribute
zero, mirroring the SupCon reference implementation [Khosla et al. 2020]).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Default contrastive temperature (FarSLIP paper Section 3.3).
DEFAULT_TEMPERATURE: float = 0.07
#: Default weight of ``L_loc`` in the combination (paper Table 3 combines without
#: reweighting).
DEFAULT_LAMBDA_LOC: float = 1.0


def _validate_temperature(temperature: float) -> None:
    """Validates that the temperature is strictly positive and finite.

    Args:
        temperature: softmax temperature ``tau``.

    Raises:
        ValueError: if ``temperature`` is not a finite positive number.
    """
    if not (temperature > 0.0) or not torch.isfinite(torch.tensor(temperature)):
        raise ValueError(f"temperature must be a finite positive number: {temperature}")


def _directional_mpcl(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    """Computes one direction of the multi-positive contrastive loss.

    Implements, for a similarity-derived ``logits`` matrix ``(A, K)`` of anchors
    against keys and a boolean ``positive_mask`` ``(A, K)`` marking which keys are
    positives of each anchor::

        L = (1/A) sum_i  -(1/|P(i)|) sum_{j in P(i)} log( exp(logit_ij) / sum_k exp(logit_ik) )

    The denominator ``sum_k exp(logit_ik)`` runs over ALL keys (the full row);
    the numerator averages the log-softmax over the positives of the anchor. This
    is the SupCon ``L_out`` form (Khosla et al. 2020), numerically stable via
    ``log_softmax`` (internally a ``logsumexp``). Anchors with an empty positive
    set contribute zero (no ``NaN``) and are EXCLUDED from the ``A`` average so a
    single positive-less anchor does not deflate the loss.

    Args:
        logits: tensor ``(A, K)`` of pre-softmax scores (similarity / tau).
        positive_mask: bool tensor ``(A, K)``; ``True`` where key ``j`` is a
            positive of anchor ``i``.

    Returns:
        Scalar loss tensor (mean over anchors with at least one positive). Zero
        if no anchor has any positive.
    """
    # log p_ij = logit_ij - logsumexp_k logit_ik  (stable softmax over the row).
    log_prob = F.log_softmax(logits, dim=1)  # (A, K)
    mask = positive_mask.to(log_prob.dtype)  # (A, K)
    n_pos = mask.sum(dim=1)  # (A,) -> |P(i)|
    has_pos = n_pos > 0
    # Mean log-prob over the positives of each anchor; clamp the denominator to
    # avoid 0/0 on positive-less anchors (their contribution is zeroed below).
    pos_log_prob_sum = (log_prob * mask).sum(dim=1)  # (A,)
    mean_log_prob_pos = pos_log_prob_sum / n_pos.clamp(min=1.0)  # (A,)
    per_anchor_loss = -mean_log_prob_pos  # (A,)
    # Only average over anchors that actually have positives (SupCon convention).
    n_valid = has_pos.sum()
    if n_valid == 0:
        return logits.sum() * 0.0  # keep grad graph, value 0.0
    return (per_anchor_loss * has_pos.to(per_anchor_loss.dtype)).sum() / n_valid


class MultiPositiveRegionCategoryLoss(nn.Module):
    """Region-category Multi-Positive Contrastive Loss (FarSLIP ``L_loc``, eq. 3-4).

    For a batch of ``R`` regions, each with a visual CLS embedding and an integer
    ``category_id`` indexing a bank of ``C`` text-category prototypes, the loss is
    the symmetric average of two directions:

        - ``L_{R->C}``: each region anchor is contrasted against the ``C`` category
          texts; its positives ``P(i)`` are all category texts whose id equals the
          anchor's category (with 1 text per category this is the single matching
          column, so ``|P(i)|`` over the text axis is 1, but the multi-positive
          grouping over the region axis is what differentiates anchors that share
          a category).
        - ``L_{C->R}``: each category text anchor is contrasted against the ``R``
          regions; its positives are all regions of that category (genuine
          multi-positive set ``|P| >= 1``).

    Equivalence with v1 (golden test): when each category appears EXACTLY once in
    the batch (``R == C`` and a bijection region<->category), both directions
    reduce to ``F.cross_entropy`` and the total equals
    ``F.cross_entropy(region_visual_n @ category_text_n.T / tau, region_cat_ids)``
    to ``atol=1e-5``. This proves MPCL generalizes the single-positive v1 loss
    rather than contradicting it.

    Similarity ``S`` is cosine (both sides L2-normalized). Stability is via
    ``log_softmax``; empty positive sets never yield ``NaN``.

    Args:
        temperature: softmax temperature ``tau`` (default 0.07, paper Section 3.3).
    """

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE) -> None:
        super().__init__()
        _validate_temperature(temperature)
        self.temperature = temperature

    def forward(
        self,
        region_visual: torch.Tensor,
        category_text: torch.Tensor,
        region_cat_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Computes ``L_loc = 1/2 (L_{R->C} + L_{C->R})``.

        Args:
            region_visual: tensor ``(R, D)`` with the CLS visual embedding of each
                region (paper Section 4.3 Takeaway-1: CLS, not RoI nor pooled
                patches). Gradients flow through this tensor.
            category_text: tensor ``(C, D)`` with one text prototype per category
                (text encoder frozen upstream). Detached internally.
            region_cat_ids: long tensor ``(R,)`` with the category index in
                ``[0, C)`` of each region.

        Returns:
            Scalar loss tensor differentiable with respect to ``region_visual``.

        Raises:
            ValueError: on rank/shape mismatch or out-of-range category ids.
        """
        if region_visual.dim() != 2:
            raise ValueError(f"region_visual must be (R, D); got {tuple(region_visual.shape)}")
        if category_text.dim() != 2:
            raise ValueError(f"category_text must be (C, D); got {tuple(category_text.shape)}")
        if region_visual.shape[1] != category_text.shape[1]:
            raise ValueError(
                f"embedding dim mismatch region={region_visual.shape[1]} "
                f"category={category_text.shape[1]}"
            )
        if region_cat_ids.dim() != 1 or region_cat_ids.shape[0] != region_visual.shape[0]:
            raise ValueError(
                f"region_cat_ids must be (R,) with R={region_visual.shape[0]}; "
                f"got {tuple(region_cat_ids.shape)}"
            )
        n_categories = category_text.shape[0]
        cat_ids = region_cat_ids.long()
        if cat_ids.numel() > 0 and ((cat_ids < 0).any() or (cat_ids >= n_categories).any()):
            raise ValueError(
                f"region_cat_ids out of range [0, {n_categories}); "
                f"got min={int(cat_ids.min())} max={int(cat_ids.max())}"
            )
        if region_visual.shape[0] == 0:
            return region_visual.sum() * 0.0

        region_n = F.normalize(region_visual, p=2, dim=-1)
        category_n = F.normalize(category_text.detach(), p=2, dim=-1)

        # Cosine similarity scaled by temperature; (R, C) region-against-text.
        logits_rc = region_n @ category_n.t() / self.temperature  # (R, C)

        # ids of the category bank columns: column j represents category j.
        category_axis_ids = torch.arange(n_categories, device=cat_ids.device)

        # L_{R->C}: anchor = region (rows), keys = category texts (cols).
        # Positive of region i = the text column whose category == region's id.
        pos_rc = cat_ids.unsqueeze(1) == category_axis_ids.unsqueeze(0)  # (R, C)
        loss_rc = _directional_mpcl(logits_rc, pos_rc)

        # L_{C->R}: anchor = category text (rows), keys = regions (cols).
        # Positive of category text c = every region whose id == c (multi-positive).
        logits_cr = logits_rc.t()  # (C, R)
        pos_cr = category_axis_ids.unsqueeze(1) == cat_ids.unsqueeze(0)  # (C, R)
        loss_cr = _directional_mpcl(logits_cr, pos_cr)

        return 0.5 * (loss_rc + loss_cr)


class GlobalImageTextLoss(nn.Module):
    """Global image-text InfoNCE alignment (FarSLIP ``L_glo``, eq. 1-2).

    Standard symmetric CLIP contrastive loss between the image CLS and the caption
    CLS of a batch of ``B`` image-caption pairs::

        L_glo = 1/2 (L_{I->T} + L_{T->I})
        L_{I->T} = -(1/B) sum_i log( exp(S(V_i, T_i)/tau) / sum_j exp(S(V_i, T_j)/tau) )

    The positive of image ``i`` is its own caption ``i`` (diagonal); every other
    caption in the batch is a negative. ``S`` is cosine similarity (L2-normalized
    both sides). Implemented as two ``F.cross_entropy`` over the symmetric logits
    with diagonal targets (numerically stable, standard CLIP form).

    Args:
        temperature: softmax temperature ``tau`` (default 0.07).
    """

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE) -> None:
        super().__init__()
        _validate_temperature(temperature)
        self.temperature = temperature

    def forward(
        self,
        image_cls: torch.Tensor,
        caption_cls: torch.Tensor,
    ) -> torch.Tensor:
        """Computes ``L_glo = 1/2 (L_{I->T} + L_{T->I})``.

        Args:
            image_cls: tensor ``(B, D)`` image CLS embeddings. Gradients flow here.
            caption_cls: tensor ``(B, D)`` caption CLS embeddings (text encoder
                frozen upstream). Detached internally.

        Returns:
            Scalar loss tensor differentiable with respect to ``image_cls``.

        Raises:
            ValueError: on rank/shape mismatch between the two sides.
        """
        if image_cls.dim() != 2 or caption_cls.dim() != 2:
            raise ValueError(
                f"image_cls and caption_cls must be 2-D (B, D); got "
                f"{tuple(image_cls.shape)} and {tuple(caption_cls.shape)}"
            )
        if image_cls.shape != caption_cls.shape:
            raise ValueError(
                f"image_cls {tuple(image_cls.shape)} and caption_cls "
                f"{tuple(caption_cls.shape)} must have the same shape"
            )
        if image_cls.shape[0] == 0:
            return image_cls.sum() * 0.0

        image_n = F.normalize(image_cls, p=2, dim=-1)
        caption_n = F.normalize(caption_cls.detach(), p=2, dim=-1)

        # (B, B) symmetric logits; row i col j = S(V_i, T_j) / tau.
        logits_it = image_n @ caption_n.t() / self.temperature
        targets = torch.arange(image_n.shape[0], device=image_n.device)
        loss_it = F.cross_entropy(logits_it, targets)  # image -> text
        loss_ti = F.cross_entropy(logits_it.t(), targets)  # text -> image
        return 0.5 * (loss_it + loss_ti)


def combine_losses(
    loss_glo: torch.Tensor,
    loss_loc: torch.Tensor,
    lambda_loc: float = DEFAULT_LAMBDA_LOC,
) -> torch.Tensor:
    """Combines the two FarSLIP objectives: ``L_total = L_glo + lambda_loc * L_loc``.

    Faithful to the paper Table 3 row ``L_glo + L_loc`` (default ``lambda_loc=1.0``
    combines without reweighting). With ``lambda_loc=0`` the total equals
    ``L_glo`` exactly (ablation).

    Args:
        loss_glo: scalar global image-text loss (``L_glo``).
        loss_loc: scalar region-category MPCL loss (``L_loc``).
        lambda_loc: non-negative weight of ``L_loc`` (default 1.0).

    Returns:
        Scalar combined loss tensor (keeps the grad graph of both inputs).

    Raises:
        ValueError: if ``lambda_loc`` is negative or not finite.
    """
    if not (lambda_loc >= 0.0) or not torch.isfinite(torch.tensor(lambda_loc)):
        raise ValueError(f"lambda_loc must be a finite non-negative number: {lambda_loc}")
    return loss_glo + lambda_loc * loss_loc


__all__ = [
    "DEFAULT_LAMBDA_LOC",
    "DEFAULT_TEMPERATURE",
    "GlobalImageTextLoss",
    "MultiPositiveRegionCategoryLoss",
    "combine_losses",
]
