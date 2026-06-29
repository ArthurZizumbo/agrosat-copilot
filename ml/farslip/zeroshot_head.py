"""FarSLIP zero-shot text<->image scoring head (US-080).

Given a parcel crop and one phenology description per candidate crop class, this
computes a per-class score by CLIP image<->text cosine similarity through FarSLIP
(the project's region-aware CLIP, :class:`ml.extractors.farslip_extractor.
FarSLIPExtractor`). Both the image and text embeddings FarSLIP returns are already
L2-normalized, so the dot product IS the cosine; a temperature-scaled softmax turns
the similarities into a per-class distribution.

This is the "perceiver" side of the US-080 second stage: the LLM writes the
per-class descriptions (:mod:`ml.farslip.class_prompts`), this head scores them
against the image, and :mod:`ml.agent.refine` fuses the result with the Voting-3
posterior. The FarSLIP encoder is injected (any object exposing
``extract_embeddings`` + ``encode_text``), so tests run with a deterministic fake
and zero network / zero GPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import structlog

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

logger = structlog.get_logger(__name__)

__all__ = ["FarSLIPEncoder", "farslip_zeroshot_scores", "farslip_zeroshot_scores_one"]

#: Default softmax temperature over the cosine similarities. Small (CLIP-style
#: ``logit_scale`` ~ 1/0.01 = 100) so the distribution is peaked on the best match.
_DEFAULT_TEMPERATURE: float = 0.01


@runtime_checkable
class FarSLIPEncoder(Protocol):
    """The slice of :class:`FarSLIPExtractor` the zero-shot head needs (injectable).

    Any object exposing these two methods works -- the real extractor in
    production, a deterministic fake in tests (no torch model, no network).
    """

    def extract_embeddings(self, crops: torch.Tensor) -> torch.Tensor:
        """Return ``(B, D)`` L2-normalized image embeddings for ``(B, C, H, W)``."""
        ...

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Return ``(N, D)`` L2-normalized text embeddings for ``N`` strings."""
        ...


def farslip_zeroshot_scores(
    encoder: FarSLIPEncoder,
    crops: torch.Tensor,
    class_names: Sequence[str],
    class_texts: Sequence[str],
    *,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> list[dict[str, float]]:
    """Score each crop against the per-class descriptions (zero-shot CLIP).

    Args:
        encoder: A :class:`FarSLIPEncoder` (the FarSLIP extractor or a test fake).
        crops: ``(B, C, H, W)`` Sentinel-2 crops for the parcels to score.
        class_names: The candidate crop class names (aligned with ``class_texts``).
        class_texts: One phenology description per class (aligned with
            ``class_names``).
        temperature: Softmax temperature over the cosine similarities.

    Returns:
        One ``{class_name: probability}`` dict per crop (the softmax over the class
        similarities), in ``crops`` order.

    Raises:
        ValueError: if ``class_names`` and ``class_texts`` differ in length or are
            empty.
    """
    import torch

    if len(class_names) != len(class_texts):
        raise ValueError(
            f"class_names ({len(class_names)}) and class_texts ({len(class_texts)}) "
            "must have the same length"
        )
    if not class_names:
        raise ValueError("class_names/class_texts must not be empty")

    image_embeds = encoder.extract_embeddings(crops)  # (B, D), L2-norm
    text_embeds = encoder.encode_text(list(class_texts))  # (N, D), L2-norm
    # Cosine similarity (both already normalized) -> temperature-scaled softmax.
    sims = image_embeds @ text_embeds.transpose(0, 1)  # (B, N)
    probs = torch.softmax(sims / max(float(temperature), 1e-6), dim=-1)  # (B, N)

    results: list[dict[str, float]] = []
    for row in probs:
        scored = {name: float(score) for name, score in zip(class_names, row.tolist(), strict=True)}
        results.append(scored)
    logger.info("farslip_zeroshot_scored", n_crops=len(results), n_classes=len(class_names))
    return results


def farslip_zeroshot_scores_one(
    encoder: FarSLIPEncoder,
    crop: torch.Tensor,
    class_names: Sequence[str],
    class_texts: Sequence[str],
    *,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> dict[str, float]:
    """Score a SINGLE crop against the per-class descriptions.

    Convenience wrapper over :func:`farslip_zeroshot_scores` for the per-parcel
    perceiver path: accepts a ``(C, H, W)`` crop (or a ``(1, C, H, W)`` batch) and
    returns the single class->probability mapping.

    Args:
        encoder: A :class:`FarSLIPEncoder`.
        crop: ``(C, H, W)`` or ``(1, C, H, W)`` Sentinel-2 crop.
        class_names: Candidate class names (aligned with ``class_texts``).
        class_texts: One phenology description per class.
        temperature: Softmax temperature.

    Returns:
        The ``{class_name: probability}`` mapping for the crop.
    """
    batch = crop if crop.dim() == 4 else crop.unsqueeze(0)
    return farslip_zeroshot_scores(
        encoder, batch, class_names, class_texts, temperature=temperature
    )[0]
