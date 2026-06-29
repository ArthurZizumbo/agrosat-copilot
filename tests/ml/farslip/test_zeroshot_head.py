"""Tests for the FarSLIP zero-shot head (US-080, :mod:`ml.farslip.zeroshot_head`).

A deterministic fake encoder (fixed image/text embeddings, no torch model, no
network) drives the cosine-softmax scoring so the head is verified offline.
"""

from __future__ import annotations

import pytest
import torch

from ml.farslip.zeroshot_head import (
    farslip_zeroshot_scores,
    farslip_zeroshot_scores_one,
)


class _FakeEncoder:
    """Returns fixed L2-normalized image/text embeddings (ignores the pixels)."""

    def __init__(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> None:
        self._image = image_embeds
        self._text = text_embeds

    def extract_embeddings(self, crops: torch.Tensor) -> torch.Tensor:
        return self._image

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        return self._text


def test_zeroshot_scores_favours_the_aligned_class() -> None:
    # Image embedding aligned with class 0 ('wheat'); text embeds are the basis.
    image = torch.tensor([[1.0, 0.0]])
    text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # wheat, corn
    scores = farslip_zeroshot_scores(
        _FakeEncoder(image, text),
        torch.zeros(1, 4, 8, 8),
        ["wheat", "corn"],
        ["wheat prompt", "corn prompt"],
        temperature=0.1,
    )
    assert len(scores) == 1
    assert scores[0]["wheat"] > scores[0]["corn"]
    assert sum(scores[0].values()) == pytest.approx(1.0, abs=1e-5)


def test_zeroshot_scores_length_mismatch_raises() -> None:
    enc = _FakeEncoder(torch.zeros(1, 2), torch.zeros(2, 2))
    with pytest.raises(ValueError):
        farslip_zeroshot_scores(enc, torch.zeros(1, 4, 8, 8), ["a"], ["t1", "t2"])


def test_zeroshot_scores_empty_classes_raises() -> None:
    enc = _FakeEncoder(torch.zeros(1, 2), torch.zeros(0, 2))
    with pytest.raises(ValueError):
        farslip_zeroshot_scores(enc, torch.zeros(1, 4, 8, 8), [], [])


def test_zeroshot_scores_one_unsqueezes_single_crop() -> None:
    image = torch.tensor([[0.0, 1.0]])  # aligned with class 1 ('corn')
    text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    out = farslip_zeroshot_scores_one(
        _FakeEncoder(image, text),
        torch.zeros(4, 8, 8),  # (C, H, W) -> unsqueezed to (1, C, H, W)
        ["wheat", "corn"],
        ["wheat prompt", "corn prompt"],
        temperature=0.1,
    )
    assert out["corn"] > out["wheat"]
