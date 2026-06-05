"""ml/ensemble/voting.py
======================
Soft-voting and hard-voting ensembles for dense semantic segmentation.

**Homogeneous ensemble**: combines models of the same family (temporal
transformers: TSViT-pheno, TSViT-base, U-TAE) by averaging their
logit tensors before argmax.  All models must share the same output
resolution and number of classes.

Usage (inference-time, no retraining)::

    from ml.ensemble.voting import SoftVotingEnsemble

    ensemble = SoftVotingEnsemble(
        checkpoint_paths=[
            "reports/segmentation/checkpoints/tsvit_pheno_best.pt",
            "reports/segmentation/checkpoints/tsvit_base_best.pt",
            "reports/segmentation/checkpoints/utae_best.pt",
        ],
        model_builders=[build_tsvit_pheno, build_tsvit_base, build_utae],
        weights=[0.50, 0.30, 0.20],   # None → uniform
        num_classes=20,
        device="cuda",
    )
    preds = ensemble.predict(dataloader)          # (N, H, W) int64
    logits = ensemble.predict_logits(dataloader)  # (N, C, H, W) float32

The module also exposes :func:`hard_vote_from_logits` and
:func:`soft_vote_from_logits` as pure-tensor utilities for use inside
notebooks when logits are already cached on disk.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Pure-tensor utilities (notebook-friendly)
# ─────────────────────────────────────────────────────────────────────────────

def soft_vote_from_logits(
    logits_list: list[torch.Tensor],
    weights: list[float] | None = None,
) -> torch.Tensor:
    """Average logits (after softmax) across models → class probabilities.

    Args:
        logits_list: list of (B, C, H, W) tensors, one per model.
        weights:     per-model scalar weights (unnormalised). None → uniform.

    Returns:
        (B, C, H, W) float32 probability tensor.
    """
    if weights is None:
        weights = [1.0] * len(logits_list)
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum()

    probs = torch.stack(
        [F.softmax(lg.float(), dim=1) for lg in logits_list], dim=0
    )  # (M, B, C, H, W)
    w = w.to(probs.device).view(-1, 1, 1, 1, 1)
    return (probs * w).sum(dim=0)  # (B, C, H, W)


def hard_vote_from_logits(
    logits_list: list[torch.Tensor],
    num_classes: int,
    weights: list[float] | None = None,
) -> torch.Tensor:
    """Majority vote on argmax predictions, with optional per-model weights.

    Args:
        logits_list:  list of (B, C, H, W) tensors.
        num_classes:  C.
        weights:      integer vote counts per model. None → 1 vote each.

    Returns:
        (B, H, W) int64 predicted class map.
    """
    if weights is None:
        weights = [1] * len(logits_list)
    # Accumulate one-hot vote counts
    B, C, H, W = logits_list[0].shape
    votes = torch.zeros(B, num_classes, H, W, device=logits_list[0].device)
    for lg, w in zip(logits_list, weights):
        pred = lg.argmax(dim=1)  # (B, H, W)
        one_hot = F.one_hot(pred, num_classes).permute(0, 3, 1, 2).float()
        votes += one_hot * float(w)
    return votes.argmax(dim=1)  # (B, H, W)


def load_cached_logits(path: str | Path) -> torch.Tensor:
    """Load a cached logit tensor saved with ``torch.save``."""
    return torch.load(str(path), map_location="cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Full inference wrapper
# ─────────────────────────────────────────────────────────────────────────────

class SoftVotingEnsemble:
    """Homogeneous soft-voting ensemble for dense segmentation.

    Loads N model checkpoints, runs inference on a DataLoader and
    returns averaged probability maps.  Designed to work with the
    PASTIS-R loaders used across the segmentation notebooks.

    Args:
        checkpoint_paths: paths to ``torch.save``'d state-dict files.
        model_builders:   callables that return an ``nn.Module`` ready
                          to receive the state-dict.  Must match the
                          order of ``checkpoint_paths``.
        weights:          per-model averaging weights (unnormalised).
                          None → uniform.
        num_classes:      number of output classes (default: 20 for PASTIS-R).
        device:           "cuda" or "cpu".
        input_key:        key inside the DataLoader batch dict for images.
        label_key:        key for ground-truth labels.
        positions_key:    key for temporal positions (U-TAE / TSViT).
                          None if the model does not need positions.
    """

    def __init__(
        self,
        checkpoint_paths: Sequence[str | Path],
        model_builders: Sequence[Callable[[], nn.Module]],
        weights: Sequence[float] | None = None,
        num_classes: int = 20,
        device: str = "cuda",
        input_key: str = "pixel_values",
        label_key: str = "labels",
        positions_key: str | None = "positions",
    ) -> None:
        assert len(checkpoint_paths) == len(model_builders), (
            "checkpoint_paths and model_builders must have the same length"
        )
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.weights = list(weights) if weights is not None else None
        self.input_key = input_key
        self.label_key = label_key
        self.positions_key = positions_key

        self.models: list[nn.Module] = []
        for ckpt, builder in zip(checkpoint_paths, model_builders):
            model = builder()
            state = torch.load(str(ckpt), map_location=self.device)
            # Support both raw state_dict and wrapped {"model_state_dict": …}
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state, strict=False)
            model.eval().to(self.device)
            self.models.append(model)

    # ------------------------------------------------------------------
    def _forward_single(
        self, model: nn.Module, batch: dict
    ) -> torch.Tensor:
        """Run one model forward pass → (B, C, H, W) logits."""
        x = batch[self.input_key].to(self.device, non_blocking=True)
        with torch.no_grad():
            if self.positions_key and self.positions_key in batch:
                pos = batch[self.positions_key].to(self.device, non_blocking=True)
                out = model(x, pos)
            else:
                out = model(x)
        # Support HuggingFace output objects
        if hasattr(out, "logits"):
            out = out.logits
        # Upsample if spatial resolution is downscaled (e.g. SegFormer /4)
        target_hw = x.shape[-2:]
        if out.shape[-2:] != target_hw:
            out = F.interpolate(out, size=target_hw, mode="bilinear", align_corners=False)
        return out.float()

    # ------------------------------------------------------------------
    def predict_logits(self, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        """Run full dataset → averaged logits and labels.

        Returns:
            (all_probs, all_labels) — shapes (N, C, H, W) and (N, H, W).
        """
        all_probs: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for batch in loader:
            logits_list = [self._forward_single(m, batch) for m in self.models]
            probs = soft_vote_from_logits(logits_list, self.weights)
            all_probs.append(probs.cpu())
            if self.label_key in batch:
                all_labels.append(batch[self.label_key])

        all_probs_t = torch.cat(all_probs, dim=0)
        all_labels_t = torch.cat(all_labels, dim=0) if all_labels else torch.tensor([])
        return all_probs_t, all_labels_t

    def predict(self, loader: DataLoader) -> torch.Tensor:
        """Run full dataset → argmax predictions (N, H, W) int64."""
        probs, _ = self.predict_logits(loader)
        return probs.argmax(dim=1)

    # ------------------------------------------------------------------
    def evaluate(
        self,
        loader: DataLoader,
        ignore_index: int = 19,
    ) -> dict[str, float]:
        """Compute mIoU, F1-macro and pixel-accuracy on ``loader``.

        Returns a dict with keys: ``miou``, ``f1_macro``, ``pixel_accuracy``.
        """
        from torchmetrics import JaccardIndex, F1Score, Accuracy

        miou_m = JaccardIndex(task="multiclass", num_classes=self.num_classes,
                              ignore_index=ignore_index, average="macro").to(self.device)
        f1_m = F1Score(task="multiclass", num_classes=self.num_classes,
                       ignore_index=ignore_index, average="macro").to(self.device)
        acc_m = Accuracy(task="multiclass", num_classes=self.num_classes,
                         ignore_index=ignore_index).to(self.device)

        t0 = time.time()
        for batch in loader:
            logits_list = [self._forward_single(m, batch) for m in self.models]
            probs = soft_vote_from_logits(logits_list, self.weights).to(self.device)
            preds = probs.argmax(dim=1)
            labels = batch[self.label_key].to(self.device)
            miou_m.update(preds, labels)
            f1_m.update(preds, labels)
            acc_m.update(preds, labels)

        return {
            "miou": miou_m.compute().item(),
            "f1_macro": f1_m.compute().item(),
            "pixel_accuracy": acc_m.compute().item(),
            "inference_time_s": time.time() - t0,
        }


class HardVotingEnsemble(SoftVotingEnsemble):
    """Hard-voting variant: majority vote on argmax, not averaged probabilities."""

    def evaluate(self, loader: DataLoader, ignore_index: int = 19) -> dict[str, float]:
        from torchmetrics import JaccardIndex, F1Score, Accuracy

        miou_m = JaccardIndex(task="multiclass", num_classes=self.num_classes,
                              ignore_index=ignore_index, average="macro").to(self.device)
        f1_m = F1Score(task="multiclass", num_classes=self.num_classes,
                       ignore_index=ignore_index, average="macro").to(self.device)
        acc_m = Accuracy(task="multiclass", num_classes=self.num_classes,
                         ignore_index=ignore_index).to(self.device)

        t0 = time.time()
        for batch in loader:
            logits_list = [self._forward_single(m, batch) for m in self.models]
            preds = hard_vote_from_logits(
                logits_list, self.num_classes, self.weights
            ).to(self.device)
            labels = batch[self.label_key].to(self.device)
            miou_m.update(preds, labels)
            f1_m.update(preds, labels)
            acc_m.update(preds, labels)

        return {
            "miou": miou_m.compute().item(),
            "f1_macro": f1_m.compute().item(),
            "pixel_accuracy": acc_m.compute().item(),
            "inference_time_s": time.time() - t0,
        }
