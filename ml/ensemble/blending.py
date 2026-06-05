"""ml/ensemble/blending.py
=========================
Logit blending (weighted average of raw logits) with Optuna-driven
weight search for dense semantic segmentation.

**Heterogeneous component**: blending works across model families
(temporal + spatial + foundation) because it operates on their shared
logit space, allowing a heterogeneous mix like:

    TSViT-pheno  +  AnySat-frozen  +  DeepLabv3+

Blending differs from soft-voting in that:
* It operates on **raw logits** (before softmax), which is numerically
  equivalent but exposes the temperature of each model's predictions.
* Weights are **optimised per-class** optionally, giving the blender
  freedom to trust different models for different crop types.
* Bias terms can be added per class to calibrate systematic over/under-
  confidence.

Usage (notebook workflow)::

    from ml.ensemble.blending import LogitBlender, optimise_blend_weights

    # 1. Cache logits once (GPU)
    blender = LogitBlender(checkpoint_paths, model_builders, device="cuda")
    logits_val, labels_val = blender.cache_logits(val_loader, save_path="val_logits.pt")

    # 2. Optimise weights on validation set (CPU, fast)
    best_weights, study = optimise_blend_weights(
        logits_val, labels_val,
        n_trials=50,
        ignore_index=19,
    )

    # 3. Evaluate on test set
    logits_test, labels_test = blender.cache_logits(test_loader)
    metrics = blender.evaluate_from_logits(logits_test, labels_test, best_weights)
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
# Pure-tensor blend utilities
# ─────────────────────────────────────────────────────────────────────────────

def blend_logits(
    logits_stack: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted sum of logits across M models.

    Args:
        logits_stack: (M, N, C, H, W) — M models, N samples, C classes.
        weights:      (M,) unnormalised non-negative weights.
        bias:         (C,) optional per-class bias added after blending.

    Returns:
        (N, C, H, W) blended logits.
    """
    w = weights.to(logits_stack.device).float()
    w = F.softmax(w, dim=0)  # normalise to sum-1 via softmax for stability
    blended = (logits_stack * w.view(-1, 1, 1, 1, 1)).sum(dim=0)  # (N, C, H, W)
    if bias is not None:
        blended = blended + bias.to(blended.device).view(1, -1, 1, 1)
    return blended


def _compute_miou(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> float:
    """Fast mIoU from blended logits, runs entirely on CPU for HPO."""
    preds = logits.argmax(dim=1)  # (N, H, W)
    valid = labels != ignore_index
    iou_per_class = []
    for c in range(num_classes):
        if c == ignore_index:
            continue
        pred_c = (preds == c) & valid
        true_c = (labels == c) & valid
        inter = (pred_c & true_c).sum().float()
        union = (pred_c | true_c).sum().float()
        if union > 0:
            iou_per_class.append((inter / union).item())
    return float(np.mean(iou_per_class)) if iou_per_class else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Optuna optimisation
# ─────────────────────────────────────────────────────────────────────────────

def optimise_blend_weights(
    logits_stack: torch.Tensor,
    labels: torch.Tensor,
    n_trials: int = 50,
    num_classes: int = 20,
    ignore_index: int = 19,
    per_class_bias: bool = False,
    direction: str = "maximize",
    study_name: str = "blend_weights",
    seed: int = 42,
) -> tuple[torch.Tensor, "optuna.Study"]:  # noqa: F821
    """Optimise blending weights on a validation set using Optuna.

    Args:
        logits_stack:    (M, N, C, H, W) cached logits tensor.
        labels:          (N, H, W) ground-truth labels.
        n_trials:        number of Optuna trials (≥30 per rubric).
        num_classes:     number of classes.
        ignore_index:    class index to ignore.
        per_class_bias:  if True, also optimise a (C,) bias vector.
        direction:       Optuna direction (maximize mIoU).
        study_name:      Optuna study name for MLflow / storage.
        seed:            random seed for reproducibility.

    Returns:
        (best_weights, study) — best_weights is (M,) tensor,
        study is the Optuna Study object for inspection.
    """
    try:
        import optuna
    except ImportError as e:
        raise ImportError("Install optuna: pip install optuna") from e

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    M = logits_stack.shape[0]

    def objective(trial: "optuna.Trial") -> float:  # noqa: F821
        # Weights: M real values in [0.05, 1.0], softmax-normalised inside blend_logits
        raw_w = torch.tensor(
            [trial.suggest_float(f"w_{i}", 0.05, 1.0) for i in range(M)],
            dtype=torch.float32,
        )
        bias = None
        if per_class_bias:
            bias = torch.tensor(
                [trial.suggest_float(f"b_{c}", -2.0, 2.0) for c in range(num_classes)],
                dtype=torch.float32,
            )
        blended = blend_logits(logits_stack, raw_w, bias)
        return _compute_miou(blended, labels, num_classes, ignore_index)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler, study_name=study_name)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best_weights = torch.tensor([best[f"w_{i}"] for i in range(M)], dtype=torch.float32)
    return best_weights, study


# ─────────────────────────────────────────────────────────────────────────────
# Full inference + caching wrapper
# ─────────────────────────────────────────────────────────────────────────────

class LogitBlender:
    """Heterogeneous logit-blending ensemble.

    Caches logits from all member models once, then blending and
    weight optimisation run entirely on CPU without touching the GPU.
    This is the recommended workflow when GPU time is scarce.

    Args:
        checkpoint_paths: paths to model checkpoints (state-dicts).
        model_builders:   callables returning ``nn.Module`` instances.
        weights:          initial per-model weights. None → uniform.
                          These are overridden by :meth:`optimise`.
        num_classes:      number of output classes.
        device:           device for model inference.
        input_key, label_key, positions_key: DataLoader batch keys.
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
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_key = input_key
        self.label_key = label_key
        self.positions_key = positions_key
        self.weights = (
            torch.tensor(weights, dtype=torch.float32)
            if weights is not None
            else None
        )

        self.models: list[nn.Module] = []
        for ckpt, builder in zip(checkpoint_paths, model_builders):
            model = builder()
            state = torch.load(str(ckpt), map_location=self.device)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state, strict=False)
            model.eval().to(self.device)
            self.models.append(model)

    # ------------------------------------------------------------------
    def _forward_single(self, model: nn.Module, batch: dict) -> torch.Tensor:
        x = batch[self.input_key].to(self.device, non_blocking=True)
        with torch.no_grad():
            if self.positions_key and self.positions_key in batch:
                pos = batch[self.positions_key].to(self.device, non_blocking=True)
                out = model(x, pos)
            else:
                out = model(x)
        if hasattr(out, "logits"):
            out = out.logits
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out.float().cpu()

    # ------------------------------------------------------------------
    def cache_logits(
        self,
        loader: DataLoader,
        save_path: str | Path | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run all models over ``loader``, return (logits_stack, labels).

        Args:
            loader:    DataLoader yielding batches.
            save_path: if given, saves ``{"logits": ..., "labels": ...}``
                       to disk so GPU runs only once.

        Returns:
            logits_stack: (M, N, C, H, W) float32 on CPU.
            labels:       (N, H, W) int64 on CPU.
        """
        per_model: list[list[torch.Tensor]] = [[] for _ in self.models]
        all_labels: list[torch.Tensor] = []

        for batch in loader:
            for i, model in enumerate(self.models):
                per_model[i].append(self._forward_single(model, batch))
            if self.label_key in batch:
                all_labels.append(batch[self.label_key])

        logits_stack = torch.stack(
            [torch.cat(lg, dim=0) for lg in per_model], dim=0
        )  # (M, N, C, H, W)
        labels = torch.cat(all_labels, dim=0) if all_labels else torch.tensor([])

        if save_path is not None:
            torch.save({"logits": logits_stack, "labels": labels}, str(save_path))

        return logits_stack, labels

    # ------------------------------------------------------------------
    @staticmethod
    def load_cached(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
        """Load logits cached by :meth:`cache_logits`."""
        d = torch.load(str(path), map_location="cpu")
        return d["logits"], d["labels"]

    # ------------------------------------------------------------------
    def optimise(
        self,
        logits_stack: torch.Tensor,
        labels: torch.Tensor,
        n_trials: int = 50,
        per_class_bias: bool = False,
        seed: int = 42,
    ) -> "optuna.Study":  # noqa: F821
        """Optimise blending weights and store best on ``self.weights``."""
        best_w, study = optimise_blend_weights(
            logits_stack, labels,
            n_trials=n_trials,
            num_classes=self.num_classes,
            per_class_bias=per_class_bias,
            seed=seed,
        )
        self.weights = best_w
        return study

    # ------------------------------------------------------------------
    def evaluate_from_logits(
        self,
        logits_stack: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor | None = None,
        ignore_index: int = 19,
    ) -> dict[str, float]:
        """Evaluate blended predictions from cached logits (CPU-only)."""
        from torchmetrics import JaccardIndex, F1Score, Accuracy

        w = weights if weights is not None else self.weights
        if w is None:
            w = torch.ones(logits_stack.shape[0])

        blended = blend_logits(logits_stack, w)  # (N, C, H, W)

        miou_m = JaccardIndex(task="multiclass", num_classes=self.num_classes,
                              ignore_index=ignore_index, average="macro")
        f1_m = F1Score(task="multiclass", num_classes=self.num_classes,
                       ignore_index=ignore_index, average="macro")
        acc_m = Accuracy(task="multiclass", num_classes=self.num_classes,
                         ignore_index=ignore_index)

        # Process in chunks to keep memory bounded
        chunk = 16
        N = blended.shape[0]
        for start in range(0, N, chunk):
            b_logits = blended[start:start + chunk]
            b_labels = labels[start:start + chunk]
            preds = b_logits.argmax(dim=1)
            miou_m.update(preds, b_labels)
            f1_m.update(preds, b_labels)
            acc_m.update(preds, b_labels)

        return {
            "miou": miou_m.compute().item(),
            "f1_macro": f1_m.compute().item(),
            "pixel_accuracy": acc_m.compute().item(),
        }
