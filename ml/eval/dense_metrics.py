"""Pixel-level metrics for dense semantic segmentation (EPIC 5/6).

Complements :mod:`ml.eval.metrics` (which operates at the parcel level) with the
three segmentation metrics required by the Avance 4 rubric: **mIoU**,
**F1-macro** and **pixel-accuracy**, computed at the pixel level over 2D maps.

The implementation accumulates a ``(C, C)`` confusion matrix in pure torch (no
``torchmetrics`` dependency), which allows aggregating batches in streaming
during validation and deriving the three metrics exactly at the end. The
``ignore_index`` class (void = 19 in PASTIS-R) is excluded from both the
accumulation and the macro average.
"""

from __future__ import annotations

import numpy as np
import torch
from matplotlib.figure import Figure

from ml.eval.metrics import confusion_matrix_figure

__all__ = [
    "DenseConfusionAccumulator",
    "compute_dense_metrics",
    "dense_confusion_figure",
]


def _as_long_tensor(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Convert a numpy/torch input to a ``torch.Tensor`` ``long`` on CPU."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).long()
    return x.detach().long()


class DenseConfusionAccumulator:
    """Pixel-level confusion matrix accumulator for dense metrics.

    Allows ``update`` per batch during validation and ``compute`` at the end,
    deriving mIoU, F1-macro and pixel-accuracy from the accumulated matrix. The
    ``ignore_index`` class is filtered out of the ground truth before
    accumulating.

    Attributes:
        num_classes: Number of classes in the problem.
        ignore_index: Class to ignore (contributes neither to the confusion nor
            to the macro).
    """

    def __init__(
        self,
        num_classes: int,
        *,
        ignore_index: int | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize the accumulator with a ``(C, C)`` zeroed matrix.

        Args:
            num_classes: Number of classes ``C``.
            ignore_index: Class to ignore (``None`` to ignore none).
            device: Device on which to keep the accumulated matrix.
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self._device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        """Reset the accumulated confusion matrix to zeros."""
        self._confusion = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.int64, device=self._device
        )

    def update(self, preds: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> None:
        """Accumulate a batch of predictions against the ground truth.

        Args:
            preds: Predicted class map(s), integers of any shape.
            target: Ground truth class map(s), same shape as ``preds``.

        Raises:
            ValueError: if ``preds`` and ``target`` differ in shape.
        """
        preds_t = _as_long_tensor(preds).to(self._device).reshape(-1)
        target_t = _as_long_tensor(target).to(self._device).reshape(-1)
        if preds_t.shape != target_t.shape:
            raise ValueError(
                f"`preds` and `target` must have the same number of pixels; "
                f"received {preds_t.numel()} vs {target_t.numel()}."
            )

        valid = torch.ones_like(target_t, dtype=torch.bool)
        if self.ignore_index is not None:
            valid &= target_t != self.ignore_index
        # Defensive: discard out-of-range pixels (e.g. pred==num_classes).
        valid &= (target_t >= 0) & (target_t < self.num_classes)
        valid &= (preds_t >= 0) & (preds_t < self.num_classes)

        t = target_t[valid]
        p = preds_t[valid]
        if t.numel() == 0:
            return
        indices = t * self.num_classes + p
        binned = torch.bincount(indices, minlength=self.num_classes**2)
        self._confusion += binned.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, float]:
        """Derive mIoU, F1-macro and pixel-accuracy from the accumulated matrix.

        The macro average (mIoU and F1) is taken only over the classes present in
        the ground truth (support > 0), excluding ``ignore_index``. This avoids
        biasing the metric downward due to classes absent from the val split.

        Returns:
            Dictionary with ``miou``, ``f1_macro`` and ``pixel_accuracy`` (floats
            in ``[0, 1]``). If no valid pixel was accumulated, it returns zeros.
        """
        conf = self._confusion.double()
        total = conf.sum()
        if total <= 0:
            return {"miou": 0.0, "f1_macro": 0.0, "pixel_accuracy": 0.0}

        diag = torch.diag(conf)
        row_sum = conf.sum(dim=1)  # real support per class
        col_sum = conf.sum(dim=0)  # predictions per class

        union = row_sum + col_sum - diag
        iou = torch.where(union > 0, diag / union, torch.zeros_like(diag))

        precision = torch.where(col_sum > 0, diag / col_sum, torch.zeros_like(diag))
        recall = torch.where(row_sum > 0, diag / row_sum, torch.zeros_like(diag))
        denom = precision + recall
        f1 = torch.where(denom > 0, 2 * precision * recall / denom, torch.zeros_like(diag))

        present = row_sum > 0
        if self.ignore_index is not None and 0 <= self.ignore_index < self.num_classes:
            present[self.ignore_index] = False

        n_present = int(present.sum().item())
        miou = float(iou[present].mean().item()) if n_present > 0 else 0.0
        f1_macro = float(f1[present].mean().item()) if n_present > 0 else 0.0
        pixel_accuracy = float((diag.sum() / total).item())
        return {"miou": miou, "f1_macro": f1_macro, "pixel_accuracy": pixel_accuracy}

    def per_class_iou(self) -> dict[int, float]:
        """Return the per-class IoU (for the per-class IoU barplot).

        Returns:
            Dictionary ``{class_id: iou}`` only for the classes with support in
            the ground truth (excluding ``ignore_index``). Empty if there are no
            pixels.
        """
        conf = self._confusion.double()
        if conf.sum() <= 0:
            return {}
        diag = torch.diag(conf)
        row_sum = conf.sum(dim=1)
        col_sum = conf.sum(dim=0)
        union = row_sum + col_sum - diag
        iou = torch.where(union > 0, diag / union, torch.zeros_like(diag))
        out: dict[int, float] = {}
        for c in range(self.num_classes):
            if c == self.ignore_index or row_sum[c] <= 0:
                continue
            out[c] = float(iou[c].item())
        return out


def compute_dense_metrics(
    preds: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    *,
    num_classes: int,
    ignore_index: int | None = None,
) -> dict[str, float]:
    """Compute mIoU + F1-macro + pixel-accuracy in a single pass (one-shot).

    Convenience over :class:`DenseConfusionAccumulator` to evaluate a full
    ``(preds, target)`` pair at once (tests, final evaluation).

    Args:
        preds: Predicted class map(s).
        target: Ground truth class map(s).
        num_classes: Number of classes ``C``.
        ignore_index: Class to ignore (default ``None``).

    Returns:
        Dictionary with ``miou``, ``f1_macro`` and ``pixel_accuracy``.
    """
    acc = DenseConfusionAccumulator(num_classes, ignore_index=ignore_index)
    acc.update(preds, target)
    return acc.compute()


def dense_confusion_figure(
    preds: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    *,
    class_names: dict[int, str] | None = None,
    ignore_index: int | None = None,
    normalize: bool = True,
) -> Figure:
    """Pixel-level confusion matrix reusing :func:`confusion_matrix_figure`.

    Flattens the 2D maps into pixel vectors, discards the pixels whose ground
    truth is ``ignore_index`` and delegates the rendering to the existing
    baseline helper (DRY, same visual style as the parcel-level matrices).

    Args:
        preds: Predicted class map(s).
        target: Ground truth class map(s).
        class_names: Map ``{class_id: name}`` to label the axes.
        ignore_index: Class to exclude from the plot (default ``None``).
        normalize: If ``True`` normalizes by row (per-class recall).

    Returns:
        matplotlib figure ready for ``savefig``/``display``.
    """
    p = _as_long_tensor(preds).reshape(-1).cpu().numpy()
    t = _as_long_tensor(target).reshape(-1).cpu().numpy()
    if ignore_index is not None:
        mask = t != ignore_index
        p, t = p[mask], t[mask]
    return confusion_matrix_figure(t, p, class_names=class_names, normalize=normalize)
