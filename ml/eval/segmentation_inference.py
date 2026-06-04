"""Inference and visualization of dense segmentation predictions.

Loads a trained checkpoint (``best.pt`` from
:mod:`ml.train.train_segmentation`), predicts over PASTIS-R patches and
generates the comparison figure ``Input (RGB) | Ground truth | Prediction``
per patch, plus the metrics of the loaded model. It is the module the
``notebooks/models/5*`` notebook invokes for visual analysis; the notebook
calls, it does not implement the logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import structlog
import torch

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = structlog.get_logger(__name__)

#: RGB bands in the PASTIS-R .npy files (S2 order of 10: B2,B3,B4,...): B4(red)=2,
#: B3(green)=1, B2(blue)=0.
_RGB_BANDS = (2, 1, 0)


def load_segmentation_model(
    checkpoint_path: Path | str,
    *,
    model_kind: Literal["deeplabv3plus", "tsvit", "tsvit-pheno"],
    num_classes: int = 18,
    n_timesteps: int = 10,
    device: str = "auto",
) -> torch.nn.Module:
    """Rebuild the model and load the checkpoint weights.

    Args:
        checkpoint_path: Path to ``best.pt`` (full training state).
        model_kind: Architecture to rebuild the exact topology.
        num_classes: Number of head classes (18 semantic or 6 HCAT).
        n_timesteps: Subsampled T (temporal models only).
        device: ``"auto"``, ``"cuda"`` or ``"cpu"``.

    Returns:
        Model in ``eval()`` mode with the best-epoch weights loaded.
    """
    from ml.models.deeplabv3plus import build_deeplabv3plus_mobilenet
    from ml.models.tsvit_wrapper import build_tsvit

    resolved_device = torch.device(
        "cuda" if (device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    )
    if model_kind == "deeplabv3plus":
        model: torch.nn.Module = build_deeplabv3plus_mobilenet(
            in_channels=10, classes=num_classes
        )
    else:
        model = build_tsvit(
            num_classes=num_classes,
            n_timesteps=n_timesteps,
            img_size=128,
            in_channels=10,
            semantic_dim=384,
        )
    ckpt = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(resolved_device).eval()
    logger.info(
        "segmentation_model_loaded",
        checkpoint=str(checkpoint_path),
        model_kind=model_kind,
        num_classes=num_classes,
        best_epoch=ckpt.get("best_metrics", {}).get("best_epoch"),
        device=str(resolved_device),
    )
    return model


@torch.no_grad()
def predict_patch(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    model_kind: str,
    doy: torch.Tensor | None = None,
) -> np.ndarray:
    """Predict the dense mask of a patch.

    Args:
        model: Model loaded in ``eval()``.
        x: Patch tensor: ``(10, H, W)`` (2D) or ``(T, 10, H, W)``
            (temporal). The batch dimension is added internally.
        model_kind: Architecture (decides whether ``doy`` is passed).
        doy: DOY vector ``(T,)`` for the temporal models.

    Returns:
        Mask ``(H, W)`` int with the predicted class per pixel (in the
        ``[0..num_classes-1]`` space).
    """
    device = next(model.parameters()).device
    xb = x.unsqueeze(0).to(device).float()
    if model_kind == "deeplabv3plus":
        logits = model(xb)
    else:
        doy_b = doy.unsqueeze(0).to(device) if doy is not None else None
        out = model(xb, doy=doy_b)
        logits = out[0] if isinstance(out, tuple) else out
    return logits.argmax(dim=1).squeeze(0).cpu().numpy()


def prediction_figure(
    rgb: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int = 18,
    ignore_index: int = 255,
    titles: tuple[str, str, str] = ("Input (RGB)", "Ground truth", "Prediction"),
) -> Figure:
    """Build the comparison figure RGB | ground truth | prediction.

    Args:
        rgb: RGB image ``(H, W, 3)`` in ``[0, 1]``.
        y_true: Ground-truth mask ``(H, W)`` (classes ``[0..C-1]`` + ``ignore_index``).
        y_pred: Predicted mask ``(H, W)``.
        num_classes: Number of classes for the discrete colormap.
        ignore_index: Value ignored in ``y_true`` (drawn neutral).
        titles: Titles of the three panels.

    Returns:
        1x3 matplotlib figure ready for ``display(fig)`` in the notebook.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors

    cmap = plt.get_cmap("tab20", num_classes)
    norm = colors.Normalize(vmin=0, vmax=num_classes - 1)

    yt = np.where(y_true == ignore_index, np.nan, y_true.astype(float))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(np.clip(rgb, 0.0, 1.0))
    axes[1].imshow(yt, cmap=cmap, norm=norm, interpolation="nearest")
    axes[2].imshow(y_pred.astype(float), cmap=cmap, norm=norm, interpolation="nearest")
    for ax, title in zip(axes, titles, strict=True):
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    return fig


def rgb_from_patch(x_2d: np.ndarray) -> np.ndarray:
    """Extract a normalized RGB image ``(H, W, 3)`` from a 2D patch.

    Takes the B4/B3/B2 bands and rescales by percentiles (2-98) for a
    reasonable visual contrast (raw S2 reflectances are dark).

    Args:
        x_2d: Patch ``(10, H, W)`` (already time-collapsed, any scale).

    Returns:
        ``(H, W, 3)`` float array in ``[0, 1]``.
    """
    rgb = np.stack([x_2d[b] for b in _RGB_BANDS], axis=-1).astype(np.float32)
    lo, hi = np.nanpercentile(rgb, 2), np.nanpercentile(rgb, 98)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)


@torch.no_grad()
def evaluate_checkpoint(
    model: torch.nn.Module,
    dataset: object,
    *,
    model_kind: str,
    num_classes: int = 18,
    ignore_index: int = 255,
    max_patches: int | None = None,
) -> tuple[dict[str, object], np.ndarray]:
    """Evaluate a checkpoint over a split accumulating the confusion matrix.

    Walks the validation ``dataset`` patch by patch, predicts and accumulates
    the dense confusion matrix; at the end it derives all metrics with
    :func:`ml.eval.metrics.dense_metrics_from_cm` (mIoU, F1-macro, pixel_acc,
    balanced accuracy, Cohen kappa, IoU and F1 per class). It is the helper the
    ``5*`` notebooks invoke to reproduce the training figures without
    retraining: they load ``best.pt`` and call here.

    Args:
        model: Model loaded in ``eval()`` (see :func:`load_segmentation_model`).
        dataset: ``PASTISSegmentationDataset`` of the validation split.
        model_kind: Architecture (decides the forward signature).
        num_classes: Number of classes (18 semantic or 6 HCAT).
        ignore_index: Value ignored in the labels.
        max_patches: If given, limits the number of evaluated patches (smoke).

    Returns:
        Tuple ``(metrics, cm)``: ``metrics`` is the full dict of
        ``dense_metrics_from_cm`` and ``cm`` the accumulated confusion matrix
        ``(num_classes, num_classes)``.
    """
    from ml.eval.metrics import dense_confusion_matrix, dense_metrics_from_cm

    n = len(dataset)  # type: ignore[arg-type]
    if max_patches is not None:
        n = min(n, max_patches)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for idx in range(n):
        x, y = dataset[idx]  # type: ignore[index]
        pred = predict_patch(model, x, model_kind=model_kind)
        cm += dense_confusion_matrix(
            pred, y.numpy(), n_classes=num_classes, ignore_index=ignore_index
        )
    metrics = dense_metrics_from_cm(cm)
    logger.info(
        "checkpoint_evaluated",
        model_kind=model_kind,
        n_patches=n,
        num_classes=num_classes,
        miou=round(float(metrics["miou"]), 4),
        f1_macro=round(float(metrics["f1_macro"]), 4),
        pixel_acc=round(float(metrics["pixel_acc"]), 4),
    )
    return metrics, cm


def predict_examples(
    model: torch.nn.Module,
    dataset: object,
    *,
    model_kind: str,
    indices: list[int],
    num_classes: int = 18,
    ignore_index: int = 255,
) -> list[Figure]:
    """Generate the RGB|GT|pred figures for a list of dataset patches.

    High-level helper for the notebook: for each index, it gets the patch,
    predicts, builds the RGB and constructs the comparison figure. For
    temporal models it collapses the series by median only for the RGB panel.

    Args:
        model: Model loaded in ``eval()``.
        dataset: ``PASTISSegmentationDataset`` (2D or temporal depending on the model).
        model_kind: Architecture.
        indices: Indices of the patches to visualize.
        num_classes: Number of classes.
        ignore_index: Ignored value.

    Returns:
        List of matplotlib figures (one per index).
    """
    figs: list[Figure] = []
    for idx in indices:
        x, y = dataset[idx]  # type: ignore[index]
        x_np = x.numpy()
        if x_np.ndim == 4:  # temporal (T,10,H,W) -> RGB from the temporal median
            rgb = rgb_from_patch(np.median(x_np, axis=0))
        else:  # 2D (10,H,W)
            rgb = rgb_from_patch(x_np)
        pred = predict_patch(model, x, model_kind=model_kind)
        figs.append(
            prediction_figure(
                rgb,
                y.numpy(),
                pred,
                num_classes=num_classes,
                ignore_index=ignore_index,
            )
        )
    return figs
