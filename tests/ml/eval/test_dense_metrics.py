"""Tests de las metricas pixel-level de segmentacion (ml.eval.dense_metrics)."""

from __future__ import annotations

import numpy as np
import torch
from matplotlib.figure import Figure

from ml.eval.dense_metrics import (
    DenseConfusionAccumulator,
    compute_dense_metrics,
    dense_confusion_figure,
)


def test_perfect_prediction_scores_one() -> None:
    """Prediccion perfecta -> mIoU = F1 = pixel_accuracy = 1."""
    target = torch.tensor([[0, 1, 2], [2, 1, 0]])
    metrics = compute_dense_metrics(target, target, num_classes=3)
    assert metrics["miou"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["pixel_accuracy"] == 1.0


def test_pixel_accuracy_half() -> None:
    """La mitad de los pixeles correctos -> pixel_accuracy = 0.5."""
    target = torch.tensor([0, 0, 1, 1])
    preds = torch.tensor([0, 0, 0, 0])
    metrics = compute_dense_metrics(preds, target, num_classes=2)
    assert metrics["pixel_accuracy"] == 0.5


def test_ignore_index_excludes_void_pixels() -> None:
    """Los pixeles con target == ignore_index no afectan las metricas."""
    # Clase 2 = void: aunque se prediga mal, no entra en el computo.
    target = torch.tensor([0, 1, 2, 2])
    preds = torch.tensor([0, 1, 0, 1])
    metrics = compute_dense_metrics(preds, target, num_classes=3, ignore_index=2)
    # Solo cuentan los pixeles 0 y 1, ambos correctos.
    assert metrics["pixel_accuracy"] == 1.0
    assert metrics["miou"] == 1.0


def test_accumulator_streaming_matches_oneshot() -> None:
    """Acumular por batches da el mismo resultado que one-shot."""
    rng = np.random.default_rng(0)
    target = rng.integers(0, 4, size=(2, 8, 8))
    preds = rng.integers(0, 4, size=(2, 8, 8))

    oneshot = compute_dense_metrics(preds, target, num_classes=4, ignore_index=3)

    acc = DenseConfusionAccumulator(4, ignore_index=3)
    acc.update(preds[0], target[0])
    acc.update(preds[1], target[1])
    streamed = acc.compute()

    for key in ("miou", "f1_macro", "pixel_accuracy"):
        assert abs(streamed[key] - oneshot[key]) < 1e-9


def test_reset_clears_state() -> None:
    """``reset`` deja el acumulador en cero."""
    acc = DenseConfusionAccumulator(3, ignore_index=2)
    acc.update(torch.tensor([0, 1]), torch.tensor([0, 1]))
    acc.reset()
    metrics = acc.compute()
    assert metrics == {"miou": 0.0, "f1_macro": 0.0, "pixel_accuracy": 0.0}


def test_dense_confusion_figure_returns_figure() -> None:
    """El helper de matriz de confusion densa devuelve una Figure."""
    target = torch.tensor([[0, 1], [1, 0]])
    preds = torch.tensor([[0, 1], [0, 0]])
    fig = dense_confusion_figure(preds, target, ignore_index=None)
    assert isinstance(fig, Figure)
