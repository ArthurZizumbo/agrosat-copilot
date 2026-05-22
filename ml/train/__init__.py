"""Paquete de entrenamiento: baseline tabular RF/XGB y futuros modelos (US-019+)."""

from __future__ import annotations

from ml.train.baseline import (
    BaselineResult,
    ModelKind,
    evaluate_with_spatial_cv,
    train_one_model,
    tune_baseline,
)

__all__ = [
    "BaselineResult",
    "ModelKind",
    "evaluate_with_spatial_cv",
    "train_one_model",
    "tune_baseline",
]
