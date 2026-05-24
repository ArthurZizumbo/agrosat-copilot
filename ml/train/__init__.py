"""Paquete de entrenamiento: baseline tabular RF/XGB y futuros modelos (US-019+)."""

from __future__ import annotations

from ml.train.baseline import (
    BaselineResult,
    ModelKind,
    evaluate_with_spatial_cv,
    train_one_model,
    tune_baseline,
)
from ml.train.phenology_models import (
    TemporalDataset,
    TemporalModelKind,
    TemporalModelResult,
    build_temporal_tensor,
    train_temporal_model,
)

__all__ = [
    "BaselineResult",
    "ModelKind",
    "TemporalDataset",
    "TemporalModelKind",
    "TemporalModelResult",
    "build_temporal_tensor",
    "evaluate_with_spatial_cv",
    "train_one_model",
    "train_temporal_model",
    "tune_baseline",
]
