"""Paquete de entrenamiento: baseline tabular RF/XGB y modelos de segmentacion (US-019+)."""

from __future__ import annotations

# The segmentation phenology loss does not depend on heavy packages (geopandas,
# h3, etc.) and must always be importable, even in a Colab environment. The
# `train_segmentation` function is not re-exported to avoid shadowing the submodule of
# the same name (consumers import from the direct module `ml.train.train_segmentation`).
from ml.train.train_segmentation import phenology_contrastive_loss

__all__ = [
    "phenology_contrastive_loss",
]

# The tabular baseline and the phenology models depend on heavy packages
# (geopandas, h3, etc.) that are not always installed. They are imported
# tolerantly to avoid blocking the rest of the package.
try:
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

    __all__ += [
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
except ImportError:  # pragma: no cover - optional dependencies absent
    pass
