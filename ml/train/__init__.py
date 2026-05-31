"""Paquete de entrenamiento: baseline tabular RF/XGB y modelos de segmentacion (US-019+)."""

from __future__ import annotations

# El baseline tabular y los modelos de fenologia dependen de paquetes pesados
# (geopandas, h3, etc.) que no siempre estan instalados, por ejemplo en un entorno
# de segmentacion en Colab. Se importan de forma tolerante para no bloquear el
# resto del paquete: train_segmentation no usa ninguna de esas dependencias y debe
# poder importarse sin ellas.
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
except ImportError:  # pragma: no cover - dependencias opcionales ausentes
    __all__ = []
