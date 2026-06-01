"""Paquete de entrenamiento: baseline tabular RF/XGB y modelos de segmentacion (US-019+)."""

from __future__ import annotations

# El loss fenologico de segmentacion no depende de paquetes pesados (geopandas,
# h3, etc.) y debe poder importarse siempre, incluso en un entorno de Colab. No
# se re-exporta la funcion `train_segmentation` para no sombrear el submodulo del
# mismo nombre (los consumidores importan del modulo directo `ml.train.train_segmentation`).
from ml.train.train_segmentation import phenology_contrastive_loss

__all__ = [
    "phenology_contrastive_loss",
]

# El baseline tabular y los modelos de fenologia dependen de paquetes pesados
# (geopandas, h3, etc.) que no siempre estan instalados. Se importan de forma
# tolerante para no bloquear el resto del paquete.
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
except ImportError:  # pragma: no cover - dependencias opcionales ausentes
    pass
