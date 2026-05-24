"""Definiciones de assets Dagster para AgroSatCopilot.

Convenio: cada asset declara explícitamente sus dependencias (lineage) para que
Dagster pueda materializar el DAG. Los datasets versionados con DVC se exponen
como assets con un IOManager que valida el hash ``data_version``.
"""

from dagster_project.assets.farslip import farslip_embeddings_italy
from dagster_project.assets.farslip_pipeline import (
    FARSLIP_MODEL_ASSET_KEY,
    farslip_clip_italy_v1_spec,
    farslip_embeddings_consolidated,
    farslip_pairs_italy_spec,
)
from dagster_project.assets.features import (
    parcel_features_fused,
    parcel_features_scaler,
    parcel_splits_spatial_kfold,
)
from dagster_project.assets.health import hello_world
from dagster_project.assets.phenology_models import (
    phenology_model_inceptiontime,
    phenology_model_tempcnn,
    temporal_models_comparison,
)
from dagster_project.assets.sentinel2_crops import (
    ITALY_REGIONS,
    sentinel2_crops_256,
)

__all__ = [
    "FARSLIP_MODEL_ASSET_KEY",
    "ITALY_REGIONS",
    "farslip_clip_italy_v1_spec",
    "farslip_embeddings_consolidated",
    "farslip_embeddings_italy",
    "farslip_pairs_italy_spec",
    "hello_world",
    "parcel_features_fused",
    "parcel_features_scaler",
    "parcel_splits_spatial_kfold",
    "phenology_model_inceptiontime",
    "phenology_model_tempcnn",
    "sentinel2_crops_256",
    "temporal_models_comparison",
]
