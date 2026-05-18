"""Definiciones de assets Dagster para AgroSatCopilot.

Convenio: cada asset declara explícitamente sus dependencias (lineage) para que
Dagster pueda materializar el DAG. Los datasets versionados con DVC se exponen
como assets con un IOManager que valida el hash ``data_version``.
"""

from dagster_project.assets.farslip import farslip_embeddings_italy
from dagster_project.assets.features import (
    parcel_features_fused,
    parcel_features_scaler,
    parcel_splits_spatial_kfold,
)
from dagster_project.assets.health import hello_world
from dagster_project.assets.sentinel2_crops import (
    ITALY_REGIONS,
    sentinel2_crops_256,
)

__all__ = [
    "ITALY_REGIONS",
    "farslip_embeddings_italy",
    "hello_world",
    "parcel_features_fused",
    "parcel_features_scaler",
    "parcel_splits_spatial_kfold",
    "sentinel2_crops_256",
]
