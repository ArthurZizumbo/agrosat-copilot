"""Definiciones de assets Dagster para AgroSatCopilot.

Convenio: cada asset declara explícitamente sus dependencias (lineage) para que
Dagster pueda materializar el DAG. Los datasets versionados con DVC se exponen
como assets con un IOManager que valida el hash ``data_version``.
"""

from dagster_project.assets.features import (
    parcel_features_fused,
    parcel_features_scaler,
    parcel_splits_spatial_kfold,
)
from dagster_project.assets.health import hello_world

__all__ = [
    "hello_world",
    "parcel_features_fused",
    "parcel_features_scaler",
    "parcel_splits_spatial_kfold",
]
