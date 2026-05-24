"""Punto de entrada Dagster — agrega assets, resources, jobs y schedules.

Arranque: ``poetry run dagster dev -m dagster_project.definitions``.

US-022b-B: registra resource ``mlflow`` (dagster-mlflow) para tracking del
pipeline FarSLIP y el job ``farslip_full_pipeline_job`` que orquesta el flujo
``sentinel2_crops_256 -> farslip_embeddings_italy ->
farslip_embeddings_consolidated``. Los AssetSpec externos
(``farslip_pairs_italy``, ``farslip_clip_italy_v1``) se anaden al lineage para
visualizar el flujo del paper Wen et al. 2025 / Li et al. 2025 en la UI.
"""

from dagster import Definitions, load_assets_from_modules

from dagster_project.assets import (
    farslip,
    farslip_pipeline,
    features,
    health,
    phenology_models,
    sentinel2_crops,
)
from dagster_project.assets.farslip_pipeline import (
    farslip_clip_italy_v1_spec,
    farslip_pairs_italy_spec,
)
from dagster_project.jobs import farslip_full_pipeline_job
from dagster_project.resources import farslip_mlflow_resource

all_assets = load_assets_from_modules(
    [health, features, sentinel2_crops, farslip, farslip_pipeline, phenology_models]
)

defs = Definitions(
    assets=[
        *all_assets,
        # External AssetSpec — declaran lineage del paper FarSLIP sin
        # materializacion (modelo MLflow + alias semantico de pairs).
        farslip_pairs_italy_spec,
        farslip_clip_italy_v1_spec,
    ],
    resources={
        "mlflow": farslip_mlflow_resource,
    },
    jobs=[farslip_full_pipeline_job],
)
