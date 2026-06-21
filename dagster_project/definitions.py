"""Dagster entry point — aggregates assets, resources, jobs and schedules.

Startup: ``poetry run dagster dev -m dagster_project.definitions``.

US-022b-B: registers the ``mlflow`` resource (dagster-mlflow) for tracking the
FarSLIP pipeline and the job ``farslip_full_pipeline_job`` that orchestrates the
flow ``sentinel2_crops_256 -> farslip_embeddings_italy ->
farslip_embeddings_consolidated``. The external AssetSpec
(``farslip_pairs_italy``, ``farslip_clip_italy_v1``) are added to the lineage to
visualize the flow of the paper Wen et al. 2025 / Li et al. 2025 in the UI.

US-060: registers the ``drift_check`` asset (weekly Evidently drift monitor),
its ``drift_notifier`` resource (email alert with structlog fallback) and the
first schedule of the project (``drift_check_weekly_schedule``, Mondays 06:00).
"""

from dagster import Definitions, load_assets_from_modules

from dagster_project.assets import (
    drift,
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
from dagster_project.resources import build_drift_notifier, farslip_mlflow_resource
from dagster_project.schedules import drift_check_job, drift_check_weekly_schedule

all_assets = load_assets_from_modules(
    [health, features, sentinel2_crops, farslip, farslip_pipeline, phenology_models, drift]
)

defs = Definitions(
    assets=[
        *all_assets,
        # External AssetSpec — declare the FarSLIP paper lineage without
        # materialization (MLflow model + semantic alias of pairs).
        farslip_pairs_italy_spec,
        farslip_clip_italy_v1_spec,
    ],
    resources={
        "mlflow": farslip_mlflow_resource,
        "drift_notifier": build_drift_notifier(),
    },
    jobs=[farslip_full_pipeline_job, drift_check_job],
    schedules=[drift_check_weekly_schedule],
)
