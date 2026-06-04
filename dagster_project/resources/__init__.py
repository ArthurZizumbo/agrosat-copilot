"""Dagster resources of the AgroSatCopilot project.

Each external resource (MLflow, GCS, Earth Engine, Postgres) is declared here
and injected into ``Definitions.resources`` so the assets receive
already-configured clients (do not instantiate in the asset body).
"""

from dagster_project.resources.mlflow import (
    FARSLIP_EXPERIMENT,
    FARSLIP_RUN_NAME,
    build_mlflow_resource,
    farslip_mlflow_resource,
    get_mlflow_tracking_uri,
)

__all__ = [
    "FARSLIP_EXPERIMENT",
    "FARSLIP_RUN_NAME",
    "build_mlflow_resource",
    "farslip_mlflow_resource",
    "get_mlflow_tracking_uri",
]
