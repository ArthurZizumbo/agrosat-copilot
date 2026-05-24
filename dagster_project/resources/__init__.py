"""Resources Dagster del proyecto AgroSatCopilot.

Cada resource externo (MLflow, GCS, Earth Engine, Postgres) se declara aqui
y se inyecta en ``Definitions.resources`` para que los assets reciban
clientes ya configurados (no instanciar en el cuerpo del asset).
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
