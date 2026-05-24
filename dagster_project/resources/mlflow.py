"""Resource MLflow para tracking de assets Dagster (US-022b-B).

Wrappea ``dagster_mlflow.mlflow_tracking`` con la configuracion estandar del
proyecto: endpoint persistente Cloud Run (US-022b-A) en vez del local
``http://localhost:5010`` que servia para el baseline.

Uso (asset):

    @asset(required_resource_keys={"mlflow"})
    def my_asset(context):
        context.resources.mlflow.log_param("data_version", "...")
        context.resources.mlflow.log_metric("n_embeddings", 30000)

El resource gestiona automaticamente el ciclo del run (start/end) si se
configura un ``experiment_name``. Si ``MLFLOW_TRACKING_URI`` no esta
disponible (CI sin endpoint, dev offline), el resource cae a un stub local
``file:./mlruns`` para no romper tests — los assets siguen reportando logs
estructurados; solo se pierde la persistencia remota.

Referencia: ``dagster_mlflow.mlflow_tracking`` (dagster-mlflow ^0.29).
"""

from __future__ import annotations

import os

from dagster import ResourceDefinition
from dagster_mlflow import mlflow_tracking

#: URI por defecto cuando no hay endpoint persistente configurado.
#: En produccion (US-022b-A) se debe sobreescribir via ``MLFLOW_TRACKING_URI``
#: apuntando al servicio Cloud Run scale-to-zero.
_DEFAULT_LOCAL_URI = "file:./mlruns"

#: Nombre del experimento MLflow para el pipeline FarSLIP.
FARSLIP_EXPERIMENT = "farslip-clip-italy"

#: Run name canonico para el modelo destilado (B-5 tag MLflow Registry).
FARSLIP_RUN_NAME = "farslip-clip-italy-v1"


def get_mlflow_tracking_uri() -> str:
    """Resuelve el URI MLflow desde env var con fallback a file local.

    Returns:
        URI del MLflow tracking server. Prioriza ``MLFLOW_TRACKING_URI``;
        si no existe usa ``file:./mlruns`` (modo offline reproducible).
    """
    return os.environ.get("MLFLOW_TRACKING_URI", _DEFAULT_LOCAL_URI)


def build_mlflow_resource(
    experiment_name: str = FARSLIP_EXPERIMENT,
    run_name: str | None = None,
    extra_tags: dict[str, str] | None = None,
) -> ResourceDefinition:
    """Construye un ``mlflow_tracking`` configurado para US-022b-B.

    Args:
        experiment_name: nombre del experimento (default ``farslip-clip-italy``).
        run_name: nombre del run (default None — MLflow autogenera).
        extra_tags: tags adicionales a propagar al run (e.g. ``data_version``).

    Returns:
        ``ResourceDefinition`` configurada lista para inyectar en ``Definitions``.

    Notes:
        El resource gestiona ciclo del run via el hook
        ``end_mlflow_on_run_finished`` de ``dagster-mlflow``. Si el asset
        necesita control granular del run (multiple runs por materializacion),
        usar la API low-level ``mlflow.start_run(...)`` directamente desde
        el cuerpo del asset.
    """
    config: dict[str, object] = {
        "experiment_name": experiment_name,
        "mlflow_tracking_uri": get_mlflow_tracking_uri(),
    }
    if run_name is not None:
        config["mlflow_run_name"] = run_name
    if extra_tags:
        config["extra_tags"] = extra_tags

    return mlflow_tracking.configured(config)


#: Resource por defecto para el pipeline FarSLIP (US-022b-B).
#: Se inyecta en ``Definitions.resources`` bajo la key ``"mlflow"``.
farslip_mlflow_resource = build_mlflow_resource(
    experiment_name=FARSLIP_EXPERIMENT,
    run_name=FARSLIP_RUN_NAME,
    extra_tags={
        # Tags B-5 del plan US-022b: data + code + model versions.
        "us": "US-022b",
        "epic": "E4",
        "pipeline": "farslip",
    },
)
