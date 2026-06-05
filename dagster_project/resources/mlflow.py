"""MLflow resource for tracking Dagster assets (US-022b-B).

Wraps ``dagster_mlflow.mlflow_tracking`` with the project's standard
configuration: persistent Cloud Run endpoint (US-022b-A) instead of the local
``http://localhost:5010`` that served for the baseline.

Usage (asset):

    @asset(required_resource_keys={"mlflow"})
    def my_asset(context):
        context.resources.mlflow.log_param("data_version", "...")
        context.resources.mlflow.log_metric("n_embeddings", 30000)

The resource automatically manages the run lifecycle (start/end) if an
``experiment_name`` is configured. If ``MLFLOW_TRACKING_URI`` is not
available (CI without endpoint, offline dev), the resource falls back to a local
stub ``file:./mlruns`` so as not to break tests — the assets keep reporting
structured logs; only the remote persistence is lost.

Reference: ``dagster_mlflow.mlflow_tracking`` (dagster-mlflow ^0.29).
"""

from __future__ import annotations

import os

from dagster import ResourceDefinition
from dagster_mlflow import mlflow_tracking

#: Default URI when no persistent endpoint is configured.
#: In production (US-022b-A) it must be overridden via ``MLFLOW_TRACKING_URI``
#: pointing to the Cloud Run scale-to-zero service.
_DEFAULT_LOCAL_URI = "file:./mlruns"

#: MLflow experiment name for the FarSLIP pipeline.
FARSLIP_EXPERIMENT = "farslip-clip-italy"

#: Canonical run name for the distilled model (B-5 MLflow Registry tag).
FARSLIP_RUN_NAME = "farslip-clip-italy-v1"


def get_mlflow_tracking_uri() -> str:
    """Resolves the MLflow URI from env var with fallback to local file.

    Returns:
        URI of the MLflow tracking server. Prioritizes ``MLFLOW_TRACKING_URI``;
        if it does not exist it uses ``file:./mlruns`` (reproducible offline mode).
    """
    return os.environ.get("MLFLOW_TRACKING_URI", _DEFAULT_LOCAL_URI)


def build_mlflow_resource(
    experiment_name: str = FARSLIP_EXPERIMENT,
    run_name: str | None = None,
    extra_tags: dict[str, str] | None = None,
) -> ResourceDefinition:
    """Builds an ``mlflow_tracking`` configured for US-022b-B.

    Args:
        experiment_name: experiment name (default ``farslip-clip-italy``).
        run_name: run name (default None — MLflow autogenerates).
        extra_tags: additional tags to propagate to the run (e.g. ``data_version``).

    Returns:
        Configured ``ResourceDefinition`` ready to inject into ``Definitions``.

    Notes:
        The resource manages the run lifecycle via the
        ``end_mlflow_on_run_finished`` hook of ``dagster-mlflow``. If the asset
        needs granular control of the run (multiple runs per materialization),
        use the low-level API ``mlflow.start_run(...)`` directly from
        the asset body.
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


#: Default resource for the FarSLIP pipeline (US-022b-B).
#: Injected into ``Definitions.resources`` under the key ``"mlflow"``.
farslip_mlflow_resource = build_mlflow_resource(
    experiment_name=FARSLIP_EXPERIMENT,
    run_name=FARSLIP_RUN_NAME,
    extra_tags={
        # B-5 tags of the US-022b plan: data + code + model versions.
        "us": "US-022b",
        "epic": "E4",
        "pipeline": "farslip",
    },
)
