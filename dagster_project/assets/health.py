"""Smoke-test asset to verify that Dagster starts correctly.

Used to validate the environment installation (Dagster + Polars + Python 3.12)
before starting to materialize the real ingestion assets. It is removed when
the first productive asset (``alphaearth_annual``) goes into operation.
"""

from datetime import UTC, datetime

from dagster import AssetExecutionContext, MaterializeResult, MetadataValue, asset


@asset(
    group_name="bootstrap",
    description="Smoke-test asset: confirma que Dagster ejecuta y registra metadata.",
)
def hello_world(context: AssetExecutionContext) -> MaterializeResult:
    """Return a timestamp as a liveness check of the orchestrator."""
    now = datetime.now(UTC)
    context.log.info("Dagster bootstrap smoke-test executed at %s", now.isoformat())
    return MaterializeResult(
        metadata={
            "executed_at": MetadataValue.text(now.isoformat()),
            "project": MetadataValue.text("agrosat-copilot"),
        }
    )
