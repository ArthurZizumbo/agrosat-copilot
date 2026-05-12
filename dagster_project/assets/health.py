"""Asset de smoke-test para verificar que Dagster arranca correctamente.

Sirve para validar la instalación del entorno (Dagster + Polars + Python 3.12)
antes de empezar a materializar los assets reales de ingesta. Se elimina cuando
el primer asset productivo (``alphaearth_annual``) entra en operación.
"""

from datetime import UTC, datetime

from dagster import AssetExecutionContext, MaterializeResult, MetadataValue, asset


@asset(
    group_name="bootstrap",
    description="Smoke-test asset: confirma que Dagster ejecuta y registra metadata.",
)
def hello_world(context: AssetExecutionContext) -> MaterializeResult:
    """Devuelve un timestamp como prueba de vida del orquestador."""
    now = datetime.now(UTC)
    context.log.info("Dagster bootstrap smoke-test executed at %s", now.isoformat())
    return MaterializeResult(
        metadata={
            "executed_at": MetadataValue.text(now.isoformat()),
            "project": MetadataValue.text("agrosat-copilot"),
        }
    )
