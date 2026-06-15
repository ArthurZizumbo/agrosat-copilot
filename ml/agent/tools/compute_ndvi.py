"""FunctionTool: report per-parcel NDVI metrics from precomputed features.

The NDVI statistics (mean/min/max), phenology descriptors, integral
(``ndvi_auc``) and ``peak_value`` are computed upstream (EPIC 3) and stored in
the feature row. This tool only READS them through the ``ParcelReader`` port and
shapes them into :class:`Finding` objects with a citation — there is no heavy
recomputation inline (forbidden in a chat turn; see ``ml/agent/CLAUDE.md``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

from ml.agent.events import Citation, Finding

if TYPE_CHECKING:
    from ml.agent.ports import ParcelReader

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Schemas.
# ---------------------------------------------------------------------------


class ComputeNdviInput(BaseModel):
    """Input contract for :func:`compute_ndvi`."""

    session_id: str = Field(description="Tenant scope (multi-tenant NON-NEGOTIABLE).")
    aoi_id: int | None = Field(default=None, description="AOI to scope parcels.")
    year: int | None = Field(default=None, description="Feature year filter.")


class ComputeNdviOutput(BaseModel):
    """Output contract: NDVI findings plus a human-readable summary."""

    findings: list[Finding] = Field(default_factory=list)
    summary: str


# ---------------------------------------------------------------------------
# Tool entry-point.
# ---------------------------------------------------------------------------


def _ndvi_mean(stats: dict[str, float]) -> float | None:
    """Pick the mean NDVI from the stats dict tolerating key aliases."""
    for key in ("mean", "ndvi_mean", "avg"):
        if key in stats:
            return float(stats[key])
    return None


async def compute_ndvi(
    payload: ComputeNdviInput,
    *,
    parcels: ParcelReader,
) -> ComputeNdviOutput:
    """Return NDVI metrics for each parcel in the AOI from stored features.

    Args:
        payload: Validated input (session, AOI, year).
        parcels: Read-only parcel/feature port (injected from ``AgentDeps``).

    Returns:
        A :class:`ComputeNdviOutput` with one :class:`Finding` per parcel that
        has a feature row. Parcels without features are skipped (not invented).

    Note:
        ``citation.tool_call_id`` is stamped by the caller (vision agent).
    """
    records = await parcels.list_parcels_in_aoi(
        session_id=payload.session_id, aoi_id=payload.aoi_id, year=payload.year
    )

    findings: list[Finding] = []
    for rec in records:
        feature = await parcels.get_features(
            session_id=payload.session_id, parcel_id=rec.id, year=rec.year
        )
        if feature is None:
            continue

        metrics: dict[str, float] = {}
        ndvi_mean = _ndvi_mean(feature.ndvi_stats)
        for source in (feature.ndvi_stats, feature.phenology):
            for key, value in source.items():
                if value is not None:
                    metrics[key] = float(value)
        if feature.ndvi_auc is not None:
            metrics["ndvi_auc"] = float(feature.ndvi_auc)
        if feature.peak_value is not None:
            metrics["peak_value"] = float(feature.peak_value)

        findings.append(
            Finding(
                parcel_id=rec.id,
                crop_class=rec.crop_class,
                area_ha=rec.area_ha,
                ndvi_mean=ndvi_mean,
                metrics=metrics,
                citation=Citation(
                    tool_call_id="",
                    source="ndvi_stats:features",
                    parcel_id=rec.id,
                    aoi_id=rec.aoi_id,
                ),
            )
        )

    logger.info(
        "compute_ndvi_done",
        session_id=payload.session_id,
        aoi_id=payload.aoi_id,
        n_parcels=len(findings),
    )
    summary = f"Metricas NDVI calculadas para {len(findings)} parcelas."
    return ComputeNdviOutput(findings=findings, summary=summary)


__all__ = [
    "ComputeNdviInput",
    "ComputeNdviOutput",
    "compute_ndvi",
]
