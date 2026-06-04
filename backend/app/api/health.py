"""Healthcheck endpoints for Cloud Run liveness/readiness probes."""

from datetime import UTC, datetime

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Response of the health endpoint."""

    status: str
    service: str
    timestamp: datetime


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """Liveness probe — returns 200 if the process is alive."""
    return HealthResponse(
        status="ok",
        service="agrosat-api",
        timestamp=datetime.now(UTC),
    )


@router.get("/readyz", response_model=HealthResponse)
async def readyz() -> HealthResponse:
    """Readiness probe — TODO: check Postgres + Redis once integrated."""
    return HealthResponse(
        status="ready",
        service="agrosat-api",
        timestamp=datetime.now(UTC),
    )
