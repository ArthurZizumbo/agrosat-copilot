"""Healthcheck endpoints para Cloud Run liveness/readiness probes."""

from datetime import UTC, datetime

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Respuesta del endpoint de salud."""

    status: str
    service: str
    timestamp: datetime


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """Liveness probe — responde 200 si el proceso está vivo."""
    return HealthResponse(
        status="ok",
        service="agrosat-api",
        timestamp=datetime.now(UTC),
    )


@router.get("/readyz", response_model=HealthResponse)
async def readyz() -> HealthResponse:
    """Readiness probe — TODO: verificar Postgres + Redis cuando estén integrados."""
    return HealthResponse(
        status="ready",
        service="agrosat-api",
        timestamp=datetime.now(UTC),
    )
