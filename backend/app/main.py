"""Punto de entrada de la API AgroSatCopilot.

Arranque: ``poetry run uvicorn backend.app.main:app --reload --port 8000``.

Routers se montan progresivamente conforme cierran las US:
- /healthz, /readyz             — operativos desde el bootstrap
- /chat (SSE)                   — EPIC 7 (Google ADK agent)
- /aois, /timeseries            — EPIC 2 (datos satelitales)
- /stac/search, /tiles          — EPIC 2 (catálogo + TiTiler)
- /llm/switch                   — EPIC 7 (switch A/B Gemini ↔ Qwen3.5)
- /jobs                         — EPIC 8 (inferencia asíncrona vía Pub/Sub)
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import health
from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Ciclo de vida de la aplicación: setup de conexiones y cleanup."""
    settings = get_settings()
    configure_logging(env=settings.env, log_level=settings.log_level)
    logger = structlog.get_logger()
    logger.info("startup", env=settings.env, region=settings.gcp_region)
    yield
    logger.info("shutdown")


def create_app() -> FastAPI:
    """Factory de la aplicación FastAPI."""
    settings = get_settings()
    app = FastAPI(
        title="AgroSatCopilot API",
        version="0.1.0",
        description="SaaS conversacional agrícola con Foundation Models satelitales.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    return app


app = create_app()
