"""Entry point of the AgroSatCopilot API.

Startup: ``poetry run uvicorn backend.app.main:app --reload --port 8000``.

Routers are mounted progressively as the US are closed:
- /healthz, /readyz             — operational from the bootstrap
- /chat (SSE)                   — EPIC 7 (Google ADK agent)
- /aois, /timeseries            — EPIC 2 (satellite data)
- /stac/search, /tiles          — EPIC 2 (catalog + TiTiler)
- /llm/switch                   — EPIC 7 (A/B switch Gemini <-> Qwen3.5)
- /jobs                         — EPIC 8 (asynchronous inference via Pub/Sub)
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import aois, chat, health, llm, sessions
from backend.app.core.config import get_settings
from backend.app.core.db import dispose_engine
from backend.app.core.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifecycle: connection setup and cleanup."""
    settings = get_settings()
    configure_logging(env=settings.env, log_level=settings.log_level)
    logger = structlog.get_logger()
    logger.info("startup", env=settings.env, region=settings.gcp_region)
    yield
    await dispose_engine()
    logger.info("shutdown")


def create_app() -> FastAPI:
    """FastAPI application factory."""
    settings = get_settings()
    app = FastAPI(
        title="AgroSatCopilot API",
        version="0.1.0",
        description="SaaS conversacional agrícola con Foundation Models satelitales.",
        lifespan=lifespan,
    )
    # CORS with explicit allow_headers (SEC hardening): combining allow_credentials=True
    # with allow_headers=["*"] exposes the API to abuse. Whitelist the minimum headers
    # that the Nuxt frontend + SSE client actually send.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "Accept",
            "X-Request-ID",
            "X-Session-ID",
        ],
        expose_headers=["X-Request-ID", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
    )
    app.include_router(health.router)
    app.include_router(sessions.router)
    app.include_router(chat.router)
    app.include_router(aois.router)
    app.include_router(llm.router)
    return app


app = create_app()
