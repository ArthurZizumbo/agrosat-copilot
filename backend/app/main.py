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

# slowapi ships an async-friendly handler that builds the ``429`` response (with
# the ``Retry-After`` / ``X-RateLimit-*`` headers when enabled). Imported under a
# private alias so it is not re-exported from this module.
from slowapi import _rate_limit_exceeded_handler as _slowapi_rate_limit_handler
from slowapi.errors import RateLimitExceeded

from backend.app.api import aois, chat, health, llm, stac, tiles, timeseries
from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging
from backend.app.core.rate_limit import limiter


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifecycle: connection setup and cleanup."""
    settings = get_settings()
    configure_logging(env=settings.env, log_level=settings.log_level)
    logger = structlog.get_logger()
    logger.info("startup", env=settings.env, region=settings.gcp_region)
    yield
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
    # Per-session rate limiting (US-052). slowapi reads the limiter from
    # ``app.state.limiter``; the ``@limiter.limit`` decorator on ``/chat`` does
    # the enforcement and the registered handler renders a JSON ``429`` (no
    # global ``SlowAPIMiddleware`` -- only ``/chat`` is limited, keyed per
    # session). The handler is evaluated before the SSE stream opens.
    app.state.limiter = limiter
    # slowapi types its handler against the concrete ``RateLimitExceeded`` while
    # Starlette's signature expects the base ``Exception``; the registration is
    # the documented slowapi pattern, so the variance is narrowed here.
    app.add_exception_handler(RateLimitExceeded, _slowapi_rate_limit_handler)  # type: ignore[arg-type]
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
    app.include_router(chat.router)
    # US-054 hot-swap of the per-session reasoner variant (session-scoped, RLS).
    app.include_router(llm.router)
    # US-053 geospatial data endpoints (all session-scoped via RLS).
    app.include_router(aois.router)
    app.include_router(timeseries.router)
    app.include_router(stac.router)
    # TiTiler mount point (US-055): replace this documented ``501`` stub router
    # with ``TilerFactory(...).router`` (prefix "/tiles") to serve dynamic COG
    # tiles. The contract path ``GET /tiles/{z}/{x}/{y}.png`` stays identical.
    app.include_router(tiles.router)
    return app


app = create_app()
