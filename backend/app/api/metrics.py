"""Prometheus scrape endpoint (US-059).

Thin router (SoC): it only delegates to
:func:`backend.app.middleware.metrics.render_latest`; all instrumentation
logic lives in the middleware. The endpoint exposes process-level metrics, so
it is intentionally NOT filtered by ``session_id`` (it carries no tenant data)
and is not rate-limited. In production it is restricted by network ingress /
Cloud Run, not by app-level auth (see ``infrastructure/grafana/README.md``).
"""

from fastapi import APIRouter
from starlette.responses import Response

from backend.app.middleware.metrics import render_latest

router = APIRouter(tags=["observability"])


@router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Expose the Prometheus metrics of the running process.

    Returns:
        The metric registry serialised in Prometheus text exposition format
        with the canonical ``CONTENT_TYPE_LATEST`` media type.
    """
    return render_latest()
