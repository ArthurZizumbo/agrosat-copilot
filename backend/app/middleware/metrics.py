"""Prometheus instrumentation middleware for the FastAPI app (US-059).

Exposes real, per-process HTTP metrics so Grafana can derive latency
percentiles (p50/p95/p99), request rate (RPS) and error rate per endpoint.
The metrics are emitted from the live request path -- every request that
reaches the server increments the counters and observes the histogram. No
synthetic traffic is fabricated here (Arthur's real-data rule); the chat
panels are populated with real ``/chat`` traffic in US-065.

Cardinality control: the ``path`` label is normalised to the *route template*
(``request.scope["route"].path``, e.g. ``/aois/{aoi_id}``) and never the raw
URL, so a per-``session_id`` / per-``parcel_id`` series explosion cannot blow
up the Prometheus time-series database (Riesgo R1).

Single-worker note: ``prometheus_client`` uses a global registry per process.
With one uvicorn worker (the Cloud Run target) this is correct out of the box;
multi-worker deployments require ``PROMETHEUS_MULTIPROC_DIR`` (multiprocess
mode) or a push gateway (see ``infrastructure/grafana/README.md``).
"""

import time

import structlog
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Match
from starlette.types import ASGIApp

logger = structlog.get_logger()

# Path used for requests that did not match any route (404 / unmatched). A
# constant label keeps probes and scanners from inflating cardinality.
UNMATCHED_PATH = "<unmatched>"

# Endpoint whose own scrape traffic is excluded from the latency histogram and
# request counters: instrumenting ``/metrics`` would pollute the percentiles
# with the scraper's own polling cadence.
METRICS_PATH = "/metrics"

# Latency buckets (seconds) tuned for a web API: sub-100ms health probes up to
# the 3s p99 alert threshold and beyond. ``histogram_quantile`` in Grafana
# interpolates p50/p95/p99 from these bucket boundaries.
LATENCY_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    3.0,
    5.0,
    10.0,
)


#: Dedicated process registry for the app's HTTP metrics. A *dedicated*
#: registry (not the implicit ``prometheus_client`` global ``REGISTRY``) is what
#: makes re-importing / re-instantiating the app idempotent: the collectors are
#: built exactly once via :func:`get_default_metrics` and reused, so calling
#: ``create_app()`` twice in one process (module import + the integration-test
#: factory) no longer raises ``Duplicated timeseries`` (US-059 fix). Tests that
#: want isolation pass their own :class:`~prometheus_client.CollectorRegistry`.
DEFAULT_REGISTRY = CollectorRegistry()

#: Process-singleton collectors registered on :data:`DEFAULT_REGISTRY`, built on
#: first use. ``None`` until :func:`get_default_metrics` initialises it.
_default_metrics: tuple[Histogram, Counter, Counter] | None = None


def get_default_metrics() -> tuple[Histogram, Counter, Counter]:
    """Return the process-singleton HTTP collectors on :data:`DEFAULT_REGISTRY`.

    Built once and memoised so repeated :func:`~backend.app.main.create_app`
    calls (module import plus the test factory) reuse the same collectors
    instead of re-registering them, which ``prometheus_client`` rejects with
    ``Duplicated timeseries``.

    Returns:
        The shared ``(duration_histogram, requests_counter, exceptions_counter)``.
    """
    global _default_metrics
    if _default_metrics is None:
        _default_metrics = build_request_metrics(DEFAULT_REGISTRY)
    return _default_metrics


def build_request_metrics(
    registry: CollectorRegistry | None = None,
) -> tuple[Histogram, Counter, Counter]:
    """Build the HTTP request metric collectors.

    Factory used by :func:`get_default_metrics` (the shared
    :data:`DEFAULT_REGISTRY`) and by tests (an isolated
    :class:`~prometheus_client.CollectorRegistry`) so unit tests do not collide
    with the process registry.

    Args:
        registry: Optional Prometheus registry to register the collectors on.
            When ``None`` the collectors attach to the implicit global registry;
            production code passes :data:`DEFAULT_REGISTRY` via
            :func:`get_default_metrics`.

    Returns:
        A ``(duration_histogram, requests_counter, exceptions_counter)`` tuple.
    """
    kwargs = {"registry": registry} if registry is not None else {}
    duration = Histogram(
        "http_request_duration_seconds",
        "HTTP request latency in seconds, labelled by method, route template and status.",
        labelnames=("method", "path", "status"),
        buckets=LATENCY_BUCKETS,
        **kwargs,  # type: ignore[arg-type]
    )
    requests_total = Counter(
        "http_requests_total",
        "Total HTTP requests, labelled by method, route template and status.",
        labelnames=("method", "path", "status"),
        **kwargs,  # type: ignore[arg-type]
    )
    exceptions_total = Counter(
        "http_request_exceptions_total",
        "Total unhandled exceptions raised while handling a request.",
        labelnames=("method", "path"),
        **kwargs,  # type: ignore[arg-type]
    )
    return duration, requests_total, exceptions_total


def resolve_route_template(request: Request) -> str:
    """Resolve the route template for a request to bound label cardinality.

    Prefers the route matched by Starlette's router (e.g. ``/aois/{aoi_id}``).
    Falls back to manually matching the app's routes when the middleware runs
    before the router has populated ``scope["route"]``. Returns
    :data:`UNMATCHED_PATH` when no route matches (404 / unknown paths), so
    scanners and stray clients cannot create one series per probed URL.

    Args:
        request: The incoming Starlette/FastAPI request.

    Returns:
        The route template string, or :data:`UNMATCHED_PATH` if unmatched.
    """
    route = request.scope.get("route")
    if route is not None and getattr(route, "path", None):
        return str(route.path)

    app = request.scope.get("app")
    routes = getattr(getattr(app, "router", None), "routes", None) or getattr(app, "routes", None)
    if routes:
        for candidate in routes:
            match, _ = candidate.matches(request.scope)
            if match == Match.FULL and getattr(candidate, "path", None):
                return str(candidate.path)
    return UNMATCHED_PATH


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Starlette middleware recording real HTTP latency and request counts.

    Observes :data:`http_request_duration_seconds`, increments
    :data:`http_requests_total` for every request, and increments
    :data:`http_request_exceptions_total` on unhandled exceptions (which it
    re-raises so the existing error handlers still run). The ``/metrics``
    endpoint is excluded so the scraper's own traffic does not skew the
    percentiles.
    """

    def __init__(self, app: ASGIApp, registry: CollectorRegistry | None = None) -> None:
        """Initialise the middleware and its metric collectors.

        Args:
            app: The wrapped ASGI application.
            registry: Optional Prometheus registry (tests pass an isolated one).
                When ``None`` the middleware reuses the process-singleton
                collectors on :data:`DEFAULT_REGISTRY`, so re-instantiating the
                app does not re-register the metrics.
        """
        super().__init__(app)
        metrics = build_request_metrics(registry) if registry is not None else get_default_metrics()
        self._duration, self._requests_total, self._exceptions_total = metrics

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Time the request, record metrics and propagate the response.

        Args:
            request: The incoming request.
            call_next: Callable invoking the downstream handler.

        Returns:
            The downstream :class:`~starlette.responses.Response`.

        Raises:
            Exception: Re-raised after counting it so the app's exception
                handlers produce the final response.
        """
        path = resolve_route_template(request)
        method = request.method

        if path == METRICS_PATH:
            return await call_next(request)

        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            self._exceptions_total.labels(method=method, path=path).inc()
            logger.warning("metrics_request_exception", method=method, path=path)
            raise
        elapsed = time.perf_counter() - start
        status = str(response.status_code)
        self._duration.labels(method=method, path=path, status=status).observe(elapsed)
        self._requests_total.labels(method=method, path=path, status=status).inc()
        return response


def render_latest(registry: CollectorRegistry | None = None) -> Response:
    """Render the current metric registry in Prometheus text exposition format.

    Used by the thin ``GET /metrics`` router (SoC: the router only delegates).

    Args:
        registry: Optional registry to serialise; defaults to the app's
            dedicated :data:`DEFAULT_REGISTRY` (the one the middleware writes to),
            not the implicit global registry.

    Returns:
        A :class:`~starlette.responses.Response` carrying the exposition text
        with the canonical ``CONTENT_TYPE_LATEST`` media type.
    """
    # Touch the singleton so ``/metrics`` exposes the collectors even if no
    # request has hit the middleware yet (collectors render with zero samples).
    get_default_metrics()
    payload = generate_latest(registry if registry is not None else DEFAULT_REGISTRY)
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
