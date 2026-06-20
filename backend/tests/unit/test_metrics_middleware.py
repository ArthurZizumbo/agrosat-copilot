"""Unit tests for the Prometheus observability scaffolding (US-059).

Scope: ONLY the files this US touches -- ``backend/app/middleware/metrics.py``
and ``backend/app/api/metrics.py`` (mounted via ``main.py``). No network, no
real LLM/DB; the middleware is exercised against a tiny in-process FastAPI app
with an isolated :class:`~prometheus_client.CollectorRegistry` so the global
default registry is never mutated by the tests.

Assertions:

- ``/metrics`` responds 200 with the Prometheus exposition content type.
- A real request increments ``http_requests_total`` and observes the latency
  histogram (real process metrics, no synthetic values).
- The ``path`` label is normalised to the ROUTE TEMPLATE (``/items/{item_id}``),
  not the raw URL, so per-id requests do not explode label cardinality (R1).
- An unmatched path is bucketed under ``<unmatched>``.
"""

from __future__ import annotations

from typing import ClassVar

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from starlette.testclient import TestClient

from backend.app.middleware.metrics import (
    UNMATCHED_PATH,
    PrometheusMiddleware,
    render_latest,
    resolve_route_template,
)


def _build_app(registry: CollectorRegistry) -> FastAPI:
    """Build a minimal app wired with the middleware on an isolated registry.

    Args:
        registry: The isolated Prometheus registry the middleware writes to.

    Returns:
        A FastAPI app exposing a templated route plus ``/metrics``.
    """
    app = FastAPI()
    app.add_middleware(PrometheusMiddleware, registry=registry)

    @app.get("/items/{item_id}")
    async def _item(item_id: str) -> dict[str, str]:
        return {"item_id": item_id}

    @app.get("/metrics", include_in_schema=False)
    async def _metrics() -> object:
        return render_latest(registry)

    return app


def test_metrics_endpoint_returns_prometheus_content_type() -> None:
    """``/metrics`` returns 200 with the canonical Prometheus media type."""
    registry = CollectorRegistry()
    client = TestClient(_build_app(registry))

    response = client.get("/metrics")

    assert response.status_code == 200
    # Assert against the library constant, not a hardcoded version string:
    # prometheus-client 0.25 ships ``version=1.0.0`` (not the legacy 0.0.4).
    assert response.headers["content-type"] == CONTENT_TYPE_LATEST
    assert response.headers["content-type"].startswith("text/plain")


def test_request_increments_counter_and_histogram() -> None:
    """A real request increments the request counter and observes latency."""
    registry = CollectorRegistry()
    client = TestClient(_build_app(registry))

    client.get("/items/abc")

    body = generate_latest(registry).decode()
    # Counter incremented for the matched route template with a 200 status.
    assert 'http_requests_total{method="GET",path="/items/{item_id}",status="200"}' in body
    # Histogram observed at least one sample on the same labels.
    hist = 'http_request_duration_seconds_count{method="GET",path="/items/{item_id}",status="200"}'
    assert hist in body


def test_path_label_is_route_template_not_raw_url() -> None:
    """Per-id requests collapse onto one route-template series (anti-cardinality)."""
    registry = CollectorRegistry()
    client = TestClient(_build_app(registry))

    for item_id in ("1", "2", "3"):
        client.get(f"/items/{item_id}")

    body = generate_latest(registry).decode()
    # Single series keyed by the template -- no per-id series leaked.
    assert 'path="/items/{item_id}"' in body
    assert 'path="/items/1"' not in body
    assert 'path="/items/3"' not in body
    # The three requests aggregate into one counter value.
    line = next(
        ln
        for ln in body.splitlines()
        if ln.startswith('http_requests_total{method="GET",path="/items/{item_id}",status="200"}')
    )
    assert line.split()[-1] == "3.0"


def test_unmatched_path_is_bucketed_under_constant_label() -> None:
    """A 404 (unknown path) is recorded under the ``<unmatched>`` template."""
    registry = CollectorRegistry()
    client = TestClient(_build_app(registry))

    client.get("/no/such/route/xyz")

    body = generate_latest(registry).decode()
    assert f'path="{UNMATCHED_PATH}"' in body
    assert 'path="/no/such/route/xyz"' not in body


def test_resolve_route_template_returns_unmatched_for_empty_scope() -> None:
    """``resolve_route_template`` degrades to ``<unmatched>`` with no route/app."""

    class _FakeRequest:
        scope: ClassVar[dict[str, object]] = {}

    assert resolve_route_template(_FakeRequest()) == UNMATCHED_PATH  # type: ignore[arg-type]
