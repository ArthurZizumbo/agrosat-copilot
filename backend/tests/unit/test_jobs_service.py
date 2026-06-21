"""Unit tests for the inference :class:`JobsService` (US-056 scaffolding).

US-056 (the async Pub/Sub + Cloud Run GPU worker) is DEFERRED to Full
(ADR-009 / ADR-012); the MVP runs inference synchronously. These tests cover ONLY
what genuinely works today -- there is no Pub/Sub here, mocked or otherwise:

- the ``sync`` mode runs the (injected) inference inline and returns a ``DONE``
  job with a trazable ``job_id`` and the inline result;
- ``get_job_status`` reflects the terminal state immediately;
- the ``pubsub`` mode raises a CLEAR ``NotImplementedError`` (Full-only) -- it
  never silently no-ops or fakes success;
- a heavy / unknown ``model_id`` in ``sync`` mode degrades to ``FAILED`` with a
  structured error (not a faked inline success);
- an inference failure surfaces as ``FAILED`` with the error message;
- Pydantic validates the ``JobRequest`` (the GeoJSON AOI in particular) before the
  service runs.

The inference itself is mocked (a stub ``runner``) so no XGBoost model is loaded
and no database is touched -- the SYNC PATH IS REAL (it really invokes the runner
and threads its result through), only the heavy ML/DB boundaries are doubled.
"""

from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from backend.app.services.jobs_service import (
    JobRequest,
    JobsService,
    JobState,
    JobStatus,
)
from ml.agent.schemas import ClassificationResult, ClassifyParcelInput, GeoJSONGeometry

_SESSION = UUID("22222222-2222-2222-2222-222222222222")

#: A valid GeoJSON polygon (a 1x1 degree square) the AOI field accepts.
_POLYGON = GeoJSONGeometry(
    type="Polygon",
    coordinates=[[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]],
)

_RESULT = ClassificationResult(
    crop_class="wheat",
    confidence=0.91,
    class_probabilities={"wheat": 0.91, "maize": 0.09},
)


class _SettingsStub:
    """Minimal settings stub so the service never reads the real ``.env.local``."""

    pubsub_inference_topic = "inference-jobs"
    gcp_project_id = "agrosat-test"
    gcs_artifacts_bucket = ""


class _FakePool:
    """asyncpg pool double: the sync path builds a ToolContext with it but the
    stub runner never touches it, so no method needs to do anything real."""


async def _fake_pool_factory() -> _FakePool:
    """Awaitable returning the fake pool (injected ``pool_factory``)."""
    return _FakePool()


def _make_service(
    *,
    enqueue_mode: str = "sync",
    runner=None,
) -> JobsService:
    """Build a :class:`JobsService` with every heavy boundary doubled.

    Args:
        enqueue_mode: ``"sync"`` (default) or ``"pubsub"``.
        runner: Optional inference-runner stub; defaults to one returning
            :data:`_RESULT` and recording the input it was called with.
    """
    captured: dict[str, object] = {}

    async def _default_stub(inp: ClassifyParcelInput, ctx) -> ClassificationResult:
        captured["input"] = inp
        captured["ctx"] = ctx
        return _RESULT

    service = JobsService(
        _SettingsStub(),  # type: ignore[arg-type]
        enqueue_mode=enqueue_mode,  # type: ignore[arg-type]
        runner=runner or _default_stub,
        pool_factory=_fake_pool_factory,  # type: ignore[arg-type]
    )
    service._captured = captured  # type: ignore[attr-defined]  # test introspection
    return service


def _request(model_id: str = "xgb-alphaearth", **params: object) -> JobRequest:
    """Build a valid :class:`JobRequest` for the session over :data:`_POLYGON`."""
    return JobRequest(
        session_id=_SESSION,
        aoi_geojson=_POLYGON,
        model_id=model_id,
        params=params,
    )


# ---------------------------------------------------------------------------
# sync mode (MVP, REAL inference path)
# ---------------------------------------------------------------------------
async def test_sync_mode_runs_inline_and_returns_done() -> None:
    """``sync`` mode runs the inference inline and returns a DONE job."""
    service = _make_service()
    job_id = await service.submit_job(_request())

    assert isinstance(job_id, str) and len(job_id) == 32  # uuid4 hex, trazable

    status = await service.get_job_status(job_id)
    assert status is not None
    assert status.job_id == job_id
    assert status.state is JobState.DONE
    assert status.error is None
    assert status.result == _RESULT  # the inline result is threaded through


async def test_sync_mode_passes_request_through_to_runner() -> None:
    """The JobRequest fields reach the inference runner (session-scoped, AOI)."""
    service = _make_service()
    await service.submit_job(
        _request(model_id="stacking-5", use_stacking=True, label_space="france-9")
    )

    captured = service._captured  # type: ignore[attr-defined]
    classify_input = captured["input"]
    assert isinstance(classify_input, ClassifyParcelInput)
    assert classify_input.session_id == _SESSION  # multi-tenant: the session flows
    assert classify_input.aoi == _POLYGON
    assert classify_input.use_stacking is True
    assert classify_input.label_space == "france-9"
    # The ToolContext is session-scoped over the injected fake pool.
    assert captured["ctx"].session_id == _SESSION


async def test_get_job_status_unknown_id_is_none() -> None:
    """An unknown job id yields ``None`` (no crash, no fabricated status)."""
    service = _make_service()
    assert await service.get_job_status("does-not-exist") is None


async def test_sync_mode_inference_failure_is_failed_state() -> None:
    """An inference failure surfaces as a FAILED job with the error message."""

    async def _boom(inp: ClassifyParcelInput, ctx) -> ClassificationResult:
        raise RuntimeError("classifier exploded")

    service = _make_service(runner=_boom)
    job_id = await service.submit_job(_request())

    status = await service.get_job_status(job_id)
    assert status is not None
    assert status.state is JobState.FAILED
    assert status.result is None
    assert "classifier exploded" in (status.error or "")


async def test_sync_mode_heavy_model_degrades_to_failed_not_faked() -> None:
    """A model the sync path cannot serve returns FAILED honestly (never faked)."""
    called = False

    async def _should_not_run(inp: ClassifyParcelInput, ctx) -> ClassificationResult:
        nonlocal called
        called = True
        return _RESULT

    service = _make_service(runner=_should_not_run)
    job_id = await service.submit_job(_request(model_id="segformer-b2-gpu"))

    status = await service.get_job_status(job_id)
    assert status is not None
    assert status.state is JobState.FAILED
    assert "Pub/Sub" in (status.error or "") or "synchronous" in (status.error or "")
    assert called is False  # the runner was never invoked for the heavy model


# ---------------------------------------------------------------------------
# pubsub mode (FUTURE, Full-only) -- must fail LOUD and clear
# ---------------------------------------------------------------------------
async def test_pubsub_mode_raises_clear_not_implemented() -> None:
    """``pubsub`` mode raises a clear NotImplementedError (US-056 deferred)."""
    service = _make_service(enqueue_mode="pubsub")

    with pytest.raises(NotImplementedError) as excinfo:
        await service.submit_job(_request())

    message = str(excinfo.value)
    # The error must name the deferral so it is self-documenting (no silent no-op).
    assert "Pub/Sub mode is Full-only" in message
    assert "US-056" in message
    assert "ADR-009" in message
    assert "inference-jobs" in message  # points at the future topic


def test_default_enqueue_mode_is_sync() -> None:
    """The MVP default is synchronous inference (ADR-012)."""
    service = JobsService(_SettingsStub())  # type: ignore[arg-type]
    assert service.enqueue_mode == "sync"


# ---------------------------------------------------------------------------
# Pydantic validation of the JobRequest contract
# ---------------------------------------------------------------------------
def test_job_request_accepts_valid_payload() -> None:
    """A well-formed payload validates into a JobRequest with defaults."""
    request = JobRequest(session_id=_SESSION, aoi_geojson=_POLYGON)
    assert request.model_id == "xgb-alphaearth"  # default model
    assert request.year == 2019  # default campaign year
    assert request.params == {}


def test_job_request_rejects_invalid_geometry_type() -> None:
    """An unsupported GeoJSON geometry type is rejected before any inference."""
    with pytest.raises(ValidationError):
        JobRequest(
            session_id=_SESSION,
            aoi_geojson=GeoJSONGeometry(type="Tetrahedron", coordinates=[[0.0, 0.0]]),
        )


def test_job_request_rejects_empty_coordinates() -> None:
    """An empty coordinate array is rejected (a geometry must have vertices)."""
    with pytest.raises(ValidationError):
        JobRequest(
            session_id=_SESSION,
            aoi_geojson=GeoJSONGeometry(type="Polygon", coordinates=[]),
        )


def test_job_request_rejects_extra_keys() -> None:
    """``extra='forbid'`` rejects keys hallucinated by a caller / LLM."""
    with pytest.raises(ValidationError):
        JobRequest(
            session_id=_SESSION,
            aoi_geojson=_POLYGON,
            unexpected="value",  # type: ignore[call-arg]
        )


def test_job_status_terminal_carries_no_partial_result() -> None:
    """A FAILED status carries an error and no result (honest degradation)."""
    status = JobStatus(job_id="abc", state=JobState.FAILED, error="boom")
    assert status.result is None
    assert status.result_url is None
    assert status.error == "boom"
