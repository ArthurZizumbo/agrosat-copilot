r"""Pub/Sub inference worker skeleton (US-056 -- DEFERRED to Full, ADR-009 / ADR-012).

This module is HONEST scaffolding, not a running worker. US-056 (the async
Pub/Sub + Cloud Run GPU L4 worker) is OUT of the MVP scope (plan v8 §1, §9.2): the
MVP serves inference synchronously through
:class:`backend.app.services.jobs_service.JobsService` in ``sync`` mode. Here we
fix the worker's contract -- the callback signature, the message schema, the
result-publishing flow -- so the Full implementation is a matter of wiring real
GCP clients, NOT redesigning. **The subscription loop does not start** (see the
``__main__`` guard at the bottom): it requires a live GCP Pub/Sub topic
(``inference-jobs``) and a Cloud Run GPU service, neither of which exists in the
MVP.

Full data flow (what this worker WILL do, once US-056 is activated)::

    Pub/Sub topic ``inference-jobs``                  (backend publishes JobRequest)
        |  message.data = JobRequest JSON {session_id, aoi_geojson, model_id, year, params}
        v
    handle_message(message)                           (this callback, per message)
        |-- parse + validate -> JobRequest            (Pydantic; bad message -> ack+drop or DLQ)
        |-- run inference (GPU model_id)              (heavy dense model on Cloud Run GPU L4)
        |-- persist result -> GCS                     (gs://<artifacts>/inference/<job_id>.json)
        |-- publish -> topic ``inference-results``     (InferenceResult {job_id, state, result_url})
        |-- message.ack()                             (on success)
        \-- message.nack()  on transient failure      (Pub/Sub redelivers; after 3 -> DLQ topic)
        v
    Backend SSE notifier                              (subscribes to ``inference-results``)
        |  pushes a ``job_done`` SSE frame to the frontend session
        v
    Frontend                                          (US-057 ChatPanel updates the job card)

Reliability (Full, configured in Terraform, NOT in this file):

- Retries + DLQ: the ``inference-jobs`` subscription has a dead-letter policy with
  ``max_delivery_attempts = 3``; a message that ``nack``s three times is routed to
  the ``inference-jobs-dlq`` topic, and a Cloud Monitoring alert fires on DLQ depth.
- Idempotency: results are keyed by ``job_id`` in GCS, so a redelivered message
  overwrites the same object rather than duplicating work.

Everything below is the SHAPE of that worker. The inference call and the GCS /
Pub/Sub publish steps are explicit ``NotImplementedError`` stops -- they fail loud
when called outside Full, never fake a result.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, ConfigDict, ValidationError

from backend.app.core.config import Settings, get_settings
from backend.app.services.jobs_service import JobRequest, JobState

if TYPE_CHECKING:  # pragma: no cover - typing only; the real types are GCP-only
    from google.cloud.pubsub_v1.subscriber.message import Message

logger = structlog.get_logger(__name__)

__all__ = [
    "InferenceResult",
    "handle_message",
    "parse_job_request",
    "run_subscriber",
]

#: Marker shared by every Full-only stop so the message is uniform and greppable.
_DEFERRED = "US-056 deferred to Full (ADR-009 / ADR-012)"


class InferenceResult(BaseModel):
    """Message published to the ``inference-results`` topic when a job completes.

    The result payload itself is persisted to GCS (``result_url``); this message
    is the lightweight notification the backend's SSE notifier subscribes to.

    Attributes:
        job_id: The trazable job identifier (matches the originating
            :class:`~backend.app.services.jobs_service.JobRequest`).
        session_id: Tenant session the result belongs to (for SSE routing).
        state: Terminal :class:`~backend.app.services.jobs_service.JobState`
            (``DONE`` or ``FAILED``).
        result_url: ``gs://`` URI of the persisted result object (``None`` when
            ``FAILED``).
        error: Structured error message when ``FAILED`` (``None`` otherwise).
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str
    session_id: str
    state: JobState
    result_url: str | None = None
    error: str | None = None


def parse_job_request(data: bytes) -> JobRequest:
    """Parse + validate a raw ``inference-jobs`` message body into a :class:`JobRequest`.

    This step is REAL (no GCP dependency): it is the boundary that protects the
    inference path from a malformed message. A body that is not valid JSON or does
    not satisfy the :class:`JobRequest` schema raises -- the caller decides whether
    to ack-and-drop (poison message) or route to the DLQ.

    Args:
        data: The raw Pub/Sub ``message.data`` bytes (UTF-8 JSON of a JobRequest).

    Returns:
        The validated :class:`JobRequest`.

    Raises:
        ValueError: When ``data`` is not valid UTF-8 JSON.
        pydantic.ValidationError: When the JSON does not match the schema.
    """
    try:
        payload: dict[str, Any] = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"inference-jobs message is not valid UTF-8 JSON: {exc}") from exc
    return JobRequest.model_validate(payload)


async def _run_inference(request: JobRequest, settings: Settings) -> InferenceResult:
    """Run the heavy GPU inference for a job and persist the result (FUTURE).

    The Full implementation will: load ``request.model_id`` on the Cloud Run GPU,
    run inference over ``request.aoi_geojson``, write the result JSON to
    ``gs://<gcs_artifacts_bucket>/inference/<job_id>.json`` and return an
    :class:`InferenceResult` pointing at it. It is a deliberate
    :class:`NotImplementedError` today -- the worker never fabricates a result.

    Args:
        request: The validated job request.
        settings: Typed application settings (GCS bucket, topics).

    Returns:
        The :class:`InferenceResult` to publish (Full only).

    Raises:
        NotImplementedError: Always, in the MVP -- the GPU inference + GCS persist
            path is Full-only (US-056 deferred).
    """
    raise NotImplementedError(
        f"GPU inference + GCS persist is {_DEFERRED}. The Full worker will serve "
        f"model_id={request.model_id!r} on Cloud Run GPU and write the result to "
        f"gs://{settings.gcs_artifacts_bucket or '<artifacts-bucket>'}/inference/<job_id>.json. "
        "The MVP serves inference synchronously via JobsService(enqueue_mode='sync')."
    )


async def _publish_result(result: InferenceResult, settings: Settings) -> None:
    """Publish an :class:`InferenceResult` to the ``inference-results`` topic (FUTURE).

    The Full implementation publishes via a ``pubsub_v1.PublisherClient``; the
    backend's SSE notifier subscribes and pushes a ``job_done`` frame to the
    frontend session. A deliberate :class:`NotImplementedError` today.

    Args:
        result: The result notification to publish.
        settings: Typed application settings (topic names, project id).

    Raises:
        NotImplementedError: Always, in the MVP -- publishing is Full-only.
    """
    raise NotImplementedError(
        f"Publishing to the inference-results topic is {_DEFERRED}. The Full worker "
        f"will publish via pubsub_v1.PublisherClient on project "
        f"{settings.gcp_project_id!r}; the backend SSE notifier forwards it to the "
        "frontend (US-057)."
    )


async def handle_message(message: Message) -> None:
    """Pub/Sub subscriber callback: process one ``inference-jobs`` message (FUTURE).

    This is the SHAPE the Full subscriber registers. The flow is: validate ->
    infer (GPU) -> persist (GCS) -> publish (inference-results) -> ack; on a
    transient failure ``nack`` so Pub/Sub redelivers (DLQ after 3 attempts). The
    inference and publish steps raise :class:`NotImplementedError` today (Full
    only); the callback is written so the only work left for Full is removing those
    stops, not restructuring.

    A message that fails validation is a poison message: it is logged and ``ack``ed
    (dropped) so it does not loop forever -- a malformed payload will never succeed
    on redelivery. Genuine inference failures ``nack`` so they retry / reach the DLQ.

    Args:
        message: The Pub/Sub message (``message.data`` is a JobRequest JSON body).
    """
    settings = get_settings()
    try:
        request = parse_job_request(message.data)
    except (ValueError, ValidationError) as exc:
        # Poison message: it can never validate -> drop it (ack), do not retry.
        logger.error("inference_worker_bad_message", error=str(exc))
        message.ack()
        return

    job_id = str(getattr(message, "message_id", "") or "")
    log = logger.bind(
        job_id=job_id,
        session_id=str(request.session_id),
        model_id=request.model_id,
    )
    log.info("inference_worker_message_received")

    try:
        result = await _run_inference(request, settings)
        await _publish_result(result, settings)
    except NotImplementedError:
        # The deferred stops: re-raise so the deferral is loud (the loop is not
        # running in the MVP anyway). In Full these branches are removed.
        log.warning("inference_worker_deferred", reason=_DEFERRED)
        raise
    except Exception as exc:  # transient failure in Full -> nack -> retry / DLQ
        log.exception("inference_worker_failed", error=str(exc), state=JobState.FAILED.value)
        message.nack()
        return

    log.info("inference_worker_done", result_url=result.result_url)
    message.ack()


def run_subscriber() -> None:
    """Start the Pub/Sub subscription loop (FUTURE -- does NOT run in the MVP).

    The Full implementation builds a ``pubsub_v1.SubscriberClient``, subscribes to
    the ``inference-jobs`` subscription with :func:`handle_message` as the
    callback, and blocks on the streaming pull future. It requires a live GCP
    Pub/Sub topic and a Cloud Run GPU L4 service -- neither exists in the MVP -- so
    it raises a clear :class:`NotImplementedError` rather than pretending to listen.

    Raises:
        NotImplementedError: Always, in the MVP -- the subscriber is Full-only
            (US-056 deferred, ADR-009 / ADR-012).
    """
    settings = get_settings()
    raise NotImplementedError(
        f"The Pub/Sub subscription loop is {_DEFERRED}. It requires a live GCP "
        f"Pub/Sub topic ({settings.pubsub_inference_topic!r}) and a Cloud Run GPU L4 "
        "service. The Full implementation: "
        "subscriber = pubsub_v1.SubscriberClient(); "
        "future = subscriber.subscribe(subscription_path, callback=handle_message); "
        "future.result()  # blocks. "
        "The MVP serves inference synchronously via "
        "backend.app.services.jobs_service.JobsService(enqueue_mode='sync')."
    )


if __name__ == "__main__":  # pragma: no cover - intentionally not runnable in the MVP
    # US-056 is deferred to Full (ADR-009 / ADR-012). This worker is documented
    # scaffolding: starting the subscriber requires GCP Pub/Sub + Cloud Run GPU L4.
    # We refuse to fake a running worker -- ``run_subscriber`` raises and explains.
    logger.warning(
        "inference_worker_not_runnable_in_mvp",
        reason=_DEFERRED,
        hint="MVP serves inference synchronously via JobsService(enqueue_mode='sync').",
    )
    run_subscriber()
