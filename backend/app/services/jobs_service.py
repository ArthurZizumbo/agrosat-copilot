"""Inference jobs service: the public interface a Pub/Sub worker will fulfil (US-056).

US-056 (the async Pub/Sub + Cloud Run GPU L4 worker) is DEFERRED to Full
post-presentation (plan v8 §1 OUT, §9.2; ADR-009) and a dedicated decision record,
ADR-012 (``docs/decisions/ADR-012-inferencia-sincrona-mvp.md``). The MVP demo runs
inference **synchronously** -- the parcels are pre-loaded PASTIS and the served
model is the CPU-light ``xgb-alphaearth`` tabular classifier, fast enough not to
block the user. This module is the HONEST scaffolding of that decision: it fixes
the public contract a future Pub/Sub worker will implement, while only the ``sync``
path actually runs today.

Two enqueue modes, neither faked:

- ``sync`` (MVP, REAL): :meth:`JobsService.submit_job` runs the inference inline
  -- it delegates to the same ``classify_new_parcel`` tool the ``/chat`` reasoner
  calls (:func:`ml.agent.tools.classify.run`) -- and returns a job already in the
  ``DONE`` state (or ``FAILED`` with a structured error). No queue, no GCS, no
  worker process. This path is genuinely exercised by the unit tests.
- ``pubsub`` (FUTURE, Full-only): :meth:`JobsService.submit_job` raises a clear
  :class:`NotImplementedError` -- it does NOT silently no-op or fake success. The
  message names US-056 / ADR-009 so the failure is self-documenting. The Full
  implementation will publish the :class:`JobRequest` to the ``inference-jobs``
  topic (``settings.pubsub_inference_topic``) for the Cloud Run GPU worker
  (:mod:`ml.workers.inference_worker`) to consume, persist the result to GCS,
  publish to ``inference-results`` and notify the frontend over SSE.

Multi-tenant: every :class:`JobRequest` carries a ``session_id`` and every job is
logged with a trazable ``job_id`` (structlog). The GeoJSON AOI is validated by
Pydantic (:class:`~ml.agent.schemas.GeoJSONGeometry`) before the service runs, so
a malformed polygon never reaches the inference path.

Router -> service -> model: a ``/jobs`` router (not built in the MVP -- US-056 is
deferred) would adapt HTTP to :meth:`submit_job` / :meth:`get_job_status` and
contain no logic of its own.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, ConfigDict, Field

from backend.app.core.config import Settings, get_settings
from ml.agent.context import ToolContext
from ml.agent.schemas import ClassificationResult, ClassifyParcelInput, GeoJSONGeometry

if TYPE_CHECKING:
    import asyncpg

logger = structlog.get_logger(__name__)

__all__ = [
    "EnqueueMode",
    "InferenceRunner",
    "JobRequest",
    "JobState",
    "JobStatus",
    "JobsService",
    "PoolFactory",
]

#: The two enqueue modes. ``sync`` runs inline (MVP, real); ``pubsub`` is the Full
#: path that publishes to the ``inference-jobs`` topic -- deferred (US-056).
EnqueueMode = Literal["sync", "pubsub"]

#: Default campaign year for an inference job (matches the perceiver / classify
#: tool default, AlphaEarth annual coverage starts 2017).
_DEFAULT_YEAR: int = 2019

#: The set of model ids the MVP sync path can serve inline (CPU-light). Anything
#: else is rejected with a clear error rather than silently mis-served. The Full
#: Pub/Sub worker (GPU) will serve the heavier dense models out of band.
_SYNC_MODELS: frozenset[str] = frozenset({"xgb-alphaearth", "stacking-5"})


class JobState(StrEnum):
    """Lifecycle state of an inference job.

    The MVP sync path only ever returns ``DONE`` or ``FAILED`` (it runs inline, so
    there is no real ``QUEUED`` / ``RUNNING`` window). ``QUEUED`` and ``RUNNING``
    are defined for the Full Pub/Sub worker, which transitions a job through them
    as the message is published, picked up and processed.
    """

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobRequest(BaseModel):
    """Body of an inference job submission (the ``inference-jobs`` message schema).

    Mirrors the US-056 Pub/Sub message contract ``{aoi_geojson, model_id, params}``
    plus the multi-tenant ``session_id`` and the campaign ``year`` the classifier
    needs. The AOI is a validated :class:`~ml.agent.schemas.GeoJSONGeometry`, so a
    malformed polygon is rejected before any inference runs.

    Attributes:
        session_id: Tenant session that owns the job (multi-tenant isolation).
        aoi_geojson: Validated GeoJSON polygon of the parcel/AOI to infer over.
        model_id: Identifier of the model to serve (e.g. ``"xgb-alphaearth"``).
        year: Campaign year of the AlphaEarth annual embedding.
        params: Free-form, JSON-serialisable model parameters forwarded verbatim
            (e.g. ``{"use_stacking": true, "label_space": "france-9"}``).
    """

    model_config = ConfigDict(extra="forbid")

    session_id: UUID
    aoi_geojson: GeoJSONGeometry
    model_id: str = "xgb-alphaearth"
    year: int = _DEFAULT_YEAR
    params: dict[str, Any] = Field(default_factory=dict)


class JobStatus(BaseModel):
    """Status of a submitted inference job, returned by the service.

    Attributes:
        job_id: Server-generated identifier, trazable across logs (structlog
            ``job_id`` key) and -- in Full -- across the Pub/Sub message and the
            GCS result object.
        state: Lifecycle :class:`JobState`.
        result_url: Where the result lives once ``DONE``. In the MVP sync path
            this is ``None`` (the result is returned inline via :attr:`result`);
            in Full it is the ``gs://`` URI of the persisted result in GCS.
        result: The inline inference result for the MVP sync path (``None`` until
            ``DONE``; never set by the Full async path, which uses ``result_url``).
        error: Structured error message when ``state`` is ``FAILED`` (``None``
            otherwise). Honest degradation: a failed job carries why, never a
            faked success.
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str
    state: JobState
    result_url: str | None = None
    result: ClassificationResult | None = None
    error: str | None = None


#: Signature of the injectable inference runner: it receives a validated
#: :class:`ClassifyParcelInput` and the shared :class:`ToolContext`, and returns a
#: :class:`ClassificationResult`. The default is the production
#: ``classify_new_parcel`` tool (:func:`ml.agent.tools.classify.run`), so the sync
#: path runs the SAME inference the ``/chat`` reasoner serves (DRY). Tests inject a
#: stub so the sync path is exercised without loading the XGBoost model or touching
#: a database.
InferenceRunner = Callable[[ClassifyParcelInput, ToolContext], Awaitable[ClassificationResult]]

#: Signature of the injectable pool factory (an awaitable returning the shared
#: asyncpg pool the inference context runs against). Defaults to
#: :func:`backend.app.core.db.get_pool`. Tests inject a fake so the sync path runs
#: without a real database.
PoolFactory = Callable[[], "Awaitable[asyncpg.Pool]"]


async def _default_runner(inp: ClassifyParcelInput, ctx: ToolContext) -> ClassificationResult:
    """Run the production ``classify_new_parcel`` tool (the real sync inference).

    Imported lazily so this module loads without pulling in the heavy ML stack
    (polars / xgboost) until a sync job actually runs.

    Args:
        inp: Validated classify arguments derived from the :class:`JobRequest`.
        ctx: Shared, session-scoped tool execution context.

    Returns:
        The :class:`ClassificationResult` produced by the tool.
    """
    from ml.agent.tools.classify import run as classify_run

    return await classify_run(inp, ctx)


class JobsService:
    """Submit inference jobs and read their status (US-056 public interface).

    In the MVP this fixes the contract a future Pub/Sub worker will implement and
    runs the ``sync`` mode for real; the ``pubsub`` mode raises a clear
    :class:`NotImplementedError` (US-056 deferred to Full -- ADR-009 / ADR-012). It
    owns the business logic of a would-be ``/jobs`` router (router -> service ->
    model): the router only adapts HTTP to :meth:`submit_job` / :meth:`get_job_status`.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        enqueue_mode: EnqueueMode | None = None,
        runner: InferenceRunner | None = None,
        pool_factory: PoolFactory | None = None,
    ) -> None:
        """Initialise the service with typed settings and injectable boundaries.

        Args:
            settings: Application settings; defaults to the cached singleton via
                :func:`~backend.app.core.config.get_settings` (never ``os.environ``).
            enqueue_mode: ``"sync"`` (MVP, runs inline) or ``"pubsub"`` (Full,
                raises :class:`NotImplementedError`). Defaults to ``"sync"`` -- the
                MVP runs synchronous inference (ADR-012).
            runner: Coroutine running the inference for the sync path; defaults to
                the production ``classify_new_parcel`` tool. Tests inject a stub so
                the sync path runs without the XGBoost model or a database.
            pool_factory: Awaitable returning the asyncpg pool the inference
                context runs against; defaults to
                :func:`backend.app.core.db.get_pool`. Tests inject a fake.
        """
        self._settings = settings or get_settings()
        self._enqueue_mode: EnqueueMode = enqueue_mode or "sync"
        self._runner: InferenceRunner = runner or _default_runner
        self._pool_factory: PoolFactory | None = pool_factory
        # In-memory job registry. The MVP sync path completes a job before
        # ``submit_job`` returns, so the registry is a simple dict (no DB table is
        # needed for a one-shot inline job). The Full path will persist job state
        # in PostgreSQL so a worker on another process can update it.
        self._jobs: dict[str, JobStatus] = {}

    @property
    def enqueue_mode(self) -> EnqueueMode:
        """The active enqueue mode (``"sync"`` in the MVP, ``"pubsub"`` in Full)."""
        return self._enqueue_mode

    async def submit_job(self, request: JobRequest) -> str:
        """Submit an inference job and return its trazable ``job_id``.

        In ``sync`` mode (MVP) the inference runs INLINE before this returns: the
        job is already ``DONE`` (or ``FAILED`` with a structured error) when the id
        comes back -- :meth:`get_job_status` reflects the terminal state
        immediately. In ``pubsub`` mode (Full) this would publish the request to
        the ``inference-jobs`` topic for the Cloud Run GPU worker; that path is
        DEFERRED (US-056) and raises a clear :class:`NotImplementedError`.

        Args:
            request: The validated job request (AOI already validated by Pydantic).

        Returns:
            The server-generated ``job_id`` (UUID4 hex), trazable across logs.

        Raises:
            NotImplementedError: When ``enqueue_mode == "pubsub"`` -- the async
                Pub/Sub worker is Full-only (US-056 deferred, ADR-009 / ADR-012).
        """
        job_id = uuid4().hex
        logger.info(
            "inference_job_submitted",
            job_id=job_id,
            session_id=str(request.session_id),
            model_id=request.model_id,
            enqueue_mode=self._enqueue_mode,
            geometry_type=request.aoi_geojson.type,
        )

        if self._enqueue_mode == "pubsub":
            # FUTURE (Full): publish ``request`` to ``settings.pubsub_inference_topic``
            # for the GPU worker. Deferred -- fail LOUD and clear, never silently.
            logger.warning(
                "inference_job_pubsub_deferred",
                job_id=job_id,
                topic=self._settings.pubsub_inference_topic,
            )
            raise NotImplementedError(
                "Pub/Sub mode is Full-only (US-056 deferred, ADR-009). The MVP runs "
                "synchronous inference; set enqueue_mode='sync'. The Full worker will "
                f"publish to the {self._settings.pubsub_inference_topic!r} topic and "
                "the Cloud Run GPU worker (ml/workers/inference_worker.py) will consume "
                "it, persist to GCS and notify the frontend over SSE."
            )

        status = await self._run_sync(job_id, request)
        self._jobs[job_id] = status
        return job_id

    async def get_job_status(self, job_id: str) -> JobStatus | None:
        """Return the status of a submitted job, or ``None`` if unknown.

        In the MVP the job is in its terminal state (``DONE`` / ``FAILED``) as soon
        as :meth:`submit_job` returns. In Full this would read the job row the
        worker updates in PostgreSQL.

        Args:
            job_id: Identifier returned by :meth:`submit_job`.

        Returns:
            The :class:`JobStatus`, or ``None`` when ``job_id`` is not registered.
        """
        return self._jobs.get(job_id)

    async def _run_sync(self, job_id: str, request: JobRequest) -> JobStatus:
        """Run the inference inline and return the terminal job status (MVP path).

        Translates the :class:`JobRequest` into a :class:`ClassifyParcelInput`,
        builds a session-scoped :class:`ToolContext`, runs the injected inference
        runner (the real ``classify_new_parcel`` tool by default) and wraps the
        outcome in a ``DONE`` (or ``FAILED``) :class:`JobStatus`. The inference is
        REAL -- there is no fabricated success; a failure is surfaced as ``FAILED``
        with a structured ``error``.

        Args:
            job_id: The trazable job identifier.
            request: The validated job request.

        Returns:
            The terminal :class:`JobStatus` (``DONE`` with an inline ``result``, or
            ``FAILED`` with ``error``).
        """
        if request.model_id not in _SYNC_MODELS:
            # Honest: the MVP sync path only serves the CPU-light members; a heavy
            # GPU model is a Full/Pub/Sub job, not faked inline.
            error = (
                f"model_id={request.model_id!r} is not served by the synchronous MVP path "
                f"(supported: {sorted(_SYNC_MODELS)}). Heavy models run on the Full "
                "Pub/Sub GPU worker (US-056 deferred)."
            )
            logger.warning("inference_job_unsupported_model", job_id=job_id, error=error)
            return JobStatus(job_id=job_id, state=JobState.FAILED, error=error)

        classify_input = ClassifyParcelInput(
            session_id=request.session_id,
            aoi=request.aoi_geojson,
            year=request.year,
            use_stacking=bool(request.params.get("use_stacking", request.model_id == "stacking-5")),
            restrict_to_resolved_classes=bool(
                request.params.get("restrict_to_resolved_classes", True)
            ),
            label_space=str(request.params.get("label_space", "france-9")),
        )

        start = time.perf_counter()
        try:
            ctx = await self._build_context(request.session_id)
            result = await self._runner(classify_input, ctx)
        except Exception as exc:  # surface ANY inference failure as a FAILED job
            logger.exception(
                "inference_job_failed",
                job_id=job_id,
                session_id=str(request.session_id),
                model_id=request.model_id,
                error=str(exc),
            )
            return JobStatus(job_id=job_id, state=JobState.FAILED, error=str(exc))

        logger.info(
            "inference_job_done",
            job_id=job_id,
            session_id=str(request.session_id),
            model_id=request.model_id,
            crop_class=result.crop_class,
            confidence=round(result.confidence, 4),
            duration_ms=round((time.perf_counter() - start) * 1000.0, 2),
        )
        return JobStatus(job_id=job_id, state=JobState.DONE, result=result)

    async def _build_context(self, session_id: UUID) -> ToolContext:
        """Build the session-scoped :class:`ToolContext` for the inference run.

        Args:
            session_id: Tenant session driving every downstream DB read.

        Returns:
            A :class:`ToolContext` with the shared asyncpg pool, settings and
            session id. ``defer`` is ``None`` -- the MVP has no deferred executor
            wired (the Full Pub/Sub worker IS that executor).
        """
        if self._pool_factory is None:
            from backend.app.core.db import get_pool

            self._pool_factory = get_pool
        pool = await self._pool_factory()
        return ToolContext(pool=pool, settings=self._settings, session_id=session_id)
