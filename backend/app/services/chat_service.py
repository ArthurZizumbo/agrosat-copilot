"""Chat dispatch service: bridges HTTP/WS transport and the agent core.

ADR-011 §3: ``POST /chat`` returns ``202`` immediately after ``dispatch`` creates
a job and launches a detached background task. That task consumes
``run_chat(...) -> AsyncIterator[AgentEvent]`` and publishes each event (as
``model_dump()``) into the :class:`JobRegistry`. WS / SSE adapters then stream
the events to the client. The backend never builds prompts or calls the LLM.

``run_chat`` is imported lazily inside the task so this module imports cleanly
even before ``ml/agent/orchestrator.py`` exists (built in parallel).
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Literal

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.core.db import get_session_factory
from backend.app.services.agent_adapters import SqlChatMemory, SqlParcelReader
from backend.app.services.job_registry import JobRegistry

logger = structlog.get_logger(__name__)

LlmVariant = Literal["gemini", "qwen35"]


@lru_cache(maxsize=1)
def get_job_registry() -> JobRegistry:
    """Return the process-wide :class:`JobRegistry` singleton."""
    return JobRegistry()


class ChatService:
    """Dispatches chat turns to the agent and publishes the event stream."""

    def __init__(
        self,
        *,
        registry: JobRegistry,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._registry = registry
        self._session_factory = session_factory
        # Keep strong references so background tasks are not GC'd mid-flight.
        self._tasks: set[asyncio.Task[None]] = set()

    def dispatch(
        self,
        *,
        session_id: str,
        message: str,
        llm_variant: LlmVariant,
        aoi_id: int | None = None,
    ) -> str:
        """Create a job, launch the background agent task, return the ``job_id``.

        Returns immediately (non-blocking): the caller responds ``202`` while the
        task streams events into the registry.
        """
        job_id: str = self._registry.create_job(session_id=session_id)
        task = asyncio.create_task(
            self._run_job(
                job_id=job_id,
                session_id=session_id,
                message=message,
                llm_variant=llm_variant,
                aoi_id=aoi_id,
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job_id

    async def _run_job(
        self,
        *,
        job_id: str,
        session_id: str,
        message: str,
        llm_variant: LlmVariant,
        aoi_id: int | None = None,
    ) -> None:
        """Consume ``run_chat`` and publish every event into the registry."""
        # Deferred import: orchestrator is built in parallel and may be absent
        # at module-import time. AgentDeps / events live next to it.
        from ml.agent.events import AgentError, Done
        from ml.agent.orchestrator import run_chat
        from ml.agent.ports import AgentDeps

        deps = AgentDeps(
            parcels=SqlParcelReader(self._session_factory),
            memory=SqlChatMemory(self._session_factory),
        )

        log = logger.bind(job_id=job_id, session_id=session_id, llm_variant=llm_variant)
        log.info("chat_job_started")
        try:
            async for event in run_chat(
                session_id=session_id,
                user_message=message,
                llm_variant=llm_variant,
                deps=deps,
                aoi_id=aoi_id,
            ):
                self._registry.publish(job_id, event.model_dump())
            self._registry.finish(job_id, status="done")
            log.info("chat_job_done")
        except Exception as exc:  # noqa: BLE001 — surface any failure to the client
            log.error("chat_job_failed", error=str(exc))
            self._registry.publish(
                job_id,
                AgentError(code="agent_error", message=str(exc)).model_dump(),
            )
            self._registry.publish(job_id, Done(job_id=job_id).model_dump())
            self._registry.finish(job_id, status="error")


def get_chat_service() -> ChatService:
    """Build a :class:`ChatService` wired to the process singletons."""
    return ChatService(
        registry=get_job_registry(),
        session_factory=get_session_factory(),
    )
