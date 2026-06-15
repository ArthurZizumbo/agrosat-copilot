"""In-memory job registry for the dispatch + WebSocket transport (ADR-011 §3).

A ``JobRegistry`` maps ``job_id`` to a replay buffer, a "new data" event and a
status. The background task that consumes ``run_chat`` publishes each serialized
:class:`AgentEvent` into the job; WebSocket / SSE subscribers replay the buffer
(backlog on late connect) and then follow new appends.

The buffer is the single source of truth: every subscriber tracks its own cursor
into it, so multiple subscribers (e.g. WS + SSE fallback) each receive the full
ordered stream. An :class:`asyncio.Event` wakes subscribers when data arrives or
the job finishes.

Single-process, single-instance only (demo scope). Replaceable by Pub/Sub
post-presentation without touching the routers.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

import structlog

logger = structlog.get_logger(__name__)

JobStatus = Literal["running", "done", "error"]

EventJson = dict[str, object]


@dataclass
class _Job:
    """State of a single dispatched job."""

    job_id: str
    session_id: str
    status: JobStatus = "running"
    buffer: list[EventJson] = field(default_factory=list)
    updated: asyncio.Event = field(default_factory=asyncio.Event)


class JobRegistry:
    """Process-wide registry of in-flight chat jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, _Job] = {}
        # Last job dispatched per session, so a WS without ?job_id can attach.
        self._latest_by_session: dict[str, str] = {}

    def create_job(self, *, session_id: str) -> str:
        """Register a new job for a session and return its ``job_id``."""
        job_id = f"job_{uuid4().hex[:12]}"
        self._jobs[job_id] = _Job(job_id=job_id, session_id=session_id)
        self._latest_by_session[session_id] = job_id
        logger.info("job_created", job_id=job_id, session_id=session_id)
        return job_id

    def publish(self, job_id: str, event: EventJson) -> None:
        """Append a serialized event to the job buffer and wake subscribers."""
        job = self._jobs.get(job_id)
        if job is None:
            logger.warning("publish_unknown_job", job_id=job_id)
            return
        job.buffer.append(event)
        job.updated.set()

    def finish(self, job_id: str, *, status: JobStatus = "done") -> None:
        """Mark a job terminal and unblock its subscribers."""
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = status
        job.updated.set()
        logger.info("job_finished", job_id=job_id, status=status)

    def latest_job_for_session(self, session_id: str) -> str | None:
        """Return the most recently dispatched job_id for a session."""
        return self._latest_by_session.get(session_id)

    def status(self, job_id: str) -> JobStatus | None:
        """Return the status of a job, or ``None`` if unknown."""
        job = self._jobs.get(job_id)
        return job.status if job is not None else None

    async def subscribe(self, job_id: str) -> AsyncIterator[EventJson]:
        """Yield events for a job: replay the buffer, then follow new appends.

        Each subscriber keeps a private cursor into the shared buffer, so a late
        subscriber still replays the full backlog and a job that already finished
        is fully drained before the iterator stops.
        """
        job = self._jobs.get(job_id)
        if job is None:
            logger.warning("subscribe_unknown_job", job_id=job_id)
            return

        cursor = 0
        while True:
            # Clear before draining: any publish/finish that races with the drain
            # below re-sets the event, so the subsequent wait() returns at once
            # and no wakeup is lost.
            job.updated.clear()

            # Emit everything appended since our cursor.
            while cursor < len(job.buffer):
                yield job.buffer[cursor]
                cursor += 1

            if job.status != "running":
                return

            # Wait for the next publish/finish, then re-check the buffer.
            await job.updated.wait()
