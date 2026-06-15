"""Unit tests for the in-memory JobRegistry transport."""

from __future__ import annotations

import asyncio

import pytest

from backend.app.services.job_registry import JobRegistry


async def _collect(registry: JobRegistry, job_id: str) -> list[dict]:
    return [event async for event in registry.subscribe(job_id)]


@pytest.mark.asyncio
async def test_subscribe_replays_buffer_after_finish() -> None:
    """A subscriber attaching after the job finished replays the full backlog."""
    registry = JobRegistry()
    job_id = registry.create_job(session_id="s1")
    registry.publish(job_id, {"type": "plan_created", "steps": ["a"]})
    registry.publish(job_id, {"type": "done", "job_id": job_id})
    registry.finish(job_id, status="done")

    events = await _collect(registry, job_id)

    assert [e["type"] for e in events] == ["plan_created", "done"]


@pytest.mark.asyncio
async def test_subscribe_streams_live_then_stops_on_finish() -> None:
    """A live subscriber receives events as they are published, then stops."""
    registry = JobRegistry()
    job_id = registry.create_job(session_id="s1")

    received: list[dict] = []

    async def consumer() -> None:
        async for event in registry.subscribe(job_id):
            received.append(event)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0)  # let the consumer start and block

    registry.publish(job_id, {"type": "token", "text": "hi"})
    await asyncio.sleep(0.01)
    registry.publish(job_id, {"type": "done", "job_id": job_id})
    registry.finish(job_id, status="done")

    await asyncio.wait_for(task, timeout=1.0)
    assert [e["type"] for e in received] == ["token", "done"]


@pytest.mark.asyncio
async def test_multiple_subscribers_each_get_full_stream() -> None:
    """Two subscribers (WS + SSE) each receive the full ordered stream."""
    registry = JobRegistry()
    job_id = registry.create_job(session_id="s1")
    registry.publish(job_id, {"type": "token", "text": "a"})
    registry.finish(job_id, status="done")

    first, second = await asyncio.gather(_collect(registry, job_id), _collect(registry, job_id))
    assert [e["type"] for e in first] == ["token"]
    assert [e["type"] for e in second] == ["token"]


@pytest.mark.asyncio
async def test_latest_job_for_session_and_status() -> None:
    """The registry tracks the latest job per session and its status."""
    registry = JobRegistry()
    j1 = registry.create_job(session_id="s1")
    j2 = registry.create_job(session_id="s1")
    assert registry.latest_job_for_session("s1") == j2
    assert registry.latest_job_for_session("unknown") is None
    assert registry.status(j1) == "running"
    registry.finish(j1, status="error")
    assert registry.status(j1) == "error"
    assert registry.status("nope") is None


@pytest.mark.asyncio
async def test_subscribe_unknown_job_yields_nothing() -> None:
    """Subscribing to an unknown job id yields an empty stream."""
    registry = JobRegistry()
    assert await _collect(registry, "missing") == []
