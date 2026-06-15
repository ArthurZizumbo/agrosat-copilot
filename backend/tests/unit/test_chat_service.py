"""Unit tests for ChatService.dispatch with a fake agent orchestrator.

The agent core (``ml/agent/orchestrator.py``) is built in parallel and is
imported lazily by the service. We inject a fake module so dispatch consumes a
controlled event stream without any LLM/DB calls.
"""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import AsyncIterator

import pytest

from backend.app.services.chat_service import ChatService
from backend.app.services.job_registry import JobRegistry
from ml.agent.events import Done, FinalAnswer, PlanCreated
from ml.agent.ports import AgentDeps


def _install_fake_orchestrator(monkeypatch: pytest.MonkeyPatch, events: list) -> None:
    """Install a fake ``ml.agent.orchestrator`` yielding ``events``."""

    async def fake_run_chat(
        *,
        session_id: str,
        user_message: str,
        llm_variant: str,
        deps: AgentDeps,
        aoi_id: int | None = None,
    ) -> AsyncIterator[object]:
        assert isinstance(deps, AgentDeps)
        for event in events:
            yield event

    module = types.ModuleType("ml.agent.orchestrator")
    module.run_chat = fake_run_chat  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ml.agent.orchestrator", module)


@pytest.mark.asyncio
async def test_dispatch_publishes_agent_events(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [
        PlanCreated(steps=["list", "classify"]),
        FinalAnswer(text="3 parcels", citations=[]),
        Done(job_id="placeholder"),
    ]
    _install_fake_orchestrator(monkeypatch, events)

    registry = JobRegistry()
    service = ChatService(registry=registry, session_factory=lambda: None)  # type: ignore[arg-type]

    job_id = service.dispatch(
        session_id="11111111-1111-1111-1111-111111111111",
        message="hello",
        llm_variant="gemini",
    )

    collected = []
    async for event in registry.subscribe(job_id):
        collected.append(event)

    assert [e["type"] for e in collected] == ["plan_created", "final_answer", "done"]
    assert registry.status(job_id) == "done"


@pytest.mark.asyncio
async def test_dispatch_surfaces_agent_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the orchestrator raises, dispatch emits error + done and marks error."""

    async def boom(
        *,
        session_id: str,
        user_message: str,
        llm_variant: str,
        deps: AgentDeps,
        aoi_id: int | None = None,
    ) -> AsyncIterator[object]:
        if False:  # pragma: no cover - make this an async generator
            yield None
        raise RuntimeError("vision agent exploded")

    module = types.ModuleType("ml.agent.orchestrator")
    module.run_chat = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ml.agent.orchestrator", module)

    registry = JobRegistry()
    service = ChatService(registry=registry, session_factory=lambda: None)  # type: ignore[arg-type]
    job_id = service.dispatch(
        session_id="11111111-1111-1111-1111-111111111111",
        message="hi",
        llm_variant="qwen35",
    )

    await asyncio.sleep(0.02)
    collected = [event async for event in registry.subscribe(job_id)]

    types_seen = [e["type"] for e in collected]
    assert "error" in types_seen
    assert types_seen[-1] == "done"
    assert registry.status(job_id) == "error"
