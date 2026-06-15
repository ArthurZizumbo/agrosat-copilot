"""Tests for run_chat: event sequence, citations, persistence, error path."""

from __future__ import annotations

import pytest

from ml.agent.events import (
    AgentError,
    Done,
    FinalAnswer,
    PlanCreated,
    ToolCall,
    ToolResult,
)
from ml.agent.orchestrator import run_chat
from ml.agent.ports import AgentDeps, FeatureRecord, ParcelRecord
from tests.ml.agent.fakes import FakeChatMemory, FakeLLMBackend, FakeParcelReader


@pytest.fixture(autouse=True)
def _no_model(monkeypatch) -> None:
    from ml.agent.tools import classify_parcel as mod

    monkeypatch.setattr(mod, "_load_classifier", lambda uri: None)


def _deps(memory: FakeChatMemory | None = None) -> AgentDeps:
    parcels = [
        ParcelRecord(id=10, aoi_id=1, crop_class="Meadow", confidence=0.9, area_ha=4.2, year=2023),
    ]
    features = {10: FeatureRecord(parcel_id=10, year=2023, ndvi_stats={"mean": 0.7})}
    return AgentDeps(
        parcels=FakeParcelReader(parcels, features),
        memory=memory or FakeChatMemory(),
    )


async def _collect(gen):  # type: ignore[no-untyped-def]
    return [event async for event in gen]


async def test_event_sequence_with_llm(monkeypatch) -> None:
    """Happy path: plan -> tool_call -> tool_result -> final_answer -> done."""
    fake = FakeLLMBackend(text="El AOI tiene 1 parcela de pradera (4.2 ha).")
    monkeypatch.setattr("ml.agent.orchestrator.get_backend", lambda variant: fake)

    memory = FakeChatMemory()
    events = await _collect(
        run_chat(
            session_id="s1",
            user_message="Que cultivos hay en el AOI?",
            llm_variant="gemini",
            deps=_deps(memory),
        )
    )

    types = [type(e) for e in events]
    assert types[0] is PlanCreated
    assert ToolCall in types
    assert ToolResult in types
    assert types[-2] is FinalAnswer
    assert types[-1] is Done

    final = next(e for e in events if isinstance(e, FinalAnswer))
    assert final.text == "El AOI tiene 1 parcela de pradera (4.2 ha)."
    assert final.citations  # citations propagated from findings
    assert final.citations[0].tool_call_id

    # Both turns persisted.
    roles = [t.role for t in memory.turns]
    assert roles == ["user", "assistant"]


async def test_template_fallback_when_llm_unavailable(monkeypatch) -> None:
    """If the backend raises, the answer is a marked deterministic template."""

    class BrokenBackend:
        name = "broken"

        async def generate(self, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("vertex unreachable")

    monkeypatch.setattr(
        "ml.agent.orchestrator.get_backend", lambda variant: BrokenBackend()
    )

    events = await _collect(
        run_chat(
            session_id="s1",
            user_message="cultivos?",
            llm_variant="qwen35",
            deps=_deps(),
        )
    )
    final = next(e for e in events if isinstance(e, FinalAnswer))
    assert "[respuesta determinista sin LLM]" in final.text
    assert "Meadow" in final.text
    assert final.citations


async def test_plan_includes_ndvi_step_for_health_question(monkeypatch) -> None:
    monkeypatch.setattr(
        "ml.agent.orchestrator.get_backend", lambda variant: FakeLLMBackend()
    )
    events = await _collect(
        run_chat(
            session_id="s1",
            user_message="Como esta la salud NDVI de las parcelas?",
            llm_variant="gemini",
            deps=_deps(),
        )
    )
    plan = next(e for e in events if isinstance(e, PlanCreated))
    assert any("NDVI" in s for s in plan.steps)
    tool_calls = [e for e in events if isinstance(e, ToolCall)]
    assert {c.tool for c in tool_calls} == {"classify_parcel", "compute_ndvi"}


async def test_error_path_emits_error_then_done(monkeypatch) -> None:
    """An exception inside the loop surfaces as AgentError + Done."""

    class BrokenReader:
        async def list_parcels_in_aoi(self, **kwargs):  # type: ignore[no-untyped-def]
            raise ValueError("db down")

        async def get_features(self, **kwargs):  # type: ignore[no-untyped-def]
            return None

    monkeypatch.setattr(
        "ml.agent.orchestrator.get_backend", lambda variant: FakeLLMBackend()
    )
    deps = AgentDeps(parcels=BrokenReader(), memory=FakeChatMemory())

    events = await _collect(
        run_chat(
            session_id="s1",
            user_message="cultivos?",
            llm_variant="gemini",
            deps=deps,
        )
    )
    assert isinstance(events[-2], AgentError)
    assert isinstance(events[-1], Done)
    assert "db down" in events[-2].message
