"""Tests for the vision agent's tool-selection heuristic and event emission."""

from __future__ import annotations

import pytest

from ml.agent.ports import AgentDeps, FeatureRecord, ParcelRecord
from ml.agent.vision_agent import analyze
from tests.ml.agent.fakes import FakeChatMemory, FakeParcelReader


@pytest.fixture(autouse=True)
def _no_model(monkeypatch) -> None:
    from ml.agent.tools import classify_parcel as mod

    monkeypatch.setattr(mod, "_load_classifier", lambda uri: None)


def _deps() -> AgentDeps:
    parcels = [ParcelRecord(id=10, aoi_id=1, crop_class="Meadow", area_ha=4.2, year=2023)]
    features = {10: FeatureRecord(parcel_id=10, year=2023, ndvi_stats={"mean": 0.7})}
    return AgentDeps(parcels=FakeParcelReader(parcels, features), memory=FakeChatMemory())


async def test_classify_only_when_no_health_question() -> None:
    result = await analyze(
        session_id="s1", aoi_id=1, question="Que cultivos hay?", deps=_deps()
    )
    tools = [c.tool for c in result.tool_calls]
    assert tools == ["classify_parcel"]
    assert all(c.agent == "vision" for c in result.tool_calls)


async def test_also_ndvi_when_health_question() -> None:
    result = await analyze(
        session_id="s1", aoi_id=1, question="Como esta la salud / NDVI?", deps=_deps()
    )
    tools = [c.tool for c in result.tool_calls]
    assert tools == ["classify_parcel", "compute_ndvi"]


async def test_ndvi_trigger_multilingual() -> None:
    for question in ("vegetation health", "stato di salute della vegetazione", "estres hidrico"):
        result = await analyze(session_id="s1", aoi_id=1, question=question, deps=_deps())
        assert "compute_ndvi" in [c.tool for c in result.tool_calls]


async def test_findings_carry_real_call_id() -> None:
    result = await analyze(
        session_id="s1", aoi_id=1, question="cultivos y ndvi", deps=_deps()
    )
    call_ids = {c.call_id for c in result.tool_calls}
    # Every finding's citation must reference an emitted tool call.
    assert result.findings
    for f in result.findings:
        assert f.citation.tool_call_id in call_ids
    # tool_results pair 1:1 with tool_calls.
    assert {r.call_id for r in result.tool_results} == call_ids
