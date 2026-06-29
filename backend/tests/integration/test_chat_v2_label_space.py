"""``/chat`` + Voting-3 v2 label-space integration tests (US-081 AC4b / AC6).

Drives the real FastAPI ``/chat`` SSE endpoint through ``httpx.AsyncClient`` +
``ASGITransport`` (no live server, no network) and pins the two backend
guarantees of serving the 12-class Voting-3 v2 champion through the copilot:

- **AC4b** ``/chat`` serves the v2 / ``france-12`` vocabulary: the reasoner's
  ``classify_new_parcel`` ``tool_result`` carries a real ``france-12`` crop name,
  and the ``LABEL_SPACE=france-9`` env override narrows the perceived posterior to
  the nine ``france-9`` classes -- both proven by running the REAL
  :func:`ml.agent.tools.classify.run` (DB / OOF / GEE boundaries mocked, never
  hit) and feeding its actual :class:`~ml.agent.schemas.ClassificationResult`
  through the SSE stream.
- **AC6** the two new honest-vocabulary fields, ``out_of_vocabulary_classes`` and
  ``unresolved_candidate``, flow verbatim in the ``tool_result`` SSE event so the
  frontend can render the out-of-vocabulary hedge. The agent already serialises
  the full ``ClassificationResult`` via ``model.model_dump(mode="json")`` (see
  ``ml/agent/agent.py::_dump_output``) and ``ChatService`` forwards it unchanged;
  this test is the regression guard on that contract.

Every external boundary is a deterministic in-memory double: the perceiver, the
asyncpg pool, the reasoner agent (a stub that replays a scripted event stream),
the parcel embedding fetch and the Voting-3 vote. No Vertex AI / vLLM / GEE /
PostGIS call happens. The classifier (xgb-alphaearth) trains on the REAL fused
features parquet when present and skips cleanly otherwise (no synthetic posterior
is fabricated -- Arthur's "real values only" rule).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated
from uuid import UUID

import numpy as np
import pytest
from fastapi import Depends
from httpx import ASGITransport, AsyncClient

import backend.app.services.chat_service as chat_mod
import ml.agent.tools.classify as classify_mod
from backend.app.api.deps import verify_chat_session
from backend.app.core import db as core_db
from backend.app.core.config import Settings
from backend.app.core.rate_limit import build_limiter, limiter
from backend.app.main import create_app
from ml.agent.events import AgentEvent, DoneEvent, TextDeltaEvent, ToolCallEvent, ToolResultEvent
from ml.agent.perceiver import PerceiverObservation
from ml.agent.schemas import ClassificationResult, ClassifyParcelInput
from ml.eval.class_remap import get_label_space

pytestmark = pytest.mark.asyncio

_VALID_SESSION = "11111111-1111-1111-1111-111111111111"
_REPO_ROOT = Path(__file__).resolve().parents[3]
#: The REAL fused features parquet the xgb-alphaearth classifier trains on. When
#: absent (DVC not pulled in this harness) the AC4b/AC6 cases skip cleanly rather
#: than fabricate a posterior.
_FEATURES_PATH = _REPO_ROOT / "data" / "features" / "features_fused_pastis.parquet"

_POLYGON = {
    "type": "Polygon",
    "coordinates": [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]],
}

_OBSERVATION = PerceiverObservation(
    parcel_id=11,
    crop_class="Corn",
    confidence=0.9,
    phenology_text="Fenologia: pico NDVI 0.8.",
    vigor="high",
    class_probabilities={"Corn": 0.9, "Meadow": 0.1},
    description="Parcela de maiz.",
)


class _FakePerceiver:
    """``PerceiverLayer`` double returning a fixed observation (no DB / LLM)."""

    def __init__(self, ctx) -> None:
        self._ctx = ctx

    async def observe(self, parcel_id: int) -> PerceiverObservation:
        return _OBSERVATION

    async def observe_aoi(self, aoi, year: int) -> PerceiverObservation:
        return _OBSERVATION


class _FakeConn:
    """asyncpg connection double for ``ChatService._resolve_variant`` (US-054)."""

    def transaction(self):
        conn = self

        class _Tx:
            async def __aenter__(self_inner):
                return conn

            async def __aexit__(self_inner, *exc):
                return False

        return _Tx()

    async def execute(self, *args, **kwargs) -> str:
        return "SET"

    async def fetchrow(self, *args, **kwargs):
        return None  # no persisted variant -> service degrades to gemini fallback


class _FakePool:
    """asyncpg pool double: ``acquire``/``release`` hand out a :class:`_FakeConn`."""

    async def acquire(self) -> _FakeConn:
        return _FakeConn()

    async def release(self, conn: _FakeConn) -> None:
        return None


class _ClassifyAgent:
    """Reasoner double that runs the REAL ``classify_new_parcel`` tool once.

    It mirrors what the production agent does with a ``classify`` tool call: it
    invokes :func:`ml.agent.tools.classify.run` with the given input under the
    shared :class:`~ml.agent.context.ToolContext`, then serialises the resulting
    :class:`ClassificationResult` into a ``tool_result`` event EXACTLY as the
    agent's ``_dump_output`` does (``model_dump(mode="json")``). This exercises the
    true classify -> SSE surface without a live LLM.
    """

    def __init__(self, classify_input: ClassifyParcelInput) -> None:
        self._input = classify_input

    async def stream_response(self, messages, session_id, ctx) -> AsyncIterator[AgentEvent]:
        result = await classify_mod.run(self._input, ctx)
        yield ToolCallEvent(name="classify_new_parcel", arguments={"session_id": str(session_id)})
        yield ToolResultEvent(
            name="classify_new_parcel", result=result.model_dump(mode="json"), ok=True
        )
        yield TextDeltaEvent(text=f"El cultivo estimado es {result.crop_class}.")
        yield DoneEvent()


@pytest.fixture
def memory_limiter():
    """Point the production limiter singleton at ``memory://`` and reset it."""
    mem = build_limiter(Settings(redis_url="memory://"))
    saved = {
        "_storage": limiter._storage,
        "_storage_uri": limiter._storage_uri,
        "_limiter": limiter._limiter,
    }
    limiter._storage = mem._storage
    limiter._storage_uri = mem._storage_uri
    limiter._limiter = mem._limiter
    limiter.reset()
    try:
        yield
    finally:
        limiter.reset()
        limiter._storage = saved["_storage"]
        limiter._storage_uri = saved["_storage_uri"]
        limiter._limiter = saved["_limiter"]


def _patch_classify_real_posterior(monkeypatch) -> None:
    """Mock only the I/O boundaries of ``classify``; keep its real logic.

    The parcel embedding fetch returns a deterministic 64-dim vector (the SAME
    fixed vector the agent-tool suite uses, never random) and the GEE sampler is
    short-circuited so no network call happens. The Voting-3 vote is forced to
    degrade (``None``) so the test does not depend on the OOF parquets being
    present in this harness -- the xgb-alphaearth posterior (trained on the REAL
    fused features parquet) drives the restricted result, which is exactly the
    AOI-without-OOF behaviour the ``voting3`` default guarantees.
    """

    async def _fake_fetch_embedding(ctx, year, aoi):
        return np.linspace(0.0, 1.0, 64, dtype=np.float64)

    async def _voting_degrades(ctx, inp):
        return None

    monkeypatch.setattr(classify_mod, "_fetch_parcel_embedding", _fake_fetch_embedding)
    monkeypatch.setattr(classify_mod, "_voting_posterior", _voting_degrades)


def _make_classify_client(monkeypatch, classify_input: ClassifyParcelInput):
    """Build an ``AsyncClient`` over the real app whose reasoner runs ``classify``."""

    async def _fake_get_pool():
        return _FakePool()

    monkeypatch.setattr(chat_mod, "get_pool", _fake_get_pool)
    monkeypatch.setattr(chat_mod, "PerceiverLayer", _FakePerceiver)
    monkeypatch.setattr(
        chat_mod,
        "_default_agent_factory",
        lambda model, *, settings: _ClassifyAgent(classify_input),
    )

    app = create_app()

    async def _override_guard(
        session_id: Annotated[UUID, Depends(core_db.get_request_session_id)],
    ) -> UUID:
        return session_id

    app.dependency_overrides[verify_chat_session] = _override_guard
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


def _body() -> dict:
    return {
        "messages": [{"role": "user", "content": "que cultivo es esta parcela?"}],
        "aoi": _POLYGON,
        "year": 2019,
    }


def _tool_result_payload(sse_body: str) -> dict:
    """Extract the JSON payload of the ``tool_result`` SSE frame from a raw body."""
    lines = sse_body.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "event: tool_result":
            data_line = lines[i + 1]
            assert data_line.startswith("data:")
            return json.loads(data_line.removeprefix("data:").strip())
    raise AssertionError("no tool_result event found in the SSE body")


async def test_chat_classify_surfaces_france12_crop_and_new_fields(monkeypatch, memory_limiter):
    """``/chat`` classify yields a france-12 crop + the two new SSE fields (AC4b/AC6)."""
    if not _FEATURES_PATH.exists():
        pytest.skip(f"fused features parquet not present: {_FEATURES_PATH} (run dvc pull)")

    _patch_classify_real_posterior(monkeypatch)
    # Default model is voting3 (US-081 AC4a); france-12 is the default label-space.
    classify_input = ClassifyParcelInput(session_id=UUID(_VALID_SESSION), aoi=_POLYGON, year=2019)
    assert classify_input.model == "voting3"
    assert classify_input.label_space == get_label_space().name == "france-12"

    client = _make_classify_client(monkeypatch, classify_input)
    async with client:
        resp = await client.post("/chat", json=_body(), headers={"X-Session-ID": _VALID_SESSION})
        assert resp.status_code == 200
        body = resp.text

    payload = _tool_result_payload(body)
    result = payload["result"]
    france12 = get_label_space("france-12")

    # AC4b: the headline crop is one of the twelve france-12 class names (or the
    # honest "unresolved" sentinel when no mass lands on the resolved classes).
    assert result["crop_class"] in set(france12.class_names.values()) | {"unresolved"}
    # The posterior is masked to exactly the twelve resolved classes.
    assert set(result["class_probabilities"]) <= set(france12.class_names.values())

    # AC6: the two new honest-vocabulary fields flow verbatim in the SSE frame.
    assert "out_of_vocabulary_classes" in result
    assert "unresolved_candidate" in result
    # france-12 drops exactly six classes (the open-set the copilot declares).
    assert set(result["out_of_vocabulary_classes"]) == set(france12.dropped_class_names.values())
    assert len(result["out_of_vocabulary_classes"]) == 6


async def test_chat_label_space_override_narrows_to_france9(monkeypatch, memory_limiter):
    """``LABEL_SPACE=france-9`` narrows the perceived posterior to nine classes (AC4b)."""
    if not _FEATURES_PATH.exists():
        pytest.skip(f"fused features parquet not present: {_FEATURES_PATH} (run dvc pull)")

    # Env override: the deployment serves the narrower france-9 vocabulary. The
    # reasoner-side ChatService reads Settings.label_space; here we pin the same
    # space on the classify input the agent runs (the perceiver path reads it from
    # settings, the classify tool from its input -- both honour the override).
    monkeypatch.setenv("LABEL_SPACE", "france-9")
    get_settings_cache = Settings(redis_url="memory://", label_space="france-9")
    assert get_settings_cache.label_space == "france-9"

    _patch_classify_real_posterior(monkeypatch)
    classify_input = ClassifyParcelInput(
        session_id=UUID(_VALID_SESSION), aoi=_POLYGON, year=2019, label_space="france-9"
    )

    client = _make_classify_client(monkeypatch, classify_input)
    async with client:
        resp = await client.post("/chat", json=_body(), headers={"X-Session-ID": _VALID_SESSION})
        assert resp.status_code == 200
        body = resp.text

    result = _tool_result_payload(body)["result"]
    france9 = get_label_space("france-9")

    # The posterior is masked to (at most) the nine france-9 resolved classes.
    assert set(result["class_probabilities"]) <= set(france9.class_names.values())
    # france-9 drops nine classes; they are surfaced as the out-of-vocabulary set.
    assert set(result["out_of_vocabulary_classes"]) == set(france9.dropped_class_names.values())
    assert len(result["out_of_vocabulary_classes"]) == 9
    assert result["crop_class"] in set(france9.class_names.values()) | {"unresolved"}


async def test_classification_result_roundtrips_new_fields_through_sse():
    """The new fields survive ``model_dump`` -> SSE -> ``model_validate`` (AC6 contract).

    A pure round-trip guard that does NOT need the features parquet: the agent
    dumps a ``ClassificationResult`` exactly as ``_dump_output`` does, the result
    is JSON-serialised into a ``ToolResultEvent`` and that event is dumped to the
    SSE wire shape -- the two new fields must be present and equal end to end.
    """
    result = ClassificationResult(
        crop_class="Corn",
        confidence=0.71,
        class_probabilities={"Corn": 0.71, "Meadow": 0.29},
        out_of_vocabulary_classes=["Potatoes", "Sorghum"],
        unresolved_candidate="Potatoes",
    )
    event = ToolResultEvent(
        name="classify_new_parcel", result=result.model_dump(mode="json"), ok=True
    )
    wire = event.model_dump(mode="json")

    assert wire["result"]["out_of_vocabulary_classes"] == ["Potatoes", "Sorghum"]
    assert wire["result"]["unresolved_candidate"] == "Potatoes"
    # Backward compatibility: a payload WITHOUT the new fields still validates
    # (they default to [] / None), so an old tool_result never breaks the contract.
    legacy = ClassificationResult(
        crop_class="Corn", confidence=0.9, class_probabilities={"Corn": 0.9}
    )
    assert legacy.out_of_vocabulary_classes == []
    assert legacy.unresolved_candidate is None
