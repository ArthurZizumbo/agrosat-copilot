"""``/chat`` SSE stream tests (US-046 AC-3: perceiver observation before done).

The :class:`~backend.app.services.chat_service.ChatService` MVP wires the
perceiver into an SSE stream. These tests consume the async generator with the
perceiver fully mocked (no DB, no LLM, no Gemini loop) and assert:

- a ``perceiver_observation`` event is emitted BEFORE the terminal ``done`` event
  when the request carries a subject (a ``parcel_id``);
- the observation frame carries the real ``to_prompt_block`` rendering and the
  structured fields, and the order is
  ``[perceiver_observation, text_delta, done]``;
- with no subject (no parcel, no AOI) the stream still completes with ``done`` and
  emits no perceiver observation;
- a perceiver failure yields a terminal ``error`` frame instead of ``done`` (the
  client always gets a terminal event).

External boundaries mocked: ``get_pool`` (so no asyncpg pool is built) and
``PerceiverLayer`` (so no classifier / descriptor / DB runs).
"""

from __future__ import annotations

import json
from uuid import UUID

import backend.app.services.chat_service as chat_mod
from backend.app.services.chat_service import ChatMessage, ChatRequest, ChatService
from ml.agent.perceiver import PerceiverObservation

_SESSION = UUID("11111111-1111-1111-1111-111111111111")

_OBSERVATION = PerceiverObservation(
    parcel_id=11,
    crop_class="wheat",
    confidence=0.88,
    phenology_text="Fenologia: pico NDVI 0.820 en el dia 185.",
    vigor="high",
    class_probabilities={"wheat": 0.88, "maize": 0.12},
    description="Parcela de trigo en senescencia.",
)


class _SettingsStub:
    """Minimal settings stub so ``ChatService`` never reads the real .env.local."""

    rag_enabled = False


class _FakePerceiver:
    """``PerceiverLayer`` double returning a fixed observation (no DB / LLM)."""

    def __init__(self, ctx) -> None:
        self._ctx = ctx

    async def observe(self, parcel_id: int) -> PerceiverObservation:
        return _OBSERVATION

    async def observe_aoi(self, aoi, year: int) -> PerceiverObservation:
        return _OBSERVATION


class _RaisingPerceiver:
    """``PerceiverLayer`` double whose ``observe`` raises (failure path)."""

    def __init__(self, ctx) -> None:
        self._ctx = ctx

    async def observe(self, parcel_id: int) -> PerceiverObservation:
        raise RuntimeError("classifier exploded")

    async def observe_aoi(self, aoi, year: int) -> PerceiverObservation:
        raise RuntimeError("classifier exploded")


def _service(monkeypatch, perceiver_cls) -> ChatService:
    """Build a ``ChatService`` with the pool + perceiver mocked out."""

    async def _fake_get_pool():
        return object()  # never used: the perceiver is mocked

    monkeypatch.setattr(chat_mod, "get_pool", _fake_get_pool)
    monkeypatch.setattr(chat_mod, "PerceiverLayer", perceiver_cls)
    return ChatService(settings=_SettingsStub())  # type: ignore[arg-type]


def _parse_frames(raw_frames: list[str]) -> list[tuple[str, dict]]:
    """Parse SSE frames into ``(event, data_dict)`` tuples."""
    parsed: list[tuple[str, dict]] = []
    for frame in raw_frames:
        event_line, data_line = frame.strip().split("\n", 1)
        event = event_line.removeprefix("event: ").strip()
        data = json.loads(data_line.removeprefix("data: ").strip())
        parsed.append((event, data))
    return parsed


async def _collect(service: ChatService, request: ChatRequest) -> list[tuple[str, dict]]:
    """Drain the SSE async generator into parsed ``(event, data)`` tuples."""
    frames = [
        frame
        async for frame in service.stream(
            request.messages, _SESSION, request=request
        )
    ]
    return _parse_frames(frames)


# ---------------------------------------------------------------------------
# AC-3: perceiver_observation before done
# ---------------------------------------------------------------------------
async def test_perceiver_observation_before_done(monkeypatch) -> None:
    """A request with a parcel emits ``perceiver_observation`` before ``done``."""
    service = _service(monkeypatch, _FakePerceiver)
    request = ChatRequest(
        messages=[ChatMessage(role="user", content="describe la parcela")],
        session_id=_SESSION,
        parcel_id=11,
    )

    events = await _collect(service, request)
    names = [name for name, _ in events]

    assert names == ["perceiver_observation", "text_delta", "done"]
    # The perceiver frame is strictly before the terminal frame.
    assert names.index("perceiver_observation") < names.index("done")


async def test_perceiver_observation_payload_is_real_text(monkeypatch) -> None:
    """The observation frame carries the prompt block and structured fields."""
    service = _service(monkeypatch, _FakePerceiver)
    request = ChatRequest(session_id=_SESSION, parcel_id=11)

    events = dict(await _collect(service, request))
    payload = events["perceiver_observation"]

    assert payload["observation"]["crop_class"] == "wheat"
    assert payload["observation"]["vigor"] == "high"
    block = payload["prompt_block"]
    assert isinstance(block, str) and block.strip()
    assert "wheat" in block
    # Be My Eyes: the reasoner-facing block is pure text, no tensor/array reprs.
    # (The block header reads "sin logits", so the bare word is legitimate.)
    for forbidden in ("tensor(", "array(", "ndarray", "dtype", "predict_proba"):
        assert forbidden not in block
    # ``done`` reports that the observation was emitted.
    assert events["done"]["perceiver_observation_emitted"] is True


async def test_no_subject_completes_without_observation(monkeypatch) -> None:
    """No parcel and no AOI -> stream completes with ``done`` and no observation."""
    service = _service(monkeypatch, _FakePerceiver)
    request = ChatRequest(
        messages=[ChatMessage(role="user", content="hola")], session_id=_SESSION
    )

    events = await _collect(service, request)
    names = [name for name, _ in events]

    assert "perceiver_observation" not in names
    assert names[-1] == "done"
    assert dict(events)["done"]["perceiver_observation_emitted"] is False


async def test_perceiver_failure_yields_terminal_error(monkeypatch) -> None:
    """A perceiver exception emits a terminal ``error`` frame instead of ``done``."""
    service = _service(monkeypatch, _RaisingPerceiver)
    request = ChatRequest(session_id=_SESSION, parcel_id=11)

    events = await _collect(service, request)
    names = [name for name, _ in events]

    assert names[-1] == "error"
    assert "done" not in names
    assert dict(events)["error"]["message"] == "perceiver_observation_failed"
