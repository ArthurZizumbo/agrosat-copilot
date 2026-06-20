"""``/chat`` SSE stream tests (US-046 perceiver + US-047 real reasoner).

The :class:`~backend.app.services.chat_service.ChatService` wires the perceiver
*and* the real reasoner agent into an SSE stream. These tests consume the async
generator with the perceiver mocked (no DB, no LLM) and the reasoner injected as
a stub ``agent_factory`` (no ``google-genai`` client, no network), and assert:

US-046 (perceiver):

- a ``perceiver_observation`` event is emitted BEFORE the terminal ``done`` event
  when the request carries a subject (a ``parcel_id``);
- the observation frame carries the real ``to_prompt_block`` rendering and the
  structured fields;
- with no subject (no parcel, no AOI) the stream still completes and emits no
  perceiver observation;
- a perceiver failure yields a terminal ``error`` frame instead of ``done``.

US-047 (reasoner forwarding):

- the agent's own events (``tool_call`` / ``tool_result`` / ``text_delta`` /
  ``done``) are forwarded as SSE frames after the perceiver observation, so a
  full subject request streams
  ``[perceiver_observation, tool_call, tool_result, text_delta, done]``;
- the perceiver's ``to_prompt_block`` is injected as the leading ``system``
  grounding turn in the message history handed to the agent;
- an agent-emitted terminal ``error`` is forwarded as the terminal SSE frame.

External boundaries mocked: ``get_pool`` (no asyncpg pool), ``PerceiverLayer``
(no classifier / DB) and the reasoner ``agent_factory`` (no Gemini / vLLM call).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from uuid import UUID

import pytest

import backend.app.services.chat_service as chat_mod
from backend.app.services.chat_service import ChatMessage, ChatRequest, ChatService
from ml.agent.events import (
    AgentEvent,
    DoneEvent,
    ErrorEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)
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
    llm_variant_default = "gemini"
    gemini_model = "gemini-3.5-flash"


class _FakeConn:
    """asyncpg connection double for ``ChatService._resolve_variant`` (US-054).

    Supports the exact contract that ``_resolve_variant`` exercises:
    ``conn.transaction()`` (async context manager), ``conn.execute`` (the
    ``set_config`` RLS prime) and ``conn.fetchrow`` (the ``SELECT llm_model``
    row). ``fetchrow`` returns ``{"llm_model": variant}`` so the service reads
    the persisted variant; ``variant=None`` mimics a missing row so the service
    degrades to its ``gemini`` fallback.
    """

    def __init__(self, variant: str | None) -> None:
        self._variant = variant

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
        return None if self._variant is None else {"llm_model": self._variant}


class _FakePool:
    """asyncpg pool double: ``acquire``/``release`` hand out a :class:`_FakeConn`."""

    def __init__(self, variant: str | None = None) -> None:
        self._variant = variant

    async def acquire(self) -> _FakeConn:
        return _FakeConn(self._variant)

    async def release(self, conn: _FakeConn) -> None:
        return None


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


class _StubAgent:
    """Reasoner double: yields a scripted :data:`AgentEvent` stream (no network).

    Records the ``messages`` history and ``session_id`` it was driven with so the
    grounding-injection and multi-tenancy assertions can inspect them. The default
    script (``text_delta`` then ``done``) mimics a plain answer; tests that need a
    tool round-trip or an error pass an explicit ``events`` list.
    """

    last_messages: list[dict] | None = None
    last_session_id: UUID | None = None

    def __init__(self, events: list[AgentEvent] | None = None) -> None:
        self._events: list[AgentEvent] = (
            events
            if events is not None
            else [TextDeltaEvent(text="Respuesta del agente."), DoneEvent()]
        )

    async def stream_response(
        self, messages: list[dict], session_id: UUID, ctx
    ) -> AsyncIterator[AgentEvent]:
        """Record the call inputs and replay the scripted events."""
        type(self).last_messages = list(messages)
        type(self).last_session_id = session_id
        for event in self._events:
            yield event


class _RaisingAgent:
    """Reasoner double whose ``stream_response`` raises mid-stream.

    Models a backend that surfaces a failure as an exception (rather than the
    documented terminal :class:`ErrorEvent`): used to pin the service boundary
    -- the perceiver observation is emitted first and only then does the reasoner
    error surface.
    """

    def __init__(self, events: list[AgentEvent] | None = None) -> None:
        self._events = events or []

    async def stream_response(
        self, messages: list[dict], session_id: UUID, ctx
    ) -> AsyncIterator[AgentEvent]:
        for event in self._events:
            yield event
        raise RuntimeError("backend connection reset")


def _service(monkeypatch, perceiver_cls, agent_events=None) -> ChatService:
    """Build a ``ChatService`` with the pool, perceiver and reasoner mocked out.

    The reasoner is injected as a stub ``agent_factory`` so no ``google-genai``
    client is constructed and no network call happens; ``agent_events`` scripts
    the events the stub agent replays (defaults to ``text_delta`` + ``done``).
    """

    async def _fake_get_pool():
        return _FakePool()  # _resolve_variant reads llm_model -> None -> gemini fallback

    monkeypatch.setattr(chat_mod, "get_pool", _fake_get_pool)
    monkeypatch.setattr(chat_mod, "PerceiverLayer", perceiver_cls)
    _StubAgent.last_messages = None
    _StubAgent.last_session_id = None

    def _stub_factory(variant: str, *, settings) -> _StubAgent:
        return _StubAgent(events=agent_events)

    return ChatService(
        settings=_SettingsStub(),  # type: ignore[arg-type]
        agent_factory=_stub_factory,
    )


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
    frames = [frame async for frame in service.stream(request.messages, _SESSION, request=request)]
    return _parse_frames(frames)


# ---------------------------------------------------------------------------
# AC-2: default reasoner model is gemini-3.5-flash (deviation from the original
# AC's gemini-2.5-pro, decided by Arthur for cost/latency).
# ---------------------------------------------------------------------------
# NOTE (US-054): the former ``test_default_reasoner_model_is_gemini_flash`` /
# ``test_reasoner_model_reads_gemini_model_setting`` /
# ``test_default_model_passed_to_agent_factory`` /
# ``test_qwen_variant_passes_vllm_model_to_factory`` cases asserted the old
# ``_reasoner_model()`` + ``(model)`` factory contract, both removed by the D3
# change (the variant is now read per-request from ``chat_sessions.llm_model``).
# Their replacements live in ``tests/integration/test_chat_uses_session_model.py``
# (the factory now receives the persisted *variant*, with a gemini fallback).


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
    # The terminal frame is the agent's ``done`` (no extra payload fields).
    assert events["done"] == {}


async def test_no_subject_completes_without_observation(monkeypatch) -> None:
    """No parcel and no AOI -> stream completes with ``done`` and no observation."""
    service = _service(monkeypatch, _FakePerceiver)
    request = ChatRequest(messages=[ChatMessage(role="user", content="hola")], session_id=_SESSION)

    events = await _collect(service, request)
    names = [name for name, _ in events]

    assert "perceiver_observation" not in names
    # No grounding -> the reasoner streams its answer then ``done``.
    assert names == ["text_delta", "done"]


async def test_perceiver_failure_yields_terminal_error(monkeypatch) -> None:
    """A perceiver exception emits a terminal ``error`` frame instead of ``done``."""
    service = _service(monkeypatch, _RaisingPerceiver)
    request = ChatRequest(session_id=_SESSION, parcel_id=11)

    events = await _collect(service, request)
    names = [name for name, _ in events]

    assert names[-1] == "error"
    assert "done" not in names
    assert dict(events)["error"]["message"] == "perceiver_observation_failed"


# ---------------------------------------------------------------------------
# US-047: the real reasoner's events are forwarded over SSE
# ---------------------------------------------------------------------------
async def test_reasoner_tool_events_forwarded_after_observation(monkeypatch) -> None:
    """A subject request streams perceiver + the agent's tool/text/done frames.

    The stub agent replays a full reasoner round-trip (a ``tool_call`` /
    ``tool_result`` pair, a ``text_delta`` and a terminal ``done``); the service
    must emit the ``perceiver_observation`` first and then forward every agent
    event verbatim, in order.
    """
    agent_events: list[AgentEvent] = [
        ToolCallEvent(name="list_parcels", arguments={"session_id": str(_SESSION)}),
        ToolResultEvent(name="list_parcels", result={"count": 2}, ok=True),
        TextDeltaEvent(text="Tienes 2 parcelas."),
        DoneEvent(),
    ]
    service = _service(monkeypatch, _FakePerceiver, agent_events=agent_events)
    request = ChatRequest(
        messages=[ChatMessage(role="user", content="lista mis parcelas")],
        session_id=_SESSION,
        parcel_id=11,
    )

    events = await _collect(service, request)
    names = [name for name, _ in events]

    assert names == [
        "perceiver_observation",
        "tool_call",
        "tool_result",
        "text_delta",
        "done",
    ]

    by_name = dict(events)
    # The forwarded frames carry the agent payload minus the ``type`` discriminant.
    assert by_name["tool_call"] == {
        "name": "list_parcels",
        "arguments": {"session_id": str(_SESSION)},
        "call_id": None,
    }
    assert by_name["tool_result"] == {
        "name": "list_parcels",
        "result": {"count": 2},
        "ok": True,
    }
    assert by_name["text_delta"] == {"text": "Tienes 2 parcelas."}


async def test_perceiver_block_injected_as_grounding_turn(monkeypatch) -> None:
    """The perceiver ``to_prompt_block`` leads the agent's message history.

    The reasoner must reason over the structured TEXT the agent "saw"; the service
    injects it as a leading ``system`` turn before the user's messages.
    """
    service = _service(monkeypatch, _FakePerceiver)
    request = ChatRequest(
        messages=[ChatMessage(role="user", content="describe la parcela")],
        session_id=_SESSION,
        parcel_id=11,
    )

    await _collect(service, request)

    messages = _StubAgent.last_messages
    assert messages is not None
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == _OBSERVATION.to_prompt_block()
    assert messages[1] == {"role": "user", "content": "describe la parcela"}
    # The tenant session id is threaded into the reasoner unchanged.
    assert _StubAgent.last_session_id == _SESSION


async def test_locale_injects_language_instruction_as_leading_turn(monkeypatch) -> None:
    """US-057: ``locale`` prepends a language-instruction ``system`` turn.

    A request with ``locale='it'`` and no subject (no perceiver grounding) must
    lead the reasoner history with the Italian language instruction so the frozen
    reasoner answers in the user's UI language.
    """
    service = _service(monkeypatch, _FakePerceiver)
    request = ChatRequest(
        messages=[ChatMessage(role="user", content="ciao")],
        session_id=_SESSION,
        locale="it",
    )

    await _collect(service, request)

    messages = _StubAgent.last_messages
    assert messages is not None
    assert messages[0] == {"role": "system", "content": "Rispondi sempre in italiano."}
    assert messages[-1] == {"role": "user", "content": "ciao"}


async def test_no_locale_has_no_language_instruction(monkeypatch) -> None:
    """Without ``locale`` no language-instruction turn is injected (agent default)."""
    service = _service(monkeypatch, _FakePerceiver)
    request = ChatRequest(
        messages=[ChatMessage(role="user", content="hola")],
        session_id=_SESSION,
    )

    await _collect(service, request)

    messages = _StubAgent.last_messages
    assert messages is not None
    assert all("sempre" not in m["content"] and "siempre" not in m["content"] for m in messages)


async def test_no_subject_history_has_no_grounding_turn(monkeypatch) -> None:
    """Without a subject the agent history is the bare user messages (no system)."""
    service = _service(monkeypatch, _FakePerceiver)
    request = ChatRequest(messages=[ChatMessage(role="user", content="hola")], session_id=_SESSION)

    await _collect(service, request)

    messages = _StubAgent.last_messages
    assert messages == [{"role": "user", "content": "hola"}]


async def test_agent_error_event_forwarded_as_terminal_frame(monkeypatch) -> None:
    """An agent-emitted ``error`` is forwarded as the terminal SSE frame.

    The agent turns its own backend failures into a terminal ``error`` event; the
    service forwards it as-is (it does not re-wrap it), so the stream ends with
    ``error`` and the perceiver observation still precedes it.
    """
    agent_events: list[AgentEvent] = [ErrorEvent(message="backend 503")]
    service = _service(monkeypatch, _FakePerceiver, agent_events=agent_events)
    request = ChatRequest(session_id=_SESSION, parcel_id=11)

    events = await _collect(service, request)
    names = [name for name, _ in events]

    assert names == ["perceiver_observation", "error"]
    assert dict(events)["error"] == {"message": "backend 503"}


async def test_reasoner_raise_surfaces_after_observation(monkeypatch) -> None:
    """If the reasoner *raises* (not emits ``ErrorEvent``) the error propagates.

    The production agent (:meth:`~ml.agent.agent.Agent.stream_response`) never
    raises -- it converts any backend or tool failure into a terminal
    ``ErrorEvent`` (covered by
    :func:`test_agent_error_event_forwarded_as_terminal_frame`). This test pins
    the service contract for a misbehaving reasoner that *does* raise: the
    perceiver observation is still emitted first, then the exception surfaces out
    of the generator (the service does not silently swallow reasoner failures).
    """

    def _raising_factory(variant: str, *, settings) -> _RaisingAgent:
        return _RaisingAgent()

    async def _fake_get_pool():
        return _FakePool()

    monkeypatch.setattr(chat_mod, "get_pool", _fake_get_pool)
    monkeypatch.setattr(chat_mod, "PerceiverLayer", _FakePerceiver)
    service = ChatService(
        settings=_SettingsStub(),  # type: ignore[arg-type]
        agent_factory=_raising_factory,
    )
    request = ChatRequest(session_id=_SESSION, parcel_id=11)

    emitted: list[tuple[str, dict]] = []
    with pytest.raises(RuntimeError, match="backend connection reset"):
        async for frame in service.stream(request.messages, _SESSION, request=request):
            emitted.extend(_parse_frames([frame]))

    # The perceiver frame was flushed to the client before the reasoner blew up.
    assert [name for name, _ in emitted] == ["perceiver_observation"]
