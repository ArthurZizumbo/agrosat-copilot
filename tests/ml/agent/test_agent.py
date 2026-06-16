"""Tests for the agent factory and the manual function-calling loop (US-047).

The agent (:mod:`ml.agent.agent`) drives a manual ``google-genai`` 2.6
function-calling loop: it asks a backend to generate, parses any function calls,
validates their arguments with the tool's Pydantic ``*Input`` model, runs the
real tool coroutine under a session-scoped :class:`~ml.agent.context.ToolContext`,
feeds the responses back, and finally streams the model's text answer.

Every external boundary is replaced by an in-memory double:

* the LLM backend is a :class:`FakeBackend` yielding *scripted* chunks per turn
  (a ``function_call`` chunk first, then a text chunk) -- no ``google-genai``
  client, no network;
* the database is the US-045 :class:`~tests.ml.agent.conftest.FakeConn`, injected
  by monkeypatching ``session_scoped_conn`` in the ``list_parcels`` tool module,
  so the real tool runs but never hits PostgreSQL.

These doubles let us assert the *order* of the emitted
:data:`~ml.agent.events.AgentEvent` stream end-to-end (happy path, invalid args,
tool exception) and the ``MAX_TURNS`` guard, all offline and deterministically.
"""

from __future__ import annotations

import sys
import types as pytypes
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import pytest

import ml.agent.tools.parcels as parcels_mod
from ml.agent.agent import MAX_TURNS, Agent, create_agent
from ml.agent.events import (
    DoneEvent,
    ErrorEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from ml.agent.tools import ToolSpec, get_sync_tools, get_tool
from tests.ml.agent.conftest import SESSION_A, FakeConn, fake_session_scoped_conn

# ---------------------------------------------------------------------------
# Backend / chunk doubles
# ---------------------------------------------------------------------------


@dataclass
class FakeFunctionCall:
    """Duck-typed stand-in for ``google.genai.types.FunctionCall``.

    The agent reads ``name`` / ``args`` (and an optional ``id``) off the chunk's
    ``function_call`` attribute, so a plain dataclass is enough.
    """

    name: str
    args: dict[str, Any]
    id: str | None = None


@dataclass
class FakeChunk:
    """One scripted backend chunk: a text delta and/or a function call.

    Mirrors the ``BackendChunk`` duck type the agent's ``_read_chunk`` expects:
    ``chunk.text`` (incremental text) and ``chunk.function_call`` (a requested
    call). Either may be ``None``.
    """

    text: str | None = None
    function_call: FakeFunctionCall | None = None


@dataclass
class FakeBackend:
    """Scripted :class:`~ml.agent.backends.LLMBackend` double (no network).

    ``turns`` is a list of per-turn chunk lists: turn *i* yields ``turns[i]``.
    Each ``generate_stream`` call consumes the next turn; calls beyond the script
    yield a single terminal text chunk so a runaway loop still terminates in the
    happy-path helpers (the dedicated ``MAX_TURNS`` test overrides this).

    Every call records the ``(tools, system_instruction)`` it was invoked with so
    tests can assert the agent advertised the right declarations / prompt.
    """

    turns: list[list[FakeChunk]]
    model: str = "fake-model"
    calls: list[dict[str, Any]] = field(default_factory=list)
    _turn: int = 0

    async def generate_stream(
        self,
        *,
        contents: list,
        tools: list,
        system_instruction: str,
    ) -> AsyncIterator[FakeChunk]:
        """Yield the next scripted turn's chunks, recording the call."""
        self.calls.append(
            {
                "contents": list(contents),
                "tools": tools,
                "system_instruction": system_instruction,
            }
        )
        index = self._turn
        self._turn += 1
        chunks = self.turns[index] if index < len(self.turns) else [FakeChunk(text="(fin)")]
        for chunk in chunks:
            yield chunk


@dataclass
class LoopingBackend:
    """Backend that *always* requests the same tool: drives the MAX_TURNS guard.

    It never yields a text-only turn, so the loop can only stop via the
    ``MAX_TURNS`` cap. Records how many turns it was asked to generate.
    """

    tool_name: str = "list_parcels"
    model: str = "looping-model"
    generate_count: int = 0

    async def generate_stream(
        self, *, contents: list, tools: list, system_instruction: str
    ) -> AsyncIterator[FakeChunk]:
        """Always yield a single function-call chunk for ``tool_name``."""
        self.generate_count += 1
        yield FakeChunk(
            function_call=FakeFunctionCall(name=self.tool_name, args={}, id=None)
        )


def _list_parcels_spec() -> ToolSpec:
    """Resolve the real ``list_parcels`` :class:`ToolSpec` (its module imported)."""
    return get_tool("list_parcels")


def _agent_with_list_parcels(backend: Any) -> Agent:
    """Build an :class:`Agent` exposing only the real ``list_parcels`` tool."""
    return Agent(
        backend=backend,
        tools=[_list_parcels_spec()],
        instruction="system-prompt-de-prueba",
    )


def _patch_db(monkeypatch: pytest.MonkeyPatch, conn: FakeConn) -> None:
    """Point the ``list_parcels`` tool at the in-memory :class:`FakeConn`."""
    monkeypatch.setattr(
        parcels_mod, "session_scoped_conn", fake_session_scoped_conn(conn)
    )


async def _collect(agent: Agent, messages: list[dict], ctx) -> list:
    """Drain ``stream_response`` into a list of events."""
    return [event async for event in agent.stream_response(messages, SESSION_A, ctx)]


# ---------------------------------------------------------------------------
# create_agent / factory
# ---------------------------------------------------------------------------
def test_create_agent_exposes_the_five_sync_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """``create_agent`` wires the 5 sync tools and the analyst instruction.

    ``backends.py`` / ``prompts.py`` (sibling sub-task ml/A) may not exist yet, so
    the lazy imports inside ``create_agent`` are stubbed: a fake ``make_backend``
    returns an offline backend and ``ANALYST_SYSTEM_PROMPT`` is a sentinel string.
    """
    captured: dict[str, Any] = {}

    def _fake_make_backend(model: str, settings):
        captured["model"] = model
        captured["settings"] = settings
        return FakeBackend(turns=[[FakeChunk(text="ok")]], model=model)

    # Stub the lazily-imported sibling modules regardless of whether they exist.
    fake_backends = pytypes.ModuleType("ml.agent.backends")
    fake_backends.make_backend = _fake_make_backend  # type: ignore[attr-defined]
    fake_prompts = pytypes.ModuleType("ml.agent.prompts")
    fake_prompts.ANALYST_SYSTEM_PROMPT = "ANALYST-PROMPT-SENTINEL"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ml.agent.backends", fake_backends)
    monkeypatch.setitem(sys.modules, "ml.agent.prompts", fake_prompts)

    agent = create_agent()

    assert isinstance(agent, Agent)
    assert {spec.name for spec in agent.tools} == {spec.name for spec in get_sync_tools()}
    assert len(agent.tools) == 5
    assert agent.instruction == "ANALYST-PROMPT-SENTINEL"
    # The 5 sync tools each yield a declaration advertised to the model.
    assert len(agent._declarations) == 5
    assert {decl.name for decl in agent._declarations} == {
        spec.name for spec in get_sync_tools()
    }


def test_create_agent_routes_model_to_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model string passed to ``create_agent`` reaches ``make_backend``."""
    seen: dict[str, Any] = {}

    def _fake_make_backend(model: str, settings):
        seen["model"] = model
        return FakeBackend(turns=[[FakeChunk(text="ok")]], model=model)

    fake_backends = pytypes.ModuleType("ml.agent.backends")
    fake_backends.make_backend = _fake_make_backend  # type: ignore[attr-defined]
    fake_prompts = pytypes.ModuleType("ml.agent.prompts")
    fake_prompts.ANALYST_SYSTEM_PROMPT = "P"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ml.agent.backends", fake_backends)
    monkeypatch.setitem(sys.modules, "ml.agent.prompts", fake_prompts)

    create_agent(model="qwen35")
    assert seen["model"] == "qwen35"


def test_agent_declarations_filtered_to_its_tools() -> None:
    """An ``Agent`` only advertises declarations for the tools it was given."""
    agent = _agent_with_list_parcels(FakeBackend(turns=[[FakeChunk(text="x")]]))
    assert [decl.name for decl in agent._declarations] == ["list_parcels"]


# ---------------------------------------------------------------------------
# stream_response: happy path
# ---------------------------------------------------------------------------
async def test_stream_response_executes_tool_then_answers(
    monkeypatch: pytest.MonkeyPatch, make_ctx
) -> None:
    """Turn 1 calls ``list_parcels``; turn 2 answers -> ordered event stream.

    The backend first requests ``list_parcels`` (no args; the agent injects the
    session id), the real tool runs against a :class:`FakeConn` returning one
    parcel, and the second turn yields the final text.
    """
    conn = FakeConn(fetch_rows=[{"id": 7, "crop_class": "wheat", "confidence": 0.9}])
    _patch_db(monkeypatch, conn)

    backend = FakeBackend(
        turns=[
            [FakeChunk(function_call=FakeFunctionCall(name="list_parcels", args={}))],
            [FakeChunk(text="Tienes 1 parcela de trigo.")],
        ]
    )
    agent = _agent_with_list_parcels(backend)
    ctx = make_ctx()

    events = await _collect(agent, [{"role": "user", "content": "lista mis parcelas"}], ctx)
    kinds = [type(event) for event in events]

    assert kinds == [ToolCallEvent, ToolResultEvent, TextDeltaEvent, DoneEvent]

    tool_call, tool_result, text_delta, done = events
    assert isinstance(tool_call, ToolCallEvent)
    assert tool_call.name == "list_parcels"
    # The tenant session id was injected by the agent (the model never sent it).
    assert tool_call.arguments["session_id"] == str(SESSION_A)

    assert isinstance(tool_result, ToolResultEvent)
    assert tool_result.ok is True
    assert tool_result.result["count"] == 1
    assert tool_result.result["parcels"][0]["parcel_id"] == 7

    assert isinstance(text_delta, TextDeltaEvent)
    assert text_delta.text == "Tienes 1 parcela de trigo."
    assert isinstance(done, DoneEvent)


async def test_session_id_injected_with_raw_uuid_type(
    monkeypatch: pytest.MonkeyPatch, make_ctx
) -> None:
    """The injected ``session_id`` reaches the DB as the bound query parameter.

    ``ListParcelsInput`` is ``strict``: a string would be rejected. That the tool
    ran at all (and primed the RLS ``set_config`` with the session id) proves the
    raw :class:`~uuid.UUID` was injected, not its string form.
    """
    conn = FakeConn(fetch_rows=[])
    _patch_db(monkeypatch, conn)

    backend = FakeBackend(
        turns=[
            [FakeChunk(function_call=FakeFunctionCall(name="list_parcels", args={}))],
            [FakeChunk(text="Sin parcelas.")],
        ]
    )
    agent = _agent_with_list_parcels(backend)

    events = await _collect(agent, [{"role": "user", "content": "hola"}], make_ctx())
    assert any(isinstance(e, DoneEvent) for e in events)

    set_config_calls = conn.set_config_calls()
    assert set_config_calls, "the session-scoped RLS hook must have run"
    assert set_config_calls[0][1] == (str(SESSION_A),)


async def test_no_tool_call_streams_text_only(make_ctx) -> None:
    """A text-only first turn streams ``text_delta`` then ``done`` (no tools)."""
    backend = FakeBackend(turns=[[FakeChunk(text="Hola"), FakeChunk(text=", mundo.")]])
    agent = _agent_with_list_parcels(backend)

    events = await _collect(agent, [{"role": "user", "content": "saluda"}], make_ctx())
    kinds = [type(e) for e in events]

    assert kinds == [TextDeltaEvent, TextDeltaEvent, DoneEvent]
    assert [e.text for e in events if isinstance(e, TextDeltaEvent)] == ["Hola", ", mundo."]


async def test_grounding_system_turn_folds_into_contents(make_ctx) -> None:
    """A non user/model role (perceiver ``system`` block) reaches the backend.

    The agent folds any ``system``-like role into ``user`` so the grounding text
    is sent as context. We assert the backend saw two ``contents`` turns.
    """
    backend = FakeBackend(turns=[[FakeChunk(text="ok")]])
    agent = _agent_with_list_parcels(backend)

    messages = [
        {"role": "system", "content": "Observacion del perceiver: trigo, vigor alto."},
        {"role": "user", "content": "describe"},
    ]
    await _collect(agent, messages, make_ctx())

    first_call_contents = backend.calls[0]["contents"]
    assert len(first_call_contents) == 2
    # Both grounding + user folded to non-model roles (the grounding is "user").
    assert first_call_contents[0].role == "user"
    assert first_call_contents[1].role == "user"


# ---------------------------------------------------------------------------
# stream_response: multi-tool turn
# ---------------------------------------------------------------------------
async def test_two_tool_calls_in_one_turn(
    monkeypatch: pytest.MonkeyPatch, make_ctx
) -> None:
    """Two function calls in a single turn -> two call/result pairs, then text."""
    conn = FakeConn(fetch_rows=[{"id": 1, "crop_class": "maize", "confidence": 0.7}])
    _patch_db(monkeypatch, conn)

    backend = FakeBackend(
        turns=[
            [
                FakeChunk(function_call=FakeFunctionCall(name="list_parcels", args={})),
                FakeChunk(function_call=FakeFunctionCall(name="list_parcels", args={})),
            ],
            [FakeChunk(text="Listo.")],
        ]
    )
    agent = _agent_with_list_parcels(backend)

    events = await _collect(agent, [{"role": "user", "content": "x"}], make_ctx())
    kinds = [type(e) for e in events]

    assert kinds == [
        ToolCallEvent,
        ToolResultEvent,
        ToolCallEvent,
        ToolResultEvent,
        TextDeltaEvent,
        DoneEvent,
    ]


async def test_mixed_text_and_tool_call_turn_streams_text_and_runs_tool(
    monkeypatch: pytest.MonkeyPatch, make_ctx
) -> None:
    """A turn emitting reasoning text *and* a function call must do both (B-9).

    Gemini and Qwen frequently interleave reasoning text with a function call in
    the same turn. The agent must (1) stream that text as a ``TextDeltaEvent``
    *before* it runs the tool (so the reasoning reaches the SSE stream even though
    the turn also requested a tool) and (2) keep the text in the reconstructed
    ``model`` content so the conversation history -- and the next backend turn --
    retain the model's rationale. Previously the text was dropped on both counts.
    """
    conn = FakeConn(fetch_rows=[{"id": 3, "crop_class": "barley", "confidence": 0.8}])
    _patch_db(monkeypatch, conn)

    backend = FakeBackend(
        turns=[
            # Mixed turn: reasoning text first, then the function call, together.
            [
                FakeChunk(text="Voy a revisar tus parcelas. "),
                FakeChunk(
                    text="Consulto la base de datos.",
                    function_call=FakeFunctionCall(name="list_parcels", args={}),
                ),
            ],
            [FakeChunk(text="Tienes 1 parcela de cebada.")],
        ]
    )
    agent = _agent_with_list_parcels(backend)

    events = await _collect(agent, [{"role": "user", "content": "x"}], make_ctx())
    kinds = [type(e) for e in events]

    # The interleaved reasoning text is streamed BEFORE the tool runs; the tool
    # still executes and the loop finishes with the final answer.
    assert kinds == [
        TextDeltaEvent,  # "Voy a revisar tus parcelas. "
        TextDeltaEvent,  # "Consulto la base de datos."
        ToolCallEvent,
        ToolResultEvent,
        TextDeltaEvent,  # final answer turn
        DoneEvent,
    ]
    reasoning = [e.text for e in events[:2]]
    assert reasoning == ["Voy a revisar tus parcelas. ", "Consulto la base de datos."]
    tool_call = next(e for e in events if isinstance(e, ToolCallEvent))
    assert tool_call.name == "list_parcels"

    # The second backend turn must see the model's reasoning text preserved in the
    # reconstructed ``model`` content (before the function-call part), so the
    # history stays coherent across turns.
    second_turn_contents = backend.calls[1]["contents"]
    model_turn = next(
        c
        for c in second_turn_contents
        if c.role == "model"
        and any(getattr(p, "function_call", None) is not None for p in c.parts)
    )
    text_in_model_turn = [p.text for p in model_turn.parts if getattr(p, "text", None)]
    assert text_in_model_turn == [
        "Voy a revisar tus parcelas. ",
        "Consulto la base de datos.",
    ]
    # The function call still follows the reasoning text in the same model turn.
    assert any(getattr(p, "function_call", None) is not None for p in model_turn.parts)


# ---------------------------------------------------------------------------
# stream_response: error handling
# ---------------------------------------------------------------------------
async def test_invalid_args_emit_error_without_crashing(
    monkeypatch: pytest.MonkeyPatch, make_ctx
) -> None:
    """Bad tool arguments -> ``ErrorEvent`` (no ``ToolResult``), loop recovers.

    The model asks for ``get_parcel_timeseries`` with a malformed date; the strict
    ``*Input`` model rejects it, the agent emits a controlled ``ErrorEvent`` and a
    ``{"error": ...}`` function response, then the next turn answers normally.
    """
    timeseries_spec = get_tool("get_parcel_timeseries")
    backend = FakeBackend(
        turns=[
            [
                FakeChunk(
                    function_call=FakeFunctionCall(
                        name="get_parcel_timeseries",
                        # ``parcel_id`` missing + ``index`` not in the Literal set.
                        args={"start": "not-a-date", "end": "2019-12-31", "index": "bad"},
                    )
                )
            ],
            [FakeChunk(text="No pude leer la serie.")],
        ]
    )
    agent = Agent(
        backend=backend,
        tools=[timeseries_spec],
        instruction="p",
    )

    events = await _collect(agent, [{"role": "user", "content": "serie"}], make_ctx())
    kinds = [type(e) for e in events]

    # Validation failed before execution: error, no tool_call/tool_result pair,
    # then the loop continued to a final text answer.
    assert ToolCallEvent not in kinds
    assert ToolResultEvent not in kinds
    assert any(isinstance(e, ErrorEvent) for e in events)
    assert kinds[-1] is DoneEvent
    error = next(e for e in events if isinstance(e, ErrorEvent))
    assert "argumentos invalidos" in error.message


async def test_unknown_tool_emits_error(make_ctx) -> None:
    """A call to a tool the agent does not expose -> ``ErrorEvent`` then recovers."""
    backend = FakeBackend(
        turns=[
            [FakeChunk(function_call=FakeFunctionCall(name="does_not_exist", args={}))],
            [FakeChunk(text="ok")],
        ]
    )
    agent = _agent_with_list_parcels(backend)

    events = await _collect(agent, [{"role": "user", "content": "x"}], make_ctx())

    assert any(isinstance(e, ErrorEvent) for e in events)
    error = next(e for e in events if isinstance(e, ErrorEvent))
    assert "desconocida" in error.message
    assert isinstance(events[-1], DoneEvent)


async def test_tool_exception_yields_failed_tool_result(
    monkeypatch: pytest.MonkeyPatch, make_ctx
) -> None:
    """A tool raising at runtime -> ``ToolResultEvent(ok=False)``, no crash.

    ``session_scoped_conn`` is patched to raise, so the real tool body explodes;
    the agent must surface a failed result (with an ``{"error": ...}`` response fed
    back to the model) and let the loop continue to a final answer.
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _boom(_session_id: UUID):
        raise RuntimeError("conexion caida")
        yield  # pragma: no cover - unreachable, makes this an async generator

    monkeypatch.setattr(parcels_mod, "session_scoped_conn", _boom)

    backend = FakeBackend(
        turns=[
            [FakeChunk(function_call=FakeFunctionCall(name="list_parcels", args={}))],
            [FakeChunk(text="Hubo un problema con la base de datos.")],
        ]
    )
    agent = _agent_with_list_parcels(backend)

    events = await _collect(agent, [{"role": "user", "content": "x"}], make_ctx())
    kinds = [type(e) for e in events]

    assert kinds == [ToolCallEvent, ToolResultEvent, TextDeltaEvent, DoneEvent]
    tool_result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert tool_result.ok is False
    assert "error" in tool_result.result


async def test_backend_exception_yields_terminal_error(make_ctx) -> None:
    """A backend that raises mid-stream -> terminal ``ErrorEvent`` (never crashes)."""

    @dataclass
    class ExplodingBackend:
        model: str = "boom"

        async def generate_stream(self, *, contents, tools, system_instruction):
            raise RuntimeError("backend 503")
            yield  # pragma: no cover - makes this an async generator

    agent = _agent_with_list_parcels(ExplodingBackend())
    events = await _collect(agent, [{"role": "user", "content": "x"}], make_ctx())

    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    assert "agent loop failed" in events[0].message


# ---------------------------------------------------------------------------
# MAX_TURNS guard
# ---------------------------------------------------------------------------
async def test_max_turns_guard_stops_infinite_tool_loop(
    monkeypatch: pytest.MonkeyPatch, make_ctx
) -> None:
    """A backend that never answers stops at ``MAX_TURNS`` with an ``ErrorEvent``.

    The tool DB is stubbed so each call succeeds; the backend keeps asking for the
    same tool forever. The loop must bound itself to ``MAX_TURNS`` generate calls
    and end with a terminal error rather than spinning indefinitely.
    """
    conn = FakeConn(fetch_rows=[])
    _patch_db(monkeypatch, conn)

    backend = LoopingBackend(tool_name="list_parcels")
    agent = _agent_with_list_parcels(backend)

    events = await _collect(agent, [{"role": "user", "content": "x"}], make_ctx())

    assert backend.generate_count == MAX_TURNS
    assert isinstance(events[-1], ErrorEvent)
    assert str(MAX_TURNS) in events[-1].message
    # No DoneEvent: the loop never produced a final text answer.
    assert not any(isinstance(e, DoneEvent) for e in events)
    # Exactly MAX_TURNS tool calls were executed (one per turn).
    assert sum(isinstance(e, ToolCallEvent) for e in events) == MAX_TURNS
