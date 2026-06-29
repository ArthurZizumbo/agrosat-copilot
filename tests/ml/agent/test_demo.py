"""Tests for the copilot-demo helpers (:mod:`ml.agent.demo`).

These helpers are the shared driver + renderers behind the two copilot notebooks
(Avance 6 and the US-079 transfer copilot). The notebooks call them with
``display=True`` (rendering via IPython/Polars); the tests call them with
``display=False`` and assert on the returned data, so nothing here imports a
notebook runtime or touches a network/LLM/database.

The agent-turn driver is exercised against the same in-memory doubles the rest of
the suite uses: a scripted backend yielding chunks (no ``google-genai`` client)
and the US-045 :class:`~tests.ml.agent.conftest.FakeConn` injected into the real
``list_parcels`` tool, so a turn runs end to end offline and deterministically.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

import ml.agent.tools.parcels as parcels_mod
from ml.agent import demo
from ml.agent.agent import Agent
from ml.agent.backends import GeminiBackend, OllamaBackend, VLLMOpenAIBackend
from ml.agent.prompts import ANALYST_SYSTEM_PROMPT
from ml.agent.tools import get_tool
from tests.ml.agent.conftest import (
    SESSION_A,
    FakeConn,
    FakeRecord,
    FakeSettings,
    fake_session_scoped_conn,
)

# ---------------------------------------------------------------------------
# Scripted backend doubles (mirror tests/ml/agent/test_agent.py, kept local)
# ---------------------------------------------------------------------------


@dataclass
class FakeFunctionCall:
    """Duck-typed ``google.genai`` function-call stand-in (name/args/id)."""

    name: str
    args: dict[str, Any]
    id: str | None = None
    thought_signature: bytes | None = None


@dataclass
class FakeChunk:
    """One scripted backend chunk: a text delta and/or a function call."""

    text: str | None = None
    function_call: FakeFunctionCall | None = None


@dataclass
class FakeBackend:
    """Scripted ``LLMBackend`` double: ``turns[i]`` is yielded on the i-th call."""

    turns: list[list[FakeChunk]]
    model: str = "fake-model"
    _turn: int = 0

    async def generate_stream(
        self, *, contents: list, tools: list, system_instruction: str
    ) -> AsyncIterator[FakeChunk]:
        index = self._turn
        self._turn += 1
        chunks = self.turns[index] if index < len(self.turns) else [FakeChunk(text="(fin)")]
        for chunk in chunks:
            yield chunk


@dataclass
class RaisingBackend:
    """Backend whose ``generate_stream`` raises -- drives the honest-failure path."""

    model: str = "down-model"
    message: str = "endpoint caido"

    async def generate_stream(
        self, *, contents: list, tools: list, system_instruction: str
    ) -> AsyncIterator[FakeChunk]:
        raise ConnectionError(self.message)
        yield FakeChunk(text="unreachable")  # pragma: no cover - generator marker


def _agent_with(backend: Any, tool_names: list[str]) -> Agent:
    """Build a real :class:`Agent` wired to a scripted backend and given tools."""
    return Agent(
        backend=backend,
        tools=[get_tool(name) for name in tool_names],
        instruction=ANALYST_SYSTEM_PROMPT,
    )


# ---------------------------------------------------------------------------
# Pure helpers: summarize / availability / endpoint liveness
# ---------------------------------------------------------------------------


def test_summarize_result_truncates_long_payloads() -> None:
    short = demo.summarize_result({"a": 1})
    assert short == '{"a": 1}'
    long = demo.summarize_result({"k": "x" * 500}, limit=40)
    assert len(long) == 44 and long.endswith(" ...")  # 40 chars + " ..."


def test_gemini_available_detects_key_vertex_and_neither() -> None:
    class _WithKey(FakeSettings):
        gemini_api_key = "AIza-demo"

    class _WithVertex(FakeSettings):
        google_genai_use_vertexai = "true"
        google_cloud_project = "agrosat-dev"

    ok_key, _ = demo.gemini_available(_WithKey())
    ok_vertex, _ = demo.gemini_available(_WithVertex())
    ok_none, why_none = demo.gemini_available(FakeSettings())
    assert ok_key is True
    assert ok_vertex is True
    assert ok_none is False and "sin GEMINI_API_KEY" in why_none


def test_endpoint_alive_with_injected_opener() -> None:
    class _Resp:
        status = 200

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def _up_2xx(url: str, timeout: float = 0.0) -> _Resp:
        return _Resp()

    def _down(url: str, timeout: float = 0.0) -> _Resp:
        raise OSError("connection refused")

    alive, _ = demo.endpoint_alive("http://127.0.0.1:8002/v1", urlopen=_up_2xx)
    dead, _ = demo.endpoint_alive("http://127.0.0.1:8002/v1", urlopen=_down)
    empty, _ = demo.endpoint_alive("", urlopen=_up_2xx)
    assert alive is True
    assert dead is False
    assert empty is False


def test_probe_availability_mixes_gemini_and_endpoints() -> None:
    def _down(url: str, timeout: float = 0.0) -> Any:
        raise OSError("refused")

    availability, rows = demo.probe_availability(
        demo.BACKEND_CATALOG.keys(),
        FakeSettings(),  # no gemini key, no vertex
        display=False,
        urlopen=_down,
    )
    # Gemini has no creds here; both on-prem endpoints are down -> all unavailable.
    assert availability == {
        "gemini-3.5-flash": False,
        "qwen3.6-vl": False,
        "qwen35": False,
    }
    assert {r["modelo"] for r in rows} == set(demo.BACKEND_CATALOG)
    assert all(r["disponible"] == "NO" for r in rows)


# ---------------------------------------------------------------------------
# Backend resolution table
# ---------------------------------------------------------------------------


def test_backend_overview_resolves_classes_and_endpoints() -> None:
    rows = demo.backend_overview(demo.BACKEND_CATALOG.keys(), FakeSettings(), display=False)
    by_model = {r["modelo"]: r for r in rows}
    assert by_model["gemini-3.5-flash"]["backend"] == GeminiBackend.__name__
    assert by_model["qwen3.6-vl"]["backend"] == OllamaBackend.__name__
    assert by_model["qwen35"]["backend"] == VLLMOpenAIBackend.__name__
    # On-prem rows expose a concrete OpenAI-compatible endpoint; Gemini does not.
    assert "8003" in by_model["qwen3.6-vl"]["endpoint"]
    assert "8002" in by_model["qwen35"]["endpoint"]
    assert by_model["gemini-3.5-flash"]["endpoint"] == "Vertex AI / GenAI"


# ---------------------------------------------------------------------------
# Tool inventory
# ---------------------------------------------------------------------------


def test_tool_inventory_lists_sync_and_deferred() -> None:
    rows = demo.tool_inventory(display=False)
    names = {r["herramienta"] for r in rows}
    assert {"classify_new_parcel", "explain_prediction", "retrieve_context"} <= names
    n_sync = sum(1 for r in rows if r["tipo"] == "sincrona")
    # 6 synchronous in-loop tools (incl. the Spatial-RAG grounding tool), 4 deferred.
    assert n_sync == 6
    assert all(
        r["comportamiento"] == ("NON_BLOCKING" if r["tipo"] == "diferida" else "BLOCKING")
        for r in rows
    )


# ---------------------------------------------------------------------------
# Agent-turn driver
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_agent_turn_text_only(make_ctx: Any) -> None:
    agent = _agent_with(FakeBackend(turns=[[FakeChunk(text="12 parcelas de trigo.")]]), [])
    record = await demo.run_agent_turn(
        agent,
        "Cuantas parcelas tengo?",
        ctx=make_ctx(),
        session_id=SESSION_A,
        render=False,
    )
    assert record["ok"] is True
    assert record["answer"] == "12 parcelas de trigo."
    assert record["n_tool_calls"] == 0
    assert record["tool_calls"] == []
    assert record["error"] is None
    assert record["latency_ms"] is not None


@pytest.mark.asyncio
async def test_run_agent_turn_runs_tool_then_answers(
    monkeypatch: pytest.MonkeyPatch, make_ctx: Any
) -> None:
    # Turn 1: the model asks for list_parcels; turn 2: it answers in text.
    backend = FakeBackend(
        turns=[
            [FakeChunk(function_call=FakeFunctionCall(name="list_parcels", args={}))],
            [FakeChunk(text="Tienes 1 parcela.")],
        ]
    )
    agent = _agent_with(backend, ["list_parcels"])
    # Inject a scripted DB connection into the real list_parcels tool (no Postgres).
    conn = FakeConn(fetch_rows=[FakeRecord(id=7, crop_class="wheat", confidence=0.91)])
    monkeypatch.setattr(parcels_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    record = await demo.run_agent_turn(
        agent, "Lista mis parcelas.", ctx=make_ctx(), session_id=SESSION_A, render=False
    )
    assert record["tool_calls"] == ["list_parcels"]
    assert record["n_tool_calls"] == 1
    assert record["ok"] is True
    assert record["answer"] == "Tienes 1 parcela."


@pytest.mark.asyncio
async def test_run_agent_turn_surfaces_backend_failure(make_ctx: Any) -> None:
    agent = _agent_with(RaisingBackend(message="tunel caido"), [])
    record = await demo.run_agent_turn(
        agent, "Hola", ctx=make_ctx(), session_id=SESSION_A, render=False
    )
    assert record["ok"] is False
    assert record["answer"] == ""
    # The agent loop wraps a backend error into an ErrorEvent message.
    assert record["error"] is not None and "tunel caido" in record["error"]


@pytest.mark.asyncio
async def test_run_tool_direct_success(monkeypatch: pytest.MonkeyPatch, make_ctx: Any) -> None:
    conn = FakeConn(fetch_rows=[FakeRecord(id=3, crop_class="barley", confidence=0.7)])
    monkeypatch.setattr(parcels_mod, "session_scoped_conn", fake_session_scoped_conn(conn))
    record = await demo.run_tool("list_parcels", {}, make_ctx(), render=False)
    assert record["ok"] is True
    assert record["error"] is None
    assert record["result"]["count"] == 1
    # The tenant session id is injected from ctx, not supplied by the caller.
    assert any("set_config" in sql for sql, _ in conn.calls)


@pytest.mark.asyncio
async def test_run_tool_direct_captures_validation_error(make_ctx: Any) -> None:
    # classify_new_parcel requires an aoi; omitting it must degrade honestly.
    record = await demo.run_tool("classify_new_parcel", {}, make_ctx(), render=False)
    assert record["ok"] is False
    assert record["result"] == {}
    assert record["error"] is not None


@pytest.mark.asyncio
async def test_run_backend_turn_skips_unavailable(make_ctx: Any) -> None:
    record = await demo.run_backend_turn(
        "qwen35",
        "Hola",
        settings=FakeSettings(),
        ctx=make_ctx(),
        session_id=SESSION_A,
        availability={"qwen35": False},
        render=False,
    )
    assert record == {
        "model": "qwen35",
        "ok": False,
        "available": False,
        "answer": "",
        "n_tool_calls": 0,
        "tool_calls": [],
        "latency_ms": None,
        "error": "backend no disponible",
    }


# ---------------------------------------------------------------------------
# Tables: cross-backend / perceiver / RAG
# ---------------------------------------------------------------------------


def test_cross_backend_table_shape() -> None:
    records = [
        {
            "model": "gemini-3.5-flash",
            "ok": True,
            "available": True,
            "answer": "ok",
            "n_tool_calls": 1,
            "tool_calls": ["classify_new_parcel"],
            "latency_ms": 800.0,
        },
        demo._blank_record("qwen35", available=False, error="no disponible"),
    ]
    rows = demo.cross_backend_table(records, display=False)
    by_backend = {r["backend"]: r for r in rows}
    assert by_backend["gemini-3.5-flash"]["respondio"] == "si"
    assert by_backend["gemini-3.5-flash"]["herramientas"] == "classify_new_parcel"
    assert by_backend["qwen35"]["disponible"] == "NO"
    assert by_backend["qwen35"]["herramientas"] == "-"


def test_save_and_load_backend_records(tmp_path: Any) -> None:
    ok_rec = {
        "model": "qwen35",
        "ok": True,
        "available": True,
        "answer": "hola",
        "n_tool_calls": 0,
        "tool_calls": [],
        "latency_ms": 12.0,
        "error": None,
    }
    skip_rec = demo._blank_record("qwen3.6-vl", available=False, error="backend no disponible")
    written = demo.save_backend_record(ok_rec, tmp_path)
    skipped = demo.save_backend_record(skip_rec, tmp_path)  # only_ok -> not written
    assert written is not None and written.exists()
    assert skipped is None
    loaded = demo.load_persisted_records(["qwen35", "qwen3.6-vl", "gemini-3.5-flash"], tmp_path)
    # Only the successful, persisted backend is reloaded.
    assert set(loaded) == {"qwen35"}
    assert loaded["qwen35"]["answer"] == "hola"


def test_save_backend_record_only_ok_false_writes_failures(tmp_path: Any) -> None:
    rec = demo._blank_record("qwen35", available=True, error="timeout")
    path = demo.save_backend_record(rec, tmp_path, only_ok=False)
    assert path is not None and path.exists()


def test_perceiver_table_shape() -> None:
    @dataclass
    class _Obs:
        parcel_id: int = 5
        crop_class: str = "maize"
        confidence: float = 0.8123
        vigor: str = "high"

    rows = demo.perceiver_table([(_Obs(), 12.3)], display=False)
    assert rows == [
        {
            "parcela": 5,
            "cultivo": "maize",
            "confianza": 0.812,
            "vigor": "high",
            "latencia_ms": 12.3,
        }
    ]


def test_rag_table_truncates_content() -> None:
    @dataclass
    class _Doc:
        id: int = 1
        source: str = "phenology_caption"
        parcel_id: str | None = "p-1"
        distance_m: float | None = 1234.56
        score: float = 0.98765
        content: str = "x" * 200

    rows = demo.rag_table([_Doc()], display=False)
    assert rows[0]["doc_id"] == 1
    assert rows[0]["distancia_m"] == 1234.6
    assert rows[0]["score"] == 0.9877
    assert rows[0]["contenido"].endswith("...") and len(rows[0]["contenido"]) == 93


def test_rag_table_empty() -> None:
    assert demo.rag_table([], display=False) == []
