"""``POST /chat`` endpoint integration tests (US-052 hardening).

Drives the real FastAPI app through ``httpx.AsyncClient`` + ``ASGITransport``
(no live server, no network) and asserts the three US-052 controls layered on
the already-working SSE stream:

- **AC-1** SSE round-trip: a guarded request streams ``text/event-stream`` frames
  ending in ``event: done``.
- **AC-4** rate limit: the 11th request in the window (same session) is ``429``,
  and the budget is **per session** -- a second session is unaffected.
- **AC-5** auth-guard: a missing/malformed ``X-Session-ID`` is ``400`` and an
  unknown/foreign session is ``403``.
- **AC-6** RLS wiring: ``ChatService`` builds its ``ToolContext`` from the
  ``app_database_url`` pool (role ``agrosat_app`` NOBYPASSRLS), so the tools run
  under the US-051 policies. Full cross-session isolation lives in the US-051
  suite (testcontainers); here we assert the wiring (the pool factory) that makes
  it hold -- the documented degradation from the plan when testcontainers is not
  part of this harness.

Boundaries mocked: the perceiver, the asyncpg pool and the reasoner agent (a
stubbed ``agent_factory``) so no DB / Gemini / vLLM call happens. The limiter is
switched to ``memory://`` storage and reset between tests for determinism without
Redis.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Annotated
from uuid import UUID, uuid4

import pytest
from fastapi import Depends
from httpx import ASGITransport, AsyncClient

import backend.app.services.chat_service as chat_mod
from backend.app.api.deps import verify_chat_session
from backend.app.core import db as core_db
from backend.app.core.config import Settings
from backend.app.core.rate_limit import build_limiter, limiter
from backend.app.main import create_app
from backend.app.services.chat_service import ChatService
from ml.agent.events import (
    AgentEvent,
    DoneEvent,
    ErrorEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from ml.agent.perceiver import PerceiverObservation

_VALID_SESSION = "11111111-1111-1111-1111-111111111111"
_OTHER_SESSION = "22222222-2222-2222-2222-222222222222"

_OBSERVATION = PerceiverObservation(
    parcel_id=11,
    crop_class="wheat",
    confidence=0.9,
    phenology_text="Fenologia: pico NDVI 0.8.",
    vigor="high",
    class_probabilities={"wheat": 0.9, "maize": 0.1},
    description="Parcela de trigo.",
)


class _FakePerceiver:
    """``PerceiverLayer`` double returning a fixed observation (no DB / LLM)."""

    def __init__(self, ctx) -> None:
        self._ctx = ctx

    async def observe(self, parcel_id: int) -> PerceiverObservation:
        return _OBSERVATION

    async def observe_aoi(self, aoi, year: int) -> PerceiverObservation:
        return _OBSERVATION


class _StubAgent:
    """Reasoner double yielding a scripted event stream (no network)."""

    def __init__(self, events: list[AgentEvent] | None = None) -> None:
        self._events = events or [TextDeltaEvent(text="Hola."), DoneEvent()]

    async def stream_response(self, messages, session_id, ctx) -> AsyncIterator[AgentEvent]:
        for event in self._events:
            yield event


@pytest.fixture
def memory_limiter() -> Iterator[None]:
    """Point the production limiter singleton at ``memory://`` and reset it.

    The ``@limiter.limit`` decorator binds the module-level singleton, so the
    test swaps that singleton's storage in place (rather than a parallel
    instance) and resets the counters before and after each test for
    determinism without a live Redis.
    """
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


def _make_app_client(monkeypatch, agent_events: list[AgentEvent] | None = None):
    """Build an ``AsyncClient`` over the real app with the boundaries mocked.

    Mocks the perceiver, the asyncpg pool and the reasoner so the real
    ``ChatService`` runs without DB / network, and overrides the auth-guard to
    accept the test sessions (the guard itself is exercised separately with a
    fake DB connection). ``agent_events`` scripts the reasoner's event stream so a
    test can drive a plain answer (default), a tool round-trip or an error frame.
    """

    async def _fake_get_pool():
        return object()

    monkeypatch.setattr(chat_mod, "get_pool", _fake_get_pool)
    monkeypatch.setattr(chat_mod, "PerceiverLayer", _FakePerceiver)
    monkeypatch.setattr(
        chat_mod,
        "_default_agent_factory",
        lambda model, *, settings: _StubAgent(events=agent_events),
    )

    app = create_app()

    # Bypass the DB existence check so the stream/limit assertions do not need a
    # live Postgres; the guard's own 400/403 logic is exercised separately.
    async def _override_guard(
        session_id: Annotated[UUID, Depends(core_db.get_request_session_id)],
    ) -> UUID:
        return session_id

    app.dependency_overrides[verify_chat_session] = _override_guard

    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test"), app


@pytest.fixture
def app_client(monkeypatch, memory_limiter):
    """Default app client: the stub agent replays ``text_delta`` + ``done``."""
    return _make_app_client(monkeypatch)


def _body() -> dict:
    return {
        "messages": [{"role": "user", "content": "describe la parcela"}],
        "parcel_id": 11,
    }


def _event_names(sse_body: str) -> list[str]:
    """Extract the ordered ``event:`` names from a raw SSE response body."""
    return [
        line.removeprefix("event:").strip()
        for line in sse_body.splitlines()
        if line.startswith("event:")
    ]


async def test_round_trip_emits_done(app_client) -> None:
    """A guarded request streams SSE frames ending in ``event: done`` (AC-1)."""
    client, _app = app_client
    async with client:
        resp = await client.post("/chat", json=_body(), headers={"X-Session-ID": _VALID_SESSION})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.text
    assert "event: perceiver_observation" in body
    assert body.rstrip().endswith("data: {}") or "event: done" in body
    assert "event: done" in body


async def test_rate_limit_429_on_11th(app_client) -> None:
    """The 11th request in the window for one session returns ``429`` (AC-4)."""
    client, _app = app_client
    headers = {"X-Session-ID": _VALID_SESSION}
    async with client:
        for _ in range(10):
            ok = await client.post("/chat", json=_body(), headers=headers)
            assert ok.status_code == 200
        limited = await client.post("/chat", json=_body(), headers=headers)
    assert limited.status_code == 429


async def test_rate_limit_isolated_per_session(app_client) -> None:
    """A second session is unaffected by the first's exhausted budget (AC-4)."""
    client, _app = app_client
    async with client:
        for _ in range(10):
            await client.post("/chat", json=_body(), headers={"X-Session-ID": _VALID_SESSION})
        exhausted = await client.post(
            "/chat", json=_body(), headers={"X-Session-ID": _VALID_SESSION}
        )
        other = await client.post("/chat", json=_body(), headers={"X-Session-ID": _OTHER_SESSION})
    assert exhausted.status_code == 429
    assert other.status_code == 200


async def test_guard_400_missing_session(monkeypatch, memory_limiter) -> None:
    """A request without ``X-Session-ID`` is rejected with ``400`` (AC-5)."""
    app = create_app()  # real guard (no override): the header dependency runs
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/chat", json=_body())
    assert resp.status_code == 400


async def test_guard_400_malformed_session(monkeypatch, memory_limiter) -> None:
    """A malformed ``X-Session-ID`` is rejected with ``400`` (AC-5)."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/chat", json=_body(), headers={"X-Session-ID": "not-a-uuid"})
    assert resp.status_code == 400


async def test_guard_403_unknown_session(monkeypatch, memory_limiter) -> None:
    """An unknown/foreign session (0 rows under RLS) is rejected with ``403`` (AC-5).

    The auth-guard runs against a fake scoped connection whose ``fetchrow``
    returns ``None`` -- exactly what the RLS policy yields for a session the
    caller does not own. The guard must translate that into a ``403``.
    """

    class _EmptyConn:
        async def fetchrow(self, *_args):
            return None

    async def _empty_scoped_conn(
        session_id: Annotated[UUID, Depends(core_db.get_request_session_id)],
    ) -> AsyncIterator[_EmptyConn]:
        yield _EmptyConn()

    app = create_app()
    # Override only the scoped-conn dependency so the real ``verify_chat_session``
    # logic (the SELECT 1 -> 403 mapping) is exercised end to end.
    app.dependency_overrides[core_db.get_scoped_conn] = _empty_scoped_conn
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/chat", json=_body(), headers={"X-Session-ID": _VALID_SESSION})
    assert resp.status_code == 403


def test_chat_service_uses_app_database_pool() -> None:
    """``ChatService`` defaults its tool pool to the ``app_database_url`` pool (AC-6).

    This is the B2 wiring that makes the US-051 RLS apply to the ``/chat`` tools:
    the ``ToolContext`` pool must come from ``backend.app.core.db.get_pool``
    (role ``agrosat_app`` NOBYPASSRLS), not the superuser ``ml.agent.db`` pool.
    Full cross-session isolation is covered by the US-051 testcontainers suite.
    """
    service = ChatService(settings=Settings(redis_url="memory://"))
    assert service._pool_factory is core_db.get_pool


async def test_rls_pool_factory_is_invoked_for_context() -> None:
    """The injected pool factory feeds the ``ToolContext`` (RLS pool plumbing, AC-6)."""
    sentinel_pool = object()
    calls: list[int] = []

    async def _pool_factory():
        calls.append(1)
        return sentinel_pool

    service = ChatService(settings=Settings(redis_url="memory://"), pool_factory=_pool_factory)
    ctx = await service._build_context(uuid4())
    assert ctx.pool is sentinel_pool
    assert calls == [1]


async def test_round_trip_function_calling_order(monkeypatch, memory_limiter) -> None:
    """A tool round-trip streams perceiver + tool_call/tool_result/text/done (AC-1).

    Drives the full SSE contract through ``httpx`` with the reasoner scripted to
    invoke a tool: the HTTP response must carry the ``perceiver_observation`` and
    then forward the agent's ``tool_call`` -> ``tool_result`` -> ``text_delta`` ->
    ``done`` frames *in order*, proving function-calling events survive the
    StreamingResponse round-trip (not just a plain answer).
    """
    agent_events: list[AgentEvent] = [
        ToolCallEvent(name="list_parcels", arguments={"session_id": _VALID_SESSION}),
        ToolResultEvent(name="list_parcels", result={"count": 3}, ok=True),
        TextDeltaEvent(text="Tienes 3 parcelas."),
        DoneEvent(),
    ]
    client, _app = _make_app_client(monkeypatch, agent_events=agent_events)
    async with client:
        resp = await client.post("/chat", json=_body(), headers={"X-Session-ID": _VALID_SESSION})
        assert resp.status_code == 200
        body = resp.text

    assert _event_names(body) == [
        "perceiver_observation",
        "tool_call",
        "tool_result",
        "text_delta",
        "done",
    ]
    # The tool_result payload survives JSON serialisation through the stream.
    assert '"count":3' in body or '"count": 3' in body


async def test_agent_error_frame_forwarded_and_stream_closes(monkeypatch, memory_limiter) -> None:
    """An agent ``ErrorEvent`` is the terminal SSE frame; the stream still closes (error path).

    The reasoner turns its backend failures into a terminal ``ErrorEvent`` (it
    never raises). Over the HTTP round-trip the response must end with
    ``event: error`` (no ``done``) and the connection close cleanly with a 200 --
    the failure is in-band SSE, not a transport error.
    """
    agent_events: list[AgentEvent] = [ErrorEvent(message="backend 503")]
    client, _app = _make_app_client(monkeypatch, agent_events=agent_events)
    async with client:
        resp = await client.post("/chat", json=_body(), headers={"X-Session-ID": _VALID_SESSION})
        assert resp.status_code == 200
        body = resp.text

    names = _event_names(body)
    assert names == ["perceiver_observation", "error"]
    assert "event: done" not in body
    assert '"message":"backend 503"' in body or '"message": "backend 503"' in body


async def test_rate_limit_429_body_is_json_not_stream(app_client) -> None:
    """The 11th request returns a JSON ``429`` (not a half-open SSE stream) (AC-4).

    slowapi evaluates the limit before the StreamingResponse opens, so the over-
    budget response is the handler's plain JSON error -- not ``text/event-stream``
    -- which the frontend can surface without parsing SSE frames.
    """
    client, _app = app_client
    headers = {"X-Session-ID": _VALID_SESSION}
    async with client:
        for _ in range(10):
            assert (await client.post("/chat", json=_body(), headers=headers)).status_code == 200
        limited = await client.post("/chat", json=_body(), headers=headers)

    assert limited.status_code == 429
    assert not limited.headers["content-type"].startswith("text/event-stream")
    # The body is the slowapi rate-limit error, not an SSE frame.
    assert "event:" not in limited.text


async def test_session_id_threaded_from_header_to_service(monkeypatch, memory_limiter) -> None:
    """The ``X-Session-ID`` header drives the session id the agent is run with (AC-5).

    Captures the ``session_id`` the reasoner ``stream_response`` receives and
    asserts it equals the header UUID -- proving the router resolves the header
    (not the body) and threads it unchanged into the tenant-scoped reasoner.
    """
    seen: dict[str, UUID] = {}

    class _RecordingAgent(_StubAgent):
        async def stream_response(self, messages, session_id, ctx):
            seen["session_id"] = session_id
            for event in self._events:
                yield event

    async def _fake_get_pool():
        return object()

    monkeypatch.setattr(chat_mod, "get_pool", _fake_get_pool)
    monkeypatch.setattr(chat_mod, "PerceiverLayer", _FakePerceiver)
    monkeypatch.setattr(
        chat_mod, "_default_agent_factory", lambda model, *, settings: _RecordingAgent()
    )
    app = create_app()

    async def _override_guard(
        session_id: Annotated[UUID, Depends(core_db.get_request_session_id)],
    ) -> UUID:
        return session_id

    app.dependency_overrides[verify_chat_session] = _override_guard
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/chat", json=_body(), headers={"X-Session-ID": _VALID_SESSION})
        assert resp.status_code == 200
        _ = resp.text  # drain the stream so the generator runs to completion

    assert seen["session_id"] == UUID(_VALID_SESSION)
