"""US-054 AC-2 + AC-6: ``/chat`` uses the session's model; switch is rate-limited.

These cover the two US-054 acceptance criteria that do NOT need a real database
(the persistence / cross-session cases live in ``test_llm_switch.py`` over
testcontainers). They reuse the US-052 ``/chat`` harness: the reasoner agent and
the asyncpg pool are stubbed (no Gemini / vLLM / Postgres) and the slowapi
limiter is pointed at ``memory://`` and reset per test.

- **AC-2** after a ``/llm/switch`` the next ``/chat`` builds the backend of the
  PERSISTED variant: the ``ChatService`` reads ``chat_sessions.llm_model`` per
  request and hands that variant tag to the injectable ``agent_factory``. The
  test fakes the pool so the row read returns a chosen variant and asserts the
  factory received exactly it (so ``make_backend_for_variant`` would build the
  matching backend) -- proving the model is per-session, not the global default.

- **AC-6** the switch endpoint is rate-limited to **5 / minute per session**: the
  6th switch in the window is ``429``, and the budget is per-session (a second
  session still succeeds).

No real LLM call ever happens; validating the live switch to Qwen on ``:8002`` is
the QA manual step.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Annotated
from uuid import UUID

import pytest
from fastapi import Depends
from httpx import ASGITransport, AsyncClient

from backend.app.api.deps import verify_chat_session
from backend.app.core import db as core_db
from backend.app.core.config import Settings
from backend.app.core.rate_limit import build_limiter, limiter
from backend.app.main import create_app
from backend.app.services.chat_service import ChatMessage, ChatRequest, ChatService
from ml.agent.events import AgentEvent, DoneEvent, TextDeltaEvent

_VALID_SESSION = "11111111-1111-1111-1111-111111111111"
_OTHER_SESSION = "22222222-2222-2222-2222-222222222222"


# ---------------------------------------------------------------------------
# Memory limiter fixture (US-052 pattern): swap the singleton's storage in place.
# ---------------------------------------------------------------------------
@pytest.fixture
def memory_limiter() -> Iterator[None]:
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


# ---------------------------------------------------------------------------
# AC-2: ChatService resolves the per-session variant and feeds it to the factory.
# ---------------------------------------------------------------------------
class _FakeTransaction:
    """async-context-manager double for ``conn.transaction()`` (no-op)."""

    async def __aenter__(self) -> _FakeTransaction:
        return self

    async def __aexit__(self, *_exc) -> bool:
        return False


class _FakeConn:
    """asyncpg connection double: ``execute`` no-ops, ``fetchrow`` returns a row."""

    def __init__(self, llm_model: str | None) -> None:
        self._llm_model = llm_model

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction()

    async def execute(self, *_args) -> str:
        return "SET"

    async def fetchrow(self, *_args):
        if self._llm_model is None:
            return None
        return {"llm_model": self._llm_model}


class _FakePool:
    """asyncpg pool double handing out a single :class:`_FakeConn`."""

    def __init__(self, llm_model: str | None) -> None:
        self._conn = _FakeConn(llm_model)

    async def acquire(self):
        return self._conn

    async def release(self, _conn) -> None:
        return None


class _StubAgent:
    """Reasoner double with a static ``backend.model`` and a scripted stream."""

    class _Backend:
        model = "stub-model"

    def __init__(self) -> None:
        self.backend = self._Backend()

    async def stream_response(self, messages, session_id, ctx) -> AsyncIterator[AgentEvent]:
        yield TextDeltaEvent(text="ok")
        yield DoneEvent()


async def _drive_chat_capturing_variant(persisted: str | None) -> str:
    """Run ``ChatService.stream`` with a pool returning ``persisted`` and capture
    the variant handed to the ``agent_factory``."""
    seen: dict[str, str] = {}

    def _factory(variant: str, *, settings) -> _StubAgent:
        seen["variant"] = variant
        return _StubAgent()

    async def _pool_factory():
        return _FakePool(persisted)

    service = ChatService(
        settings=Settings(redis_url="memory://"),
        agent_factory=_factory,
        pool_factory=_pool_factory,
    )
    session_id = UUID(_VALID_SESSION)
    request = ChatRequest(messages=[ChatMessage(role="user", content="hola")])
    # Drain the stream so ``_stream_reasoner`` runs and calls the factory.
    async for _frame in service.stream(request.messages, session_id, request=request):
        pass
    return seen["variant"]


@pytest.mark.parametrize("persisted", ["gemini", "qwen-api", "qwen-onprem", "gemma"])
async def test_chat_uses_persisted_session_variant(persisted: str) -> None:
    """``/chat`` builds the backend of the variant persisted on the session (AC-2)."""
    variant = await _drive_chat_capturing_variant(persisted)
    assert variant == persisted


async def test_chat_falls_back_to_default_when_row_missing() -> None:
    """A session with no persisted variant falls back to the default (AC-2)."""
    variant = await _drive_chat_capturing_variant(None)
    assert variant == "gemini"


async def test_chat_falls_back_when_persisted_value_unknown() -> None:
    """An out-of-range persisted value degrades to the fallback variant (AC-2)."""
    variant = await _drive_chat_capturing_variant("mystery-model")
    assert variant == "gemini"


# ---------------------------------------------------------------------------
# AC-6: /llm/switch is rate-limited to 5/minute per session.
# ---------------------------------------------------------------------------
class _SwitchStubResult:
    def __init__(self, model: str) -> None:
        self.model = model
        from datetime import UTC, datetime

        self.applied_at = datetime.now(UTC)


def _make_switch_client(monkeypatch) -> tuple[AsyncClient, object]:
    """Build a client over the real app with the switch service + guard stubbed.

    The ``verify_chat_session`` guard is overridden to accept the test session
    (its own ``400``/``403`` logic is covered by the testcontainers suite) and
    ``LLMSwitchService.switch`` is patched to a no-op so the rate-limit assertions
    need neither Postgres nor a real ``UPDATE``.
    """
    import backend.app.api.llm as llm_mod

    async def _fake_switch(conn, session_id, model):  # type: ignore[no-untyped-def]
        return _SwitchStubResult(model)

    monkeypatch.setattr(llm_mod.LLMSwitchService, "switch", staticmethod(_fake_switch))

    app = create_app()

    async def _override_guard(
        session_id: Annotated[UUID, Depends(core_db.get_request_session_id)],
    ) -> UUID:
        return session_id

    async def _override_scoped(
        session_id: Annotated[UUID, Depends(core_db.get_request_session_id)],
    ) -> AsyncIterator[object]:
        yield object()

    app.dependency_overrides[verify_chat_session] = _override_guard
    app.dependency_overrides[core_db.get_scoped_conn] = _override_scoped
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test"), app


def _switch_body() -> dict:
    return {"model": "qwen-onprem"}


async def test_switch_rate_limit_429_on_sixth(monkeypatch, memory_limiter) -> None:
    """The 6th switch in the window for one session returns ``429`` (AC-6)."""
    client, _app = _make_switch_client(monkeypatch)
    headers = {"X-Session-ID": _VALID_SESSION}
    async with client:
        for _ in range(5):
            ok = await client.post("/llm/switch", json=_switch_body(), headers=headers)
            assert ok.status_code == 200, ok.text
        limited = await client.post("/llm/switch", json=_switch_body(), headers=headers)
    assert limited.status_code == 429


async def test_switch_rate_limit_isolated_per_session(monkeypatch, memory_limiter) -> None:
    """A second session is unaffected by the first's exhausted switch budget (AC-6)."""
    client, _app = _make_switch_client(monkeypatch)
    async with client:
        for _ in range(5):
            await client.post(
                "/llm/switch", json=_switch_body(), headers={"X-Session-ID": _VALID_SESSION}
            )
        exhausted = await client.post(
            "/llm/switch", json=_switch_body(), headers={"X-Session-ID": _VALID_SESSION}
        )
        other = await client.post(
            "/llm/switch", json=_switch_body(), headers={"X-Session-ID": _OTHER_SESSION}
        )
    assert exhausted.status_code == 429
    assert other.status_code == 200


async def test_switch_429_body_is_json_not_stream(monkeypatch, memory_limiter) -> None:
    """The over-budget switch returns a plain JSON ``429`` (AC-6)."""
    client, _app = _make_switch_client(monkeypatch)
    headers = {"X-Session-ID": _VALID_SESSION}
    async with client:
        for _ in range(5):
            assert (
                await client.post("/llm/switch", json=_switch_body(), headers=headers)
            ).status_code == 200
        limited = await client.post("/llm/switch", json=_switch_body(), headers=headers)
    assert limited.status_code == 429
    assert "event:" not in limited.text
