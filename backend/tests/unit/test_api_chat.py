"""API-level tests for the chat dispatch + SSE endpoints with fakes.

The DB session dependency and the service layer are overridden so the routers
are exercised without Postgres or the real agent.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from backend.app.core.db import get_session
from backend.app.main import create_app
from backend.app.services import chat_service as chat_service_module
from backend.app.services.job_registry import JobRegistry

SESSION_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")


class _FakeSession:
    """A stand-in ChatSession with the attributes the router reads."""

    def __init__(self) -> None:
        self.id = SESSION_ID
        self.user_id = "demo@agrosat.dev"
        self.llm_variant = "gemini"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    app = create_app()

    async def fake_get_session():  # type: ignore[no-untyped-def]
        yield None

    async def fake_owner(*, session_id, user_id, db):  # type: ignore[no-untyped-def]
        return _FakeSession()

    app.dependency_overrides[get_session] = fake_get_session
    # Patch the ownership guard used inside the router module.
    monkeypatch.setattr("backend.app.api.chat._check_session_owner", fake_owner)

    # Fresh registry + a chat service whose dispatch records the call.
    registry = JobRegistry()

    class _StubChatService:
        def dispatch(self, *, session_id, message, llm_variant, aoi_id=None):  # type: ignore[no-untyped-def]
            job_id = registry.create_job(session_id=session_id)
            registry.publish(job_id, {"type": "done", "job_id": job_id})
            registry.finish(job_id, status="done")
            return job_id

    monkeypatch.setattr("backend.app.api.chat.get_chat_service", lambda: _StubChatService())
    monkeypatch.setattr("backend.app.api.chat.get_job_registry", lambda: registry)
    monkeypatch.setattr(chat_service_module, "get_job_registry", lambda: registry)

    return TestClient(app)


def test_post_chat_returns_202_with_job_and_ws_url(client: TestClient) -> None:
    resp = client.post(
        "/chat",
        json={"session_id": str(SESSION_ID), "message": "hola"},
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["job_id"].startswith("job_")
    assert f"/ws/chat/{SESSION_ID}" in body["ws_url"]
    assert body["ws_url"].startswith("ws")


def test_post_chat_rejects_empty_message(client: TestClient) -> None:
    resp = client.post("/chat", json={"session_id": str(SESSION_ID), "message": ""})
    assert resp.status_code == 422


def test_sse_fallback_streams_events(client: TestClient) -> None:
    dispatch = client.post("/chat", json={"session_id": str(SESSION_ID), "message": "hi"})
    job_id = dispatch.json()["job_id"]
    with client.stream("GET", f"/chat/{job_id}/events") as stream:
        assert stream.status_code == 200
        body = b"".join(stream.iter_bytes())
    assert b'"type":"done"' in body


def test_sse_unknown_job_404(client: TestClient) -> None:
    resp = client.get("/chat/job_unknown/events")
    assert resp.status_code == 404
