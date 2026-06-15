"""Repository unit tests with a fake AsyncSession (no Postgres).

These cover the session-scoped read logic that does not depend on a live
database: ownership checks and the chat-history chronological ordering.
"""

from __future__ import annotations

import uuid

import pytest

from backend.app.models.chat_message import ChatMessage
from backend.app.models.session import ChatSession
from backend.app.repositories.chat_message import ChatMessageRepository
from backend.app.repositories.session import SessionRepository


class _ScalarResult:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    def scalars(self):  # type: ignore[no-untyped-def]
        return self

    def all(self):  # type: ignore[no-untyped-def]
        return list(self._rows)


class _FakeSession:
    """Minimal AsyncSession stand-in for read-only repository paths."""

    def __init__(self, *, get_obj=None, rows=None) -> None:  # type: ignore[no-untyped-def]
        self._get_obj = get_obj
        self._rows = rows or []

    async def get(self, model, obj_id):  # type: ignore[no-untyped-def]
        return self._get_obj

    async def execute(self, stmt):  # type: ignore[no-untyped-def]
        return _ScalarResult(self._rows)


@pytest.mark.asyncio
async def test_get_owned_returns_none_for_wrong_user() -> None:
    sid = uuid.uuid4()
    session_row = ChatSession(user_id="owner@x.dev", llm_variant="gemini")
    repo = SessionRepository(_FakeSession(get_obj=session_row))  # type: ignore[arg-type]

    assert await repo.get_owned(session_id=sid, user_id="other@x.dev") is None
    assert await repo.get_owned(session_id=sid, user_id="owner@x.dev") is session_row


@pytest.mark.asyncio
async def test_get_owned_missing_session() -> None:
    repo = SessionRepository(_FakeSession(get_obj=None))  # type: ignore[arg-type]
    assert await repo.get_owned(session_id=uuid.uuid4(), user_id="a@x.dev") is None


@pytest.mark.asyncio
async def test_list_recent_reverses_to_chronological() -> None:
    sid = uuid.uuid4()
    # DB returns newest-first (id desc); repo must reverse to oldest-first.
    newest_first = [
        ChatMessage(id=3, session_id=sid, role="assistant", content="c"),
        ChatMessage(id=2, session_id=sid, role="user", content="b"),
        ChatMessage(id=1, session_id=sid, role="assistant", content="a"),
    ]
    repo = ChatMessageRepository(_FakeSession(rows=newest_first))  # type: ignore[arg-type]
    history = await repo.list_recent(session_id=sid, limit=10)
    assert [m.content for m in history] == ["a", "b", "c"]
