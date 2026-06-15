"""Repository for ``chat_messages`` (session-scoped conversation memory)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.chat_message import ChatMessage
from backend.app.repositories.base import BaseRepository


class ChatMessageRepository(BaseRepository[ChatMessage]):
    """Session-scoped access to ``chat_messages``."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(ChatMessage, session)

    async def list_recent(self, *, session_id: uuid.UUID, limit: int = 20) -> Sequence[ChatMessage]:
        """Return the most recent turns of a session, oldest first.

        Fetches the newest ``limit`` rows by id, then reverses to chronological
        order so the agent reads history in the order it happened.
        """
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.id.desc())  # type: ignore[union-attr]
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        rows = list(result.scalars().all())
        rows.reverse()
        return rows

    async def append(self, *, session_id: uuid.UUID, role: str, content: str) -> ChatMessage:
        """Persist one conversation turn."""
        obj = ChatMessage(session_id=session_id, role=role, content=content)
        return await self.add_commit_refresh(obj)
