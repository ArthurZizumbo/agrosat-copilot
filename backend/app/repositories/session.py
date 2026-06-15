"""Repository for ``chat_sessions``."""

from __future__ import annotations

import uuid

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.session import ChatSession
from backend.app.repositories.base import BaseRepository


class SessionRepository(BaseRepository[ChatSession]):
    """Session-scoped access to ``chat_sessions``."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(ChatSession, session)

    async def create(self, *, user_id: str, llm_variant: str) -> ChatSession:
        """Insert a new chat session and return it with its generated UUID."""
        obj = ChatSession(user_id=user_id, llm_variant=llm_variant)
        return await self.add_commit_refresh(obj)

    async def get_owned(self, *, session_id: uuid.UUID, user_id: str) -> ChatSession | None:
        """Return the session only if it belongs to ``user_id`` (else ``None``)."""
        obj = await self.get(session_id)
        if obj is None or obj.user_id != user_id:
            return None
        return obj

    async def get_latest_for_user(self, *, user_id: str) -> ChatSession | None:
        """Return the most recently created session for a user, or ``None``."""
        from sqlalchemy import select

        result = await self.session.execute(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def set_llm_variant(self, *, session_id: uuid.UUID, llm_variant: str) -> None:
        """Update the ``llm_variant`` of a session and bump ``updated_at``."""
        from sqlalchemy import text

        await self.session.execute(
            update(ChatSession)
            .where(ChatSession.id == session_id)
            .values(llm_variant=llm_variant, updated_at=text("now()"))
        )
        await self.commit()
