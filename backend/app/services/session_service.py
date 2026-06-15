"""Session service: create sessions, switch LLM variant, ownership checks."""

from __future__ import annotations

import uuid
from typing import Literal

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.session import ChatSession
from backend.app.repositories.session import SessionRepository

logger = structlog.get_logger(__name__)

LlmVariant = Literal["gemini", "qwen35"]


class SessionService:
    """Business logic around ``chat_sessions``."""

    def __init__(self, session: AsyncSession) -> None:
        self._repo = SessionRepository(session)

    async def create(self, *, user_id: str, llm_variant: LlmVariant) -> ChatSession:
        """Create a new chat session for a user."""
        obj = await self._repo.create(user_id=user_id, llm_variant=llm_variant)
        logger.info("session_created", session_id=str(obj.id), user_id=user_id)
        return obj

    async def get_or_create_latest(self, *, user_id: str, llm_variant: LlmVariant) -> ChatSession:
        """Return the user's latest session, or create one if none exists.

        Demo behaviour (single-tenant, no real auth): the frontend lands on the
        existing seeded session so the chat is scoped to pre-loaded parcels
        instead of a fresh empty session.
        """
        existing = await self._repo.get_latest_for_user(user_id=user_id)
        if existing is not None:
            return existing
        return await self.create(user_id=user_id, llm_variant=llm_variant)

    async def get_owned_or_none(self, *, session_id: uuid.UUID, user_id: str) -> ChatSession | None:
        """Return the session iff it belongs to ``user_id``, else ``None``."""
        return await self._repo.get_owned(session_id=session_id, user_id=user_id)

    async def switch_llm_variant(self, *, session_id: uuid.UUID, llm_variant: LlmVariant) -> None:
        """Persist a new ``llm_variant`` for the session."""
        await self._repo.set_llm_variant(session_id=session_id, llm_variant=llm_variant)
        logger.info(
            "llm_variant_switched",
            session_id=str(session_id),
            llm_variant=llm_variant,
        )
