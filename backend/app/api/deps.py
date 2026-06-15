"""Shared API dependencies: current user, ownership guard.

``_check_session_owner`` is the multi-tenant gate every session-scoped endpoint
must call (root rule 3 + ADR-011). Until Clerk auth lands (US-051) the current
user is the configurable demo user; the ownership check is real regardless.
"""

from __future__ import annotations

import uuid

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import Settings, get_settings
from backend.app.models.session import ChatSession
from backend.app.services.session_service import SessionService


def get_current_user_id(settings: Settings = Depends(get_settings)) -> str:
    """Return the acting user id (demo user until Clerk auth in US-051)."""
    user_id: str = settings.demo_user_id
    return user_id


async def _check_session_owner(
    *,
    session_id: uuid.UUID,
    user_id: str,
    db: AsyncSession,
) -> ChatSession:
    """Return the session iff owned by ``user_id``, else raise 403/404.

    A missing session yields 404; a session owned by another user yields 403.
    Every session-scoped endpoint calls this before touching session data.
    """
    service = SessionService(db)
    owned = await service.get_owned_or_none(session_id=session_id, user_id=user_id)
    if owned is None:
        # Distinguish absence from ownership to avoid leaking existence? For the
        # demo single-user setup we use 404 for both unknown and not-owned.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="session not found",
        )
    return owned
