"""Session endpoints: create a chat session."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import get_current_user_id
from backend.app.api.schemas import CreateSessionRequest, SessionResponse
from backend.app.core.config import get_settings
from backend.app.core.db import get_session
from backend.app.services.session_service import SessionService

router = APIRouter(tags=["sessions"])


@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    body: CreateSessionRequest | None = None,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_session),
) -> SessionResponse:
    """Create a chat session owned by the current (demo) user."""
    variant = (body.llm_variant if body else None) or get_settings().llm_variant_default
    service = SessionService(db)
    # Demo: reuse the user's existing session (seeded with parcels) when present,
    # so the UI chat is scoped to real data instead of an empty fresh session.
    created = await service.get_or_create_latest(user_id=user_id, llm_variant=variant)
    return SessionResponse(
        session_id=created.id,  # type: ignore[arg-type]
        user_id=created.user_id,
        llm_variant=created.llm_variant,  # type: ignore[arg-type]
    )
