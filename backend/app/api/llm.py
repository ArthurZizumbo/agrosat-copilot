"""LLM switch endpoint: change the active variant of a session."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import _check_session_owner, get_current_user_id
from backend.app.api.schemas import LlmSwitchRequest, LlmSwitchResponse
from backend.app.core.db import get_session
from backend.app.services.session_service import SessionService

router = APIRouter(tags=["llm"])


@router.post("/llm/switch", response_model=LlmSwitchResponse)
async def switch_llm(
    body: LlmSwitchRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_session),
) -> LlmSwitchResponse:
    """Switch the ``llm_variant`` (``gemini`` | ``qwen35``) of a session."""
    await _check_session_owner(session_id=body.session_id, user_id=user_id, db=db)
    service = SessionService(db)
    await service.switch_llm_variant(session_id=body.session_id, llm_variant=body.llm_variant)
    return LlmSwitchResponse(session_id=body.session_id, llm_variant=body.llm_variant)
