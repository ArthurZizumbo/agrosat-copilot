"""``/llm/switch`` router: hot-swap the session's reasoner variant (US-054).

Thin HTTP adapter (router -> service -> DB). Two cross-cutting controls run
before any business logic (router stays thin -- SoC):

1. ``Depends(verify_chat_session)`` authorises the session under RLS: a
   missing/malformed ``X-Session-ID`` is ``400`` and an unknown/foreign session
   is ``403`` (fail-closed).
2. ``@limiter.limit`` enforces **5 requests / minute per session** (key =
   ``X-Session-ID``, US-054 AC-6). The 6th request in the window returns ``429``.

The persisted variant (one of ``gemini`` / ``qwen-api`` / ``qwen-onprem`` /
``gemma``) drives which backend the subsequent ``/chat`` requests of this session
build (US-054 AC-2). No SQL lives here; the ``UPDATE`` runs in
:class:`~backend.app.services.llm_switch_service.LLMSwitchService` on the
RLS-scoped connection.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

import asyncpg
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict

from backend.app.api.deps import verify_chat_session
from backend.app.core.db import get_scoped_conn
from backend.app.core.rate_limit import LLM_SWITCH_RATE_LIMIT, limiter, session_id_key
from backend.app.services.llm_switch_service import LLMSwitchService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/llm", tags=["llm"])

#: The four supported variant tags, exposed as a ``Literal`` so FastAPI rejects
#: any other value with ``422`` before the handler runs (1:1 with the DB CHECK
#: constraint and ``ml.agent.llm_routing.VARIANTS``).
LLMVariant = Literal["gemini", "qwen-api", "qwen-onprem", "gemma"]


class LLMSwitchRequest(BaseModel):
    """Body of a ``POST /llm/switch`` request.

    Attributes:
        model: The reasoner variant to activate for this session.
    """

    model_config = ConfigDict(extra="forbid")

    model: LLMVariant


class LLMSwitchResponse(BaseModel):
    """Response of a successful switch.

    Attributes:
        model: The variant now persisted on the session.
        applied_at: When the switch was persisted (the row's ``updated_at``).
    """

    model_config = ConfigDict(extra="forbid")

    model: LLMVariant
    applied_at: datetime


@router.post("/switch", response_model=LLMSwitchResponse)
@limiter.limit(LLM_SWITCH_RATE_LIMIT, key_func=session_id_key)
async def switch_llm(
    request: Request,
    response: Response,
    body: LLMSwitchRequest,
    session_id: Annotated[UUID, Depends(verify_chat_session)],
    conn: Annotated[asyncpg.Connection, Depends(get_scoped_conn)],
) -> LLMSwitchResponse:
    """Persist the active reasoner variant for the calling session.

    Authorised and rate-limited (see module docstring). Delegates the scoped
    ``UPDATE`` to :class:`~backend.app.services.llm_switch_service.LLMSwitchService`
    and echoes the applied variant. The 6th call within a minute returns ``429``.

    Args:
        request: The raw request (required by slowapi to evaluate the limit).
        response: The response object slowapi injects the ``X-RateLimit-*``
            headers into; FastAPI merges it into the serialised body response.
            Required because the limiter runs with ``headers_enabled=True``, so it
            needs a real :class:`~starlette.responses.Response` to write the
            rate-limit headers onto (a pydantic return value alone would raise).
        body: Validated switch request (one of the four variants). Named ``body``
            because slowapi reserves ``request`` for the :class:`~fastapi.Request`.
        session_id: Authorised tenant session injected by
            :func:`~backend.app.api.deps.verify_chat_session`.
        conn: RLS-scoped connection bound to the session.

    Returns:
        The :class:`LLMSwitchResponse` with the applied variant and timestamp.

    Raises:
        HTTPException: ``400`` when the variant is rejected by the service (the
            ``Literal`` body already blocks unknown values with ``422``, so this
            is a defensive mapping of the service's ``ValueError``).
    """
    logger.info("llm_switch_received", session_id=str(session_id), model=body.model)
    try:
        result = await LLMSwitchService.switch(conn, session_id, body.model)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    return LLMSwitchResponse(model=result.model, applied_at=result.applied_at)  # type: ignore[arg-type]
