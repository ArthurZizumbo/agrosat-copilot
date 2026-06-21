"""``/chat`` router: SSE endpoint streaming the perceiver-reasoner response.

Thin HTTP adapter (router -> service -> model). It only resolves the effective
``session_id`` (``X-Session-ID`` header takes precedence over the request body)
and hands the validated request to :class:`~backend.app.services.chat_service.ChatService`,
returning its async SSE generator as a ``StreamingResponse``. All business logic
(perceiver wiring, event ordering) lives in the service.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from backend.app.api.deps import verify_chat_session
from backend.app.core.rate_limit import CHAT_RATE_LIMIT, limiter, session_id_key
from backend.app.services.chat_service import ChatRequest, ChatService

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["chat"])

#: SSE media type required so the browser ``EventSource`` parses the stream.
_SSE_MEDIA_TYPE = "text/event-stream"

#: Headers that keep an SSE stream open and un-buffered across proxies.
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # Disable proxy buffering (nginx) so events flush as they are produced.
    "X-Accel-Buffering": "no",
}


def _resolve_session_id(header_value: str | None, body_value: UUID | None) -> UUID:
    """Resolve the effective session id from header or body.

    The ``X-Session-ID`` header wins over the body so a session-scoped frontend
    client can drive multi-tenancy from a single place. Exactly one source must
    yield a valid UUID.

    Args:
        header_value: Raw ``X-Session-ID`` header value, if present.
        body_value: ``session_id`` from the request body, if present.

    Returns:
        The resolved tenant session id.

    Raises:
        HTTPException: 400 when no session id is supplied, or the header is not
            a valid UUID.
    """
    if header_value:
        try:
            return UUID(header_value)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="X-Session-ID header is not a valid UUID.",
            ) from exc
    if body_value is not None:
        return body_value
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="A session id is required via the X-Session-ID header or the body.",
    )


@router.post("/chat")
@limiter.limit(CHAT_RATE_LIMIT, key_func=session_id_key)
async def chat(
    request: Request,
    body: ChatRequest,
    _session: Annotated[UUID, Depends(verify_chat_session)],
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
) -> StreamingResponse:
    """Stream the chat response as Server-Sent Events (rate-limited, guarded).

    Two cross-cutting controls run before any business logic (router stays thin
    -- SoC):

    1. ``@limiter.limit`` enforces **10 requests / minute per session** (key =
       ``X-Session-ID``). The 11th request in the window returns ``429`` before
       the stream opens.
    2. ``Depends(verify_chat_session)`` authorises the session under RLS: a
       missing/malformed header is ``400`` and an unknown/foreign session is
       ``403``.

    Once both pass, it resolves the session id (header over body) and streams the
    service's SSE generator. The stream emits a real ``perceiver_observation``
    event before the reasoner's answer and a terminal ``done``/``error`` event.

    Args:
        request: The raw request (required by slowapi to evaluate the limit and
            by FastAPI to host the SSE response).
        body: Validated chat request (history + parcel/AOI subject). Named
            ``body`` because slowapi reserves ``request`` for the
            :class:`~fastapi.Request`.
        _session: Authorised tenant session injected by
            :func:`~backend.app.api.deps.verify_chat_session` (guard side effect;
            value unused here, the router re-resolves it from header/body).
        x_session_id: Optional ``X-Session-ID`` header (preferred source).

    Returns:
        A ``StreamingResponse`` of ``text/event-stream`` frames.
    """
    session_id = _resolve_session_id(x_session_id, body.session_id)
    logger.info("chat_request_received", session_id=str(session_id))

    service = ChatService()
    generator = service.stream(body.messages, session_id, request=body)
    return StreamingResponse(
        generator,
        media_type=_SSE_MEDIA_TYPE,
        headers=_SSE_HEADERS,
    )
