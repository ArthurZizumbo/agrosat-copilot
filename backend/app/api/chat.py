"""``/chat`` router: SSE endpoint streaming the perceiver-reasoner response.

Thin HTTP adapter (router -> service -> model). It only resolves the effective
``session_id`` (``X-Session-ID`` header takes precedence over the request body)
and hands the validated request to :class:`~backend.app.services.chat_service.ChatService`,
returning its async SSE generator as a ``StreamingResponse``. All business logic
(perceiver wiring, event ordering) lives in the service.
"""

from __future__ import annotations

from uuid import UUID

import structlog
from fastapi import APIRouter, Header, HTTPException, status
from fastapi.responses import StreamingResponse

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
async def chat(
    request: ChatRequest,
    x_session_id: str | None = Header(default=None, alias="X-Session-ID"),
) -> StreamingResponse:
    """Stream the chat response as Server-Sent Events.

    Resolves the session id (header over body), then streams the service's SSE
    generator. The stream emits a real ``perceiver_observation`` event before
    the (US-047-pending) final answer and a terminal ``done`` event.

    Args:
        request: Validated chat request (history + parcel/AOI subject).
        x_session_id: Optional ``X-Session-ID`` header (preferred source).

    Returns:
        A ``StreamingResponse`` of ``text/event-stream`` frames.
    """
    session_id = _resolve_session_id(x_session_id, request.session_id)
    logger.info("chat_request_received", session_id=str(session_id))

    service = ChatService()
    generator = service.stream(request.messages, session_id, request=request)
    return StreamingResponse(
        generator,
        media_type=_SSE_MEDIA_TYPE,
        headers=_SSE_HEADERS,
    )
