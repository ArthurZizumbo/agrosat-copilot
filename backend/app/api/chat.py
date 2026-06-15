"""Chat transport endpoints: dispatch, WebSocket stream, SSE fallback.

ADR-011 §3: ``POST /chat`` dispatches a background job and returns ``202``; the
client then streams ``AgentEvent`` JSON over ``WS /ws/chat/{session_id}`` (or the
``GET /chat/{job_id}/events`` SSE fallback). The backend only forwards events.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.websockets import WebSocket, WebSocketDisconnect

from backend.app.api.deps import _check_session_owner, get_current_user_id
from backend.app.api.schemas import ChatDispatchResponse, ChatRequest
from backend.app.core.config import get_settings
from backend.app.core.db import get_session, get_session_factory
from backend.app.services.chat_service import get_chat_service, get_job_registry
from backend.app.services.session_service import SessionService

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["chat"])


def _ws_url(session_id: uuid.UUID, job_id: str) -> str:
    """Build the WebSocket URL the client should connect to."""
    base = (
        get_settings().nuxt_public_api_url.replace("http://", "ws://").replace("https://", "wss://")
    )
    return f"{base}/ws/chat/{session_id}?job_id={job_id}"


@router.post("/chat", response_model=ChatDispatchResponse, status_code=status.HTTP_202_ACCEPTED)
async def post_chat(
    body: ChatRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_session),
) -> ChatDispatchResponse:
    """Dispatch a chat turn to the agent; return ``202 {job_id, ws_url}``."""
    session = await _check_session_owner(session_id=body.session_id, user_id=user_id, db=db)
    # Resolve the active variant: explicit override > session default.
    variant = body.llm_variant or session.llm_variant  # type: ignore[assignment]

    service = get_chat_service()
    job_id = service.dispatch(
        session_id=str(body.session_id),
        message=body.message,
        llm_variant=variant,  # type: ignore[arg-type]
        aoi_id=body.aoi_id,
    )
    return ChatDispatchResponse(job_id=job_id, ws_url=_ws_url(body.session_id, job_id))


async def _resolve_job_id(session_id: str, job_id: str | None) -> str | None:
    """Return the explicit job_id or the latest job dispatched for the session."""
    registry = get_job_registry()
    if job_id is not None:
        return job_id
    latest: str | None = registry.latest_job_for_session(session_id)
    return latest


@router.websocket("/ws/chat/{session_id}")
async def ws_chat(
    websocket: WebSocket,
    session_id: uuid.UUID,
    job_id: str | None = Query(default=None),
) -> None:
    """Stream ``AgentEvent`` JSON for a session's job over WebSocket."""
    user_id = get_settings().demo_user_id
    factory = get_session_factory()
    async with factory() as db:
        service = SessionService(db)
        owned = await service.get_owned_or_none(session_id=session_id, user_id=user_id)
    if owned is None:
        await websocket.close(code=4404)
        return

    resolved = await _resolve_job_id(str(session_id), job_id)
    if resolved is None:
        await websocket.close(code=4404)
        return

    await websocket.accept()
    registry = get_job_registry()
    log = logger.bind(session_id=str(session_id), job_id=resolved)
    log.info("ws_connected")
    try:
        async for event in registry.subscribe(resolved):
            await websocket.send_json(event)
            if event.get("type") == "done":
                break
    except WebSocketDisconnect:
        log.info("ws_disconnected")
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass


@router.get("/chat/{job_id}/events")
async def chat_events_sse(job_id: str) -> StreamingResponse:
    """SSE fallback streaming the same ``AgentEvent`` JSON as the WebSocket."""
    registry = get_job_registry()
    if registry.status(job_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")

    async def _event_stream() -> AsyncIterator[bytes]:
        async for event in registry.subscribe(job_id):
            payload = json.dumps(event, separators=(",", ":"))
            yield f"data: {payload}\n\n".encode()
            if event.get("type") == "done":
                break

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
