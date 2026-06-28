"""``/sessions`` router: chat-session lifecycle + transcript (US-080).

Thin HTTP adapter (router -> service -> DB). ``POST /sessions`` creates a session
WITHOUT an ``X-Session-ID`` (there is none yet) -- the id is minted by the
service. The other endpoints are session-scoped: they require the
``X-Session-ID`` header to match the path id and be owned by the caller
(:func:`~backend.app.api.deps.verify_session` + RLS).
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

import asyncpg
import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Response, status

from backend.app.api.deps import verify_session
from backend.app.core.db import get_scoped_conn
from backend.app.models.session import (
    ChatMessageOut,
    SessionCreate,
    SessionOut,
    SessionRename,
)
from backend.app.services.session_service import SessionService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions"])

#: Owner tag used until Clerk auth lands: an anonymous per-browser id passed in
#: the ``X-User-ID`` header (so a browser can list its own sessions later),
#: defaulting to a shared "anonymous" bucket when absent.
_DEFAULT_USER_ID = "anonymous"


@router.post("", status_code=status.HTTP_201_CREATED, response_model=SessionOut)
async def create_session(
    body: SessionCreate,
    x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
) -> SessionOut:
    """Create a chat session (one per chat tab) and return its id.

    No ``X-Session-ID`` is required: the session does not exist yet. The owning
    ``user_id`` comes from the optional ``X-User-ID`` header (anonymous browser
    id) so it is ready for per-user listing once auth lands.

    Args:
        body: Optional title + reasoner variant.
        x_user_id: Anonymous browser id (optional).

    Returns:
        The created :class:`SessionOut` (HTTP ``201``).
    """
    row = await SessionService.create(
        user_id=x_user_id or _DEFAULT_USER_ID,
        title=body.title,
        llm_model=body.llm_model,
    )
    return SessionOut(**row)


@router.get("", response_model=list[SessionOut])
async def list_sessions(
    x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
) -> list[SessionOut]:
    """List the caller's chat sessions (newest first) to restore the tabs.

    Scoped by the ``X-User-ID`` header (anonymous browser/user id) so the chat
    switcher can be rebuilt from the server, not only from localStorage. Without
    auth this is the source of truth for which chats exist for that id.

    Args:
        x_user_id: Anonymous browser/user id (optional; defaults to the shared
            bucket).

    Returns:
        The user's sessions as :class:`SessionOut` (may be empty).
    """
    rows = await SessionService.list_for_user(x_user_id or _DEFAULT_USER_ID)
    return [SessionOut(**row) for row in rows]


@router.get("/{session_id}/messages", response_model=list[ChatMessageOut])
async def list_messages(
    session_id: UUID,
    session: Annotated[UUID, Depends(verify_session)],
    conn: Annotated[asyncpg.Connection, Depends(get_scoped_conn)],
) -> list[ChatMessageOut]:
    """Return the session's transcript (RLS-scoped, ordered).

    The ``X-Session-ID`` header must match the path id (and be owned by the
    caller); otherwise the request is rejected.

    Args:
        session_id: Path session id.
        session: Authorised tenant session from the header (guard + identity).
        conn: RLS-scoped connection bound to the header session.

    Returns:
        The ordered list of :class:`ChatMessageOut`; empty for a new session.
    """
    _assert_path_matches_session(session_id, session)
    rows = await SessionService.list_messages(conn, session_id)
    return [ChatMessageOut(**row) for row in rows]


@router.patch("/{session_id}", response_model=SessionOut)
async def rename_session(
    session_id: UUID,
    body: SessionRename,
    session: Annotated[UUID, Depends(verify_session)],
    conn: Annotated[asyncpg.Connection, Depends(get_scoped_conn)],
) -> SessionOut:
    """Rename a chat session (session-scoped)."""
    _assert_path_matches_session(session_id, session)
    row = await SessionService.rename(conn, session_id, body.title)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    return SessionOut(**row)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: UUID,
    session: Annotated[UUID, Depends(verify_session)],
    conn: Annotated[asyncpg.Connection, Depends(get_scoped_conn)],
) -> Response:
    """Delete a chat session and its transcript/AOIs (CASCADE)."""
    _assert_path_matches_session(session_id, session)
    deleted = await SessionService.delete(conn, session_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _assert_path_matches_session(path_id: UUID, header_session: UUID) -> None:
    """Reject a request whose path id differs from the authorised header session.

    The RLS-scoped connection is bound to the header session, so a mismatching
    path id would silently read/write nothing; failing fast with ``403`` is
    clearer than a confusing empty result.
    """
    if path_id != header_session:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Path session id does not match the X-Session-ID header.",
        )
