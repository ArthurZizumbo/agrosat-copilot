"""Shared FastAPI dependencies for the API routers (auth-guard).

US-052 hardens ``/chat`` with a per-session authorisation guard. The MVP guard
(decision D5, no productive Clerk yet) validates that the request carries a
well-formed ``X-Session-ID`` (handled upstream by
:func:`~backend.app.core.db.get_request_session_id`, which maps a missing or
malformed header to ``400``) AND that the session **exists and belongs to the
caller**.

Existence/ownership is checked under an RLS-scoped connection
(:func:`~backend.app.core.db.get_scoped_conn`, role ``agrosat_app``
NOBYPASSRLS): the US-051 policies already restrict ``chat_sessions`` to the
current session, so a ``SELECT 1`` that returns zero rows means the session is
either unknown or owned by another tenant -- both fail closed as ``403``. The
guard therefore needs no explicit ``user_id`` comparison; the database enforces
isolation.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

import asyncpg
import structlog
from fastapi import Depends, HTTPException, status

from backend.app.core.db import get_request_session_id, get_scoped_conn

__all__ = ["verify_chat_session"]

logger = structlog.get_logger(__name__)

#: Probe that the requested session exists and is visible to the caller. Under
#: the RLS-scoped connection this returns a row only for the caller's own
#: session (the ``chat_sessions`` policy filters by ``app.current_session``).
_SESSION_EXISTS_SQL = "SELECT 1 FROM chat_sessions WHERE id = $1"


async def verify_chat_session(
    session_id: Annotated[UUID, Depends(get_request_session_id)],
    conn: Annotated[asyncpg.Connection, Depends(get_scoped_conn)],
) -> UUID:
    """Authorise a ``/chat`` request against its tenant session.

    Runs ``SELECT 1 FROM chat_sessions WHERE id = $1`` on the RLS-scoped
    connection. Because the connection is primed with ``app.current_session =
    session_id`` (``SET LOCAL``), the US-051 policy only exposes the caller's own
    session: zero rows means the session is unknown or belongs to another tenant
    -- either way the request is rejected with ``403`` (fail-closed). A missing
    or malformed ``X-Session-ID`` already raised ``400`` in
    :func:`~backend.app.core.db.get_request_session_id`.

    Args:
        session_id: Validated tenant session id (``400`` on a bad header).
        conn: RLS-scoped connection bound to ``session_id``.

    Returns:
        The validated and authorised tenant session id.

    Raises:
        HTTPException: ``403 Forbidden`` when the session does not exist or is
            not owned by the caller.
    """
    row = await conn.fetchrow(_SESSION_EXISTS_SQL, session_id)
    if row is None:
        logger.warning("chat_session_forbidden", session_id=str(session_id))
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Session not found or not accessible.",
        )
    return session_id
