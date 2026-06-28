"""Chat-session lifecycle + transcript persistence (US-080).

Business logic for the in-app multi-chat UI: create a session (one per chat
tab), rename/delete it, and persist/restore its transcript. Multi-tenant
isolation is the US-051 RLS contract -- every query runs under a connection
scoped to ``app.current_session``.

Creating a session generates the UUID in the application (``uuid4()``) rather
than relying on the ``gen_random_uuid()`` default, so the INSERT can run under a
connection already scoped to that id and satisfy the RLS ``WITH CHECK
(id = app.current_session)`` policy (otherwise a fresh row would be invisible to
its own creating scope).
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID, uuid4

import asyncpg
import structlog

from backend.app.core.db import get_pool, session_scoped_conn

__all__ = ["SessionService"]

logger = structlog.get_logger(__name__)


class SessionService:
    """Create/rename/delete sessions and persist their chat transcript."""

    @staticmethod
    async def create(user_id: str, title: str | None, llm_model: str) -> dict[str, Any]:
        """Create a new chat session owned by ``user_id``.

        The id is minted here so the INSERT runs under a scope bound to it
        (RLS ``WITH CHECK``). Returns the created session row.

        Args:
            user_id: Owner tag (anonymous browser id until Clerk auth lands).
            title: Optional human-facing chat name (NULL -> UI default).
            llm_model: Reasoner variant (CHECK-constrained).

        Returns:
            The created session as a dict (``id, title, llm_model, created_at``).
        """
        session_id = uuid4()
        async with session_scoped_conn(session_id) as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO chat_sessions (id, user_id, llm_model, title)
                VALUES ($1, $2, $3, $4)
                RETURNING id, title, llm_model, created_at
                """,
                session_id,
                user_id,
                llm_model,
                title,
            )
        logger.info("session_created", session_id=str(session_id), user_id=user_id)
        return dict(row)

    @staticmethod
    async def list_for_user(user_id: str) -> list[dict[str, Any]]:
        """List a user's sessions (newest first) via the SECURITY DEFINER fn.

        Per-session RLS hides "all sessions of a user" from the app role, so the
        listing goes through ``list_chat_sessions(user_id)`` (a controlled,
        parameterized bypass that returns only this user's rows). No session
        scope is needed, so a plain pool connection is used.

        Args:
            user_id: Owner tag (anonymous browser/user id).

        Returns:
            Session rows ``id, title, llm_model, created_at`` (newest first).
        """
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, title, llm_model, created_at FROM list_chat_sessions($1)",
                user_id,
            )
        return [dict(row) for row in rows]

    @staticmethod
    async def rename(
        conn: asyncpg.Connection, session_id: UUID, title: str
    ) -> dict[str, Any] | None:
        """Rename a session (session-scoped). Returns the row, or None if absent."""
        row = await conn.fetchrow(
            """
            UPDATE chat_sessions SET title = $1, updated_at = now()
            WHERE id = $2
            RETURNING id, title, llm_model, created_at
            """,
            title,
            session_id,
        )
        return dict(row) if row is not None else None

    @staticmethod
    async def delete(conn: asyncpg.Connection, session_id: UUID) -> bool:
        """Delete a session (CASCADE removes its messages/AOIs). True if deleted."""
        result: str = await conn.execute("DELETE FROM chat_sessions WHERE id = $1", session_id)
        return result.rsplit(" ", 1)[-1] == "1"

    @staticmethod
    async def list_messages(conn: asyncpg.Connection, session_id: UUID) -> list[dict[str, Any]]:
        """Return the session's transcript in insertion order (RLS-scoped)."""
        rows = await conn.fetch(
            """
            SELECT id, role, content, extra, created_at
            FROM chat_messages
            WHERE session_id = $1
            ORDER BY created_at, id
            """,
            session_id,
        )
        messages: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            raw = item.get("extra")
            # asyncpg returns JSONB as a JSON string unless a codec is registered;
            # decode it so the API emits a real object.
            item["extra"] = json.loads(raw) if isinstance(raw, str) else raw
            messages.append(item)
        return messages

    @staticmethod
    async def save_message(
        session_id: UUID,
        role: str,
        content: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Persist one chat turn under its own session-scoped connection.

        Used by the chat stream to store the user turn and the assistant's final
        answer so the transcript survives a reload. Best-effort: the caller wraps
        it so a persistence failure never breaks the live stream.

        Args:
            session_id: Owning tenant session.
            role: ``user`` | ``assistant`` | ``system``.
            content: Plain-text turn content.
            extra: Optional structured payload (model/variant, citations).
        """
        async with session_scoped_conn(session_id) as conn:
            await conn.execute(
                """
                INSERT INTO chat_messages (session_id, role, content, extra)
                VALUES ($1, $2, $3, $4::jsonb)
                """,
                session_id,
                role,
                content,
                json.dumps(extra) if extra is not None else None,
            )
