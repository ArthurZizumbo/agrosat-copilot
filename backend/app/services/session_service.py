"""Chat-session service: create and delete ``chat_sessions`` rows (RLS-scoped).

Owns all SQL for the ``/sessions`` endpoints; the router stays thin (SoC). Every
method receives the request's RLS-scoped :class:`asyncpg.Connection`
(``app.current_session`` already primed inside an open transaction by
:func:`backend.app.core.db.get_scoped_conn`), so the US-051 ``tenant_isolation``
policies enforce isolation.

This closes the "session-creation gap": the rest of the API (``/aois``,
``/chat``, ...) requires a ``chat_sessions`` row to exist (``403`` otherwise via
the auth guard), but nothing created it. Here the client generates the new
chat's UUID, sends it in ``X-Session-ID`` (so it becomes ``app.current_session``)
and this service inserts ``id = that same UUID``: the RLS WITH CHECK policy
(``id = current_setting('app.current_session')::uuid``) then authorises the
write. The DELETE works the same way (the RLS USING clause only exposes the
current session), so a session can only ever delete itself.

The insert is idempotent: ``ON CONFLICT (id) DO NOTHING`` plus a fallback
``SELECT`` returns the existing row when the chat was already created (e.g. a
client retry), so ``POST /sessions`` is safe to call more than once.
"""

from __future__ import annotations

from uuid import UUID

import asyncpg
import structlog

from backend.app.models.session import LLM_MODELS, SessionCreate, SessionRead

__all__ = ["SessionService"]

logger = structlog.get_logger(__name__)

# INSERT the new chat. ``$1`` is the session id from the ``X-Session-ID`` header
# (= the primed ``app.current_session``), so the RLS WITH CHECK policy authorises
# the write. ``ON CONFLICT (id) DO NOTHING`` makes the call idempotent: a retry
# with the same id returns no row here and is resolved by ``_SELECT_SQL``.
_INSERT_SQL = """
INSERT INTO chat_sessions (id, user_id, llm_model)
VALUES ($1, $2, $3)
ON CONFLICT (id) DO NOTHING
RETURNING id, user_id, llm_model, created_at
"""

# Fallback read for the idempotent path: when the row already existed the INSERT
# returned nothing, so fetch the persisted row (RLS scopes it to the session).
_SELECT_SQL = "SELECT id, user_id, llm_model, created_at FROM chat_sessions WHERE id = $1"

# DELETE under RLS removes the row only if it is the current session; the ON
# DELETE CASCADE on dependent tables (aois, parcels, chat_messages, ...) removes
# the chat's data with it.
_DELETE_SQL = "DELETE FROM chat_sessions WHERE id = $1"


def _row_to_read(row: asyncpg.Record) -> SessionRead:
    """Map a ``chat_sessions`` row to a :class:`SessionRead` response model.

    Args:
        row: An asyncpg record exposing ``id``, ``user_id``, ``llm_model`` and
            ``created_at``.

    Returns:
        The session as a :class:`SessionRead`.
    """
    return SessionRead(
        id=row["id"],
        user_id=row["user_id"],
        llm_model=row["llm_model"],
        created_at=row["created_at"],
    )


class SessionService:
    """Stateless create/delete operations over ``chat_sessions`` (RLS-scoped)."""

    @staticmethod
    def validate_llm_model(model: str) -> str:
        """Validate ``model`` against the database CHECK set (defence in depth).

        The router already constrains the body to the :data:`LlmModel` literal,
        so this keeps the service independently correct (the single place that
        re-checks the persisted domain, :data:`LLM_MODELS`).

        Args:
            model: The requested reasoner variant.

        Returns:
            The validated variant (unchanged).

        Raises:
            ValueError: When ``model`` is not one of the supported variants.
        """
        if model not in LLM_MODELS:
            raise ValueError(f"Unsupported llm_model: {model!r}. Expected one of {LLM_MODELS}.")
        return model

    @classmethod
    async def create(
        cls,
        conn: asyncpg.Connection,
        session_id: UUID,
        body: SessionCreate,
    ) -> tuple[SessionRead, bool]:
        """Create (or return the existing) chat session for ``session_id``.

        Inserts ``id = session_id`` so the RLS WITH CHECK policy authorises the
        write (``session_id`` equals the primed ``app.current_session``). The
        call is idempotent: when the row already exists, ``ON CONFLICT DO
        NOTHING`` yields no row and the persisted one is read back instead.

        Args:
            conn: RLS-scoped connection (session primed, in a transaction).
            session_id: The new chat's UUID, taken from ``X-Session-ID`` (equals
                ``app.current_session``).
            body: Validated request body (``user_id`` + optional ``llm_model``).

        Returns:
            A ``(session, created)`` tuple where ``created`` is ``True`` when a
            new row was inserted and ``False`` when an existing row was returned
            (the router maps this to ``201`` vs ``200``).
        """
        model = cls.validate_llm_model(body.llm_model)
        row = await conn.fetchrow(_INSERT_SQL, session_id, body.user_id, model)
        if row is not None:
            session = _row_to_read(row)
            logger.info("session_created", session_id=str(session.id), user_id=session.user_id)
            return session, True
        # ON CONFLICT DO NOTHING returned no row: the session already exists.
        # Read it back (RLS scopes it to the current session) and return it
        # unchanged so the POST is idempotent.
        existing = await conn.fetchrow(_SELECT_SQL, session_id)
        # ``existing`` is never None here: the conflict proves a row with this id
        # is visible under the current RLS scope.
        session = _row_to_read(existing)
        logger.info("session_create_idempotent", session_id=str(session.id))
        return session, False

    @staticmethod
    async def delete(conn: asyncpg.Connection, session_id: UUID) -> None:
        """Delete the chat session ``session_id`` (and its data, via CASCADE).

        Under RLS the ``DELETE`` only ever matches the current session, so a
        foreign id removes zero rows. The endpoint is ``204`` regardless (no
        foreign-existence leak), so this returns ``None`` either way.

        Args:
            conn: RLS-scoped connection (session primed, in a transaction).
            session_id: The chat session to delete (equals
                ``app.current_session``).
        """
        status_tag = await conn.execute(_DELETE_SQL, session_id)
        if status_tag.endswith(" 0"):
            logger.info("session_delete_noop", session_id=str(session_id))
        else:
            logger.info("session_deleted", session_id=str(session_id))
