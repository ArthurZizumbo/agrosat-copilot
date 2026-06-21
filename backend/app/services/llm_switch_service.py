"""LLM switch service: persist the active reasoner variant per session (US-054).

Owns the business logic of ``POST /llm/switch`` (router -> service -> DB): it
validates the requested variant against the routing table and persists it on the
caller's ``chat_sessions`` row. The router stays thin and runs no SQL.

Tenant isolation is enforced by the US-051 RLS policies, NOT by an
application-level ``WHERE``: the ``UPDATE`` runs on the RLS-scoped connection
(:func:`~backend.app.core.db.get_scoped_conn`, role ``agrosat_app`` NOBYPASSRLS,
``app.current_session`` primed), so it can only touch the caller's own row. A
foreign session id is invisible and the ``UPDATE`` affects zero rows.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

import structlog

from ml.agent.llm_routing import VARIANTS

if TYPE_CHECKING:  # pragma: no cover - typing only
    import asyncpg

logger = structlog.get_logger(__name__)

__all__ = ["LLMSwitchResult", "LLMSwitchService"]

#: Persist the variant on the caller's own session and return the applied value
#: plus the new ``updated_at``. RLS scopes the row to ``app.current_session``;
#: the ``RETURNING`` row is ``None`` when no row matched (foreign / unknown
#: session, already rejected by the auth guard upstream).
_SWITCH_SQL = (
    "UPDATE chat_sessions SET llm_model = $1, updated_at = now() "
    "WHERE id = $2 RETURNING llm_model, updated_at"
)


class LLMSwitchResult:
    """Outcome of a switch: the applied variant and when it was persisted.

    Attributes:
        model: The variant tag now persisted on the session.
        applied_at: The row's ``updated_at`` after the switch.
    """

    __slots__ = ("applied_at", "model")

    def __init__(self, model: str, applied_at: datetime) -> None:
        self.model = model
        self.applied_at = applied_at


class LLMSwitchService:
    """Persist the per-session reasoner variant under RLS."""

    @staticmethod
    def validate_variant(model: str) -> str:
        """Validate that ``model`` is one of the four supported variants.

        The router already constrains the body to a ``Literal`` of the four
        variants, so this is a defence-in-depth check keeping the service
        independently correct (and the single place that knows the valid set,
        :data:`~ml.agent.llm_routing.VARIANTS`).

        Args:
            model: The requested variant tag.

        Returns:
            The validated variant (unchanged).

        Raises:
            ValueError: When ``model`` is not a supported variant.
        """
        if model not in VARIANTS:
            raise ValueError(f"Unsupported LLM variant: {model!r}. Expected one of {VARIANTS}.")
        return model

    @classmethod
    async def switch(
        cls,
        conn: asyncpg.Connection,
        session_id: UUID,
        model: str,
    ) -> LLMSwitchResult:
        """Persist ``model`` as the active reasoner variant for the session.

        Validates the variant, then runs the scoped ``UPDATE``. Because the
        connection is RLS-scoped to ``session_id``, the update only ever touches
        the caller's own row; a zero-row result means the session is unknown to
        this tenant (which the auth guard already prevents, so it is treated as a
        defensive error here). Logs ``llm_switch`` with latency (AC-5, FinOps
        US-065 input).

        Args:
            conn: RLS-scoped connection bound to ``session_id``.
            session_id: The caller's tenant session id (matches the RLS hook).
            model: The requested variant tag (validated against
                :data:`~ml.agent.llm_routing.VARIANTS`).

        Returns:
            The :class:`LLMSwitchResult` with the applied variant and timestamp.

        Raises:
            ValueError: When ``model`` is not a supported variant, or when no row
                was updated (session not visible under RLS).
        """
        variant = cls.validate_variant(model)
        start = time.perf_counter()
        row = await conn.fetchrow(_SWITCH_SQL, variant, session_id)
        latency_ms = round((time.perf_counter() - start) * 1000.0, 2)
        if row is None:
            logger.warning("llm_switch_no_row", session_id=str(session_id), model=variant)
            raise ValueError("Session not found or not accessible.")
        applied: datetime = row["updated_at"]
        logger.info(
            "llm_switch",
            session_id=str(session_id),
            model=variant,
            latency_ms=latency_ms,
        )
        return LLMSwitchResult(model=row["llm_model"], applied_at=applied)
