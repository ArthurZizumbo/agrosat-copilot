"""Chat service: SSE stream wiring the perceiver layer into ``/chat``.

This service is the MVP skeleton of the conversational endpoint (US-046). It does
NOT yet run the reasoner loop (the real Gemini Plan-and-React loop lands in
US-047). What it *does* implement for real is the perceiver hook of the "Be My
Eyes" pattern: it instantiates :class:`~ml.agent.perceiver.PerceiverLayer`, asks
it to observe the requested parcel/AOI, and emits the resulting *structured TEXT*
observation as a dedicated ``perceiver_observation`` SSE event BEFORE the final
answer (AC-3 of US-046).

Event order emitted by :meth:`ChatService.stream`:

1. ``perceiver_observation`` — REAL output of the perceiver (its
   :meth:`~ml.agent.perceiver.PerceiverObservation.to_prompt_block` plus the
   structured fields), the text the future reasoner will consume.
2. ``text_delta`` — incremental answer chunks. In this MVP it is a single,
   documented placeholder chunk (the real token stream arrives with the US-047
   reasoner); when the perceiver cannot observe anything it is omitted.
3. ``done`` — terminal event closing the stream.
4. ``error`` — emitted instead of ``done`` if the perceiver raises, so the
   client always receives a terminal event.

Business logic lives here (router -> service -> model): the router only adapts
HTTP <-> this async generator. Every DB read performed by the perceiver is
session-scoped through the tools it reuses, so multi-tenancy holds.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any
from uuid import UUID

import structlog
from pydantic import BaseModel, ConfigDict, Field

from backend.app.core.config import Settings, get_settings
from ml.agent.context import ToolContext
from ml.agent.db import get_pool
from ml.agent.perceiver import PerceiverLayer, PerceiverObservation
from ml.agent.schemas import GeoJSONGeometry

logger = structlog.get_logger(__name__)

__all__ = ["ChatMessage", "ChatRequest", "ChatService"]

#: Default campaign year for an AOI observation (matches the perceiver default).
_DEFAULT_YEAR: int = 2019

#: Documented placeholder answer. The real reasoner (Gemini loop) lands in
#: US-047; until then the final text is this explicit notice so no consumer
#: mistakes it for a grounded model answer.
_PENDING_REASONER_NOTICE: str = (
    "Respuesta del razonador pendiente (US-047). Esta capa MVP ya entrega la "
    "observacion textual del perceiver mostrada arriba; el bucle Gemini que "
    "razona sobre ese texto se habilita en una historia posterior."
)


class ChatMessage(BaseModel):
    """A single chat turn in the conversation history.

    Attributes:
        role: Author of the turn (``"user"``, ``"assistant"`` or ``"system"``).
        content: Plain-text content of the turn.
    """

    model_config = ConfigDict(extra="forbid")

    role: str
    content: str


class ChatRequest(BaseModel):
    """Body of a ``/chat`` request.

    The perceiver needs an explicit subject to observe. The request therefore
    carries either a stored ``parcel_id`` or a drawn ``aoi`` polygon (mutually
    optional); when both are absent the stream still completes but emits no
    perceiver observation (the future reasoner would then answer from history
    alone). ``session_id`` is optional in the body because the router also
    accepts it from the ``X-Session-ID`` header.

    Attributes:
        messages: Conversation history (most recent turn last).
        session_id: Tenant session; when omitted here it must arrive via header.
        parcel_id: Stored parcel to observe with the perceiver, if any.
        aoi: Drawn AOI polygon to observe with the perceiver, if any.
        year: Campaign year of the AlphaEarth annual embedding for an AOI.
    """

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(default_factory=list)
    session_id: UUID | None = None
    parcel_id: int | None = None
    aoi: GeoJSONGeometry | None = None
    year: int = _DEFAULT_YEAR


def _sse_event(event: str, data: dict[str, Any]) -> str:
    """Serialise an SSE frame as ``event: <type>\\ndata: <json>\\n\\n``.

    Args:
        event: SSE event type (e.g. ``"perceiver_observation"``).
        data: JSON-serialisable payload rendered as the ``data:`` line.

    Returns:
        A single, fully-terminated SSE frame.
    """
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


class ChatService:
    """Stream the chat response as Server-Sent Events.

    Owns the business logic of the ``/chat`` endpoint: it builds the tool
    execution context, drives the perceiver to produce a real observation, and
    yields SSE frames. The router consumes :meth:`stream` verbatim into a
    ``StreamingResponse`` and contains no logic of its own.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialise the service with typed settings.

        Args:
            settings: Application settings; defaults to the cached singleton via
                :func:`~backend.app.core.config.get_settings`. Always injected
                rather than read from ``os.environ`` (backend convention).
        """
        self._settings = settings or get_settings()

    async def _build_context(self, session_id: UUID) -> ToolContext:
        """Build the :class:`ToolContext` shared by the perceiver and tools.

        Args:
            session_id: Tenant session driving every downstream DB read.

        Returns:
            A :class:`ToolContext` with the shared asyncpg pool, settings and
            session id (no deferred executor in this MVP).
        """
        pool = await get_pool()
        return ToolContext(
            pool=pool,
            settings=self._settings,
            session_id=session_id,
        )

    async def _observe(
        self, request: ChatRequest, session_id: UUID
    ) -> PerceiverObservation | None:
        """Run the perceiver over the request's parcel/AOI, if any.

        Args:
            request: The validated chat request (subject + history).
            session_id: Effective tenant session.

        Returns:
            The :class:`PerceiverObservation` for the requested subject, or
            ``None`` when the request carries neither a parcel nor an AOI.
        """
        if request.parcel_id is None and request.aoi is None:
            return None

        ctx = await self._build_context(session_id)
        perceiver = PerceiverLayer(ctx)
        if request.parcel_id is not None:
            return await perceiver.observe(request.parcel_id)
        # ``aoi`` is non-None here by the guard above.
        assert request.aoi is not None
        return await perceiver.observe_aoi(request.aoi, request.year)

    async def stream(
        self, messages: Sequence[ChatMessage], session_id: UUID, *, request: ChatRequest
    ) -> AsyncIterator[str]:
        """Yield the chat response as a sequence of SSE frames.

        Emits, in order: a real ``perceiver_observation`` event (when a subject
        was supplied), a placeholder ``text_delta`` (documented US-047 stub),
        and a terminal ``done``. On perceiver failure it emits ``error`` instead
        of ``done`` so the client always sees a terminal frame.

        Args:
            messages: Conversation history (kept for the future reasoner; unused
                by the MVP beyond echoing its length in the ``done`` event).
            session_id: Effective tenant session for every DB read.
            request: The full validated request (carries the parcel/AOI subject).

        Yields:
            SSE frames as ``str`` (``event:``/``data:`` pairs).
        """
        start = time.perf_counter()
        logger.info(
            "chat_stream_started",
            session_id=str(session_id),
            n_messages=len(messages),
            has_parcel=request.parcel_id is not None,
            has_aoi=request.aoi is not None,
        )

        try:
            observation = await self._observe(request, session_id)
        except Exception as exc:  # surface any perceiver failure as a terminal SSE error
            logger.exception(
                "chat_stream_perceiver_failed",
                session_id=str(session_id),
                error=str(exc),
            )
            yield _sse_event(
                "error",
                {
                    "message": "perceiver_observation_failed",
                    "detail": str(exc),
                },
            )
            return

        if observation is not None:
            yield _sse_event(
                "perceiver_observation",
                {
                    "observation": observation.model_dump(),
                    "prompt_block": observation.to_prompt_block(),
                },
            )

        # MVP placeholder: the grounded reasoner answer is US-047. The notice is
        # explicit so no client renders it as a real model conclusion.
        yield _sse_event("text_delta", {"text": _PENDING_REASONER_NOTICE})

        yield _sse_event(
            "done",
            {
                "session_id": str(session_id),
                "perceiver_observation_emitted": observation is not None,
                "reasoner": "pending_us_047",
            },
        )
        logger.info(
            "chat_stream_finished",
            session_id=str(session_id),
            perceiver_observation_emitted=observation is not None,
            duration_ms=round((time.perf_counter() - start) * 1000.0, 2),
        )
