"""Chat service: SSE stream wiring the perceiver + reasoner into ``/chat``.

This service owns the business logic of the conversational endpoint. It realises
the "Be My Eyes" pattern in two stages and serialises both to Server-Sent
Events:

1. **Perceiver (US-046)** — it instantiates
   :class:`~ml.agent.perceiver.PerceiverLayer`, asks it to observe the requested
   parcel/AOI, and emits the resulting *structured TEXT* observation as a
   dedicated ``perceiver_observation`` SSE event BEFORE the reasoner speaks. The
   same text block is injected into the reasoner's message history as a grounding
   turn so the model reasons over what the perceiver "saw" (never over logits).
2. **Reasoner (US-047)** — it builds the real :class:`~ml.agent.agent.Agent` via
   :func:`~ml.agent.agent.create_agent` (backend selected from
   ``settings.llm_variant_default``: ``gemini`` -> Gemini, ``qwen35`` -> vLLM
   Qwen), runs its manual function-calling loop, and forwards every
   :data:`~ml.agent.events.AgentEvent` it yields (``tool_call``, ``tool_result``,
   ``text_delta``, ``done``, ``error``) as an SSE frame.

Event order emitted by :meth:`ChatService.stream`:

1. ``perceiver_observation`` — REAL output of the perceiver (its
   :meth:`~ml.agent.perceiver.PerceiverObservation.to_prompt_block` plus the
   structured fields), the text the reasoner consumes. Omitted when the request
   carries neither a parcel nor an AOI.
2. ``tool_call`` / ``tool_result`` — zero or more pairs, one per tool the
   reasoner invokes during its loop.
3. ``text_delta`` — incremental chunks of the reasoner's grounded answer.
4. ``done`` — terminal event closing the stream once the reasoner answers.
5. ``error`` — emitted instead of ``done`` when the perceiver raises before the
   loop starts (the agent's own failures arrive as the loop's terminal ``error``
   event), so the client always receives a terminal event.

Business logic lives here (router -> service -> model): the router only adapts
HTTP <-> this async generator. Every DB read (perceiver + tools) is session
scoped through the shared :class:`~ml.agent.context.ToolContext`, so
multi-tenancy holds. The reasoner is injectable (``agent_factory``) so tests
drive the stream with a stubbed agent and never touch the network.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable, Sequence
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog
from pydantic import BaseModel, ConfigDict, Field

from backend.app.core.config import Settings, get_settings
from ml.agent.agent import create_agent
from ml.agent.context import ToolContext
from ml.agent.db import get_pool
from ml.agent.perceiver import PerceiverLayer, PerceiverObservation
from ml.agent.schemas import GeoJSONGeometry

if TYPE_CHECKING:
    from ml.agent.agent import Agent

logger = structlog.get_logger(__name__)

__all__ = ["AgentFactory", "ChatMessage", "ChatRequest", "ChatService"]

#: Default campaign year for an AOI observation (matches the perceiver default).
_DEFAULT_YEAR: int = 2019

#: Maps the persisted ``llm_variant_default`` value to the concrete reasoner
#: model name understood by :func:`~ml.agent.agent.create_agent` (which routes
#: ``gemini-*`` to the Gemini backend and ``qwen*`` to the vLLM backend). Kept
#: here so the HTTP layer never hardcodes a model string.
_VARIANT_TO_MODEL: dict[str, str] = {
    "gemini": "gemini-2.5-pro",
    "qwen35": "qwen35",
}

#: Fallback model when ``llm_variant_default`` carries an unmapped value (the
#: setting is a CHECK-constrained ``Literal``, so this is defensive only).
_DEFAULT_MODEL: str = "gemini-2.5-pro"

#: Signature of the injectable reasoner factory: it receives the resolved model
#: name and the typed ``settings`` keyword and returns a ready-to-stream
#: :class:`Agent`. The default adapter delegates to
#: :func:`~ml.agent.agent.create_agent` (forwarding ``settings`` by keyword so it
#: is never mistaken for the positional ``tools`` argument); tests pass a stub so
#: the stream runs without the real ``google-genai`` client or any network call.
AgentFactory = Callable[..., "Agent"]


def _default_agent_factory(model: str, *, settings: Settings) -> Agent:
    """Build the production reasoner for ``(model, settings)``.

    Thin adapter over :func:`~ml.agent.agent.create_agent` that pins the call to
    keyword ``settings`` (``create_agent``'s second positional parameter is
    ``tools``), keeping the :data:`AgentFactory` contract unambiguous.

    Args:
        model: Resolved reasoner model name.
        settings: Typed application settings forwarded to the backend.

    Returns:
        A ready-to-stream :class:`~ml.agent.agent.Agent`.
    """
    return create_agent(model=model, settings=settings)


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
    optional); when both are absent the stream still runs the reasoner over the
    conversation history alone (no perceiver grounding). ``session_id`` is
    optional in the body because the router also accepts it from the
    ``X-Session-ID`` header.

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

    Owns the business logic of the ``/chat`` endpoint: it builds the shared tool
    execution context, drives the perceiver to produce a real observation, then
    runs the reasoner loop and forwards its events. The router consumes
    :meth:`stream` verbatim into a ``StreamingResponse`` and contains no logic of
    its own.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        agent_factory: AgentFactory | None = None,
    ) -> None:
        """Initialise the service with typed settings and an injectable reasoner.

        Args:
            settings: Application settings; defaults to the cached singleton via
                :func:`~backend.app.core.config.get_settings`. Always injected
                rather than read from ``os.environ`` (backend convention).
            agent_factory: Callable building the reasoner from ``(model, *,
                settings)``; defaults to :func:`_default_agent_factory` (a thin
                adapter over :func:`~ml.agent.agent.create_agent`). Tests pass a
                stub so the stream runs without the real ``google-genai`` client
                or any network call.
        """
        self._settings = settings or get_settings()
        self._agent_factory: AgentFactory = agent_factory or _default_agent_factory

    def _reasoner_model(self) -> str:
        """Resolve the reasoner model name from the configured LLM variant.

        Returns:
            The concrete model name for :func:`~ml.agent.agent.create_agent`
            (e.g. ``"gemini-2.5-pro"`` for the ``gemini`` variant, ``"qwen35"``
            for the on-prem vLLM variant).
        """
        variant = getattr(self._settings, "llm_variant_default", "gemini")
        return _VARIANT_TO_MODEL.get(variant, _DEFAULT_MODEL)

    async def _build_context(self, session_id: UUID) -> ToolContext:
        """Build the :class:`ToolContext` shared by the perceiver and the tools.

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

    async def _observe(self, request: ChatRequest, ctx: ToolContext) -> PerceiverObservation | None:
        """Run the perceiver over the request's parcel/AOI, if any.

        Args:
            request: The validated chat request (subject + history).
            ctx: The shared, session-scoped tool execution context.

        Returns:
            The :class:`PerceiverObservation` for the requested subject, or
            ``None`` when the request carries neither a parcel nor an AOI.
        """
        if request.parcel_id is None and request.aoi is None:
            return None

        perceiver = PerceiverLayer(ctx)
        if request.parcel_id is not None:
            return await perceiver.observe(request.parcel_id)
        # ``aoi`` is non-None here by the guard above.
        assert request.aoi is not None
        return await perceiver.observe_aoi(request.aoi, request.year)

    @staticmethod
    def _agent_messages(
        messages: Sequence[ChatMessage],
        observation: PerceiverObservation | None,
    ) -> list[dict[str, str]]:
        """Build the reasoner's message history, grounded by the perceiver.

        The perceiver's :meth:`~ml.agent.perceiver.PerceiverObservation.to_prompt_block`
        rendering is injected as a leading grounding turn so the reasoner reasons
        over the structured TEXT the agent "saw" (Be My Eyes), never over raw
        logits. The agent maps any non ``user``/``model`` role into the ``user``
        role, so a ``system`` grounding turn reaches the model as context.

        Args:
            messages: The validated conversation history.
            observation: The perceiver observation to ground the reasoner with,
                or ``None`` when there is no subject to observe.

        Returns:
            A list of ``{"role", "content"}`` dicts for
            :meth:`~ml.agent.agent.Agent.stream_response`.
        """
        history: list[dict[str, str]] = []
        if observation is not None:
            history.append({"role": "system", "content": observation.to_prompt_block()})
        history.extend({"role": m.role, "content": m.content} for m in messages)
        return history

    async def stream(
        self, messages: Sequence[ChatMessage], session_id: UUID, *, request: ChatRequest
    ) -> AsyncIterator[str]:
        """Yield the chat response as a sequence of SSE frames.

        Emits, in order: a real ``perceiver_observation`` event (when a subject
        was supplied), then the reasoner's own event stream (``tool_call`` /
        ``tool_result`` pairs, ``text_delta`` chunks, and a terminal ``done`` or
        ``error``). On perceiver failure -- before the reasoner starts -- it emits
        a terminal ``error`` frame so the client always sees a terminal event.

        Args:
            messages: Conversation history (most recent turn last); fed to the
                reasoner together with the perceiver grounding block.
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

        ctx = await self._build_context(session_id)

        try:
            observation = await self._observe(request, ctx)
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
                    "observation": observation.model_dump(mode="json"),
                    "prompt_block": observation.to_prompt_block(),
                },
            )

        async for frame in self._stream_reasoner(messages, observation, session_id, ctx):
            yield frame

        logger.info(
            "chat_stream_finished",
            session_id=str(session_id),
            perceiver_observation_emitted=observation is not None,
            duration_ms=round((time.perf_counter() - start) * 1000.0, 2),
        )

    async def _stream_reasoner(
        self,
        messages: Sequence[ChatMessage],
        observation: PerceiverObservation | None,
        session_id: UUID,
        ctx: ToolContext,
    ) -> AsyncIterator[str]:
        """Run the reasoner loop and forward its events as SSE frames.

        Builds the grounded reasoner, drives
        :meth:`~ml.agent.agent.Agent.stream_response`, and serialises each
        :data:`~ml.agent.events.AgentEvent` to an SSE frame whose event name is
        the event's ``type`` literal (``tool_call``, ``tool_result``,
        ``text_delta``, ``done``, ``error``) and whose payload is the remaining
        fields. The agent already turns its own failures into a terminal
        ``error`` event, so the stream always ends with ``done`` or ``error``.

        Args:
            messages: The conversation history.
            observation: The perceiver grounding observation, if any.
            session_id: Effective tenant session.
            ctx: Shared, session-scoped tool execution context.

        Yields:
            SSE frames for each reasoner event.
        """
        agent = self._agent_factory(self._reasoner_model(), settings=self._settings)
        agent_messages = self._agent_messages(messages, observation)

        async for event in agent.stream_response(agent_messages, session_id, ctx):
            payload = event.model_dump(mode="json")
            event_name = payload.pop("type")
            yield _sse_event(event_name, payload)
