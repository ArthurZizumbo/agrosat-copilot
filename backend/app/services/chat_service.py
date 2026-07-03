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
2. **Reasoner (US-047 / US-054)** — it reads the session's persisted LLM variant
   from ``chat_sessions.llm_model`` (per-request, RLS-scoped), resolves it through
   the backend-agnostic routing table (:mod:`ml.agent.llm_routing`) and builds the
   matching :class:`~ml.agent.backends.LLMBackend` (``gemini`` -> Gemini,
   ``qwen-api`` / ``qwen-onprem`` -> OpenAI-compatible vLLM, ``gemma`` -> Ollama,
   ``qwen-vl`` -> multimodal Ollama/llama.cpp, E12),
   wires it into a real :class:`~ml.agent.agent.Agent`, runs its manual
   function-calling loop, and forwards every :data:`~ml.agent.events.AgentEvent`
   it yields (``tool_call``, ``tool_result``, ``text_delta``, ``done``,
   ``error``) as an SSE frame.

   The variant is read PER REQUEST (in :meth:`ChatService.stream`, not
   ``__init__``) because the active model depends on the request's
   ``session_id``: a ``/llm/switch`` on that session must take effect on the next
   ``/chat`` (US-054 AC-2). ``settings.llm_variant_default`` is kept only as the
   startup/fallback variant when the row carries no value or an unknown one.

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
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

import structlog
from pydantic import BaseModel, ConfigDict, Field

from backend.app.core.config import Settings, get_settings
from backend.app.core.db import get_pool
from backend.app.services.session_service import SessionService
from backend.app.utils.chat_metrics import ChatMetricsAccumulator, emit_chat_turn_metrics
from ml.agent.context import ToolContext
from ml.agent.llm_routing import (
    DEFAULT_VARIANT,
    VARIANTS,
    make_backend_for_variant_available,
)
from ml.agent.perceiver import PerceiverLayer, PerceiverObservation
from ml.agent.schemas import GeoJSONGeometry

if TYPE_CHECKING:
    import asyncpg

    from ml.agent.agent import Agent

logger = structlog.get_logger(__name__)

__all__ = ["AgentFactory", "ChatMessage", "ChatRequest", "ChatService", "PoolFactory"]

#: Reasoner language instruction per UI locale (US-057). Injected as a leading
#: ``system`` turn so the frozen reasoner answers in the user's language without
#: changing the analyst system prompt.
_LOCALE_INSTRUCTION: dict[str, str] = {
    "it": "Rispondi sempre in italiano.",
    "es": "Responde siempre en español.",
    "en": "Always answer in English.",
}

#: Crop-classification models the user can pin for ``classify_new_parcel``
#: (mirror of :data:`ml.agent.schemas.ClassifyParcelInput.model`). When the
#: request carries one, a leading ``system`` turn instructs the reasoner to pass
#: ``model='<value>'`` to the tool, so the choice is honoured WITHOUT hardcoding
#: the model in the tool itself (the tool keeps its ``voting3`` default).
CropModel = Literal["voting3", "xgb", "stacking5"]

#: System instruction injected (as a leading ``system`` turn, same mechanism as
#: ``_LOCALE_INSTRUCTION``) when the request pins a crop model. ``{model}`` is the
#: validated :data:`CropModel` value, so the reasoner forwards it verbatim to the
#: ``classify_new_parcel`` tool's ``model`` argument.
_CROP_MODEL_INSTRUCTION = (
    "When you call the classify_new_parcel tool, you MUST pass model='{model}' "
    "as its argument; the user explicitly selected this crop-classification model."
)

#: Default campaign year for an AOI observation (matches the perceiver default).
_DEFAULT_YEAR: int = 2019

#: Read the persisted reasoner variant for a session (US-054 AC-2). Run on the
#: RLS-scoped pool connection (``app.current_session`` primed), so it only ever
#: returns the caller's own row.
_SELECT_VARIANT_SQL = "SELECT llm_model FROM chat_sessions WHERE id = $1"

#: Signature of the injectable reasoner factory: it receives the resolved LLM
#: VARIANT tag (``gemini`` / ``qwen-api`` / ``qwen-onprem`` / ``gemma`` /
#: ``qwen-vl``) and the typed ``settings`` keyword and returns a ready-to-stream
#: :class:`Agent`. The
#: default adapter (:func:`_default_agent_factory`) resolves the variant through
#: the backend-agnostic routing table (:mod:`ml.agent.llm_routing`) and wires the
#: matching backend into the agent; tests pass a stub so the stream runs without
#: the real ``google-genai`` client or any network call, and assert the backend
#: type selected per variant (US-054 AC-2).
AgentFactory = Callable[..., "Agent"]

#: Signature of the injectable pool factory: an awaitable returning the shared
#: asyncpg pool the :class:`ToolContext` runs the tools against. The default is
#: :func:`backend.app.core.db.get_pool` (the ``app_database_url`` pool, role
#: ``agrosat_app`` WITHOUT ``BYPASSRLS``) so the per-session ``SET LOCAL
#: app.current_session`` emitted by the tools actually enforces the US-051 RLS
#: policies (B2). Using the ``ml.agent.db`` superuser pool would silently bypass
#: RLS. Tests inject a fake so the stream runs without a real database.
PoolFactory = Callable[[], "Awaitable[asyncpg.Pool]"]


def _default_agent_factory(variant: str, *, settings: Settings) -> Agent:
    """Build the production reasoner for a persisted LLM ``variant``.

    Resolves the variant through the AVAILABILITY-AWARE routing table
    (:func:`~ml.agent.llm_routing.make_backend_for_variant_available`, US-081
    AC10 / E12) and wires the resulting :class:`~ml.agent.backends.LLMBackend`
    into an :class:`~ml.agent.agent.Agent` with the default tool set and analyst
    prompt.

    Honest degradation (E12): the on-prem variants (``qwen-onprem`` / ``qwen-vl``
    / ``gemma``) are reachable only behind the demo VM tunnel (``make demo-vm``).
    The availability-aware resolver PROBES the on-prem host and, when it is
    unreachable, falls back to the always-resolvable ``gemini`` route instead of
    failing the ``/chat`` request at request time. The ``/llm/switch`` endpoint
    only persists the chosen variant (it does not probe), so this is where a dead
    host is caught -- the user keeps a working reasoner rather than a timeout.

    The agent is assembled here (rather than via
    :func:`~ml.agent.agent.create_agent`, which selects the backend by
    model-name prefix and so cannot tell ``qwen-api`` from ``qwen-onprem`` apart)
    so the backend is selected PER VARIANT and never by heuristic.

    Args:
        variant: The persisted LLM variant tag (one of
            :data:`~ml.agent.llm_routing.VARIANTS`).
        settings: Typed application settings forwarded to the backend resolver.

    Returns:
        A ready-to-stream :class:`~ml.agent.agent.Agent`.
    """
    from ml.agent.agent import Agent
    from ml.agent.prompts import ANALYST_SYSTEM_PROMPT
    from ml.agent.tools import get_sync_tools

    backend, _decision = make_backend_for_variant_available(variant, settings)
    return Agent(
        backend=backend,
        tools=get_sync_tools(),
        instruction=ANALYST_SYSTEM_PROMPT,
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
    #: Active UI locale (US-057). Conditions the reasoner's answer language so a
    #: trilingual frontend gets replies in the user's language. ``None`` -> the
    #: agent answers in its default (Spanish, per the analyst system prompt).
    locale: Literal["it", "es", "en"] | None = None
    #: Crop-classification model the USER pinned for ``classify_new_parcel``
    #: (mirror of :data:`ml.agent.schemas.ClassifyParcelInput.model`). When set, a
    #: leading ``system`` turn instructs the reasoner to forward ``model=<value>``
    #: to the tool (see :data:`_CROP_MODEL_INSTRUCTION`); the model is never
    #: hardcoded in the tool. ``None`` -> the tool keeps its ``voting3`` default
    #: and the LLM may still choose another model on its own.
    crop_model: CropModel | None = None


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
        pool_factory: PoolFactory | None = None,
    ) -> None:
        """Initialise the service with typed settings and injectable boundaries.

        Args:
            settings: Application settings; defaults to the cached singleton via
                :func:`~backend.app.core.config.get_settings`. Always injected
                rather than read from ``os.environ`` (backend convention).
            agent_factory: Callable building the reasoner from ``(variant, *,
                settings)``; defaults to :func:`_default_agent_factory` (which
                resolves the variant through :mod:`ml.agent.llm_routing` and wires
                the matching backend into the agent). Tests pass a stub so the
                stream runs without the real ``google-genai`` client or any
                network call, and assert the backend type per variant (AC-2).
            pool_factory: Awaitable returning the asyncpg pool the tools run
                against; defaults to :func:`backend.app.core.db.get_pool` (the
                ``app_database_url`` pool, role ``agrosat_app`` NOBYPASSRLS) so
                the US-051 RLS policies enforce on the ``/chat`` tools (B2).
                Tests inject a fake so the stream runs without a real database.
        """
        self._settings = settings or get_settings()
        self._agent_factory: AgentFactory = agent_factory or _default_agent_factory
        self._pool_factory: PoolFactory = pool_factory or get_pool

    def _fallback_variant(self) -> str:
        """Return the startup/fallback LLM variant from settings.

        US-054 D3: ``settings.llm_variant_default`` is no longer the per-request
        source (the session row is), only the honest-degradation target when the
        row carries no value or an unknown one. Guarded against an out-of-range
        value (defensive: the setting is a CHECK-constrained ``Literal``).

        Returns:
            A valid variant tag (one of :data:`~ml.agent.llm_routing.VARIANTS`),
            defaulting to :data:`~ml.agent.llm_routing.DEFAULT_VARIANT`.
        """
        variant = str(getattr(self._settings, "llm_variant_default", DEFAULT_VARIANT))
        return variant if variant in VARIANTS else DEFAULT_VARIANT

    async def _resolve_variant(self, ctx: ToolContext, session_id: UUID) -> str:
        """Read the session's persisted LLM variant (US-054 AC-2).

        Runs ``SELECT llm_model FROM chat_sessions WHERE id = $1`` on a
        connection from the shared pool with the per-session RLS hook primed
        (``app.current_session`` = ``session_id``, ``SET LOCAL`` semantics), so it
        only ever reads the caller's own row. A missing row, a ``NULL`` value or
        an unknown variant degrades honestly to :meth:`_fallback_variant`
        (``gemini`` by default) with a ``logger.warning``.

        Args:
            ctx: The shared, session-scoped tool execution context (its ``pool``
                is the ``agrosat_app`` NOBYPASSRLS pool, so RLS enforces).
            session_id: Tenant session whose persisted variant is read.

        Returns:
            The active variant tag for this request (one of
            :data:`~ml.agent.llm_routing.VARIANTS`).
        """
        conn = await ctx.pool.acquire()
        try:
            async with conn.transaction():
                await conn.execute(
                    "SELECT set_config('app.current_session', $1, true)",
                    str(session_id),
                )
                row = await conn.fetchrow(_SELECT_VARIANT_SQL, session_id)
        finally:
            await ctx.pool.release(conn)

        value = row["llm_model"] if row is not None else None
        if value in VARIANTS:
            return str(value)
        fallback = self._fallback_variant()
        logger.warning(
            "chat_variant_fallback",
            session_id=str(session_id),
            persisted=value,
            fallback=fallback,
        )
        return fallback

    async def _build_context(self, session_id: UUID) -> ToolContext:
        """Build the :class:`ToolContext` shared by the perceiver and the tools.

        Args:
            session_id: Tenant session driving every downstream DB read.

        The pool comes from :attr:`_pool_factory` (by default the
        ``app_database_url`` pool of :func:`backend.app.core.db.get_pool`, role
        ``agrosat_app`` NOBYPASSRLS), so the per-session ``SET LOCAL`` the tools
        emit isolates them under the US-051 RLS policies (B2).

        Returns:
            A :class:`ToolContext` with the shared asyncpg pool, settings and
            session id (no deferred executor in this MVP).
        """
        pool = await self._pool_factory()
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
        # ``aoi`` is non-None here by the guard above. Forward the user-picked crop
        # model so the AOI observation runs (and reports) the SAME model the user
        # selected in the UI -- voting3 by default -- instead of always XGBoost.
        assert request.aoi is not None
        return await perceiver.observe_aoi(request.aoi, request.year, request.crop_model)

    @staticmethod
    def _agent_messages(
        messages: Sequence[ChatMessage],
        observation: PerceiverObservation | None,
        locale: Literal["it", "es", "en"] | None = None,
        crop_model: CropModel | None = None,
    ) -> list[dict[str, str]]:
        """Build the reasoner's message history, grounded by the perceiver.

        The perceiver's :meth:`~ml.agent.perceiver.PerceiverObservation.to_prompt_block`
        rendering is injected as a leading grounding turn so the reasoner reasons
        over the structured TEXT the agent "saw" (Be My Eyes), never over raw
        logits. The agent maps any non ``user``/``model`` role into the ``user``
        role, so a ``system`` grounding turn reaches the model as context. When a
        ``locale`` is supplied (US-057) a language-instruction ``system`` turn is
        prepended so the reasoner answers in the user's UI language. When a
        ``crop_model`` is supplied a second ``system`` turn instructs the reasoner
        to forward ``model='<crop_model>'`` to the ``classify_new_parcel`` tool, so
        the user's choice is honoured without hardcoding the model in the tool.

        Args:
            messages: The validated conversation history.
            observation: The perceiver observation to ground the reasoner with,
                or ``None`` when there is no subject to observe.
            locale: Active UI locale (``it``/``es``/``en``) or ``None`` for the
                agent's default answer language.
            crop_model: Crop-classification model the user pinned for
                ``classify_new_parcel`` (``voting3``/``xgb``/``stacking5``), or
                ``None`` to leave the tool's own default in place.

        Returns:
            A list of ``{"role", "content"}`` dicts for
            :meth:`~ml.agent.agent.Agent.stream_response`.
        """
        history: list[dict[str, str]] = []
        if locale is not None:
            history.append({"role": "system", "content": _LOCALE_INSTRUCTION[locale]})
        if crop_model is not None:
            history.append(
                {
                    "role": "system",
                    "content": _CROP_MODEL_INSTRUCTION.format(model=crop_model),
                }
            )
        if observation is not None:
            # Frame the perceiver block so the reasoner treats it as the answer
            # source for the area the user already selected, instead of asking the
            # user to draw an AOI that is in fact already provided (B: the reasoner
            # was replying "draw the area" despite a valid observation).
            grounding = (
                "El usuario ya selecciono un area de interes (AOI) en el mapa y el "
                "perceiver del equipo la observo. Responde la pregunta del usuario "
                "USANDO esta observacion como fuente; NO pidas que dibuje el area, "
                "porque ya esta seleccionada. Reporta la clase de cultivo estimada y "
                "su confianza tal cual; si la confianza es baja, advierte que es una "
                "estimacion preliminar, pero igualmente da la clase.\n\n"
                f"{observation.to_prompt_block()}"
            )
            history.append({"role": "system", "content": grounding})
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

        # Persist the new user turn (US-080) so the transcript survives a reload.
        # Best-effort: a persistence failure must never break the live stream.
        if messages and messages[-1].role == "user" and messages[-1].content.strip():
            try:
                await SessionService.save_message(session_id, "user", messages[-1].content)
            except Exception:  # noqa: BLE001
                logger.warning("chat_user_message_persist_failed", session_id=str(session_id))

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

        async for frame in self._stream_reasoner(
            messages,
            observation,
            session_id,
            ctx,
            locale=request.locale,
            crop_model=request.crop_model,
            start=start,
        ):
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
        *,
        locale: Literal["it", "es", "en"] | None = None,
        crop_model: CropModel | None = None,
        start: float,
    ) -> AsyncIterator[str]:
        """Run the reasoner loop and forward its events as SSE frames.

        Builds the grounded reasoner, drives
        :meth:`~ml.agent.agent.Agent.stream_response`, and serialises each
        :data:`~ml.agent.events.AgentEvent` to an SSE frame whose event name is
        the event's ``type`` literal (``tool_call``, ``tool_result``,
        ``text_delta``, ``done``, ``error``) and whose payload is the remaining
        fields. The agent already turns its own failures into a terminal
        ``error`` event, so the stream always ends with ``done`` or ``error``.

        Observability (US-065): the reasoner events are observed in flight to
        build a per-turn :class:`~backend.app.utils.chat_metrics.ChatMetricsAccumulator`
        -- one increment per ``tool_call``, one ``usage`` capture on the terminal
        ``done`` -- and a single ``chat_turn_metrics`` line is emitted once the
        stream ends, scored against the latency SLO using the END-TO-END timer
        owned by :meth:`stream` (``start``), so there is a single source of truth
        for the turn latency.

        Args:
            messages: The conversation history.
            observation: The perceiver grounding observation, if any.
            session_id: Effective tenant session.
            ctx: Shared, session-scoped tool execution context.
            locale: Active UI locale conditioning the answer language, if any.
            crop_model: Crop-classification model the user pinned for
                ``classify_new_parcel``, forwarded to the reasoner as a system
                instruction, if any.
            start: The ``time.perf_counter`` mark taken by :meth:`stream` at the
                turn's start, reused so the SLO latency is not double-measured.

        Yields:
            SSE frames for each reasoner event.
        """
        resolve_start = time.perf_counter()
        variant = await self._resolve_variant(ctx, session_id)
        agent = self._agent_factory(variant, settings=self._settings)
        # ``model`` is the concrete reasoner id behind the variant (surfaced by the
        # backend for logging); ``latency_ms`` here is the per-request resolution
        # cost (DB read + backend build), the FinOps US-065 input (AC-5).
        model = getattr(getattr(agent, "backend", None), "model", None)
        logger.info(
            "chat_model_resolved",
            session_id=str(session_id),
            variant=variant,
            model=model,
            latency_ms=round((time.perf_counter() - resolve_start) * 1000.0, 2),
        )
        agent_messages = self._agent_messages(messages, observation, locale, crop_model)

        metrics = ChatMetricsAccumulator(variant=variant, model=model)
        answer_parts: list[str] = []
        async for event in agent.stream_response(agent_messages, session_id, ctx):
            payload = event.model_dump(mode="json")
            event_name = payload.pop("type")
            # Observe the FinOps/SLO signal without altering the forwarded frame:
            # count tool calls, capture provider token usage off the terminal
            # ``done`` (``None`` when the provider reports none -- never invented).
            if event_name == "tool_call":
                metrics = metrics.observe_tool_call()
            elif event_name == "done":
                metrics = metrics.observe_usage(payload.get("usage"))
            elif event_name == "text_delta":
                # Accumulate the answer to persist the assistant turn (US-080).
                answer_parts.append(str(payload.get("text", "")))
            yield _sse_event(event_name, payload)

        # Persist the assistant's final answer (US-080). Best-effort: never break
        # the (already-finished) stream on a storage error.
        answer = "".join(answer_parts).strip()
        if answer:
            try:
                await SessionService.save_message(
                    session_id,
                    "assistant",
                    answer,
                    extra={"model": model, "variant": variant},
                )
            except Exception:  # noqa: BLE001
                logger.warning("chat_assistant_message_persist_failed", session_id=str(session_id))

        duration_ms = round((time.perf_counter() - start) * 1000.0, 2)
        emit_chat_turn_metrics(
            metrics.finalise(duration_ms),
            session_id=str(session_id),
            settings=self._settings,
        )
