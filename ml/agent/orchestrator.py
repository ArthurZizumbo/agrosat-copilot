"""Orchestrator — the "brain" of the Be My Eyes pattern (ADR-011).

Entry-point ``run_chat`` is the single contract the backend consumes: it streams
:class:`AgentEvent` objects and the backend forwards them verbatim over
WebSocket/SSE. The backend builds NO prompts and calls NO LLM.

Plan-and-React loop for the MVP:
    1. Emit ``PlanCreated`` (deterministic plan for the single-AOI question).
    2. Delegate perception to :func:`ml.agent.vision_agent.analyze`, re-emitting
       its ``ToolCall`` / ``ToolResult`` events.
    3. Synthesise the answer with the selected LLM backend; if the LLM is
       unavailable in the environment, fall back to a deterministic template
       (clearly marked) built from the findings.
    4. Emit ``FinalAnswer`` whose citations come from the findings, then ``Done``.
    5. On any exception emit ``AgentError`` + ``Done`` (never crash the stream).

Both the user turn and the assistant turn are persisted via ``deps.memory``.
Tool timing is logged with structlog ``tool_call_started`` / ``tool_call_finished``
inside the vision agent; the orchestrator logs the synthesis step.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import Literal

import structlog

from ml.agent.backends import BackendVariant, get_backend
from ml.agent.events import (
    AgentError,
    AgentEvent,
    Citation,
    Done,
    FinalAnswer,
    Finding,
    PlanCreated,
)
from ml.agent.memory import history_to_messages
from ml.agent.ports import AgentDeps, ChatTurn
from ml.agent.vision_agent import analyze as vision_analyze

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are AgroSatCopilot, an agronomy assistant that answers questions about "
    "agricultural parcels using ONLY the structured findings provided by the "
    "vision agent. Never invent figures (hectares, NDVI, crop classes); every "
    "number must come from a finding. Answer in the user's language (it/es/en). "
    "Be concise and cite the parcels you reference."
)


def _extract_aoi_id(deps: AgentDeps, user_message: str) -> int | None:
    """Best-effort AOI id from the message (MVP: None = all session parcels).

    The frontend passes the active AOI explicitly in later iterations; for the
    MVP the orchestrator scopes by session and lets the reader return all
    parcels when no AOI is pinned. Kept as a hook so callers can override.
    """
    return None


def _plan_steps(user_message: str) -> list[str]:
    """Deterministic plan shown to the user before acting."""
    steps = [
        "Listar parcelas del AOI de la sesion",
        "Clasificar el cultivo de cada parcela",
    ]
    lowered = user_message.lower()
    if any(k in lowered for k in ("ndvi", "salud", "estres", "estrés", "health", "stress")):
        steps.append("Calcular metricas NDVI por parcela")
    steps.append("Sintetizar la respuesta con citas")
    return steps


def _citations_from_findings(findings: list[Finding]) -> list[Citation]:
    """Collect the citations backing every finding (deduplicated by key)."""
    seen: set[tuple[str, int | None, int | None]] = set()
    citations: list[Citation] = []
    for finding in findings:
        cit = finding.citation
        key = (cit.tool_call_id, cit.parcel_id, cit.aoi_id)
        if key in seen:
            continue
        seen.add(key)
        citations.append(cit)
    return citations


def _template_answer(findings: list[Finding]) -> str:
    """Deterministic answer built from findings when the LLM is unavailable.

    Clearly a fallback (no LLM prose). Reports crop classes and, when present,
    NDVI means — every figure traces back to a finding.
    """
    if not findings:
        return (
            "No se encontraron parcelas en el AOI de la sesion para analizar. "
            "[respuesta determinista sin LLM]"
        )
    crop_findings = [f for f in findings if f.crop_class is not None]
    ndvi_findings = [f for f in findings if f.ndvi_mean is not None]
    parts: list[str] = []
    n_parcels = len({f.parcel_id for f in findings})
    parts.append(f"El AOI tiene {n_parcels} parcela(s).")
    for f in crop_findings:
        conf = (
            f" (confianza {f.confidence:.2f})"
            if f.confidence is not None and f.confidence == f.confidence  # not NaN
            else ""
        )
        area = f", {f.area_ha:.1f} ha" if f.area_ha is not None else ""
        parts.append(f"Parcela {f.parcel_id}: cultivo {f.crop_class}{conf}{area}.")
    for f in ndvi_findings:
        parts.append(f"Parcela {f.parcel_id}: NDVI medio {f.ndvi_mean:.3f}.")
    parts.append("[respuesta determinista sin LLM]")
    return " ".join(parts)


def _findings_context(findings: list[Finding]) -> str:
    """Serialise findings as compact JSON-ish text for the LLM synthesis prompt."""
    lines = ["Findings disponibles (usa solo estas cifras):"]
    for f in findings:
        bits = [f"parcel_id={f.parcel_id}"]
        if f.crop_class is not None:
            bits.append(f"crop={f.crop_class}")
        if f.confidence is not None and f.confidence == f.confidence:
            bits.append(f"conf={f.confidence:.2f}")
        if f.area_ha is not None:
            bits.append(f"area_ha={f.area_ha:.2f}")
        if f.ndvi_mean is not None:
            bits.append(f"ndvi_mean={f.ndvi_mean:.3f}")
        lines.append("- " + ", ".join(bits))
    return "\n".join(lines)


async def run_chat(
    *,
    session_id: str,
    user_message: str,
    llm_variant: Literal["gemini", "qwen35"],
    deps: AgentDeps,
    aoi_id: int | None = None,
) -> AsyncIterator[AgentEvent]:
    """Run one conversational turn, streaming :class:`AgentEvent` objects.

    See :mod:`ml.agent.orchestrator` module docstring for the event sequence.
    The job id (carried in the terminal ``Done``) is generated here so single
    turns are self-contained; the backend may override it upstream.

    Args:
        session_id: Tenant scope (multi-tenant NON-NEGOTIABLE).
        user_message: The user's natural-language message.
        llm_variant: ``"gemini"`` (Vertex AI) or ``"qwen35"`` (vLLM).
        deps: Injected ports (``parcels``, ``memory``).

    Yields:
        ``PlanCreated`` -> (``ToolCall`` -> ``ToolResult``)* -> ``FinalAnswer``
        -> ``Done`` on success, or ``AgentError`` -> ``Done`` on failure.
    """
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    log = logger.bind(session_id=session_id, job_id=job_id, llm_variant=llm_variant)

    # Persist the user turn up front so it is recorded even if synthesis fails.
    try:
        await deps.memory.append_turn(
            session_id=session_id, turn=ChatTurn(role="user", content=user_message)
        )
    except Exception as exc:  # noqa: BLE001 - memory must not break the stream
        log.warning("memory_append_user_failed", error=str(exc))

    try:
        # --- 1. Plan. ------------------------------------------------------
        steps = _plan_steps(user_message)
        yield PlanCreated(steps=steps)

        # --- 2. Delegate perception to the vision agent. -------------------
        # Prefer the explicit AOI selected on the map; fall back to extraction.
        resolved_aoi_id = aoi_id if aoi_id is not None else _extract_aoi_id(deps, user_message)
        vision = await vision_analyze(
            session_id=session_id,
            aoi_id=resolved_aoi_id,
            question=user_message,
            deps=deps,
        )
        for call in vision.tool_calls:
            yield call
        for result in vision.tool_results:
            yield result

        # --- 3. Synthesise the final answer. -------------------------------
        answer_text = await _synthesise(
            llm_variant=llm_variant,
            session_id=session_id,
            user_message=user_message,
            findings=vision.findings,
            deps=deps,
            log=log,
        )
        citations = _citations_from_findings(vision.findings)

        # --- 4. Final answer + persist assistant turn. ---------------------
        yield FinalAnswer(text=answer_text, citations=citations)
        try:
            await deps.memory.append_turn(
                session_id=session_id,
                turn=ChatTurn(role="assistant", content=answer_text),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("memory_append_assistant_failed", error=str(exc))

        yield Done(job_id=job_id)

    except Exception as exc:  # surface any failure as an event, never crash stream
        log.error("run_chat_failed", error=str(exc), exc_info=True)
        yield AgentError(code="agent_error", message=str(exc))
        yield Done(job_id=job_id)


async def _synthesise(
    *,
    llm_variant: BackendVariant,
    session_id: str,
    user_message: str,
    findings: list[Finding],
    deps: AgentDeps,
    log: structlog.BoundLogger,
) -> str:
    """Produce the final answer text, falling back to a template if no LLM.

    Loads recent history, builds the prompt and calls the backend. Any backend
    failure (no SDK, unreachable endpoint, auth) degrades to the deterministic
    template so the turn still completes with cited figures.
    """
    try:
        history = await deps.memory.load_history(session_id=session_id, limit=20)
    except Exception as exc:  # noqa: BLE001
        log.warning("memory_load_history_failed", error=str(exc))
        history = []

    messages = history_to_messages(history, system_prompt=_SYSTEM_PROMPT)
    # The latest user turn was already persisted; append the findings context so
    # the LLM grounds its answer strictly on tool outputs.
    from ml.agent.backends import ChatMessage

    messages.append(
        ChatMessage(
            role="user",
            content=f"{user_message}\n\n{_findings_context(findings)}",
        )
    )

    started = time.perf_counter()
    log.info("synthesis_started", n_findings=len(findings))
    try:
        backend = get_backend(llm_variant)
        result = await backend.generate(messages=messages)
        duration_ms = int((time.perf_counter() - started) * 1000)
        if not result.text:
            raise ValueError("LLM returned empty text")
        log.info("synthesis_finished", duration_ms=duration_ms, model=result.model)
        return result.text
    except Exception as exc:  # noqa: BLE001 - any LLM failure -> template
        duration_ms = int((time.perf_counter() - started) * 1000)
        log.warning(
            "synthesis_fallback_template",
            duration_ms=duration_ms,
            error=str(exc),
        )
        return _template_answer(findings)


__all__ = ["run_chat"]
