"""Vision agent — the "eyes" of the Be My Eyes pattern (ADR-011 section 1).

The orchestrator (the "brain") never looks at the raster; it delegates
perception to this agent, which runs the trained ML stack over the scene and
returns structured :class:`Finding` objects with citations. The interface is
deliberately stable so a future ``describe_scene_vlm`` (Gemma / Qwen-VL) sub-tool
plugs in WITHOUT changing the orchestrator or this signature.

MVP logic is heuristic and REAL (no LLM): it always classifies the AOI's
parcels and, when the question mentions health / NDVI / stress, it also computes
NDVI. Each sub-tool invocation produces a paired ``ToolCall`` + ``ToolResult``
(agent ``"vision"``) and the findings' citations are stamped with the real
``call_id`` so the wire-level provenance link is authoritative.
"""

from __future__ import annotations

import time
import uuid

import structlog

from ml.agent.events import Finding, ToolCall, ToolResult
from ml.agent.ports import AgentDeps
from ml.agent.tools.classify_parcel import ClassifyParcelInput, classify_parcel
from ml.agent.tools.compute_ndvi import ComputeNdviInput, compute_ndvi

logger = structlog.get_logger(__name__)

# Trigger words (it/es/en) that make the vision agent also compute NDVI.
_NDVI_TRIGGERS: tuple[str, ...] = (
    "ndvi",
    "salud",
    "saludable",
    "estres",
    "estrés",
    "vigor",
    "vegetacion",
    "vegetación",
    "health",
    "stress",
    "vegetation",
    "salute",
    "stress idrico",
    "vegetazione",
)


class VisionResult:
    """Bundle returned by :func:`analyze`.

    Attributes:
        findings: All findings produced across the sub-tools (citations already
            carry the real ``call_id``).
        tool_calls: The ``ToolCall`` events emitted, in order.
        tool_results: The ``ToolResult`` events emitted, in order (paired by
            ``call_id`` with ``tool_calls``).
    """

    __slots__ = ("findings", "tool_calls", "tool_results")

    def __init__(
        self,
        findings: list[Finding],
        tool_calls: list[ToolCall],
        tool_results: list[ToolResult],
    ) -> None:
        self.findings = findings
        self.tool_calls = tool_calls
        self.tool_results = tool_results


def _wants_ndvi(question: str) -> bool:
    """Heuristic: does the question ask about plant health / NDVI?"""
    lowered = question.lower()
    return any(trigger in lowered for trigger in _NDVI_TRIGGERS)


def _stamp(findings: list[Finding], call_id: str) -> list[Finding]:
    """Stamp the wire ``call_id`` into each finding's citation."""
    stamped: list[Finding] = []
    for finding in findings:
        citation = finding.citation.model_copy(update={"tool_call_id": call_id})
        stamped.append(finding.model_copy(update={"citation": citation}))
    return stamped


async def analyze(
    *,
    session_id: str,
    aoi_id: int | None,
    question: str,
    deps: AgentDeps,
    year: int | None = None,
) -> VisionResult:
    """Run the perception sub-tools over the AOI and return findings + events.

    Contract (stable across future VLM sub-tools):
        - Always classifies the AOI's parcels (``classify_parcel``).
        - If ``question`` mentions health/NDVI/stress, also runs ``compute_ndvi``.
        - Emits one ``ToolCall`` + one ``ToolResult`` per sub-tool, with
          ``agent="vision"`` and a unique ``call_id`` per pair.

    Args:
        session_id: Tenant scope (multi-tenant NON-NEGOTIABLE).
        aoi_id: AOI to analyse (None = all session parcels).
        question: The user's natural-language question (drives tool selection).
        deps: Injected ports (``parcels``, ``memory``).
        year: Optional feature/label year filter.

    Returns:
        A :class:`VisionResult` with findings, tool calls and tool results.
    """
    tool_calls: list[ToolCall] = []
    tool_results: list[ToolResult] = []
    all_findings: list[Finding] = []

    # --- 1. Always classify the AOI's parcels. -----------------------------
    classify_call_id = f"c-{uuid.uuid4().hex[:8]}"
    classify_args: dict[str, object] = {"aoi_id": aoi_id, "year": year}
    tool_calls.append(
        ToolCall(
            call_id=classify_call_id,
            tool="classify_parcel",
            args=classify_args,
            agent="vision",
        )
    )
    started = time.perf_counter()
    logger.info("tool_call_started", tool="classify_parcel", call_id=classify_call_id)
    classify_out = await classify_parcel(
        ClassifyParcelInput(session_id=session_id, aoi_id=aoi_id, year=year),
        parcels=deps.parcels,
    )
    duration_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "tool_call_finished",
        tool="classify_parcel",
        call_id=classify_call_id,
        duration_ms=duration_ms,
        n_findings=len(classify_out.findings),
    )
    classify_findings = _stamp(classify_out.findings, classify_call_id)
    all_findings.extend(classify_findings)
    tool_results.append(
        ToolResult(
            call_id=classify_call_id,
            tool="classify_parcel",
            ok=True,
            summary=classify_out.summary,
            duration_ms=duration_ms,
            data={"used_model": classify_out.used_model},
            findings=classify_findings,
        )
    )

    # --- 2. Conditionally compute NDVI when health is asked about. ---------
    if _wants_ndvi(question):
        ndvi_call_id = f"c-{uuid.uuid4().hex[:8]}"
        ndvi_args: dict[str, object] = {"aoi_id": aoi_id, "year": year}
        tool_calls.append(
            ToolCall(
                call_id=ndvi_call_id,
                tool="compute_ndvi",
                args=ndvi_args,
                agent="vision",
            )
        )
        started = time.perf_counter()
        logger.info("tool_call_started", tool="compute_ndvi", call_id=ndvi_call_id)
        ndvi_out = await compute_ndvi(
            ComputeNdviInput(session_id=session_id, aoi_id=aoi_id, year=year),
            parcels=deps.parcels,
        )
        duration_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "tool_call_finished",
            tool="compute_ndvi",
            call_id=ndvi_call_id,
            duration_ms=duration_ms,
            n_findings=len(ndvi_out.findings),
        )
        ndvi_findings = _stamp(ndvi_out.findings, ndvi_call_id)
        all_findings.extend(ndvi_findings)
        tool_results.append(
            ToolResult(
                call_id=ndvi_call_id,
                tool="compute_ndvi",
                ok=True,
                summary=ndvi_out.summary,
                duration_ms=duration_ms,
                findings=ndvi_findings,
            )
        )

    return VisionResult(all_findings, tool_calls, tool_results)


__all__ = ["VisionResult", "analyze"]
