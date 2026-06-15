"""Agent event contract shared by the agent core and the backend transport.

The orchestrator emits a stream of :class:`AgentEvent` objects (a discriminated
union on the ``type`` field). The backend serialises each event to JSON and
pushes it to the client over WebSocket (or SSE as a fallback). The frontend
renders them progressively. Keeping this contract in one place guarantees the
three layers (agent, backend, frontend) agree on the wire format.

Design notes:
- Every numeric claim surfaced to the user must trace back to a ``tool_result``
  (NON-NEGOTIABLE: no figure without a tool call). ``Citation`` enforces that
  link at the type level.
- The union is ordered by a typical turn: ``plan_created`` -> N x
  (``tool_call`` -> ``tool_result``) -> optional ``token`` stream ->
  ``final_answer`` -> ``done`` (or ``error``).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Value objects.
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """Provenance of a figure or claim in the final answer.

    At least one identifier (``scene_id``, ``parcel_id`` or ``aoi_id``) plus the
    originating ``tool_call_id`` must be present so the claim is auditable.
    """

    tool_call_id: str = Field(description="Id of the tool_call that produced the data.")
    source: str = Field(description="Human-readable origin, e.g. 'XGBoost+AlphaEarth'.")
    scene_id: str | None = None
    parcel_id: int | None = None
    aoi_id: int | None = None
    dates: list[str] | None = Field(default=None, description="ISO dates backing the claim.")


class Finding(BaseModel):
    """A single structured observation produced by the vision agent."""

    parcel_id: int
    crop_class: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    area_ha: float | None = None
    ndvi_mean: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    geometry: dict[str, object] | None = Field(
        default=None, description="Parcel boundary as a GeoJSON Polygon (for map rendering)."
    )
    citation: Citation


# ---------------------------------------------------------------------------
# Event union.
# ---------------------------------------------------------------------------


class PlanCreated(BaseModel):
    """The orchestrator published its plan before acting."""

    type: Literal["plan_created"] = "plan_created"
    steps: list[str]


class ToolCall(BaseModel):
    """The orchestrator (or the vision agent) is invoking a tool."""

    type: Literal["tool_call"] = "tool_call"
    call_id: str
    tool: str
    args: dict[str, object] = Field(default_factory=dict)
    agent: Literal["orchestrator", "vision"] = "orchestrator"


class ToolResult(BaseModel):
    """A tool finished; ``summary`` is safe to show, ``data`` is structured."""

    type: Literal["tool_result"] = "tool_result"
    call_id: str
    tool: str
    ok: bool
    summary: str
    duration_ms: int
    data: dict[str, object] = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)


class Token(BaseModel):
    """A partial chunk of the final answer (optional token streaming)."""

    type: Literal["token"] = "token"
    text: str


class FinalAnswer(BaseModel):
    """The orchestrator's final natural-language answer with citations."""

    type: Literal["final_answer"] = "final_answer"
    text: str
    citations: list[Citation] = Field(default_factory=list)


class AgentError(BaseModel):
    """A recoverable error surfaced to the client."""

    type: Literal["error"] = "error"
    code: str
    message: str


class Done(BaseModel):
    """Terminal event: the job finished (success or after an error)."""

    type: Literal["done"] = "done"
    job_id: str


AgentEvent = Annotated[
    PlanCreated | ToolCall | ToolResult | Token | FinalAnswer | AgentError | Done,
    Field(discriminator="type"),
]
"""Discriminated union of every event the orchestrator can emit."""


__all__ = [
    "AgentError",
    "AgentEvent",
    "Citation",
    "Done",
    "FinalAnswer",
    "Finding",
    "PlanCreated",
    "Token",
    "ToolCall",
    "ToolResult",
]
