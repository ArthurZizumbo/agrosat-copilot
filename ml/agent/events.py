"""Stream event types emitted by the agent's function-calling loop.

The agent (``ml/agent/agent.py``, US-047) drives a manual function-calling loop
and yields a sequence of these typed events. ``ChatService`` (US-046) consumes
them and serialises each one to a Server-Sent Event. Every event carries a
``type`` literal so the discriminated :data:`AgentEvent` union round-trips
cleanly to JSON and back (``model_dump`` keeps the tag, the union re-parses it).

Design notes:
    * Models are immutable (``frozen=True``) and reject unknown keys
      (``extra="forbid"``) so a malformed event never leaks to the SSE wire.
    * The ``type`` field is the SSE discriminant; it is the first key of every
      payload and never overlaps between event classes.
    * Payloads only ever hold JSON-serialisable scalars / mappings -- no tensors,
      no Pydantic models of the tool layer -- so the wire stays text-only.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AgentEvent",
    "DoneEvent",
    "ErrorEvent",
    "PerceiverObservationEvent",
    "TextDeltaEvent",
    "ToolCallEvent",
    "ToolResultEvent",
]

# Shared config: events are immutable snapshots and reject typos / unknown keys.
_EVENT_CONFIG = ConfigDict(frozen=True, extra="forbid")


class ToolCallEvent(BaseModel):
    """The reasoner decided to call a tool (one event per requested call).

    Emitted before the tool runs, so the UI can show "calling <tool>" with the
    arguments the model produced.

    Attributes:
        type: Discriminant literal ``"tool_call"``.
        name: Registered tool name the model wants to invoke.
        arguments: Validated tool arguments (already parsed by the tool's
            Pydantic ``*Input`` model and dumped back to a plain mapping).
        call_id: Provider-supplied call identifier when available (Gemini omits
            it; OpenAI-compatible backends supply it), used to correlate the
            result; ``None`` when the backend does not provide one.
    """

    model_config = _EVENT_CONFIG

    type: Literal["tool_call"] = "tool_call"
    name: str
    arguments: dict
    call_id: str | None = None


class ToolResultEvent(BaseModel):
    """A tool finished and returned a result (or failed in a controlled way).

    Attributes:
        type: Discriminant literal ``"tool_result"``.
        name: Tool name that produced the result.
        result: The tool's output model dumped to a JSON-serialisable mapping;
            on failure, a ``{"error": ...}`` mapping describing the controlled
            failure.
        ok: ``True`` when the tool ran to completion, ``False`` when it failed
            (validation error, exception) and ``result`` carries the error.
    """

    model_config = _EVENT_CONFIG

    type: Literal["tool_result"] = "tool_result"
    name: str
    result: dict
    ok: bool = True


class TextDeltaEvent(BaseModel):
    """An incremental chunk of the reasoner's final natural-language answer.

    Attributes:
        type: Discriminant literal ``"text_delta"``.
        text: The next piece of streamed answer text (may be a single token or a
            larger span depending on the backend).
    """

    model_config = _EVENT_CONFIG

    type: Literal["text_delta"] = "text_delta"
    text: str


class PerceiverObservationEvent(BaseModel):
    """The perceiver's initial TEXT observation injected before reasoning.

    Carries the ``PerceiverObservation.to_prompt_block()`` rendering so the UI
    can surface what the agent "sees" (crop, phenology, vigor) before the
    reasoner speaks. It is plain text -- never logits/tensors -- honouring the
    Be My Eyes contract.

    Attributes:
        type: Discriminant literal ``"perceiver_observation"``.
        text: The rendered grounding block from the perceiver.
    """

    model_config = _EVENT_CONFIG

    type: Literal["perceiver_observation"] = "perceiver_observation"
    text: str


class DoneEvent(BaseModel):
    """Terminal event signalling the loop produced its final answer.

    Attributes:
        type: Discriminant literal ``"done"``.
    """

    model_config = _EVENT_CONFIG

    type: Literal["done"] = "done"


class ErrorEvent(BaseModel):
    """Terminal event signalling the loop aborted with an error.

    Emitted instead of :class:`DoneEvent` when the agent cannot recover (backend
    failure, unrecoverable loop state). The message is a human-readable summary
    safe to surface; it never carries secrets or stack traces.

    Attributes:
        type: Discriminant literal ``"error"``.
        message: Human-readable error summary for the client.
    """

    model_config = _EVENT_CONFIG

    type: Literal["error"] = "error"
    message: str


# Discriminated union of every event the agent loop may yield. ``type`` is the
# tag, so ``AgentEvent`` parsers (e.g. ``TypeAdapter(AgentEvent)``) reconstruct
# the right subclass from a serialised SSE payload.
AgentEvent = Annotated[
    ToolCallEvent
    | ToolResultEvent
    | TextDeltaEvent
    | PerceiverObservationEvent
    | DoneEvent
    | ErrorEvent,
    Field(discriminator="type"),
]
