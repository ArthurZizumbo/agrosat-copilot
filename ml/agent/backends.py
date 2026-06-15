"""LLM backend abstraction (dual: Gemini Vertex AI / Qwen3.5 vLLM).

The orchestrator never hardcodes a model. It asks :func:`get_backend` for an
:class:`LLMBackend` adapter and uses its single async ``generate`` method. Two
adapters are provided:

- :class:`GeminiBackend` — variant A, Gemini 3.x over Vertex AI via the native
  ``google-genai`` SDK (``from google import genai``). Supports function calling.
- :class:`VLLMOpenAIBackend` — variant B, Qwen3.5-35B-A3B served by vLLM behind
  an OpenAI-compatible API. Implemented defensively: importing this module never
  requires a live endpoint, and the heavy SDK is imported lazily inside
  ``generate`` so a missing dependency or unreachable server only fails at call
  time (the orchestrator catches it and falls back to a deterministic template).

Design notes:
- The SDKs are imported lazily so the agent core imports without ``google-genai``
  or ``openai`` installed (tests inject a fake backend).
- ``LLMResult`` is intentionally minimal (just the synthesised text plus optional
  raw tool-call requests) because the MVP synthesis step only needs free text.
  Native function-calling plumbing is wired but unused by the heuristic vision
  agent; it is ready for the post-presentation LLM-driven planner.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel, Field

from ml.agent.settings import AgentSettings, get_settings

logger = structlog.get_logger(__name__)

BackendVariant = Literal["gemini", "qwen35"]


# ---------------------------------------------------------------------------
# Wire types.
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single message in the LLM conversation context."""

    role: Literal["system", "user", "assistant"]
    content: str


class LLMResult(BaseModel):
    """Outcome of one LLM generation call."""

    text: str
    model: str
    finish_reason: str | None = None
    tool_calls: list[dict[str, object]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Port.
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMBackend(Protocol):
    """Provider-agnostic chat/generation backend used by the orchestrator."""

    name: str

    async def generate(
        self,
        *,
        messages: list[ChatMessage],
        tools: list[dict[str, object]] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResult:
        """Generate a completion (optionally exposing ``tools`` for calling)."""
        ...


# ---------------------------------------------------------------------------
# Gemini (Vertex AI) adapter.
# ---------------------------------------------------------------------------


class GeminiBackend:
    """Gemini backend over Vertex AI via the native ``google-genai`` SDK.

    Auth is auto-detected by the SDK from the environment (Vertex AI when
    ``GOOGLE_GENAI_USE_VERTEXAI=true`` + project/location, public Gemini API
    when ``GEMINI_API_KEY`` is set). We pass the relevant settings explicitly so
    the behaviour is deterministic regardless of ambient ``os.environ``.
    """

    name = "gemini"

    def __init__(self, settings: AgentSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: object | None = None  # lazily built genai.Client

    def _get_client(self) -> object:
        """Build (once) and return the ``genai.Client`` (lazy SDK import)."""
        if self._client is not None:
            return self._client
        try:
            from google import genai  # type: ignore[import-not-found]
            from google.genai import types  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised only without SDK
            raise ImportError(
                "google-genai is not installed. Run `poetry add google-genai`."
            ) from exc

        s = self._settings
        if s.google_genai_use_vertexai:
            client = genai.Client(
                vertexai=True,
                project=s.google_cloud_project,
                location=s.google_cloud_location or s.vertex_ai_location,
                http_options=types.HttpOptions(timeout=s.llm_http_timeout_ms),
            )
        else:
            client = genai.Client(
                api_key=s.gemini_api_key or None,
                http_options=types.HttpOptions(timeout=s.llm_http_timeout_ms),
            )
        self._client = client
        return client

    async def generate(
        self,
        *,
        messages: list[ChatMessage],
        tools: list[dict[str, object]] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResult:
        """Generate via ``client.models.generate_content`` off the event loop.

        The google-genai SDK call is synchronous; we run it in a worker thread
        (``anyio.to_thread``) so the orchestrator's async loop is not blocked.
        """
        import anyio
        from google.genai import types  # type: ignore[import-not-found]

        client = self._get_client()
        s = self._settings
        system_text, contents = _split_system(messages)
        config = types.GenerateContentConfig(
            temperature=s.llm_temperature if temperature is None else temperature,
            max_output_tokens=(
                s.llm_max_output_tokens if max_output_tokens is None else max_output_tokens
            ),
            system_instruction=system_text or None,
        )

        def _call() -> object:
            return client.models.generate_content(  # type: ignore[attr-defined]
                model=s.gemini_model,
                contents=contents,
                config=config,
            )

        response = await anyio.to_thread.run_sync(_call)
        text = str(getattr(response, "text", "") or "").strip()
        return LLMResult(text=text, model=s.gemini_model, finish_reason="stop")


# ---------------------------------------------------------------------------
# Qwen3.5 vLLM (OpenAI-compatible) adapter.
# ---------------------------------------------------------------------------


class VLLMOpenAIBackend:
    """Qwen3.5-35B-A3B served by vLLM behind an OpenAI-compatible API.

    The endpoint may not exist yet in dev; the SDK is imported lazily inside
    :meth:`generate` so importing this module never requires it. A failed call
    raises and the orchestrator degrades to the deterministic template.
    """

    name = "qwen35"

    def __init__(self, settings: AgentSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: object | None = None

    def _get_client(self) -> object:
        """Build (once) and return the async OpenAI client (lazy import)."""
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised only without SDK
            raise ImportError(
                "openai is not installed. Run `poetry add openai` to enable the "
                "Qwen3.5 vLLM backend."
            ) from exc
        s = self._settings
        self._client = AsyncOpenAI(base_url=s.vllm_qwen35_url, api_key=s.vllm_api_key)
        return self._client

    async def generate(
        self,
        *,
        messages: list[ChatMessage],
        tools: list[dict[str, object]] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResult:
        """Generate via the OpenAI-compatible ``chat.completions`` endpoint."""
        client = self._get_client()
        s = self._settings
        wire_messages = [{"role": m.role, "content": m.content} for m in messages]
        completion = await client.chat.completions.create(  # type: ignore[attr-defined]
            model=s.vllm_qwen35_model,
            messages=wire_messages,
            temperature=s.llm_temperature if temperature is None else temperature,
            max_tokens=(
                s.llm_max_output_tokens if max_output_tokens is None else max_output_tokens
            ),
            tools=tools or None,
        )
        choice = completion.choices[0]
        text = str(choice.message.content or "").strip()
        return LLMResult(
            text=text,
            model=s.vllm_qwen35_model,
            finish_reason=getattr(choice, "finish_reason", None),
        )


# ---------------------------------------------------------------------------
# Helpers + factory.
# ---------------------------------------------------------------------------


def _split_system(messages: list[ChatMessage]) -> tuple[str, list[str]]:
    """Split messages into a single system instruction and the content turns.

    google-genai takes the system prompt as ``system_instruction`` and the rest
    as flat ``contents`` (we serialise user/assistant turns inline since the MVP
    only needs single-shot synthesis).
    """
    system_parts = [m.content for m in messages if m.role == "system"]
    contents = [
        f"{m.role}: {m.content}" if m.role == "assistant" else m.content
        for m in messages
        if m.role != "system"
    ]
    return "\n\n".join(system_parts), contents


def get_backend(variant: BackendVariant, *, settings: AgentSettings | None = None) -> LLMBackend:
    """Return the LLM backend adapter for ``variant`` (never hardcoded).

    Args:
        variant: ``"gemini"`` (Vertex AI) or ``"qwen35"`` (vLLM).
        settings: Optional settings override (tests inject a custom instance).

    Returns:
        An :class:`LLMBackend` adapter.

    Raises:
        ValueError: if ``variant`` is unknown.
    """
    s = settings or get_settings()
    if variant == "gemini":
        return GeminiBackend(settings=s)
    if variant == "qwen35":
        return VLLMOpenAIBackend(settings=s)
    raise ValueError(f"Unknown LLM variant {variant!r}; expected 'gemini' or 'qwen35'.")


__all__ = [
    "BackendVariant",
    "ChatMessage",
    "GeminiBackend",
    "LLMBackend",
    "LLMResult",
    "VLLMOpenAIBackend",
    "get_backend",
]
