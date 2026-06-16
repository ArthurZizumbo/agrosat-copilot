"""LLM backend abstraction for the conversational agent (US-047).

The agent loop in :mod:`ml.agent.agent` is backend-agnostic: it drives the
function-calling turns through a single coroutine,
:meth:`LLMBackend.generate_stream`, that yields duck-typed
:class:`BackendChunk` objects (a text delta and/or a requested function call).
Two concrete backends implement it:

- :class:`GeminiBackend` -- the default reasoner, Gemini 2.5 Pro on Vertex AI or
  the public GenAI API, via the ``google-genai`` SDK with automatic function
  calling DISABLED (the agent runs the tool loop itself so tools stay async and
  receive the tenant :class:`~ml.agent.context.ToolContext`).
- :class:`VLLMOpenAIBackend` -- the on-prem variant, Qwen3.5-35B-A3B served by
  vLLM behind an OpenAI-compatible ``/v1/chat/completions`` endpoint (US-048).

:func:`make_backend` selects the backend from the model name so ``/llm/switch``
swaps the reasoner without touching the agent loop (AC-5).

The OpenAI-compatible path uses the ``openai`` SDK (async client); both clients
are injectable so tests mock them with zero network calls.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog
from google import genai
from google.genai import types

if TYPE_CHECKING:  # pragma: no cover - typing only
    from backend.app.core.config import Settings

logger = structlog.get_logger(__name__)

__all__ = [
    "BackendChunk",
    "BackendFunctionCall",
    "GeminiBackend",
    "LLMBackend",
    "OllamaBackend",
    "VLLMOpenAIBackend",
    "make_backend",
]

#: Default Gemini reasoner model when the caller does not pin one.
_DEFAULT_GEMINI_MODEL: str = "gemini-2.5-pro"

#: Default Qwen model id served by vLLM (OpenAI-compatible). The on-prem serving
#: of US-048 publishes the model under this name; overridable via settings.
_DEFAULT_QWEN_MODEL: str = "qwen35"

#: Fallback vLLM endpoint when ``settings.vllm_qwen35_url`` is empty. Matches the
#: serving script default port of US-048.
_DEFAULT_VLLM_BASE_URL: str = "http://127.0.0.1:8002/v1"

#: Default Ollama OpenAI-compatible endpoint for the local Gemma variant. The
#: benchmark reaches the H100 Ollama through a local SSH forward (default :11435
#: to avoid clashing with a local Ollama on :11434).
_DEFAULT_OLLAMA_BASE_URL: str = "http://127.0.0.1:11435/v1"

#: Per-request timeout so a stalled call RAISES instead of hanging forever (a
#: dropped tunnel or a wedged socket otherwise blocks the eval/agent loop with no
#: recovery -- US-049 hardening). ``genai`` takes milliseconds; the OpenAI client
#: takes seconds. The on-prem (Gemma multimodal) value is generous on purpose.
_GENAI_TIMEOUT_MS: int = 120_000
_OPENAI_TIMEOUT_S: float = 180.0


@dataclass(frozen=True)
class BackendFunctionCall:
    """A function call requested by the model, normalised across backends.

    Attributes:
        name: Tool name the model wants to invoke.
        args: Argument mapping decoded from the model output (pre-validation).
        id: Provider-supplied call identifier when available (Gemini omits it;
            the OpenAI-compatible path carries it).
    """

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


@dataclass(frozen=True)
class BackendChunk:
    """One streamed unit from a backend: a text delta and/or a function call.

    The agent's ``_read_chunk`` reads ``text`` and ``function_call`` off this
    object (duck-typed), so any backend that yields chunks with these two
    attributes is compatible. Either attribute may be ``None`` on a given chunk.

    Attributes:
        text: Incremental text delta, or ``None`` when this chunk carries none.
        function_call: A requested call, or ``None`` when this chunk carries none.
    """

    text: str | None = None
    function_call: BackendFunctionCall | None = None


class LLMBackend(ABC):
    """Backend abstraction the agent loop drives.

    Concrete backends translate the provider-specific streaming/tool protocol
    into the common :class:`BackendChunk` stream. The agent never sees provider
    types beyond the ``google.genai.types`` it builds for ``contents`` (Gemini's
    content schema is reused as the neutral wire format; the vLLM backend maps it
    to OpenAI messages).
    """

    #: Model identifier, surfaced for logging (``agent_turn_started``).
    model: str

    @abstractmethod
    def generate_stream(
        self,
        *,
        contents: list[types.Content],
        tools: list[types.FunctionDeclaration],
        system_instruction: str,
    ) -> AsyncIterator[BackendChunk]:
        """Stream the model response for one turn.

        Args:
            contents: Conversation so far as ``google.genai.types.Content``.
            tools: Function declarations advertised to the model this turn.
            system_instruction: The system prompt (analyst "Be My Eyes").

        Yields:
            :class:`BackendChunk` values carrying text deltas and/or the
            function calls the model requests this turn.
        """
        raise NotImplementedError


class GeminiBackend(LLMBackend):
    """Gemini backend via ``google-genai`` with manual function calling.

    Automatic function calling is disabled so the agent owns the tool loop: the
    backend just advertises the declarations and surfaces the model's requested
    calls. The client is injectable for tests; in production it is built from the
    ambient credentials (Vertex AI or ``GEMINI_API_KEY``), mirroring
    :mod:`ml.features.phenology_description`.
    """

    def __init__(
        self,
        model: str = _DEFAULT_GEMINI_MODEL,
        *,
        api_key: str = "",
        use_vertexai: bool = False,
        project: str = "",
        location: str = "",
        client: genai.Client | None = None,
    ) -> None:
        """Initialise the Gemini backend.

        Args:
            model: Gemini model id (e.g. ``gemini-2.5-pro``/``gemini-2.5-flash``).
            api_key: Public GenAI API key. Passed explicitly because the project
                stores it in ``.env.local`` (read by ``Settings``), which is NOT
                exported to ``os.environ`` for the SDK to pick up automatically.
            use_vertexai: Use Vertex AI instead of the public API.
            project: GCP project id (Vertex AI mode).
            location: Vertex AI location.
            client: Optional pre-built ``genai.Client`` (injected in tests). When
                ``None`` a client is created lazily on first use.
        """
        self.model = model
        self._api_key = api_key
        self._use_vertexai = use_vertexai
        self._project = project
        self._location = location
        self._client = client

    def _get_client(self) -> genai.Client:
        """Return the ``genai.Client``, building it lazily from the configuration.

        Uses Vertex AI when ``use_vertexai`` is set (with project/location), or
        the public GenAI API with the explicit ``api_key``. Falls back to the
        ambient credentials only when neither was provided.

        Returns:
            The cached or newly created client.
        """
        if self._client is None:
            http_options = types.HttpOptions(timeout=_GENAI_TIMEOUT_MS)
            if self._use_vertexai:
                self._client = genai.Client(
                    vertexai=True,
                    project=self._project or None,
                    location=self._location or None,
                    http_options=http_options,
                )
            elif self._api_key:
                self._client = genai.Client(api_key=self._api_key, http_options=http_options)
            else:
                self._client = genai.Client(http_options=http_options)
        return self._client

    async def generate_stream(
        self,
        *,
        contents: list[types.Content],
        tools: list[types.FunctionDeclaration],
        system_instruction: str,
    ) -> AsyncIterator[BackendChunk]:
        """Stream Gemini output for one turn, surfacing text and function calls.

        Args:
            contents: Conversation contents accumulated so far.
            tools: Function declarations to advertise this turn.
            system_instruction: The analyst system prompt.

        Yields:
            :class:`BackendChunk` per streamed response chunk.
        """
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[types.Tool(function_declarations=tools)] if tools else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        client = self._get_client()
        # A turn is resolved with a SINGLE non-streaming generation: it is the
        # canonical surface for ``function_calls`` and also carries the answer
        # ``text``. Re-running a streaming generation to surface the text would be
        # a second full model call (double cost/latency) and could sample a
        # different completion than the one inspected for tool calls (B-4).
        response = await self._generate(client, contents, config)
        calls = list(getattr(response, "function_calls", None) or [])
        if calls:
            for call in calls:
                if getattr(call, "name", None):
                    yield BackendChunk(
                        function_call=BackendFunctionCall(
                            name=call.name,
                            args=dict(getattr(call, "args", None) or {}),
                            id=getattr(call, "id", None),
                        )
                    )
            return
        # No tool calls: re-emit the text of the response already obtained. This
        # trades token-by-token streaming for a single model call; the text the
        # caller sees is exactly the one inspected for tool calls.
        for chunk in self._chunks_from_response(response):
            yield chunk

    async def _generate(
        self,
        client: Any,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
    ) -> Any:
        """Run a non-streaming generation, preferring the async client surface.

        Production clients expose ``client.aio.models`` (no event-loop blocking);
        test doubles often expose only the synchronous ``client.models``. This
        adapter uses whichever is available.

        Args:
            client: The ``genai.Client`` (or a test double).
            contents: Conversation contents for this turn.
            config: The generation config (tools, system instruction).

        Returns:
            The provider response object (exposes ``function_calls`` / ``text``).
        """
        aio = getattr(client, "aio", None)
        if aio is not None:
            return await aio.models.generate_content(
                model=self.model, contents=contents, config=config
            )
        return client.models.generate_content(model=self.model, contents=contents, config=config)

    @staticmethod
    def _chunks_from_response(response: Any) -> list[BackendChunk]:
        """Split one Gemini stream response into neutral :class:`BackendChunk`.

        A response chunk may carry text parts and/or function-call parts. Each is
        emitted as its own :class:`BackendChunk` so the agent buffers text and
        collects calls independently.

        Args:
            response: One item yielded by ``generate_content_stream``.

        Returns:
            The chunks extracted from the response (possibly empty).
        """
        out: list[BackendChunk] = []
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                text = getattr(part, "text", None)
                if text:
                    out.append(BackendChunk(text=text))
                fc = getattr(part, "function_call", None)
                if fc is not None and getattr(fc, "name", None):
                    out.append(
                        BackendChunk(
                            function_call=BackendFunctionCall(
                                name=fc.name,
                                args=dict(getattr(fc, "args", None) or {}),
                                id=getattr(fc, "id", None),
                            )
                        )
                    )
        # Flat surfaces: the SDK and the test doubles also expose
        # ``response.function_calls`` and ``response.text`` directly. Fold these
        # in when the candidate parts did not already yield them.
        if not any(c.function_call for c in out):
            for fc in getattr(response, "function_calls", None) or []:
                if getattr(fc, "name", None):
                    out.append(
                        BackendChunk(
                            function_call=BackendFunctionCall(
                                name=fc.name,
                                args=dict(getattr(fc, "args", None) or {}),
                                id=getattr(fc, "id", None),
                            )
                        )
                    )
        if not out:
            text = getattr(response, "text", None)
            if text:
                out.append(BackendChunk(text=text))
        return out


class VLLMOpenAIBackend(LLMBackend):
    """On-prem Qwen backend via an OpenAI-compatible vLLM endpoint (US-048).

    Maps the neutral ``google.genai.types.Content`` history to OpenAI chat
    messages and the function declarations to OpenAI ``tools``, then streams the
    completion and surfaces text deltas and tool calls as :class:`BackendChunk`.
    The async OpenAI client is injectable for tests (no network).
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_VLLM_BASE_URL,
        model: str = _DEFAULT_QWEN_MODEL,
        *,
        api_key: str = "EMPTY",
        client: Any | None = None,
    ) -> None:
        """Initialise the vLLM OpenAI-compatible backend.

        Args:
            base_url: Base URL of the vLLM server (``.../v1``).
            model: Model id served by vLLM.
            api_key: API key (vLLM ignores it but the client requires a value).
            client: Optional pre-built async OpenAI client (injected in tests).
        """
        self.model = model
        self._base_url = base_url
        self._api_key = api_key or "EMPTY"
        self._client = client

    def _get_client(self) -> Any:
        """Return the async OpenAI client, building it lazily.

        Returns:
            The cached or newly created ``openai.AsyncOpenAI`` pointed at the
            vLLM endpoint.
        """
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                timeout=_OPENAI_TIMEOUT_S,
            )
        return self._client

    async def generate_stream(
        self,
        *,
        contents: list[types.Content],
        tools: list[types.FunctionDeclaration],
        system_instruction: str,
    ) -> AsyncIterator[BackendChunk]:
        """Stream Qwen output for one turn via the OpenAI-compatible API.

        Args:
            contents: Conversation contents accumulated so far.
            tools: Function declarations to advertise this turn.
            system_instruction: The analyst system prompt.

        Yields:
            :class:`BackendChunk` per streamed delta / tool call.
        """
        messages = self._messages_from_contents(contents, system_instruction)
        openai_tools = self._tools_from_declarations(tools)
        client = self._get_client()
        stream = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=openai_tools or None,
            stream=True,
        )
        # vLLM streams tool-call arguments incrementally; accumulate per index and
        # emit the call once the stream completes.
        pending: dict[int, dict[str, Any]] = {}
        async for event in stream:
            choices = getattr(event, "choices", None) or []
            for choice in choices:
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                text = getattr(delta, "content", None)
                if text:
                    yield BackendChunk(text=text)
                for tc in getattr(delta, "tool_calls", None) or []:
                    self._accumulate_tool_call(pending, tc)
        for call in self._finalise_tool_calls(pending):
            yield BackendChunk(function_call=call)

    @staticmethod
    def _messages_from_contents(
        contents: list[types.Content],
        system_instruction: str,
    ) -> list[dict[str, Any]]:
        """Map ``genai`` contents to OpenAI chat messages.

        ``user`` -> user, ``model`` -> assistant, ``tool`` parts (function
        responses) -> tool messages. Function-call parts on a model turn become
        ``tool_calls`` on the assistant message.

        Args:
            contents: The neutral conversation history.
            system_instruction: The system prompt prepended as a system message.

        Returns:
            OpenAI-format messages.
        """
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_instruction}]
        # The OpenAI/vLLM API requires every ``tool`` message to carry a
        # ``tool_call_id`` that matches the ``id`` of the assistant ``tool_calls``
        # entry it answers. ``genai`` function-call/response parts do not carry
        # ids, so synthesise stable per-(turn, index) ids on the assistant turn and
        # replay them, in order, onto the following tool messages (the agent loop
        # emits responses in the same order it emitted the calls).
        pending_call_ids: list[str] = []
        turn = 0
        for content in contents:
            role = getattr(content, "role", "user")
            parts = getattr(content, "parts", None) or []
            if role == "tool":
                response_idx = 0
                for part in parts:
                    fr = getattr(part, "function_response", None)
                    if fr is not None:
                        if response_idx < len(pending_call_ids):
                            call_id = pending_call_ids[response_idx]
                        else:
                            call_id = f"call_{turn}_{response_idx}"
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "name": getattr(fr, "name", ""),
                                "content": json.dumps(getattr(fr, "response", {})),
                            }
                        )
                        response_idx += 1
                pending_call_ids = []
                continue
            mapped_role = "assistant" if role == "model" else "user"
            text_bits: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    text_bits.append(text)
                fc = getattr(part, "function_call", None)
                if fc is not None and getattr(fc, "name", None):
                    # Unique per call so two invocations of the same tool in one
                    # turn do not collide on ``id`` (OpenAI requires uniqueness).
                    call_id = getattr(fc, "id", None) or f"call_{turn}_{len(tool_calls)}"
                    tool_calls.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": fc.name,
                                "arguments": json.dumps(dict(getattr(fc, "args", None) or {})),
                            },
                        }
                    )
            message: dict[str, Any] = {"role": mapped_role, "content": "".join(text_bits)}
            if tool_calls:
                message["tool_calls"] = tool_calls
                pending_call_ids = [tc["id"] for tc in tool_calls]
            messages.append(message)
            turn += 1
        return messages

    @staticmethod
    def _tools_from_declarations(
        tools: list[types.FunctionDeclaration],
    ) -> list[dict[str, Any]]:
        """Map ``genai`` function declarations to OpenAI ``tools`` schema.

        Args:
            tools: The function declarations advertised this turn.

        Returns:
            OpenAI-format tool definitions.
        """
        out: list[dict[str, Any]] = []
        for decl in tools:
            parameters = decl.parameters
            schema = parameters.model_dump(exclude_none=True) if parameters is not None else {}
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": decl.name,
                        "description": decl.description or "",
                        "parameters": schema,
                    },
                }
            )
        return out

    @staticmethod
    def _accumulate_tool_call(pending: dict[int, dict[str, Any]], tc: Any) -> None:
        """Accumulate a streamed OpenAI tool-call delta into ``pending``.

        Args:
            pending: Per-index accumulator of ``{id, name, arguments}``.
            tc: One ``tool_calls`` delta item from the stream.
        """
        index = getattr(tc, "index", 0) or 0
        slot = pending.setdefault(index, {"id": None, "name": None, "arguments": ""})
        if getattr(tc, "id", None):
            slot["id"] = tc.id
        fn = getattr(tc, "function", None)
        if fn is not None:
            if getattr(fn, "name", None):
                slot["name"] = fn.name
            if getattr(fn, "arguments", None):
                slot["arguments"] += fn.arguments

    @staticmethod
    def _finalise_tool_calls(
        pending: dict[int, dict[str, Any]],
    ) -> list[BackendFunctionCall]:
        """Turn accumulated tool-call slots into :class:`BackendFunctionCall`.

        Args:
            pending: The per-index accumulator built during streaming.

        Returns:
            The decoded function calls (slots with no name are dropped).
        """
        calls: list[BackendFunctionCall] = []
        for slot in pending.values():
            name = slot.get("name")
            if not name:
                continue
            raw_args = slot.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                logger.warning("vllm_tool_args_decode_failed", name=name, raw=raw_args[:200])
                args = {}
            calls.append(BackendFunctionCall(name=name, args=args, id=slot.get("id")))
        return calls


class OllamaBackend(VLLMOpenAIBackend):
    """On-prem multimodal backend served by Ollama (OpenAI-compatible).

    Used for the local Gemma variant (``gemma4:31b-it-q8_0`` on the H100), so the
    benchmark's Gemma runs at zero API cost. Differs from the plain vLLM backend:

    - Multimodal: image parts in ``contents`` are forwarded as OpenAI
      ``image_url`` data URIs (Gemma 4 is a VLM), so AgroMind images reach it.
    - ``max_tokens``: Gemma 4 is a *thinking* model that spends tokens on a
      ``reasoning`` trace before the answer; a generous budget ensures the final
      ``content`` is emitted (a small budget leaves ``content`` empty).
    - Fallback: if the API returns an empty ``content`` but a ``reasoning`` trace,
      the reasoning tail is used so the answer is never silently lost.
    """

    #: Generous output budget so Gemma's reasoning trace does not starve the
    #: final answer. Overridable per instance.
    _DEFAULT_MAX_TOKENS: int = 1024

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "EMPTY",
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        client: Any | None = None,
    ) -> None:
        """Initialise the Ollama backend.

        Args:
            base_url: Ollama OpenAI-compatible base URL (``.../v1``).
            model: Ollama model tag (e.g. ``gemma4:31b-it-q8_0``).
            api_key: Ignored by Ollama; the client requires a value.
            max_tokens: Output token budget per call.
            client: Optional pre-built async OpenAI client (tests).
        """
        super().__init__(base_url=base_url, model=model, api_key=api_key, client=client)
        self._max_tokens = max_tokens

    async def generate_stream(
        self,
        *,
        contents: list[types.Content],
        tools: list[types.FunctionDeclaration],
        system_instruction: str,
    ) -> AsyncIterator[BackendChunk]:
        """Stream Gemma output via Ollama, forwarding images and a token budget.

        Args:
            contents: Conversation contents (may carry image parts).
            tools: Function declarations. Accepted for interface parity but
                ignored by this multimodal benchmark backend (Gemma is used for
                direct-answer evaluation, not tool calling).
            system_instruction: System prompt.

        Yields:
            :class:`BackendChunk` text deltas (the benchmark ignores tool calls).
        """
        messages = self._messages_with_images(contents, system_instruction)
        client = self._get_client()
        stream = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self._max_tokens,
            stream=True,
        )
        produced = False
        reasoning_tail: list[str] = []
        async for event in stream:
            for choice in getattr(event, "choices", None) or []:
                delta = getattr(choice, "delta", None)
                if delta is None:
                    continue
                text = getattr(delta, "content", None)
                if text:
                    produced = True
                    yield BackendChunk(text=text)
                reasoning = getattr(delta, "reasoning", None)
                if reasoning:
                    reasoning_tail.append(reasoning)
        # Fallback: if the model only emitted a reasoning trace (no final content),
        # surface the tail so the answer is never lost (Gemma thinking models).
        if not produced and reasoning_tail:
            yield BackendChunk(text="".join(reasoning_tail)[-512:])

    @classmethod
    def _messages_with_images(
        cls,
        contents: list[types.Content],
        system_instruction: str,
    ) -> list[dict[str, Any]]:
        """Map contents to OpenAI messages, forwarding image parts as data URIs.

        Args:
            contents: The neutral conversation history (may carry image parts).
            system_instruction: System prompt (omitted when empty).

        Returns:
            OpenAI-format messages with multimodal user content blocks.
        """
        import base64

        messages: list[dict[str, Any]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        for content in contents:
            role = getattr(content, "role", "user")
            mapped_role = "assistant" if role == "model" else "user"
            blocks: list[dict[str, Any]] = []
            for part in getattr(content, "parts", None) or []:
                text = getattr(part, "text", None)
                if text:
                    blocks.append({"type": "text", "text": text})
                    continue
                inline = getattr(part, "inline_data", None)
                data = getattr(inline, "data", None) if inline is not None else None
                if data is not None:
                    mime = getattr(inline, "mime_type", "image/jpeg")
                    b64 = base64.b64encode(data).decode("ascii")
                    blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        }
                    )
            if not blocks:
                continue
            # Collapse to a plain string when there is a single text block (Ollama
            # accepts both forms; the string form is simpler for text-only turns).
            if len(blocks) == 1 and blocks[0]["type"] == "text":
                messages.append({"role": mapped_role, "content": blocks[0]["text"]})
            else:
                messages.append({"role": mapped_role, "content": blocks})
        return messages


def make_backend(model: str, settings: Settings | None = None) -> LLMBackend:
    """Select the backend from the model name (AC-5).

    ``gemini*`` resolves to :class:`GeminiBackend`; ``qwen*`` (and the legacy
    ``qwen35`` variant tag) resolves to :class:`VLLMOpenAIBackend` pointed at the
    on-prem vLLM endpoint from ``settings``. The default is the Gemini backend.

    Args:
        model: The reasoner model name or variant tag.
        settings: Optional typed settings carrying the vLLM URL / API key.

    Returns:
        The concrete :class:`LLMBackend`.
    """
    name = (model or "").lower()
    if name.startswith("gemma"):
        # Local Gemma served by Ollama (OpenAI-compatible) -> zero API cost.
        base_url = os.environ.get("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_BASE_URL)
        if settings is not None:
            base_url = getattr(settings, "ollama_base_url", "") or base_url
        logger.info("backend_selected", kind="ollama", model=model, base_url=base_url)
        return OllamaBackend(base_url=base_url, model=model)
    if name.startswith("qwen"):
        base_url = _DEFAULT_VLLM_BASE_URL
        api_key = "EMPTY"
        served_model = _DEFAULT_QWEN_MODEL
        if settings is not None:
            base_url = getattr(settings, "vllm_qwen35_url", "") or base_url
            api_key = getattr(settings, "vllm_api_key", "") or api_key
        logger.info("backend_selected", kind="vllm", model=served_model, base_url=base_url)
        return VLLMOpenAIBackend(base_url=base_url, model=served_model, api_key=api_key)
    gemini_model = model if name.startswith("gemini") else _DEFAULT_GEMINI_MODEL
    if settings is not None and not name.startswith("gemini"):
        gemini_model = getattr(settings, "gemini_model", "") or gemini_model
    # Credentials come from Settings (read from .env.local), not the ambient
    # environment: pass them explicitly so the SDK does not raise "No API key".
    api_key = ""
    use_vertexai = False
    project = ""
    location = ""
    if settings is not None:
        api_key = getattr(settings, "gemini_api_key", "") or getattr(settings, "google_api_key", "")
        use_vertexai = str(getattr(settings, "google_genai_use_vertexai", "")).lower() in (
            "1",
            "true",
            "yes",
        )
        project = getattr(settings, "google_cloud_project", "") or getattr(
            settings, "gcp_project_id", ""
        )
        location = getattr(settings, "google_cloud_location", "") or getattr(
            settings, "vertex_ai_location", ""
        )
    logger.info("backend_selected", kind="gemini", model=gemini_model, vertexai=use_vertexai)
    return GeminiBackend(
        model=gemini_model,
        api_key=api_key,
        use_vertexai=use_vertexai,
        project=project,
        location=location,
    )
