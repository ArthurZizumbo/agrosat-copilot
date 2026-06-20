"""Backend-agnostic LLM routing table for the conversational agent (US-054).

The ``/llm/switch`` endpoint persists one of four **variant** tags on the chat
session (``gemini``, ``qwen-api``, ``qwen-onprem``, ``gemma``); the ``/chat``
endpoint then reads that tag back per request and builds the matching backend.
This module is the single declarative source mapping each variant to *where* its
backend lives, so swapping the host of a model (H100 -> L4 -> a hosted API) is a
change of an **environment variable only, never code** (AC-4):

    variant -> Route(backend_type, base_url_env, api_key_env, model_id)

``backend_type`` is one of ``{"gemini", "openai_compat"}``:

- ``gemini``        -> :class:`~ml.agent.backends.GeminiBackend` (cloud, the SDK
  reads its credentials from settings; there is no base URL).
- ``openai_compat`` -> an OpenAI-compatible endpoint reached by
  :class:`~ml.agent.backends.VLLMOpenAIBackend` (or :class:`OllamaBackend` for
  the multimodal-capable Gemma host). Any of vLLM / Ollama / Together /
  Fireworks / OpenRouter speaks this protocol, so the same code path serves
  on-prem and serverless-cloud variants alike.

The resolution reads the concrete URL / key / model id from the typed
:class:`~backend.app.core.config.Settings` (never ``os.environ`` directly): the
``*_env`` fields name the **settings attribute** to read (lower-cased, matching
pydantic-settings' env-var-to-field mapping). When a variant's URL is missing the
resolver logs ``llm_route_env_missing`` and the caller falls back to ``gemini``.

DRY: both the ``ChatService`` (per-request) and any future call-site build their
backend through :func:`make_backend_for_variant`, which delegates to the concrete
backend classes in :mod:`ml.agent.backends`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import structlog

if TYPE_CHECKING:  # pragma: no cover - typing only
    from backend.app.core.config import Settings
    from ml.agent.backends import LLMBackend

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_VARIANT",
    "VARIANTS",
    "ResolvedRoute",
    "Route",
    "make_backend_for_variant",
    "resolve_route",
]

#: The four supported persisted variant tags (1:1 with the DB CHECK constraint
#: of US-054). Order matters only for documentation / iteration.
VARIANTS: tuple[str, ...] = ("gemini", "qwen-api", "qwen-onprem", "gemma")

#: Honest degradation target when a session carries no variant or an unknown one
#: (AC-2 fallback). ``gemini`` is always resolvable from settings.
DEFAULT_VARIANT: str = "gemini"

#: The kinds of backend the agent loop can drive. ``gemini`` is the native SDK
#: path; ``openai_compat`` is every OpenAI-compatible HTTP endpoint.
BackendType = Literal["gemini", "openai_compat"]


@dataclass(frozen=True)
class Route:
    """Declarative routing entry for one variant (host-agnostic).

    The ``*_env`` fields name the :class:`~backend.app.core.config.Settings`
    **attribute** that carries the concrete value, so moving a model between
    hosts is an env-var edit (``QWEN_API_URL=...``) with zero code change. They
    are not the raw values themselves -- :func:`resolve_route` reads them off the
    typed settings at request time.

    Attributes:
        variant: The persisted variant tag (one of :data:`VARIANTS`).
        backend_type: ``"gemini"`` (native SDK) or ``"openai_compat"`` (HTTP).
        base_url_env: Settings attribute holding the endpoint URL, or ``None``
            for the ``gemini`` variant (the SDK has no base URL).
        api_key_env: Settings attribute holding the API key, or ``None`` when the
            backend needs no key (an on-prem vLLM/Ollama accepts ``"EMPTY"``).
        model_id_env: Settings attribute holding the served model id. ``None``
            means the model id is a constant carried by ``model_id_const``.
        model_id_const: Hard-wired served model id used when ``model_id_env`` is
            ``None`` (e.g. ``"qwen35"`` is the vLLM served-model alias of US-048,
            decoupled from any host).
    """

    variant: str
    backend_type: BackendType
    base_url_env: str | None
    api_key_env: str | None
    model_id_env: str | None
    model_id_const: str | None = None


@dataclass(frozen=True)
class ResolvedRoute:
    """A :class:`Route` with its env-driven values read off the settings.

    Attributes:
        variant: The resolved variant tag.
        backend_type: ``"gemini"`` or ``"openai_compat"``.
        base_url: Concrete endpoint URL (empty for ``gemini``).
        api_key: Concrete API key (empty / ``"EMPTY"`` for keyless on-prem).
        model_id: Concrete served model id passed to the backend.
    """

    variant: str
    backend_type: BackendType
    base_url: str
    api_key: str
    model_id: str


#: The four routing entries (D2 of the US-054 plan). ``gemini`` reads its model
#: id from ``settings.gemini_model`` (single source of truth) and its
#: credentials are resolved inside the Gemini backend, so it needs no base URL.
#: The OpenAI-compatible variants name the env (settings) attribute for each of
#: URL / key / model so a host swap is an env edit only.
_ROUTES: dict[str, Route] = {
    "gemini": Route(
        variant="gemini",
        backend_type="gemini",
        base_url_env=None,
        api_key_env="gemini_api_key",
        model_id_env="gemini_model",
    ),
    "qwen-api": Route(
        variant="qwen-api",
        backend_type="openai_compat",
        base_url_env="qwen_api_url",
        api_key_env="qwen_api_key",
        model_id_env="qwen_api_model",
    ),
    "qwen-onprem": Route(
        variant="qwen-onprem",
        backend_type="openai_compat",
        base_url_env="vllm_qwen35_url",
        api_key_env="vllm_api_key",
        model_id_env=None,
        model_id_const="qwen35",
    ),
    "gemma": Route(
        variant="gemma",
        backend_type="openai_compat",
        base_url_env="gemma_api_url",
        api_key_env="gemma_api_key",
        model_id_env="gemma_model",
    ),
}

#: Fallback served-model id for the ``qwen-onprem`` variant (the vLLM alias of
#: US-048) when no constant is set on the route (defensive only).
_DEFAULT_QWEN_ONPREM_MODEL: str = "qwen35"

#: Last-resort base URL for an OpenAI-compatible variant whose URL env is unset
#: (matches the US-048 vLLM serving default; the resolver already warned).
_FALLBACK_OPENAI_BASE_URL: str = "http://127.0.0.1:8002/v1"

#: Default Gemini model id mirrored from the backend so resolving ``gemini``
#: never returns an empty model when ``settings.gemini_model`` is blank.
_DEFAULT_GEMINI_MODEL: str = "gemini-3.5-flash"

#: Secondary settings attribute consulted for the Gemma / Ollama base URL when
#: the primary ``gemma_api_url`` is empty (Ollama on-prem reuses
#: ``OLLAMA_BASE_URL``), so an Ollama host needs no dedicated ``GEMMA_API_URL``.
_GEMMA_FALLBACK_URL_ENV: str = "ollama_base_url"


def _setting(settings: Settings, attr: str | None) -> str:
    """Read a string setting attribute, tolerating missing / non-string values.

    Args:
        settings: The typed application settings.
        attr: The settings attribute name to read, or ``None``.

    Returns:
        The attribute value as a stripped ``str``, or ``""`` when ``attr`` is
        ``None`` or the attribute is unset / falsy.
    """
    if attr is None:
        return ""
    value = getattr(settings, attr, "")
    return str(value).strip() if value else ""


def resolve_route(variant: str, settings: Settings) -> ResolvedRoute:
    """Resolve a variant tag to its concrete ``(backend, url, key, model)``.

    Reads the env-driven values (URL / key / model id) off the typed settings for
    the variant's :class:`Route`. An unknown variant degrades honestly to
    :data:`DEFAULT_VARIANT` (``gemini``) with a ``logger.warning``. For an
    OpenAI-compatible variant whose URL env is unset, the route still resolves
    (the caller decides whether to fall back) but a ``llm_route_env_missing``
    warning is logged so the gap is visible (AC-4 risk mitigation).

    Args:
        variant: One of :data:`VARIANTS`; anything else falls back to ``gemini``.
        settings: Typed settings carrying the per-host URL / key / model values.

    Returns:
        The :class:`ResolvedRoute` to build a backend from.
    """
    route = _ROUTES.get(variant)
    if route is None:
        logger.warning("llm_route_unknown_variant", variant=variant, fallback=DEFAULT_VARIANT)
        route = _ROUTES[DEFAULT_VARIANT]

    if route.backend_type == "gemini":
        model_id = _setting(settings, route.model_id_env) or _DEFAULT_GEMINI_MODEL
        return ResolvedRoute(
            variant=route.variant,
            backend_type="gemini",
            base_url="",
            api_key=_setting(settings, route.api_key_env),
            model_id=model_id,
        )

    base_url = _setting(settings, route.base_url_env)
    if not base_url and route.variant == "gemma":
        # Ollama on-prem hosts Gemma under OLLAMA_BASE_URL; reuse it so a
        # dedicated GEMMA_API_URL is optional.
        base_url = _setting(settings, _GEMMA_FALLBACK_URL_ENV)
    if not base_url:
        logger.warning(
            "llm_route_env_missing",
            variant=route.variant,
            env=route.base_url_env,
        )

    model_id = _setting(settings, route.model_id_env) or (
        route.model_id_const or _DEFAULT_QWEN_ONPREM_MODEL
    )
    return ResolvedRoute(
        variant=route.variant,
        backend_type="openai_compat",
        base_url=base_url,
        api_key=_setting(settings, route.api_key_env),
        model_id=model_id,
    )


def make_backend_for_variant(variant: str, settings: Settings) -> LLMBackend:
    """Build the concrete :class:`LLMBackend` for a persisted variant tag.

    This is the per-variant entry point the ``ChatService`` uses (AC-2): it
    resolves the route (env-driven, AC-4) and constructs the matching backend
    without any name-prefix heuristic. ``gemini`` builds a
    :class:`~ml.agent.backends.GeminiBackend` from the resolved Gemini model and
    the credentials the backend reads from settings; the OpenAI-compatible
    variants build a :class:`~ml.agent.backends.VLLMOpenAIBackend` pointed at the
    resolved URL / model / key.

    The legacy :func:`ml.agent.backends.make_backend` (by-model-name) is kept
    untouched for the eval call-sites; this function is the by-variant sibling.

    Args:
        variant: One of :data:`VARIANTS` (anything else falls back to ``gemini``).
        settings: Typed settings carrying the per-host URL / key / model values.

    Returns:
        A ready :class:`LLMBackend` for the variant.
    """
    from ml.agent.backends import VLLMOpenAIBackend

    resolved = resolve_route(variant, settings)
    if resolved.backend_type == "gemini":
        # Reuse the by-name builder so the Gemini credential wiring (Vertex vs
        # API key, project, location) stays in one place (DRY with make_backend).
        from ml.agent.backends import make_backend

        backend = make_backend(resolved.model_id, settings)
        logger.info(
            "backend_for_variant_selected",
            variant=resolved.variant,
            backend_type=resolved.backend_type,
            model=resolved.model_id,
        )
        return backend

    logger.info(
        "backend_for_variant_selected",
        variant=resolved.variant,
        backend_type=resolved.backend_type,
        model=resolved.model_id,
        base_url=resolved.base_url,
    )
    return VLLMOpenAIBackend(
        base_url=resolved.base_url or _FALLBACK_OPENAI_BASE_URL,
        model=resolved.model_id,
        api_key=resolved.api_key or "EMPTY",
    )
