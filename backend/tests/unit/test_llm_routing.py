"""Unit tests for the backend-agnostic LLM routing table (US-054).

Pins the env-driven resolution of every persisted variant to its concrete
``(backend_type, base_url, api_key, model_id)`` and the per-variant backend
construction, with the settings mocked (no real Gemini / vLLM / Ollama call).

These cover US-054:

- **AC-3** the four variants (``gemini`` / ``qwen-api`` / ``qwen-onprem`` /
  ``gemma``) all resolve; an unknown one degrades to ``gemini``.
- **AC-4** routing is env-driven: each OpenAI-compatible variant reads its URL /
  key / model off the typed settings, so a host swap is an env edit (verified by
  feeding distinct settings values and asserting they flow through).

The Gemini path is mocked at :func:`ml.agent.llm_routing.make_backend` so the
test never touches the ``google-genai`` SDK; the OpenAI-compatible path builds a
real :class:`~ml.agent.backends.VLLMOpenAIBackend` (a thin HTTP client whose
constructor performs no network I/O).
"""

from __future__ import annotations

import pytest

from backend.app.core.config import Settings
from ml.agent.backends import VLLMOpenAIBackend
from ml.agent.llm_routing import (
    DEFAULT_VARIANT,
    VARIANTS,
    make_backend_for_variant,
    resolve_route,
)


def _settings(**overrides: str) -> Settings:
    """Build a ``Settings`` with ``memory://`` redis and the given env overrides."""
    base: dict[str, object] = {"redis_url": "memory://"}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AC-3: the four variants all resolve to the expected backend type.
# ---------------------------------------------------------------------------
def test_variants_tuple_is_the_supported_tags() -> None:
    """``VARIANTS`` is exactly the hybrid variants (1:1 with the DB CHECK, E12)."""
    assert VARIANTS == ("gemini", "qwen-api", "qwen-onprem", "gemma", "qwen-vl")
    assert DEFAULT_VARIANT == "gemini"


def test_resolve_gemini_uses_settings_model_and_no_base_url() -> None:
    """``gemini`` -> native backend, model from ``settings.gemini_model``, no URL (AC-3)."""
    resolved = resolve_route("gemini", _settings(gemini_model="gemini-3.5-flash"))
    assert resolved.backend_type == "gemini"
    assert resolved.model_id == "gemini-3.5-flash"
    assert resolved.base_url == ""


def test_resolve_qwen_onprem_reads_vllm_env_and_constant_model() -> None:
    """``qwen-onprem`` reads ``VLLM_QWEN35_URL`` and serves the ``qwen35`` alias (AC-4)."""
    resolved = resolve_route(
        "qwen-onprem",
        _settings(vllm_qwen35_url="http://127.0.0.1:8002/v1", vllm_api_key="secret-onprem"),
    )
    assert resolved.backend_type == "openai_compat"
    assert resolved.base_url == "http://127.0.0.1:8002/v1"
    assert resolved.api_key == "secret-onprem"
    assert resolved.model_id == "qwen35"


def test_resolve_qwen_api_reads_hosted_env_triplet() -> None:
    """``qwen-api`` reads URL / key / model from the ``QWEN_API_*`` env triplet (AC-4)."""
    resolved = resolve_route(
        "qwen-api",
        _settings(
            qwen_api_url="https://hosted.example/v1",
            qwen_api_key="hosted-key",
            qwen_api_model="qwen3-30b",
        ),
    )
    assert resolved.backend_type == "openai_compat"
    assert resolved.base_url == "https://hosted.example/v1"
    assert resolved.api_key == "hosted-key"
    assert resolved.model_id == "qwen3-30b"


def test_resolve_gemma_reads_gemma_env_triplet() -> None:
    """``gemma`` reads the ``GEMMA_*`` env triplet (AC-4)."""
    resolved = resolve_route(
        "gemma",
        _settings(
            gemma_api_url="http://127.0.0.1:11434/v1",
            gemma_api_key="gemma-key",
            gemma_model="gemma4:27b",
        ),
    )
    assert resolved.backend_type == "openai_compat"
    assert resolved.base_url == "http://127.0.0.1:11434/v1"
    assert resolved.model_id == "gemma4:27b"


def test_resolve_gemma_falls_back_to_ollama_base_url() -> None:
    """When ``GEMMA_API_URL`` is empty, ``gemma`` reuses ``OLLAMA_BASE_URL`` (AC-4)."""
    resolved = resolve_route(
        "gemma",
        _settings(gemma_api_url="", ollama_base_url="http://127.0.0.1:11434/v1"),
    )
    assert resolved.base_url == "http://127.0.0.1:11434/v1"


def test_resolve_unknown_variant_degrades_to_gemini() -> None:
    """An unknown variant falls back to ``gemini`` (honest degradation, AC-2/AC-3)."""
    resolved = resolve_route("does-not-exist", _settings(gemini_model="gemini-3.5-flash"))
    assert resolved.variant == "gemini"
    assert resolved.backend_type == "gemini"


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_variant_resolves_without_error(variant: str) -> None:
    """Every variant resolves even with empty env (defensive, AC-3 / E12)."""
    resolved = resolve_route(variant, _settings())
    assert resolved.variant == variant
    assert resolved.backend_type in {"gemini", "openai_compat", "ollama"}


def test_resolve_qwen_vl_reads_env_and_serves_multimodal_alias() -> None:
    """``qwen-vl`` reads ``QWEN36_VL_URL`` and serves the multimodal alias (E12)."""
    resolved = resolve_route(
        "qwen-vl",
        _settings(qwen36_vl_url="http://127.0.0.1:8003/v1"),
    )
    assert resolved.backend_type == "ollama"
    assert resolved.base_url == "http://127.0.0.1:8003/v1"
    assert resolved.model_id == "qwen36-vl"


def test_make_backend_for_qwen_vl_builds_ollama_backend() -> None:
    """``qwen-vl`` builds the image-forwarding ``OllamaBackend`` (E12)."""
    from ml.agent.backends import OllamaBackend
    from ml.agent.llm_routing import make_backend_for_variant

    settings = _settings(qwen36_vl_url="http://127.0.0.1:8003/v1")
    backend = make_backend_for_variant("qwen-vl", settings)
    assert isinstance(backend, OllamaBackend)
    assert backend.model == "qwen36-vl"
    assert "127.0.0.1:8003" in str(backend._base_url)


# ---------------------------------------------------------------------------
# AC-2/AC-4: make_backend_for_variant builds the right backend type per variant.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("variant", ["qwen-api", "qwen-onprem", "gemma"])
def test_make_backend_openai_compat_variants_build_vllm_backend(variant: str) -> None:
    """The three OpenAI-compatible variants build a ``VLLMOpenAIBackend`` (AC-2)."""
    settings = _settings(
        vllm_qwen35_url="http://127.0.0.1:8002/v1",
        qwen_api_url="https://hosted.example/v1",
        qwen_api_model="qwen3-30b",
        gemma_api_url="http://127.0.0.1:11434/v1",
        gemma_model="gemma4:27b",
    )
    backend = make_backend_for_variant(variant, settings)
    assert isinstance(backend, VLLMOpenAIBackend)


def test_make_backend_for_qwen_onprem_points_at_vllm_url() -> None:
    """``qwen-onprem`` builds a backend pointed at the configured vLLM URL/model (AC-4)."""
    settings = _settings(vllm_qwen35_url="http://127.0.0.1:8002/v1", vllm_api_key="EMPTY")
    backend = make_backend_for_variant("qwen-onprem", settings)
    assert isinstance(backend, VLLMOpenAIBackend)
    assert backend.model == "qwen35"
    assert "127.0.0.1:8002" in str(backend._base_url)


def test_make_backend_for_gemini_delegates_to_make_backend(monkeypatch) -> None:
    """``gemini`` delegates to the by-name ``make_backend`` (mocked; no SDK call) (AC-2).

    ``make_backend_for_variant`` imports ``make_backend`` from
    :mod:`ml.agent.backends` at call time, so the patch targets that module (not
    the ``llm_routing`` namespace) to intercept the Gemini credential wiring
    without instantiating the real ``google-genai`` client.
    """
    import ml.agent.backends as backends_mod

    seen: dict[str, object] = {}

    class _FakeGeminiBackend:
        def __init__(self, model: str) -> None:
            self.model = model

    def _fake_make_backend(model_id: str, settings: object) -> _FakeGeminiBackend:
        seen["model_id"] = model_id
        return _FakeGeminiBackend(model_id)

    monkeypatch.setattr(backends_mod, "make_backend", _fake_make_backend, raising=False)
    backend = make_backend_for_variant("gemini", _settings(gemini_model="gemini-3.5-flash"))
    assert seen["model_id"] == "gemini-3.5-flash"
    assert getattr(backend, "model", None) == "gemini-3.5-flash"
