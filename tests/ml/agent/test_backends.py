"""Tests for the LLM backend factory (no real Vertex/vLLM calls)."""

from __future__ import annotations

import pytest

from ml.agent.backends import (
    GeminiBackend,
    LLMBackend,
    VLLMOpenAIBackend,
    get_backend,
)
from ml.agent.settings import AgentSettings


def test_get_backend_gemini() -> None:
    backend = get_backend("gemini", settings=AgentSettings())
    assert isinstance(backend, GeminiBackend)
    assert isinstance(backend, LLMBackend)
    assert backend.name == "gemini"


def test_get_backend_qwen35() -> None:
    backend = get_backend("qwen35", settings=AgentSettings())
    assert isinstance(backend, VLLMOpenAIBackend)
    assert isinstance(backend, LLMBackend)
    assert backend.name == "qwen35"


def test_get_backend_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown LLM variant"):
        get_backend("gpt5", settings=AgentSettings())  # type: ignore[arg-type]


def test_constructing_backend_does_not_touch_network() -> None:
    """Instantiating either backend must not require a live endpoint/SDK call."""
    # No exception, no client built yet (lazy).
    assert GeminiBackend(settings=AgentSettings())._client is None
    assert VLLMOpenAIBackend(settings=AgentSettings())._client is None
