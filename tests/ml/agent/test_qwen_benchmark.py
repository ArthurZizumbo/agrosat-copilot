"""US-048: tests for the Qwen serving benchmark logic (no GPU, no network).

The vLLM endpoint cannot run in CI (and is blocked on the H100 VM by the
nested-virt host issue), so these tests exercise the benchmark's pure logic with
a fake OpenAI client: the smoke call, the latency sampling, and the percentile
aggregation. MLflow logging is not exercised here (it needs the server).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "benchmark_qwen35", _REPO_ROOT / "scripts" / "benchmark_qwen35.py"
)
assert _SPEC is not None and _SPEC.loader is not None
benchmark_qwen35 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark_qwen35)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeCompletion:
        self.calls.append(kwargs)
        # Echo a deterministic answer so the smoke text is non-empty.
        return _FakeCompletion("respuesta deterministica del modelo")


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()


@pytest.fixture
def patched_client(monkeypatch: pytest.MonkeyPatch) -> _FakeOpenAIClient:
    """Patch the benchmark's client factory to return a fake OpenAI client."""
    client = _FakeOpenAIClient()
    monkeypatch.setattr(benchmark_qwen35, "_client", lambda base_url, api_key: client)
    return client


def test_run_benchmark_collects_metrics(patched_client: _FakeOpenAIClient) -> None:
    """run_benchmark returns the expected metric keys and a matching sample count."""
    metrics = benchmark_qwen35.run_benchmark(
        base_url="http://x/v1", model="qwen35", api_key="EMPTY", n=6
    )
    assert set(metrics) == {
        "smoke_latency_s",
        "latency_p50_s",
        "latency_p95_s",
        "latency_mean_s",
        "n_samples",
    }
    assert metrics["n_samples"] == 6.0
    # 1 smoke + 6 latency samples = 7 completion calls.
    assert len(patched_client.chat.completions.calls) == 7


def test_run_benchmark_uses_zero_temperature(patched_client: _FakeOpenAIClient) -> None:
    """Every benchmark query is deterministic (temperature 0) for reproducibility."""
    benchmark_qwen35.run_benchmark(base_url="http://x/v1", model="qwen35", api_key="EMPTY", n=3)
    assert all(call["temperature"] == 0.0 for call in patched_client.chat.completions.calls)
    assert all(call["model"] == "qwen35" for call in patched_client.chat.completions.calls)


def test_percentiles_are_ordered(patched_client: _FakeOpenAIClient) -> None:
    """p50 <= p95 (the aggregation sorts the samples)."""
    metrics = benchmark_qwen35.run_benchmark(
        base_url="http://x/v1", model="qwen35", api_key="EMPTY", n=10
    )
    assert metrics["latency_p50_s"] <= metrics["latency_p95_s"]
