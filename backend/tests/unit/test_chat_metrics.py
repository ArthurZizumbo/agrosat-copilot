"""Unit tests for the chat observability helper (US-065).

Cover :mod:`backend.app.utils.chat_metrics` in isolation -- no network, no DB,
no real settings: the accumulator and the classifier are pure, and the emitter
is exercised against a captured structlog stream (``structlog.testing``). The
cases pin the US-065 acceptance criteria:

* a turn with no tool calls classifies as ``simple`` with the 3 s SLO; a turn
  with >=1 tool call classifies as ``multi_step`` with the 15 s SLO;
* the SLO is ``met`` only when the latency is within the target;
* tool calls are tallied one per ``observe_tool_call``;
* missing provider usage leaves the token fields ``None`` (never synthesised);
* the emitter logs a single ``chat_turn_metrics`` line and never uses ``print``.
"""

from __future__ import annotations

import inspect

import structlog

from backend.app.utils import chat_metrics as cm
from backend.app.utils.chat_metrics import (
    MULTI_STEP_SLO_MS,
    SIMPLE_SLO_MS,
    ChatMetricsAccumulator,
    emit_chat_turn_metrics,
)


def test_simple_turn_classified_with_3s_slo() -> None:
    """A turn with no tool calls is ``simple`` and scored against 3 s."""
    metrics = ChatMetricsAccumulator(variant="gemini", model="gemini-3.5-flash").finalise(1200.0)

    assert metrics.turn_type == "simple"
    assert metrics.tool_calls == 0
    assert metrics.slo_target_ms == SIMPLE_SLO_MS
    assert metrics.slo_met is True


def test_multi_step_turn_classified_with_15s_slo() -> None:
    """A turn with >=1 tool call is ``multi_step`` and scored against 15 s."""
    acc = ChatMetricsAccumulator(variant="qwen-onprem", model="qwen35")
    acc = acc.observe_tool_call().observe_tool_call()
    metrics = acc.finalise(8000.0)

    assert metrics.turn_type == "multi_step"
    assert metrics.tool_calls == 2
    assert metrics.slo_target_ms == MULTI_STEP_SLO_MS
    assert metrics.slo_met is True


def test_simple_slo_breached_when_over_3s() -> None:
    """A simple turn slower than 3 s reports ``slo_met=False``."""
    metrics = ChatMetricsAccumulator(variant="gemini").finalise(SIMPLE_SLO_MS + 1.0)

    assert metrics.turn_type == "simple"
    assert metrics.slo_met is False


def test_multi_step_slo_breached_when_over_15s() -> None:
    """A multi-step turn slower than 15 s reports ``slo_met=False``."""
    acc = ChatMetricsAccumulator(variant="gemini").observe_tool_call()
    metrics = acc.finalise(MULTI_STEP_SLO_MS + 1.0)

    assert metrics.turn_type == "multi_step"
    assert metrics.slo_met is False


def test_slo_met_at_exact_boundary() -> None:
    """Latency exactly equal to the SLO target counts as met (``<=``)."""
    metrics = ChatMetricsAccumulator(variant="gemini").finalise(SIMPLE_SLO_MS)

    assert metrics.slo_met is True


def test_accumulator_is_immutable() -> None:
    """``observe_tool_call`` returns a new accumulator, never mutating in place."""
    acc0 = ChatMetricsAccumulator(variant="gemini")
    acc1 = acc0.observe_tool_call()

    assert acc0.tool_calls == 0
    assert acc1.tool_calls == 1
    assert acc1 is not acc0


def test_usage_captured_into_token_fields() -> None:
    """Provider usage is surfaced verbatim into the three token fields."""
    acc = ChatMetricsAccumulator(variant="gemini", model="gemini-3.5-flash")
    acc = acc.observe_usage(
        {"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165}
    )
    metrics = acc.finalise(900.0)

    assert metrics.tokens_prompt == 120
    assert metrics.tokens_completion == 45
    assert metrics.tokens_total == 165


def test_missing_usage_leaves_tokens_none() -> None:
    """No provider usage -> token fields stay ``None`` (never invented)."""
    metrics = ChatMetricsAccumulator(variant="gemini").observe_usage(None).finalise(900.0)

    assert metrics.tokens_prompt is None
    assert metrics.tokens_completion is None
    assert metrics.tokens_total is None


def test_later_none_usage_does_not_wipe_real_capture() -> None:
    """A subsequent ``None`` usage never clears a real earlier capture."""
    acc = ChatMetricsAccumulator(variant="gemini")
    acc = acc.observe_usage({"total_tokens": 10}).observe_usage(None)

    assert acc.finalise(100.0).tokens_total == 10


def test_emit_logs_single_chat_turn_metrics_line() -> None:
    """The emitter logs exactly one ``chat_turn_metrics`` event with the fields."""
    acc = ChatMetricsAccumulator(variant="gemini", model="gemini-3.5-flash")
    acc = acc.observe_tool_call().observe_usage({"total_tokens": 50})
    metrics = acc.finalise(2500.0)

    with structlog.testing.capture_logs() as logs:
        emit_chat_turn_metrics(metrics, session_id="sess-1", settings=None)

    entries = [e for e in logs if e["event"] == "chat_turn_metrics"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["turn_type"] == "multi_step"
    assert entry["tool_calls"] == 1
    assert entry["duration_ms"] == 2500.0
    assert entry["slo_target_ms"] == MULTI_STEP_SLO_MS
    assert entry["slo_met"] is True
    assert entry["tokens_total"] == 50
    assert entry["model"] == "gemini-3.5-flash"
    assert entry["variant"] == "gemini"


def test_emit_skips_prometheus_when_settings_none() -> None:
    """With ``settings=None`` no Prometheus export is attempted (structlog only)."""
    metrics = ChatMetricsAccumulator(variant="gemini").finalise(100.0)
    # Sentinel: the export must not be invoked, so the singleton stays untouched.
    cm._COLLECTORS = None
    emit_chat_turn_metrics(metrics, session_id="sess-1", settings=None)
    assert cm._COLLECTORS is None


def test_helper_module_has_no_print() -> None:
    """The helper never uses ``print`` (structlog-only observability rule)."""
    source = inspect.getsource(cm)
    assert "print(" not in source


class _PrometheusSettingsStub:
    """Settings double that turns the Prometheus export of the chat metrics on."""

    chat_metrics_prometheus_enabled = True


def test_emit_prometheus_enabled_records_without_error() -> None:
    """With the flag on, the export builds the collectors and records the turn.

    The collectors are a module-level singleton (Prometheus forbids registering
    a metric name twice), so two consecutive enabled emits must reuse them and
    never raise a registry clash -- the honest-degradation contract of US-065 R3.
    """
    cm._COLLECTORS = None
    settings = _PrometheusSettingsStub()
    acc = ChatMetricsAccumulator(variant="gemini", model="gemini-3.5-flash")
    acc = acc.observe_tool_call().observe_usage({"total_tokens": 30})
    metrics = acc.finalise(2500.0)

    emit_chat_turn_metrics(metrics, session_id="sess-1", settings=settings)  # type: ignore[arg-type]
    first = cm._COLLECTORS
    assert first is not None
    # Second emit must not raise (collectors reused, not re-registered).
    emit_chat_turn_metrics(metrics, session_id="sess-2", settings=settings)  # type: ignore[arg-type]
    assert cm._COLLECTORS is first
