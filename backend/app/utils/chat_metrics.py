"""Per-turn chat observability helper (US-065).

Centralises the FinOps/SLO instrumentation of the ``/chat`` SSE stream so the
business logic in :class:`~backend.app.services.chat_service.ChatService` only
*observes* the agent events and the metric emission stays DRY and testable:

* a pure classifier turns the per-turn tally (tool-calls observed, end-to-end
  latency, active model/variant and optional provider token usage) into a typed
  :class:`ChatTurnMetrics` snapshot, classifying the turn as ``"simple"`` (no
  tool calls) or ``"multi_step"`` (>=1) and evaluating it against the matching
  latency SLO (``3000 ms`` simple / ``15000 ms`` multi-step);
* an immutable :class:`ChatMetricsAccumulator` the service mutates as the stream
  unfolds (one increment per ``tool_call`` event, one ``usage`` capture on the
  terminal ``done`` event) -- ``frozen`` so each ``observe_*`` returns a new
  snapshot and the service never mutates shared state in place;
* a single structlog emission under the canonical ``chat_turn_metrics`` key
  (never :func:`print`), plus a best-effort Prometheus export gated by settings
  that becomes a no-op when ``prometheus-client`` is absent (lazy import, so the
  app boots and the tests run without the exporter).

Token usage is reported verbatim from the provider (Gemini exposes it; vLLM
streaming only with ``include_usage``) or left ``None`` -- it is never
synthesised, honouring the project's "real data, no placeholders" rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

import structlog

if TYPE_CHECKING:  # pragma: no cover - typing only
    from prometheus_client import Counter, Histogram

    from backend.app.core.config import Settings

logger = structlog.get_logger(__name__)

__all__ = [
    "MULTI_STEP_SLO_MS",
    "SIMPLE_SLO_MS",
    "ChatMetricsAccumulator",
    "ChatTurnMetrics",
    "TurnType",
    "emit_chat_turn_metrics",
]

#: Turn classification: ``"simple"`` (no tool calls) vs ``"multi_step"`` (>=1).
TurnType = Literal["simple", "multi_step"]

#: Latency SLO for a simple turn (no tool calls), in milliseconds (US-065 AC:
#: p95 < 3 s for a simple query).
SIMPLE_SLO_MS: float = 3000.0

#: Latency SLO for a multi-step turn (>=1 tool call), in milliseconds (US-065
#: AC: p95 < 15 s for a multi-step query).
MULTI_STEP_SLO_MS: float = 15000.0

#: Canonical structlog event key for the single per-turn observability log.
_LOG_KEY: str = "chat_turn_metrics"

#: Provider-neutral token keys read off the terminal event's ``usage`` mapping.
_PROMPT_KEY: str = "prompt_tokens"
_COMPLETION_KEY: str = "completion_tokens"
_TOTAL_KEY: str = "total_tokens"


@dataclass(frozen=True)
class ChatTurnMetrics:
    """Immutable observability snapshot of a single ``/chat`` turn.

    Attributes:
        tool_calls: Number of ``tool_call`` events observed during the turn.
        duration_ms: End-to-end turn latency in milliseconds (the service's
            ``time.perf_counter`` span, the single source of truth).
        turn_type: ``"simple"`` when no tool was called, ``"multi_step"`` when
            at least one was.
        slo_target_ms: The latency SLO the turn is measured against
            (:data:`SIMPLE_SLO_MS` / :data:`MULTI_STEP_SLO_MS`).
        slo_met: ``True`` when ``duration_ms <= slo_target_ms``.
        model: Concrete reasoner model id behind the active variant (e.g.
            ``gemini-3.5-flash``), as resolved by the service; ``None`` when the
            backend exposed none.
        variant: Active LLM variant tag (``gemini`` / ``qwen-api`` /
            ``qwen-onprem`` / ``gemma``), the FinOps grouping key.
        tokens_prompt: Prompt tokens reported by the provider, or ``None``.
        tokens_completion: Completion tokens reported by the provider, or
            ``None``.
        tokens_total: Total tokens reported by the provider, or ``None``.
    """

    tool_calls: int
    duration_ms: float
    turn_type: TurnType
    slo_target_ms: float
    slo_met: bool
    model: str | None
    variant: str | None
    tokens_prompt: int | None
    tokens_completion: int | None
    tokens_total: int | None


def _classify(tool_calls: int) -> tuple[TurnType, float]:
    """Classify a turn by its tool-call count and pick the matching SLO.

    Args:
        tool_calls: Number of ``tool_call`` events observed during the turn.

    Returns:
        ``(turn_type, slo_target_ms)``: ``("simple", SIMPLE_SLO_MS)`` when no
        tool was called, ``("multi_step", MULTI_STEP_SLO_MS)`` otherwise.
    """
    if tool_calls >= 1:
        return "multi_step", MULTI_STEP_SLO_MS
    return "simple", SIMPLE_SLO_MS


@dataclass(frozen=True)
class ChatMetricsAccumulator:
    """Immutable per-turn tally updated as the agent event stream unfolds.

    The service threads one accumulator through a single chat turn, calling
    :meth:`observe_tool_call` for every ``tool_call`` event and
    :meth:`observe_usage` once when the terminal ``done`` event carries provider
    token usage. Being ``frozen``, each observation returns a *new* accumulator
    rather than mutating shared state, so concurrent streams never interfere.

    Attributes:
        variant: Active LLM variant tag for the turn (FinOps grouping key).
        model: Concrete reasoner model id behind the variant, when known.
        tool_calls: Running count of observed ``tool_call`` events.
        usage: Provider token mapping captured from the terminal event, or
            ``None`` when the provider reported none (never synthesised).
    """

    variant: str | None = None
    model: str | None = None
    tool_calls: int = 0
    usage: dict[str, int] | None = field(default=None)

    def observe_tool_call(self) -> ChatMetricsAccumulator:
        """Return a copy with the tool-call counter incremented by one.

        Returns:
            A new :class:`ChatMetricsAccumulator` with ``tool_calls + 1``.
        """
        return replace(self, tool_calls=self.tool_calls + 1)

    def observe_usage(self, usage: dict[str, int] | None) -> ChatMetricsAccumulator:
        """Return a copy carrying the provider's token usage, if any.

        Args:
            usage: The terminal event's ``usage`` mapping (provider-neutral
                ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``), or
                ``None`` when the provider reported none.

        Returns:
            A new :class:`ChatMetricsAccumulator` carrying ``usage`` (unchanged
            when ``usage`` is ``None``, so a later ``None`` never wipes a real
            capture).
        """
        if usage is None:
            return self
        return replace(self, usage=dict(usage))

    def finalise(self, duration_ms: float) -> ChatTurnMetrics:
        """Build the typed turn snapshot from the tally and the turn latency.

        Args:
            duration_ms: End-to-end turn latency in milliseconds, measured by
                the service (single source of truth -- no timer is created here).

        Returns:
            The :class:`ChatTurnMetrics` for the turn, classified and scored
            against the matching latency SLO.
        """
        turn_type, slo_target_ms = _classify(self.tool_calls)
        usage = self.usage or {}
        return ChatTurnMetrics(
            tool_calls=self.tool_calls,
            duration_ms=duration_ms,
            turn_type=turn_type,
            slo_target_ms=slo_target_ms,
            slo_met=duration_ms <= slo_target_ms,
            model=self.model,
            variant=self.variant,
            tokens_prompt=usage.get(_PROMPT_KEY),
            tokens_completion=usage.get(_COMPLETION_KEY),
            tokens_total=usage.get(_TOTAL_KEY),
        )


def emit_chat_turn_metrics(
    metrics: ChatTurnMetrics,
    *,
    session_id: str,
    settings: Settings | None = None,
) -> None:
    """Emit the per-turn metrics to structlog (and Prometheus when enabled).

    Logs exactly one structured event under the canonical
    :data:`chat_turn_metrics <_LOG_KEY>` key -- never :func:`print` -- so the
    SLO dashboard and the FinOps cost-per-model query both read a single line per
    turn. When ``settings.chat_metrics_prometheus_enabled`` is set, it also
    updates the Prometheus collectors via :func:`_export_prometheus`, which is a
    best-effort no-op if ``prometheus-client`` is not installed.

    Args:
        metrics: The finalised turn snapshot to record.
        session_id: Tenant session id (already stringified) for the log line.
        settings: Typed application settings; the Prometheus export reads
            ``chat_metrics_prometheus_enabled`` off it. ``None`` skips the export.
    """
    logger.info(
        _LOG_KEY,
        session_id=session_id,
        turn_type=metrics.turn_type,
        duration_ms=metrics.duration_ms,
        slo_target_ms=metrics.slo_target_ms,
        slo_met=metrics.slo_met,
        tool_calls=metrics.tool_calls,
        tokens_prompt=metrics.tokens_prompt,
        tokens_completion=metrics.tokens_completion,
        tokens_total=metrics.tokens_total,
        model=metrics.model,
        variant=metrics.variant,
    )
    if settings is not None and getattr(settings, "chat_metrics_prometheus_enabled", False):
        _export_prometheus(metrics)


def _export_prometheus(metrics: ChatTurnMetrics) -> None:
    """Update the Prometheus collectors for a turn, gated and best-effort.

    The exporter is imported lazily and any failure (``prometheus-client``
    absent, registry clash) is swallowed with a ``debug`` log: chat observability
    must never break the response stream over a missing optional dependency
    (US-065 R3 -- honest degradation to structlog-only).

    Args:
        metrics: The finalised turn snapshot to export.
    """
    try:
        histogram, tool_calls_total, tokens_total = _prometheus_collectors()
    except Exception as exc:  # noqa: BLE001 - optional exporter must never break /chat
        logger.debug("chat_metrics_prometheus_unavailable", error=str(exc))
        return

    labels = {"model": metrics.model or "unknown", "variant": metrics.variant or "unknown"}
    histogram.labels(turn_type=metrics.turn_type).observe(metrics.duration_ms / 1000.0)
    if metrics.tool_calls:
        tool_calls_total.labels(**labels).inc(metrics.tool_calls)
    if metrics.tokens_total is not None:
        tokens_total.labels(**labels).inc(metrics.tokens_total)


def _prometheus_collectors() -> tuple[Histogram, Counter, Counter]:
    """Return the singleton Prometheus collectors, creating them once.

    The collectors live on the module so repeated turns reuse the same metric
    families (Prometheus forbids re-registering a name). ``prometheus-client`` is
    imported here so the app and the tests do not depend on it at import time.

    Returns:
        ``(histogram, tool_calls_counter, tokens_counter)``.
    """
    global _COLLECTORS
    if _COLLECTORS is None:
        from prometheus_client import Counter, Histogram

        histogram = Histogram(
            "chat_turn_duration_seconds",
            "End-to-end /chat turn latency in seconds, by turn type.",
            labelnames=("turn_type",),
            buckets=(0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 15.0, 20.0, 30.0),
        )
        tool_calls_total = Counter(
            "chat_tool_calls_total",
            "Tool calls invoked across /chat turns, by model and variant.",
            labelnames=("model", "variant"),
        )
        tokens_total = Counter(
            "chat_tokens_total",
            "Reasoner tokens consumed across /chat turns, by model and variant.",
            labelnames=("model", "variant"),
        )
        _COLLECTORS = (histogram, tool_calls_total, tokens_total)
    return _COLLECTORS


#: Lazily-built singleton of the Prometheus collectors (``None`` until the first
#: enabled export). Module-level so the metric families are registered once.
_COLLECTORS: tuple[Histogram, Counter, Counter] | None = None
