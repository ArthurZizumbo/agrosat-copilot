"""Execution context shared by every agent tool.

Each tool's ``run`` coroutine receives, besides its validated ``*Input`` model,
a :class:`ToolContext` carrying the shared resources it needs: the asyncpg pool,
the typed settings, the active session id, an optional deferred-execution hook
used by background/deferred tools to enqueue work instead of running it inline,
and the per-turn USER choices that must outrank whatever the reasoner asks for
(``crop_model``).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import asyncpg

from backend.app.core.config import Settings
from ml.agent.schemas import CropModel

__all__ = ["DeferHook", "ToolContext"]

# Signature of the deferred-execution hook. Deferred tools call it with a job
# name and a JSON-serialisable payload; it returns a handle/identifier for the
# enqueued job. The concrete implementation (Pub/Sub worker) is wired by the
# agent loop in US-047; tools must not assume it is present.
DeferHook = Callable[[str, dict[str, Any]], Awaitable[Any]]


@dataclass
class ToolContext:
    """Resources passed to every tool's ``run`` coroutine.

    Attributes:
        pool: Shared asyncpg pool (use ``session_scoped_conn`` to acquire a
            session-scoped connection rather than the raw pool directly).
        settings: Typed application settings (DSNs, GEE/TiTiler config, etc.).
        session_id: Active tenant session; every DB query must filter by it.
        defer: Optional hook to enqueue background work. Deferred tools call it
            when a heavy/async backend is available; ``None`` means no deferred
            executor is wired, in which case deferred tools return a controlled
            ``NotConfigured`` result instead of crashing.
        crop_model: Crop-classification model the USER pinned for this turn in the
            UI (``ChatRequest.crop_model``), or ``None`` when they pinned nothing.
            This is a CONTRACT, not a hint: when set,
            :func:`ml.agent.tools.classify.run` serves exactly this model and
            ignores the ``model`` argument the reasoner passed. Enforcing it here
            -- at the tool boundary, on the context every tool already receives --
            is what makes the UI's promise real: an LLM cannot opt out of it, and
            both entry points (the reasoner's tool call and the perceiver's AOI
            observation) read the choice from this one place. ``None`` leaves the
            model selection to the caller/tool default (``voting3``).
    """

    pool: asyncpg.Pool
    settings: Settings
    session_id: UUID
    defer: DeferHook | None = None
    crop_model: CropModel | None = None
