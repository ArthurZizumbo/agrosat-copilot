"""Execution context shared by every agent tool.

Each tool's ``run`` coroutine receives, besides its validated ``*Input`` model,
a :class:`ToolContext` carrying the shared resources it needs: the asyncpg pool,
the typed settings, the active session id, and an optional deferred-execution
hook used by background/deferred tools to enqueue work instead of running it
inline.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import asyncpg

from backend.app.core.config import Settings
from ml.agent.schemas import GeoJSONGeometry

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
        request_aoi: The AOI polygon the user actually drew on this request, if
            any. The agent loop injects it into any tool whose call already
            carries an ``aoi`` argument, so geo tools operate on the EXACT drawn
            polygon instead of the one the reasoner reconstructs (which can drift
            from what the user outlined). ``None`` when the request had no AOI.
    """

    pool: asyncpg.Pool
    settings: Settings
    session_id: UUID
    defer: DeferHook | None = None
    request_aoi: GeoJSONGeometry | None = None
