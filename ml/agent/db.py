"""Async PostgreSQL access layer for the agent tools.

Provides a lazily-initialised singleton :class:`asyncpg.Pool` and a
session-scoped connection context manager that primes the per-session RLS hook.

The Row-Level-Security policy keyed on ``app.current_session`` is owned by
EPIC 11 and is not part of this US. Until it exists, every tool still filters by
``session_id`` in its ``WHERE`` clause (defence in depth); this module emits the
``SET LOCAL`` so that the policy works transparently the moment it is created.

The DSN is read once from ``backend.app.core.config.get_settings().database_url``.
That URL uses the SQLAlchemy driver form ``postgresql+asyncpg://...``; asyncpg's
own ``connect`` does not understand the ``+asyncpg`` suffix, so it is stripped.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import asyncpg
import structlog

from backend.app.core.config import get_settings

__all__ = ["close_pool", "get_pool", "session_scoped_conn", "to_asyncpg_dsn"]

logger = structlog.get_logger(__name__)

# Process-wide singleton. Guarded so concurrent first-callers do not each build
# a pool; asyncpg pool creation is async, hence the explicit None-check pattern.
_POOL: asyncpg.Pool | None = None


def to_asyncpg_dsn(database_url: str) -> str:
    """Convert a SQLAlchemy-style URL into a plain asyncpg DSN.

    ``asyncpg.connect`` / ``create_pool`` accept the libpq URL form
    ``postgresql://...`` but not the SQLAlchemy driver form
    ``postgresql+asyncpg://...``. This strips the ``+asyncpg`` driver marker.

    Args:
        database_url: URL as stored in settings, e.g.
            ``postgresql+asyncpg://user:pass@host:5432/db``.

    Returns:
        The same URL with the ``+asyncpg`` driver marker removed.
    """
    return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def get_pool() -> asyncpg.Pool:
    """Return the lazily-initialised singleton connection pool.

    The first call creates the pool from the configured DSN; subsequent calls
    return the same instance. Safe to call from any tool before acquiring a
    connection.

    Returns:
        The shared :class:`asyncpg.Pool`.
    """
    global _POOL
    if _POOL is None:
        settings = get_settings()
        dsn = to_asyncpg_dsn(settings.database_url)
        _POOL = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=10)
        logger.info("agent_db_pool_created", min_size=1, max_size=10)
    return _POOL


@asynccontextmanager
async def session_scoped_conn(session_id: UUID) -> AsyncIterator[asyncpg.Connection]:
    """Acquire a connection with the per-session RLS hook primed.

    Inside the context, ``app.current_session`` is set as a transaction-local
    setting (``SET LOCAL`` semantics via ``set_config(..., is_local => true)``)
    so any future RLS policy keyed on it applies to the queries run here. The
    parameter is bound through ``set_config`` rather than string-interpolated to
    avoid SQL injection. The setting is scoped to the surrounding transaction,
    so the body is wrapped in one.

    Args:
        session_id: Tenant session whose data the connection may access.

    Yields:
        An :class:`asyncpg.Connection` from the pool, with the RLS session set.
    """
    pool = await get_pool()
    conn = await pool.acquire()
    try:
        # ``SET LOCAL`` only persists inside a transaction; open one so the
        # session setting survives for every query in the body.
        async with conn.transaction():
            # ``is_local => true`` => transaction-local, equivalent to SET LOCAL.
            await conn.execute(
                "SELECT set_config('app.current_session', $1, true)",
                str(session_id),
            )
            logger.debug("agent_db_session_scoped", session_id=str(session_id))
            yield conn
    finally:
        await pool.release(conn)


async def close_pool() -> None:
    """Close the singleton pool on application shutdown.

    Idempotent: a no-op if the pool was never created.
    """
    global _POOL
    if _POOL is not None:
        await _POOL.close()
        logger.info("agent_db_pool_closed")
        _POOL = None
