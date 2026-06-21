"""Async PostgreSQL access layer for the backend API (RLS-aware).

This module owns the backend's lazily-initialised singleton
:class:`asyncpg.Pool` and the per-request machinery that primes the multi-tenant
Row-Level-Security hook before any query runs.

Multi-tenant isolation (US-051) is enforced at the database layer by RLS
policies keyed on the ``app.current_session`` runtime setting. Every request
must set that key to its tenant ``session_id`` *inside the same transaction* as
its queries, otherwise the ``FORCE ROW LEVEL SECURITY`` policies fail closed
(zero rows). The contract is **byte-identical** to
:func:`ml.agent.db.session_scoped_conn`: same key ``app.current_session``, same
``set_config(..., is_local => true)`` (``SET LOCAL`` semantics), and the value
is always **bound** as a parameter, never string-interpolated.

DSN handling reuses :func:`ml.agent.db.to_asyncpg_dsn` (DRY) to strip the
SQLAlchemy ``+asyncpg`` driver marker. The backend connects with the
application-role DSN (``app_database_url``, role ``agrosat_app``), which has no
``BYPASSRLS`` — this is what makes the policies actually enforce.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

import asyncpg
import structlog
from fastapi import Depends, Header, HTTPException, status

from backend.app.core.config import get_settings
from backend.app.utils.session import SESSION_ID_HEADER, parse_session_id
from ml.agent.db import to_asyncpg_dsn

__all__ = [
    "close_pool",
    "get_pool",
    "get_request_session_id",
    "get_scoped_conn",
    "session_scoped_conn",
]

logger = structlog.get_logger(__name__)

# SQL emitted to prime the per-session RLS hook. Kept identical to the agent
# layer (``ml.agent.db``) so RLS policies apply uniformly whichever layer
# touches the database. ``$1`` is bound (never interpolated) and
# ``set_config(..., true)`` is ``SET LOCAL`` (transaction-scoped) semantics.
_SET_SESSION_SQL = "SELECT set_config('app.current_session', $1, true)"

# Process-wide singleton. Guarded by an explicit None-check because asyncpg pool
# creation is async and cannot live in a module-level initialiser.
_POOL: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    """Return the lazily-initialised singleton backend connection pool.

    The first call builds the pool from the application-role DSN
    (``settings.app_database_url``, role ``agrosat_app`` without ``BYPASSRLS``);
    subsequent calls return the same instance.

    Returns:
        The shared :class:`asyncpg.Pool`.
    """
    global _POOL
    if _POOL is None:
        settings = get_settings()
        dsn = to_asyncpg_dsn(settings.app_database_url)
        _POOL = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=10)
        logger.info("backend_db_pool_created", min_size=1, max_size=10)
    return _POOL


@asynccontextmanager
async def session_scoped_conn(session_id: UUID) -> AsyncIterator[asyncpg.Connection]:
    """Acquire a backend pool connection with the per-session RLS hook primed.

    Inside the context, ``app.current_session`` is set as a transaction-local
    setting (``SET LOCAL`` semantics via ``set_config(..., is_local => true)``)
    so the RLS policies keyed on it apply to every query run with the yielded
    connection. The value is bound through ``set_config`` rather than
    string-interpolated to avoid SQL injection. Because ``SET LOCAL`` only
    persists inside a transaction, the body is wrapped in one.

    This is the identical contract emitted by
    :func:`ml.agent.db.session_scoped_conn`.

    Args:
        session_id: Tenant session whose data the connection may access.

    Yields:
        An :class:`asyncpg.Connection` from the pool, with the RLS session set
        and an open transaction.
    """
    pool = await get_pool()
    conn = await pool.acquire()
    try:
        async with conn.transaction():
            await conn.execute(_SET_SESSION_SQL, str(session_id))
            logger.debug("backend_db_session_scoped", session_id=str(session_id))
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
        logger.info("backend_db_pool_closed")
        _POOL = None


async def get_request_session_id(
    x_session_id: Annotated[str | None, Header(alias=SESSION_ID_HEADER)] = None,
) -> UUID:
    """FastAPI dependency: resolve the tenant session id from the request header.

    Reads and validates the ``X-Session-ID`` header. Kept thin so routers never
    touch the raw header — they depend on :func:`get_scoped_conn` (or this) and
    receive a typed :class:`uuid.UUID`.

    Args:
        x_session_id: Raw value of the ``X-Session-ID`` header, injected by
            FastAPI.

    Returns:
        The validated tenant session id.

    Raises:
        HTTPException: ``400 Bad Request`` if the header is missing or malformed.
    """
    try:
        session_id: UUID = parse_session_id(x_session_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    return session_id


async def get_scoped_conn(
    session_id: Annotated[UUID, Depends(get_request_session_id)],
) -> AsyncIterator[asyncpg.Connection]:
    """FastAPI dependency yielding an RLS-scoped connection for the request.

    Business routers (US-052/053/...) consume this via::

        conn: Annotated[asyncpg.Connection, Depends(get_scoped_conn)]

    The yielded connection has ``app.current_session`` primed (``SET LOCAL``)
    inside an open transaction, so every query the handler runs with it is
    subject to the tenant-isolation RLS policies. FastAPI closes the generator
    (committing/rolling back the transaction and releasing the connection) when
    the request ends.

    Args:
        session_id: Validated tenant session id, injected by
            :func:`get_request_session_id`.

    Yields:
        An RLS-scoped :class:`asyncpg.Connection`.
    """
    async with session_scoped_conn(session_id) as conn:
        yield conn
