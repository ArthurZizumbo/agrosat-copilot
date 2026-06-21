"""Redis tile cache (15 min) keyed by the full endpoint hash.

This is an *explicit* helper, NOT an ASGI middleware: a global middleware would
cache every response (``/healthz``, the ``/chat`` SSE stream, error bodies),
which is wrong and dangerous for streaming. The cache grain is one tile request,
so the key is the hash of the complete endpoint -- method + path + ordered
querystring -- which covers ``url``, ``expression``, ``rescale``,
``colormap_name`` and ``z/x/y`` at once.

The Redis client is ``redis.asyncio`` over ``settings.redis_url`` (the same Redis
``rate_limit.py`` uses). Tests inject ``fakeredis.aioredis.FakeRedis`` -- no
network.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from typing import Protocol, cast

import redis.asyncio as aioredis
import structlog
from fastapi import Request
from fastapi.responses import Response

from backend.app.core.config import get_settings

logger = structlog.get_logger(__name__)

__all__ = [
    "CACHE_CONTROL_VALUE",
    "TILE_TTL_SECONDS",
    "RedisLike",
    "cached_tile",
    "make_redis",
    "tile_cache_key",
]

#: Tile TTL in seconds (AC-3: 15 minutes). Module constant -- intentionally NOT a
#: ``Settings`` field, so it cannot break startup under ``extra="forbid"``.
TILE_TTL_SECONDS = 900

#: HTTP cache header value, aligned with ``TILE_TTL_SECONDS`` so MapLibre / the
#: browser will not re-request the same tile within the window.
CACHE_CONTROL_VALUE = f"public, max-age={TILE_TTL_SECONDS}"


class RedisLike(Protocol):
    """Minimal async Redis surface used by the tile cache (real + fakeredis)."""

    async def get(self, key: str) -> bytes | None: ...

    async def setex(self, key: str, ttl: int, value: bytes) -> object: ...


def make_redis() -> RedisLike:
    """Build the async Redis client from ``settings.redis_url``.

    Returns:
        A ``redis.asyncio`` client (same Redis as ``rate_limit.py``). Tests
        override this with a ``fakeredis`` instance.
    """
    settings = get_settings()
    # ``redis.asyncio`` types ``get``/``setex`` more loosely than ``RedisLike``;
    # the structural surface we use (async ``get``/``setex``) matches at runtime.
    return cast(RedisLike, aioredis.from_url(settings.redis_url))


def tile_cache_key(method: str, path: str, query: str) -> str:
    """Build the cache key as a namespaced hash of the full endpoint.

    The querystring is sorted so the same tile requested with parameters in a
    different order collapses to a single entry.

    Args:
        method: HTTP method (e.g. ``GET``).
        path: Request path (e.g. ``/tiles/14/8629/5887.png``).
        query: Raw querystring (``request.url.query``); ordered internally.

    Returns:
        A ``tile:<sha256>`` key.
    """
    ordered = "&".join(sorted(query.split("&"))) if query else ""
    raw = f"{method}|{path}|{ordered}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"tile:{digest}"


async def cached_tile(
    request: Request,
    render: Callable[[], Awaitable[bytes]],
    *,
    redis: RedisLike | None = None,
) -> Response:
    """Serve a PNG tile through Redis: MISS renders + ``SETEX``, HIT serves blob.

    On a cache HIT the blob is returned without ever touching rio-tiler. On a
    MISS ``render`` is awaited, the PNG is stored with a 900 s TTL, and the bytes
    are returned. Both paths carry ``Cache-Control`` and an ``X-Tile-Cache``
    (HIT|MISS) header.

    Args:
        request: The incoming tile request (its method/path/query form the key).
        render: Awaitable producing the PNG bytes on a cache miss.
        redis: Optional Redis client (tests inject ``fakeredis``); defaults to
            :func:`make_redis`.

    Returns:
        A ``200`` PNG :class:`~fastapi.responses.Response`.
    """
    client = redis if redis is not None else make_redis()
    key = tile_cache_key(request.method, request.url.path, request.url.query)

    blob = await client.get(key)
    if blob is not None:
        logger.debug("tile_cache_hit", key=key)
        return _png_response(blob, cache_state="HIT")

    png = await render()
    await client.setex(key, TILE_TTL_SECONDS, png)
    logger.debug("tile_cache_miss", key=key, bytes=len(png))
    return _png_response(png, cache_state="MISS")


def _png_response(blob: bytes, *, cache_state: str) -> Response:
    """Wrap PNG bytes in a tile response with cache headers.

    Args:
        blob: PNG-encoded tile bytes.
        cache_state: ``HIT`` or ``MISS`` for the ``X-Tile-Cache`` header.

    Returns:
        A ``200`` ``image/png`` response.
    """
    return Response(
        content=blob,
        media_type="image/png",
        headers={
            "Cache-Control": CACHE_CONTROL_VALUE,
            "X-Tile-Cache": cache_state,
        },
    )
