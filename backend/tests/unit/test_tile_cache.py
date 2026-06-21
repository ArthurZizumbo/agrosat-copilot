"""Unit tests for the Redis tile cache helper (US-055 AC-3)."""

from __future__ import annotations

import fakeredis.aioredis
from starlette.datastructures import URL, Headers
from starlette.requests import Request

from backend.app.services.tile_cache import (
    TILE_TTL_SECONDS,
    cached_tile,
    tile_cache_key,
)

_PNG = b"\x89PNG\r\n\x1a\nfake-bytes"


def _make_request(path: str, query: str) -> Request:
    """Build a minimal GET request with the given path + querystring."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "query_string": query.encode("utf-8"),
        "headers": Headers({}).raw,
    }
    request = Request(scope)
    # Force the parsed URL so request.url.path/query are stable.
    request._url = URL(scope=scope)
    return request


def test_key_is_order_insensitive() -> None:
    """Same params in different order collapse to one key (AC-3)."""
    a = tile_cache_key("GET", "/tiles/14/1/1.png", "url=x&index=ndvi")
    b = tile_cache_key("GET", "/tiles/14/1/1.png", "index=ndvi&url=x")
    assert a == b
    assert a.startswith("tile:")


def test_key_changes_with_url_and_expression() -> None:
    """Different url / expression / tile yield different keys (AC-3)."""
    base = tile_cache_key("GET", "/tiles/14/1/1.png", "url=x&index=ndvi")
    assert base != tile_cache_key("GET", "/tiles/14/1/1.png", "url=y&index=ndvi")
    assert base != tile_cache_key("GET", "/tiles/14/1/1.png", "url=x&index=ndwi")
    assert base != tile_cache_key("GET", "/tiles/14/2/1.png", "url=x&index=ndvi")


async def test_miss_then_hit_with_fakeredis() -> None:
    """First call MISS (renders + SETEX 900), second HIT (no render)."""
    redis = fakeredis.aioredis.FakeRedis()
    calls = {"n": 0}

    async def _render() -> bytes:
        calls["n"] += 1
        return _PNG

    request = _make_request("/tiles/14/1/1.png", "url=cog.tif&index=ndvi")

    first = await cached_tile(request, _render, redis=redis)
    assert first.status_code == 200
    assert first.media_type == "image/png"
    assert first.headers["X-Tile-Cache"] == "MISS"
    assert first.headers["Cache-Control"] == f"public, max-age={TILE_TTL_SECONDS}"
    assert calls["n"] == 1

    second = await cached_tile(request, _render, redis=redis)
    assert second.status_code == 200
    assert second.headers["X-Tile-Cache"] == "HIT"
    assert second.body == _PNG
    assert calls["n"] == 1  # render NOT called again -- served from Redis.


async def test_setex_uses_900_ttl() -> None:
    """The stored blob carries a 900 s TTL (AC-3)."""
    redis = fakeredis.aioredis.FakeRedis()

    async def _render() -> bytes:
        return _PNG

    request = _make_request("/tiles/14/1/1.png", "url=cog.tif&index=ndvi")
    await cached_tile(request, _render, redis=redis)

    key = tile_cache_key("GET", "/tiles/14/1/1.png", "url=cog.tif&index=ndvi")
    ttl = await redis.ttl(key)
    assert 0 < ttl <= TILE_TTL_SECONDS
