"""Integration tests for COG tile rendering with a REAL COG (US-055).

Drives the real FastAPI app through ``httpx.AsyncClient`` + ``ASGITransport`` (no
live server, no network). The COG is built from REAL farslip Sentinel-2 pixels +
REAL lat/lon (``conftest.real_cog``); the Redis tile cache is backed by
``fakeredis``. Covers AC-2 (``/cog`` factory), AC-5 (``/tiles`` contract), AC-3 /
AC-4 (cache MISS->HIT + headers) and AC-7 (honest validation/errors).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import fakeredis.aioredis
import morecantile
import pytest
import rasterio
from httpx import ASGITransport, AsyncClient

from backend.app.api.tiles import get_tile_redis
from backend.app.main import create_app

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _xyz_for_cog(cog: Path, zoom: int = 14) -> tuple[int, int, int]:
    """Return an XYZ tile covering the COG centre at ``zoom`` (WebMercatorQuad)."""
    tms = morecantile.tms.get("WebMercatorQuad")
    with rasterio.open(cog) as src:
        bounds = src.bounds
        cx = (bounds.left + bounds.right) / 2
        cy = (bounds.top + bounds.bottom) / 2
    # COG is EPSG:3857; convert centre to lon/lat for the tms.tile() lookup.
    from rasterio.warp import transform as warp_transform

    lon, lat = warp_transform("EPSG:3857", "EPSG:4326", [cx], [cy])
    tile = tms.tile(lon[0], lat[0], zoom)
    return tile.x, tile.y, zoom


@pytest.fixture()
def client(real_cog: Path) -> Iterator[tuple[AsyncClient, Path]]:
    """App client with the tile-cache Redis overridden by fakeredis."""
    app = create_app()
    shared = fakeredis.aioredis.FakeRedis()
    app.dependency_overrides[get_tile_redis] = lambda: shared
    transport = ASGITransport(app=app)
    yield AsyncClient(transport=transport, base_url="http://test"), real_cog
    app.dependency_overrides.clear()


async def test_cog_factory_renders_ndvi_png(client: tuple[AsyncClient, Path]) -> None:
    """AC-2: ``/cog/tiles/{TMS}/{z}/{x}/{y}.png`` renders a real NDVI PNG."""
    ac, cog = client
    x, y, z = _xyz_for_cog(cog)
    async with ac:
        resp = await ac.get(
            f"/cog/tiles/WebMercatorQuad/{z}/{x}/{y}.png",
            params={
                "url": str(cog),
                "expression": "(b4-b3)/(b4+b3)",
                "rescale": "-1,1",
                "colormap_name": "rdylgn",
            },
        )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/png"
    assert resp.content.startswith(_PNG_SIGNATURE)


async def test_tiles_contract_renders_ndvi(client: tuple[AsyncClient, Path]) -> None:
    """AC-5: ``/tiles/{z}/{x}/{y}.png?url&index=ndvi`` -> 200 PNG (no longer 501)."""
    ac, cog = client
    x, y, z = _xyz_for_cog(cog)
    async with ac:
        resp = await ac.get(f"/tiles/{z}/{x}/{y}.png", params={"url": str(cog), "index": "ndvi"})
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/png"
    assert resp.content.startswith(_PNG_SIGNATURE)


async def test_tile_cache_miss_then_hit_with_headers(client: tuple[AsyncClient, Path]) -> None:
    """AC-3/AC-4: first MISS then HIT, both with Cache-Control + X-Tile-Cache."""
    ac, cog = client
    x, y, z = _xyz_for_cog(cog)
    params = {"url": str(cog), "index": "ndvi"}
    async with ac:
        first = await ac.get(f"/tiles/{z}/{x}/{y}.png", params=params)
        second = await ac.get(f"/tiles/{z}/{x}/{y}.png", params=params)
    assert first.status_code == 200
    assert first.headers["x-tile-cache"] == "MISS"
    assert first.headers["cache-control"] == "public, max-age=900"
    assert second.status_code == 200
    assert second.headers["x-tile-cache"] == "HIT"
    assert second.content == first.content


async def test_tiles_missing_url_is_422(client: tuple[AsyncClient, Path]) -> None:
    """AC-7: a missing ``url`` query param is a 422 (FastAPI validation)."""
    ac, cog = client
    x, y, z = _xyz_for_cog(cog)
    async with ac:
        resp = await ac.get(f"/tiles/{z}/{x}/{y}.png", params={"index": "ndvi"})
    assert resp.status_code == 422


async def test_tiles_ndmi_is_honest_422(client: tuple[AsyncClient, Path]) -> None:
    """AC-7/R5: NDMI on a 4-band COG (no SWIR) is an honest 422, not a fake tile."""
    ac, cog = client
    x, y, z = _xyz_for_cog(cog)
    async with ac:
        resp = await ac.get(f"/tiles/{z}/{x}/{y}.png", params={"url": str(cog), "index": "ndmi"})
    assert resp.status_code == 422
    assert "SWIR" in resp.json()["detail"]


async def test_tiles_unreadable_cog_is_404(client: tuple[AsyncClient, Path]) -> None:
    """AC-7: a non-existent COG maps to 404 (no bare 500)."""
    ac, _cog = client
    async with ac:
        resp = await ac.get(
            "/tiles/14/1/1.png", params={"url": "/no/such/file.tif", "index": "ndvi"}
        )
    assert resp.status_code == 404


async def test_tiles_ssrf_url_is_403(client: tuple[AsyncClient, Path]) -> None:
    """SSRF guard: an arbitrary remote host in ``url`` is rejected with 403.

    ``http://169.254.169.254/...`` (cloud metadata) is not in
    ``tile_url_allowed_hosts`` -> the tiler never fetches it server-side.
    """
    ac, _cog = client
    async with ac:
        resp = await ac.get(
            "/tiles/14/1/1.png",
            params={"url": "http://169.254.169.254/latest/meta-data/x.tif", "index": "ndvi"},
        )
    assert resp.status_code == 403
