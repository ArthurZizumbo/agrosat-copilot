"""Integration tests for the TiTiler ``/cog`` factory + ``/tiles`` adapter (US-055).

These tests exercise the mounted ``TilerFactory`` surface and the cross-cutting
HTTP concerns (CORS, HTTP cache headers, Redis tile cache) against a **REAL**
Cloud-Optimized GeoTIFF -- never an ``np.random`` raster.

REAL COG (regla Arthur, cero sintetico): the ``real_cog`` session fixture
(``conftest.py``) takes the **untouched** Sentinel-2 pixels of a farslip crop
(``data/farslip_pairs/pianura_padana/crops/<crop_id>.tif`` -- 4-band uint16
256x256, real S2 reflectance) and the crop's **real** lat/lon from
``manifest.parquet`` (e.g. ``45.0890, 9.6022``, Pianura Padana, ``cap_class=soia``).
The raw crop is *not* a COG (``crs=None``, no overviews, untiled), so the fixture
assigns a legitimate georeference -- CRS ``EPSG:3857`` + a ``from_origin``
transform centred on the real coordinates at 10 m/px (S2 GSD) -- and runs the
pixels through ``rio_cogeo.cog_translate(web_optimized=True)``; ``cog_validate``
returns True. Only georeferencing is added: the pixel values are the real ones.

Band order is the farslip S2 convention ``[B02, B03, B04, B08]`` -> 1-indexed
``b3=Red, b4=NIR`` -> NDVI ``= (b4-b3)/(b4+b3) = (B8-B4)/(B8+B4)``.

The app is driven through ``httpx.AsyncClient`` + ``ASGITransport`` (no live
server, no network); the Redis tile cache is backed by ``fakeredis``.

Covers AC-2 (``/cog`` factory literal endpoint + extras), AC-3 (Redis cache key),
AC-4 (HTTP cache + CORS headers) and AC-5 (``/tiles`` contract no longer 501).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import fakeredis.aioredis
import morecantile
import pytest
import rasterio
from httpx import ASGITransport, AsyncClient
from rasterio.warp import transform as warp_transform

from backend.app.api.tiles import get_tile_redis
from backend.app.main import create_app
from backend.app.services.tile_cache import tile_cache_key

#: PNG magic bytes -- the first 8 octets of every PNG file.
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

#: NDVI band-math on the 4-band farslip COG (1-indexed: b3=Red, b4=NIR).
_NDVI_EXPRESSION = "(b4-b3)/(b4+b3)"

#: A configured CORS origin (matches ``settings.cors_allow_origins``).
_ALLOWED_ORIGIN = "http://localhost:3000"


def _xyz_for_cog(cog: Path, zoom: int = 14) -> tuple[int, int, int]:
    """Return an XYZ ``(x, y, z)`` tile covering the COG centre (WebMercatorQuad).

    Args:
        cog: Path to the EPSG:3857 COG.
        zoom: WebMercatorQuad zoom level.

    Returns:
        The ``(x, y, z)`` of the tile containing the COG centroid.
    """
    tms = morecantile.tms.get("WebMercatorQuad")
    with rasterio.open(cog) as src:
        bounds = src.bounds
    cx = (bounds.left + bounds.right) / 2
    cy = (bounds.top + bounds.bottom) / 2
    lon, lat = warp_transform("EPSG:3857", "EPSG:4326", [cx], [cy])
    tile = tms.tile(lon[0], lat[0], zoom)
    return tile.x, tile.y, zoom


@pytest.fixture()
def client(real_cog: Path) -> Iterator[tuple[AsyncClient, Path, fakeredis.aioredis.FakeRedis]]:
    """App client with the tile-cache Redis overridden by a shared fakeredis.

    The same ``fakeredis`` instance is returned so tests can assert directly on
    the cache key the adapter writes.

    Args:
        real_cog: Session-scoped real COG path (from ``conftest``).

    Yields:
        ``(client, cog_path, fake_redis)``.
    """
    app = create_app()
    shared = fakeredis.aioredis.FakeRedis()
    app.dependency_overrides[get_tile_redis] = lambda: shared
    transport = ASGITransport(app=app)
    yield AsyncClient(transport=transport, base_url="http://test"), real_cog, shared
    app.dependency_overrides.clear()


# --------------------------------------------------------------------------- #
# AC-2 -- the TiTiler ``/cog`` factory surface (literal endpoint + extras).
# --------------------------------------------------------------------------- #


async def test_cog_factory_literal_endpoint_renders_png(
    client: tuple[AsyncClient, Path, fakeredis.aioredis.FakeRedis],
) -> None:
    """AC-2 literal: ``/cog/tiles/{TMS}/{z}/{x}/{y}.png?url&expression&rescale&colormap_name``.

    The factory returns a valid PNG (magic bytes + ``image/png``) rendered from
    the real COG via the explicit NDVI expression, rescale and colormap.
    """
    ac, cog, _redis = client
    x, y, z = _xyz_for_cog(cog)
    async with ac:
        resp = await ac.get(
            f"/cog/tiles/WebMercatorQuad/{z}/{x}/{y}.png",
            params={
                "url": str(cog),
                "expression": _NDVI_EXPRESSION,
                "rescale": "-1,1",
                "colormap_name": "rdylgn",
            },
        )
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/png"
    assert resp.content.startswith(_PNG_SIGNATURE)
    assert len(resp.content) > len(_PNG_SIGNATURE)


async def test_cog_info_reports_four_real_bands(
    client: tuple[AsyncClient, Path, fakeredis.aioredis.FakeRedis],
) -> None:
    """AC-2 extra: ``/cog/info`` describes the real 4-band uint16 COG."""
    ac, cog, _redis = client
    async with ac:
        resp = await ac.get("/cog/info", params={"url": str(cog)})
    assert resp.status_code == 200, resp.text
    info = resp.json()
    assert info["count"] == 4  # real farslip S2 [B02, B03, B04, B08].
    assert info["dtype"] == "uint16"
    # web-optimized COG: EPSG:3857 georeference assigned over the real pixels.
    assert "3857" in info["crs"]


async def test_cog_tilejson_exposes_tile_template(
    client: tuple[AsyncClient, Path, fakeredis.aioredis.FakeRedis],
) -> None:
    """AC-2 extra: ``/cog/{TMS}/tilejson.json`` returns a usable XYZ tile template."""
    ac, cog, _redis = client
    async with ac:
        resp = await ac.get(
            "/cog/WebMercatorQuad/tilejson.json",
            params={"url": str(cog), "expression": _NDVI_EXPRESSION},
        )
    assert resp.status_code == 200, resp.text
    doc = resp.json()
    assert doc["tilejson"].startswith("3.")
    assert any("{z}/{x}/{y}" in tpl for tpl in doc["tiles"])


async def test_cog_point_returns_real_pixel_values(
    client: tuple[AsyncClient, Path, fakeredis.aioredis.FakeRedis],
) -> None:
    """AC-2 extra: ``/cog/point/{lon},{lat}`` samples the real S2 reflectance."""
    ac, cog, _redis = client
    with rasterio.open(cog) as src:
        bounds = src.bounds
    cx = (bounds.left + bounds.right) / 2
    cy = (bounds.top + bounds.bottom) / 2
    lon, lat = warp_transform("EPSG:3857", "EPSG:4326", [cx], [cy])
    async with ac:
        resp = await ac.get(f"/cog/point/{lon[0]},{lat[0]}", params={"url": str(cog)})
    assert resp.status_code == 200, resp.text
    values = resp.json()["values"]
    assert len(values) == 4  # one sample per real band.


# --------------------------------------------------------------------------- #
# AC-5 -- the US-053 ``/tiles`` contract is the real adapter (not 501).
# --------------------------------------------------------------------------- #


async def test_tiles_contract_not_501(
    client: tuple[AsyncClient, Path, fakeredis.aioredis.FakeRedis],
) -> None:
    """AC-5: the frontend ``/tiles/{z}/{x}/{y}.png`` contract responds 200, not 501."""
    ac, cog, _redis = client
    x, y, z = _xyz_for_cog(cog)
    async with ac:
        resp = await ac.get(
            f"/tiles/{z}/{x}/{y}.png",
            params={"url": str(cog), "index": "ndvi"},
        )
    assert resp.status_code != 501
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "image/png"
    assert resp.content.startswith(_PNG_SIGNATURE)


# --------------------------------------------------------------------------- #
# AC-4 -- HTTP cache headers + CORS on the tile surfaces.
# --------------------------------------------------------------------------- #


async def test_tile_has_http_cache_headers(
    client: tuple[AsyncClient, Path, fakeredis.aioredis.FakeRedis],
) -> None:
    """AC-4: every ``/tiles`` response carries ``Cache-Control`` + ``X-Tile-Cache``."""
    ac, cog, _redis = client
    x, y, z = _xyz_for_cog(cog)
    async with ac:
        resp = await ac.get(
            f"/tiles/{z}/{x}/{y}.png",
            params={"url": str(cog), "index": "ndvi"},
        )
    assert resp.status_code == 200, resp.text
    assert resp.headers["cache-control"] == "public, max-age=900"
    assert resp.headers["x-tile-cache"] in {"HIT", "MISS"}


async def test_tile_request_echoes_cors_origin(
    client: tuple[AsyncClient, Path, fakeredis.aioredis.FakeRedis],
) -> None:
    """AC-4: a tile GET with an allowed ``Origin`` echoes the CORS allow-origin header."""
    ac, cog, _redis = client
    x, y, z = _xyz_for_cog(cog)
    async with ac:
        resp = await ac.get(
            f"/tiles/{z}/{x}/{y}.png",
            params={"url": str(cog), "index": "ndvi"},
            headers={"Origin": _ALLOWED_ORIGIN},
        )
    assert resp.status_code == 200, resp.text
    assert resp.headers["access-control-allow-origin"] == _ALLOWED_ORIGIN


async def test_tile_cors_preflight_allows_get(
    client: tuple[AsyncClient, Path, fakeredis.aioredis.FakeRedis],
) -> None:
    """AC-4: an ``OPTIONS`` preflight from an allowed origin permits ``GET`` on /tiles."""
    ac, cog, _redis = client
    x, y, z = _xyz_for_cog(cog)
    async with ac:
        resp = await ac.options(
            f"/tiles/{z}/{x}/{y}.png",
            headers={
                "Origin": _ALLOWED_ORIGIN,
                "Access-Control-Request-Method": "GET",
            },
        )
    assert resp.status_code in {200, 204}
    assert resp.headers["access-control-allow-origin"] == _ALLOWED_ORIGIN
    assert "GET" in resp.headers["access-control-allow-methods"]


# --------------------------------------------------------------------------- #
# AC-3 -- Redis tile cache: second identical call is a HIT; key is in fakeredis.
# --------------------------------------------------------------------------- #


async def test_second_identical_call_hits_redis_cache(
    client: tuple[AsyncClient, Path, fakeredis.aioredis.FakeRedis],
) -> None:
    """AC-3: the 2nd identical request is served from Redis (HIT) with the same bytes.

    Asserts the documented cache key (``tile:sha256(method|path|sorted(qs))``) is
    actually written to the shared ``fakeredis`` after the first (MISS) call, so
    the second call short-circuits before rio-tiler.
    """
    ac, cog, redis = client
    x, y, z = _xyz_for_cog(cog)
    params = {"url": str(cog), "index": "ndvi"}
    async with ac:
        first = await ac.get(f"/tiles/{z}/{x}/{y}.png", params=params)
        # The expected key is the hash of the full endpoint as sent by httpx.
        query = str(first.request.url.query, "utf-8")
        expected_key = tile_cache_key("GET", f"/tiles/{z}/{x}/{y}.png", query)
        cached_blob = await redis.get(expected_key)
        second = await ac.get(f"/tiles/{z}/{x}/{y}.png", params=params)

    assert first.status_code == 200, first.text
    assert first.headers["x-tile-cache"] == "MISS"
    # The MISS populated the documented cache key in fakeredis.
    assert cached_blob is not None
    assert cached_blob == first.content
    assert second.status_code == 200
    assert second.headers["x-tile-cache"] == "HIT"
    assert second.content == first.content
