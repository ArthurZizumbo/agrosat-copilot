"""``/tiles`` router: the stable US-053 contract over the COG render service.

US-055 replaces the documented ``501`` stub with the real tile adapter. The
public contract ``GET /tiles/{z}/{x}/{y}.png?url&index`` (US-053) is unchanged:
the adapter fixes ``WebMercatorQuad`` (the MapLibre/XYZ default) and resolves the
``index`` (ndvi/ndwi/ndmi) to an expression/rescale/colormap, then delegates to
the shared :func:`~backend.app.services.cog_tiler.render_cog_tile` wrapped by the
Redis tile cache. It does **not** reimplement tiling.

The full TiTiler ``TilerFactory`` is mounted separately under ``/cog`` in
``main.py`` for the literal AC endpoint with free ``expression``/``colormap``.
"""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import Response

from backend.app.services.cog_tiler import (
    CogUrlNotAllowedError,
    TileRenderError,
    UnsupportedIndexError,
    render_cog_tile,
    resolve_index,
)
from backend.app.services.tile_cache import RedisLike, cached_tile, make_redis

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/tiles", tags=["tiles"])


def get_tile_redis() -> RedisLike:
    """Provide the Redis client for the tile cache (overridable in tests).

    Returns:
        The async Redis client backed by ``settings.redis_url``.
    """
    return make_redis()


@router.get(
    "/{z}/{x}/{y}.png",
    responses={
        200: {"content": {"image/png": {}}, "description": "Rendered XYZ tile."},
        422: {"description": "Missing ``url`` or an unsupported ``index``."},
    },
)
async def get_tile(
    request: Request,
    z: int,
    x: int,
    y: int,
    url: Annotated[str, Query(description="COG location (path, file://, http(s)://, gs://).")],
    redis: Annotated[RedisLike, Depends(get_tile_redis)],
    index: Annotated[str, Query(description="Spectral index: ndvi | ndwi | ndmi.")] = "ndvi",
) -> Response:
    """Render an NDVI/NDWI XYZ PNG tile for a COG (US-053 contract).

    Fixes ``WebMercatorQuad``, resolves ``index`` to render parameters, and serves
    through the Redis tile cache (``X-Tile-Cache: HIT|MISS``).

    Args:
        request: Incoming request (forms the cache key).
        z: XYZ zoom level.
        x: XYZ tile column.
        y: XYZ tile row.
        url: COG location (local/file/http/gs).
        redis: Tile-cache Redis client (injected; ``fakeredis`` in tests).
        index: Spectral index name (``ndvi`` default).

    Returns:
        A ``200`` ``image/png`` tile response.

    Raises:
        HTTPException: ``422`` if ``index`` is unknown or unsupported (e.g.
            ``ndmi`` on a 4-band COG without SWIR); ``404`` if the COG cannot be
            read.
    """
    try:
        spec = resolve_index(index)
    except UnsupportedIndexError as exc:
        logger.info("tile_index_unsupported", index=index, z=z, x=x, y=y)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    # ``resolve_index`` guarantees a non-None expression for a supported index;
    # bind it to a local so the type narrows inside the closure below.
    expression = spec.expression
    assert expression is not None

    async def _render() -> bytes:
        try:
            png: bytes = render_cog_tile(
                url,
                z,
                x,
                y,
                expression=expression,
                rescale=spec.rescale,
                colormap_name=spec.colormap_name,
            )
        except CogUrlNotAllowedError as exc:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"COG url not allowed: {exc}",
            ) from exc
        except TileRenderError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"COG not readable: {exc}",
            ) from exc
        return png

    response: Response = await cached_tile(request, _render, redis=redis)
    return response
