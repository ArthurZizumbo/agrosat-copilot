"""``/tiles`` router: documented ``501`` stub until US-055 mounts TiTiler.

US-055 will mount ``titiler.core.factory.TilerFactory(...).router`` with
``prefix="/tiles"`` onto the backend image to serve dynamic COG/AlphaEarth tiles.
For US-053 the ``TilerFactory`` is **not** mounted yet (it would couple the image
to rio-tiler/GDAL runtime concerns prematurely), and there is no upstream tiler
to proxy. The honest degradation is therefore a ``501 Not Implemented`` that names
the contract and the US that fulfils it -- not a fabricated empty PNG.

The mount point is marked in ``backend/app/main.py``; replacing this stub router
with the ``TilerFactory`` router there is the entire US-055 wiring change.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/tiles", tags=["tiles"])

#: Body returned by the stub so the frontend (and tests) can distinguish a
#: deliberate not-yet-implemented state from a real server error.
_NOT_IMPLEMENTED_BODY = {
    "detail": ("Tile serving is mounted by US-055 (TiTiler TilerFactory); not yet available."),
    "contract": "GET /tiles/{z}/{x}/{y}.png",
}


@router.get("/{z}/{x}/{y}.png")
async def get_tile(z: int, x: int, y: int) -> JSONResponse:
    """Return a documented ``501`` until US-055 mounts the TiTiler ``TilerFactory``.

    Args:
        z: Tile zoom level (XYZ scheme).
        x: Tile column.
        y: Tile row.

    Returns:
        A JSON ``501 Not Implemented`` response describing the deferred contract.
    """
    logger.info("tiles_not_implemented", z=z, x=x, y=y)
    return JSONResponse(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        content=_NOT_IMPLEMENTED_BODY,
    )
