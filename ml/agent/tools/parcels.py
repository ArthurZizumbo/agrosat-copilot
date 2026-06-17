"""``list_parcels`` tool: list the session's parcels, optionally within an AOI.

This synchronous demo tool reads the ``parcels`` table for the active session.
Every row is filtered by ``session_id`` (multi-tenant defence in depth) and the
query runs inside :func:`ml.agent.db.session_scoped_conn` so the per-session RLS
hook is primed. When the caller supplies an ``aoi`` polygon, the listing is
spatially restricted with ``ST_Intersects`` against the parcel geometry built
from the GeoJSON via ``ST_GeomFromGeoJSON``.

No data is fabricated: if the session owns no parcels (or none intersect the
AOI), an empty :class:`ParcelList` with ``count == 0`` is returned.
"""

from __future__ import annotations

import json
import time

import structlog

from ml.agent.context import ToolContext
from ml.agent.db import session_scoped_conn
from ml.agent.schemas import ListParcelsInput, ParcelList, ParcelRef

__all__ = ["run"]

logger = structlog.get_logger(__name__)

# Listing is capped so a runaway session cannot stream unbounded rows into the
# LLM context. Parcels are returned in stable ``id`` order for determinism.
_MAX_PARCELS: int = 1000

# Base query: parcels of the session, newest geometry id first is irrelevant;
# stable ascending ``id`` order keeps results reproducible across calls.
_LIST_SQL_NO_AOI = """
SELECT id, crop_class, confidence
FROM parcels
WHERE session_id = $1
ORDER BY id ASC
LIMIT $2
"""

# AOI-restricted variant: only parcels whose geometry intersects the supplied
# GeoJSON polygon (assumed EPSG:4326, matching ``parcels.geom`` SRID).
_LIST_SQL_WITH_AOI = """
SELECT id, crop_class, confidence
FROM parcels
WHERE session_id = $1
  AND ST_Intersects(geom, ST_SetSRID(ST_GeomFromGeoJSON($2), 4326))
ORDER BY id ASC
LIMIT $3
"""


async def run(inp: ListParcelsInput, ctx: ToolContext) -> ParcelList:
    """List the parcels visible to the session, optionally clipped to an AOI.

    Args:
        inp: Validated arguments (session id and optional AOI polygon).
        ctx: Tool execution context (session-scoped pool access).

    Returns:
        A :class:`ParcelList` of the matching parcels (empty when the session
        owns no parcels or none intersect the AOI).
    """
    started = time.perf_counter()
    logger.info(
        "tool_call_started",
        tool="list_parcels",
        session_id=str(ctx.session_id),
        has_aoi=inp.aoi is not None,
    )

    async with session_scoped_conn(inp.session_id) as conn:
        if inp.aoi is None:
            records = await conn.fetch(_LIST_SQL_NO_AOI, inp.session_id, _MAX_PARCELS)
        else:
            aoi_geojson = json.dumps(inp.aoi.model_dump())
            records = await conn.fetch(
                _LIST_SQL_WITH_AOI,
                inp.session_id,
                aoi_geojson,
                _MAX_PARCELS,
            )

    parcels = [
        ParcelRef(
            parcel_id=record["id"],
            crop_class=record["crop_class"],
            confidence=record["confidence"],
        )
        for record in records
    ]

    duration_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "tool_call_finished",
        tool="list_parcels",
        session_id=str(ctx.session_id),
        count=len(parcels),
        duration_ms=round(duration_ms, 2),
    )
    return ParcelList(parcels=parcels, count=len(parcels))
