"""``add_aoi`` tool: persist a drawn Area Of Interest (deferred).

This deferred tool inserts a user-drawn polygon into the ``aois`` table for the
current session. The geometry arrives as GeoJSON and is materialised in
PostGIS with ``ST_GeomFromGeoJSON`` (forced to SRID 4326 to match the
``GEOMETRY(POLYGON, 4326)`` column); the area in hectares is computed
server-side via ``ST_Area`` over the ``geography`` cast (square metres) divided
by ``10_000``. The ``name`` argument becomes the ``label`` column.

The insert runs inside :func:`ml.agent.db.session_scoped_conn`, so the
per-session RLS hook is primed and ``session_id`` is bound explicitly (defence in
depth) before the row is written. The generated ``id`` is returned as an
:class:`AoiRef`.
"""

from __future__ import annotations

import json
import time

import structlog

from ml.agent.context import ToolContext
from ml.agent.db import session_scoped_conn
from ml.agent.schemas import AddAoiInput, AoiRef

__all__ = ["run"]

logger = structlog.get_logger(__name__)

# Insert the AOI and return the generated id plus the server-computed area.
# ST_SetSRID guards against GeoJSON without an explicit CRS; the column is 4326.
# area_ha is derived from the geography area (m^2) so it is geodesically correct.
_INSERT_AOI_SQL = """
INSERT INTO aois (session_id, geom, label, area_ha)
VALUES (
    $1,
    ST_SetSRID(ST_GeomFromGeoJSON($2), 4326),
    $3,
    ST_Area(ST_SetSRID(ST_GeomFromGeoJSON($2), 4326)::geography) / 10000.0
)
RETURNING id, area_ha
"""


async def run(inp: AddAoiInput, ctx: ToolContext) -> AoiRef:
    """Persist the drawn AOI for the session and return its reference.

    Args:
        inp: Validated arguments (session id, GeoJSON polygon, AOI name).
        ctx: Tool execution context (session-scoped pool access).

    Returns:
        An :class:`AoiRef` with the generated ``aoi_id``, the ``name`` as label
        and the server-computed ``area_ha``.
    """
    started = time.perf_counter()
    geojson = json.dumps({"type": inp.aoi.type, "coordinates": inp.aoi.coordinates})
    logger.info(
        "tool_call_started",
        tool="add_aoi",
        session_id=str(ctx.session_id),
        name=inp.name,
        geometry_type=inp.aoi.type,
    )
    async with session_scoped_conn(ctx.session_id) as conn:
        row = await conn.fetchrow(_INSERT_AOI_SQL, ctx.session_id, geojson, inp.name)

    aoi_id = int(row["id"])
    area_ha = float(row["area_ha"]) if row["area_ha"] is not None else None
    duration_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "tool_call_finished",
        tool="add_aoi",
        session_id=str(ctx.session_id),
        aoi_id=aoi_id,
        area_ha=area_ha,
        duration_ms=round(duration_ms, 2),
    )
    return AoiRef(aoi_id=aoi_id, label=inp.name, area_ha=area_ha)
