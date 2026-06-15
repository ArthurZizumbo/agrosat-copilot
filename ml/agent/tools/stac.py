"""``search_stac`` tool: STAC item search backed by pgstac (deferred).

This deferred tool queries a STAC catalogue stored in PostgreSQL through the
``pgstac.search`` function, which evaluates a STAC API ``search`` request encoded
as a single JSON document and returns a GeoJSON ``FeatureCollection`` of matching
items.

pgstac is **not deployed yet** in this project: the ``CREATE EXTENSION pgstac``
statement is commented out in the initial migration because it requires a
container image bundling the extension. This tool therefore implements the real
query against ``pgstac.search`` but degrades gracefully when the function does
not exist: it catches the "undefined function / schema" class of asyncpg errors,
logs ``pgstac_not_deployed`` and returns an empty :class:`SceneList`. It never
fabricates synthetic scenes.

The query still runs inside :func:`ml.agent.db.session_scoped_conn` so the
per-session RLS hook is primed, consistent with every other DB-touching tool.
"""

from __future__ import annotations

import json
import time

import asyncpg
import structlog

from ml.agent.context import ToolContext
from ml.agent.db import session_scoped_conn
from ml.agent.schemas import BBox, SceneList, SearchStacInput

__all__ = ["run"]

logger = structlog.get_logger(__name__)

# asyncpg raises subclasses of these when ``pgstac.search`` (or the ``pgstac``
# schema itself) is absent because the extension was never installed.
_PGSTAC_MISSING_ERRORS: tuple[type[Exception], ...] = (
    asyncpg.UndefinedFunctionError,
    asyncpg.InvalidSchemaNameError,
    asyncpg.UndefinedTableError,
)


def _build_search_request(bbox: BBox, datetime_range: str, cloud_cover_max: float) -> dict:
    """Build the STAC API ``search`` request consumed by ``pgstac.search``.

    Args:
        bbox: Bounding box to search within (EPSG:4326 lon/lat).
        datetime_range: RFC 3339 STAC datetime interval (e.g.
            ``"2019-01-01/2019-12-31"``).
        cloud_cover_max: Maximum acceptable ``eo:cloud_cover`` percentage.

    Returns:
        A JSON-serialisable dict in STAC API ``search`` shape, including a CQL2
        filter pushing the cloud-cover bound down into pgstac.
    """
    return {
        "bbox": [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy],
        "datetime": datetime_range,
        "filter-lang": "cql2-json",
        "filter": {
            "op": "<=",
            "args": [{"property": "eo:cloud_cover"}, cloud_cover_max],
        },
    }


def _extract_scenes(result: object) -> list[dict]:
    """Normalise the ``pgstac.search`` return value into a list of STAC items.

    ``pgstac.search`` returns a GeoJSON ``FeatureCollection`` as JSON. asyncpg
    surfaces it either as a JSON string or as an already-decoded ``dict``
    depending on type codecs; both are handled.

    Args:
        result: Raw scalar returned by ``pgstac.search``.

    Returns:
        The ``features`` list of the collection, or an empty list if absent.
    """
    if result is None:
        return []
    collection = json.loads(result) if isinstance(result, (str, bytes)) else result
    if not isinstance(collection, dict):
        return []
    features = collection.get("features", [])
    return features if isinstance(features, list) else []


async def run(inp: SearchStacInput, ctx: ToolContext) -> SceneList:
    """Search STAC scenes via ``pgstac.search``; empty result if pgstac is absent.

    Args:
        inp: Validated search arguments (bbox, datetime range, cloud-cover max).
        ctx: Tool execution context (session-scoped pool access).

    Returns:
        A :class:`SceneList` with the matching STAC items, or an empty list when
        pgstac is not deployed (logged as ``pgstac_not_deployed``).
    """
    started = time.perf_counter()
    request = _build_search_request(inp.bbox, inp.datetime_range, inp.cloud_cover_max)
    logger.info(
        "tool_call_started",
        tool="search_stac",
        session_id=str(ctx.session_id),
        cloud_cover_max=inp.cloud_cover_max,
        datetime_range=inp.datetime_range,
    )
    try:
        async with session_scoped_conn(ctx.session_id) as conn:
            # pgstac.search takes the STAC search request as a single JSONB arg.
            raw = await conn.fetchval("SELECT pgstac.search($1::jsonb)", json.dumps(request))
        scenes = _extract_scenes(raw)
    except _PGSTAC_MISSING_ERRORS as exc:
        logger.warning(
            "pgstac_not_deployed",
            tool="search_stac",
            session_id=str(ctx.session_id),
            error=str(exc),
        )
        return SceneList(scenes=[], count=0)

    duration_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "tool_call_finished",
        tool="search_stac",
        session_id=str(ctx.session_id),
        count=len(scenes),
        duration_ms=round(duration_ms, 2),
    )
    return SceneList(scenes=scenes, count=len(scenes))
