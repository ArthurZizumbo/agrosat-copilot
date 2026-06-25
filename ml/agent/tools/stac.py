"""``search_stac`` tool: real Sentinel-2 scene search via CDSE, pgstac fallback.

This deferred tool resolves Sentinel-2 scenes for a bbox / datetime / cloud-cover
query. It prefers the **Copernicus Data Space Ecosystem (CDSE)** catalogue, the
real source of Sentinel-2 products, and falls back to the in-database
``pgstac.search`` function when CDSE is not configured.

Resolution order:

1. **CDSE (real scenes).** When the CDSE confidential OAuth client-credentials are
   configured (``settings.cdse_client_id`` / ``cdse_client_secret``), the tool
   builds a :class:`ml.ingest.cdse_client.CDSEClient` and calls ``search_s2`` over
   the CDSE OData product catalogue, returning real Sentinel-2 L2A scenes. Each
   :class:`~ml.ingest.cdse_client.CDSEScene` is mapped to a STAC-shaped item dict
   (``id`` / ``bbox`` / ``properties.datetime`` / ``properties.eo:cloud_cover``)
   so the :class:`SceneList` contract is unchanged and every scene stays citable
   by its ``scene_id`` and acquisition date.
2. **pgstac fallback.** When the CDSE credentials are empty (a dev box without
   ``.env.local``), the tool logs ``search_stac_cdse_unavailable`` and degrades
   cleanly to the previous behaviour: querying ``pgstac.search`` inside
   :func:`ml.agent.db.session_scoped_conn` (so the per-session RLS hook is primed,
   consistent with every other DB-touching tool).

pgstac is **not deployed yet** in this project (the ``CREATE EXTENSION pgstac``
statement is commented out in the initial migration), so the fallback itself
degrades gracefully too: it catches the "undefined function / schema" class of
asyncpg errors, logs ``pgstac_not_deployed`` and returns an empty
:class:`SceneList`. The tool never fabricates synthetic scenes.
"""

from __future__ import annotations

import json
import time

import asyncpg
import structlog

from ml.agent.context import ToolContext
from ml.agent.db import session_scoped_conn
from ml.agent.schemas import BBox, SceneList, SearchStacInput
from ml.ingest.cdse_client import (
    CDSECredentialsMissing,
    CDSEScene,
    cdse_client_from_settings,
)

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


def _cdse_credentials_present(settings: object) -> bool:
    """Report whether the CDSE confidential client-credentials are configured.

    The settings object may be the typed app ``Settings`` or a lightweight test
    stub that does not declare the CDSE fields at all; both are handled via
    :func:`getattr` defaults so a missing attribute reads as "not configured"
    rather than raising.

    Args:
        settings: The application settings (or a stub) carried by the tool
            context.

    Returns:
        ``True`` only when both ``cdse_client_id`` and ``cdse_client_secret`` are
        non-empty.
    """
    client_id = getattr(settings, "cdse_client_id", "") or ""
    client_secret = getattr(settings, "cdse_client_secret", "") or ""
    return bool(client_id and client_secret)


def _scene_to_stac_item(scene: CDSEScene) -> dict:
    """Map a :class:`CDSEScene` to a STAC-shaped item dict.

    The returned shape matches the ``features`` produced by ``pgstac.search`` so
    the :class:`SceneList` contract is identical regardless of the backing
    source. Every field needed to cite the scene in ``final_answer`` (the scene
    id, the acquisition datetime and the cloud cover) is preserved.

    Args:
        scene: A scene returned by :meth:`CDSEClient.search_s2`.

    Returns:
        A STAC item dict with ``id``, ``bbox``, ``geometry`` and ``properties``
        (``datetime`` and ``eo:cloud_cover``), tagged with its CDSE origin.
    """
    min_lon, min_lat, max_lon, max_lat = scene.bbox
    return {
        "type": "Feature",
        "id": scene.scene_id,
        "bbox": [min_lon, min_lat, max_lon, max_lat],
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [min_lon, min_lat],
                    [max_lon, min_lat],
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat],
                ]
            ],
        },
        "properties": {
            "datetime": scene.datetime,
            "eo:cloud_cover": scene.cloud_cover,
        },
        "collection": "SENTINEL-2",
        "source": "cdse",
    }


def _search_cdse(inp: SearchStacInput, ctx: ToolContext) -> list[dict]:
    """Search real Sentinel-2 scenes via CDSE and map them to STAC items.

    Args:
        inp: Validated search arguments (bbox, datetime range, cloud-cover max).
        ctx: Tool execution context (its ``settings`` carry the CDSE credentials).

    Returns:
        A list of STAC item dicts (possibly empty when the catalogue returns no
        matching scene). Never fabricates scenes.

    Raises:
        CDSECredentialsMissing: if the credentials turn out to be empty when the
            client is built (guarded by :func:`_cdse_credentials_present`).
    """
    client = cdse_client_from_settings(ctx.settings)
    scenes = client.search_s2(
        bbox=(inp.bbox.minx, inp.bbox.miny, inp.bbox.maxx, inp.bbox.maxy),
        datetime_range=inp.datetime_range,
        cloud_cover_max=inp.cloud_cover_max,
        l2a_only=True,
    )
    return [_scene_to_stac_item(scene) for scene in scenes]


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


async def _run_pgstac_fallback(inp: SearchStacInput, ctx: ToolContext) -> list[dict]:
    """Run the legacy ``pgstac.search`` query; empty list when pgstac is absent.

    Args:
        inp: Validated search arguments (bbox, datetime range, cloud-cover max).
        ctx: Tool execution context (session-scoped pool access).

    Returns:
        The matching STAC item dicts, or an empty list when ``pgstac.search`` is
        not deployed (logged as ``pgstac_not_deployed``).
    """
    request = _build_search_request(inp.bbox, inp.datetime_range, inp.cloud_cover_max)
    try:
        async with session_scoped_conn(ctx.session_id) as conn:
            # pgstac.search takes the STAC search request as a single JSONB arg.
            raw = await conn.fetchval("SELECT pgstac.search($1::jsonb)", json.dumps(request))
        return _extract_scenes(raw)
    except _PGSTAC_MISSING_ERRORS as exc:
        logger.warning(
            "pgstac_not_deployed",
            tool="search_stac",
            session_id=str(ctx.session_id),
            error=str(exc),
        )
        return []


async def run(inp: SearchStacInput, ctx: ToolContext) -> SceneList:
    """Search Sentinel-2 scenes via CDSE, falling back to ``pgstac.search``.

    Prefers real CDSE scenes when the confidential client-credentials are
    configured; otherwise logs ``search_stac_cdse_unavailable`` and degrades to
    the in-database ``pgstac.search`` query (which itself returns an empty list
    when pgstac is not deployed). Never fabricates synthetic scenes.

    Args:
        inp: Validated search arguments (bbox, datetime range, cloud-cover max).
        ctx: Tool execution context (settings carry the CDSE credentials; the
            session-scoped pool backs the pgstac fallback).

    Returns:
        A :class:`SceneList` with the matching STAC items (CDSE-sourced when
        configured, pgstac-sourced otherwise), or an empty list when neither
        source is available.
    """
    started = time.perf_counter()
    logger.info(
        "tool_call_started",
        tool="search_stac",
        session_id=str(ctx.session_id),
        cloud_cover_max=inp.cloud_cover_max,
        datetime_range=inp.datetime_range,
    )

    source = "cdse"
    if _cdse_credentials_present(ctx.settings):
        try:
            scenes = _search_cdse(inp, ctx)
        except CDSECredentialsMissing as exc:
            # Defence in depth: the presence check passed but the client still
            # rejected the pair (e.g. whitespace-only). Degrade to pgstac.
            logger.warning(
                "search_stac_cdse_unavailable",
                tool="search_stac",
                session_id=str(ctx.session_id),
                reason="credentials_missing",
                error=str(exc),
            )
            source = "pgstac"
            scenes = await _run_pgstac_fallback(inp, ctx)
    else:
        logger.warning(
            "search_stac_cdse_unavailable",
            tool="search_stac",
            session_id=str(ctx.session_id),
            reason="credentials_not_configured",
        )
        source = "pgstac"
        scenes = await _run_pgstac_fallback(inp, ctx)

    duration_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "tool_call_finished",
        tool="search_stac",
        session_id=str(ctx.session_id),
        source=source,
        count=len(scenes),
        duration_ms=round(duration_ms, 2),
    )
    return SceneList(scenes=scenes, count=len(scenes))
