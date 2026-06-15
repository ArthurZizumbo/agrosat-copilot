"""``get_tiles`` tool: build a TiTiler XYZ tile-template URL (deferred).

This deferred tool composes the XYZ tile-template URL that the frontend MapLibre
raster source consumes to render a STAC scene as a spectral index (NDVI/NDWI/EVI)
or natural-colour RGB. It performs **no** network I/O: it only assembles the URL
string from the configured TiTiler host/port and the requested scene/index, so
it is cheap, deterministic and safe to call inline despite being flagged
deferred (the deferral is for consistency with the other backend-dependent
tools and because TiTiler may not be deployed in every environment).

TiTiler is reached at ``http://localhost:{titiler_host_port}`` where
``titiler_host_port`` defaults to ``8001`` in
:class:`backend.app.core.config.Settings`. The STAC tiler endpoint renders a
single STAC item identified by ``scene_id``; spectral indices are expressed as
TiTiler ``expression`` parameters over the Sentinel-2 bands, while ``rgb`` maps
to a true-colour band selection.
"""

from __future__ import annotations

import time
from urllib.parse import urlencode

import structlog

from ml.agent.context import ToolContext
from ml.agent.schemas import GetTilesInput, TileUrl

__all__ = ["run"]

logger = structlog.get_logger(__name__)

# Default TiTiler host used when settings expose no ``titiler_host_port``.
_DEFAULT_TITILER_PORT = 8001

# TiTiler band-math ``expression`` per spectral index, over Sentinel-2 L2A assets
# (B08 = NIR, B04 = red, B03 = green, B02 = blue, B8A used for EVI's NIR). These
# are standard normalised-difference / enhanced-vegetation formulations.
_INDEX_EXPRESSION: dict[str, str] = {
    "ndvi": "(B08-B04)/(B08+B04)",
    "ndwi": "(B03-B08)/(B03+B08)",
    "evi": "2.5*(B08-B04)/(B08+6*B04-7.5*B02+1)",
}

# Per-index single-band colormap (TiTiler ``colormap_name``); RGB uses an
# explicit asset/band selection instead of a colormap.
_INDEX_COLORMAP: dict[str, str] = {
    "ndvi": "rdylgn",
    "ndwi": "blues",
    "evi": "rdylgn",
}

# Natural-colour band selection for the ``rgb`` product (red, green, blue).
_RGB_ASSETS: tuple[tuple[str, str], ...] = (
    ("assets", "B04"),
    ("assets", "B03"),
    ("assets", "B02"),
)


def _titiler_base_url(ctx: ToolContext) -> str:
    """Resolve the TiTiler base URL from settings.

    Reads ``titiler_host_port`` from the typed settings when present, falling
    back to the documented default port. Host is ``localhost`` (the docker
    service is published on the host in dev); cloud deployments override the
    port via ``.env.local``.

    Args:
        ctx: Tool execution context carrying the typed settings.

    Returns:
        The TiTiler base URL, e.g. ``http://localhost:8001``.
    """
    port = getattr(ctx.settings, "titiler_host_port", _DEFAULT_TITILER_PORT)
    return f"http://localhost:{port}"


def _build_tile_url(base_url: str, scene_id: str, index: str) -> str:
    """Assemble the XYZ tile-template URL for a scene/index against TiTiler.

    The path keeps the literal ``{z}/{x}/{y}`` placeholders that MapLibre fills
    in per tile request; the rendering parameters (expression + colormap, or RGB
    band selection) are appended as a query string.

    Args:
        base_url: TiTiler base URL (scheme, host, port).
        scene_id: STAC item identifier to render.
        index: One of ``ndvi``, ``ndwi``, ``evi`` or ``rgb``.

    Returns:
        The fully-composed XYZ tile-template URL.
    """
    path = f"{base_url}/stac/tiles/{{z}}/{{x}}/{{y}}"
    params: list[tuple[str, str]] = [("url", scene_id)]
    if index == "rgb":
        params.extend(_RGB_ASSETS)
    else:
        params.append(("expression", _INDEX_EXPRESSION[index]))
        params.append(("colormap_name", _INDEX_COLORMAP[index]))
        params.append(("rescale", "-1,1"))
    # ``safe="{}/"`` keeps the XYZ placeholders and slashes literal (they live in
    # the path, but urlencode only touches the query, so this only guards values).
    query = urlencode(params, safe="{}")
    return f"{path}?{query}"


async def run(inp: GetTilesInput, ctx: ToolContext) -> TileUrl:
    """Build the TiTiler XYZ tile-template URL for a scene and visual product.

    Args:
        inp: Validated arguments (scene id and index/RGB product).
        ctx: Tool execution context carrying the typed settings.

    Returns:
        A :class:`TileUrl` echoing the scene/index and carrying the XYZ template.
    """
    started = time.perf_counter()
    logger.info(
        "tool_call_started",
        tool="get_tiles",
        session_id=str(ctx.session_id),
        scene_id=inp.scene_id,
        index=inp.index,
    )
    base_url = _titiler_base_url(ctx)
    tile_url = _build_tile_url(base_url, inp.scene_id, inp.index)
    duration_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "tool_call_finished",
        tool="get_tiles",
        session_id=str(ctx.session_id),
        scene_id=inp.scene_id,
        index=inp.index,
        duration_ms=round(duration_ms, 2),
    )
    return TileUrl(scene_id=inp.scene_id, index=inp.index, tile_url=tile_url)
