"""COG tiling service: TiTiler ``TilerFactory`` + a shared render helper.

Two tile surfaces share one render path (DRY):

- ``/cog`` -- the full :class:`~titiler.core.factory.TilerFactory` (mounted in
  ``main.py``), giving the literal AC endpoint
  ``GET /cog/tiles/{tileMatrixSetId}/{z}/{x}/{y}.png?url&expression&rescale&colormap_name``
  plus ``/cog/info``, ``/cog/tilejson.json``, ``/cog/statistics`` for free.
- ``/tiles`` -- the stable US-053 contract ``GET /tiles/{z}/{x}/{y}.png?url&index``.
  The adapter (in ``api/tiles.py``) fixes ``WebMercatorQuad`` and resolves the
  index to an expression/rescale/colormap, then calls :func:`render_cog_tile`.

:func:`render_cog_tile` renders the same way TiTiler does internally
(``rio_tiler.io.Reader`` over a ``morecantile`` TMS, expression -> rescale ->
colormap -> PNG), wrapped in the per-request GDAL/VSI env from
:func:`~backend.app.services.gdal_env.gdal_gcs_environment`. This keeps the
adapter a thin router and the render logic in the service layer.
"""

from __future__ import annotations

# isort: off
# PROJ_DATA must be pinned before rasterio/rio-tiler/titiler initialise GDAL's
# PROJ on the Windows dev box (Riesgo R1). This side-effect import MUST stay
# above the rasterio/rio-tiler/titiler imports below -- do not reorder.
from backend.app.services import proj_env as _proj_env  # noqa: F401

import structlog
from rasterio import Env as RasterioEnv
from rasterio.errors import RasterioError
from rio_tiler.colormap import cmap as default_colormaps
from rio_tiler.errors import RioTilerError
from rio_tiler.io import Reader
from titiler.core.factory import TilerFactory

# isort: on
from urllib.parse import urlparse

from backend.app.core.config import get_settings
from backend.app.services.gdal_env import gdal_gcs_environment

logger = structlog.get_logger(__name__)

#: Local/file COGs are always allowed (dev + the real farslip fixtures); ``gs://``
#: is allowed only for the configured data bucket; ``http(s)://`` only for hosts
#: in ``settings.tile_url_allowed_hosts`` (localhost in dev). Anything else is
#: rejected before GDAL fetches it -- closes the SSRF surface of the open ``url``
#: param (a remote tiler would otherwise fetch any host server-side).
_LOCAL_SCHEMES = frozenset({"", "file"})


class CogUrlNotAllowedError(Exception):
    """Raised when a COG ``url`` is outside the SSRF allowlist."""


def validate_cog_url(url: str) -> None:
    """Reject COG URLs outside the allowlist (SSRF guard for the ``url`` param).

    Local paths and ``file://`` are always allowed; ``gs://`` only for
    ``settings.gcs_data_bucket``; ``http(s)://`` only for an explicitly allowed
    host (``settings.tile_url_allowed_hosts``). Everything else raises so GDAL
    never makes a server-side request to an attacker-controlled host.

    Args:
        url: The COG location from the request.

    Raises:
        CogUrlNotAllowedError: If the URL is not in the allowlist.
    """
    # A Windows path ("C:\\...") parses as scheme "c"; a POSIX path has no
    # scheme. Single-letter schemes are drive letters -> treat as local paths.
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme in _LOCAL_SCHEMES or len(scheme) == 1:
        return
    settings = get_settings()
    if scheme == "gs":
        bucket = settings.gcs_data_bucket
        if bucket and parsed.netloc == bucket:
            return
        raise CogUrlNotAllowedError(f"gs:// bucket not allowed: {parsed.netloc!r}")
    if scheme in ("http", "https"):
        allowed = {h.strip() for h in settings.tile_url_allowed_hosts.split(",") if h.strip()}
        if parsed.hostname in allowed:
            return
        raise CogUrlNotAllowedError(f"http(s) host not allowed: {parsed.hostname!r}")
    raise CogUrlNotAllowedError(f"COG url scheme not allowed: {scheme!r}")


__all__ = [
    "INDEX_RENDER",
    "TILE_SIZE",
    "CogUrlNotAllowedError",
    "IndexRender",
    "TileRenderError",
    "UnsupportedIndexError",
    "cog_tiler",
    "render_cog_tile",
    "resolve_index",
    "validate_cog_url",
]

#: XYZ tile edge in pixels (the MapLibre/WebMercatorQuad default).
TILE_SIZE = 256

#: TiTiler factory singleton. ``environment_dependency`` injects the GDAL/VSI env
#: (``vsigs`` + range tuning) into a per-request ``rasterio.Env`` -- this replaces
#: the deprecated ``gdal_config=`` argument. Mounted under ``/cog`` in ``main.py``.
cog_tiler: TilerFactory = TilerFactory(environment_dependency=gdal_gcs_environment)


class TileRenderError(Exception):
    """Raised when a COG cannot be read/rendered (bad url, unreadable raster)."""


class UnsupportedIndexError(Exception):
    """Raised when a requested ``index`` cannot be computed from the COG bands."""


class IndexRender:
    """Render parameters for a named spectral index on the 4-band farslip/S2 COG.

    Band order is the farslip Sentinel-2 convention ``[B02, B03, B04, B08]`` ->
    1-indexed ``b1=blue, b2=green, b3=red, b4=NIR``. The AC writes NDVI as
    ``(B8-B4)/(B8+B4)``, i.e. ``(NIR-Red)`` = ``(b4-b3)/(b4+b3)`` in COG band
    indices.

    Attributes:
        expression: rio-tiler band-math expression (1-indexed bands), or ``None``
            when the index is not computable (no suitable band present).
        rescale: ``(min, max)`` input range mapped to ``0..255`` before colormap.
        colormap_name: A registered rio-tiler colormap name.
        unsupported_reason: Human-readable reason when ``expression`` is ``None``.
    """

    def __init__(
        self,
        expression: str | None,
        rescale: tuple[float, float],
        colormap_name: str,
        unsupported_reason: str | None = None,
    ) -> None:
        self.expression = expression
        self.rescale = rescale
        self.colormap_name = colormap_name
        self.unsupported_reason = unsupported_reason


#: ``index`` (US-053 contract) -> render parameters. NDMI is honestly
#: unsupported on a 4-band COG (no SWIR): the adapter returns ``422`` rather than
#: fabricating a fake NDMI (Arthur's real-data rule).
INDEX_RENDER: dict[str, IndexRender] = {
    "ndvi": IndexRender("(b4-b3)/(b4+b3)", (-1.0, 1.0), "rdylgn"),
    "ndwi": IndexRender("(b2-b4)/(b2+b4)", (-1.0, 1.0), "rdbu_r"),
    "ndmi": IndexRender(
        None,
        (-1.0, 1.0),
        "viridis",
        unsupported_reason="ndmi requires a SWIR band, not present in the 4-band COG",
    ),
}


def resolve_index(index: str) -> IndexRender:
    """Resolve an ``index`` name to its render parameters.

    Args:
        index: One of ``ndvi`` / ``ndwi`` / ``ndmi`` (case-insensitive).

    Returns:
        The :class:`IndexRender` for the index.

    Raises:
        UnsupportedIndexError: If the index is unknown, or known but not
            computable from the available bands (e.g. ``ndmi`` without SWIR).
    """
    render = INDEX_RENDER.get(index.lower())
    if render is None:
        raise UnsupportedIndexError(
            f"unknown index {index!r}; expected one of {sorted(INDEX_RENDER)}"
        )
    if render.expression is None:
        raise UnsupportedIndexError(render.unsupported_reason or f"index {index!r} unsupported")
    return render


def render_cog_tile(
    url: str,
    z: int,
    x: int,
    y: int,
    *,
    expression: str,
    rescale: tuple[float, float],
    colormap_name: str,
) -> bytes:
    """Render one XYZ PNG tile from a COG (shared by ``/tiles`` and the cache).

    Renders exactly like TiTiler does internally -- ``rio_tiler.io.Reader`` over
    the default ``WebMercatorQuad`` TMS, applying the band-math expression, the
    rescale, then the colormap -- inside the per-request GDAL/VSI env so ``gs://``
    (vsigs) and local/http COGs both work.

    Args:
        url: COG location (local path, ``file://``, ``http(s)://`` or ``gs://``).
        z: XYZ zoom level.
        x: XYZ tile column.
        y: XYZ tile row.
        expression: rio-tiler band-math expression (1-indexed bands).
        rescale: ``(min, max)`` input range mapped to ``0..255``.
        colormap_name: Registered rio-tiler colormap name.

    Returns:
        PNG-encoded tile bytes.

    Raises:
        TileRenderError: If the COG cannot be opened or the tile cannot be read.
    """
    validate_cog_url(url)
    colormap = default_colormaps.get(colormap_name)
    try:
        with RasterioEnv(**gdal_gcs_environment()):
            with Reader(url) as reader:
                image = reader.tile(x, y, z, tilesize=TILE_SIZE, expression=expression)
            image.rescale(in_range=(rescale,))
            tile = image.apply_colormap(colormap)
            png = tile.render(img_format="PNG")
    except (RioTilerError, RasterioError) as exc:
        logger.info("cog_tile_render_failed", url=url, z=z, x=x, y=y, error=str(exc))
        raise TileRenderError(str(exc)) from exc
    return bytes(png)
