"""Copernicus Data Space Ecosystem (CDSE) client -- OAuth2 client-credentials.

Authenticates against the CDSE Keycloak realm with a *confidential server-side*
OAuth client (``client_id`` + ``client_secret``, NEVER a single-page-application
client) and exposes a thin wrapper over the CDSE STAC catalogue to discover
Sentinel-2 scenes by area, date window and maximum cloud cover -- the same
``bbox`` / ``datetime`` / ``cloud_cover`` contract the agent's ``search_stac``
tool already speaks.

The token endpoint and the client-credentials flow are taken verbatim from the
official CDSE documentation (``eu-cdse/documentation``: ``sh_token_url`` =
``https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token``).
The STAC catalogue lives at ``https://catalogue.dataspace.copernicus.eu/stac``.

Credentials are read from settings (``.env.local``, gitignored); this module
never hardcodes them and raises a clear error when they are absent so a missing
secret fails loudly instead of silently fabricating an empty result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

__all__ = ["CDSEClient", "CDSECredentialsMissing", "CDSEScene"]

#: CDSE OData product catalogue (the canonical CDSE discovery API). The STAC
#: ``/search`` endpoint exists but its collection ids are CLMS/CCM products, NOT
#: the standard ``SENTINEL-2`` -- verified empirically that OData is the working
#: path for Sentinel product discovery by area/date/cloud.
_ODATA_PRODUCTS_URL: str = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

#: Sentinel collection name in the OData ``Collection/Name`` filter.
_S2_COLLECTION_NAME: str = "SENTINEL-2"

#: Sentinel-2 L2A product-type token (surface reflectance) used to keep only
#: analysis-ready L2A scenes (the ``MSIL2A`` substring of the product name).
_S2_L2A_TOKEN: str = "MSIL2A"  # noqa: S105 - product-name token, not a secret

#: OData attribute name carrying the scene cloud-cover percentage.
_CLOUD_ATTR: str = "cloudCover"

#: Refresh the access token this many seconds before it actually expires, so an
#: in-flight request never races the expiry boundary.
_TOKEN_REFRESH_MARGIN_S: float = 30.0


class CDSECredentialsMissing(RuntimeError):
    """Raised when the CDSE client-credentials pair is not configured."""


@dataclass(frozen=True)
class CDSEScene:
    """A single Sentinel scene returned by the CDSE STAC search.

    Attributes:
        scene_id: STAC item id (the canonical product name).
        datetime: Acquisition timestamp (ISO-8601, UTC).
        cloud_cover: Scene-level cloud cover percentage in ``[0, 100]``.
        bbox: Scene bounding box ``[min_lon, min_lat, max_lon, max_lat]``.
    """

    scene_id: str
    datetime: str
    cloud_cover: float
    bbox: tuple[float, float, float, float]


class CDSEClient:
    """OAuth2 client-credentials client over the CDSE STAC catalogue.

    The access token is fetched lazily on the first request and cached until
    shortly before its expiry, then transparently refreshed. All HTTP calls go
    through a single injected :class:`httpx.Client` so tests can stub the network.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        *,
        token_url: str,
        http_client: httpx.Client | None = None,
    ) -> None:
        """Initialise the client with confidential OAuth credentials.

        Args:
            client_id: CDSE OAuth client id (server-side confidential client).
            client_secret: CDSE OAuth client secret (kept only in ``.env.local``).
            token_url: Keycloak token endpoint of the CDSE realm.
            http_client: Optional injected HTTP client (defaults to a fresh
                :class:`httpx.Client` with a sane timeout).

        Raises:
            CDSECredentialsMissing: if ``client_id`` or ``client_secret`` is empty.
        """
        if not client_id or not client_secret:
            raise CDSECredentialsMissing(
                "CDSE_CLIENT_ID / CDSE_CLIENT_SECRET are not set. Create a "
                "confidential OAuth client (Client Credentials flow, NOT a "
                "single-page application) at https://shapps.dataspace.copernicus.eu "
                "and put the pair in .env.local."
            )
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_url = token_url
        self._http = http_client or httpx.Client(timeout=60.0)
        self._access_token: str | None = None
        self._token_expiry: float = 0.0

    def _ensure_token(self) -> str:
        """Return a valid access token, fetching/refreshing it when needed.

        Returns:
            A bearer access token valid for the CDSE STAC catalogue.

        Raises:
            httpx.HTTPStatusError: if the token endpoint rejects the credentials.
        """
        now = time.monotonic()
        if self._access_token is not None and now < self._token_expiry:
            return self._access_token

        response = self._http.post(
            self._token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        self._access_token = str(payload["access_token"])
        # ``expires_in`` is in seconds; refresh a little early to avoid races.
        self._token_expiry = now + float(payload.get("expires_in", 600.0)) - _TOKEN_REFRESH_MARGIN_S
        logger.info("cdse_token_acquired", expires_in=payload.get("expires_in"))
        return self._access_token

    def search_s2(
        self,
        bbox: tuple[float, float, float, float],
        datetime_range: str,
        *,
        cloud_cover_max: float = 10.0,
        limit: int = 50,
        l2a_only: bool = True,
    ) -> list[CDSEScene]:
        """Search Sentinel-2 scenes by area, date window and cloud cover (OData).

        Mirrors the agent ``search_stac`` contract -- a ``bbox`` (EPSG:4326
        lon/lat), a date range and a maximum cloud cover -- over the CDSE OData
        product catalogue (the working CDSE discovery API; the STAC ``/search``
        collections are CLMS/CCM products, not Sentinel). ``cloud_cover_max=10``
        keeps scenes that are >= 90% cloud-free.

        Args:
            bbox: ``[min_lon, min_lat, max_lon, max_lat]`` in EPSG:4326.
            datetime_range: ``"start/end"`` ISO-8601 (e.g.
                ``"2021-05-01T00:00:00Z/2021-09-30T23:59:59Z"``); a bare
                ``"start/end"`` of dates is accepted and normalised.
            cloud_cover_max: Maximum scene cloud cover percentage to keep.
            limit: Maximum number of products to return.
            l2a_only: Keep only L2A (surface reflectance) products.

        Returns:
            A list of :class:`CDSEScene`, newest first, never fabricated: an
            empty list means the catalogue returned no matching scene.

        Raises:
            httpx.HTTPStatusError: on a non-2xx response from the catalogue.
            ValueError: if ``datetime_range`` is not a ``start/end`` pair.
        """
        token = self._ensure_token()
        start, _, end = datetime_range.partition("/")
        if not start or not end:
            raise ValueError(
                f"datetime_range must be 'start/end'; got {datetime_range!r}"
            )
        start_iso = _to_odata_instant(start)
        end_iso = _to_odata_instant(end)
        min_lon, min_lat, max_lon, max_lat = bbox
        polygon = (
            f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},"
            f"{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"
        )
        # OData $filter: Sentinel-2, intersecting the AOI, in the date window,
        # with a cloud-cover attribute below the threshold (>= 90% cloud-free).
        cloud_filter = (
            f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq "
            f"'{_CLOUD_ATTR}' and att/OData.CSC.DoubleAttribute/Value "
            f"lt {cloud_cover_max})"
        )
        filt = (
            f"Collection/Name eq '{_S2_COLLECTION_NAME}' "
            f"and OData.CSC.Intersects(area=geography'SRID=4326;{polygon}') "
            f"and ContentDate/Start gt {start_iso} "
            f"and ContentDate/Start lt {end_iso} "
            f"and {cloud_filter}"
        )
        params = {
            "$filter": filt,
            "$orderby": "ContentDate/Start desc",
            "$top": str(limit),
            "$expand": "Attributes",
        }
        response = self._http.get(
            _ODATA_PRODUCTS_URL,
            params=params,
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()
        products = response.json().get("value", [])
        scenes: list[CDSEScene] = []
        for prod in products:
            if l2a_only and _S2_L2A_TOKEN not in str(prod.get("Name", "")):
                continue
            scene = _scene_from_product(prod)
            if scene is not None:
                scenes.append(scene)
        logger.info(
            "cdse_s2_search_done",
            n_scenes=len(scenes),
            cloud_cover_max=cloud_cover_max,
            datetime=datetime_range,
        )
        return scenes


def _to_odata_instant(value: str) -> str:
    """Normalise a date/datetime to the OData instant format ``...Z``.

    Args:
        value: A date (``"2021-05-01"``) or ISO-8601 datetime, optionally with a
            trailing ``Z``.

    Returns:
        An OData-compatible instant string (always with a ``T..:..:..Z`` time).
    """
    v = value.strip()
    if "T" not in v:
        v = f"{v}T00:00:00.000Z"
    elif not v.endswith("Z"):
        v = f"{v}Z"
    return v


def _scene_from_product(product: dict[str, Any]) -> CDSEScene | None:
    """Convert a CDSE OData product into a :class:`CDSEScene` (``None`` if bad).

    Args:
        product: One product dict from the OData ``value`` array (with
            ``Attributes`` expanded, so ``cloudCover`` and ``GeoFootprint`` are
            available).

    Returns:
        The parsed :class:`CDSEScene`, or ``None`` when required fields are
        missing (logged, never raised, so one bad product does not fail search).
    """
    name = product.get("Name")
    when = (product.get("ContentDate") or {}).get("Start")
    if not name or not when:
        logger.warning("cdse_scene_skipped", reason="missing fields", name=name)
        return None
    cloud = 0.0
    for att in product.get("Attributes", []):
        if att.get("Name") == _CLOUD_ATTR:
            cloud = float(att.get("Value", 0.0))
            break
    bbox = _bbox_from_footprint(product.get("GeoFootprint"))
    return CDSEScene(
        scene_id=str(name),
        datetime=str(when),
        cloud_cover=cloud,
        bbox=bbox,
    )


def _bbox_from_footprint(footprint: Any) -> tuple[float, float, float, float]:
    """Compute a lon/lat bbox from a GeoJSON ``GeoFootprint`` (``0`` quad if absent).

    Args:
        footprint: A GeoJSON geometry (Polygon/MultiPolygon) or ``None``.

    Returns:
        ``(min_lon, min_lat, max_lon, max_lat)``; a zero quad when the footprint
        is missing or unparseable (the scene id + date are still usable).
    """
    if not isinstance(footprint, dict):
        return (0.0, 0.0, 0.0, 0.0)
    coords = footprint.get("coordinates")
    lons: list[float] = []
    lats: list[float] = []

    def _walk(node: Any) -> None:
        if (
            isinstance(node, (list, tuple))
            and len(node) == 2
            and all(isinstance(x, (int, float)) for x in node)
        ):
            lons.append(float(node[0]))
            lats.append(float(node[1]))
        elif isinstance(node, (list, tuple)):
            for child in node:
                _walk(child)

    _walk(coords)
    if not lons or not lats:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(lons), min(lats), max(lons), max(lats))


def _scene_from_feature(feature: dict[str, Any]) -> CDSEScene | None:
    """Convert a STAC feature into a :class:`CDSEScene` (``None`` if malformed).

    Args:
        feature: A STAC item dict from the CDSE catalogue ``search`` response.

    Returns:
        The parsed :class:`CDSEScene`, or ``None`` when required fields are
        missing (logged, never raised, so one bad item does not fail the search).
    """
    props = feature.get("properties", {})
    scene_id = feature.get("id")
    when = props.get("datetime")
    raw_bbox = feature.get("bbox")
    if not scene_id or not when or not raw_bbox or len(raw_bbox) < 4:
        logger.warning("cdse_scene_skipped", reason="missing fields", id=scene_id)
        return None
    cloud = props.get("eo:cloud_cover", props.get("cloudCover", 0.0))
    return CDSEScene(
        scene_id=str(scene_id),
        datetime=str(when),
        cloud_cover=float(cloud),
        bbox=(float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3])),
    )


def cdse_client_from_settings(settings: Any) -> CDSEClient:
    """Build a :class:`CDSEClient` from the application settings.

    Args:
        settings: The app settings object exposing ``cdse_client_id``,
            ``cdse_client_secret`` and ``cdse_token_url``.

    Returns:
        A ready :class:`CDSEClient`.

    Raises:
        CDSECredentialsMissing: if the client-credentials pair is unset.
    """
    return CDSEClient(
        client_id=getattr(settings, "cdse_client_id", "") or "",
        client_secret=getattr(settings, "cdse_client_secret", "") or "",
        token_url=getattr(
            settings,
            "cdse_token_url",
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        ),
    )
