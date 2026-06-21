"""Pydantic response/request models for the geospatial data endpoints (US-053).

These are **not** ORM models. The backend data layer is asyncpg + PostGIS I/O
(see ``backend/app/core/db.py`` and the decision recorded in
``docs/us-planning/us-053.md`` Section 2.1): geometry <-> GeoJSON conversion is
resolved in SQL (``ST_AsGeoJSON`` / ``ST_SetSRID(ST_GeomFromGeoJSON, 4326)``) and
these models only shape the request/response contract so routers return typed
Pydantic objects (never a raw asyncpg ``Record``).

The contracts are GeoJSON-compatible (AC-6): an AOI is a GeoJSON ``Feature``, the
listing is a ``FeatureCollection`` and the STAC search returns a STAC
``FeatureCollection`` (the ``dict`` produced by ``pystac.ItemCollection.to_dict``).
The input geometry validator is reused from :class:`ml.agent.schemas.GeoJSONGeometry`
(DRY) so the endpoint and the agent's ``add_aoi`` tool speak the same vocabulary.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from ml.agent.schemas import GeoJSONGeometry

__all__ = [
    "AoiCreate",
    "AoiFeature",
    "AoiFeatureCollection",
    "AoiProperties",
    "StacItemCollection",
    "StacSearchQuery",
    "TimeSeriesResponse",
]

# ``extra="forbid"`` rejects unknown keys (typos / spoofed fields such as a
# client-supplied ``session_id``); the session always comes from the RLS hook.
_FORBID = ConfigDict(extra="forbid")

#: Spectral indices the timeseries endpoint accepts. NDVI is anchored on the
#: stored phenology peak; NDWI/NDMI have no temporal anchor persisted in the DB
#: and degrade to an empty (honest) series -- see ``timeseries_service``.
TimeSeriesIndex = Literal["NDVI", "NDWI", "NDMI"]


# ---------------------------------------------------------------------------
# /aois -- GeoJSON Feature contract
# ---------------------------------------------------------------------------
class AoiCreate(BaseModel):
    """Request body of ``POST /aois``.

    Only the geometry and an optional label are accepted. The owning
    ``session_id`` is never read from the client: it is injected by the RLS hook
    (``current_setting('app.current_session')``) and the WITH CHECK policy
    rejects any attempt to write another tenant's row (US-051).

    Attributes:
        geometry: GeoJSON geometry validated against the OGC primitives by
            :class:`ml.agent.schemas.GeoJSONGeometry`. Must be a ``Polygon`` to
            match the ``GEOMETRY(POLYGON, 4326)`` column.
        label: Optional human-readable label stored in ``aois.label``.
    """

    model_config = _FORBID

    geometry: GeoJSONGeometry
    label: str | None = None

    @field_validator("geometry")
    @classmethod
    def _require_polygon(cls, value: GeoJSONGeometry) -> GeoJSONGeometry:
        """Reject non-Polygon geometries before they reach PostGIS.

        The ``aois.geom`` column is ``GEOMETRY(POLYGON, 4326)``; inserting a
        ``MultiPolygon`` or a point would raise a database error mid-request.
        Validating here turns that into a clean ``422`` at the edge.
        """
        if value.type != "Polygon":
            raise ValueError(f"AOI geometry must be a Polygon, got {value.type!r}")
        return value


class AoiProperties(BaseModel):
    """``properties`` block of an AOI GeoJSON Feature.

    Attributes:
        label: Human-readable AOI label, if any.
        area_ha: Server-computed area in hectares (geodesic, via
            ``ST_Area(geom::geography) / 10000``).
        created_at: Row creation timestamp.
    """

    model_config = _FORBID

    label: str | None = None
    area_ha: float | None = None
    created_at: datetime


class AoiFeature(BaseModel):
    """A persisted AOI rendered as a GeoJSON ``Feature``.

    Attributes:
        type: Always ``"Feature"``.
        id: Primary key in the ``aois`` table.
        geometry: GeoJSON geometry produced by ``ST_AsGeoJSON``.
        properties: Non-geometry attributes (label, area, timestamp).
    """

    model_config = _FORBID

    type: Literal["Feature"] = "Feature"
    id: int
    geometry: dict
    properties: AoiProperties


class AoiFeatureCollection(BaseModel):
    """A GeoJSON ``FeatureCollection`` wrapping the session's AOIs.

    Attributes:
        type: Always ``"FeatureCollection"``.
        features: AOIs visible to the calling session (RLS-filtered).
    """

    model_config = _FORBID

    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[AoiFeature]


# ---------------------------------------------------------------------------
# /aois/{id}/timeseries
# ---------------------------------------------------------------------------
class TimeSeriesResponse(BaseModel):
    """Result of ``GET /aois/{aoi_id}/timeseries``.

    Consistent with :class:`ml.agent.schemas.TimeSeries`: ``dates`` and
    ``values`` are aligned one-to-one. The series is short by design (at most the
    NDVI phenology peak); NDWI/NDMI degrade to an empty series because no temporal
    anchor for them is persisted (honest-by-construction, see the timeseries
    service / ``ml/agent/tools/timeseries.py``).

    Attributes:
        aoi_id: AOI the series belongs to.
        index: Spectral index echoed back.
        dates: Observation dates (ascending), aligned with ``values``.
        values: Index values aligned one-to-one with ``dates``.
    """

    model_config = _FORBID

    aoi_id: int
    index: TimeSeriesIndex
    dates: list[date]
    values: list[float]

    @field_validator("values")
    @classmethod
    def _validate_aligned(cls, value: list[float], info: object) -> list[float]:
        """Ensure ``values`` and ``dates`` have matching length."""
        dates = info.data.get("dates")  # type: ignore[attr-defined]
        if dates is not None and len(value) != len(dates):
            raise ValueError(f"values length {len(value)} != dates length {len(dates)}")
        return value


# ---------------------------------------------------------------------------
# /stac/search
# ---------------------------------------------------------------------------
class StacSearchQuery(BaseModel):
    """Query parameters of ``GET /stac/search``.

    Mirrors the STAC API ``search`` vocabulary (same fields the agent's
    ``search_stac`` tool speaks). pgstac is not deployed yet, so a query against
    an absent catalogue degrades to an empty (valid) ``FeatureCollection`` rather
    than erroring -- see :class:`backend.app.services.stac_service.StacService`.

    Attributes:
        bbox: ``[minx, miny, maxx, maxy]`` in EPSG:4326, if filtering spatially.
        datetime: RFC 3339 single instant or interval (e.g.
            ``"2019-01-01T00:00:00Z/2019-12-31T23:59:59Z"``).
        collections: STAC collection ids to search within.
        limit: Maximum number of items to return.
    """

    model_config = _FORBID

    bbox: list[float] | None = None
    datetime: str | None = None
    collections: list[str] | None = None
    limit: int = 10

    @field_validator("bbox")
    @classmethod
    def _validate_bbox(cls, value: list[float] | None) -> list[float] | None:
        """Require a 4-element ``[minx, miny, maxx, maxy]`` box with valid ranges."""
        if value is None:
            return value
        if len(value) != 4:
            raise ValueError(f"bbox must have 4 elements [minx,miny,maxx,maxy], got {len(value)}")
        minx, miny, maxx, maxy = value
        if not (-180.0 <= minx <= 180.0 and -180.0 <= maxx <= 180.0):
            raise ValueError("bbox longitudes out of range [-180, 180]")
        if not (-90.0 <= miny <= 90.0 and -90.0 <= maxy <= 90.0):
            raise ValueError("bbox latitudes out of range [-90, 90]")
        if minx > maxx or miny > maxy:
            raise ValueError("bbox min edge must not exceed max edge")
        return value

    @field_validator("limit")
    @classmethod
    def _validate_limit(cls, value: int) -> int:
        """Constrain the page size to a sane positive bound."""
        if not 1 <= value <= 1000:
            raise ValueError(f"limit {value} out of range [1, 1000]")
        return value


# A STAC ``FeatureCollection`` is returned as the plain ``dict`` produced by
# ``pystac.ItemCollection.to_dict()`` (GeoJSON-compatible). Aliased for clarity at
# the router boundary without forcing a rigid model over pystac's output shape.
StacItemCollection = dict
