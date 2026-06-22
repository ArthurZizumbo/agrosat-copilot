"""Parcel read service: bbox-clipped GeoJSON over an RLS-scoped connection.

The service owns all SQL for the ``/parcels`` endpoint; the router stays thin
(SoC). It receives the request's RLS-scoped :class:`asyncpg.Connection`
(``app.current_session`` already primed inside an open transaction by
:func:`backend.app.core.db.get_scoped_conn`), so the US-051 ``tenant_isolation``
policy filters every row to the calling session automatically -- there is no
application-level ``WHERE session_id`` (DRY with the agent tools and ``/aois``).

The spatial filter uses the GiST index on ``parcels.geom`` via the ``&&``
bounding-box overlap operator against ``ST_MakeEnvelope(...)``, and geometry is
serialised in SQL (``ST_AsGeoJSON(geom)::json``) so no GeoAlchemy2 ``WKBElement``
is mapped in Python.
"""

from __future__ import annotations

import json
from typing import Any, cast

import asyncpg
import structlog

from backend.app.models.geo import (
    ParcelFeature,
    ParcelFeatureCollection,
    ParcelProperties,
)

__all__ = ["ParcelService"]

logger = structlog.get_logger(__name__)

# RLS filters every SELECT to the current session; no manual WHERE session_id.
# ``geom && ST_MakeEnvelope(...)`` uses the GiST index (bbox overlap), then the
# exact ``year`` equality narrows the page. ``LIMIT`` is applied last after a
# stable ``ORDER BY id``. All values are bound (never interpolated).
_LIST_BY_BBOX_SQL = """
SELECT id, crop_class, confidence, area_ha, ST_AsGeoJSON(geom)::json AS geometry
FROM parcels
WHERE geom && ST_MakeEnvelope($1, $2, $3, $4, 4326)
  AND year = $5
ORDER BY id
LIMIT $6
"""


def _parse_geometry(value: object) -> dict:
    """Normalise an ``ST_AsGeoJSON(...)::json`` value into a GeoJSON ``dict``.

    asyncpg may deliver a ``json`` column either already decoded (a ``dict``,
    when a JSON codec is registered) or as the raw JSON ``str``. This accepts
    both so the response is always a parsed GeoJSON object.

    Args:
        value: The ``geometry`` field from a ``parcels`` row.

    Returns:
        The geometry as a GeoJSON ``dict``.
    """
    if isinstance(value, str):
        return cast("dict[str, Any]", json.loads(value))
    return cast("dict[str, Any]", value)


def _row_to_feature(row: asyncpg.Record) -> ParcelFeature:
    """Map a ``parcels`` row (with ``ST_AsGeoJSON`` geometry) to a GeoJSON Feature.

    Args:
        row: An asyncpg record exposing ``id``, ``crop_class``, ``confidence``,
            ``area_ha`` and ``geometry`` (from ``ST_AsGeoJSON(geom)::json``).

    Returns:
        The parcel as a :class:`ParcelFeature` whose ``properties.parcel_id`` is
        the row's ``id``.
    """
    return ParcelFeature(
        geometry=_parse_geometry(row["geometry"]),
        properties=ParcelProperties(
            parcel_id=int(row["id"]),
            crop_class=row["crop_class"],
            confidence=float(row["confidence"]) if row["confidence"] is not None else None,
            area_ha=float(row["area_ha"]) if row["area_ha"] is not None else None,
        ),
    )


class ParcelService:
    """Stateless read operations over the ``parcels`` table (RLS-scoped)."""

    @staticmethod
    async def list_in_bbox(
        conn: asyncpg.Connection,
        *,
        min_lng: float,
        min_lat: float,
        max_lng: float,
        max_lat: float,
        year: int,
        limit: int,
    ) -> ParcelFeatureCollection:
        """List the session's parcels overlapping a bounding box for a year.

        Args:
            conn: RLS-scoped connection (session primed, in a transaction).
            min_lng: West edge of the bounding box (EPSG:4326).
            min_lat: South edge of the bounding box (EPSG:4326).
            max_lng: East edge of the bounding box (EPSG:4326).
            max_lat: North edge of the bounding box (EPSG:4326).
            year: Acquisition year filter (``parcels.year``).
            limit: Maximum number of parcels to return.

        Returns:
            A GeoJSON :class:`ParcelFeatureCollection`; empty when the session
            owns no parcels intersecting the box for the given year.
        """
        rows = await conn.fetch(
            _LIST_BY_BBOX_SQL,
            min_lng,
            min_lat,
            max_lng,
            max_lat,
            year,
            limit,
        )
        logger.info(
            "parcels_listed",
            count=len(rows),
            year=year,
            limit=limit,
        )
        return ParcelFeatureCollection(features=[_row_to_feature(r) for r in rows])
