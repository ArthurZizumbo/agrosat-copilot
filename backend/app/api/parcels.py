"""``/parcels`` router: session-scoped, bbox-clipped GeoJSON of parcels.

Thin HTTP adapter (router -> service -> DB). The handler authorises the request
against its tenant session (:func:`~backend.app.api.deps.verify_chat_session`),
resolves the RLS-scoped connection (:func:`~backend.app.core.db.get_scoped_conn`)
and delegates the SQL to :class:`~backend.app.services.parcel_service.ParcelService`.
No SQL lives here; the only router-level work is parsing the raw ``bbox`` query
string into four floats (a transport concern) and clamping the page size.

Isolation is enforced by the US-051 ``tenant_isolation`` RLS policy, not by an
application-level ``WHERE session_id``: ``GET`` returns only the calling
session's parcels (the row is invisible otherwise). This layer feeds the
frontend's independent "parcel universe" map layer (decoupled from ``/chat``).
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

import asyncpg
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.app.api.deps import verify_chat_session
from backend.app.core.db import get_scoped_conn
from backend.app.models.geo import ParcelFeatureCollection
from backend.app.services.parcel_service import ParcelService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/parcels", tags=["parcels"])

#: Hard cap on the page size; a larger ``limit`` is clamped down to this bound to
#: keep a single response (and the map layer it feeds) bounded.
_MAX_LIMIT = 5000


def _parse_bbox(bbox: str) -> tuple[float, float, float, float]:
    """Parse a ``"minLng,minLat,maxLng,maxLat"`` string into four floats.

    Args:
        bbox: Comma-separated bounding box in EPSG:4326.

    Returns:
        The ``(min_lng, min_lat, max_lng, max_lat)`` tuple.

    Raises:
        HTTPException: ``422`` when the string does not hold exactly four
            comma-separated numbers, or the min edge exceeds the max edge.
    """
    parts = bbox.split(",")
    if len(parts) != 4:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="bbox must be 'minLng,minLat,maxLng,maxLat' (4 comma-separated numbers).",
        )
    try:
        min_lng, min_lat, max_lng, max_lat = (float(p) for p in parts)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="bbox values must be numbers.",
        ) from exc
    if min_lng > max_lng or min_lat > max_lat:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="bbox min edge must not exceed max edge.",
        )
    return min_lng, min_lat, max_lng, max_lat


@router.get("", response_model=ParcelFeatureCollection)
async def list_parcels(
    _session: Annotated[UUID, Depends(verify_chat_session)],
    conn: Annotated[asyncpg.Connection, Depends(get_scoped_conn)],
    bbox: Annotated[
        str,
        Query(description="Bounding box 'minLng,minLat,maxLng,maxLat' in EPSG:4326."),
    ],
    year: Annotated[int, Query(description="Acquisition year filter.")] = 2019,
    limit: Annotated[
        int,
        Query(ge=1, description="Maximum parcels to return (clamped to 5000)."),
    ] = 2000,
) -> ParcelFeatureCollection:
    """List the session's parcels overlapping a bbox, as a GeoJSON FeatureCollection.

    Authorisation (``Depends(verify_chat_session)``) rejects a missing/malformed
    ``X-Session-ID`` with ``400`` and an unknown/foreign session with ``403``
    (fail-closed under RLS), exactly like ``/chat``. The ``bbox`` query string is
    parsed into four floats (``422`` if malformed) and the page size is clamped to
    ``5000`` before the service runs the GiST-indexed bbox query.

    Args:
        _session: Authorised tenant session (guard side effect; value unused, the
            RLS hook already primes the connection).
        conn: RLS-scoped connection bound to the session.
        bbox: Raw ``"minLng,minLat,maxLng,maxLat"`` query string (EPSG:4326).
        year: Acquisition year filter (defaults to ``2019``).
        limit: Maximum number of parcels (defaults to ``2000``, clamped to
            ``5000``).

    Returns:
        A GeoJSON :class:`ParcelFeatureCollection`; empty when no owned parcel
        intersects the box for the given year.

    Raises:
        HTTPException: ``422`` when ``bbox`` is malformed.
    """
    min_lng, min_lat, max_lng, max_lat = _parse_bbox(bbox)
    effective_limit = min(limit, _MAX_LIMIT)
    return await ParcelService.list_in_bbox(
        conn,
        min_lng=min_lng,
        min_lat=min_lat,
        max_lng=max_lng,
        max_lat=max_lat,
        year=year,
        limit=effective_limit,
    )
