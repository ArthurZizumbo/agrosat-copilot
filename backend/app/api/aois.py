"""``/aois`` router: session-scoped CRUD of Areas Of Interest (US-053).

Thin HTTP adapter (router -> service -> DB). Each handler resolves the request's
RLS-scoped connection (:func:`~backend.app.core.db.get_scoped_conn`) and the
auth-guard (:func:`~backend.app.api.deps.verify_session`), then delegates to
:class:`~backend.app.services.aoi_service.AoiService`. No SQL lives here.

Isolation is enforced by the US-051 ``tenant_isolation`` RLS policies, not by an
application-level ``WHERE session_id``: ``GET`` lists only the session's AOIs,
``GET /{id}`` / ``DELETE /{id}`` return ``404`` for a foreign id (the row is
invisible), and ``POST`` cannot write another tenant's row (WITH CHECK).
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

import asyncpg
import structlog
from fastapi import APIRouter, Depends, HTTPException, Response, status

from backend.app.api.deps import verify_session
from backend.app.core.db import get_scoped_conn
from backend.app.models.geo import AoiCreate, AoiFeature, AoiFeatureCollection
from backend.app.services.aoi_service import AoiService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/aois", tags=["aois"])


@router.post("", status_code=status.HTTP_201_CREATED, response_model=AoiFeature)
async def create_aoi(
    body: AoiCreate,
    _session: Annotated[UUID, Depends(verify_session)],
    conn: Annotated[asyncpg.Connection, Depends(get_scoped_conn)],
) -> AoiFeature:
    """Create an AOI from a GeoJSON Polygon for the calling session.

    The owning ``session_id`` is taken from the RLS hook, never from the body
    (anti-spoofing). Returns the persisted AOI as a GeoJSON ``Feature`` with its
    generated id and server-computed ``area_ha``.

    Args:
        body: Validated request (Polygon geometry + optional label).
        _session: Authorised tenant session (guard side effect; value unused).
        conn: RLS-scoped connection bound to the session.

    Returns:
        The created :class:`AoiFeature` (HTTP ``201``).
    """
    return await AoiService.create(conn, body)


@router.get("", response_model=AoiFeatureCollection)
async def list_aois(
    _session: Annotated[UUID, Depends(verify_session)],
    conn: Annotated[asyncpg.Connection, Depends(get_scoped_conn)],
) -> AoiFeatureCollection:
    """List every AOI visible to the calling session (RLS-filtered).

    Args:
        _session: Authorised tenant session (guard side effect; value unused).
        conn: RLS-scoped connection bound to the session.

    Returns:
        A GeoJSON :class:`AoiFeatureCollection`; empty for a session with no AOIs.
    """
    return await AoiService.list(conn)


@router.get("/{aoi_id}", response_model=AoiFeature)
async def get_aoi(
    aoi_id: int,
    _session: Annotated[UUID, Depends(verify_session)],
    conn: Annotated[asyncpg.Connection, Depends(get_scoped_conn)],
) -> AoiFeature:
    """Fetch one AOI by id if it belongs to the calling session.

    Args:
        aoi_id: Primary key of the AOI.
        _session: Authorised tenant session (guard side effect; value unused).
        conn: RLS-scoped connection bound to the session.

    Returns:
        The :class:`AoiFeature`.

    Raises:
        HTTPException: ``404`` when the AOI does not exist or is owned by another
            tenant (RLS hides it -- no foreign existence leak).
    """
    feature = await AoiService.get(conn, aoi_id)
    if feature is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AOI not found.")
    return feature


@router.delete("/{aoi_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_aoi(
    aoi_id: int,
    _session: Annotated[UUID, Depends(verify_session)],
    conn: Annotated[asyncpg.Connection, Depends(get_scoped_conn)],
) -> Response:
    """Delete an AOI by id if it belongs to the calling session.

    Args:
        aoi_id: Primary key of the AOI to delete.
        _session: Authorised tenant session (guard side effect; value unused).
        conn: RLS-scoped connection bound to the session.

    Returns:
        An empty ``204`` response on success.

    Raises:
        HTTPException: ``404`` when the AOI does not exist or is owned by another
            tenant (RLS yields ``DELETE 0``).
    """
    deleted = await AoiService.delete(conn, aoi_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AOI not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
