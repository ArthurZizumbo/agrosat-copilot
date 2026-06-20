"""``/aois/{id}/timeseries`` router: session-scoped index series (US-053).

Thin HTTP adapter (router -> service -> DB). Resolves the RLS-scoped connection
and the auth-guard, validates the spectral index via the path/query Pydantic
types, and delegates to
:class:`~backend.app.services.timeseries_service.TimeseriesService`.

The series is honest-by-construction (see the service): ``NDVI`` yields at most
the stored phenology peak; ``NDWI``/``NDMI`` degrade to an empty series because no
temporal anchor for them is persisted. A foreign or unknown AOI is ``404`` (RLS
hides it). An invalid ``index`` is ``422`` (enforced by the ``Literal`` type).
"""

from __future__ import annotations

from datetime import date
from typing import Annotated
from uuid import UUID

import asyncpg
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.app.api.deps import verify_session
from backend.app.core.db import get_scoped_conn
from backend.app.models.geo import TimeSeriesIndex, TimeSeriesResponse
from backend.app.services.timeseries_service import TimeseriesService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/aois", tags=["timeseries"])

#: Default window covers a full campaign year so a stored NDVI peak (any day of
#: year) falls inside it unless the caller narrows the range explicitly.
_DEFAULT_START = date(2017, 1, 1)
_DEFAULT_END = date(2100, 12, 31)


@router.get("/{aoi_id}/timeseries", response_model=TimeSeriesResponse)
async def get_aoi_timeseries(
    aoi_id: int,
    _session: Annotated[UUID, Depends(verify_session)],
    conn: Annotated[asyncpg.Connection, Depends(get_scoped_conn)],
    index: Annotated[TimeSeriesIndex, Query()] = "NDVI",
    start: Annotated[date, Query()] = _DEFAULT_START,
    end: Annotated[date, Query()] = _DEFAULT_END,
) -> TimeSeriesResponse:
    """Return the AOI's spectral-index series from stored phenology anchors.

    Args:
        aoi_id: AOI whose parcels are summarised.
        _session: Authorised tenant session (guard side effect; value unused).
        conn: RLS-scoped connection bound to the session.
        index: Spectral index (``NDVI``/``NDWI``/``NDMI``); invalid -> ``422``.
        start: Inclusive window start (defaults to full AlphaEarth coverage).
        end: Inclusive window end.

    Returns:
        A :class:`TimeSeriesResponse`. ``NDVI`` carries at most the in-window
        peak; ``NDWI``/``NDMI`` are empty (no anchor persisted).

    Raises:
        HTTPException: ``404`` when the AOI is unknown or owned by another tenant;
            ``422`` when ``end`` precedes ``start``.
    """
    if end < start:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"end {end} must not be before start {start}.",
        )
    if not await TimeseriesService.aoi_exists(conn, aoi_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="AOI not found.")
    return await TimeseriesService.for_aoi(conn, aoi_id, index, start, end)
