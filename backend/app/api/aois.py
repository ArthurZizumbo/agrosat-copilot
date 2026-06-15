"""AOI endpoints: list / create AOIs of a session (GeoJSON)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import _check_session_owner, get_current_user_id
from backend.app.api.schemas import (
    AoiListResponse,
    AoiResponse,
    CreateAoiRequest,
)
from backend.app.core.db import get_session
from backend.app.services.aoi_service import AoiService, AoiView

router = APIRouter(tags=["aois"])


def _to_response(view: AoiView) -> AoiResponse:
    """Map an internal AOI view to its public response model."""
    return AoiResponse(
        id=view.id,
        session_id=view.session_id,
        label=view.label,
        area_ha=view.area_ha,
        geometry=view.geometry,
    )


@router.get("/aois", response_model=AoiListResponse)
async def list_aois(
    session_id: uuid.UUID = Query(...),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_session),
) -> AoiListResponse:
    """List the AOIs belonging to a session."""
    await _check_session_owner(session_id=session_id, user_id=user_id, db=db)
    service = AoiService(db)
    views = await service.list_for_session(session_id=session_id)
    return AoiListResponse(items=[_to_response(v) for v in views])


@router.post("/aois", response_model=AoiResponse, status_code=status.HTTP_201_CREATED)
async def create_aoi(
    body: CreateAoiRequest,
    session_id: uuid.UUID = Query(...),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_session),
) -> AoiResponse:
    """Create an AOI from a GeoJSON Polygon for a session."""
    await _check_session_owner(session_id=session_id, user_id=user_id, db=db)
    service = AoiService(db)
    try:
        view = await service.create(
            session_id=session_id,
            geometry=body.geometry.as_mapping(),
            label=body.label,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    return _to_response(view)
