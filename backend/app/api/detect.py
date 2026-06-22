"""``/detect`` router: on-demand crop map of an AOI from the trained model.

Thin HTTP adapter (router -> service -> model). ``POST /detect`` takes a drawn
AOI and returns the crop polygons the ``xgb-alphaearth`` classifier detects in
it (GeoJSON), i.e. *model output for the zone*, not a pre-loaded catalogue. All
logic (grid sampling, Earth Engine, classification, vectorisation) lives in
:class:`~backend.app.services.detect_service.DetectService`.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from backend.app.core.db import get_request_session_id
from backend.app.models.detect import DetectionFeatureCollection
from backend.app.services.detect_service import DetectService
from ml.agent.schemas import GeoJSONGeometry

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/detect", tags=["detect"])

#: Default campaign year (AlphaEarth annual; matches the perceiver / classify tool).
_DEFAULT_YEAR: int = 2019


class DetectRequest(BaseModel):
    """Body of a ``POST /detect`` request.

    Attributes:
        aoi: AOI polygon to analyse (EPSG:4326), validated as a GeoJSON geometry.
        year: Campaign year of the AlphaEarth annual embedding.
    """

    model_config = ConfigDict(extra="forbid")

    aoi: GeoJSONGeometry
    year: int = Field(default=_DEFAULT_YEAR, ge=2017, le=2100)


@router.post("", response_model=DetectionFeatureCollection)
async def detect(
    body: DetectRequest,
    session_id: Annotated[UUID, Depends(get_request_session_id)],
) -> DetectionFeatureCollection:
    """Detect crops over the AOI and return them as merged GeoJSON polygons.

    Args:
        body: Validated request (AOI polygon + campaign year).
        session_id: Tenant session resolved from ``X-Session-ID`` (``400`` on a
            missing/malformed header); used for traceable logging.

    Returns:
        A :class:`DetectionFeatureCollection` of the detected crop regions.
    """
    logger.info("detect_request_received", session_id=str(session_id), year=body.year)
    return await DetectService().crop_map(body.aoi, body.year)
