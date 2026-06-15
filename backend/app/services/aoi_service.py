"""AOI service: GeoJSON validation, area computation and persistence.

Router stays thin (router -> service -> model, root SoC rule): it hands the
validated GeoJSON geometry here, the service computes ``area_ha``, persists via
the repository and reads geometries back as GeoJSON for the response.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import structlog
from geoalchemy2.functions import ST_AsGeoJSON
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.aoi import Aoi
from backend.app.repositories.aoi import AoiRepository
from backend.app.utils.geo import (
    geojson_to_shape,
    geometry_to_geojson_str,
    polygon_area_ha,
)

logger = structlog.get_logger(__name__)


class AoiView:
    """Plain projection of an AOI row plus its geometry as GeoJSON."""

    def __init__(
        self,
        *,
        id: int,
        session_id: uuid.UUID,
        label: str | None,
        area_ha: float | None,
        geometry: dict[str, Any],
    ) -> None:
        self.id = id
        self.session_id = session_id
        self.label = label
        self.area_ha = area_ha
        self.geometry = geometry


class AoiService:
    """Create and list AOIs of a session."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._repo = AoiRepository(session)

    async def create(
        self,
        *,
        session_id: uuid.UUID,
        geometry: dict[str, Any],
        label: str | None,
    ) -> AoiView:
        """Validate a GeoJSON Polygon, compute area and persist the AOI.

        Raises:
            ValueError: if the geometry is not a valid Polygon.
        """
        geom = geojson_to_shape(geometry)
        area_ha = round(polygon_area_ha(geom), 4)
        aoi = await self._repo.create_from_geojson(
            session_id=session_id,
            geometry_json=geometry_to_geojson_str(geometry),
            label=label,
            area_ha=area_ha,
        )
        logger.info("aoi_created", aoi_id=aoi.id, session_id=str(session_id), area_ha=area_ha)
        return AoiView(
            id=aoi.id,  # type: ignore[arg-type]
            session_id=session_id,
            label=aoi.label,
            area_ha=aoi.area_ha,
            geometry=geometry,
        )

    async def list_for_session(self, *, session_id: uuid.UUID) -> list[AoiView]:
        """Return all AOIs of a session, geometry serialized as GeoJSON."""
        stmt = (
            select(
                Aoi.id,
                Aoi.label,
                Aoi.area_ha,
                ST_AsGeoJSON(Aoi.geom),
            )
            .where(Aoi.session_id == session_id)
            .order_by(Aoi.id.desc())  # type: ignore[union-attr]
        )
        result = await self._session.execute(stmt)
        views: list[AoiView] = []
        for row in result.all():
            aoi_id, label, area_ha, geojson_str = row
            views.append(
                AoiView(
                    id=aoi_id,
                    session_id=session_id,
                    label=label,
                    area_ha=area_ha,
                    geometry=json.loads(geojson_str),
                )
            )
        return views
