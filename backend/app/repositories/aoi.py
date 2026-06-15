"""Repository for ``aois`` (session-scoped, geometry-aware)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from geoalchemy2.functions import ST_GeomFromGeoJSON
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.aoi import Aoi
from backend.app.repositories.base import BaseRepository


class AoiRepository(BaseRepository[Aoi]):
    """Session-scoped access to ``aois``."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Aoi, session)

    async def list_for_session(self, *, session_id: uuid.UUID) -> Sequence[Aoi]:
        """Return all AOIs belonging to a session (newest first)."""
        stmt = (
            select(Aoi).where(Aoi.session_id == session_id).order_by(Aoi.id.desc())  # type: ignore[union-attr]
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def create_from_geojson(
        self,
        *,
        session_id: uuid.UUID,
        geometry_json: str,
        label: str | None,
        area_ha: float | None,
    ) -> Aoi:
        """Insert an AOI from a GeoJSON geometry string (POLYGON, SRID 4326).

        Args:
            session_id: Owning session.
            geometry_json: Serialized GeoJSON geometry (``json.dumps`` of a
                Polygon geometry object).
            label: Optional human label.
            area_ha: Pre-computed area in hectares.
        """
        geom = ST_GeomFromGeoJSON(geometry_json)
        obj = Aoi(session_id=session_id, geom=geom, label=label, area_ha=area_ha)
        return await self.add_commit_refresh(obj)
