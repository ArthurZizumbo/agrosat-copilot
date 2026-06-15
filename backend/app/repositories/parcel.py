"""Repository for ``parcels`` and ``features_parcels`` (session-scoped)."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.feature import FeatureParcel
from backend.app.models.parcel import Parcel
from backend.app.repositories.base import BaseRepository


class ParcelRepository(BaseRepository[Parcel]):
    """Session-scoped access to ``parcels`` plus their feature rows."""

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(Parcel, session)

    async def list_in_aoi(
        self,
        *,
        session_id: uuid.UUID,
        aoi_id: int | None = None,
        year: int | None = None,
    ) -> Sequence[Parcel]:
        """Return parcels of a session, optionally filtered by AOI / year."""
        stmt = select(Parcel).where(Parcel.session_id == session_id)
        if aoi_id is not None:
            stmt = stmt.where(Parcel.aoi_id == aoi_id)
        if year is not None:
            stmt = stmt.where(Parcel.year == year)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_in_aoi_with_geojson(
        self,
        *,
        session_id: uuid.UUID,
        aoi_id: int | None = None,
        year: int | None = None,
    ) -> list[tuple[Parcel, str | None]]:
        """Like :meth:`list_in_aoi` but also returns each parcel geometry as a
        GeoJSON string (``ST_AsGeoJSON``) so the agent can ship it to the map."""
        stmt = select(Parcel, func.ST_AsGeoJSON(Parcel.geom).label("geojson")).where(
            Parcel.session_id == session_id
        )
        if aoi_id is not None:
            stmt = stmt.where(Parcel.aoi_id == aoi_id)
        if year is not None:
            stmt = stmt.where(Parcel.year == year)
        result = await self.session.execute(stmt)
        return [(row[0], row[1]) for row in result.all()]

    async def get_features(
        self, *, session_id: uuid.UUID, parcel_id: int, year: int
    ) -> FeatureParcel | None:
        """Return the feature row for a parcel/year, scoped by session.

        Joins ``parcels`` to enforce the session filter so a caller cannot read
        features of a parcel outside its session (multi-tenant NON-NEGOTIABLE).
        """
        stmt = (
            select(FeatureParcel)
            .join(Parcel, Parcel.id == FeatureParcel.parcel_id)
            .where(
                Parcel.session_id == session_id,
                FeatureParcel.parcel_id == parcel_id,
                FeatureParcel.year == year,
            )
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()
