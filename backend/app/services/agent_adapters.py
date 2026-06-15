"""SQL adapters implementing the agent ports (hexagonal boundary).

``ml/agent/ports.py`` declares the Protocols ``ParcelReader`` and ``ChatMemory``
plus plain DTOs. The backend implements them here over the async ORM and injects
them via ``AgentDeps``. Each call opens its own short-lived ``AsyncSession`` from
the factory because the orchestrator runs in a detached background task whose
lifetime is decoupled from any request scope.

Every read is scoped by ``session_id`` (multi-tenant NON-NEGOTIABLE).
"""

from __future__ import annotations

import json
import uuid

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.models.feature import FeatureParcel
from backend.app.repositories.chat_message import ChatMessageRepository
from backend.app.repositories.parcel import ParcelRepository
from ml.agent.ports import ChatTurn, FeatureRecord, ParcelRecord


def _to_embedding(value: object) -> list[float] | None:
    """Convert a stored ``VECTOR(64)`` value to ``list[float]`` (or ``None``)."""
    if value is None:
        return None
    from collections.abc import Iterable

    if isinstance(value, Iterable):
        return [float(x) for x in value]
    return None


def _feature_to_record(row: FeatureParcel) -> FeatureRecord:
    """Map a ``features_parcels`` row to the agent ``FeatureRecord`` DTO."""
    ndvi_stats = {k: float(v) for k, v in (row.ndvi_stats or {}).items()}
    phenology = {k: float(v) for k, v in (row.phenology or {}).items()}
    return FeatureRecord(
        parcel_id=row.parcel_id,
        year=row.year,
        alphaearth_embedding=_to_embedding(row.alphaearth_embedding),
        ndvi_stats=ndvi_stats,
        phenology=phenology,
        ndvi_auc=row.ndvi_auc,
        peak_value=row.peak_value,
    )


class SqlParcelReader:
    """``ParcelReader`` adapter backed by :class:`ParcelRepository`."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def list_parcels_in_aoi(
        self, *, session_id: str, aoi_id: int | None = None, year: int | None = None
    ) -> list[ParcelRecord]:
        """Return parcels of a session, optionally filtered by AOI / year."""
        sid = uuid.UUID(session_id)
        async with self._session_factory() as db:
            repo = ParcelRepository(db)
            rows = await repo.list_in_aoi_with_geojson(session_id=sid, aoi_id=aoi_id, year=year)
            return [
                ParcelRecord(
                    id=parcel.id,  # type: ignore[arg-type]
                    aoi_id=parcel.aoi_id,
                    crop_class=parcel.crop_class,
                    confidence=parcel.confidence,
                    area_ha=parcel.area_ha,
                    year=parcel.year,
                    geometry=json.loads(geojson) if geojson else None,
                )
                for parcel, geojson in rows
            ]

    async def get_features(
        self, *, session_id: str, parcel_id: int, year: int
    ) -> FeatureRecord | None:
        """Return the feature row for a parcel/year, or ``None`` if absent."""
        sid = uuid.UUID(session_id)
        async with self._session_factory() as db:
            repo = ParcelRepository(db)
            row = await repo.get_features(session_id=sid, parcel_id=parcel_id, year=year)
            return _feature_to_record(row) if row is not None else None


class SqlChatMemory:
    """``ChatMemory`` adapter backed by :class:`ChatMessageRepository`."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def load_history(self, *, session_id: str, limit: int = 20) -> list[ChatTurn]:
        """Return the most recent turns (oldest first)."""
        sid = uuid.UUID(session_id)
        async with self._session_factory() as db:
            repo = ChatMessageRepository(db)
            rows = await repo.list_recent(session_id=sid, limit=limit)
            return [
                ChatTurn(role=row.role, content=row.content)  # type: ignore[arg-type]
                for row in rows
            ]

    async def append_turn(self, *, session_id: str, turn: ChatTurn) -> None:
        """Persist one turn."""
        sid = uuid.UUID(session_id)
        async with self._session_factory() as db:
            repo = ChatMessageRepository(db)
            await repo.append(session_id=sid, role=turn.role, content=turn.content)
