"""``parcels`` table model (inferred / ingested agricultural polygons)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from geoalchemy2 import Geometry
from sqlalchemy import Column, text
from sqlmodel import Field, SQLModel


class Parcel(SQLModel, table=True):
    """A classified parcel polygon. Mirrors ``parcels`` (US-015).

    ``session_id`` and ``aoi_id`` are nullable FKs in the schema; downstream
    features live in ``features_parcels`` (FK ``parcel_id``).
    """

    __tablename__ = "parcels"

    id: int | None = Field(default=None, primary_key=True)
    session_id: uuid.UUID | None = Field(
        default=None, foreign_key="chat_sessions.id", index=True
    )
    aoi_id: int | None = Field(default=None, foreign_key="aois.id")
    geom: Any = Field(sa_column=Column(Geometry("POLYGON", srid=4326), nullable=False))
    crop_class: str | None = None
    confidence: float | None = None
    area_ha: float | None = None
    year: int
    created_at: datetime | None = Field(
        default=None, sa_column_kwargs={"server_default": text("now()")}
    )
    updated_at: datetime | None = Field(
        default=None, sa_column_kwargs={"server_default": text("now()")}
    )
