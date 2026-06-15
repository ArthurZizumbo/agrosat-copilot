"""``aois`` table model (POLYGON 4326 AOI of a session)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from geoalchemy2 import Geometry
from sqlalchemy import Column, text
from sqlmodel import Field, SQLModel


class Aoi(SQLModel, table=True):
    """An Area Of Interest polygon attached to a session.

    Mirrors ``aois`` from the initial migration. ``geom`` is a
    ``GEOMETRY(POLYGON, 4326)`` GeoAlchemy2 column; the ORM stores/reads WKB,
    callers convert to/from GeoJSON in the service layer.
    """

    __tablename__ = "aois"

    id: int | None = Field(default=None, primary_key=True)
    session_id: uuid.UUID = Field(foreign_key="chat_sessions.id", index=True)
    geom: Any = Field(sa_column=Column(Geometry("POLYGON", srid=4326), nullable=False))
    label: str | None = None
    area_ha: float | None = None
    created_at: datetime | None = Field(
        default=None, sa_column_kwargs={"server_default": text("now()")}
    )
