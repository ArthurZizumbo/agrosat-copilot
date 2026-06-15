"""``features_parcels`` table model (temporal + spectral aggregates)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class FeatureParcel(SQLModel, table=True):
    """Per-(parcel, year) feature row backing the vision tools.

    Mirrors ``features_parcels`` (US-015): ``alphaearth_embedding`` is a nullable
    ``VECTOR(64)`` (pgvector), ``ndvi_stats`` / ``phenology`` are JSONB, plus the
    scalar phenology columns. UNIQUE (parcel_id, year) enforced by the DB.
    """

    __tablename__ = "features_parcels"

    id: int | None = Field(default=None, primary_key=True)
    parcel_id: int = Field(foreign_key="parcels.id", index=True)
    year: int
    alphaearth_embedding: Any | None = Field(
        default=None, sa_column=Column(Vector(64), nullable=True)
    )
    ndvi_stats: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    )
    phenology: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, server_default=text("'{}'::jsonb")),
    )
    sog_doy: int | None = None
    peak_doy: int | None = None
    peak_value: float | None = None
    senescence_doy: int | None = None
    ndvi_auc: float | None = None
    ndvi_slope_pre_peak: float | None = None
    ndvi_slope_post_peak: float | None = None
    maturity_duration_days: int | None = None
    created_at: datetime | None = Field(
        default=None, sa_column_kwargs={"server_default": text("now()")}
    )
    updated_at: datetime | None = Field(
        default=None, sa_column_kwargs={"server_default": text("now()")}
    )
