"""Pydantic request/response models for the API (never expose SQLModel rows)."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

LlmVariant = Literal["gemini", "qwen35"]


# --- Sessions --------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    """Optional overrides when creating a session."""

    llm_variant: LlmVariant | None = None


class SessionResponse(BaseModel):
    """Public projection of a chat session."""

    session_id: uuid.UUID
    user_id: str
    llm_variant: LlmVariant


# --- Chat ------------------------------------------------------------------


class ChatRequest(BaseModel):
    """A user turn dispatched to the agent."""

    session_id: uuid.UUID
    message: str = Field(min_length=1, max_length=4000)
    llm_variant: LlmVariant | None = None
    aoi_id: int | None = Field(
        default=None, description="Scope the analysis to this AOI; None = all session parcels."
    )


class ChatDispatchResponse(BaseModel):
    """``202`` payload pointing the client at the event stream."""

    job_id: str
    ws_url: str


# --- LLM switch ------------------------------------------------------------


class LlmSwitchRequest(BaseModel):
    """Switch the LLM variant of a session."""

    session_id: uuid.UUID
    llm_variant: LlmVariant


class LlmSwitchResponse(BaseModel):
    """Confirmation of the active variant after a switch."""

    session_id: uuid.UUID
    llm_variant: LlmVariant


# --- AOIs ------------------------------------------------------------------


class GeoJsonGeometry(BaseModel):
    """Minimal GeoJSON Polygon geometry (validated before the service)."""

    type: Literal["Polygon"]
    coordinates: list[list[list[float]]]

    @field_validator("coordinates")
    @classmethod
    def _non_empty_ring(cls, value: list[list[list[float]]]) -> list[list[list[float]]]:
        if not value or not value[0] or len(value[0]) < 4:
            raise ValueError("Polygon exterior ring needs at least 4 positions")
        return value

    def as_mapping(self) -> dict[str, Any]:
        """Return the geometry as a plain GeoJSON mapping."""
        return {"type": self.type, "coordinates": self.coordinates}


class CreateAoiRequest(BaseModel):
    """Create an AOI from a GeoJSON Polygon geometry."""

    geometry: GeoJsonGeometry
    label: str | None = Field(default=None, max_length=200)


class AoiResponse(BaseModel):
    """Public projection of an AOI, geometry as GeoJSON."""

    id: int
    session_id: uuid.UUID
    label: str | None
    area_ha: float | None
    geometry: dict[str, Any]


class AoiListResponse(BaseModel):
    """Collection of AOIs for a session."""

    items: list[AoiResponse]
