"""Ports (hexagonal boundaries) the agent core depends on.

The agent must stay free of backend / database imports so it can be unit-tested
with fakes and later moved to Vertex AI Agent Engine. The backend provides
concrete adapters (SQLModel repositories) that satisfy these Protocols and
injects them through :class:`AgentDeps`.

Records are plain Pydantic DTOs (not SQLModel rows) so the agent never touches
the ORM. Every read is scoped by ``session_id`` (multi-tenant NON-NEGOTIABLE).
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# DTOs crossing the boundary.
# ---------------------------------------------------------------------------


class ParcelRecord(BaseModel):
    """Minimal parcel projection the agent needs to reason and cite."""

    id: int
    aoi_id: int | None = None
    crop_class: str | None = None
    confidence: float | None = None
    area_ha: float | None = None
    year: int
    geometry: dict[str, object] | None = None


class FeatureRecord(BaseModel):
    """Per-parcel feature row backing the vision tools."""

    parcel_id: int
    year: int
    alphaearth_embedding: list[float] | None = None
    ndvi_stats: dict[str, float] = Field(default_factory=dict)
    phenology: dict[str, float] = Field(default_factory=dict)
    ndvi_auc: float | None = None
    peak_value: float | None = None


class ChatTurn(BaseModel):
    """One persisted conversation turn."""

    role: Literal["user", "assistant"]
    content: str


# ---------------------------------------------------------------------------
# Protocols (implemented by backend adapters).
# ---------------------------------------------------------------------------


@runtime_checkable
class ParcelReader(Protocol):
    """Read-only access to parcels and their features, session-scoped."""

    async def list_parcels_in_aoi(
        self, *, session_id: str, aoi_id: int | None = None, year: int | None = None
    ) -> list[ParcelRecord]:
        """Return parcels of a session, optionally filtered by AOI / year."""
        ...

    async def get_features(
        self, *, session_id: str, parcel_id: int, year: int
    ) -> FeatureRecord | None:
        """Return the feature row for a parcel/year, or ``None`` if absent."""
        ...


@runtime_checkable
class ChatMemory(Protocol):
    """Session conversation memory, session-scoped."""

    async def load_history(self, *, session_id: str, limit: int = 20) -> list[ChatTurn]:
        """Return the most recent turns (oldest first)."""
        ...

    async def append_turn(self, *, session_id: str, turn: ChatTurn) -> None:
        """Persist one turn."""
        ...


# ---------------------------------------------------------------------------
# Dependency container injected into the orchestrator.
# ---------------------------------------------------------------------------


class AgentDeps(BaseModel):
    """Everything the orchestrator needs from the outside world.

    ``arbitrary_types_allowed`` lets us carry adapter instances (Protocols) that
    are not Pydantic models. The backend builds this per request.
    """

    model_config = {"arbitrary_types_allowed": True}

    parcels: ParcelReader
    memory: ChatMemory


__all__ = [
    "AgentDeps",
    "ChatMemory",
    "ChatTurn",
    "FeatureRecord",
    "ParcelReader",
    "ParcelRecord",
]
