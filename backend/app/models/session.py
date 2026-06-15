"""``chat_sessions`` table model (multi-tenant root key)."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from sqlalchemy import text
from sqlmodel import Field, SQLModel

LlmVariant = Literal["gemini", "qwen35"]


class ChatSession(SQLModel, table=True):
    """A conversation session owned by a ``user_id``.

    Mirrors ``chat_sessions`` from the initial migration: UUID PK, ``llm_variant``
    constrained to ``gemini`` | ``qwen35`` (enforced by a DB CHECK), timestamps.
    """

    __tablename__ = "chat_sessions"

    id: uuid.UUID | None = Field(
        default=None,
        primary_key=True,
        sa_column_kwargs={"server_default": text("gen_random_uuid()")},
    )
    user_id: str
    llm_variant: str = Field(default="gemini")
    created_at: datetime | None = Field(
        default=None, sa_column_kwargs={"server_default": text("now()")}
    )
    updated_at: datetime | None = Field(
        default=None, sa_column_kwargs={"server_default": text("now()")}
    )
