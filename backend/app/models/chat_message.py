"""``chat_messages`` table model (persisted conversation memory)."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import text
from sqlmodel import Field, SQLModel


class ChatMessage(SQLModel, table=True):
    """One persisted conversation turn. Mirrors ``chat_messages``.

    ``role`` is constrained to ``user`` | ``assistant`` by a DB CHECK. Every read
    filters by ``session_id`` (multi-tenant NON-NEGOTIABLE).
    """

    __tablename__ = "chat_messages"

    id: int | None = Field(default=None, primary_key=True)
    session_id: uuid.UUID = Field(foreign_key="chat_sessions.id", index=True)
    role: str
    content: str
    created_at: datetime | None = Field(
        default=None, sa_column_kwargs={"server_default": text("now()")}
    )
