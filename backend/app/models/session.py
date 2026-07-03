"""Pydantic models for the chat-session lifecycle and history (US-080).

The in-app multi-chat UI creates one session per chat tab and restores its
transcript from the server. These models are the wire contract for the
``/sessions`` router; the persisted shape lives in the ``chat_sessions`` and
``chat_messages`` tables.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

__all__ = [
    "ChatMessageOut",
    "SessionCreate",
    "SessionOut",
    "SessionRename",
]

#: Allowed reasoner variants, mirroring the ``chat_sessions.llm_model`` CHECK.
#: ``qwen-vl`` is the on-prem multimodal Qwen3.6-VL host (E12).
LlmModel = Literal["gemini", "qwen-api", "qwen-onprem", "gemma", "qwen-vl"]


class SessionCreate(BaseModel):
    """Request body for ``POST /sessions`` (create a chat tab's session)."""

    title: str | None = Field(default=None, max_length=200)
    llm_model: LlmModel = "gemini"


class SessionRename(BaseModel):
    """Request body for ``PATCH /sessions/{id}`` (rename a chat tab)."""

    title: str = Field(min_length=1, max_length=200)


class SessionOut(BaseModel):
    """A chat session as returned by the API."""

    id: UUID
    title: str | None = None
    llm_model: str
    created_at: datetime


class ChatMessageOut(BaseModel):
    """A persisted chat turn restored from the server."""

    id: int
    role: str
    content: str
    extra: dict[str, Any] | None = None
    created_at: datetime
