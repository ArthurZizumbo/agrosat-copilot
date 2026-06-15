"""Agent-side conversation memory helpers (presentation / trimming only).

Real persistence lives in the backend adapter that implements the
:class:`ml.agent.ports.ChatMemory` Protocol (SQLModel over Postgres). This module
holds ONLY presentation logic: turning persisted :class:`ChatTurn` rows into the
:class:`ml.agent.backends.ChatMessage` list the LLM prompt expects, and trimming
the history so the context stays bounded.
"""

from __future__ import annotations

from ml.agent.backends import ChatMessage
from ml.agent.ports import ChatTurn

_DEFAULT_MAX_TURNS = 12


def trim_history(turns: list[ChatTurn], *, max_turns: int = _DEFAULT_MAX_TURNS) -> list[ChatTurn]:
    """Keep only the most recent ``max_turns`` turns (oldest first preserved).

    Args:
        turns: Persisted turns, oldest first (the ``ChatMemory`` contract).
        max_turns: Maximum number of trailing turns to keep.

    Returns:
        The last ``max_turns`` turns, oldest first.
    """
    if max_turns <= 0 or len(turns) <= max_turns:
        return list(turns)
    return list(turns[-max_turns:])


def history_to_messages(
    turns: list[ChatTurn], *, system_prompt: str, max_turns: int = _DEFAULT_MAX_TURNS
) -> list[ChatMessage]:
    """Build the LLM message list from persisted turns plus a system prompt.

    Args:
        turns: Persisted conversation turns, oldest first.
        system_prompt: The system instruction prepended to the context.
        max_turns: History trimming bound.

    Returns:
        ``[system, *history]`` as :class:`ChatMessage` objects.
    """
    messages: list[ChatMessage] = [ChatMessage(role="system", content=system_prompt)]
    for turn in trim_history(turns, max_turns=max_turns):
        messages.append(ChatMessage(role=turn.role, content=turn.content))
    return messages


__all__ = ["history_to_messages", "trim_history"]
