"""Helpers to extract and validate the tenant session id from a request.

Multi-tenant isolation in AgroSatCopilot is keyed on a per-request ``session_id``
(a UUID). The browser/SSE client sends it through the ``X-Session-ID`` HTTP
header (already whitelisted in the CORS configuration). This module centralises
parsing and validation so routers never read the header directly and every
handler receives a strongly-typed :class:`uuid.UUID`.
"""

from __future__ import annotations

from uuid import UUID

import structlog

__all__ = ["SESSION_ID_HEADER", "parse_session_id"]

logger = structlog.get_logger(__name__)

# Canonical header carrying the tenant session id. Kept in sync with the CORS
# ``allow_headers`` whitelist in ``backend.app.main``.
SESSION_ID_HEADER = "X-Session-ID"


def parse_session_id(raw: str | None) -> UUID:
    """Validate and parse a raw header value into a :class:`uuid.UUID`.

    Args:
        raw: The raw value of the ``X-Session-ID`` header, or ``None`` when the
            header is absent.

    Returns:
        The parsed session id.

    Raises:
        ValueError: If the header is missing, empty, or not a valid UUID. The
            FastAPI dependency layer maps this to an HTTP 400 response; the
            message never echoes the offending value to avoid leaking it into
            logs or responses.
    """
    if raw is None or not raw.strip():
        raise ValueError(f"Missing required header {SESSION_ID_HEADER!r}.")
    try:
        return UUID(raw.strip())
    except ValueError as exc:
        # Do not interpolate ``raw`` into the message: it is attacker-controlled
        # input and could pollute structured logs downstream.
        raise ValueError(f"Header {SESSION_ID_HEADER!r} is not a valid UUID.") from exc
