"""Per-session rate limiting for the ``/chat`` endpoint (slowapi).

The orchestrator forbids ``/chat`` without a rate limit. US-052 enforces
**10 requests / minute per session** (not per IP): a single tenant flooding the
copilot cannot exhaust another tenant's budget. The key is therefore the
``X-Session-ID`` header (the auth-guard already requires it), not the client IP.

Design (plan section 2.2):

- A single :class:`~slowapi.Limiter` instance with ``key_func=session_id_key``,
  ``storage_uri=settings.redis_url`` (Redis in runtime; tests use ``memory://``
  or fakeredis) and **empty** ``default_limits`` -- the limit is applied only via
  the ``@limiter.limit(...)`` decorator on ``/chat``, NOT globally through
  ``SlowAPIMiddleware`` (which would limit every route, including ``/healthz``,
  and cannot key per session per endpoint).
- ``main.create_app`` wires ``app.state.limiter = limiter`` and registers the
  :class:`~slowapi.errors.RateLimitExceeded` handler so the 11th request in the
  window returns a JSON ``429`` (evaluated before the SSE stream opens, so there
  is no streaming-buffering gotcha).

The module exposes a process-wide ``limiter`` singleton because the
``@limiter.limit`` decorator needs the object at import time. Its storage URI is
read from :func:`~backend.app.core.config.get_settings` (never ``os.environ``).
"""

from __future__ import annotations

import structlog
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.app.core.config import Settings, get_settings
from backend.app.utils.session import SESSION_ID_HEADER

__all__ = [
    "CHAT_RATE_LIMIT",
    "LLM_SWITCH_RATE_LIMIT",
    "build_limiter",
    "limiter",
    "session_id_key",
]

logger = structlog.get_logger(__name__)


def session_id_key(request: Request) -> str:
    """Return the rate-limit bucket key for a request: its tenant session id.

    Keying on ``X-Session-ID`` (instead of the client IP) makes the limit
    **per session**: distinct sessions never share a bucket, so one tenant's
    burst cannot rate-limit another. When the header is absent the function
    falls back to the remote address so the limiter still has a stable key --
    in practice the ``verify_chat_session`` guard rejects header-less requests
    with ``400`` before the limit is consumed, so the fallback is only a safety
    net (and keeps anonymous routes, if ever decorated, from sharing a global
    bucket).

    Args:
        request: The incoming request, injected by slowapi.

    Returns:
        The ``X-Session-ID`` value, or the remote address when it is missing.
    """
    header: str | None = request.headers.get(SESSION_ID_HEADER)
    if header and header.strip():
        return header.strip()
    return str(get_remote_address(request))


def build_limiter(settings: Settings) -> Limiter:
    """Build the per-session :class:`~slowapi.Limiter` from typed settings.

    Args:
        settings: Application settings supplying ``redis_url`` as the slowapi
            storage backend (use ``memory://`` in tests).

    Returns:
        A :class:`~slowapi.Limiter` keyed by :func:`session_id_key` with no
        global default limits (only the ``/chat`` decorator limits a route).
    """
    return Limiter(
        key_func=session_id_key,
        storage_uri=settings.redis_url,
        default_limits=[],
        headers_enabled=True,
    )


def _chat_rate_limit(settings: Settings) -> str:
    """Render the ``/chat`` limit string from ``rate_limit_chat_per_min``.

    Args:
        settings: Application settings (``rate_limit_chat_per_min`` defaults to
            ``10``).

    Returns:
        A slowapi limit expression such as ``"10/minute"``.
    """
    return f"{settings.rate_limit_chat_per_min}/minute"


def _llm_switch_rate_limit(settings: Settings) -> str:
    """Render the ``/llm/switch`` limit string from ``rate_limit_llm_switch_per_min``.

    Args:
        settings: Application settings (``rate_limit_llm_switch_per_min`` defaults
            to ``5``).

    Returns:
        A slowapi limit expression such as ``"5/minute"``.
    """
    return f"{settings.rate_limit_llm_switch_per_min}/minute"


#: Process-wide limiter singleton. Built at import time from the cached settings
#: so the ``@limiter.limit(...)`` decorator on the ``/chat`` endpoint can
#: reference it. Tests rebuild it with ``memory://`` storage (or fakeredis) and
#: re-attach it to ``app.state``/the decorated route to stay deterministic and
#: Redis-free.
limiter: Limiter = build_limiter(get_settings())

#: The ``/chat`` limit expression (``"10/minute"`` by default), resolved once
#: from settings so the decorator never hardcodes the per-minute budget.
CHAT_RATE_LIMIT: str = _chat_rate_limit(get_settings())

#: The ``/llm/switch`` limit expression (``"5/minute"`` by default, US-054
#: AC-6), resolved once from settings so the decorator never hardcodes it.
LLM_SWITCH_RATE_LIMIT: str = _llm_switch_rate_limit(get_settings())
