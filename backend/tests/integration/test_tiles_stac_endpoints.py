"""``/tiles`` + ``/stac/search`` endpoint tests (US-053 AC-3 / AC-4).

Drives the real FastAPI app through ``httpx.AsyncClient`` + ``ASGITransport`` (no
live server, no network, no Docker). The DB boundary is faked: the auth-guard is
overridden to accept the test session, and ``get_scoped_conn`` yields a fake
connection whose ``fetchrow`` reports pgstac **absent** -- the current production
state -- so ``/stac/search`` exercises the graceful-degradation path end to end.

- AC-3: ``GET /tiles/{z}/{x}/{y}.png`` -> ``501`` with the documented JSON body
  naming US-055 and the contract path (a deliberate not-yet-implemented state).
- AC-4: ``GET /stac/search`` -> ``200`` with a valid empty STAC
  ``FeatureCollection`` (``features: []``, ``numberMatched: 0``) when pgstac is
  not deployed; the output shape is identical to the pgstac-present case.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Annotated
from uuid import UUID

import pytest
from fastapi import Depends
from httpx import ASGITransport, AsyncClient

import backend.app.services.stac_service as stac_mod
from backend.app.api.deps import verify_session
from backend.app.core import db as core_db
from backend.app.core.db import get_request_session_id
from backend.app.main import create_app

_VALID_SESSION = "11111111-1111-1111-1111-111111111111"


class _FakeConn:
    """Fake RLS-scoped connection. ``fetchrow`` reports pgstac absent."""

    async def fetchrow(self, *_args: object) -> None:
        # ``SELECT 1 FROM pg_extension WHERE extname='pgstac'`` -> no row.
        return None

    async def fetchval(self, *_args: object) -> None:  # pragma: no cover - not reached
        return None


@pytest.fixture(autouse=True)
def _reset_pgstac_cache() -> Iterator[None]:
    """Reset the process-wide pgstac detection cache around each test.

    ``StacService`` memoises the detection; tests must not leak that state.
    """
    stac_mod._PGSTAC_AVAILABLE = None
    yield
    stac_mod._PGSTAC_AVAILABLE = None


def _client() -> tuple[AsyncClient, object]:
    """Build an ``AsyncClient`` over the real app with the DB boundary faked."""
    app = create_app()

    async def _override_guard(
        session_id: Annotated[UUID, Depends(get_request_session_id)],
    ) -> UUID:
        return session_id

    async def _fake_scoped_conn() -> AsyncIterator[_FakeConn]:
        yield _FakeConn()

    app.dependency_overrides[verify_session] = _override_guard
    app.dependency_overrides[core_db.get_scoped_conn] = _fake_scoped_conn
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test"), app


async def test_tiles_returns_documented_501() -> None:
    """``GET /tiles/10/1/1.png`` returns ``501`` with the US-055 contract body."""
    client, _app = _client()
    async with client:
        resp = await client.get("/tiles/10/1/1.png", headers={"X-Session-ID": _VALID_SESSION})
    assert resp.status_code == 501
    body = resp.json()
    assert "US-055" in body["detail"]
    assert body["contract"] == "GET /tiles/{z}/{x}/{y}.png"


async def test_stac_search_empty_collection_when_pgstac_absent() -> None:
    """``GET /stac/search`` returns a valid empty STAC FeatureCollection (AC-4)."""
    client, _app = _client()
    async with client:
        resp = await client.get(
            "/stac/search",
            params={"bbox": [11.0, 43.0, 11.2, 43.2], "datetime": "2019-01-01/2019-12-31"},
            headers={"X-Session-ID": _VALID_SESSION},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "FeatureCollection"
    assert body["features"] == []
    assert body["numberMatched"] == 0
    assert body["numberReturned"] == 0


async def test_stac_search_no_params_still_ok() -> None:
    """A parameter-less ``/stac/search`` still returns an empty collection."""
    client, _app = _client()
    async with client:
        resp = await client.get("/stac/search", headers={"X-Session-ID": _VALID_SESSION})
    assert resp.status_code == 200
    assert resp.json()["features"] == []


async def test_stac_search_rejects_bad_bbox() -> None:
    """A malformed bbox (3 elements) is a ``422`` before touching the DB."""
    client, _app = _client()
    async with client:
        resp = await client.get(
            "/stac/search",
            params={"bbox": [11.0, 43.0, 11.2]},
            headers={"X-Session-ID": _VALID_SESSION},
        )
    assert resp.status_code == 422
