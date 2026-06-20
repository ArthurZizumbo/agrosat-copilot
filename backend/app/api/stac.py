"""``/stac/search`` router: STAC item search, degrades elegantly (US-053).

Thin HTTP adapter (router -> service -> DB). Binds the query parameters into a
:class:`~backend.app.models.geo.StacSearchQuery`, resolves the RLS-scoped
connection and auth-guard, and delegates to
:class:`~backend.app.services.stac_service.StacService`.

pgstac is not deployed yet, so the service returns a valid empty STAC
``FeatureCollection`` (HTTP ``200``) rather than failing. The output shape is the
same whether or not pgstac is present, so the frontend contract is stable.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

import asyncpg
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import ValidationError

from backend.app.api.deps import verify_session
from backend.app.core.db import get_scoped_conn
from backend.app.models.geo import StacSearchQuery
from backend.app.services.stac_service import StacService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/stac", tags=["stac"])


@router.get("/search")
async def search_stac(
    _session: Annotated[UUID, Depends(verify_session)],
    conn: Annotated[asyncpg.Connection, Depends(get_scoped_conn)],
    bbox: Annotated[list[float] | None, Query()] = None,
    datetime: Annotated[str | None, Query()] = None,
    collections: Annotated[list[str] | None, Query()] = None,
    limit: Annotated[int, Query()] = 10,
) -> dict:
    """Search the STAC catalogue; returns an empty collection if pgstac is absent.

    Args:
        _session: Authorised tenant session (guard side effect; value unused).
        conn: RLS-scoped connection bound to the session.
        bbox: ``[minx, miny, maxx, maxy]`` in EPSG:4326, if filtering spatially.
        datetime: RFC 3339 instant or interval.
        collections: STAC collection ids to restrict the search.
        limit: Maximum number of items (1..1000).

    Returns:
        A STAC ``FeatureCollection`` dict (empty but valid when pgstac is not
        deployed). The contract is identical with or without pgstac.
    """
    # The query model carries domain validation (bbox shape/ranges, limit bound)
    # beyond FastAPI's per-parameter typing; surface its errors as ``422`` instead
    # of an unhandled ``500`` since it is constructed manually here.
    try:
        query = StacSearchQuery(bbox=bbox, datetime=datetime, collections=collections, limit=limit)
    except ValidationError as exc:
        # ``errors(include_context=...)`` excluded: the raw context holds the
        # ValueError object, which is not JSON-serialisable. The messages alone
        # are enough for the client to fix the query.
        messages = "; ".join(err["msg"] for err in exc.errors())
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid STAC search query: {messages}",
        ) from exc
    collection: dict = await StacService.search(conn, query)
    return collection
