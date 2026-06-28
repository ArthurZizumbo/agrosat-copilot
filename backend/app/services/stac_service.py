"""STAC search service: pgstac-backed, degrades to an empty collection (US-053).

The endpoint speaks the STAC API ``search`` vocabulary. pgstac is **not deployed
yet** (``CREATE EXTENSION pgstac`` is commented out in the initial migration;
``db/CLAUDE.md``), so this service detects the extension once per process and
degrades elegantly:

- **pgstac absent** (current state) -> returns ``pystac.ItemCollection(items=[],
  extra_fields={"numberMatched": 0, "numberReturned": 0}).to_dict()``, a valid
  empty STAC ``FeatureCollection`` (``features: []``). HTTP ``200``.
- **pgstac present** -> ``SELECT * FROM pgstac.search($1::jsonb)`` and serialises
  the returned features as STAC items.

The output **shape is identical** in both cases (a STAC ``FeatureCollection``), so
the frontend never breaks. The detection is cached on the instance/process to
avoid probing ``pg_extension`` on every request; it is also defensive against the
extension appearing/disappearing only across deploys (not mid-process).

Every query runs on the request's RLS-scoped connection, consistent with the
agent's ``search_stac`` tool and every other DB-touching path.
"""

from __future__ import annotations

import json

import asyncpg
import structlog
from pystac import ItemCollection

from backend.app.models.geo import StacSearchQuery

__all__ = ["StacService"]

logger = structlog.get_logger(__name__)

# Probe whether the pgstac extension is installed. Returns a row only when
# present; cached per process so it is evaluated at most once.
_PGSTAC_PRESENT_SQL = "SELECT 1 FROM pg_extension WHERE extname = 'pgstac'"

# pgstac.search consumes the STAC API search request as a single JSONB argument
# and returns a GeoJSON FeatureCollection of matching items.
_PGSTAC_SEARCH_SQL = "SELECT pgstac.search($1::jsonb)"

# asyncpg raises these when pgstac.search / the pgstac schema is absent despite a
# stale ``pg_extension`` read; caught so the contract still degrades, never 500s.
_PGSTAC_MISSING_ERRORS: tuple[type[Exception], ...] = (
    asyncpg.UndefinedFunctionError,
    asyncpg.InvalidSchemaNameError,
    asyncpg.UndefinedTableError,
)

#: Process-wide cache of the pgstac detection (``None`` until first probed).
_PGSTAC_AVAILABLE: bool | None = None


def _empty_collection() -> dict:
    """Build a valid empty STAC ``FeatureCollection`` via pystac.

    Returns:
        The ``dict`` form of an empty :class:`pystac.ItemCollection` with the STAC
        API context counters set to zero (``features: []``).
    """
    collection = ItemCollection(
        items=[],
        extra_fields={"numberMatched": 0, "numberReturned": 0},
    )
    # pystac is untyped, so ``to_dict`` is inferred as ``Any``; pin the contract.
    result: dict = collection.to_dict()
    return result


def _build_search_request(query: StacSearchQuery) -> dict:
    """Translate the validated query into a STAC API ``search`` request body.

    Args:
        query: Validated :class:`StacSearchQuery` (bbox, datetime, collections,
            limit).

    Returns:
        A JSON-serialisable STAC API ``search`` request consumed by
        ``pgstac.search``. Optional fields are omitted when unset.
    """
    request: dict = {"limit": query.limit}
    if query.bbox is not None:
        request["bbox"] = query.bbox
    if query.datetime is not None:
        request["datetime"] = query.datetime
    if query.collections is not None:
        request["collections"] = query.collections
    return request


def _normalise_collection(raw: object, limit: int) -> dict:
    """Normalise the ``pgstac.search`` return value into a STAC FeatureCollection.

    Args:
        raw: Raw scalar returned by ``pgstac.search`` (JSON string or dict).
        limit: Page size echoed into the context counters when pgstac omits them.

    Returns:
        A STAC ``FeatureCollection`` dict with ``numberReturned`` populated.
    """
    if raw is None:
        return _empty_collection()
    decoded = json.loads(raw) if isinstance(raw, str | bytes) else raw
    if not isinstance(decoded, dict):
        return _empty_collection()
    collection: dict = decoded
    features = collection.get("features", [])
    if not isinstance(features, list):
        features = []
        collection["features"] = features
    collection.setdefault("type", "FeatureCollection")
    collection.setdefault("numberReturned", len(features))
    return collection


class StacService:
    """STAC search with graceful pgstac-absent degradation."""

    @staticmethod
    async def _pgstac_available(conn: asyncpg.Connection) -> bool:
        """Detect (once per process) whether the pgstac extension is installed.

        Args:
            conn: Any live connection (RLS-scoped is fine; ``pg_extension`` is
                catalog data, not tenant data).

        Returns:
            ``True`` if pgstac is installed, ``False`` otherwise. The result is
            memoised in the module-level cache.
        """
        global _PGSTAC_AVAILABLE
        if _PGSTAC_AVAILABLE is None:
            row = await conn.fetchrow(_PGSTAC_PRESENT_SQL)
            _PGSTAC_AVAILABLE = row is not None
            logger.info("pgstac_detected", available=_PGSTAC_AVAILABLE)
        return _PGSTAC_AVAILABLE

    @staticmethod
    async def search(conn: asyncpg.Connection, query: StacSearchQuery) -> dict:
        """Search the STAC catalogue; return an empty collection if pgstac absent.

        Args:
            conn: RLS-scoped connection (session primed, in a transaction).
            query: Validated search parameters.

        Returns:
            A STAC ``FeatureCollection`` dict. Empty (but valid) when pgstac is
            not deployed -- the contract is identical with or without pgstac, so
            the frontend is never broken.
        """
        if not await StacService._pgstac_available(conn):
            logger.info("stac_search_degraded", reason="pgstac_not_deployed")
            return _empty_collection()

        request = _build_search_request(query)
        try:
            raw = await conn.fetchval(_PGSTAC_SEARCH_SQL, json.dumps(request))
        except _PGSTAC_MISSING_ERRORS as exc:
            # The cached detection was stale (extension dropped); fail to empty.
            logger.warning("stac_search_degraded", reason="pgstac_missing", error=str(exc))
            return _empty_collection()

        collection = _normalise_collection(raw, query.limit)
        logger.info(
            "stac_search_completed",
            count=len(collection.get("features", [])),
        )
        return collection
