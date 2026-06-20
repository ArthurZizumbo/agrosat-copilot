"""AOI CRUD service: GeoJSON <-> PostGIS over an RLS-scoped connection (US-053).

The service owns all SQL for the ``/aois`` endpoints; the router stays thin
(SoC). Every method receives the request's RLS-scoped :class:`asyncpg.Connection`
(``app.current_session`` already primed inside an open transaction by
:func:`backend.app.core.db.get_scoped_conn`), so the US-051 ``tenant_isolation``
policies enforce isolation: a ``SELECT``/``DELETE`` only ever touches the calling
session's rows, and an ``INSERT`` carrying another tenant's ``session_id`` is
rejected by the policy's WITH CHECK.

Geometry conversion lives in SQL: ``ST_SetSRID(ST_GeomFromGeoJSON($1), 4326)`` on
the way in (forced to the column SRID) and ``ST_AsGeoJSON(geom)`` on the way out.
No GeoAlchemy2 ``WKBElement`` is mapped in Python. ``area_ha`` is computed
geodesically server-side (``ST_Area(geom::geography) / 10000``), identical to the
agent's ``add_aoi`` tool (DRY of behaviour).
"""

from __future__ import annotations

import json

import asyncpg
import structlog

from backend.app.models.geo import (
    AoiCreate,
    AoiFeature,
    AoiFeatureCollection,
    AoiProperties,
)

__all__ = ["AoiService"]

logger = structlog.get_logger(__name__)

# INSERT relies on the RLS hook for ``session_id``: ``current_setting`` reads the
# primed ``app.current_session`` so the client can never spoof another owner
# (the WITH CHECK policy would reject it anyway). area_ha is geodesic.
_INSERT_SQL = """
INSERT INTO aois (session_id, geom, label, area_ha)
VALUES (
    current_setting('app.current_session')::uuid,
    ST_SetSRID(ST_GeomFromGeoJSON($1), 4326),
    $2,
    ST_Area(ST_SetSRID(ST_GeomFromGeoJSON($1), 4326)::geography) / 10000.0
)
RETURNING id, ST_AsGeoJSON(geom) AS geojson, label, area_ha, created_at
"""

# RLS filters every SELECT to the current session; no manual WHERE session_id.
_LIST_SQL = """
SELECT id, ST_AsGeoJSON(geom) AS geojson, label, area_ha, created_at
FROM aois
ORDER BY id
"""

_GET_SQL = """
SELECT id, ST_AsGeoJSON(geom) AS geojson, label, area_ha, created_at
FROM aois
WHERE id = $1
"""

# DELETE under RLS removes the row only if it belongs to the session; a foreign
# id matches zero rows (``DELETE 0``), surfaced as ``False`` -> ``404``.
_DELETE_SQL = "DELETE FROM aois WHERE id = $1"


def _row_to_feature(row: asyncpg.Record) -> AoiFeature:
    """Map an ``aois`` row (with ``ST_AsGeoJSON`` geometry) to a GeoJSON Feature.

    Args:
        row: An asyncpg record exposing ``id``, ``geojson`` (a JSON string from
            ``ST_AsGeoJSON``), ``label``, ``area_ha`` and ``created_at``.

    Returns:
        The AOI as an :class:`AoiFeature`.
    """
    return AoiFeature(
        id=int(row["id"]),
        geometry=json.loads(row["geojson"]),
        properties=AoiProperties(
            label=row["label"],
            area_ha=float(row["area_ha"]) if row["area_ha"] is not None else None,
            created_at=row["created_at"],
        ),
    )


class AoiService:
    """Stateless CRUD operations over the ``aois`` table (RLS-scoped)."""

    @staticmethod
    async def create(conn: asyncpg.Connection, body: AoiCreate) -> AoiFeature:
        """Persist a new AOI for the calling session and return it as a Feature.

        Args:
            conn: RLS-scoped connection (session primed, in a transaction).
            body: Validated request body (Polygon geometry + optional label).

        Returns:
            The created AOI as a GeoJSON :class:`AoiFeature` with its generated
            id, server-computed ``area_ha`` and ``created_at``.
        """
        geojson = json.dumps({"type": body.geometry.type, "coordinates": body.geometry.coordinates})
        row = await conn.fetchrow(_INSERT_SQL, geojson, body.label)
        # The row is never None: a successful INSERT ... RETURNING yields one row,
        # and a WITH CHECK violation raises before we get here.
        feature = _row_to_feature(row)
        logger.info("aoi_created", aoi_id=feature.id, area_ha=feature.properties.area_ha)
        return feature

    @staticmethod
    async def list(conn: asyncpg.Connection) -> AoiFeatureCollection:
        """List every AOI visible to the calling session.

        Args:
            conn: RLS-scoped connection (session primed, in a transaction).

        Returns:
            A GeoJSON :class:`AoiFeatureCollection`; empty when the session owns
            no AOIs (e.g. a foreign session sees nothing).
        """
        rows = await conn.fetch(_LIST_SQL)
        return AoiFeatureCollection(features=[_row_to_feature(r) for r in rows])

    @staticmethod
    async def get(conn: asyncpg.Connection, aoi_id: int) -> AoiFeature | None:
        """Fetch a single AOI by id, if it belongs to the calling session.

        Args:
            conn: RLS-scoped connection (session primed, in a transaction).
            aoi_id: Primary key of the AOI to fetch.

        Returns:
            The :class:`AoiFeature`, or ``None`` when the AOI does not exist or
            is owned by another tenant (RLS hides it). The router maps ``None``
            to ``404``.
        """
        row = await conn.fetchrow(_GET_SQL, aoi_id)
        return _row_to_feature(row) if row is not None else None

    @staticmethod
    async def delete(conn: asyncpg.Connection, aoi_id: int) -> bool:
        """Delete an AOI by id if it belongs to the calling session.

        Args:
            conn: RLS-scoped connection (session primed, in a transaction).
            aoi_id: Primary key of the AOI to delete.

        Returns:
            ``True`` when a row was deleted; ``False`` when the AOI does not
            exist or is owned by another tenant (RLS yields ``DELETE 0``). The
            router maps ``False`` to ``404``.
        """
        status_tag = await conn.execute(_DELETE_SQL, aoi_id)
        deleted = not status_tag.endswith(" 0")
        if not deleted:
            logger.info("aoi_delete_noop", aoi_id=aoi_id)
        return deleted
