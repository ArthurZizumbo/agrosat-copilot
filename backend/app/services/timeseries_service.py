"""AOI timeseries service: honest phenology anchors from the DB (US-053).

Aligned by construction with :mod:`ml.agent.tools.timeseries` (the agent's
``get_parcel_timeseries`` tool): AgroSatCopilot does **not** persist the raw daily
Sentinel-2 series. ``features_parcels`` stores yearly aggregates plus scalar NDVI
phenology anchors (``sog_doy``, ``peak_doy``, ``peak_value``, ``senescence_doy``).
The only point that pairs a real date with a real *measured* value is the NDVI
**peak** (``peak_value`` measured on ``peak_doy``); SOG/senescence have a
day-of-year but no stored value (threshold crossings, not observations), so
emitting them would fabricate a number -- forbidden (``ml/agent/CLAUDE.md``).

Therefore:

- ``NDVI`` -> at most one point (the in-window peak), the same anchor the tool
  surfaces.
- ``NDWI`` / ``NDMI`` -> an empty (honest) series: the DB holds no temporal
  anchor for them. NDMI in particular is not persisted at all yet (the feature
  pipeline computes NDVI/NDWI/EVI); it degrades to empty rather than fabricating.

The endpoint is AOI-scoped (the agent tool is parcel-scoped): an AOI may contain
several parcels, so the service resolves the AOI's parcels under RLS, joins their
yearly feature rows and emits the peak of the parcel whose ``peak_value`` is
highest in the window (the AOI's representative NDVI peak). Every query runs on
the request's RLS-scoped connection, so a foreign AOI/parcel is invisible.
"""

from __future__ import annotations

from datetime import date, timedelta

import asyncpg
import structlog

from backend.app.models.geo import TimeSeriesResponse

__all__ = ["TimeseriesService"]

logger = structlog.get_logger(__name__)

#: Default campaign year used when the AOI's parcels span multiple years; the
#: latest year present is selected so the series reflects the most recent data.
#: Resolved per request from the feature rows, not hardcoded.

# Confirm the AOI exists for the session (RLS hides foreign AOIs -> 0 rows).
_AOI_EXISTS_SQL = "SELECT 1 FROM aois WHERE id = $1"

# Resolve the AOI's parcels and their NDVI phenology anchors. RLS restricts both
# parcels and (indirectly) features_parcels to the session. The representative
# peak is the highest in-window peak_value across the AOI's parcels for the most
# recent year that has one.
_AOI_PEAKS_SQL = """
SELECT
    fp.year AS year,
    fp.peak_doy AS peak_doy,
    fp.peak_value AS peak_value
FROM parcels p
JOIN features_parcels fp ON fp.parcel_id = p.id
WHERE p.aoi_id = $1
  AND fp.peak_doy IS NOT NULL
  AND fp.peak_value IS NOT NULL
ORDER BY fp.year DESC, fp.peak_value DESC
"""


def _doy_to_date(year: int, doy: int) -> date | None:
    """Convert a 1-based day-of-year to a calendar date, or ``None`` if invalid.

    Mirrors :func:`ml.agent.tools.timeseries._doy_to_date`: a ``doy`` that rolls
    past the campaign year is rejected so the peak never lands in the next year.

    Args:
        year: Campaign year the day-of-year belongs to.
        doy: 1-based day of year (1..365/366).

    Returns:
        The corresponding :class:`datetime.date`, or ``None`` when out of range.
    """
    if doy < 1:
        return None
    candidate = date(year, 1, 1) + timedelta(days=doy - 1)
    if candidate.year != year:
        return None
    return candidate


class TimeseriesService:
    """Honest AOI-level timeseries reconstruction from stored phenology anchors."""

    @staticmethod
    async def aoi_exists(conn: asyncpg.Connection, aoi_id: int) -> bool:
        """Return whether the AOI exists and is visible to the calling session.

        Args:
            conn: RLS-scoped connection (session primed, in a transaction).
            aoi_id: Primary key of the AOI.

        Returns:
            ``True`` if the AOI belongs to the session; ``False`` otherwise (the
            router maps ``False`` to ``404``, never leaking foreign existence).
        """
        return await conn.fetchrow(_AOI_EXISTS_SQL, aoi_id) is not None

    @staticmethod
    async def for_aoi(
        conn: asyncpg.Connection,
        aoi_id: int,
        index: str,
        start: date,
        end: date,
    ) -> TimeSeriesResponse:
        """Build the AOI's index series from stored phenology anchors.

        For ``NDVI`` returns at most one point: the representative in-window peak
        across the AOI's parcels. For ``NDWI``/``NDMI`` returns an empty series
        (no temporal anchor persisted) -- never fabricated.

        Args:
            conn: RLS-scoped connection (session primed, in a transaction).
            aoi_id: AOI whose parcels are summarised (already verified to exist).
            index: Spectral index (``"NDVI"``/``"NDWI"``/``"NDMI"``).
            start: Inclusive window start.
            end: Inclusive window end.

        Returns:
            A :class:`TimeSeriesResponse` aligned date/value; empty when the AOI
            has no in-window NDVI peak or the index has no stored anchor.
        """
        dates: list[date] = []
        values: list[float] = []

        if index == "NDVI":
            rows = await conn.fetch(_AOI_PEAKS_SQL, aoi_id)
            for row in rows:
                peak_date = _doy_to_date(int(row["year"]), int(row["peak_doy"]))
                if peak_date is not None and start <= peak_date <= end:
                    dates.append(peak_date)
                    values.append(float(row["peak_value"]))
                    break  # representative peak: highest peak_value in newest year
            if not dates:
                logger.warning(
                    "timeseries_no_in_window_anchor",
                    aoi_id=aoi_id,
                    index=index,
                )
        else:
            # NDWI/NDMI: no measured temporal anchor in the DB -> honest empty.
            logger.warning(
                "timeseries_no_index_anchor",
                aoi_id=aoi_id,
                index=index,
                reason="only NDVI peak is persisted with a real date+value",
            )

        return TimeSeriesResponse(aoi_id=aoi_id, index=index, dates=dates, values=values)  # type: ignore[arg-type]
