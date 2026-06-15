"""``get_parcel_timeseries`` tool: reconstruct a parcel index series from DB.

Honest-by-construction series reconstruction. AgroSatCopilot does **not** persist
the raw daily Sentinel-2 reflectance series in PostgreSQL; the ``features_parcels``
table stores yearly *aggregates* per parcel (see
``ml/features/persist_features.py``):

- ``ndvi_stats`` (JSONB): nine descriptive statistics per spectral index, keyed
  ``"{INDEX}_{stat}"`` with ``INDEX`` upper-case (``NDVI``, ``NDWI``, ``EVI``)
  and ``stat`` one of ``mean/std/min/max/p05/p25/p50/p75/p95``.
- scalar NDVI phenology columns (``sog_doy``, ``peak_doy``, ``peak_value``,
  ``senescence_doy``) capturing the NDVI growth curve's key day-of-year anchors.

This tool therefore returns a **distributional summary** of the requested index
across the campaign year, not a fabricated daily curve. The summary points are
the available percentiles (p05 -> p95) of the index, laid out on evenly spaced
dates inside the requested ``[start, end]`` window in ascending value order. For
NDVI specifically, where the DB also stores phenology day-of-year anchors, the
real ``peak_value`` is additionally emitted on its true ``peak_doy`` date when it
falls inside the window — those are measured, not interpolated.

No value is invented: every emitted value comes from a stored statistic. If the
parcel is not visible to the session, or carries no stats for the requested year
and index, an **empty** series is returned and a warning is logged. Every query
filters by ``session_id`` and runs inside
:func:`ml.agent.db.session_scoped_conn`.
"""

from __future__ import annotations

import json
import time
from datetime import date, timedelta

import structlog

from ml.agent.context import ToolContext
from ml.agent.db import session_scoped_conn
from ml.agent.schemas import ParcelTimeseriesInput, TimeSeries

__all__ = ["run"]

logger = structlog.get_logger(__name__)

# Map the lower-case API index name to the upper-case JSONB key prefix used by
# ``ml.features.temporal_features`` / ``ndvi_stats``.
_INDEX_KEY: dict[str, str] = {"ndvi": "NDVI", "ndwi": "NDWI", "evi": "EVI"}

# Percentile statistics used to lay out the distributional summary, in ascending
# nominal order so the resulting series rises monotonically across the window.
_PERCENTILE_STATS: tuple[str, ...] = ("p05", "p25", "p50", "p75", "p95")

# Joins the parcel (for session ownership) with its yearly feature row. The
# parcel filter by ``session_id`` is the multi-tenant guard; the LEFT-less INNER
# join means a parcel without a feature row yields no record (empty series).
_FEATURES_SQL = """
SELECT
    fp.ndvi_stats AS ndvi_stats,
    fp.sog_doy AS sog_doy,
    fp.peak_doy AS peak_doy,
    fp.peak_value AS peak_value,
    fp.senescence_doy AS senescence_doy
FROM parcels p
JOIN features_parcels fp ON fp.parcel_id = p.id
WHERE p.id = $1
  AND p.session_id = $2
  AND fp.year = $3
"""


def _normalise_jsonb(raw: object) -> dict:
    """Decode an asyncpg JSONB value into a dict.

    asyncpg surfaces ``jsonb`` columns as a JSON ``str`` unless a custom type
    codec is registered (none is, here), so the string is parsed. An already
    decoded ``dict`` is returned as-is; anything else yields an empty dict.

    Args:
        raw: The raw ``ndvi_stats`` value returned by asyncpg.

    Returns:
        The decoded mapping, or an empty dict when the value is null/malformed.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (str, bytes)):
        decoded = json.loads(raw)
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _doy_to_date(year: int, doy: int) -> date | None:
    """Convert a 1-based day-of-year to a calendar date, or ``None`` if invalid.

    A ``doy`` that would roll past the campaign year (e.g. ``> 366``) is rejected
    so the peak anchor never silently lands in the following year.

    Args:
        year: Campaign year the day-of-year belongs to.
        doy: 1-based day of year (1..365/366).

    Returns:
        The corresponding :class:`datetime.date`, or ``None`` when ``doy`` is
        outside the valid range for ``year``.
    """
    if doy < 1:
        return None
    candidate = date(year, 1, 1) + timedelta(days=doy - 1)
    if candidate.year != year:
        return None
    return candidate


def _spread_dates(start: date, end: date, n: int) -> list[date]:
    """Return ``n`` ascending dates evenly spread across ``[start, end]``.

    Args:
        start: Inclusive window start.
        end: Inclusive window end.
        n: Number of dates to produce (``>= 1``).

    Returns:
        ``n`` dates from ``start`` to ``end`` inclusive. With ``n == 1`` the
        midpoint is returned; the span is clamped so all dates stay in-window.
    """
    span_days = (end - start).days
    if n == 1:
        return [start + timedelta(days=span_days // 2)]
    step = span_days / (n - 1)
    return [start + timedelta(days=round(step * i)) for i in range(n)]


def _empty_series(parcel_id: int, index: str) -> TimeSeries:
    """Build an empty (no-data) :class:`TimeSeries` for the parcel/index."""
    return TimeSeries(parcel_id=parcel_id, index=index, dates=[], values=[])


async def run(inp: ParcelTimeseriesInput, ctx: ToolContext) -> TimeSeries:
    """Reconstruct a distributional index summary for a parcel over a window.

    Args:
        inp: Validated arguments (session, parcel, date window, index).
        ctx: Tool execution context (session-scoped pool access).

    Returns:
        A :class:`TimeSeries` whose points are stored statistics of the index
        across the campaign year, placed inside the requested window. Empty when
        the parcel is not visible to the session or has no stored stats for the
        year/index.
    """
    started = time.perf_counter()
    logger.info(
        "tool_call_started",
        tool="get_parcel_timeseries",
        session_id=str(ctx.session_id),
        parcel_id=inp.parcel_id,
        index=inp.index,
    )

    # The campaign year is taken from the window's start; the stored aggregates
    # are annual, so the start year selects the feature row to summarise.
    feature_year = inp.start.year

    async with session_scoped_conn(inp.session_id) as conn:
        record = await conn.fetchrow(
            _FEATURES_SQL,
            inp.parcel_id,
            inp.session_id,
            feature_year,
        )

    if record is None:
        logger.warning(
            "timeseries_no_feature_row",
            tool="get_parcel_timeseries",
            session_id=str(ctx.session_id),
            parcel_id=inp.parcel_id,
            year=feature_year,
        )
        return _empty_series(inp.parcel_id, inp.index)

    ndvi_stats = _normalise_jsonb(record["ndvi_stats"])
    index_prefix = _INDEX_KEY[inp.index]

    # Collect the available percentile stats for this index, in ascending order.
    available: list[float] = []
    for stat in _PERCENTILE_STATS:
        raw = ndvi_stats.get(f"{index_prefix}_{stat}")
        if raw is not None:
            available.append(float(raw))

    if not available:
        logger.warning(
            "timeseries_no_index_stats",
            tool="get_parcel_timeseries",
            session_id=str(ctx.session_id),
            parcel_id=inp.parcel_id,
            index=inp.index,
            year=feature_year,
        )
        return _empty_series(inp.parcel_id, inp.index)

    # Lay the percentile summary on evenly spaced in-window dates.
    summary_dates = _spread_dates(inp.start, inp.end, len(available))
    points: list[tuple[date, float]] = list(zip(summary_dates, available, strict=True))

    # For NDVI, add the measured peak (real value on its real day-of-year) when
    # it is stored and falls inside the requested window. This is the one truly
    # temporal anchor the DB holds, so it is surfaced verbatim (not interpolated).
    if inp.index == "ndvi" and record["peak_doy"] is not None and record["peak_value"] is not None:
        peak_date = _doy_to_date(feature_year, int(record["peak_doy"]))
        if peak_date is not None and inp.start <= peak_date <= inp.end:
            points.append((peak_date, float(record["peak_value"])))

    # Sort chronologically and de-duplicate dates (the peak may coincide with a
    # spread date); keep the measured value when a collision occurs by appending
    # the peak last and preferring the last occurrence per date.
    by_date: dict[date, float] = {}
    for point_date, value in sorted(points, key=lambda item: item[0]):
        by_date[point_date] = value
    ordered_dates = sorted(by_date)
    dates = list(ordered_dates)
    values = [by_date[d] for d in ordered_dates]

    duration_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "tool_call_finished",
        tool="get_parcel_timeseries",
        session_id=str(ctx.session_id),
        parcel_id=inp.parcel_id,
        index=inp.index,
        n_points=len(dates),
        duration_ms=round(duration_ms, 2),
    )
    return TimeSeries(
        parcel_id=inp.parcel_id,
        index=inp.index,
        dates=dates,
        values=values,
    )
