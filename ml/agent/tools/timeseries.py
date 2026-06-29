"""``get_parcel_timeseries`` tool: surface a parcel's real phenology anchors.

Honest-by-construction series reconstruction. AgroSatCopilot does **not** persist
the raw daily Sentinel-2 reflectance series in PostgreSQL; the ``features_parcels``
table stores yearly *aggregates* per parcel (see
``ml/features/persist_features.py``):

- ``ndvi_stats`` (JSONB): nine descriptive statistics per spectral index, keyed
  ``"{INDEX}_{stat}"`` with ``INDEX`` upper-case (``NDVI``, ``NDWI``, ``EVI``)
  and ``stat`` one of ``mean/std/min/max/p05/p25/p50/p75/p95``.
- scalar NDVI phenology columns (``sog_doy``, ``peak_doy``, ``peak_value``,
  ``senescence_doy``) capturing the NDVI growth curve's key day-of-year anchors.

This tool returns the **real phenology anchors** the DB holds: points whose date
is a genuine day-of-year stored for the parcel and whose value is the measured
index value at that date. Concretely, the only anchor that pairs a real date with
a real stored value is the NDVI **peak** (``peak_value`` measured on ``peak_doy``).
SOG and senescence have a stored day-of-year but **no stored value** (they are
threshold *crossings*, not measured observations), so they are not emitted — the
agent must never read a fabricated value off them. NDWI and EVI carry no temporal
anchor at all in the DB.

Rationale (``ml/agent/CLAUDE.md``): "Cifra (NDVI, fechas) sin origen en un tool
call" is forbidden. Earlier this tool spread the percentile distribution (p05->p95)
over evenly spaced dates inside ``[start, end]`` to fake an ascending curve; those
dates corresponded to **no observation** and collapsed to duplicates on short
windows. That fabrication is removed: every emitted date is a real day-of-year and
every emitted value is the measured value at that date.

Consequently the series is short by design (at most one point: the NDVI peak). It
is **not** a daily curve and is **not** a distributional summary. If the parcel is
not visible to the session, carries no feature row, or has no in-window phenology
anchor for the requested index, an **empty** series is returned and a warning is
logged. Every query filters by ``session_id`` and runs inside
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
# ``ml.features.temporal_features`` / ``ndvi_stats``. Retained so the per-index
# feature row can still be validated as present (no fabricated values otherwise).
_INDEX_KEY: dict[str, str] = {"ndvi": "NDVI", "ndwi": "NDWI", "evi": "EVI"}

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
    if isinstance(raw, str | bytes):
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


def _empty_series(parcel_id: int, index: str) -> TimeSeries:
    """Build an empty (no-data) :class:`TimeSeries` for the parcel/index."""
    return TimeSeries(parcel_id=parcel_id, index=index, dates=[], values=[])


async def run(inp: ParcelTimeseriesInput, ctx: ToolContext) -> TimeSeries:
    """Surface the parcel's real in-window phenology anchors for an index.

    Returns only points with a genuine stored date paired with a measured value.
    In practice this is the NDVI peak (``peak_value`` on ``peak_doy``) when it is
    stored and falls inside the requested window; no value is interpolated and no
    date is fabricated (see module docstring). NDWI/EVI have no temporal anchor in
    the DB and therefore yield an empty series.

    Args:
        inp: Validated arguments (session, parcel, date window, index).
        ctx: Tool execution context (session-scoped pool access).

    Returns:
        A :class:`TimeSeries` with the real phenology anchors (at most the NDVI
        peak). Empty when the parcel is not visible to the session, has no stored
        stats for the year/index, or has no in-window anchor.
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

    # The index must actually be present in the stored aggregates; otherwise the
    # parcel carries no information for it and the honest answer is empty. (We do
    # not emit the percentile stats themselves because they have no real date.)
    has_index_stats = any(
        ndvi_stats.get(f"{index_prefix}_{stat}") is not None
        for stat in ("p05", "p25", "p50", "p75", "p95", "mean", "min", "max")
    )
    if not has_index_stats:
        logger.warning(
            "timeseries_no_index_stats",
            tool="get_parcel_timeseries",
            session_id=str(ctx.session_id),
            parcel_id=inp.parcel_id,
            index=inp.index,
            year=feature_year,
        )
        return _empty_series(inp.parcel_id, inp.index)

    # The only anchor that pairs a real date with a real *measured* value is the
    # NDVI peak. SOG/senescence store a day-of-year but no value (threshold
    # crossings, not observations), so emitting them would fabricate a value --
    # forbidden. NDWI/EVI have no phenology anchor at all in the DB.
    dates: list[date] = []
    values: list[float] = []
    if inp.index == "ndvi" and record["peak_doy"] is not None and record["peak_value"] is not None:
        peak_date = _doy_to_date(feature_year, int(record["peak_doy"]))
        if peak_date is not None and inp.start <= peak_date <= inp.end:
            dates.append(peak_date)
            values.append(float(record["peak_value"]))

    if not dates:
        logger.warning(
            "timeseries_no_in_window_anchor",
            tool="get_parcel_timeseries",
            session_id=str(ctx.session_id),
            parcel_id=inp.parcel_id,
            index=inp.index,
            year=feature_year,
        )
        return _empty_series(inp.parcel_id, inp.index)

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
