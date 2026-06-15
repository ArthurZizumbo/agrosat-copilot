"""Synchronous DB-tool tests with a mocked asyncpg connection (US-045 AC-4).

Covers ``list_parcels``, ``get_parcel_timeseries`` and ``get_aoi_stats``. Each
tool's ``session_scoped_conn`` symbol is replaced by a fake that yields a
:class:`FakeConn` returning scripted rows, so no database is needed. The tests
assert the typed output shape, that the session id reaches the RLS hook, and the
honest empty-result behaviour (no fabricated data) when the DB has no rows.
"""

from __future__ import annotations

import json
from datetime import date

import ml.agent.tools.aoi_stats as aoi_stats_mod
import ml.agent.tools.parcels as parcels_mod
import ml.agent.tools.timeseries as timeseries_mod
from ml.agent.schemas import (
    AoiStats,
    AoiStatsInput,
    ListParcelsInput,
    ParcelList,
    ParcelTimeseriesInput,
    TimeSeries,
)

from .conftest import SESSION_A, FakeConn, FakeRecord, fake_session_scoped_conn

_POLYGON = {"type": "Polygon", "coordinates": [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]]}


# ---------------------------------------------------------------------------
# list_parcels
# ---------------------------------------------------------------------------
async def test_list_parcels_returns_typed_rows(monkeypatch, make_ctx) -> None:
    """``list_parcels`` maps DB rows to a typed ``ParcelList``."""
    conn = FakeConn(
        fetch_rows=[
            FakeRecord(id=1, crop_class="wheat", confidence=0.91),
            FakeRecord(id=2, crop_class="maize", confidence=0.77),
        ]
    )
    monkeypatch.setattr(parcels_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    out = await parcels_mod.run(ListParcelsInput(session_id=SESSION_A), make_ctx())

    assert isinstance(out, ParcelList)
    assert out.count == 2
    assert [p.parcel_id for p in out.parcels] == [1, 2]
    assert out.parcels[0].crop_class == "wheat"
    # RLS hook primed with the session id.
    assert conn.set_config_calls()[0][1] == (str(SESSION_A),)


async def test_list_parcels_empty_session(monkeypatch, make_ctx) -> None:
    """An empty session yields an empty list (count 0), nothing fabricated."""
    conn = FakeConn(fetch_rows=[])
    monkeypatch.setattr(parcels_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    out = await parcels_mod.run(ListParcelsInput(session_id=SESSION_A), make_ctx())

    assert out.count == 0
    assert out.parcels == []


async def test_list_parcels_with_aoi_passes_geojson(monkeypatch, make_ctx) -> None:
    """With an AOI, the GeoJSON is serialised and bound to the spatial query."""
    conn = FakeConn(fetch_rows=[FakeRecord(id=5, crop_class=None, confidence=None)])
    monkeypatch.setattr(parcels_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    out = await parcels_mod.run(
        ListParcelsInput(session_id=SESSION_A, aoi=_POLYGON), make_ctx()
    )

    assert out.count == 1
    assert out.parcels[0].crop_class is None
    # The AOI-restricted SQL must have used ST_Intersects and bound the GeoJSON.
    aoi_call = next(c for c in conn.calls if "ST_Intersects" in c[0])
    geojson_arg = aoi_call[1][1]
    assert json.loads(geojson_arg)["type"] == "Polygon"


# ---------------------------------------------------------------------------
# get_parcel_timeseries
# ---------------------------------------------------------------------------
def _ndvi_stats_json() -> str:
    """Build a realistic ``ndvi_stats`` JSONB string (asyncpg surfaces str)."""
    return json.dumps(
        {
            "NDVI_p05": 0.12,
            "NDVI_p25": 0.31,
            "NDVI_p50": 0.55,
            "NDVI_p75": 0.74,
            "NDVI_p95": 0.88,
            "EVI_p50": 0.40,
        }
    )


async def test_timeseries_ndvi_summary_within_window(monkeypatch, make_ctx) -> None:
    """NDVI series uses the stored percentiles plus the measured peak anchor."""
    conn = FakeConn(
        fetchrow_row=FakeRecord(
            ndvi_stats=_ndvi_stats_json(),
            sog_doy=90,
            peak_doy=180,  # 2019-06-29, inside the window
            peak_value=0.93,
            senescence_doy=270,
        )
    )
    monkeypatch.setattr(timeseries_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    out = await timeseries_mod.run(
        ParcelTimeseriesInput(
            session_id=SESSION_A,
            parcel_id=7,
            start=date(2019, 1, 1),
            end=date(2019, 12, 31),
            index="ndvi",
        ),
        make_ctx(),
    )

    assert isinstance(out, TimeSeries)
    assert out.parcel_id == 7
    assert out.index == "ndvi"
    # 5 percentiles + the measured peak (distinct date) => 6 aligned points.
    assert len(out.dates) == len(out.values) == 6
    assert out.dates == sorted(out.dates)  # ascending
    # Every value comes from a stored statistic; the real peak is surfaced.
    assert 0.93 in out.values
    # All percentile dates lie within the requested window.
    assert all(date(2019, 1, 1) <= d <= date(2019, 12, 31) for d in out.dates)


async def test_timeseries_empty_when_no_feature_row(monkeypatch, make_ctx) -> None:
    """No feature row (parcel not visible / no data) => empty series."""
    conn = FakeConn(fetchrow_row=None)
    monkeypatch.setattr(timeseries_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    out = await timeseries_mod.run(
        ParcelTimeseriesInput(
            session_id=SESSION_A,
            parcel_id=7,
            start=date(2019, 1, 1),
            end=date(2019, 12, 31),
            index="ndvi",
        ),
        make_ctx(),
    )

    assert out.dates == []
    assert out.values == []


async def test_timeseries_empty_when_index_missing(monkeypatch, make_ctx) -> None:
    """A feature row without the requested index's stats => empty series."""
    conn = FakeConn(
        fetchrow_row=FakeRecord(
            ndvi_stats=json.dumps({"NDVI_p50": 0.5}),  # no NDWI keys at all
            sog_doy=None,
            peak_doy=None,
            peak_value=None,
            senescence_doy=None,
        )
    )
    monkeypatch.setattr(timeseries_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    out = await timeseries_mod.run(
        ParcelTimeseriesInput(
            session_id=SESSION_A,
            parcel_id=7,
            start=date(2019, 1, 1),
            end=date(2019, 12, 31),
            index="ndwi",
        ),
        make_ctx(),
    )

    assert out.dates == []
    assert out.values == []


# ---------------------------------------------------------------------------
# get_aoi_stats
# ---------------------------------------------------------------------------
async def test_aoi_stats_aggregates_dominant_and_fractions(monkeypatch, make_ctx) -> None:
    """``get_aoi_stats`` reports area, dominant crop and per-class fractions."""
    conn = FakeConn(
        fetchrow_row=FakeRecord(area_sqm=120_000.0),  # 12 ha
        fetch_rows=[
            FakeRecord(crop_class="wheat", n=3),
            FakeRecord(crop_class="maize", n=1),
            FakeRecord(crop_class=None, n=2),  # unlabelled parcels still counted
        ],
    )
    monkeypatch.setattr(aoi_stats_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    out = await aoi_stats_mod.run(
        AoiStatsInput(session_id=SESSION_A, aoi=_POLYGON, year=2019), make_ctx()
    )

    assert isinstance(out, AoiStats)
    assert out.area_ha == 12.0
    assert out.n_parcels == 6  # includes the two unlabelled
    assert out.dominant_crop == "wheat"
    # Fractions are over labelled parcels only (3 + 1 = 4).
    assert out.crop_fractions["wheat"] == 0.75
    assert out.crop_fractions["maize"] == 0.25
    assert abs(sum(out.crop_fractions.values()) - 1.0) < 1e-9


async def test_aoi_stats_empty_aoi(monkeypatch, make_ctx) -> None:
    """An AOI with no intersecting parcels yields empty crop stats."""
    conn = FakeConn(
        fetchrow_row=FakeRecord(area_sqm=50_000.0),  # 5 ha footprint
        fetch_rows=[],
    )
    monkeypatch.setattr(aoi_stats_mod, "session_scoped_conn", fake_session_scoped_conn(conn))

    out = await aoi_stats_mod.run(
        AoiStatsInput(session_id=SESSION_A, aoi=_POLYGON, year=2019), make_ctx()
    )

    assert out.area_ha == 5.0
    assert out.n_parcels == 0
    assert out.dominant_crop == ""
    assert out.crop_fractions == {}
