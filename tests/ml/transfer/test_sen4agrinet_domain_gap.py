"""Tests for the FR/ES domain-gap materialization (US-073, B-073-1/2).

These exercise the GEE-free parts (netCDF centroid reading, UTM->lon/lat
reprojection, macro mapping, AOI bbox) against the REAL Sen4AgriNet subset on
disk. Tests that need the patches skip cleanly when the DVC data is absent so the
suite stays green in a data-less CI; no value is fabricated.
"""

from __future__ import annotations

import glob

import polars as pl
import pytest

from ml.transfer import sen4agrinet_domain_gap as dg

_HAS_ES = bool(glob.glob(dg.ES_PATCH_GLOB))
_HAS_FR = bool(glob.glob(dg.FR_PATCH_GLOB))
_skip_es = pytest.mark.skipif(not _HAS_ES, reason="Sen4AgriNet ES subset absent (dvc pull)")
_skip_fr = pytest.mark.skipif(not _HAS_FR, reason="Sen4AgriNet FR subset absent (dvc pull)")


@_skip_es
def test_es_centroids_are_in_catalonia() -> None:
    """ES centroids reproject into the Catalonia lon/lat box (tile 31TCG)."""
    es = dg.collect_centroids(patch_glob=dg.ES_PATCH_GLOB, region="ES", max_per_class=20)
    assert es.height > 0
    assert set(es.columns) == {"px_id", "lon", "lat", "macro", "region"}
    assert es.get_column("region").unique().to_list() == ["ES"]
    # Catalonia 31TCG sits around lon 0.5-0.7E, lat 41-42N.
    assert 0.0 < float(es.get_column("lon").min()) < 1.5  # type: ignore[arg-type]
    assert 40.5 < float(es.get_column("lat").min()) < 42.5  # type: ignore[arg-type]


@_skip_fr
def test_fr_centroids_are_north_of_es() -> None:
    """FR (31TCJ) centroids land at a clearly higher latitude than ES (the gap)."""
    fr = dg.collect_centroids(patch_glob=dg.FR_PATCH_GLOB, region="FR", max_per_class=20)
    assert fr.height > 0
    # PASTIS-FR tile 31TCJ is ~2 deg north of Catalonia: median lat well above 43N.
    assert float(fr.get_column("lat").median()) > 43.0  # type: ignore[arg-type]


@_skip_es
def test_macro_mapping_only_known_crops() -> None:
    """Every collected macro is a real shared HCAT crop group (no fabricated class)."""
    es = dg.collect_centroids(patch_glob=dg.ES_PATCH_GLOB, region="ES", max_per_class=10)
    macros = set(es.get_column("macro").unique().to_list())
    assert macros
    assert macros <= set(dg.MACRO_COLORS)


@_skip_es
def test_macro_aoi_bbox_is_well_formed() -> None:
    """The AOI bbox for a present macro is an ordered lon/lat rectangle."""
    es = dg.collect_centroids(patch_glob=dg.ES_PATCH_GLOB, region="ES", max_per_class=10)
    macro = es.get_column("macro")[0]
    bbox = dg.macro_aoi_bbox(es, macro)
    assert bbox is not None
    assert bbox[0] < bbox[2] and bbox[1] < bbox[3]


def test_macro_aoi_bbox_absent_returns_none() -> None:
    """An absent macro yields None (degraded, never a fabricated AOI)."""
    empty = pl.DataFrame({"lon": [], "lat": [], "macro": []})
    assert dg.macro_aoi_bbox(empty, "vineyard") is None


def test_ndvi_empty_frame_schema() -> None:
    """The degraded NDVI frame carries the canonical (date, doy, ndvi) schema."""
    frame = dg._ndvi_empty_frame()
    assert frame.columns == ["date", "doy", "ndvi"]
    assert frame.is_empty()
