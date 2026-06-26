"""Tests for the US-078 runner ``scripts/build_italia_pastis.py``.

The runner orchestrates the builder end-to-end. These tests exercise the pure
helpers (``_parse_season``, ``_summarise``) directly and the ``run`` orchestrator
with EVERY network/IO dependency mocked:

- The Sentinel Hub client factory (``sh_client_from_settings``) returns a dummy;
  ``download_patch_series`` is monkeypatched to a deterministic toy ``_PatchStack``
  (NO ``_download_tile``, NO token, NO HTTP, ZERO SH requests).
- The labelled polygons + dense-patch selection are stubbed with toy plans so the
  test does not read the 643k-parcel DVC parquet.

The runner's value under test is the RESUME logic (a patch already on disk is not
re-downloaded -> 0 new requests) and the GATE summary aggregation, never a real
download.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
import pytest
from affine import Affine
from shapely.geometry import MultiPolygon, box

import scripts.build_italia_pastis as runner
from ml.data.eurocrops_pastis_builder import PATCH_PX, PatchPlan, PatchResult, _PatchStack

_SEASON = "2018-03-01..2018-10-31"


def _toy_stack(n_frames: int = 3, seed: int = 0) -> _PatchStack:
    rng = np.random.default_rng(seed)
    stack = rng.uniform(0.1, 0.2, size=(n_frames, 10, PATCH_PX, PATCH_PX)).astype(np.float32)
    transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, PATCH_PX * 10.0)
    return _PatchStack(stack=stack, transform=transform, crs="EPSG:3035", residual_cloud=0.04)


def _toy_plan(patch_id: int, fold: int = 0) -> PatchPlan:
    return PatchPlan(
        patch_id=patch_id,
        bbox_3035=(0.0, 0.0, 1280.0, 1280.0),
        bbox_4326=(11.0, 43.0, 11.01, 43.01),
        n_parcels=35,
        fold=fold,
        classes_present=(1, 2),
    )


def _toy_gdf() -> gpd.GeoDataFrame:
    """A minimal labelled GeoDataFrame whose parcels fall in the toy window."""
    geoms = [
        MultiPolygon([box(100.0, 1000.0, 400.0, 1200.0)]),
        MultiPolygon([box(600.0, 1000.0, 900.0, 1200.0)]),
    ]
    gdf = gpd.GeoDataFrame({"class_id": [1, 2]}, geometry=geoms, crs="EPSG:3035")
    return gdf


def _patch_runner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    plans: list[PatchPlan],
    stack_factory,
) -> dict[str, int]:
    """Wire the runner with toy data + mocked SH; return a call counter."""
    calls = {"download": 0}
    gdf = _toy_gdf()
    table = pl.DataFrame(
        {"class_id": [1, 2], "hcat4_name": ["maize", "vineyards"], "n_parcels": [20, 15]}
    )

    monkeypatch.setattr(
        runner, "load_labeled_polygons", lambda **_kw: (gdf, table)
    )
    monkeypatch.setattr(runner, "select_dense_patches", lambda _gdf, **_kw: plans)
    # No real Sentinel Hub client, no settings read from .env.local.
    monkeypatch.setattr(runner, "get_settings", lambda: object())
    monkeypatch.setattr(runner, "sh_client_from_settings", lambda _s: object())

    def _fake_download(_client, plan, **_kw):  # type: ignore[no-untyped-def]
        calls["download"] += 1
        return stack_factory(plan)

    monkeypatch.setattr(runner, "download_patch_series", _fake_download)
    return calls


# --------------------------------------------------------------------------- #
# Pure helpers.
# --------------------------------------------------------------------------- #
def test_parse_season_splits_from_to() -> None:
    assert runner._parse_season("2018-03-01..2018-10-31") == ("2018-03-01", "2018-10-31")


@pytest.mark.parametrize("bad", ["2018-03-01", "..2018-10-31", "2018-03-01..", ""])
def test_parse_season_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        runner._parse_season(bad)


def test_summarise_aggregates_gate_metrics() -> None:
    """The GATE summary averages coverage/dates/cloud and unions class support."""
    results = [
        (
            _toy_plan(0),
            PatchResult(
                patch_id=0,
                n_dates=10,
                coverage=0.8,
                n_classes_present=2,
                class_support={1: 100, 2: 50},
                residual_cloud=0.05,
                ok=True,
            ),
        ),
        (
            _toy_plan(1),
            PatchResult(
                patch_id=1,
                n_dates=20,
                coverage=0.6,
                n_classes_present=1,
                class_support={1: 200},
                residual_cloud=0.15,
                ok=True,
            ),
        ),
    ]
    summary = runner._summarise(results, n_requests=2)
    assert summary["n_patches"] == 2
    assert summary["mean_coverage"] == pytest.approx(0.7)
    assert summary["min_coverage"] == pytest.approx(0.6)
    assert summary["mean_dates"] == pytest.approx(15.0)
    assert summary["mean_residual_cloud"] == pytest.approx(0.1)
    assert summary["n_classes_present"] == 2  # union {1, 2}
    assert summary["class_support_pixels"] == {"1": 300, "2": 50}
    assert summary["n_requests"] == 2


def test_summarise_empty_is_honest() -> None:
    """No patches -> a minimal honest summary (no fabricated means)."""
    summary = runner._summarise([], n_requests=0)
    assert summary == {"n_patches": 0, "n_requests": 0}


# --------------------------------------------------------------------------- #
# run() orchestration with SH mocked.
# --------------------------------------------------------------------------- #
def test_run_builds_patches_and_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A fresh run downloads each plan once and writes the PASTIS layout."""
    plans = [_toy_plan(0, fold=0), _toy_plan(1, fold=1)]
    calls = _patch_runner(monkeypatch, plans=plans, stack_factory=lambda _p: _toy_stack())

    summary = runner.run(
        n_patches=2,
        season=_SEASON,
        min_support=2,
        n_frames=3,
        max_cloud=20.0,
        out_dir=tmp_path,
    )

    assert calls["download"] == 2
    assert summary["n_patches"] == 2
    assert summary["n_requests"] == 2
    # Artefacts on disk for both patches.
    for pid in (0, 1):
        assert (tmp_path / "DATA_S2" / f"S2_{pid}.npy").is_file()
        assert (tmp_path / "ANNOTATIONS" / f"TARGET_{pid}.npy").is_file()
    assert (tmp_path / "metadata.parquet").is_file()
    assert (tmp_path / "class_mapping.json").is_file()
    assert (tmp_path / "pilot_summary.json").is_file()


def test_run_resume_skips_written_patches_zero_new_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A second run re-uses the on-disk patches and issues 0 new SH requests."""
    plans = [_toy_plan(0, fold=0), _toy_plan(1, fold=1)]
    calls = _patch_runner(monkeypatch, plans=plans, stack_factory=lambda _p: _toy_stack())

    first = runner.run(
        n_patches=2, season=_SEASON, min_support=2, n_frames=3, max_cloud=20.0, out_dir=tmp_path
    )
    assert calls["download"] == 2 and first["n_requests"] == 2

    # Second run: nothing new is downloaded (resume); the report is still complete.
    second = runner.run(
        n_patches=2, season=_SEASON, min_support=2, n_frames=3, max_cloud=20.0, out_dir=tmp_path
    )
    assert calls["download"] == 2  # unchanged -> no re-download
    assert second["n_requests"] == 0
    assert second["n_patches"] == 2  # stats recomputed from disk


def test_run_skips_failed_tile_without_fabricating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A tile that returns None (all-cloud/failed) is dropped, the rest survive."""
    plans = [_toy_plan(0, fold=0), _toy_plan(1, fold=1)]

    def _stack_factory(plan: PatchPlan) -> _PatchStack | None:
        return None if plan.patch_id == 0 else _toy_stack()

    calls = _patch_runner(monkeypatch, plans=plans, stack_factory=_stack_factory)
    summary = runner.run(
        n_patches=2, season=_SEASON, min_support=2, n_frames=3, max_cloud=20.0, out_dir=tmp_path
    )
    assert calls["download"] == 2  # both attempted
    assert summary["n_patches"] == 1  # only the good one written
    assert summary["n_requests"] == 2  # both attempts counted honestly
    assert (tmp_path / "DATA_S2" / "S2_1.npy").is_file()
    assert not (tmp_path / "DATA_S2" / "S2_0.npy").is_file()
