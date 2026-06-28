"""Tests for :mod:`ml.data.eurocrops_pastis_builder` (US-078).

The builder turns EuroCrops Italy 2018 polygons into the PASTIS-R dense layout
(``S2_<id>.npy`` + ``TARGET_<id>.npy`` + ``dates_<id>.npy``). These tests exercise
its pure, testable steps WITHOUT any network call: Sentinel Hub / CDSE / GEE are
never touched (the only step that downloads, :func:`download_patch_series`, is
covered through the runner tests with ``_download_tile`` mocked). Here we test:

- The class table (``original_code`` -> HCAT name -> contiguous id, rare folded).
- ``load_labeled_polygons`` dropping empty geometries and attaching class ids,
  reading a tiny synthetic parquet + CSV crosswalk (no DVC blob).
- The dense-patch selection ranking cells by AREA (a dense cell of large parcels
  beats a cell of many tiny parcels) and the deterministic spatial fold.
- The rasterisation of synthetic MultiPolygons onto a known grid: the mask is
  ``(128, 128)`` with the right class ids and ``fill=0`` background.
- The PASTIS-format persistence (shapes, dtypes, DN scaling) and the resume
  helpers (``patch_artifacts_exist`` / ``load_patch_result_stats``).

All geometries are TOY MultiPolygons built with shapely and arrays seeded with a
fixed RNG (the synthetic-data ban applies to experiment RESULTS, not test
fixtures). Nothing here spends an SH request or touches a GPU.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
import pytest
from affine import Affine
from shapely.geometry import MultiPolygon, Polygon, box

from ml.data import eurocrops_pastis_builder as builder
from ml.data.eurocrops_pastis_builder import (
    BACKGROUND_ID,
    PASTIS_BANDS,
    PATCH_PX,
    PatchPlan,
    _PatchStack,
    build_class_table,
    load_labeled_polygons,
    load_patch_result_stats,
    patch_artifacts_exist,
    rasterize_patch_mask,
    save_pastis_format,
    select_dense_patches,
    write_class_mapping_doc,
    write_metadata,
)

_SEASON = ("2018-03-01", "2018-10-31")


# --------------------------------------------------------------------------- #
# Fixtures: toy crosswalk CSV + toy parcels parquet (no DVC blob, no network).
# --------------------------------------------------------------------------- #
def _square(cx: float, cy: float, side: float) -> MultiPolygon:
    """A toy square MultiPolygon centred at ``(cx, cy)`` (EuroCrops are multi)."""
    half = side / 2.0
    return MultiPolygon([box(cx - half, cy - half, cx + half, cy + half)])


def _write_crosswalk(tmp_path: Path) -> Path:
    """Write a tiny EuroCrops mapping CSV (Italy rows + a foreign row to filter)."""
    csv_path = tmp_path / "eurocrops_mapping.csv"
    pl.DataFrame(
        {
            "original_code": ["A01", "A02", "A03", "ZZ9"],
            "hcat4_name": ["maize", "vineyards", "rare_crop", "should_be_dropped"],
            "nuts": ["ITI1", "ITF3", "ITC4", "FR10"],
        }
    ).write_csv(csv_path)
    return csv_path


def _make_parcels_gdf(
    *,
    n_maize: int = 6,
    n_vine: int = 4,
    n_rare: int = 1,
    n_empty: int = 2,
) -> gpd.GeoDataFrame:
    """Build a toy labelled GeoDataFrame in EPSG:3035 (metric)."""
    rows: list[dict[str, object]] = []
    geoms: list[object] = []
    # Maize: a compact dense cluster of medium parcels near (0, 0).
    for i in range(n_maize):
        geoms.append(_square(100.0 + i * 40.0, 100.0, 30.0))
        rows.append({"original_code": "A01", "area_ha": 4.0})
    # Vineyards: in the SAME patch cell (so the patch is multi-class).
    for i in range(n_vine):
        geoms.append(_square(300.0 + i * 40.0, 300.0, 30.0))
        rows.append({"original_code": "A02", "area_ha": 3.0})
    # Rare crop: a single parcel (folded into "other" with min_support>1).
    for _ in range(n_rare):
        geoms.append(_square(700.0, 700.0, 20.0))
        rows.append({"original_code": "A03", "area_ha": 2.0})
    # Empty geometries (EuroCrops Italy ships ~908): must be dropped lossless.
    for _ in range(n_empty):
        geoms.append(Polygon())
        rows.append({"original_code": "A01", "area_ha": 0.0})
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:3035")
    return gdf


def _toy_patch_stack(
    *, n_frames: int = 3, fill: float = 0.12, seed: int = 0
) -> _PatchStack:
    """A toy aligned patch stack on a known 10 m grid at the origin.

    The transform places pixel (0, 0) at world (0, 0) with 10 m pixels (y down),
    so toy parcels at small positive world coords land near the top-left.
    """
    rng = np.random.default_rng(seed)
    stack = rng.uniform(
        fill, fill + 0.05, size=(n_frames, 10, PATCH_PX, PATCH_PX)
    ).astype(np.float32)
    # y-down transform: row 0 at top (max y), 10 m/px.
    transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, PATCH_PX * 10.0)
    return _PatchStack(stack=stack, transform=transform, crs="EPSG:3035", residual_cloud=0.05)


# --------------------------------------------------------------------------- #
# Step 1 -- class table + labelled polygons.
# --------------------------------------------------------------------------- #
def test_build_class_table_assigns_contiguous_ids_and_folds_rare() -> None:
    """HCAT names get ids [1, K]; classes below min_support fold into 'other'."""
    gdf = _make_parcels_gdf()
    gdf = gdf[~gdf.geometry.is_empty].reset_index(drop=True)
    gdf["hcat4_name"] = [
        {"A01": "maize", "A02": "vineyards", "A03": "rare_crop"}[c]
        for c in gdf["original_code"]
    ]

    table, name_to_id = build_class_table(gdf, min_support=2)

    # Background id 0 is reserved; crop ids start at 1 and are contiguous.
    ids = sorted(table["class_id"].to_list())
    assert ids == list(range(1, len(ids) + 1))
    assert BACKGROUND_ID == 0 and 0 not in ids
    # The rare single-parcel class collapses into the explicit "other" bucket.
    assert name_to_id["rare_crop"] == name_to_id_of(table, "other")
    names = set(table["hcat4_name"].to_list())
    assert "other" in names and "rare_crop" not in names
    assert {"maize", "vineyards"} <= names


def name_to_id_of(table: pl.DataFrame, name: str) -> int:
    """Helper: class_id of a name in the class table."""
    row = table.filter(pl.col("hcat4_name") == name)
    return int(row["class_id"][0])


def test_build_class_table_keeps_all_when_min_support_one() -> None:
    """With min_support=1 nothing is folded: every distinct HCAT keeps its id."""
    gdf = _make_parcels_gdf()
    gdf = gdf[~gdf.geometry.is_empty].reset_index(drop=True)
    gdf["hcat4_name"] = [
        {"A01": "maize", "A02": "vineyards", "A03": "rare_crop"}[c]
        for c in gdf["original_code"]
    ]
    table, _ = build_class_table(gdf, min_support=1)
    assert set(table["hcat4_name"].to_list()) == {"maize", "vineyards", "rare_crop"}
    assert "other" not in table["hcat4_name"].to_list()


def test_load_labeled_polygons_drops_empty_and_maps_classes(tmp_path: Path) -> None:
    """Empty geometries are dropped; class ids + projected centroids are attached."""
    parcels = tmp_path / "iti1_2018.parquet"
    gdf = _make_parcels_gdf(n_empty=2)
    gdf.to_parquet(parcels)
    csv_path = _write_crosswalk(tmp_path)

    labelled, table = load_labeled_polygons(
        parcels_parquet=parcels, mapping_csv=csv_path, min_support=2
    )

    # The two empty geometries are gone (lossless drop).
    assert len(labelled) == len(gdf) - 2
    assert (~labelled.geometry.is_empty).all()
    # class_id present, int, background never assigned to a real parcel.
    assert "class_id" in labelled.columns
    assert labelled["class_id"].min() >= 1
    # Projected centroids computed in the metric CRS (not NaN, not inf).
    assert np.isfinite(labelled["cx"].to_numpy()).all()
    assert np.isfinite(labelled["cy"].to_numpy()).all()
    # rare_crop (1 parcel) folded into 'other'.
    assert "other" in table["hcat4_name"].to_list()


def test_load_labeled_polygons_missing_parquet_raises(tmp_path: Path) -> None:
    """A missing parcels parquet raises a clear FileNotFoundError (dvc pull hint)."""
    with pytest.raises(FileNotFoundError, match="parcels parquet"):
        load_labeled_polygons(
            parcels_parquet=tmp_path / "absent.parquet",
            mapping_csv=_write_crosswalk(tmp_path),
            min_support=2,
        )


# --------------------------------------------------------------------------- #
# Step 2 -- dense-patch selection by AREA + spatial fold.
# --------------------------------------------------------------------------- #
def _labelled_with_ids(tmp_path: Path, min_support: int = 2) -> gpd.GeoDataFrame:
    parcels = tmp_path / "iti1_2018.parquet"
    _make_parcels_gdf().to_parquet(parcels)
    gdf, _ = load_labeled_polygons(
        parcels_parquet=parcels,
        mapping_csv=_write_crosswalk(tmp_path),
        min_support=min_support,
    )
    return gdf


def test_select_dense_patches_ranks_by_area_not_count() -> None:
    """A cell of few LARGE parcels beats a cell of many TINY ones (AC3 coverage).

    Two cells, each above the parcel-count floor: cell A has 31 tiny parcels,
    cell B has 31 large parcels. Ranking by AREA must put B first.
    """
    rows: list[dict[str, object]] = []
    geoms: list[object] = []
    # Cell A near (5_000, 5_000): 31 tiny parcels (small area).
    for i in range(31):
        x = 5_000.0 + (i % 6) * 20.0
        y = 5_000.0 + (i // 6) * 20.0
        geoms.append(_square(x, y, 5.0))
        rows.append({"class_id": 1, "area_ha": 0.1})
    # Cell B near (50_000, 50_000): 31 large parcels (big area).
    for i in range(31):
        x = 50_000.0 + (i % 6) * 100.0
        y = 50_000.0 + (i // 6) * 100.0
        geoms.append(_square(x, y, 80.0))
        rows.append({"class_id": 2, "area_ha": 9.0})
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:3035")
    gdf["cx"] = gdf.geometry.centroid.x.to_numpy()
    gdf["cy"] = gdf.geometry.centroid.y.to_numpy()

    plans = select_dense_patches(gdf, n_patches=2, min_parcels=30)
    assert len(plans) == 2
    # The first (highest-area) patch is the large-parcel cell B (class 2).
    assert 2 in plans[0].classes_present
    assert plans[0].n_parcels == 31


def test_select_dense_patches_drops_below_parcel_floor() -> None:
    """Cells under the parcel-count floor (single-field cells) are excluded."""
    rows: list[dict[str, object]] = []
    geoms: list[object] = []
    for i in range(5):  # only 5 parcels -> below a floor of 30
        geoms.append(_square(1_000.0 + i * 50.0, 1_000.0, 40.0))
        rows.append({"class_id": 1, "area_ha": 12.0})
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:3035")
    gdf["cx"] = gdf.geometry.centroid.x.to_numpy()
    gdf["cy"] = gdf.geometry.centroid.y.to_numpy()
    assert select_dense_patches(gdf, n_patches=4, min_parcels=30) == []


def test_select_dense_patches_bbox_and_fold_are_deterministic(tmp_path: Path) -> None:
    """The selected patch carries a 4326 bbox and a stable spatial fold."""
    rows: list[dict[str, object]] = []
    geoms: list[object] = []
    for i in range(40):
        x = 4_000_000.0 + (i % 8) * 60.0  # realistic 3035 coords (Italy band)
        y = 2_500_000.0 + (i // 8) * 60.0
        geoms.append(_square(x, y, 40.0))
        rows.append({"class_id": 1 + (i % 2), "area_ha": 5.0})
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:3035")
    gdf["cx"] = gdf.geometry.centroid.x.to_numpy()
    gdf["cy"] = gdf.geometry.centroid.y.to_numpy()

    plans = select_dense_patches(gdf, n_patches=1, min_parcels=30)
    assert len(plans) == 1
    plan = plans[0]
    # 4326 bbox is a real lon/lat box inside Italy's rough envelope.
    min_lon, min_lat, max_lon, max_lat = plan.bbox_4326
    assert -10.0 < min_lon < 25.0 and 30.0 < min_lat < 50.0
    assert min_lon < max_lon and min_lat < max_lat
    # Fold is deterministic and in range.
    assert 0 <= plan.fold < builder.N_SPATIAL_FOLDS
    again = select_dense_patches(gdf, n_patches=1, min_parcels=30)[0]
    assert again.fold == plan.fold
    assert again.bbox_4326 == plan.bbox_4326


# --------------------------------------------------------------------------- #
# Step 4 -- rasterisation onto the exact image grid.
# --------------------------------------------------------------------------- #
def _plan_at_origin() -> PatchPlan:
    return PatchPlan(
        patch_id=0,
        bbox_3035=(0.0, 0.0, 1280.0, 1280.0),
        bbox_4326=(0.0, 0.0, 0.01, 0.01),
        n_parcels=2,
        fold=0,
        classes_present=(1, 2),
    )


def test_rasterize_patch_mask_burns_class_ids_with_background_fill() -> None:
    """Two toy parcels burn their class ids; everything else stays background 0."""
    stack = _toy_patch_stack()
    # Parcel of class 1 near the top-left (world coords map to small rows/cols),
    # parcel of class 5 a bit to the right. Grid: row=top at y=1280, 10 m/px.
    gdf = gpd.GeoDataFrame(
        {"class_id": [1, 5]},
        geometry=[
            _square(200.0, 1080.0, 200.0),  # ~cols 10-30, rows 10-30
            _square(800.0, 1080.0, 200.0),  # ~cols 70-90, rows 10-30
        ],
        crs="EPSG:3035",
    )
    mask = rasterize_patch_mask(gdf, stack)

    assert mask.shape == (PATCH_PX, PATCH_PX)
    assert mask.dtype == np.int32
    present = set(np.unique(mask).tolist())
    assert present == {0, 1, 5}  # background + the two burned classes
    assert int((mask == 0).sum()) > 0  # background present (fill=0)
    assert int((mask == 1).sum()) > 0 and int((mask == 5).sum()) > 0


def test_rasterize_patch_mask_empty_when_no_parcel_in_window() -> None:
    """Parcels entirely outside the 128 px window leave an all-background mask."""
    stack = _toy_patch_stack()
    far = gpd.GeoDataFrame(
        {"class_id": [3]},
        geometry=[_square(50_000.0, 50_000.0, 100.0)],  # far outside the 1.28 km tile
        crs="EPSG:3035",
    )
    mask = rasterize_patch_mask(far, stack)
    assert mask.shape == (PATCH_PX, PATCH_PX)
    assert int(mask.sum()) == 0  # nothing burned


# --------------------------------------------------------------------------- #
# Step 5 -- PASTIS-format persistence + stats.
# --------------------------------------------------------------------------- #
def test_save_pastis_format_writes_shapes_dtypes_and_scaling(tmp_path: Path) -> None:
    """The three artefacts match PASTIS shapes/dtypes; reflectance scaled to DN."""
    plan = _plan_at_origin()
    stack = _toy_patch_stack(n_frames=4, fill=0.2)
    mask = np.zeros((PATCH_PX, PATCH_PX), dtype=np.int32)
    mask[10:40, 10:40] = 1
    mask[50:80, 50:80] = 2

    result = save_pastis_format(
        tmp_path, plan, stack, mask, date_from=_SEASON[0], date_to=_SEASON[1]
    )

    s2 = np.load(tmp_path / "DATA_S2" / "S2_0.npy")
    tgt = np.load(tmp_path / "ANNOTATIONS" / "TARGET_0.npy")
    dts = np.load(tmp_path / "ANNOTATIONS" / "dates_0.npy")
    assert s2.shape == (4, 10, PATCH_PX, PATCH_PX) and s2.dtype == np.int16
    assert tgt.shape == (PATCH_PX, PATCH_PX) and tgt.dtype == np.int32
    assert dts.shape == (4,) and dts.dtype == np.int32
    # DN scaling: reflectance ~0.2 -> ~2000 DN (x 10000).
    assert 1500 < int(np.median(s2)) < 2500
    # DOY ascending and within a year.
    assert list(dts) == sorted(dts.tolist())
    assert dts.min() >= 1 and dts.max() <= 366
    # Result stats reflect the mask (coverage = fraction non-background).
    assert result.ok is True
    assert result.n_classes_present == 2
    assert result.class_support == {1: 900, 2: 900}
    assert result.coverage == pytest.approx(1800 / mask.size)


def test_ndvi_std_is_nonzero_for_textured_stack(tmp_path: Path) -> None:
    """A spatially varying stack yields a positive NDVI std (texture proxy)."""
    plan = _plan_at_origin()
    # Strong spatial gradient in B04/B08 so NDVI has real spatial variance.
    rng = np.random.default_rng(1)
    stack = np.zeros((2, 10, PATCH_PX, PATCH_PX), dtype=np.float32)
    grad = np.linspace(0.05, 0.5, PATCH_PX, dtype=np.float32)[None, :]
    stack[:, PASTIS_BANDS.index("B04")] = grad + rng.uniform(0, 0.01, (PATCH_PX, PATCH_PX))
    stack[:, PASTIS_BANDS.index("B08")] = grad[:, ::-1] + 0.1
    ps = _PatchStack(stack=stack, transform=plan_transform(), crs="EPSG:3035", residual_cloud=0.0)
    mask = np.ones((PATCH_PX, PATCH_PX), dtype=np.int32)
    result = save_pastis_format(
        tmp_path, plan, ps, mask, date_from=_SEASON[0], date_to=_SEASON[1]
    )
    assert result.ndvi_std > 0.0


def plan_transform() -> Affine:
    """Shared toy transform (origin grid, 10 m/px, y down)."""
    return Affine(10.0, 0.0, 0.0, 0.0, -10.0, PATCH_PX * 10.0)


# --------------------------------------------------------------------------- #
# Resume + metadata + class mapping doc.
# --------------------------------------------------------------------------- #
def test_resume_helpers_skip_written_patch_and_recompute_stats(tmp_path: Path) -> None:
    """patch_artifacts_exist flags a written patch; stats reload from disk."""
    plan = _plan_at_origin()
    stack = _toy_patch_stack(n_frames=3, fill=0.15)
    mask = np.zeros((PATCH_PX, PATCH_PX), dtype=np.int32)
    mask[20:60, 20:60] = 2  # 1600 px of class 2

    assert patch_artifacts_exist(tmp_path, plan.patch_id) is False
    save_pastis_format(tmp_path, plan, stack, mask, date_from=_SEASON[0], date_to=_SEASON[1])
    assert patch_artifacts_exist(tmp_path, plan.patch_id) is True

    reloaded = load_patch_result_stats(tmp_path, plan)
    assert reloaded.ok is True
    assert reloaded.requests == 0  # a resumed patch issues no new request
    assert reloaded.n_dates == 3
    assert reloaded.class_support == {2: 1600}
    assert reloaded.coverage == pytest.approx(1600 / mask.size)


def test_write_metadata_and_class_mapping_doc(tmp_path: Path) -> None:
    """metadata.parquet carries the fold + coverage; class_mapping.json the ids."""
    gdf = _make_parcels_gdf()
    gdf = gdf[~gdf.geometry.is_empty].reset_index(drop=True)
    gdf["hcat4_name"] = [
        {"A01": "maize", "A02": "vineyards", "A03": "rare_crop"}[c]
        for c in gdf["original_code"]
    ]
    table, _ = build_class_table(gdf, min_support=2)

    plan = _plan_at_origin()
    stack = _toy_patch_stack()
    mask = np.zeros((PATCH_PX, PATCH_PX), dtype=np.int32)
    mask[:64] = 1
    result = save_pastis_format(
        tmp_path, plan, stack, mask, date_from=_SEASON[0], date_to=_SEASON[1]
    )
    result.requests = 1

    meta_path = write_metadata(tmp_path, table, [(plan, result)])
    meta = pl.read_parquet(meta_path)
    assert meta.height == 1
    assert "fold_espacial" in meta.columns and "pct_cubierto" in meta.columns
    assert int(meta["fold_espacial"][0]) == plan.fold
    assert meta["pct_cubierto"][0] == pytest.approx(0.5)

    doc_path = write_class_mapping_doc(tmp_path, table)
    import json

    doc = json.loads(doc_path.read_text(encoding="utf-8"))
    assert doc["background_id"] == 0
    assert doc["other_class_name"] == "other"
    assert any(c["hcat4_name"] == "maize" for c in doc["classes"])
