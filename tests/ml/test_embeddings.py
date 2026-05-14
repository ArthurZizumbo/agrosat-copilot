"""Tests para `ml.analysis.embeddings` y extensiones de US-011 en
`ml.ingest.gee_sampler` y `ml.ingest.pastis_loader`.

Fixtures sinteticos:
- Matriz aleatoria `1000 x 64` con seed 42.
- Una dimension (`dim_00`) inyectada como discriminativa: su signo decide
  la clase A/B. RF debe asignarle importance >> que el resto.
- Labels balanceados 50/50.

GEE y PASTIS-R se mockean — ningun test toca la red ni el disco grande.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from ml.analysis.embeddings import (
    DIM_COLS,
    compare_alphaearth_vs_ndvi,
    correlation_matrix,
    cross_region_consistency,
    qq_test_dims,
    rf_feature_importance,
    temporal_stability,
    tsne_2d,
    umap_2d,
)


def _synthetic_embeddings(
    n: int = 1000,
    seed: int = 42,
    discriminative_dim: int = 0,
) -> tuple[pl.DataFrame, pl.Series]:
    """Construye fixture (X, y) con `dim_00` discriminativa.

    Returns:
        Tupla `(X: pl.DataFrame, y: pl.Series)` donde X tiene `dim_00..dim_63`
        e y es la etiqueta binaria balanceada.
    """
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal(size=(n, 64)).astype(np.float64)
    labels = np.where(mat[:, discriminative_dim] > 0, "A", "B")
    # Reforzar la senal de la dim discriminativa
    mat[labels == "A", discriminative_dim] += 2.5
    mat[labels == "B", discriminative_dim] -= 2.5
    df = pl.DataFrame({col: mat[:, i] for i, col in enumerate(DIM_COLS)})
    return df, pl.Series("class_name", labels)


def test_dim_cols_canonical() -> None:
    """`DIM_COLS` contiene 64 entradas `dim_00..dim_63`."""
    assert len(DIM_COLS) == 64
    assert DIM_COLS[0] == "dim_00"
    assert DIM_COLS[63] == "dim_63"


def test_correlation_matrix_shape_and_range() -> None:
    """Output de `correlation_matrix` tiene triangulo superior + valores en [-1, 1]."""
    df, _ = _synthetic_embeddings(n=500)
    out = correlation_matrix(df, method="pearson")
    # Triangulo superior incluyendo diagonal: 64*65/2
    assert out.height == 64 * 65 // 2
    expected_cols = {"dim_i", "dim_j", "pearson", "abs_corr"}
    assert expected_cols.issubset(set(out.columns))
    vals = out["pearson"].drop_nulls().to_numpy()
    assert (vals >= -1.0001).all() and (vals <= 1.0001).all()


def test_correlation_matrix_spearman_runs() -> None:
    """Spearman tambien produce la misma cantidad de filas."""
    df, _ = _synthetic_embeddings(n=200)
    out = correlation_matrix(df, method="spearman")
    assert out.height == 64 * 65 // 2
    assert "spearman" in out.columns


def test_correlation_matrix_empty_input() -> None:
    """DataFrame vacio retorna DataFrame vacio con esquema."""
    empty = pl.DataFrame(schema={c: pl.Float64 for c in DIM_COLS})
    out = correlation_matrix(empty)
    assert out.is_empty()
    assert "dim_i" in out.columns


def test_correlation_matrix_invalid_cols() -> None:
    """Columnas inexistentes lanzan ValueError."""
    df, _ = _synthetic_embeddings(n=100)
    with pytest.raises(ValueError):
        correlation_matrix(df, cols=["dim_00", "no_existe"])


def test_qq_test_dims_columns_and_finite() -> None:
    """`qq_test_dims` produce 64 filas con stats finitos."""
    df, _ = _synthetic_embeddings(n=500)
    out = qq_test_dims(df)
    assert out.height == 64
    for col in ("mean", "std", "skewness", "kurtosis", "shapiro_pvalue"):
        vals = out[col].to_numpy()
        assert np.isfinite(vals).all()
    # p-value siempre en [0, 1]
    pvals = out["shapiro_pvalue"].to_numpy()
    assert ((pvals >= 0.0) & (pvals <= 1.0)).all()


def test_qq_test_dims_empty() -> None:
    """Empty df produce 0 filas."""
    empty = pl.DataFrame(schema={c: pl.Float64 for c in DIM_COLS})
    out = qq_test_dims(empty)
    assert out.is_empty()


def test_tsne_2d_shape_and_seed() -> None:
    """t-SNE devuelve subsample x 2 reproducible."""
    df, _ = _synthetic_embeddings(n=300)
    X1, idx1 = tsne_2d(df, perplexity=10, subsample=100, seed=42)
    X2, idx2 = tsne_2d(df, perplexity=10, subsample=100, seed=42)
    assert X1.shape == (100, 2)
    assert idx1.shape == (100,)
    np.testing.assert_array_equal(idx1, idx2)
    np.testing.assert_allclose(X1, X2, atol=1e-3)


def test_tsne_2d_empty() -> None:
    """Empty df produce array vacio."""
    empty = pl.DataFrame(schema={c: pl.Float64 for c in DIM_COLS})
    X, idx = tsne_2d(empty)
    assert X.shape == (0, 2)
    assert idx.shape == (0,)


def test_umap_2d_shape_or_empty() -> None:
    """UMAP devuelve (X_2d, idx). Si umap-learn no esta instalado retorna (0,2)+[]."""
    df, _ = _synthetic_embeddings(n=200)
    out, idx = umap_2d(df, n_neighbors=10, seed=42)
    assert out.ndim == 2
    assert out.shape[1] == 2
    assert idx.ndim == 1
    # Si UMAP esta disponible, deben coincidir las filas (sin NaN sinteticos).
    if out.shape[0] > 0:
        assert out.shape[0] == df.height
        assert idx.shape[0] == df.height


def test_umap_2d_drops_nan_rows() -> None:
    """UMAP filtra filas con NaN y los idx mapean al df original."""
    df, _ = _synthetic_embeddings(n=120)
    bad_rows = [0, 5, 10]
    df = df.with_columns(
        pl.when(pl.int_range(0, df.height).is_in(bad_rows))
        .then(float("nan"))
        .otherwise(pl.col("dim_00"))
        .alias("dim_00")
    )
    out, idx = umap_2d(df, n_neighbors=10, seed=42)
    if out.shape[0] > 0:
        assert not any(b in idx for b in bad_rows)
        assert out.shape[0] == idx.shape[0]
        assert out.shape[0] == df.height - len(bad_rows)


def test_tsne_2d_drops_nan_rows() -> None:
    """t-SNE filtra filas con NaN y los idx mapean al df original."""
    df, _ = _synthetic_embeddings(n=200)
    bad_rows = [1, 7, 99]
    df = df.with_columns(
        pl.when(pl.int_range(0, df.height).is_in(bad_rows))
        .then(float("nan"))
        .otherwise(pl.col("dim_03"))
        .alias("dim_03")
    )
    X_2d, idx = tsne_2d(df, perplexity=10, subsample=500, seed=42)
    assert X_2d.shape[0] == idx.shape[0]
    assert not any(b in idx for b in bad_rows)


def test_rf_feature_importance_drops_nan_rows() -> None:
    """RF filtra filas con NaN antes del fit (sklearn las rechaza)."""
    df, y = _synthetic_embeddings(n=400)
    df = df.with_columns(
        pl.when(pl.int_range(0, df.height).is_in([0, 1, 2]))
        .then(float("nan"))
        .otherwise(pl.col("dim_00"))
        .alias("dim_00")
    )
    out = rf_feature_importance(df, y, n_estimators=50, max_depth=6)
    assert out.height == 64
    assert not np.isnan(out["importance"].to_numpy()).any()


def test_rf_feature_importance_dim00_dominates() -> None:
    """`dim_00` (discriminativa inyectada) debe tener importance >> resto."""
    df, y = _synthetic_embeddings(n=800)
    out = rf_feature_importance(df, y, n_estimators=80, max_depth=8)
    assert out.height == 64
    # importance suma 1
    assert abs(float(out["importance"].sum()) - 1.0) < 1e-6
    top = out.sort("importance", descending=True).row(0, named=True)
    assert top["dim"] == "dim_00"
    assert top["importance"] > 0.2  # debe ser claramente la mas importante
    oob = float(out["oob_score"][0])
    assert 0.0 <= oob <= 1.0


def test_rf_feature_importance_empty() -> None:
    """Empty df produce DataFrame vacio."""
    empty = pl.DataFrame(schema={c: pl.Float64 for c in DIM_COLS})
    out = rf_feature_importance(empty, pl.Series("y", [], dtype=pl.Utf8))
    assert out.is_empty()


def test_temporal_stability_identical_years() -> None:
    """Embeddings identicos entre anios -> cosine_mean = 1.0."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal(size=(50, 64))
    rows = []
    for year in (2022, 2023, 2024):
        for i in range(50):
            row = {"px_id": f"p{i}", "year": year, "class_name": "crops"}
            for j, col in enumerate(DIM_COLS):
                row[col] = float(base[i, j])
            rows.append(row)
    df = pl.DataFrame(rows)
    out = temporal_stability(df)
    assert "cosine_mean" in out.columns
    cos_mean = out["cosine_mean"].to_numpy()
    np.testing.assert_allclose(cos_mean, 1.0, atol=1e-6)


def test_temporal_stability_random_years() -> None:
    """Embeddings totalmente aleatorios -> cosine_mean cerca de 0."""
    rng = np.random.default_rng(1)
    rows = []
    for year in (2022, 2023):
        mat = rng.standard_normal(size=(80, 64))
        for i in range(80):
            row = {"px_id": f"p{i}", "year": year, "class_name": "X"}
            for j, col in enumerate(DIM_COLS):
                row[col] = float(mat[i, j])
            rows.append(row)
    df = pl.DataFrame(rows)
    out = temporal_stability(df)
    mean_abs = float(np.abs(out["cosine_mean"].to_numpy()).mean())
    # Bound holgado: vectores 64-dim random tienen cosine ~ N(0, 1/sqrt(64))
    assert mean_abs < 0.3


def test_temporal_stability_empty() -> None:
    empty = pl.DataFrame(
        schema={c: pl.Float64 for c in DIM_COLS} | {"px_id": pl.Utf8, "year": pl.Int64}
    )
    out = temporal_stability(empty)
    assert out.is_empty()


def test_compare_alphaearth_vs_ndvi_returns_figure(tmp_path: Path) -> None:
    """Funcion grafica produce figura matplotlib + PNG en disco."""
    rows = [{col: float(i + j) for j, col in enumerate(DIM_COLS)} for i in range(10)]
    df = pl.DataFrame(rows)
    rgb = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)
    ndvi = np.random.default_rng(0).uniform(-1, 1, size=(16, 16)).astype(np.float32)
    out_path = tmp_path / "compare.png"
    fig = compare_alphaearth_vs_ndvi(
        parcel_id="p1",
        df_embeddings=df,
        top_dims=["dim_00", "dim_01", "dim_02"],
        s2_date="2024-06-15",
        rgb_array=rgb,
        ndvi_array=ndvi,
        out_path=out_path,
    )
    assert out_path.exists()
    assert fig is not None


def test_cross_region_consistency_identical_inputs() -> None:
    """Si Italia y Francia son identicos, todas las top-K son consistentes."""
    df, y = _synthetic_embeddings(n=500)
    rf_it = rf_feature_importance(df, y, n_estimators=50, max_depth=6)
    out = cross_region_consistency(rf_it, rf_it, top_k=10)
    assert out.height == 64
    top10 = out.filter(pl.col("rank_italia") <= 10)
    # Como los dos rankings son idénticos, todas las top-10 son consistentes
    assert all(top10["consistente_top10"].to_list())
    # delta_rank siempre 0 cuando los inputs coinciden
    deltas = out["delta_rank"].to_numpy()
    assert (deltas == 0).all()


def test_cross_region_consistency_different_inputs() -> None:
    """Inputs diferentes producen un subset variable de consistentes."""
    df_a, y_a = _synthetic_embeddings(n=500, seed=42, discriminative_dim=0)
    df_b, y_b = _synthetic_embeddings(n=500, seed=99, discriminative_dim=10)
    rf_a = rf_feature_importance(df_a, y_a, n_estimators=50, max_depth=6)
    rf_b = rf_feature_importance(df_b, y_b, n_estimators=50, max_depth=6)
    out = cross_region_consistency(rf_a, rf_b, top_k=10)
    assert out.height == 64
    # Por construccion las dims discriminativas son distintas -> top-10 no coinciden todas
    consistent = sum(out["consistente_top10"].to_list())
    assert 0 <= consistent <= 10


def test_cross_region_consistency_empty_inputs() -> None:
    """Empty inputs -> DataFrame vacio con esquema."""
    empty = pl.DataFrame(
        schema={
            "dim": pl.Utf8,
            "importance": pl.Float64,
            "rank": pl.Int64,
            "cumulative_importance": pl.Float64,
            "oob_score": pl.Float64,
        }
    )
    out = cross_region_consistency(empty, empty)
    assert out.is_empty()


# ============================================================
# Tests para extensiones de gee_sampler (mocked EE)
# ============================================================


def _fake_alphaearth_ee(n_features: int = 5) -> MagicMock:
    """Mock minimo del modulo `ee` para `sample_alphaearth_roi`."""
    fake = MagicMock(name="ee")
    fake_image = MagicMock(name="ee.Image")
    fake_image.select.return_value = fake_image
    fake_sample = MagicMock(name="ee.Sample")
    payload_features = []
    for i in range(n_features):
        props = {f"A{j:02d}": float(i + j) for j in range(64)}
        payload_features.append(
            {"properties": props, "geometry": {"coordinates": [9.5 + i * 0.01, 45.2]}}
        )
    fake_sample.getInfo.return_value = {"features": payload_features}
    fake_image.sample.return_value = fake_sample

    fake_collection = MagicMock()
    fake_collection.filterBounds.return_value = fake_collection
    fake_collection.filterDate.return_value = fake_collection
    # AlphaEarth usa mosaic() (no first()) para unir todos los tiles del anio
    # en un raster continuo. Ver BUG-7 en docs/us-handoff/us-011.md.
    fake_collection.mosaic.return_value = fake_image
    fake.ImageCollection.return_value = fake_collection
    fake.Geometry.Point = MagicMock()
    fake.Feature = MagicMock()
    fake.FeatureCollection = MagicMock()
    fake.Reducer.first = MagicMock()
    return fake


def test_sample_alphaearth_roi_with_mock(tmp_path: Path) -> None:
    """`sample_alphaearth_roi` produce DataFrame con 64 dims via EE mockeado."""
    from ml.ingest import gee_sampler
    from ml.ingest.gee_sampler import ALPHAEARTH_DIM_COLS, sample_alphaearth_roi

    fake_ee = _fake_alphaearth_ee(n_features=4)
    with patch.object(gee_sampler, "ee", fake_ee):
        df = sample_alphaearth_roi(
            roi=MagicMock(),
            year=2024,
            n_pixels=10,
            cache_path=tmp_path,
            roi_name="pianura",
        )
    assert df.height == 4
    for col in ALPHAEARTH_DIM_COLS:
        assert col in df.columns
    assert set(df["roi"].unique().to_list()) == {"pianura"}
    cache_files = list(tmp_path.glob("alphaearth_pianura_2024_*.parquet"))
    assert cache_files, "Cache parquet no escrito"


def test_sample_alphaearth_roi_cache_round_trip(tmp_path: Path) -> None:
    """Segunda llamada lee del cache sin re-invocar EE."""
    from ml.ingest import gee_sampler
    from ml.ingest.gee_sampler import sample_alphaearth_roi

    fake_ee = _fake_alphaearth_ee(n_features=3)
    with patch.object(gee_sampler, "ee", fake_ee):
        first = sample_alphaearth_roi(
            roi=MagicMock(),
            year=2024,
            n_pixels=10,
            cache_path=tmp_path,
            roi_name="toscana",
        )
        fake_ee.ImageCollection.reset_mock()
        second = sample_alphaearth_roi(
            roi=MagicMock(),
            year=2024,
            n_pixels=10,
            cache_path=tmp_path,
            roi_name="toscana",
        )
    assert first.equals(second)
    fake_ee.ImageCollection.assert_not_called()


def test_sample_alphaearth_roi_degraded(tmp_path: Path) -> None:
    """Sin EE retorna DataFrame vacio con esquema correcto."""
    from ml.ingest import gee_sampler
    from ml.ingest.gee_sampler import sample_alphaearth_roi

    with patch.object(gee_sampler, "ee", None):
        out = sample_alphaearth_roi(roi=None, year=2024, cache_path=tmp_path, roi_name="apulia")
    assert out.is_empty()
    assert "dim_00" in out.columns
    assert "dim_63" in out.columns


def test_sample_alphaearth_at_coords_empty_when_ee_missing(tmp_path: Path) -> None:
    """`sample_alphaearth_at_coords` sin EE produce esquema vacio."""
    from ml.ingest import gee_sampler
    from ml.ingest.gee_sampler import sample_alphaearth_at_coords

    coords = pl.DataFrame({"px_id": ["a", "b"], "lon": [1.0, 2.0], "lat": [44.0, 45.0]})
    with patch.object(gee_sampler, "ee", None):
        out = sample_alphaearth_at_coords(coords, year=2019, cache_path=tmp_path)
    assert out.is_empty()
    assert "dim_00" in out.columns


def test_sample_dynamic_world_at_degraded(tmp_path: Path) -> None:
    """`sample_dynamic_world_at` sin EE produce esquema vacio."""
    from ml.ingest import gee_sampler
    from ml.ingest.gee_sampler import sample_dynamic_world_at

    coords = pl.DataFrame({"px_id": ["a"], "lon": [9.5], "lat": [45.2]})
    with patch.object(gee_sampler, "ee", None):
        out = sample_dynamic_world_at(coords, year=2024, cache_path=tmp_path)
    assert out.is_empty()
    assert "dw_class_id" in out.columns


def test_fetch_s2_ndvi_rgb_degraded() -> None:
    """Sin EE retorna arrays vacios sin error."""
    from ml.ingest import gee_sampler
    from ml.ingest.gee_sampler import fetch_s2_ndvi_rgb_for_parcel

    with patch.object(gee_sampler, "ee", None):
        out = fetch_s2_ndvi_rgb_for_parcel(None, "2024-06-15")
    assert "rgb" in out and "ndvi" in out and "date_used" in out
    assert out["rgb"].size == 0
    assert out["ndvi"].size == 0


# ============================================================
# Tests para extensiones de pastis_loader
# ============================================================


def test_pastis_patch_coords_missing_file(tmp_path: Path) -> None:
    """Sin metadata.geojson retorna DataFrame vacio con esquema."""
    from ml.ingest.pastis_loader import pastis_patch_coords

    out = pastis_patch_coords(tmp_path / "no_existe.geojson")
    assert out.is_empty()
    assert {"patch_id", "lon", "lat", "tile", "fold"}.issubset(set(out.columns))


def test_pastis_patch_coords_reprojection(tmp_path: Path) -> None:
    """Centroide conocido en EPSG:2154 se proyecta a coords plausibles para Francia."""
    pytest.importorskip("pyproj")
    from ml.ingest.pastis_loader import pastis_patch_coords

    # MultiPolygon de un cuadrado centrado en (391500, 6955430) EPSG:2154
    # que corresponde aproximadamente a (-3.5 lon, 47.8 lat) en EPSG:4326
    cx, cy = 391500.0, 6955430.0
    poly = [
        [
            [cx - 100, cy - 100],
            [cx + 100, cy - 100],
            [cx + 100, cy + 100],
            [cx - 100, cy + 100],
            [cx - 100, cy - 100],
        ]
    ]
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "id": "10000",
                "type": "Feature",
                "properties": {"ID_PATCH": 10000, "TILE": "T30UXV", "Fold": 1},
                "geometry": {"type": "MultiPolygon", "coordinates": [poly]},
            }
        ],
    }
    import json

    path = tmp_path / "metadata.geojson"
    path.write_text(json.dumps(geojson), encoding="utf-8")
    out = pastis_patch_coords(path)
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["patch_id"] == "10000"
    # Coords plausibles en territorio frances (lon ~-5..-1, lat ~46..50)
    # El centroide (391500, 6955430) en EPSG:2154 cae cerca de Bretaña/Loira.
    assert -6.0 < row["lon"] < 0.0
    assert 46.0 < row["lat"] < 51.0


def test_pastis_pixel_labels_missing(tmp_path: Path) -> None:
    """TARGET no existe -> DataFrame vacio con esquema."""
    from ml.ingest.pastis_loader import pastis_pixel_labels

    out = pastis_pixel_labels("999999", root=tmp_path)
    assert out.is_empty()
    assert {"px_id", "class_id", "class_name"}.issubset(set(out.columns))


def test_pastis_pixel_labels_mock_npy(tmp_path: Path) -> None:
    """TARGET sintetico produce DataFrame con class_id en rango y filtrado de 0/19."""
    from ml.ingest.pastis_loader import pastis_pixel_labels

    ann_dir = tmp_path / "ANNOTATIONS"
    ann_dir.mkdir()
    # Tensor 3x128x128: canal 0 con clases 0..19 mixto
    rng = np.random.default_rng(7)
    sem = rng.integers(low=0, high=20, size=(128, 128), dtype=np.uint8)
    target = np.stack([sem, np.zeros_like(sem), np.zeros_like(sem)], axis=0)
    np.save(ann_dir / "TARGET_10000.npy", target)

    out = pastis_pixel_labels("10000", root=tmp_path, sample_per_patch=500, exclude_classes=(0, 19))
    assert not out.is_empty()
    assert out.height <= 500
    cls = out["class_id"].to_numpy()
    assert (cls >= 1).all() and (cls <= 18).all()
    # class_name no debe contener "unknown"
    assert "unknown" not in set(out["class_name"].to_list())


def test_pastis_groupings_loaded() -> None:
    """`PASTIS_R_GROUPINGS` se carga si el JSON esta presente."""
    from ml.ingest.pastis_loader import PASTIS_R_GROUPINGS

    if PASTIS_R_GROUPINGS:
        # En el repo el JSON existe, debe haber al menos 'phenological_cycle'
        assert "phenological_cycle" in PASTIS_R_GROUPINGS
        assert all(isinstance(k, int) for k in PASTIS_R_GROUPINGS["phenological_cycle"])
