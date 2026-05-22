"""Tests del entrenamiento del baseline ``ml.train.train_baseline`` (Avance 3).

Se usa un fixture sintetico separable (clases con centros distintos en el
espacio de features) para que los modelos de arboles superen claramente al
azar y las aserciones sean deterministas.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import polars as pl
import pytest

from ml.train.train_baseline import (
    BaselineConfig,
    XGBLabelSafeClassifier,
    build_alphaearth_pastis_table,
    filter_rare_classes,
    spatial_cv_evaluate,
    train_baselines,
)


def _make_separable_dataset(
    n_per_class: int = 80,
    n_classes: int = 4,
    n_features: int = 16,
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.Series, np.ndarray]:
    """Crea un dataset sintetico separable con folds round-robin.

    Args:
        n_per_class: Muestras por clase.
        n_classes: Numero de clases.
        n_features: Dimension del vector de features.
        n_folds: Numero de folds espaciales.
        seed: Semilla.

    Returns:
        Tupla ``(X, y, folds)`` lista para los entrenadores del baseline.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 4.0, size=(n_classes, n_features))
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for cls in range(n_classes):
        block = centers[cls] + rng.normal(0.0, 1.0, size=(n_per_class, n_features))
        rows.append(block)
        labels.extend([cls] * n_per_class)
    matrix = np.vstack(rows)
    y_arr = np.asarray(labels, dtype=np.int64)
    perm = rng.permutation(len(y_arr))
    matrix = matrix[perm]
    y_arr = y_arr[perm]
    folds = (np.arange(len(y_arr)) % n_folds + 1).astype(np.int64)
    X = pl.DataFrame(
        {f"dim_{j:02d}": matrix[:, j].tolist() for j in range(n_features)}
    )
    return X, pl.Series("class_id", y_arr.tolist()), folds


def test_dummy_is_beaten_by_random_forest() -> None:
    """El RandomForest supera con holgura al DummyClassifier en datos separables."""
    X, y, folds = _make_separable_dataset()
    config = BaselineConfig(grid_search=False)
    results = train_baselines(
        X, y, folds, config=config, models=("dummy", "random_forest"), log_mlflow=False
    )
    assert results["random_forest"].f1_macro() > results["dummy"].f1_macro() + 0.3


def test_spatial_cv_uses_all_folds() -> None:
    """``spatial_cv_evaluate`` produce una metrica por cada fold presente."""
    X, y, folds = _make_separable_dataset(n_folds=5)
    config = BaselineConfig(grid_search=False)
    result = spatial_cv_evaluate("random_forest", X, y, folds, config)
    assert len(result.fold_metrics) == 5
    assert result.oof_true.size == X.height


def test_xgboost_handles_non_contiguous_labels() -> None:
    """El wrapper XGBoost entrena con etiquetas no contiguas sin fallar."""
    X, y, folds = _make_separable_dataset(n_classes=4)
    # Re-mapeo a etiquetas no contiguas: 0,1,2,3 -> 10,25,40,77.
    remap = {0: 10, 1: 25, 2: 40, 3: 77}
    y_sparse = pl.Series("class_id", [remap[v] for v in y.to_list()])
    config = BaselineConfig(grid_search=False)
    result = spatial_cv_evaluate("xgboost", X, y_sparse, folds, config)
    assert result.f1_macro() > 0.5
    # Las predicciones viven en el espacio de etiquetas original.
    assert set(np.unique(result.oof_pred).tolist()).issubset(set(remap.values()))


def test_xgb_wrapper_predict_returns_original_labels() -> None:
    """``XGBLabelSafeClassifier`` revierte el re-mapeo en ``predict``."""
    X, y, _ = _make_separable_dataset(n_classes=3)
    matrix = X.to_numpy()
    y_arr = np.array([v * 100 + 5 for v in y.to_list()], dtype=np.int64)
    clf = XGBLabelSafeClassifier(n_estimators=30, random_state=42)
    clf.fit(matrix, y_arr)
    preds = clf.predict(matrix)
    assert set(np.unique(preds).tolist()).issubset(set(np.unique(y_arr).tolist()))


def test_xgb_wrapper_exposes_sklearn_classifier_api() -> None:
    """El wrapper expone ``classes_`` / ``predict_proba`` que GridSearchCV exige.

    Sin ``classes_``, ``sklearn.utils._response._get_response_values`` falla y
    cada candidato del grid cae en ``error_score`` -> ``best_score`` 0.0.
    """
    from sklearn.model_selection import GridSearchCV

    X, y, folds = _make_separable_dataset(n_classes=3)
    matrix = X.to_numpy()
    y_arr = np.array([v * 10 + 7 for v in y.to_list()], dtype=np.int64)  # no contiguas
    clf = XGBLabelSafeClassifier(n_estimators=30, random_state=42).fit(matrix, y_arr)
    np.testing.assert_array_equal(clf.classes_, np.unique(y_arr))
    assert clf.n_features_in_ == matrix.shape[1]
    assert clf.predict_proba(matrix).shape == (matrix.shape[0], 3)

    splits = [
        (np.where(folds != k)[0], np.where(folds == k)[0])
        for k in sorted(set(folds.tolist()))
    ]
    search = GridSearchCV(
        XGBLabelSafeClassifier(n_estimators=30, random_state=42),
        {"max_depth": [3, 6]},
        scoring="f1_macro",
        cv=splits,
        refit=False,
        error_score=0.0,
    )
    search.fit(matrix, y_arr)
    # best_score > 0 prueba que el scoring corrio (no cayo en error_score).
    assert float(search.best_score_) > 0.5


def test_grid_search_selects_params() -> None:
    """Con ``grid_search=True`` el RandomForest recibe hiperparametros elegidos."""
    X, y, folds = _make_separable_dataset()
    config = BaselineConfig(grid_search=True)
    results = train_baselines(
        X, y, folds, config=config, models=("random_forest",), log_mlflow=False
    )
    best = results["random_forest"].best_params
    assert "n_estimators" in best
    assert "max_depth" in best


def test_filter_rare_classes_drops_low_support() -> None:
    """``filter_rare_classes`` descarta las clases por debajo del soporte minimo."""
    df = pl.DataFrame(
        {
            "class_id": [1] * 50 + [2] * 50 + [3] * 5,
            "dim_00": list(range(105)),
        }
    )
    filtered, dropped = filter_rare_classes(df, min_count=30)
    assert dropped == [3]
    assert filtered.height == 100
    assert 3 not in filtered.get_column("class_id").to_list()


def test_build_table_falls_back_to_synthetic(tmp_path) -> None:
    """Si el cache AlphaEarth no existe, se genera un fixture sintetico valido."""
    missing_ae = tmp_path / "no_ae.parquet"
    missing_pastis = tmp_path / "no_pastis"
    df, mode = build_alphaearth_pastis_table(
        missing_ae, missing_pastis, n_folds=5, min_class_count=20
    )
    assert mode == "synthetic"
    assert df.height > 0
    assert "fold" in df.columns
    assert "class_id" in df.columns
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    assert len(dim_cols) == 64


def test_synthetic_table_is_separable() -> None:
    """El fixture sintetico del baseline es separable (RF supera al azar)."""
    df, mode = build_alphaearth_pastis_table(
        os.devnull and __import__("pathlib").Path("missing.parquet"),
        __import__("pathlib").Path("missing_dir"),
        min_class_count=20,
    )
    assert mode == "synthetic"
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    X = df.select(dim_cols)
    y = df.get_column("class_id").cast(pl.Int64)
    folds = df.get_column("fold").to_numpy()
    config = BaselineConfig(grid_search=False)
    result = spatial_cv_evaluate("random_forest", X, y, folds, config)
    assert result.f1_macro() > 0.5


def test_baseline_result_f1_macro_default_zero() -> None:
    """Un ``BaselineResult`` sin metricas devuelve F1-macro 0.0."""
    from ml.train.train_baseline import BaselineResult

    res = BaselineResult(model_name="dummy")
    assert res.f1_macro() == 0.0


def test_load_dataset_raises_on_missing_file(tmp_path) -> None:
    """Cargar un parquet inexistente lanza ``FileNotFoundError``."""
    from ml.train.train_baseline import load_tabular_dataset

    with pytest.raises(FileNotFoundError):
        load_tabular_dataset(tmp_path / "nope.parquet", BaselineConfig())
