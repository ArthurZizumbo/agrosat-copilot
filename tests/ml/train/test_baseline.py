"""Tests smoke de ml.train.baseline (US-019).

Conjunto minimo de validacion de la libreria del baseline tabular. La
suite exhaustiva (~22 tests, grupos A-E) la completa el sub-agente de
tests. Todos los tests core usan el fixture sintetico determinista, sin
depender del parquet PASTIS-R de 76 MB.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ml.train.baseline import (
    BaselineResult,
    build_estimator,
    evaluate_with_spatial_cv,
    train_one_model,
    tune_baseline,
)
from tests.ml.train.fixtures.baseline_synthetic import make_baseline_dataset

_METRIC_KEYS = {"f1_macro", "f1_weighted", "miou", "accuracy", "cohen_kappa"}


@pytest.fixture(scope="module")
def synthetic_df() -> pl.DataFrame:
    """DataFrame sintetico determinista compartido por el modulo."""
    return make_baseline_dataset(n=300, n_classes=4, n_features=10, n_patches=14, seed=42)


def test_train_rf_returns_baseline_result(synthetic_df: pl.DataFrame) -> None:
    """train_one_model('rf') devuelve un BaselineResult completo."""
    result = train_one_model(synthetic_df, model="rf")
    assert isinstance(result, BaselineResult)
    assert result.model_kind == "rf"
    assert set(result.metrics.keys()) == _METRIC_KEYS
    assert len(result.feature_cols) == 10


def test_train_xgb_returns_baseline_result(synthetic_df: pl.DataFrame) -> None:
    """train_one_model('xgb') maneja class_ids no contiguos via LabelEncoder."""
    result = train_one_model(synthetic_df, model="xgb")
    assert result.model_kind == "xgb"
    # Class ids del fixture no son contiguos (saltan 0); el encoder los mapea.
    assert result.label_classes == tuple(sorted(result.label_classes))
    assert min(result.label_classes) >= 1


def test_baseline_result_feature_cols_excludes_meta(
    synthetic_df: pl.DataFrame,
) -> None:
    """Las columnas de metadata no entran como features."""
    result = train_one_model(synthetic_df, model="rf")
    for meta_col in ("parcel_id", "class_id", "fold", "patch_id", "n_pixels"):
        assert meta_col not in result.feature_cols


def test_cv_metrics_have_mean_and_std(synthetic_df: pl.DataFrame) -> None:
    """cv_metrics expone (media, std) por metrica."""
    result = train_one_model(synthetic_df, model="rf")
    for key in _METRIC_KEYS:
        assert key in result.cv_metrics
        mean, std = result.cv_metrics[key]
        assert isinstance(mean, float)
        assert isinstance(std, float)


def test_evaluate_returns_oof_predictions(synthetic_df: pl.DataFrame) -> None:
    """evaluate_with_spatial_cv devuelve metricas CV y predicciones OOF."""
    from ml.train.baseline import _prepare_dataframe

    clean = _prepare_dataframe(synthetic_df)

    def factory():  # type: ignore[no-untyped-def]
        return build_estimator("rf", {"n_estimators": 50, "random_state": 42})

    cv_metrics, y_true, y_pred = evaluate_with_spatial_cv(clean, factory)
    assert set(cv_metrics.keys()) == _METRIC_KEYS
    assert y_true.shape == y_pred.shape
    assert y_true.size > 0


def test_tune_returns_best_params(synthetic_df: pl.DataFrame) -> None:
    """tune_baseline devuelve un diccionario de best_params."""
    best = tune_baseline(
        synthetic_df,
        model="rf",
        param_grid={"n_estimators": [50, 100], "max_depth": [5, None]},
    )
    assert isinstance(best, dict)
    assert "n_estimators" in best


def test_grid_combos_within_budget() -> None:
    """Las grillas por defecto no exceden 8 combinaciones (criterio AC-4)."""
    from ml.train.baseline import _RF_PARAM_GRID, _XGB_PARAM_GRID

    for grid in (_RF_PARAM_GRID, _XGB_PARAM_GRID):
        combos = int(np.prod([len(v) for v in grid.values()]))
        assert combos <= 8


def test_train_deterministic_with_seed(synthetic_df: pl.DataFrame) -> None:
    """Dos entrenamientos con la misma semilla dan el mismo F1-macro."""
    a = train_one_model(synthetic_df, model="rf", random_state=7)
    b = train_one_model(synthetic_df, model="rf", random_state=7)
    assert a.metrics["f1_macro"] == pytest.approx(b.metrics["f1_macro"])
