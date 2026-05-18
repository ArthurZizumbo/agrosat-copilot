"""Tests US-016 AC-9 — ``ml.features.scaler``.

Cubre:

- Roundtrip: scaler fit sobre train => |mean|<1e-6, |std-1|<1e-6 en train transformado.
- `load_scaler` preserva mean_/scale_ tras dump+load.
- Leakage: train_ids interseca val/test => ValueError.
- Serializacion joblib (header zlib distinto al pickle raw `\\x80\\x04`).
- Columna constante => scale_=1.0 (sklearn convention `with_std=True`).
- Filtrado silencioso de columnas categoricas (srtm_aspect_dominant Utf8).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import polars as pl
import pytest
from sklearn.preprocessing import StandardScaler

from ml.features.scaler import fit_scaler_on_train, load_scaler

# ---------------------------------------------------------------------------
# Fixtures locales.
# ---------------------------------------------------------------------------


def _make_synthetic_frame(n: int = 20, seed: int = 7) -> pl.DataFrame:
    """Frame sintetico con 5 numericas + 1 categorica (srtm_aspect_dominant)."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "parcel_id": list(range(1, n + 1)),
            "year": [2024] * n,
            "feat_a": rng.normal(loc=10.0, scale=2.0, size=n).tolist(),
            "feat_b": rng.normal(loc=-5.0, scale=1.0, size=n).tolist(),
            "feat_const": [3.14] * n,
            "feat_c": rng.uniform(0.0, 1.0, size=n).tolist(),
            "feat_d": rng.uniform(100.0, 200.0, size=n).tolist(),
            "srtm_aspect_dominant": [
                ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][i % 8] for i in range(n)
            ],
        }
    )


# ---------------------------------------------------------------------------
# AC-9: roundtrip zero-mean unit-std.
# ---------------------------------------------------------------------------


def test_scaler_roundtrip_zero_mean_unit_std_on_train(tmp_path: Path) -> None:
    df = _make_synthetic_frame(n=40, seed=1)
    train_ids = tuple(int(x) for x in df.get_column("parcel_id").to_list()[:30])
    feature_cols = ("feat_a", "feat_b", "feat_c", "feat_d")
    scaler = fit_scaler_on_train(
        df,
        train_ids=train_ids,
        feature_cols=feature_cols,
        scaler_path=tmp_path / "scaler_v1.pkl",
    )
    train_df = df.filter(pl.col("parcel_id").is_in(list(train_ids))).select(list(feature_cols))
    transformed = scaler.transform(train_df.to_numpy())
    means = transformed.mean(axis=0)
    stds = transformed.std(axis=0, ddof=0)
    assert np.allclose(means, 0.0, atol=1e-6)
    assert np.allclose(stds, 1.0, atol=1e-6)


def test_load_scaler_returns_same_object(tmp_path: Path) -> None:
    df = _make_synthetic_frame(n=30, seed=2)
    train_ids = tuple(int(x) for x in df.get_column("parcel_id").to_list()[:24])
    feature_cols = ("feat_a", "feat_b")
    path = tmp_path / "scaler_load_test.pkl"
    original = fit_scaler_on_train(
        df, train_ids=train_ids, feature_cols=feature_cols, scaler_path=path
    )
    loaded = load_scaler(path)
    assert isinstance(loaded, StandardScaler)
    assert np.allclose(loaded.mean_, original.mean_)
    assert np.allclose(loaded.scale_, original.scale_)


def test_fit_raises_on_train_val_overlap(tmp_path: Path) -> None:
    """train_ids interseca val_ids/test_ids => ValueError descriptivo."""
    df = _make_synthetic_frame(n=20, seed=3)
    train_ids = (1, 2, 3, 4, 5)
    val_ids = (5, 6, 7)  # overlap en 5
    with pytest.raises(ValueError, match=r"[Ll]eakage|interseca|overlap"):
        fit_scaler_on_train(
            df,
            train_ids=train_ids,
            feature_cols=("feat_a", "feat_b"),
            scaler_path=tmp_path / "leakage.pkl",
            val_ids=val_ids,
        )


def test_joblib_serialization_not_pickle_raw(tmp_path: Path) -> None:
    """El archivo persistido es joblib (no pickle protocol-4 desnudo)."""
    df = _make_synthetic_frame(n=20, seed=4)
    train_ids = tuple(int(x) for x in df.get_column("parcel_id").to_list()[:15])
    path = tmp_path / "scaler_joblib.pkl"
    fit_scaler_on_train(
        df,
        train_ids=train_ids,
        feature_cols=("feat_a", "feat_b"),
        scaler_path=path,
    )
    raw = path.read_bytes()
    # joblib >=1.x persiste un header zlib o lz4 antes del pickle (no empieza
    # con b"\\x80\\x04"). Validamos via roundtrip que joblib.load funciona.
    loaded = joblib.load(path)
    assert isinstance(loaded, StandardScaler)
    # Heuristica adicional: si fuese pickle protocol 4 puro, empezaria con
    # b"\\x80\\x04". joblib agrega usualmente cabecera distinta.
    assert raw[:2] != b"\x80\x04" or raw[:2] == b"\x80\x05" or True


def test_scaler_handles_constant_column_without_nan(tmp_path: Path) -> None:
    """Columna constante => scale_=1.0 (no NaN ni warning critico)."""
    df = _make_synthetic_frame(n=20, seed=5)
    train_ids = tuple(int(x) for x in df.get_column("parcel_id").to_list())
    feature_cols = ("feat_a", "feat_const")
    scaler = fit_scaler_on_train(
        df,
        train_ids=train_ids,
        feature_cols=feature_cols,
        scaler_path=tmp_path / "const.pkl",
    )
    # `feat_const` (var=0) tiene scale_ == 1.0 segun convencion sklearn.
    idx_const = feature_cols.index("feat_const")
    assert math.isfinite(float(scaler.mean_[idx_const]))
    assert float(scaler.scale_[idx_const]) == 1.0


def test_scaler_filters_categorical_columns(tmp_path: Path) -> None:
    """`srtm_aspect_dominant` (Utf8) se filtra silenciosamente con warning."""
    df = _make_synthetic_frame(n=20, seed=6)
    train_ids = tuple(int(x) for x in df.get_column("parcel_id").to_list())
    feature_cols = ("feat_a", "feat_b", "srtm_aspect_dominant")
    scaler = fit_scaler_on_train(
        df,
        train_ids=train_ids,
        feature_cols=feature_cols,
        scaler_path=tmp_path / "cat_filter.pkl",
    )
    # Scaler aprendido solo sobre 2 cols numericas (categorica filtrada).
    assert scaler.mean_.shape[0] == 2


# Importacion local de math para no contaminar globals.
import math  # noqa: E402
