"""Tests de las extensiones de US-019: device GPU de XGBoost y cache de folds.

Cubre las funciones anadidas tras el run inicial del baseline:

- :func:`ml.train.baseline.resolve_xgb_device` — deteccion de GPU NVIDIA.
- :func:`ml.train.baseline.build_estimator` — inyeccion del device en XGB.
- El cache de folds espaciales (`_spatial_folds_cache_path`,
  `_save_cached_cv_splits`, `_load_cached_cv_splits`).

Todo se mockea: ``nvidia-smi`` via ``shutil.which`` + ``subprocess.run``;
ningun test invoca una GPU real ni el parquet de 85k parcelas.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from xgboost import XGBClassifier

from ml.train import baseline

# ---------------------------------------------------------------------------
# resolve_xgb_device
# ---------------------------------------------------------------------------


def test_resolve_xgb_device_cpu_when_no_nvidia_smi(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Sin ``nvidia-smi`` en el PATH, el device degrada a ``cpu``."""
    monkeypatch.setattr(shutil, "which", lambda _: None)
    assert baseline.resolve_xgb_device() == "cpu"


def test_resolve_xgb_device_cuda_when_nvidia_smi_responds(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Con ``nvidia-smi`` respondiendo una GPU, el device es ``cuda``."""
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    def fake_run(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout="NVIDIA GeForce RTX 4070\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert baseline.resolve_xgb_device() == "cuda"


def test_resolve_xgb_device_cpu_when_nvidia_smi_fails(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Si ``nvidia-smi`` falla (returncode != 0), degrada a ``cpu``."""
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    def fake_run(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="err")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert baseline.resolve_xgb_device() == "cpu"


def test_resolve_xgb_device_cpu_on_subprocess_error(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Si ``subprocess.run`` lanza, degrada a ``cpu`` sin propagar."""
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    def fake_run(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise OSError("boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert baseline.resolve_xgb_device() == "cpu"


# ---------------------------------------------------------------------------
# build_estimator
# ---------------------------------------------------------------------------


def test_build_estimator_xgb_injects_device(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """``build_estimator`` inyecta el device resuelto en el XGBClassifier."""
    monkeypatch.setattr(baseline, "resolve_xgb_device", lambda: "cpu")
    estimator = baseline.build_estimator("xgb", {"n_estimators": 10})
    assert isinstance(estimator, XGBClassifier)
    assert estimator.get_params()["device"] == "cpu"


def test_build_estimator_xgb_respects_explicit_device() -> None:
    """Si el caller fija ``device``, ``build_estimator`` no lo sobrescribe."""
    estimator = baseline.build_estimator("xgb", {"n_estimators": 10, "device": "cpu"})
    assert estimator.get_params()["device"] == "cpu"


def test_build_estimator_rejects_unknown_model() -> None:
    """Un ``model`` no soportado levanta ``ValueError``."""
    with pytest.raises(ValueError, match=r"rf.*xgb"):
        baseline.build_estimator("svm", {})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Cache de folds espaciales
# ---------------------------------------------------------------------------


def test_spatial_folds_cache_path_encodes_key() -> None:
    """La ruta del cache codifica n_rows, k, buffer y seed."""
    path = baseline._spatial_folds_cache_path(
        n_rows=1000, k_folds=5, buffer_km=1.0, random_state=42
    )
    assert path.name == "baseline_spatial_folds_n1000_k5_b1_s42.parquet"


def test_spatial_folds_cache_path_distinct_keys_distinct_paths() -> None:
    """Claves distintas producen rutas distintas (no colision de cache)."""
    a = baseline._spatial_folds_cache_path(n_rows=1000, k_folds=5, buffer_km=1.0, random_state=42)
    b = baseline._spatial_folds_cache_path(n_rows=2000, k_folds=5, buffer_km=1.0, random_state=42)
    assert a != b


def test_save_and_load_cv_splits_roundtrip(tmp_path: Path) -> None:
    """``_save_cached_cv_splits`` + ``_load_cached_cv_splits`` preservan los splits."""
    splits = [
        (np.array([0, 1, 2]), np.array([3, 4])),
        (np.array([3, 4]), np.array([0, 1, 2])),
    ]
    cache_path = tmp_path / "folds.parquet"
    baseline._save_cached_cv_splits(cache_path, splits)
    assert cache_path.exists()

    loaded = baseline._load_cached_cv_splits(cache_path)
    assert loaded is not None
    assert len(loaded) == len(splits)
    for (orig_train, orig_test), (got_train, got_test) in zip(splits, loaded, strict=False):
        np.testing.assert_array_equal(sorted(orig_train), sorted(got_train))
        np.testing.assert_array_equal(sorted(orig_test), sorted(got_test))


def test_load_cv_splits_returns_none_when_absent(tmp_path: Path) -> None:
    """``_load_cached_cv_splits`` devuelve ``None`` si el parquet no existe."""
    assert baseline._load_cached_cv_splits(tmp_path / "missing.parquet") is None
