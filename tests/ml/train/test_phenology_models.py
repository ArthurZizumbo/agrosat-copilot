"""Tests de ``ml.train.phenology_models`` (US-022b-C).

Smoke CPU para TempCNN/InceptionTime (2 batches reales por fold) + tests
deterministicos del adaptador Polars->tensor + reconstruccion FFT.

Cobertura objetivo >=75%. Los tests se saltean si ``torch`` no esta
instalado (no aplica en el dev environment estandar pero documenta la
dependencia).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from ml.train import phenology_models as pm  # noqa: E402
from ml.train.phenology_models import (  # noqa: E402  - torch must be available first
    DEFAULT_FFT_HARMONICS,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_TEMPORAL_INDICES,
    TemporalDataset,
    _NullMlflowRun,
    _reconstruct_curve,
    _resolve_code_version,
    _resolve_data_version,
    _try_mlflow_run,
    build_temporal_tensor,
    train_temporal_model,
)


@pytest.fixture
def fft_features_df() -> pl.DataFrame:
    """DataFrame con coeficientes FFT y metadata minima."""
    rng = np.random.default_rng(42)
    n = 60
    classes = rng.integers(low=1, high=4, size=n)
    patch_ids = rng.integers(low=10000, high=10010, size=n).astype(np.int64)
    base = {
        "parcel_id": [f"{p}_{i}" for p, i in zip(patch_ids, range(n), strict=True)],
        "year": [2019] * n,
        "patch_id": patch_ids.tolist(),
        "class_id": classes.tolist(),
    }
    for idx in DEFAULT_TEMPORAL_INDICES:
        for k in range(DEFAULT_FFT_HARMONICS + 1):
            base[f"{idx}_fft_amp_{k}"] = (rng.normal(size=n) + classes * 0.2).tolist()
            base[f"{idx}_fft_phase_{k}"] = rng.uniform(-np.pi, np.pi, size=n).tolist()
    return pl.DataFrame(base)


def test_reconstruct_curve_from_fft_returns_correct_shape(
    fft_features_df: pl.DataFrame,
) -> None:
    curve = _reconstruct_curve(fft_features_df, index_name="NDVI", sequence_length=72)
    assert curve.shape == (fft_features_df.height, 72)
    assert curve.dtype == np.float32


def test_reconstruct_curve_fallback_to_mean() -> None:
    df = pl.DataFrame(
        {
            "parcel_id": [1, 2, 3],
            "year": [2019, 2019, 2019],
            "NDVI_mean": [0.5, 0.3, 0.8],
        }
    )
    curve = _reconstruct_curve(df, index_name="NDVI", sequence_length=12)
    assert curve.shape == (3, 12)
    # Cada fila es constante = mean.
    assert np.allclose(curve[0], 0.5)
    assert np.allclose(curve[1], 0.3)
    assert np.allclose(curve[2], 0.8)


def test_temporal_dataset_to_tensor_shape(fft_features_df: pl.DataFrame) -> None:
    tensor = TemporalDataset(
        fft_features_df, indices=("NDVI", "NDWI"), sequence_length=24
    ).to_tensor()
    assert tensor.shape == (fft_features_df.height, 24, 2)
    assert tensor.dtype == np.float32


def test_build_temporal_tensor_matches_dataset(fft_features_df: pl.DataFrame) -> None:
    tensor_a = build_temporal_tensor(fft_features_df, indices=("NDVI",), sequence_length=18)
    tensor_b = TemporalDataset(fft_features_df, indices=("NDVI",), sequence_length=18).to_tensor()
    assert np.allclose(tensor_a, tensor_b)


def test_train_tempcnn_smoke_returns_result(fft_features_df: pl.DataFrame) -> None:
    """Smoke CPU: TempCNN entrena 2 epocas sin reventar y devuelve metricas."""
    result = train_temporal_model(
        df=fft_features_df,
        model_kind="tempcnn",
        n_epochs=2,
        batch_size=16,
        seed=42,
        device="cpu",
        sequence_length=24,
        indices=("NDVI",),
        k_folds=3,
        buffer_km=0.5,
    )
    assert result.model_kind == "tempcnn"
    assert 0.0 <= result.f1_macro <= 1.0
    assert 0.0 <= result.miou <= 1.0
    assert result.n_parcels > 0
    assert result.train_time_s > 0.0
    assert result.mlflow_run_id is None  # sin mlflow_uri.


def test_train_inceptiontime_smoke_returns_result(
    fft_features_df: pl.DataFrame,
) -> None:
    result = train_temporal_model(
        df=fft_features_df,
        model_kind="inceptiontime",
        n_epochs=2,
        batch_size=16,
        seed=42,
        device="cpu",
        sequence_length=24,
        indices=("NDVI",),
        k_folds=3,
        buffer_km=0.5,
    )
    assert result.model_kind == "inceptiontime"
    assert result.n_classes >= 1


def test_train_invalid_model_kind_raises(fft_features_df: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="model_kind"):
        train_temporal_model(df=fft_features_df, model_kind="lstm")  # type: ignore[arg-type]


def test_train_no_df_no_path_raises() -> None:
    with pytest.raises(ValueError, match="features_path"):
        train_temporal_model(model_kind="tempcnn")


def test_train_seed_determinism(fft_features_df: pl.DataFrame) -> None:
    """Misma seed -> mismo F1-macro (CPU, sin GPU non-determinism)."""
    kwargs = dict(
        df=fft_features_df,
        model_kind="tempcnn",
        n_epochs=2,
        batch_size=16,
        device="cpu",
        sequence_length=12,
        indices=("NDVI",),
        k_folds=3,
        buffer_km=0.5,
    )
    r1 = train_temporal_model(seed=42, **kwargs)
    r2 = train_temporal_model(seed=42, **kwargs)
    # Tolerancia minima por shuffle interno de torch.randperm en CPU.
    assert abs(r1.f1_macro - r2.f1_macro) < 1e-3


def test_train_subsample_max_samples(fft_features_df: pl.DataFrame) -> None:
    result = train_temporal_model(
        df=fft_features_df,
        model_kind="tempcnn",
        n_epochs=1,
        batch_size=8,
        max_samples=30,
        device="cpu",
        sequence_length=12,
        indices=("NDVI",),
        k_folds=3,
        buffer_km=0.5,
    )
    assert result.n_parcels <= 30


def test_temporal_indices_default_constants() -> None:
    assert DEFAULT_TEMPORAL_INDICES == ("NDVI", "NDWI", "EVI")
    assert DEFAULT_SEQUENCE_LENGTH == 72


def test_train_filters_pastis_drop_classes() -> None:
    """class_id 0 (Background) y 19 (Void) deben filtrarse antes del CV."""
    rng = np.random.default_rng(0)
    n = 30
    base = {
        "parcel_id": [str(i) for i in range(n)],
        "year": [2019] * n,
        "patch_id": list(range(10000, 10000 + n)),
        # 10 con class_id=0 (Background) -> deben drop.
        "class_id": [0] * 10 + [1] * 10 + [2] * 10,
    }
    for idx in ("NDVI",):
        for k in range(DEFAULT_FFT_HARMONICS + 1):
            base[f"{idx}_fft_amp_{k}"] = rng.normal(size=n).tolist()
            base[f"{idx}_fft_phase_{k}"] = rng.normal(size=n).tolist()
    df = pl.DataFrame(base)
    result = train_temporal_model(
        df=df,
        model_kind="tempcnn",
        n_epochs=1,
        batch_size=8,
        device="cpu",
        sequence_length=12,
        indices=("NDVI",),
        k_folds=3,
        buffer_km=0.5,
    )
    # 10 parcelas con class_id 0 deben haberse filtrado.
    assert result.n_parcels == 20


# ---------------------------------------------------------------------------
# QA-3: tests de MLflow helpers + _resolve_data_version / _resolve_code_version
# ---------------------------------------------------------------------------


def test_null_mlflow_run_context_returns_none_and_noop_logging():
    """``_NullMlflowRun`` debe entrar/salir sin tocar mlflow y aceptar log_*."""
    null_ctx = _NullMlflowRun()
    with null_ctx as run:
        assert run is None
        # Los metodos noop no deben levantar.
        null_ctx.log_params({"foo": "bar"})  # type: ignore[func-returns-value]
        null_ctx.log_metric("acc", 0.5)  # type: ignore[func-returns-value]
    assert null_ctx.run_id is None


def test_try_mlflow_run_returns_null_when_uri_is_none():
    ctx = _try_mlflow_run(None, model_kind="tempcnn")
    assert isinstance(ctx, _NullMlflowRun)


def test_try_mlflow_run_returns_mlflow_run_when_uri_provided(monkeypatch):
    """Si ``mlflow`` importa, ``_try_mlflow_run`` debe devolver ``_MlflowRun``."""
    # Asegura que el import dentro del helper no toque la red: mockeamos
    # ``mlflow`` antes de la llamada para que parezca disponible.
    import sys
    import types

    fake_mlflow = types.ModuleType("mlflow")
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    ctx = _try_mlflow_run("file:///tmp/mlruns", model_kind="tempcnn")
    assert ctx.__class__.__name__ == "_MlflowRun"


def test_mlflow_run_lifecycle_with_mocked_mlflow(monkeypatch):
    """Lifecycle completo: __enter__ inicia run, set_tags, log_*; __exit__ cierra.

    Mockeamos el modulo ``mlflow`` con un stub minimo (set_tracking_uri,
    start_run, set_tags, end_run, log_params, log_metric, log_artifact).
    """
    import sys
    import types

    calls: dict[str, list] = {
        "set_tracking_uri": [],
        "set_tags": [],
        "log_params": [],
        "log_metric": [],
        "log_artifact": [],
        "end_run": [],
    }

    class _Run:
        class info:
            run_id = "RUN-123"

    fake = types.ModuleType("mlflow")
    fake.set_tracking_uri = lambda uri: calls["set_tracking_uri"].append(uri)  # type: ignore[attr-defined]
    fake.start_run = lambda run_name=None: _Run()  # type: ignore[attr-defined]
    fake.set_tags = lambda tags: calls["set_tags"].append(tags)  # type: ignore[attr-defined]
    fake.log_params = lambda params: calls["log_params"].append(params)  # type: ignore[attr-defined]
    fake.log_metric = lambda key, value, step=None: calls["log_metric"].append(  # type: ignore[attr-defined]
        (key, value, step)
    )
    fake.log_artifact = lambda path, artifact_path=None: calls["log_artifact"].append(  # type: ignore[attr-defined]
        (path, artifact_path)
    )
    fake.end_run = lambda: calls["end_run"].append(True)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", fake)

    ctx = _try_mlflow_run("file:///tmp/mlruns", model_kind="inceptiontime")
    with ctx as run_ctx:
        assert run_ctx is ctx
        assert ctx.run_id == "RUN-123"
        ctx.log_params({"lr": 1e-3})
        ctx.log_metric("f1_macro", 0.42, step=1)
        # NaN debe filtrarse (no se loguea).
        ctx.log_metric("nan_metric", float("nan"))
        ctx.log_artifact("dummy.txt", artifact_path="checkpoints")

    assert calls["set_tracking_uri"] == ["file:///tmp/mlruns"]
    assert len(calls["set_tags"]) == 1
    tags = calls["set_tags"][0]
    assert tags["model_kind"] == "inceptiontime"
    assert "data_version" in tags and "code_version" in tags
    assert calls["log_params"] == [{"lr": 1e-3}]
    # Solo el log_metric con valor finito se llamo (NaN filtrado).
    assert len(calls["log_metric"]) == 1
    assert calls["log_metric"][0][:2] == ("f1_macro", 0.42)
    assert calls["log_artifact"] == [("dummy.txt", "checkpoints")]
    assert calls["end_run"] == [True]


def test_mlflow_run_log_state_dict_persists_artifact(monkeypatch, tmp_path):
    """``log_state_dict`` debe serializar el state_dict con torch y registrar artifact."""
    import sys
    import types

    captured: dict = {}

    class _Run:
        class info:
            run_id = "RUN-456"

    fake = types.ModuleType("mlflow")
    fake.set_tracking_uri = lambda uri: None  # type: ignore[attr-defined]
    fake.start_run = lambda run_name=None: _Run()  # type: ignore[attr-defined]
    fake.set_tags = lambda tags: None  # type: ignore[attr-defined]
    fake.log_params = lambda params: None  # type: ignore[attr-defined]
    fake.log_metric = lambda key, value, step=None: None  # type: ignore[attr-defined]
    fake.end_run = lambda: None  # type: ignore[attr-defined]

    def _log_artifact(path, artifact_path=None):
        captured["path"] = path
        captured["artifact_path"] = artifact_path

    fake.log_artifact = _log_artifact  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", fake)

    # Modelo trivial con un parametro.
    model = torch.nn.Linear(4, 2)
    ctx = _try_mlflow_run("file:///tmp/mlruns", model_kind="tempcnn")
    with ctx:
        ctx.log_state_dict(model, name="weights.pt")

    assert captured["artifact_path"] == "checkpoints"
    assert captured["path"].endswith("weights.pt")


def test_resolve_data_version_reads_dvc_md5(monkeypatch, tmp_path):
    """Si existe el ``.dvc`` con linea ``md5:``, debe extraer el hash corto."""
    repo_root = tmp_path
    (repo_root / "data" / "test_fixtures").mkdir(parents=True)
    dvc_file = repo_root / "data" / "test_fixtures" / "feature_selection_parcels_subset.parquet.dvc"
    dvc_file.write_text(
        "outs:\n- md5: abcdef1234567890abcdef\n  size: 80000000\n  path: foo.parquet\n",
        encoding="utf-8",
    )
    # Apunta el modulo a un repo_root falso reescribiendo __file__.
    fake_file = repo_root / "ml" / "train" / "phenology_models.py"
    fake_file.parent.mkdir(parents=True)
    fake_file.write_text("# stub", encoding="utf-8")
    monkeypatch.setattr(pm, "__file__", str(fake_file))

    version = _resolve_data_version()
    assert version == "abcdef123456"  # primeros 12 chars


def test_resolve_data_version_fallback_to_unknown_when_dvc_missing(monkeypatch, tmp_path):
    """Sin ``.dvc`` y sin git, debe devolver ``unknown`` (no levantar)."""
    fake_file = tmp_path / "ml" / "train" / "phenology_models.py"
    fake_file.parent.mkdir(parents=True)
    fake_file.write_text("# stub", encoding="utf-8")
    monkeypatch.setattr(pm, "__file__", str(fake_file))

    # Fuerza subprocess.run a devolver stdout vacio.
    import subprocess

    def _fake_run(*args, **kwargs):
        class _R:
            stdout = ""

        return _R()

    monkeypatch.setattr(subprocess, "run", _fake_run)
    version = _resolve_data_version()
    assert version == "unknown"


def test_resolve_code_version_returns_git_short_sha_or_unknown():
    """Devuelve un valor truthy (sha corto si git OK, ``unknown`` si no)."""
    version = _resolve_code_version()
    # En CI/dev este repo tiene git, asi que esperamos sha de 7 chars o ``unknown``.
    assert isinstance(version, str)
    assert version != ""
    assert version == "unknown" or len(version) >= 4
