"""Tests smoke de ml.utils.mlflow_utils (US-019).

Conjunto minimo de validacion. Ningun test contacta al servidor MLflow
Docker real: el probe se desactiva o se mockea (R11 del plan: CI sin
Docker debe quedar verde).
"""

from __future__ import annotations

import mlflow

from ml.utils.mlflow_utils import resolve_tracking_uri, track_experiment


def test_resolve_tracking_uri_respects_override() -> None:
    """Un override explicito gana sobre cualquier otra fuente."""
    assert resolve_tracking_uri("file:/tmp/custom") == "file:/tmp/custom"


def test_resolve_tracking_uri_falls_back_to_file_store(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Sin override ni env var, y sin probe, cae al file store."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert resolve_tracking_uri(probe_server=False) == "file:./mlruns"


def test_resolve_tracking_uri_prefers_env_var(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """La variable de entorno gana sobre el server/file store."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://example:5000")
    assert resolve_tracking_uri(probe_server=False) == "http://example:5000"


def test_track_experiment_sets_versioning_tags(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """El run abierto lleva los tags code_version y data_version."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    uri = f"file:{tmp_path / 'mlruns'}"
    with track_experiment(
        "test-exp", run_name="smoke-run", tracking_uri=uri, probe_server=False
    ) as run:
        run_id = run.info.run_id

    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    fetched = client.get_run(run_id)
    assert "code_version" in fetched.data.tags
    assert "data_version" in fetched.data.tags
    assert fetched.data.tags["data_version"] == "untracked"
    assert fetched.info.run_name == "smoke-run"
