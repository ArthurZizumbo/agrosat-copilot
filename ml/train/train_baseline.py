"""CLI Typer para entrenar los baselines RF/XGB del EPIC 4 (US-019).

Orquesta el entrenamiento de los dos modelos tabulares, el logging a
MLflow (runs ``baseline-{rf,xgb}-alphaearth-v1`` bajo el experimento
``agrosat-baseline``) y la persistencia de los joblib en ``artifacts/``.
Toda la logica de modelado vive en :mod:`ml.train.baseline`; este modulo
solo orquesta (separation of concerns, regla CLAUDE.md 8).

Uso::

    poetry run python ml/train/train_baseline.py \\
        --features-path data/test_fixtures/feature_selection_parcels_subset.parquet \\
        --model both \\
        --tune \\
        --mlflow-uri file:./mlruns \\
        --max-samples 0 \\
        --output-dir artifacts/

Operativo permanente (NO viola el anti-patron ``scripts/_*.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import joblib
import mlflow
import structlog
import typer

from ml.train.baseline import (
    BaselineResult,
    ModelKind,
    train_one_model,
    tune_baseline,
)
from ml.train.baseline import (
    _load_baseline_dataset as load_baseline_dataset,
)
from ml.train.baseline import (
    _prepare_dataframe as prepare_dataframe,
)
from ml.utils.mlflow_utils import track_experiment

logger = structlog.get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)

# MLflow 3.x writes emojis (run/experiment views) to stdout when closing
# a run; the Windows console uses cp1252 by default and that triggers
# UnicodeEncodeError. We force UTF-8 on the streams (no-op on Linux/macOS).
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")

_EXPERIMENT_NAME = "agrosat-baseline"
_REGISTERED_MODEL_NAME = "agrosat-baseline"
_DEFAULT_FEATURES = "data/test_fixtures/feature_selection_parcels_subset.parquet"


def _resolve_models(model: str) -> list[ModelKind]:
    """Traduce el flag ``--model`` a la lista de modelos a entrenar.

    Args:
        model: Valor del flag (``rf``, ``xgb`` o ``both``).

    Returns:
        Lista de ``ModelKind`` a entrenar.

    Raises:
        typer.BadParameter: si ``model`` no es un valor valido.
    """
    if model == "both":
        return ["rf", "xgb"]
    if model in ("rf", "xgb"):
        return [model]  # type: ignore[list-item]
    raise typer.BadParameter("`--model` debe ser 'rf', 'xgb' o 'both'.")


def _stratified_subsample(df, max_samples: int, seed: int = 42):  # type: ignore[no-untyped-def]
    """Submuestrea el DataFrame de forma estratificada por clase.

    Args:
        df: DataFrame Polars de features.
        max_samples: Tamano objetivo; ``0`` o negativo devuelve ``df``
            intacto.
        seed: Semilla determinista.

    Returns:
        El DataFrame submuestreado (o el original si ``max_samples <= 0``).
    """
    if max_samples <= 0 or df.height <= max_samples:
        return df
    from ml.utils.sampling import stratified_sample

    return stratified_sample(df, by=["class_id"], n=max_samples, seed=seed)


def _log_baseline_run(
    result: BaselineResult,
    *,
    run_name: str,
    output_dir: Path,
) -> Path:
    """Loggea un :class:`BaselineResult` a MLflow y persiste el joblib.

    Registra params (hiperparametros), metricas (OOF + CV mean/std),
    artefactos (matriz de confusion, reporte de clasificacion, joblib) y
    el modelo en el Model Registry ``agrosat-baseline``.

    Args:
        result: Resultado del entrenamiento.
        run_name: Nombre del run MLflow (``baseline-{rf,xgb}-alphaearth-v1``).
        output_dir: Directorio destino del joblib.

    Returns:
        La ruta del joblib persistido.
    """
    model_kind = result.model_kind

    mlflow.log_params({f"hp_{k}": v for k, v in result.best_params.items()})
    mlflow.log_param("algo", model_kind)
    mlflow.log_param("n_features", len(result.feature_cols))
    mlflow.log_param("n_classes", len(result.label_classes))

    for metric, value in result.metrics.items():
        mlflow.log_metric(f"oof_{metric}", value)
    for metric, (mean, std) in result.cv_metrics.items():
        mlflow.log_metric(f"cv_{metric}", mean)
        mlflow.log_metric(f"cv_{metric}_std", std)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib_path = output_dir / f"baseline_{model_kind}_v1.joblib"
    payload = {
        "model": result.model,
        "model_kind": model_kind,
        "feature_cols": list(result.feature_cols),
        "label_encoder": result.label_encoder,
        "label_classes": list(result.label_classes),
        "metrics": result.metrics,
        "cv_metrics": result.cv_metrics,
        "best_params": result.best_params,
        "data_version": mlflow.active_run().data.tags.get("data_version", "untracked"),
        "code_version": mlflow.active_run().data.tags.get("code_version", "unknown"),
    }
    joblib.dump(payload, joblib_path)
    mlflow.log_artifact(str(joblib_path))

    # Textual metrics summary as an inspectable artifact. The per-class
    # classification report and the confusion matrix over the out-of-fold
    # predictions are generated in the notebook 04_baseline (§6).
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        summary_path = Path(tmp) / "metrics_summary.txt"
        summary_lines = [f"baseline {model_kind} ({run_name})", ""]
        summary_lines += [f"oof_{m} = {v:.4f}" for m, v in result.metrics.items()]
        summary_lines.append("")
        summary_lines += [
            f"cv_{m} = {mean:.4f} +/- {std:.4f}" for m, (mean, std) in result.cv_metrics.items()
        ]
        summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
        mlflow.log_artifact(str(summary_path))

    # Register the model in the Model Registry. `await_registration_for=0`
    # prevents the client from blocking while waiting for the state transition
    # of the registered version (the synchronous polling can hang against the
    # local Docker server).
    if model_kind == "rf":
        mlflow.sklearn.log_model(
            sk_model=result.model,
            name="model",
            registered_model_name=_REGISTERED_MODEL_NAME,
            await_registration_for=0,
        )
    else:
        mlflow.xgboost.log_model(
            xgb_model=result.model,
            name="model",
            registered_model_name=_REGISTERED_MODEL_NAME,
            await_registration_for=0,
        )

    logger.info(
        "baseline_run_logged",
        run_name=run_name,
        model=model_kind,
        joblib=str(joblib_path),
        f1_macro=result.metrics.get("f1_macro"),
    )
    return joblib_path


@app.command()
def main(
    features_path: Annotated[
        Path,
        typer.Option(help="Parquet de features del EPIC 3."),
    ] = Path(_DEFAULT_FEATURES),
    model: Annotated[
        str,
        typer.Option(help="Modelo(s) a entrenar: rf, xgb o both."),
    ] = "both",
    tune: Annotated[
        bool,
        typer.Option("--tune/--no-tune", help="Ejecutar GridSearchCV ligero."),
    ] = True,
    mlflow_uri: Annotated[
        str,
        typer.Option(help="Tracking URI MLflow (vacio = autoresolucion)."),
    ] = "",
    max_samples: Annotated[
        int,
        typer.Option(help="Submuestreo estratificado para dev/CI (0 = todo)."),
    ] = 0,
    output_dir: Annotated[
        Path,
        typer.Option(help="Destino de los joblib."),
    ] = Path("artifacts"),
) -> None:
    """Entrena los baselines RF/XGB y los registra en MLflow.

    Args:
        features_path: Ruta al parquet de features del EPIC 3.
        model: ``rf``, ``xgb`` o ``both``.
        tune: Si ``True`` ejecuta tuning ligero via ``GridSearchCV``.
        mlflow_uri: Override del tracking URI; cadena vacia delega en
            :func:`resolve_tracking_uri`.
        max_samples: Tamano del submuestreo estratificado (0 = dataset
            completo).
        output_dir: Directorio destino de los joblib.
    """
    if not features_path.exists():
        logger.warning(
            "features_parquet_missing",
            path=str(features_path),
            note="Dataset ausente; nada que entrenar. Genera el subset del EPIC 3.",
        )
        raise typer.Exit(code=0)

    models = _resolve_models(model)
    df = load_baseline_dataset(features_path)
    df = prepare_dataframe(df)
    df = _stratified_subsample(df, max_samples)
    logger.info(
        "baseline_cli_start",
        models=models,
        n_samples=df.height,
        tune=tune,
        features_path=str(features_path),
    )

    tracking_override = mlflow_uri or None
    dvc_path = str(features_path)

    for model_kind in models:
        run_name = f"baseline-{model_kind}-alphaearth-v1"
        with track_experiment(
            _EXPERIMENT_NAME,
            run_name=run_name,
            tracking_uri=tracking_override,
            dvc_path=dvc_path,
        ):
            hyperparams: dict[str, object] | None = None
            if tune:
                hyperparams = tune_baseline(df, model=model_kind)
                logger.info("baseline_best_params", model=model_kind, params=hyperparams)
            result = train_one_model(df, model=model_kind, hyperparams=hyperparams)
            joblib_path = _log_baseline_run(result, run_name=run_name, output_dir=output_dir)
            typer.echo(
                f"[{model_kind}] F1-macro OOF={result.metrics['f1_macro']:.4f} -> {joblib_path}"
            )

    logger.info("baseline_cli_done", models=models)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(app())
