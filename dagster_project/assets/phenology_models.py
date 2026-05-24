"""Assets Dagster — US-022b-C entrenamiento de modelos temporales.

Declara los assets que entrenan TempCNN e InceptionTime sobre la FE
fenologica del subset US-018 con spatial CV 5-fold, reusando
:func:`ml.train.phenology_models.train_temporal_model`.

El training NO se ejecuta automaticamente desde un schedule (decision de
infra US-022b-A: training a la carta via ``make reencuadre-notebook-full``
o invocacion manual del asset). Los assets se materializan cuando el
usuario lo pide explicitamente desde la UI o desde la CLI. Esto evita
gastar GPU por accidente.

Mapeo a criterios de aceptacion (docs/us-planning/us-022b.md §3.3):

- **C-3**: ``phenology_model_tempcnn`` + ``phenology_model_inceptiontime``
  entrenan los dos modelos con la misma spatial CV.
- **C-4**: ``temporal_models_comparison`` consolida las metricas y
  reporta delta vs baseline tabular en ``reports/baseline/phenology_models.csv``.
- **MLflow**: cada asset registra params (model_kind, n_epochs, batch_size,
  device, n_parcels, n_classes), metricas por epoch, OOF metrics y el
  state_dict del modelo del ultimo fold como artifact.

Lineage:

::

    feature_selection_parcels_subset.parquet (data fixture US-018)
        |
        +-> phenology_model_tempcnn        (MLflow run + state_dict artifact)
        +-> phenology_model_inceptiontime  (MLflow run + state_dict artifact)
                |
                +-> temporal_models_comparison  (reports/baseline/phenology_models.csv)
"""

from pathlib import Path

import polars as pl
from dagster import AssetExecutionContext, MaterializeResult, MetadataValue, asset

#: Path canonico del subset de features (US-018 + US-015).
_FEATURES_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "test_fixtures"
    / "feature_selection_parcels_subset.parquet"
)

#: Path canonico de la tabla comparativa que produce
#: ``temporal_models_comparison`` y que consume el notebook 05.
_REPORTS_PATH = (
    Path(__file__).resolve().parents[2] / "reports" / "baseline" / "phenology_models.csv"
)


def _train_one(
    context: AssetExecutionContext,
    *,
    model_kind: str,
    n_epochs: int,
    batch_size: int,
    device: str,
    seed: int,
) -> dict[str, float | int | str]:
    """Helper interno: entrena un modelo temporal y devuelve un dict serializable.

    Importacion lazy de ``train_temporal_model`` para evitar levantar torch
    cuando solo se valida el grafo de assets (`dagster definitions validate`).
    """
    from ml.train.phenology_models import train_temporal_model

    result = train_temporal_model(
        features_path=_FEATURES_PATH,
        model_kind=model_kind,  # type: ignore[arg-type]
        n_epochs=n_epochs,
        batch_size=batch_size,
        seed=seed,
        device=device,
        # mlflow_uri se inyecta desde el resource mlflow si esta presente.
        mlflow_uri=None,
    )
    context.log.info(
        f"trained {model_kind}: F1-macro={result.f1_macro:.4f} "
        f"mIoU={result.miou:.4f} t={result.train_time_s:.1f}s "
        f"n_parcels={result.n_parcels} n_classes={result.n_classes}"
    )
    return {
        "model_kind": result.model_kind,
        "f1_macro": float(result.f1_macro),
        "f1_weighted": float(result.f1_weighted),
        "miou": float(result.miou),
        "cohen_kappa": float(result.cohen_kappa),
        "train_time_s": float(result.train_time_s),
        "n_parcels": int(result.n_parcels),
        "n_classes": int(result.n_classes),
        "mlflow_run_id": result.mlflow_run_id or "",
    }


@asset(
    group_name="phenology_models",
    description=(
        "TempCNN (Pelletier et al. 2019) nativo sobre la FE fenologica del "
        "subset US-018 con spatial CV 5-fold. Output: dict con metricas OOF "
        "y referencia al run MLflow."
    ),
    metadata={
        "owner": "isaac.avila",
        "us": "US-022b-C",
        "model_arch": "ml.models.temporal.TempCNN",
    },
)
def phenology_model_tempcnn(
    context: AssetExecutionContext,
) -> MaterializeResult:
    """Entrena TempCNN sobre la FE fenologica con spatial CV 5-fold."""
    metrics = _train_one(
        context,
        model_kind="tempcnn",
        n_epochs=30,
        batch_size=256,
        device="auto",
        seed=42,
    )
    return MaterializeResult(
        metadata={
            "f1_macro": MetadataValue.float(metrics["f1_macro"]),  # type: ignore[arg-type]
            "miou": MetadataValue.float(metrics["miou"]),  # type: ignore[arg-type]
            "train_time_s": MetadataValue.float(metrics["train_time_s"]),  # type: ignore[arg-type]
            "n_parcels": MetadataValue.int(metrics["n_parcels"]),  # type: ignore[arg-type]
            "mlflow_run_id": MetadataValue.text(str(metrics["mlflow_run_id"])),
            "data_version": MetadataValue.text("us018-phenology-subset"),
            "model_kind": MetadataValue.text("tempcnn"),
            "us_label": MetadataValue.text("US-022b-C"),
        },
    )


@asset(
    group_name="phenology_models",
    description=(
        "InceptionTime (Fawaz et al. 2020) nativo sobre la FE fenologica del "
        "subset US-018 con spatial CV 5-fold. Output: dict con metricas OOF "
        "y referencia al run MLflow."
    ),
    metadata={
        "owner": "isaac.avila",
        "us": "US-022b-C",
        "model_arch": "ml.models.temporal.InceptionTime",
    },
)
def phenology_model_inceptiontime(
    context: AssetExecutionContext,
) -> MaterializeResult:
    """Entrena InceptionTime sobre la FE fenologica con spatial CV 5-fold."""
    metrics = _train_one(
        context,
        model_kind="inceptiontime",
        n_epochs=30,
        batch_size=256,
        device="auto",
        seed=42,
    )
    return MaterializeResult(
        metadata={
            "f1_macro": MetadataValue.float(metrics["f1_macro"]),  # type: ignore[arg-type]
            "miou": MetadataValue.float(metrics["miou"]),  # type: ignore[arg-type]
            "train_time_s": MetadataValue.float(metrics["train_time_s"]),  # type: ignore[arg-type]
            "n_parcels": MetadataValue.int(metrics["n_parcels"]),  # type: ignore[arg-type]
            "mlflow_run_id": MetadataValue.text(str(metrics["mlflow_run_id"])),
            "data_version": MetadataValue.text("us018-phenology-subset"),
            "model_kind": MetadataValue.text("inceptiontime"),
            "us_label": MetadataValue.text("US-022b-C"),
        },
    )


@asset(
    group_name="phenology_models",
    description=(
        "Tabla comparativa de modelos temporales (TempCNN + InceptionTime) "
        "vs baseline tabular 0.32. Lee las metricas de los dos assets "
        "anteriores via MLflow y persiste reports/baseline/phenology_models.csv."
    ),
    deps=[phenology_model_tempcnn, phenology_model_inceptiontime],
    metadata={
        "owner": "isaac.avila",
        "us": "US-022b-C",
        "output_path": str(_REPORTS_PATH),
    },
)
def temporal_models_comparison(
    context: AssetExecutionContext,
) -> MaterializeResult:
    """Persiste la tabla comparativa de modelos temporales en CSV.

    Esta version del asset reentrenamiento es deliberadamente sencilla: vuelve
    a invocar ``train_temporal_model`` para los dos modelos y consolida.
    Para evitar reentrenar en cada materializacion, una version posterior
    podra leer las metricas de MLflow via ``mlflow.search_runs``; por ahora
    la simplicidad gana.
    """
    rows = []
    for model_kind in ("tempcnn", "inceptiontime"):
        metrics = _train_one(
            context,
            model_kind=model_kind,
            n_epochs=30,
            batch_size=256,
            device="auto",
            seed=42,
        )
        rows.append(
            {
                "model": model_kind,
                "f1_macro": round(metrics["f1_macro"], 4),  # type: ignore[arg-type]
                "f1_weighted": round(metrics["f1_weighted"], 4),  # type: ignore[arg-type]
                "miou": round(metrics["miou"], 4),  # type: ignore[arg-type]
                "cohen_kappa": round(metrics["cohen_kappa"], 4),  # type: ignore[arg-type]
                "train_time_s": round(metrics["train_time_s"], 2),  # type: ignore[arg-type]
                "n_parcels": metrics["n_parcels"],
                "delta_vs_baseline": round(
                    float(metrics["f1_macro"]) - 0.32, 4
                ),
            }
        )

    table = pl.DataFrame(rows).sort("f1_macro", descending=True)
    _REPORTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.write_csv(_REPORTS_PATH)
    context.log.info(f"Tabla comparativa persistida: {_REPORTS_PATH}")

    best = table.row(0, named=True)
    return MaterializeResult(
        metadata={
            "best_model": MetadataValue.text(best["model"]),
            "best_f1_macro": MetadataValue.float(best["f1_macro"]),
            "best_delta_vs_baseline": MetadataValue.float(best["delta_vs_baseline"]),
            "table_path": MetadataValue.path(str(_REPORTS_PATH)),
            "preview": MetadataValue.md(table.to_pandas().to_markdown(index=False)),
            "us_label": MetadataValue.text("US-022b-C"),
        },
    )
