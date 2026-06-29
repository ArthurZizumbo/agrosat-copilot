"""Runner: build the Italian ``xgb-alphaearth`` member for the Voting (US-079).

Replicates the THIRD member of the champion Voting-3 (``tsvit-pheno`` + ``utae``
+ ``xgb-alphaearth``, F1 0.9069 in France) on the Italian homologue, re-downloading
CLEAN AlphaEarth embeddings over the REAL parcels of the US-078 dataset (not the
``stab`` cache). Steps:

1. Extract the 64-dim AlphaEarth 2018 embedding per parcel polygon (real GEE via
   ADC) for the EuroCrops Italy 2018 parcels that fall in the US-078 patch bboxes,
   and materialize ``data/features/alphaearth_italia_2018.parquet``.
2. Train the per-parcel ``xgb-alphaearth-italia`` member over the Italian label
   space with anti-leakage SPATIAL cross-validation, and dump the per-parcel OOF
   post-softmax probabilities (the artifact the parcel-level Voting consumes).
3. Log the run to MLflow (server :5010, file-store fallback) with
   ``data_version`` + ``code_version``.

The runner is parametrizable on ``--patches-metadata`` so it can be re-run on the
FULL dataset (``F:\\worktrees\\us078\\data\\pastis_italia_2018\\metadata.parquet``)
once the patch download finishes -- the 20-pilot subset only validates the
end-to-end pipeline with REAL GEE data.

AlphaEarth is GEE-free, so this does NOT touch the Sentinel Hub quota the patch
download is consuming in parallel.

Usage::

    poetry run python scripts/build_alphaearth_italia.py \\
        --dataset-dir data/pastis_italia_2018 \\
        --out-path data/features/alphaearth_italia_2018.parquet

    # Full dataset (re-run when the VM download completes):
    poetry run python scripts/build_alphaearth_italia.py \\
        --patches-metadata F:/worktrees/us078/data/pastis_italia_2018/metadata.parquet
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog
import typer

from ml.ensemble.xgb_alphaearth_italia import (
    DEFAULT_OOF_DIR,
    ITALIA_XGB_MEMBER,
    train_xgb_alphaearth_italia,
)
from ml.transfer.alphaearth_italia import (
    DEFAULT_DATASET_DIR,
    DEFAULT_FEATURES_PATH,
    ITALIA_YEAR,
    ITALY_PARCELS_PARQUET,
    build_alphaearth_italia_features,
)

logger = structlog.get_logger(__name__)

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    dataset_dir: Path = typer.Option(
        DEFAULT_DATASET_DIR, help="US-078 dataset dir (class_mapping.json lives here)."
    ),
    patches_metadata: Path | None = typer.Option(
        None,
        help="Override metadata.parquet (e.g. the full dataset on the VM). Defaults "
        "to dataset_dir/metadata.parquet.",
    ),
    out_path: Path = typer.Option(
        DEFAULT_FEATURES_PATH, help="Destination parquet for the per-parcel features."
    ),
    oof_dir: Path = typer.Option(
        DEFAULT_OOF_DIR, help="Directory for the per-parcel OOF parquet (Voting input)."
    ),
    year: int = typer.Option(ITALIA_YEAR, help="AlphaEarth annual image year."),
    parcels_parquet: Path = typer.Option(
        ITALY_PARCELS_PARQUET,
        help="EuroCrops parcels parquet (e.g. iti1_2023.parquet or de4_2023.parquet "
        "to migrate the ground truth; defaults to the Italy 2018 reference).",
    ),
    region_prefix: str = typer.Option(
        "it",
        help="NUTS prefix selecting the crosswalk region (it, de4, nl, ...).",
    ),
    mapping_csv_override: Path | None = typer.Option(
        None,
        "--mapping-csv",
        help="EuroCrops crosswalk CSV (eurocrops.csv); defaults to the Italy mapping.",
    ),
    batch_size: int = typer.Option(100, help="Polygons per GEE reduceRegions request."),
    project: str | None = typer.Option(
        None, help="GCP project for the EE quota (ADC). Defaults to the active one."
    ),
    service_account_json: Path | None = typer.Option(
        None, help="Optional service-account key for GEE (else ADC)."
    ),
    buffer_km: float = typer.Option(1.0, help="Inter-fold buffer (km) for the CV fallback."),
    random_state: int = typer.Option(42, help="Deterministic seed."),
    use_mlflow: bool = typer.Option(True, help="Log the run to MLflow (:5010)."),
    reuse_features: bool = typer.Option(
        False,
        help="Skip GEE extraction and reuse an existing features parquet at --out-path "
        "(useful to re-train the xgb without re-sampling).",
    ),
) -> None:
    """Extract AlphaEarth per parcel + train xgb-alphaearth-italia + dump OOF.

    Args:
        dataset_dir: US-078 dataset directory.
        patches_metadata: Override path to the patch metadata parquet.
        out_path: Destination parquet for the per-parcel AlphaEarth features.
        oof_dir: Directory for the per-parcel OOF parquet.
        year: AlphaEarth annual image year (default 2018).
        batch_size: Polygons per GEE request.
        project: GCP project for the EE quota.
        service_account_json: Optional GEE service-account key.
        buffer_km: Inter-fold buffer for the spatial-CV fallback.
        random_state: Deterministic seed.
        use_mlflow: Log the run to MLflow.
        reuse_features: Reuse an existing features parquet instead of re-sampling.
    """
    import polars as pl

    if reuse_features and out_path.is_file():
        logger.info("alphaearth_italia_reuse_features", path=str(out_path))
        features = pl.read_parquet(out_path)
    else:
        ae_kwargs: dict[str, object] = {
            "dataset_dir": dataset_dir,
            "patches_metadata": patches_metadata,
            "parcels_parquet": parcels_parquet,
            "region_prefix": region_prefix,
            "year": year,
            "batch_size": batch_size,
            "project": project,
            "service_account_json": service_account_json,
            "out_path": out_path,
        }
        if mapping_csv_override is not None:
            ae_kwargs["mapping_csv"] = mapping_csv_override
        features = build_alphaearth_italia_features(**ae_kwargs)

    if features.is_empty():
        typer.secho(
            "GEE returned NO embeddings (auth/quota/network). The features parquet "
            "was not written and the xgb member cannot be trained. Verify ADC with "
            "`gcloud auth application-default print-access-token` and re-run.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    n_parcels = features.height
    n_classes_features = int(features["class_id"].n_unique())
    logger.info(
        "alphaearth_italia_features_ready",
        n_parcels=n_parcels,
        n_classes=n_classes_features,
        n_folds=int(features["fold"].n_unique()),
    )

    result = train_xgb_alphaearth_italia(
        features,
        buffer_km=buffer_km,
        random_state=random_state,
        oof_dir=oof_dir,
        member=ITALIA_XGB_MEMBER,
    )

    summary = {
        "member": ITALIA_XGB_MEMBER,
        "year": year,
        "n_parcels_features": n_parcels,
        "n_parcels_oof": result.n_parcels,
        "n_classes": result.n_classes,
        "class_ids": list(result.class_ids),
        "f1_macro_oof": round(result.f1_macro, 4),
        "accuracy_oof": round(result.accuracy, 4),
        "per_fold_f1": [round(f, 4) for f in result.per_fold_f1],
        "features_path": str(out_path),
        "oof_path": str(result.oof_path),
        "patches_metadata": str(patches_metadata or (dataset_dir / "metadata.parquet")),
    }

    run_id = "n/a"
    if use_mlflow:
        try:
            import mlflow

            from ml.utils.mlflow_utils import track_experiment

            with track_experiment(
                "us079-xgb-alphaearth-italia",
                run_name="xgb-alphaearth-italia",
                dvc_path=str(out_path),
            ) as active_run:
                mlflow.log_params(
                    {
                        "member": ITALIA_XGB_MEMBER,
                        "year": year,
                        "n_parcels": result.n_parcels,
                        "n_classes": result.n_classes,
                        "buffer_km": buffer_km,
                        "random_state": random_state,
                    }
                )
                mlflow.log_metrics(
                    {"f1_macro_oof": result.f1_macro, "accuracy_oof": result.accuracy}
                )
                run_id = active_run.info.run_id
        except Exception as exc:  # noqa: BLE001 - logging must never abort the run
            logger.warning("alphaearth_italia_mlflow_failed_skipping", error=str(exc))

    summary["mlflow_run_id"] = run_id
    typer.echo(json.dumps(summary, indent=2))
    typer.secho(
        f"OOF written: {result.oof_path} ({result.n_parcels} parcels, "
        f"{result.n_classes} classes, F1-macro OOF {result.f1_macro:.4f}). "
        f"DVC-track the features with: dvc add {out_path}",
        fg=typer.colors.GREEN,
    )


if __name__ == "__main__":
    app()
