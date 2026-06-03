"""Helpers DRY para los notebooks `notebooks/baseline/*.ipynb`.

Centraliza los patrones que se repiten en los 6 notebooks de baseline
(04_baseline, 04b_baseline, 04c_baseline, 04_farslip_eval_pastis,
05_reencuadre_fenologico, Avance3.Equipo17) para que cada notebook quede
como una composicion de llamadas + markdown + display, sin codigo inline.

Cubre:

- :func:`load_or_build_fused_features` — carga features fused con auto-build.
  Si `data/features/features_fused_pastis.parquet` no existe (ni su variante
  legacy `_italy`), construye desde
  `data/processed/pastis_parcels_full.geoparquet` con
  :func:`ml.features.fusion.build_fused_features`.
- :func:`load_features_dataset_with_meta` — alias seguro del subset US-018.
- :func:`load_base_plus_alphaearth_2018_2019` — base + AlphaEarth 2018
  (`ae18_NN`) + AlphaEarth 2019 (`ae19_NN`), el escenario ganador de la
  ablacion (`base_plus_ae18_ae19`).
- :func:`train_baseline_three_models` — entrena RF + XGB + LGBM con spatial CV.
- :func:`build_model_comparison_table` — DataFrame Polars con 5 metricas x N modelos.
- :func:`materialize_phenology_text_if_missing` — auto-genera bloque pheno_text.
- :func:`materialize_s2_anchors_if_missing` — auto-genera bloque S2 anchors.
- :func:`materialize_spectral_signature_if_missing` — auto-genera firma espectral.
- :func:`materialize_pastis_eval_subset_if_missing` — auto-genera PASTIS subset.
- :func:`materialize_remoteclip_if_missing` — auto-genera RemoteCLIP embeddings.
- :func:`run_ablation_and_persist` — ejecuta feature_ablation + persiste tabla.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl
import structlog

from ml.utils.dataset_paths import resolve_dataset_path
from ml.utils.parcel_id import canonical_parcel_id

logger = structlog.get_logger(__name__)

__all__ = [
    "ModelComparisonRow",
    "build_model_comparison_table",
    "load_base_plus_alphaearth_2018_2019",
    "load_features_dataset_with_meta",
    "load_or_build_fused_features",
    "load_temporal_result_from_mlflow",
    "materialize_pastis_eval_subset_if_missing",
    "materialize_phenology_text_if_missing",
    "materialize_remoteclip_if_missing",
    "materialize_s2_anchors_if_missing",
    "materialize_spectral_signature_if_missing",
    "run_ablation_and_persist",
    "train_baseline_three_models",
]


# ---------------------------------------------------------------------------
# Feature loading / construction.
# ---------------------------------------------------------------------------


_DEFAULT_SUBSET_PATH = Path("data/test_fixtures/feature_selection_parcels_subset.parquet")
_DEFAULT_PARCELS_PATH = Path("data/processed/pastis_parcels_full.geoparquet")
_DEFAULT_FUSED_PATH = Path("data/features/features_fused_pastis.parquet")
_DEFAULT_AE18_PATH = Path("data/cache/gee/alphaearth_parcels_parcels_2018_85951.parquet")
_DEFAULT_AE19_PATH = Path(
    "data/cache/gee/alphaearth_parcels_pastis_parcels_2019_85951.parquet"
)


def load_features_dataset_with_meta(
    path: Path | str = _DEFAULT_SUBSET_PATH,
    *,
    parcels_geoparquet: Path | str = _DEFAULT_PARCELS_PATH,
) -> pl.DataFrame:
    """Carga el subset de features US-018 y le adjunta metadata real.

    El subset original (`feature_selection_parcels_subset.parquet`) trae
    `parcel_id`, `year`, `class_id` y features espectrales/fenologicas,
    pero NO trae `patch_id` ni `class_name`. Esta funcion hace un LEFT JOIN
    con `pastis_parcels_full.geoparquet` para anexar esas columnas
    necesarias para el spatial CV y los reportes.

    Args:
        path: Ruta al parquet de features (default subset US-018).
        parcels_geoparquet: Ruta al geoparquet de parcelas PASTIS-R full.

    Returns:
        DataFrame Polars con `parcel_id` Utf8, `year`, `class_id`,
        `class_name`, `patch_id`, `fold`, mas todas las columnas de features.

    Raises:
        FileNotFoundError: si alguno de los dos archivos no existe.
    """
    features_path = Path(path)
    parcels_path = Path(parcels_geoparquet)
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features parquet no encontrado en {features_path}. "
            "Ejecuta los pipelines de EPIC 3 (US-013..US-018) primero."
        )
    if not parcels_path.exists():
        raise FileNotFoundError(
            f"Parcelas geoparquet no encontrado en {parcels_path}. "
            "Ejecuta `make build-parcels-geoparquet`."
        )

    import geopandas as gpd

    features = pl.read_parquet(features_path)
    features = canonical_parcel_id(features)

    parcels_gdf = gpd.read_parquet(parcels_path)
    _candidate_meta = (
        "parcel_id", "patch_id", "instance_id", "class_name", "fold", "area_m2", "n_pixels"
    )
    meta_cols = [c for c in _candidate_meta if c in parcels_gdf.columns]
    parcels_meta = pl.from_pandas(parcels_gdf[meta_cols])
    parcels_meta = canonical_parcel_id(parcels_meta)

    # If the features parquet already carries any of the meta columns (by
    # construction of the US-018 subset), we drop them from `parcels_meta` before
    # the join. Otherwise Polars suffixes with `_right` and those numeric
    # columns (patch_id, fold, n_pixels) end up in matrix X as features
    # — spatial leakage that the SHAP of 04_baseline exposed.
    overlap = [c for c in parcels_meta.columns if c != "parcel_id" and c in features.columns]
    if overlap:
        parcels_meta = parcels_meta.drop(overlap)
        logger.info("features_meta_overlap_dropped", overlap=overlap)

    enriched = features.join(parcels_meta, on="parcel_id", how="left")
    logger.info(
        "features_loaded_with_meta",
        features_shape=features.shape,
        enriched_shape=enriched.shape,
    )
    return enriched


def load_base_plus_alphaearth_2018_2019(
    *,
    features_path: Path | str = _DEFAULT_SUBSET_PATH,
    parcels_geoparquet: Path | str = _DEFAULT_PARCELS_PATH,
    alphaearth_2018_path: Path | str = _DEFAULT_AE18_PATH,
    alphaearth_2019_path: Path | str = _DEFAULT_AE19_PATH,
) -> pl.DataFrame:
    """Carga el escenario ganador: base + AlphaEarth 2018 + AlphaEarth 2019.

    Parte del subset US-018 con metadata real (185 features base) y le anexa
    los dos embeddings AlphaEarth anuales de 64 dimensiones cada uno: 2018
    (columnas ``ae18_00..ae18_63``) y 2019 (columnas ``ae19_00..ae19_63``),
    uniendo por ``parcel_id`` (join 1:1, mismo universo de 85951 parcelas).
    El resultado es el escenario ``base_plus_ae18_ae19`` que maximizo el
    F1-macro en la ablacion de escenarios.

    Los parquets de AlphaEarth traen las dimensiones como ``dim_00..dim_63``;
    se renombran a ``ae18_NN`` / ``ae19_NN`` para que ambos anios coexistan en
    la misma matriz de features sin colision de nombres.

    Args:
        features_path: Ruta al parquet de features base (subset US-018).
        parcels_geoparquet: Geoparquet de parcelas PASTIS-R full (metadata).
        alphaearth_2018_path: Parquet AlphaEarth 2018 con ``dim_NN``.
        alphaearth_2019_path: Parquet AlphaEarth 2019 con ``dim_NN``.

    Returns:
        DataFrame Polars con las features base + 64 columnas ``ae18_NN`` + 64
        columnas ``ae19_NN``, mas la metadata (``parcel_id``, ``class_id``,
        ``patch_id``, ``class_name``, ``fold``).

    Raises:
        FileNotFoundError: si falta alguno de los parquets de AlphaEarth.
    """
    base = load_features_dataset_with_meta(
        path=features_path, parcels_geoparquet=parcels_geoparquet
    )
    base = canonical_parcel_id(base)

    def _load_alphaearth(path: Path | str, prefix: str) -> pl.DataFrame:
        ae_path = Path(path)
        if not ae_path.exists():
            raise FileNotFoundError(
                f"AlphaEarth parquet no encontrado en {ae_path}. "
                "Ejecuta el pipeline GEE (US-012) o `dvc pull` del cache."
            )
        ae = canonical_parcel_id(pl.read_parquet(ae_path))
        dim_cols = [c for c in ae.columns if c.startswith("dim_")]
        rename = {c: f"{prefix}_{c.removeprefix('dim_')}" for c in dim_cols}
        return ae.select(["parcel_id", *dim_cols]).rename(rename)

    ae18 = _load_alphaearth(alphaearth_2018_path, "ae18")
    ae19 = _load_alphaearth(alphaearth_2019_path, "ae19")

    enriched = base.join(ae18, on="parcel_id", how="left").join(
        ae19, on="parcel_id", how="left"
    )
    n_ae18 = sum(1 for c in enriched.columns if c.startswith("ae18_"))
    n_ae19 = sum(1 for c in enriched.columns if c.startswith("ae19_"))
    n_null_ae18 = int(enriched.select(pl.col("ae18_00").is_null().sum()).item())
    n_null_ae19 = int(enriched.select(pl.col("ae19_00").is_null().sum()).item())
    logger.info(
        "base_plus_ae18_ae19_loaded",
        base_shape=base.shape,
        enriched_shape=enriched.shape,
        n_ae18=n_ae18,
        n_ae19=n_ae19,
        n_null_ae18=n_null_ae18,
        n_null_ae19=n_null_ae19,
    )
    return enriched


def load_or_build_fused_features(
    output_path: Path | str = _DEFAULT_FUSED_PATH,
    *,
    parcels_geoparquet: Path | str = _DEFAULT_PARCELS_PATH,
    year: int = 2023,
    overwrite: bool = False,
    include_farslip: bool = True,
    include_phenology_text: bool = False,
    include_spectral_signature: bool = False,
) -> pl.DataFrame:
    """Carga el parquet de features fused; lo construye si no existe.

    Si `output_path` existe y `overwrite=False`, lo lee y devuelve. En caso
    contrario invoca :func:`ml.features.fusion.build_fused_features` sobre
    las parcelas full y persiste el resultado.

    Args:
        output_path: Ruta al parquet fused (default canonico
            `data/features/features_fused_pastis.parquet`; al usar el default
            se resuelve via :func:`resolve_dataset_path`, que cae al legacy
            `_italy` si ya esta materializado en disco). El contenido es
            PASTIS-R frances, no italiano.
        parcels_geoparquet: Geoparquet de parcelas PASTIS-R full.
        year: Anio de referencia para los muestreos GEE.
        overwrite: Si True regenera el parquet aunque exista.
        include_farslip: Si True incluye el bloque FarSLIP.
        include_phenology_text: Si True incluye el bloque pheno_text.
        include_spectral_signature: Si True incluye la firma espectral.

    Returns:
        DataFrame Polars con todas las columnas de features.

    Raises:
        FileNotFoundError: si el geoparquet de parcelas no existe.
    """
    # Read: if the canonical default was used, resolve to the existing
    # variant (`_pastis` or legacy `_italy`). If the caller passed an explicit
    # path, it is respected as-is.
    if output_path is _DEFAULT_FUSED_PATH:
        output = resolve_dataset_path(_DEFAULT_FUSED_PATH)
    else:
        output = Path(output_path)
    if output.exists() and not overwrite:
        logger.info("fused_features_cache_hit", path=str(output))
        return pl.read_parquet(output)

    parcels_path = Path(parcels_geoparquet)
    if not parcels_path.exists():
        raise FileNotFoundError(
            f"Parcelas geoparquet no encontrado en {parcels_path}."
        )

    import geopandas as gpd

    from ml.features.fusion import build_fused_features

    parcels = gpd.read_parquet(parcels_path)
    parcels["parcel_id"] = parcels["parcel_id"].astype(str)
    if "year" not in parcels.columns:
        parcels["year"] = year

    logger.info(
        "fused_features_building",
        n_parcels=len(parcels),
        year=year,
        include_farslip=include_farslip,
        include_phenology_text=include_phenology_text,
        include_spectral_signature=include_spectral_signature,
    )
    fused = build_fused_features(
        parcels=parcels,
        year=year,
        include_farslip=include_farslip,
        include_phenology_text=include_phenology_text,
        include_spectral_signature=include_spectral_signature,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fused.write_parquet(output)
    logger.info("fused_features_persisted", path=str(output), shape=fused.shape)
    return fused


# ---------------------------------------------------------------------------
# Training of the 3 baseline models.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelComparisonRow:
    """Una fila de la tabla comparativa de modelos."""

    model: str
    f1_macro: float
    f1_weighted: float
    miou: float
    accuracy: float
    cohen_kappa: float
    train_time_s: float
    n_features: int
    n_samples: int


def train_baseline_three_models(
    df: pl.DataFrame,
    *,
    models: tuple[Literal["rf", "xgb", "lgbm"], ...] = ("rf", "xgb", "lgbm"),
    k_folds: int = 5,
    buffer_km: float = 1.0,
    random_state: int = 42,
) -> list[ModelComparisonRow]:
    """Entrena los 3 modelos baseline tabulares con spatial CV y devuelve metricas.

    Args:
        df: DataFrame con features + `parcel_id`, `class_id`, `patch_id`.
        models: Tupla de modelos a entrenar. Soporta `"rf"`, `"xgb"`, `"lgbm"`.
        k_folds: Folds del CV espacial.
        buffer_km: Buffer anti-leakage en km.
        random_state: Semilla.

    Returns:
        Lista de :class:`ModelComparisonRow` con metricas + train_time.
    """
    import time

    from ml.train.baseline import train_one_model

    rows: list[ModelComparisonRow] = []
    for model_kind in models:
        t0 = time.perf_counter()
        result = train_one_model(
            df,
            model=model_kind,
            k_folds=k_folds,
            buffer_km=buffer_km,
            random_state=random_state,
        )
        elapsed = time.perf_counter() - t0
        rows.append(
            ModelComparisonRow(
                model=model_kind,
                f1_macro=float(result.metrics["f1_macro"]),
                f1_weighted=float(result.metrics["f1_weighted"]),
                miou=float(result.metrics["miou"]),
                accuracy=float(result.metrics["accuracy"]),
                cohen_kappa=float(result.metrics["cohen_kappa"]),
                train_time_s=elapsed,
                n_features=len(result.feature_cols),
                n_samples=df.height,
            )
        )
        logger.info(
            "baseline_model_done",
            model=model_kind,
            f1_macro=round(rows[-1].f1_macro, 4),
            train_time_s=round(elapsed, 1),
        )
    return rows


def build_model_comparison_table(
    rows: Sequence[ModelComparisonRow],
    *,
    output_path: Path | str | None = None,
) -> pl.DataFrame:
    """Convierte filas de comparacion en DataFrame Polars y opcionalmente persiste.

    Args:
        rows: Lista de :class:`ModelComparisonRow`.
        output_path: Si no es None, persiste como parquet en esa ruta.

    Returns:
        DataFrame Polars ordenado por `f1_macro` descendente.
    """
    table = pl.DataFrame(
        [
            {
                "model": r.model,
                "f1_macro": round(r.f1_macro, 4),
                "f1_weighted": round(r.f1_weighted, 4),
                "miou": round(r.miou, 4),
                "accuracy": round(r.accuracy, 4),
                "cohen_kappa": round(r.cohen_kappa, 4),
                "train_time_s": round(r.train_time_s, 1),
                "n_features": r.n_features,
                "n_samples": r.n_samples,
            }
            for r in rows
        ]
    ).sort("f1_macro", descending=True)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.write_parquet(out)
        logger.info("model_comparison_persisted", path=str(out))
    return table


def load_temporal_result_from_mlflow(
    model_kind: Literal["tempcnn", "inceptiontime"],
    *,
    experiment_name: str = "baseline-05-reencuadre",
    tracking_uri: str = "http://localhost:5010",
):
    """Reconstruye un TemporalModelResult desde un MLflow run ya finalizado.

    Evita re-entrenar TempCNN/InceptionTime cuando ya hay una corrida con
    metricas registradas. Lee la run mas reciente con `params.model_kind`
    igual a `model_kind` y `status=FINISHED`, y reconstruye el dataclass
    de salida usando las metricas `oof_*` y `params.n_classes`.

    Args:
        model_kind: ``"tempcnn"`` o ``"inceptiontime"``.
        experiment_name: Nombre del experimento MLflow.
        tracking_uri: URI del tracking server.

    Returns:
        :class:`ml.train.phenology_models.TemporalModelResult` con las metricas
        out-of-fold reconstruidas, ``y_true_oof`` y ``y_pred_oof`` vacios y
        ``checkpoint_path`` apuntando al artifact si esta disponible.

    Raises:
        ValueError: si no hay run FINISHED del kind solicitado.
    """
    import mlflow

    from ml.train.phenology_models import TemporalModelResult

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"experimento `{experiment_name}` no existe en {tracking_uri}.")

    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"params.model_kind = '{model_kind}' and attributes.status = 'FINISHED'",
        max_results=1,
        order_by=["attributes.start_time DESC"],
    )
    if not runs:
        raise ValueError(
            f"no hay runs FINISHED con model_kind=`{model_kind}` en `{experiment_name}`. "
            "Re-entrena con train_temporal_model o ajusta la consulta."
        )
    run = runs[0]
    metrics = run.data.metrics
    params = run.data.params

    n_parcels = int(params.get("n_parcels", 0))
    n_classes = int(params.get("n_classes", 18))
    train_time_s = float(metrics.get("train_time_s", 0.0))

    logger.info(
        "temporal_result_loaded_from_mlflow",
        model_kind=model_kind,
        run_id=run.info.run_id[:12],
        oof_f1_macro=round(float(metrics.get("oof_f1_macro", 0.0)), 4),
    )

    return TemporalModelResult(
        model_kind=model_kind,
        f1_macro=float(metrics.get("oof_f1_macro", 0.0)),
        f1_weighted=float(metrics.get("oof_f1_weighted", 0.0)),
        miou=float(metrics.get("oof_miou", 0.0)),
        cohen_kappa=float(metrics.get("oof_cohen_kappa", 0.0)),
        train_time_s=train_time_s,
        n_parcels=n_parcels,
        n_classes=n_classes,
        mlflow_run_id=run.info.run_id,
    )


# ---------------------------------------------------------------------------
# Auto-materialization of optional blocks.
# ---------------------------------------------------------------------------


def materialize_phenology_text_if_missing(
    parcels_features_path: Path | str,
    *,
    output_path: Path | str = Path("data/features/phenology_text_pastis.parquet"),
    enforce_api_key: bool = True,
    max_parcels: int | None = None,
) -> Path:
    """Materializa el bloque `pheno_text_*` si el parquet no existe.

    Wrapper sobre :func:`ml.utils.phenology_text.materialize_phenology_text`.
    Idempotente: si `output_path` existe, no llama Gemini.

    Args:
        parcels_features_path: Ruta al parquet de features con `parcel_id`,
            `class_id` y columnas NDVI temporales.
        output_path: Path destino del bloque (parquet).
        enforce_api_key: Si True (default) raise RuntimeError sin Gemini.
        max_parcels: Limita el numero de parcelas (None = todas).

    Returns:
        Path del parquet generado o existente.
    """
    from ml.utils.phenology_text import materialize_phenology_text

    return materialize_phenology_text(
        parcels_features_path=parcels_features_path,
        output_path=Path(output_path),
        enforce_api_key=enforce_api_key,
        max_parcels=max_parcels,
        overwrite=False,
    )


def materialize_s2_anchors_if_missing(
    parcels_geoparquet: Path | str,
    *,
    output_path: Path | str = Path("data/features/s2_anchors_pastis.parquet"),
    year: int = 2023,
    phenology_anchors_path: Path | str | None = None,
) -> Path:
    """Materializa el bloque `{anchor}_b04..b08` si el parquet no existe.

    Wrapper sobre :func:`ml.ingest.s2_anchor_sampler.sample_s2_anchors_for_parcels`.

    Args:
        parcels_geoparquet: Geoparquet de parcelas PASTIS-R full.
        output_path: Path destino del bloque S2 anchors.
        year: Anio para el muestreo GEE.
        phenology_anchors_path: Parquet opcional con anclas calendario por
            parcela (schema: ``parcel_id, sog_doy, peak_doy, senescence_doy``).
            Si se provee, el sampler usa DOY especifico por parcela y evita
            el warning ``phenology_anchors_fallback_static``. Generar con
            :func:`ml.ingest.pastis_phenology_anchors.build_pastis_phenology_anchors`
            para PASTIS-R.

    Returns:
        Path del parquet generado o existente.
    """
    output = Path(output_path)
    if output.exists():
        logger.info("s2_anchors_cache_hit", path=str(output))
        return output

    import geopandas as gpd

    from ml.ingest.s2_anchor_sampler import sample_s2_anchors_for_parcels

    parcels = gpd.read_parquet(Path(parcels_geoparquet))
    parcels["parcel_id"] = parcels["parcel_id"].astype(str)
    if "year" not in parcels.columns:
        parcels["year"] = year

    anchors_path: Path | None = (
        Path(phenology_anchors_path) if phenology_anchors_path is not None else None
    )

    return sample_s2_anchors_for_parcels(
        parcels=parcels,
        year=year,
        output_path=output,
        phenology_anchors_path=anchors_path,
    )


def materialize_spectral_signature_if_missing(
    *,
    s2_anchors_path: Path | str = Path("data/features/s2_anchors_pastis.parquet"),
    output_path: Path | str = Path("data/features/spectral_signature_pastis.parquet"),
    descriptor: Literal["rep", "sam", "redge_moments"] = "rep",
) -> Path:
    """Materializa la firma espectral si no existe, desde anclas S2 ya muestreadas.

    Args:
        s2_anchors_path: Path al parquet de anclas S2 (debe existir; si no,
            invocar `materialize_s2_anchors_if_missing` primero).
        output_path: Path destino del bloque `spectral_signature_*`.
        descriptor: Tipo de descriptor (default `"rep"`, Frampton 2013).

    Returns:
        Path del parquet generado o existente.

    Raises:
        FileNotFoundError: si las anclas S2 no estan en disco.
    """
    output = Path(output_path)
    if output.exists():
        logger.info("spectral_signature_cache_hit", path=str(output))
        return output

    # Read the S2 anchors block: resolve to the existing variant
    # (`_pastis` canonical or legacy `_italy`) to avoid re-sampling if the
    # artifact is already on disk under the inherited name.
    anchors_path = resolve_dataset_path(s2_anchors_path)
    if not anchors_path.exists():
        raise FileNotFoundError(
            f"S2 anchors no encontrado en {anchors_path}. Ejecuta "
            "materialize_s2_anchors_if_missing antes."
        )

    from ml.features.spectral_signature import SpectralSignatureFeatures

    anchors = pl.read_parquet(anchors_path)
    anchors = canonical_parcel_id(anchors)
    transformer = SpectralSignatureFeatures(descriptor=descriptor)
    signature = transformer.fit_transform(anchors)

    output.parent.mkdir(parents=True, exist_ok=True)
    signature.write_parquet(output)
    logger.info(
        "spectral_signature_persisted",
        path=str(output),
        shape=signature.shape,
        descriptor=descriptor,
    )
    return output


def materialize_pastis_eval_subset_if_missing(
    *,
    output_path: Path | str = Path("data/test_fixtures/pastis_eval_subset.parquet"),
    n_samples: int = 1024,
) -> Path:
    """Materializa el subset PASTIS-R real si no existe.

    Wrapper sobre :func:`ml.ingest.pastis_eval_subset.build_pastis_eval_subset`.
    """
    output = Path(output_path)
    if output.exists():
        logger.info("pastis_eval_subset_cache_hit", path=str(output))
        return output

    from ml.ingest.pastis_eval_subset import build_pastis_eval_subset

    return build_pastis_eval_subset(
        output_path=output,
        n_samples=n_samples,
        overwrite=False,
        save_imagery=True,
    )


def materialize_remoteclip_if_missing(
    *,
    pastis_eval_subset_path: Path | str = Path(
        "data/test_fixtures/pastis_eval_subset.parquet"
    ),
    imagery_path: Path | str = Path(
        "data/test_fixtures/pastis_eval_subset.imagery.parquet"
    ),
    output_path: Path | str = Path("data/farslip/remoteclip_embeddings_pastis.parquet"),
) -> Path:
    """Materializa embeddings RemoteCLIP sobre el subset PASTIS si no existen."""
    output = Path(output_path)
    if output.exists():
        logger.info("remoteclip_cache_hit", path=str(output))
        return output

    from ml.ingest.remoteclip_extractor import extract_remoteclip_embeddings

    return extract_remoteclip_embeddings(
        pastis_eval_subset_path=Path(pastis_eval_subset_path),
        imagery_path=Path(imagery_path),
        output_path=output,
    )


# ---------------------------------------------------------------------------
# Ablation runner that persists table + figures.
# ---------------------------------------------------------------------------


def run_ablation_and_persist(
    df: pl.DataFrame,
    *,
    output_dir: Path | str = Path("reports/baseline/feature_ablation"),
    models: tuple[Literal["rf", "xgb", "lgbm"], ...] = ("xgb",),
    k_folds: int = 5,
    buffer_km: float = 1.0,
    max_samples: int | None = None,
) -> tuple[pl.DataFrame, Path]:
    """Ejecuta feature_ablation + persiste tabla parquet/csv/md.

    Args:
        df: DataFrame fused (debe incluir las columnas que se ablacionaran).
        output_dir: Carpeta destino.
        models: Modelos a ablacionar.
        k_folds: Folds del CV.
        buffer_km: Buffer anti-leakage.
        max_samples: Subsample uniforme para CI/dev. None = todos.

    Returns:
        Tupla `(tabla_polars, ruta_parquet)`.
    """
    from ml.eval.feature_ablation import (
        build_default_feature_sets,
        export_ablation_table,
        run_feature_ablation,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_sets = build_default_feature_sets(df.columns)
    results = run_feature_ablation(
        df=df,
        feature_sets=feature_sets,
        models=models,
        max_samples=max_samples,
        k_folds=k_folds,
        buffer_km=buffer_km,
    )

    stem = out_dir / "ablation_table"
    export_ablation_table(results, stem)
    parquet_path = stem.with_suffix(".parquet")
    table = pl.DataFrame(
        [
            {
                "feature_set": r.feature_set,
                "model": r.model_kind,
                "n_features": r.n_features,
                "f1_macro": r.f1_macro,
                "f1_weighted": r.f1_weighted,
                "miou": r.miou,
                "delta_vs_full": r.delta_vs_full,
            }
            for r in results
        ]
    )
    table.write_parquet(parquet_path)
    logger.info(
        "ablation_persisted",
        parquet=str(parquet_path),
        n_rows=table.height,
        models=models,
    )
    return table, parquet_path
