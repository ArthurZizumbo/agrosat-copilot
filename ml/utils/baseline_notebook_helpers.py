"""Helpers DRY para los notebooks `notebooks/baseline/*.ipynb`.

Centraliza los patrones que se repiten en los 6 notebooks de baseline
(04_baseline, 04b_baseline, 04c_baseline, 04_farslip_eval_pastis,
05_reencuadre_fenologico, Avance3.Equipo17) para que cada notebook quede
como una composicion de llamadas + markdown + display, sin codigo inline.

Cubre:

- :func:`load_or_build_fused_features` — carga features fused con auto-build.
  Si `data/features/features_fused_italy.parquet` no existe, construye
  desde `data/processed/pastis_parcels_full.geoparquet` con
  :func:`ml.features.fusion.build_fused_features`.
- :func:`load_features_dataset_with_meta` — alias seguro del subset US-018.
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

from ml.utils.parcel_id import canonical_parcel_id

logger = structlog.get_logger(__name__)

__all__ = [
    "ModelComparisonRow",
    "build_model_comparison_table",
    "load_features_dataset_with_meta",
    "load_or_build_fused_features",
    "materialize_pastis_eval_subset_if_missing",
    "materialize_phenology_text_if_missing",
    "materialize_remoteclip_if_missing",
    "materialize_s2_anchors_if_missing",
    "materialize_spectral_signature_if_missing",
    "run_ablation_and_persist",
    "train_baseline_three_models",
]


# ---------------------------------------------------------------------------
# Carga / construccion de features.
# ---------------------------------------------------------------------------


_DEFAULT_SUBSET_PATH = Path("data/test_fixtures/feature_selection_parcels_subset.parquet")
_DEFAULT_PARCELS_PATH = Path("data/processed/pastis_parcels_full.geoparquet")
_DEFAULT_FUSED_PATH = Path("data/features/features_fused_italy.parquet")


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
    meta_cols = [
        c for c in ("parcel_id", "patch_id", "instance_id", "class_name", "fold", "area_m2", "n_pixels")
        if c in parcels_gdf.columns
    ]
    parcels_meta = pl.from_pandas(parcels_gdf[meta_cols])
    parcels_meta = canonical_parcel_id(parcels_meta)

    enriched = features.join(parcels_meta, on="parcel_id", how="left")
    logger.info(
        "features_loaded_with_meta",
        features_shape=features.shape,
        enriched_shape=enriched.shape,
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
        output_path: Ruta al parquet fused (default
            `data/features/features_fused_italy.parquet`).
        parcels_geoparquet: Geoparquet de parcelas Italia full.
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
# Entrenamiento de los 3 modelos baseline.
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


# ---------------------------------------------------------------------------
# Auto-materializacion de bloques opcionales.
# ---------------------------------------------------------------------------


def materialize_phenology_text_if_missing(
    parcels_features_path: Path | str,
    *,
    output_path: Path | str = Path("data/features/phenology_text_italy.parquet"),
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
    output_path: Path | str = Path("data/features/s2_anchors_italy.parquet"),
    year: int = 2023,
) -> Path:
    """Materializa el bloque `{anchor}_b04..b08` si el parquet no existe.

    Wrapper sobre :func:`ml.ingest.s2_anchor_sampler.sample_s2_anchors_for_parcels`.

    Args:
        parcels_geoparquet: Geoparquet de parcelas Italia full.
        output_path: Path destino del bloque S2 anchors.
        year: Anio para el muestreo GEE.

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

    return sample_s2_anchors_for_parcels(
        parcels=parcels,
        year=year,
        output_path=output,
    )


def materialize_spectral_signature_if_missing(
    *,
    s2_anchors_path: Path | str = Path("data/features/s2_anchors_italy.parquet"),
    output_path: Path | str = Path("data/features/spectral_signature_italy.parquet"),
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

    anchors_path = Path(s2_anchors_path)
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
# Ablation runner que persiste tabla + figures.
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
