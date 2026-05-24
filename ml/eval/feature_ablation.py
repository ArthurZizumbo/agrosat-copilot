"""Ablation de bloques de features para el baseline (US-022b-C).

Construye la **matriz comparativa** del reencuadre fenologico:

- ``full`` — todas las features disponibles.
- ``no_geom`` — descarta ``geom_area_ha``, ``geom_perimeter_m``,
  ``geom_elongation`` (proxies geograficos, candidatos a leakage espacial).
- ``no_geom_no_era5_srtm`` — adicionalmente descarta los bloques ERA5
  (24 cols) y SRTM (3 cols), redundantes con AlphaEarth (que ya los
  codifica internamente).
- ``alphaearth_only`` — solo las 64 dimensiones ``ae_00..ae_63``.
- ``phenology_only`` — solo los 8 features fenologicos + 24 FFT
  (NDVI/NDWI/EVI).

Decisiones canonicas (plan ``docs/us-planning/us-022b.md`` §6.2):

- **D-ARQ-1**: NO reescribe ``fusion.py`` ni ``temporal_features.py`` — los
  consume. Recibe el DataFrame ya fusionado o el parquet del baseline.
- **Mismo spatial CV 5-fold** para todos los conjuntos (gracias al cache
  por ``n_rows + k + buffer + seed`` de ``_build_cv_splits``).
- **Reusa** ``ml.train.baseline.train_one_model`` (no reinventa training).
- **delta_vs_full** se calcula como ``F1-macro(set) - F1-macro(full)``
  para el mismo ``model_kind``. ``full`` es la referencia obligatoria; si
  no esta en ``feature_sets`` se levanta ``ValueError``.

El export ``export_ablation_table`` produce CSV + Markdown listos para el
notebook ``05_reencuadre_fenologico.ipynb`` y para el cierre del Avance 4.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "FeatureAblationResult",
    "build_default_feature_sets",
    "export_ablation_table",
    "run_feature_ablation",
]

#: Modelos soportados por la ablation (los temporales pasan por
#: :mod:`ml.train.phenology_models`; aqui solo se aceptan los tabulares
#: rapidos para que la matriz N x M corra en CPU sin GPU).
SupportedModel = Literal["rf", "xgb", "tempcnn", "inceptiontime"]

#: Columnas de metadata que jamas son features.
_META_COLS: frozenset[str] = frozenset(
    {
        "parcel_id",
        "year",
        "patch_id",
        "instance_id",
        "class_id",
        "class_name",
        "fold",
        "n_pixels",
        "area_m2",
        "geometry",
    }
)


@dataclass(frozen=True)
class FeatureAblationResult:
    """Resultado de un (feature_set, model) en la matriz de ablation.

    Attributes:
        feature_set: Etiqueta del conjunto (``full``, ``no_geom``, ...).
        model_kind: Modelo aplicado (``rf``, ``xgb``, ``tempcnn``,
            ``inceptiontime``).
        f1_macro: F1-macro out-of-fold del spatial CV.
        f1_weighted: F1 ponderado.
        miou: mIoU (Jaccard macro).
        n_features: Numero de features efectivas (las que existian en el
            DataFrame y eran numericas; las pedidas que no existian se
            ignoran con warning).
        delta_vs_full: ``f1_macro(set) - f1_macro(full)`` para el mismo
            modelo. ``nan`` si el set es ``full`` mismo o si ``full`` no
            esta presente.
    """

    feature_set: str
    model_kind: SupportedModel
    f1_macro: float
    f1_weighted: float
    miou: float
    n_features: int
    delta_vs_full: float


# ---------------------------------------------------------------------------
# Conjuntos de features por defecto (cuando el caller no aporta los suyos).
# ---------------------------------------------------------------------------


def build_default_feature_sets(
    available_cols: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """Construye los 5 conjuntos canonicos a partir de las columnas presentes.

    Args:
        available_cols: Todas las columnas presentes en el DataFrame
            fusionado (output de :func:`ml.features.fusion.build_fused_features`
            o del subset US-018).

    Returns:
        Mapping ``{nombre_set: (cols,)}`` con los 5 sets:

        - ``full``: todas las features numericas (excluyendo metadata).
        - ``no_geom``: ``full`` sin ``geom_*``.
        - ``no_geom_no_era5_srtm``: ``no_geom`` sin ``era5_*`` ni ``srtm_*``.
        - ``alphaearth_only``: solo ``ae_*`` o ``dim_*`` (acepta ambos
          nombres).
        - ``phenology_only``: 8 cols fenologicas + 24 FFT
          (``{idx}_fft_amp_k``, ``{idx}_fft_phase_k`` para
          ``idx in {NDVI, NDWI, EVI}``).
    """
    cols = [c for c in available_cols if c not in _META_COLS]

    full = tuple(cols)
    no_geom = tuple(c for c in cols if not c.startswith("geom_"))
    no_geom_no_era5_srtm = tuple(
        c for c in no_geom if not c.startswith("era5_") and not c.startswith("srtm_")
    )
    ae_cols = tuple(
        c
        for c in cols
        if (c.startswith("ae_") and len(c) == 5) or (c.startswith("dim_") and len(c) == 6)
    )
    pheno_cols_known = {
        "sog_doy",
        "peak_doy",
        "peak_value",
        "senescence_doy",
        "ndvi_auc",
        "ndvi_slope_pre_peak",
        "ndvi_slope_post_peak",
        "maturity_duration_days",
    }
    fft_cols = tuple(c for c in cols if "_fft_amp_" in c or "_fft_phase_" in c)
    pheno_cols = tuple(c for c in cols if c in pheno_cols_known) + fft_cols
    # Bloques opcionales (US-017 FarSLIP, US-022b-D rama semantica fenologica).
    farslip_cols = tuple(c for c in cols if c.startswith("farslip_"))
    pheno_text_cols = tuple(c for c in cols if c.startswith("pheno_text_"))

    sets: dict[str, tuple[str, ...]] = {
        "full": full,
        "no_geom": no_geom,
        "no_geom_no_era5_srtm": no_geom_no_era5_srtm,
        "alphaearth_only": ae_cols,
        "phenology_only": pheno_cols,
    }
    # Solo agrega los conjuntos with_* si las columnas correspondientes
    # estan materializadas en el DataFrame (graceful degradation).
    if farslip_cols:
        sets["with_farslip"] = pheno_cols + farslip_cols
        sets["farslip_only"] = farslip_cols
    if pheno_text_cols:
        sets["with_pheno_text"] = pheno_cols + pheno_text_cols
    return sets


# ---------------------------------------------------------------------------
# API publica.
# ---------------------------------------------------------------------------


def run_feature_ablation(
    features_path: Path | str | None = None,
    *,
    df: pl.DataFrame | None = None,
    feature_sets: dict[str, tuple[str, ...]] | None = None,
    models: tuple[SupportedModel, ...] = ("xgb",),
    max_samples: int | None = None,
    seed: int = 42,
    k_folds: int = 5,
    buffer_km: float = 1.0,
) -> list[FeatureAblationResult]:
    """Ejecuta la ablation: entrena cada modelo sobre cada conjunto de features.

    Para cada par ``(feature_set, model)`` entrena un baseline con el mismo
    spatial CV 5-fold (cacheado) y registra F1-macro + F1-weighted + mIoU +
    ``n_features``. El ``delta_vs_full`` se calcula al final, una vez que
    todos los runs terminaron.

    Args:
        features_path: Ruta al parquet de features fusionadas. Si ``df`` se
            pasa, se ignora.
        df: DataFrame Polars ya cargado.
        feature_sets: Mapping ``{nombre: (cols,)}`` con los conjuntos a
            ablacionar. Debe incluir la clave ``"full"``. Si es ``None`` se
            construyen con :func:`build_default_feature_sets`.
        models: Modelos a aplicar. ``"rf"`` y ``"xgb"`` van a
            :func:`ml.train.baseline.train_one_model`; ``"tempcnn"`` y
            ``"inceptiontime"`` van a
            :func:`ml.train.phenology_models.train_temporal_model` (solo
            si el set incluye al menos un indice temporal reconstructible).
        max_samples: Subsample uniforme determinista (CI/dev). ``None`` =
            dataset completo.
        seed: Semilla determinista.
        k_folds: Numero de folds del CV espacial.
        buffer_km: Buffer anti-leakage en km.

    Returns:
        Lista de :class:`FeatureAblationResult`, una por cada par
        ``(set, model)`` con muestras suficientes.

    Raises:
        ValueError: si ``df`` y ``features_path`` son ambos ``None`` o si
            ``feature_sets`` no contiene la clave ``"full"``.
    """
    if df is None:
        if features_path is None:
            raise ValueError("Debes pasar `features_path` o `df`.")
        df = pl.read_parquet(Path(features_path))

    if max_samples is not None and max_samples > 0 and df.height > max_samples:
        df = df.sample(n=max_samples, seed=seed, with_replacement=False)
        logger.info("ablation_subsampled", max_samples=max_samples, n=df.height)

    if feature_sets is None:
        feature_sets = build_default_feature_sets(df.columns)
    if "full" not in feature_sets:
        raise ValueError(
            "`feature_sets` debe incluir la clave 'full' (referencia para delta_vs_full)."
        )

    logger.info(
        "ablation_start",
        n_sets=len(feature_sets),
        models=models,
        n_rows=df.height,
    )

    raw_results: list[FeatureAblationResult] = []
    for set_name, requested_cols in feature_sets.items():
        present_cols = tuple(c for c in requested_cols if c in df.columns)
        missing = [c for c in requested_cols if c not in df.columns]
        if missing:
            logger.warning(
                "ablation_missing_cols",
                feature_set=set_name,
                n_missing=len(missing),
                first_missing=missing[:5],
            )
        if not present_cols:
            logger.warning("ablation_set_empty", feature_set=set_name)
            for model_kind in models:
                raw_results.append(
                    FeatureAblationResult(
                        feature_set=set_name,
                        model_kind=model_kind,
                        f1_macro=float("nan"),
                        f1_weighted=float("nan"),
                        miou=float("nan"),
                        n_features=0,
                        delta_vs_full=float("nan"),
                    )
                )
            continue

        meta_cols = [c for c in ("parcel_id", "year", "patch_id", "class_id") if c in df.columns]
        # Mantenemos solo metadata + las columnas pedidas. _META_COLS extra
        # como `fold`, `instance_id` se conservan si existen (train_one_model
        # las ignora porque no son numericas o estan en su lista negra).
        keep = meta_cols + [c for c in present_cols if c not in meta_cols]
        subset_df = df.select(keep)

        for model_kind in models:
            try:
                f1_macro, f1_weighted, miou, n_feats = _train_single(
                    subset_df,
                    model_kind=model_kind,
                    k_folds=k_folds,
                    buffer_km=buffer_km,
                    seed=seed,
                )
            except (ValueError, RuntimeError) as exc:  # pragma: no cover - safety net
                logger.warning(
                    "ablation_train_failed",
                    feature_set=set_name,
                    model_kind=model_kind,
                    error=str(exc),
                )
                f1_macro = f1_weighted = miou = float("nan")
                n_feats = len(present_cols)
            raw_results.append(
                FeatureAblationResult(
                    feature_set=set_name,
                    model_kind=model_kind,
                    f1_macro=f1_macro,
                    f1_weighted=f1_weighted,
                    miou=miou,
                    n_features=n_feats,
                    delta_vs_full=float("nan"),  # se rellena en el segundo pass
                )
            )
            logger.info(
                "ablation_cell_done",
                feature_set=set_name,
                model_kind=model_kind,
                f1_macro=round(f1_macro, 4) if not np.isnan(f1_macro) else None,
                n_features=n_feats,
            )

    # Segundo pass: rellena delta_vs_full por modelo.
    f1_full_by_model: dict[str, float] = {}
    for r in raw_results:
        if r.feature_set == "full":
            f1_full_by_model[r.model_kind] = r.f1_macro

    results: list[FeatureAblationResult] = []
    for r in raw_results:
        ref = f1_full_by_model.get(r.model_kind)
        if r.feature_set == "full" or ref is None or np.isnan(ref) or np.isnan(r.f1_macro):
            delta = float("nan")
        else:
            delta = r.f1_macro - ref
        results.append(
            FeatureAblationResult(
                feature_set=r.feature_set,
                model_kind=r.model_kind,
                f1_macro=r.f1_macro,
                f1_weighted=r.f1_weighted,
                miou=r.miou,
                n_features=r.n_features,
                delta_vs_full=delta,
            )
        )

    logger.info("ablation_done", n_rows_output=len(results))
    return results


def export_ablation_table(
    results: Sequence[FeatureAblationResult],
    output_path: Path | str,
) -> tuple[Path, Path]:
    """Persiste la tabla de ablation en CSV + Markdown.

    Args:
        results: Lista de :class:`FeatureAblationResult`.
        output_path: Ruta destino (sin extension o ``.csv``). Se generan
            ``<stem>.csv`` y ``<stem>.md`` en el mismo directorio.

    Returns:
        Tupla ``(csv_path, md_path)`` con las rutas escritas.
    """
    csv_path = Path(output_path).with_suffix(".csv")
    md_path = Path(output_path).with_suffix(".md")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    table = pl.DataFrame(
        [
            {
                "feature_set": r.feature_set,
                "model": r.model_kind,
                "n_features": r.n_features,
                "f1_macro": round(r.f1_macro, 4),
                "f1_weighted": round(r.f1_weighted, 4),
                "miou": round(r.miou, 4),
                "delta_vs_full": (
                    round(r.delta_vs_full, 4) if not np.isnan(r.delta_vs_full) else None
                ),
            }
            for r in results
        ],
        schema={
            "feature_set": pl.Utf8,
            "model": pl.Utf8,
            "n_features": pl.Int64,
            "f1_macro": pl.Float64,
            "f1_weighted": pl.Float64,
            "miou": pl.Float64,
            "delta_vs_full": pl.Float64,
        },
    )
    table.write_csv(csv_path)
    md_body = (
        "# Ablation de features — reencuadre fenologico (US-022b-C)\n\n"
        + table.to_pandas().to_markdown(index=False)
        + "\n"
    )
    md_path.write_text(md_body, encoding="utf-8")
    logger.info("ablation_table_exported", csv=str(csv_path), md=str(md_path))
    return csv_path, md_path


# ---------------------------------------------------------------------------
# Helpers privados.
# ---------------------------------------------------------------------------


def _train_single(
    df: pl.DataFrame,
    *,
    model_kind: SupportedModel,
    k_folds: int,
    buffer_km: float,
    seed: int,
) -> tuple[float, float, float, int]:
    """Entrena un modelo y devuelve ``(f1_macro, f1_weighted, miou, n_feats)``.

    Import diferido: rompe el ciclo ``baseline -> eval.metrics`` y
    ``eval.__init__ -> feature_ablation``.
    """
    if model_kind in ("rf", "xgb"):
        from ml.train.baseline import train_one_model

        tabular_result = train_one_model(
            df,
            model=model_kind,
            k_folds=k_folds,
            buffer_km=buffer_km,
            random_state=seed,
        )
        return (
            float(tabular_result.metrics["f1_macro"]),
            float(tabular_result.metrics["f1_weighted"]),
            float(tabular_result.metrics["miou"]),
            len(tabular_result.feature_cols),
        )
    if model_kind in ("tempcnn", "inceptiontime"):
        from ml.train.phenology_models import train_temporal_model

        temporal_result = train_temporal_model(
            df=df,
            model_kind=model_kind,
            k_folds=k_folds,
            buffer_km=buffer_km,
            seed=seed,
            n_epochs=10,
            batch_size=128,
        )
        return (
            float(temporal_result.f1_macro),
            float(temporal_result.f1_weighted),
            float(temporal_result.miou),
            int(temporal_result.n_classes),
        )
    raise ValueError(f"`model_kind` no soportado: {model_kind!r}.")
