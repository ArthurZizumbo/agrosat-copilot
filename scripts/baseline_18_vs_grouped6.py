"""Baseline XGBoost: 18 clases planas PASTIS-R vs 6 grupos HCAT Level-1.

Operativo permanente. Re-corre el MEJOR setup del baseline (XGBoost, spatial
CV 5-fold con buffer anti-leakage de 1 km) sobre las 85951 parcelas PASTIS-R
en DOS esquemas de etiquetas y los compara:

1. ``flat18``    : las 18 clases planas originales PASTIS-R.
2. ``hcat_l1_6`` : las 18 clases fusionadas en 6 super-clases HCAT Level-1
   (CEREALS, OILSEEDS, ROOT_CROPS, LEGUMES, PERMANENT_WOODY, OTHER).

El vector de features es el mejor encontrado en la ablation: las 185 features
base (indices espectrales + temporales + fenologia) mas los embeddings
AlphaEarth Foundations de 2018 (cols ``ae18_NN``) y 2019 (cols ``ae19_NN``),
unidos por ``parcel_id``.

Diseno apples-to-apples: ambos esquemas operan sobre EXACTAMENTE las mismas
filas en el mismo orden, por lo que comparten los mismos splits espaciales
cacheados (la clave de cache es ``n_filas + k + buffer + seed``). La unica
diferencia entre las dos corridas es el remapeo de ``class_id``.

El mapeo de 6 grupos y sus codigos HCAT viven en
``data/reference/pastis_class_mapping.json`` (grouping ``hcat_l1_6``) y se
cargan via ``ml.ingest.pastis_loader.PASTIS_R_GROUPINGS``; este script los
reusa, no los redefine.

Uso:
    python scripts/baseline_18_vs_grouped6.py                  # FULL 85951
    python scripts/baseline_18_vs_grouped6.py --max-samples 300  # validacion

Artefactos en ``reports/baseline/grouped_vs_flat/``:
    - comparison.parquet        : F1-macro y demas metricas por esquema.
    - per_class_f1_flat18.parquet     : F1 por clase (18 clases).
    - per_class_f1_hcat_l1_6.parquet  : F1 por grupo HCAT (6 grupos).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
import structlog
from sklearn.metrics import f1_score

from ml.ingest.pastis_loader import PASTIS_R_CLASSES, PASTIS_R_GROUPINGS
from ml.train.baseline import (
    build_estimator,
    compute_baseline_metrics,
    evaluate_with_spatial_cv,
)

logger = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = (
    _REPO_ROOT / "data" / "test_fixtures" / "feature_selection_parcels_subset.parquet"
)
_AE18 = (
    _REPO_ROOT
    / "data"
    / "cache"
    / "gee"
    / "alphaearth_parcels_parcels_2018_85951.parquet"
)
_AE19 = (
    _REPO_ROOT
    / "data"
    / "cache"
    / "gee"
    / "alphaearth_parcels_pastis_parcels_2019_85951.parquet"
)
_OUT_DIR = _REPO_ROOT / "reports" / "baseline" / "grouped_vs_flat"

# Grouping canonico HCAT Level-1 (6 grupos). Se reusa el que ya vive en
# data/reference/pastis_class_mapping.json; aqui solo se referencia el nombre
# y se documentan los codigos HCAT para defendibilidad. NO redefinir el mapa:
# es la fuente unica de verdad cargada via PASTIS_R_GROUPINGS.
_HCAT_GROUPING = "hcat_l1_6"

# Codigos HCAT v3 Level-1 de cada fusion (metodo Russwurm et al. 2018 /
# H2Crop arXiv:2506.06155). Se imprimen para la defensa del agrupamiento.
_HCAT_CODES: dict[str, str] = {
    "CEREALS": "3300000000 cereals (wheat 3301, barley 3302, maize 3303, "
    "triticale 3304, sorghum 3305, mixed cereal 3300010000)",
    "OILSEEDS": "3400000000 oilseed crops (rapeseed 3401010101, sunflower 3401050000)",
    "ROOT_CROPS": "3500000000 root/tuber crops (sugar beet 3500010000, potato 3500020000)",
    "LEGUMES": "3600000000 leguminous crops (leguminous fodder 3600060000, soybean 3601000000)",
    "PERMANENT_WOODY": "3900000000 permanent/woody crops "
    "(grapevine 3901000000, orchard 3902000000)",
    "OTHER": "3000000000 raiz / horticultura mixta (grassland-meadow 3370000000, "
    "fruits/vegetables/flowers 3800000000)",
}


def _load_features(max_samples: int | None) -> pl.DataFrame:
    """Carga el fixture de 185 features y une los embeddings AlphaEarth.

    Une AlphaEarth 2018 (``ae18_NN``) y 2019 (``ae19_NN``) por ``parcel_id``.
    Si ``max_samples`` se especifica, submuestrea de forma determinista
    ANTES de unir (la validacion rapida no necesita las 85951 filas).

    Args:
        max_samples: Si no es ``None``, toma las primeras ``max_samples``
            filas tras un shuffle con semilla fija.

    Returns:
        DataFrame Polars con las 185 features base + 128 dims AlphaEarth
        (64 de 2018 + 64 de 2019) + metadata (``parcel_id``, ``class_id``,
        ``patch_id``, etc.).
    """
    base = pl.read_parquet(_FIXTURE)
    if max_samples is not None and max_samples < base.height:
        base = base.sample(n=max_samples, seed=42, shuffle=True)

    ae18 = pl.read_parquet(_AE18).drop("year")
    ae18 = ae18.rename(
        {c: f"ae18_{c.removeprefix('dim_')}" for c in ae18.columns if c != "parcel_id"}
    )
    ae19 = pl.read_parquet(_AE19).drop("year")
    ae19 = ae19.rename(
        {c: f"ae19_{c.removeprefix('dim_')}" for c in ae19.columns if c != "parcel_id"}
    )

    merged = base.join(ae18, on="parcel_id", how="left").join(
        ae19, on="parcel_id", how="left"
    )
    n_ae18 = sum(c.startswith("ae18_") for c in merged.columns)
    n_ae19 = sum(c.startswith("ae19_") for c in merged.columns)
    logger.info(
        "features_loaded",
        n_rows=merged.height,
        n_cols=merged.width,
        n_ae18=n_ae18,
        n_ae19=n_ae19,
    )
    if n_ae18 != 64 or n_ae19 != 64:
        raise ValueError(
            f"Se esperaban 64 dims AlphaEarth por anio; obtenidas {n_ae18} (2018) "
            f"y {n_ae19} (2019). Revisa los caches de AlphaEarth."
        )
    return merged


def _remap_to_hcat(df: pl.DataFrame) -> pl.DataFrame:
    """Remapea ``class_id`` (1..18) a IDs enteros de los 6 grupos HCAT L1.

    Usa el grouping canonico ``hcat_l1_6`` de
    :data:`PASTIS_R_GROUPINGS`. Asigna un entero estable y ordenado a cada
    nombre de grupo para que XGBoost reciba etiquetas limpias.

    Args:
        df: DataFrame con la columna ``class_id`` (18 clases planas).

    Returns:
        Copia del DataFrame con ``class_id`` reemplazado por el ID del grupo
        HCAT (enteros contiguos por orden alfabetico de grupo).
    """
    grouping = PASTIS_R_GROUPINGS[_HCAT_GROUPING]
    group_names = sorted(set(grouping.values()))
    name_to_id = {name: i + 1 for i, name in enumerate(group_names)}
    class_to_group_id = {cid: name_to_id[grp] for cid, grp in grouping.items()}

    return df.with_columns(
        pl.col("class_id").replace_strict(class_to_group_id).alias("class_id")
    )


def _group_id_to_name() -> dict[int, str]:
    """Mapa ``{group_id: group_name}`` consistente con :func:`_remap_to_hcat`."""
    group_names = sorted(set(PASTIS_R_GROUPINGS[_HCAT_GROUPING].values()))
    return {i + 1: name for i, name in enumerate(group_names)}


def _run_scheme(
    df: pl.DataFrame,
    *,
    scheme: str,
    k_folds: int,
    buffer_km: float,
) -> tuple[dict[str, float], pl.DataFrame]:
    """Corre el CV espacial para un esquema de etiquetas y arma per-class F1.

    Llama a :func:`evaluate_with_spatial_cv` (el mismo motor que usa
    ``train_one_model``) para obtener las predicciones out-of-fold, calcula
    las metricas agregadas con :func:`compute_baseline_metrics` y el F1 por
    clase a partir de los mismos vectores OOF.

    Args:
        df: DataFrame de features con ``class_id`` ya en el esquema deseado.
        scheme: Etiqueta del esquema (``"flat18"`` o ``"hcat_l1_6"``).
        k_folds: Numero de folds del CV espacial.
        buffer_km: Buffer anti-leakage en km.

    Returns:
        Tupla ``(metrics, per_class_df)`` donde ``metrics`` son las cinco
        metricas OOF y ``per_class_df`` es un DataFrame con ``label_id``,
        ``label_name``, ``f1`` y ``support`` por clase.
    """
    from ml.train.baseline import _base_params, _encode_labels  # reuso interno

    # No fijamos `num_class`: XGBClassifier (API sklearn) lo infiere por fold.
    # Forzarlo rompe los folds donde el train no contiene las 18 clases
    # (artefacto de subsamples chicos; en el full cada fold trae todas).
    params = _base_params("xgb")

    def factory():  # type: ignore[no-untyped-def]
        return build_estimator("xgb", params)

    _, y_true_oof, y_pred_oof = evaluate_with_spatial_cv(
        df, factory, k_folds=k_folds, buffer_km=buffer_km, random_state=42
    )

    encoder, _ = _encode_labels(df)
    labels = list(range(len(encoder.classes_)))
    metrics = compute_baseline_metrics(y_true_oof, y_pred_oof, labels=labels)

    f1_per = f1_score(
        y_true_oof, y_pred_oof, labels=labels, average=None, zero_division=0
    )
    support = np.bincount(y_true_oof.astype(np.int64), minlength=len(labels))

    if scheme == "flat18":
        names = [PASTIS_R_CLASSES[int(c)] for c in encoder.classes_]
    else:
        gid_to_name = _group_id_to_name()
        names = [gid_to_name[int(c)] for c in encoder.classes_]

    per_class = pl.DataFrame(
        {
            "scheme": [scheme] * len(labels),
            "label_id": [int(c) for c in encoder.classes_],
            "label_name": names,
            "f1": [float(x) for x in f1_per],
            "support": [int(s) for s in support],
        }
    ).sort("f1")

    logger.info(
        "scheme_done",
        scheme=scheme,
        f1_macro=round(metrics["f1_macro"], 4),
        n_classes=len(labels),
    )
    return metrics, per_class


def _print_per_class(title: str, frame: pl.DataFrame) -> None:
    """Imprime un per-class F1 en ASCII puro (consola cp1252 segura)."""
    print(f"\n{title}")
    print(f"  {'id':>3}  {'clase/grupo':32s}  {'F1':>7}  {'support':>8}")
    for row in frame.iter_rows(named=True):
        print(
            f"  {row['label_id']:>3}  {row['label_name']:32s}  "
            f"{row['f1']:>7.4f}  {row['support']:>8d}"
        )


def main() -> None:
    """Punto de entrada: corre ambos esquemas, persiste y reporta."""
    # La consola Windows (cp1252) no codifica los bordes Unicode de Polars ni
    # acentos; forzamos UTF-8 en stdout para que el reporte no truene.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Submuestra N parcelas para validacion rapida (default: full 85951).",
    )
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--buffer-km", type=float, default=1.0)
    args = parser.parse_args()

    df = _load_features(args.max_samples)

    # Esquema 1: 18 clases planas.
    metrics_18, per_class_18 = _run_scheme(
        df, scheme="flat18", k_folds=args.k_folds, buffer_km=args.buffer_km
    )

    # Esquema 2: 6 grupos HCAT L1 (mismas filas -> mismos folds cacheados).
    df_grouped = _remap_to_hcat(df)
    metrics_6, per_class_6 = _run_scheme(
        df_grouped, scheme=_HCAT_GROUPING, k_folds=args.k_folds, buffer_km=args.buffer_km
    )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison = pl.DataFrame(
        [
            {"scheme": "flat18", "n_classes": 18, **metrics_18},
            {"scheme": _HCAT_GROUPING, "n_classes": 6, **metrics_6},
        ]
    )
    comparison.write_parquet(_OUT_DIR / "comparison.parquet")
    per_class_18.write_parquet(_OUT_DIR / "per_class_f1_flat18.parquet")
    per_class_6.write_parquet(_OUT_DIR / f"per_class_f1_{_HCAT_GROUPING}.parquet")

    delta = metrics_6["f1_macro"] - metrics_18["f1_macro"]

    print("\n=== Baseline XGBoost: 18 clases vs 6 grupos HCAT L1 ===")
    print(f"n_parcelas={df.height} | spatial CV {args.k_folds}-fold | buffer {args.buffer_km} km")
    print(f"\nF1-macro 18 clases planas : {metrics_18['f1_macro']:.4f}")
    print(f"F1-macro 6 grupos HCAT L1 : {metrics_6['f1_macro']:.4f}")
    print(f"Delta (6 grupos - 18)     : {delta:+.4f}")
    print("\nMetricas completas por esquema:")
    for row in comparison.iter_rows(named=True):
        print(
            f"  {row['scheme']:10s} n_classes={row['n_classes']:2d} "
            f"f1_macro={row['f1_macro']:.4f} f1_weighted={row['f1_weighted']:.4f} "
            f"miou={row['miou']:.4f} accuracy={row['accuracy']:.4f} "
            f"cohen_kappa={row['cohen_kappa']:.4f}"
        )
    _print_per_class(
        "F1 por clase (18 clases planas, peor a mejor):", per_class_18
    )
    _print_per_class(
        "F1 por grupo HCAT L1 (6 grupos):", per_class_6.sort("label_id")
    )
    print("\nCodigos HCAT de cada grupo:")
    for name, code in _HCAT_CODES.items():
        print(f"  {name:16s} {code}")
    print(f"\nArtefactos en {_OUT_DIR}")


if __name__ == "__main__":
    main()
