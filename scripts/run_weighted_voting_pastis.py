"""Experimento (PENDIENTE engram #334): Voting PONDERADO sobre PASTIS fold-5.

Mide si una capa de combinacion con POCOS pesos aprendibles (un peso convexo por
miembro, ``N``) cierra parte de la brecha F1 +0.124 entre el Voting SIMPLE (E1,
0.6225 pixel) y el Stacking meta-LogReg (E3, 0.7470 parcela, ~54 pesos), en
DISTRIBUCION (PASTIS, dominio origen Francia) antes de gastarlo en transfer.

Comparacion HONESTA -- aisla la CAPA DE COMBINACION, no el protocolo. Las cuatro
variantes corren sobre EXACTAMENTE las mismas parcelas fold-5 de PASTIS-R, con la
misma verdad-terreno reconstruida por :func:`build_parcel_ground_truth` (la que
produjo el 0.7470 del Stacking) y se reportan con la MISMA VARA: ``evaluate()``
DIRECTO (refit del combinador en TODAS las parcelas fold-5, predict sobre las
MISMAS), que es exactamente la medicion que produjo el 0.7470 oficial del Stacking
(``run_us040_ensembles.py:730-733``). La spatial K-fold CV
(:meth:`EnsembleModel.spatial_subfolds`) se conserva solo como columna SECUNDARIA
``f1_macro_spatialcv`` (cifra leakage-free, conservadora) para contraste:

- **Voting simple (parcela)**: media aritmetica de los MISMOS miembros parcela
  (pesos fijos 1/N) -- el "piso" honesto del voto.
- **Voting ponderado (parcela)**: :class:`WeightedVotingEnsemble`, ``N`` pesos
  convexos aprendidos por F1-macro (refit en todo via ``predict_proba``).
- **Stacking (parcela)**: :class:`StackingEnsemble` meta-LogReg, ``N x 18 = 54``
  pesos (el campeon US-040, su 0.7470).
- **Blending (parcela)**: :class:`BlendingEnsemble`, ``N`` pesos convexos via
  Optuna sobre un holdout espacial (referencia: pesos N pero otro regularizador).

NOTA (FIX engram #336): la version previa mezclaba varas -- reportaba el Stacking
y los votos con ``oof_cv_metrics_`` (spatial-CV) y el Blending con ``evaluate()``
directo, asi que el Stacking salia 0.536 en vez de 0.747. Ahora las cuatro usan
``evaluate()`` directo como cifra titular.

Los tres "parcela" comparten la terna ``(tsvit-pheno, utae, xgb-alphaearth)`` para
que la UNICA pieza que cambia entre el voto simple, el voto ponderado y el Stacking
sea la capa de combinacion (pesos 1/N -> N aprendibles -> 54 aprendibles).

Nota de costo (respuesta a "entrenalo en la H100"): NO entrena modelos. Carga las
posteriors OOF ya persistidas (~2 MB/miembro) y aprende N=3 pesos por sub-fold con
``scipy.optimize`` (CPU, segundos). No necesita GPU -- corre en local como el
runner US-040 original.

Uso (desde la raiz del repo):

    poetry run python scripts/run_weighted_voting_pastis.py \\
        --oof-dir ml/eval/oof \\
        --pastis-root data/PASTIS-R \\
        --out-dir reports/ensemble \\
        --fold 5

``--no-use-mlflow`` para un dry-run sin contactar el server :5010.

Convenciones: ``polars`` (nunca pandas), ``numpy`` solo en el borde de arrays,
``structlog``, ``typer``; prosa visible en espanol, identificadores en ingles; sin
emojis; SOLO datos reales PASTIS-R fold-5 (anti-leakage R-LEAK: nunca fold-4).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import structlog
import typer

# Reuse the US-040 closing-run helpers (GT + geometry + label alignment) so this
# experiment shares the EXACT ground truth that produced the 0.7470 Stacking
# number -- DRY, no second source of truth.
from scripts.run_us040_ensembles import (
    HELD_OUT_FOLD,
    PARCEL_BASE_MEMBERS,
    _aligned_labels,
    _fold5_patch_ids,
    _geoms_for_blending,
    build_parcel_geometries,
    build_parcel_ground_truth,
)

logger = structlog.get_logger(__name__)

app = typer.Typer(add_completion=False, help="Voting ponderado sobre PASTIS fold-5.")

#: Canonical key column shared by every parcel frame.
_KEY: str = "canonical_parcel_id"


def _simple_vote_oof_cv(
    members: tuple[str, ...],
    parcel_geoms_gdf: object,
    gt_labels: pl.DataFrame,
    *,
    oof_dir: Path,
    random_state: int,
    n_spatial_folds: int,
    buffer_km: float,
) -> dict[str, float]:
    """Score the SIMPLE parcel vote (fixed 1/N) with the same spatial-CV.

    Reuses :class:`WeightedVotingEnsemble`'s alignment and CV splitting but
    replaces the learned weights with the uniform ``1/N`` vector, so the simple
    vote is reported on the IDENTICAL held-out spatial sub-folds as the weighted
    vote and the Stacking -- the honest floor of the combination layer (the
    published E1 0.6225 is a PIXEL number and not comparable to the parcel cifras).

    Args:
        members: Parcel base learners to average (the Stacking terna).
        parcel_geoms_gdf: GeoDataFrame of the fold-5 parcels (``parcel_id`` int
            surrogate + ``canonical_parcel_id`` + geometry, EPSG:4326).
        gt_labels: Per-parcel GT with ``canonical_parcel_id`` + ``label``.
        oof_dir: OOF directory.
        random_state: Deterministic seed.
        n_spatial_folds: Geographic sub-folds of fold-5.
        buffer_km: Inter-fold exclusion buffer (km).

    Returns:
        ``{"f1_macro": mean, "accuracy": mean}`` over the held-out sub-folds.
    """
    from ml.ensemble.base import EnsembleModel
    from ml.ensemble.voting_weighted import WeightedVotingEnsemble

    helper = WeightedVotingEnsemble(
        members,
        oof_dir=oof_dir,
        random_state=random_state,
        n_spatial_folds=n_spatial_folds,
        buffer_km=buffer_km,
    )
    parcel_ids, member_probs = helper._align_members()
    labels = helper._labels_for(parcel_ids, gt_labels)
    splits = helper._cv_splits(parcel_ids, parcel_geoms_gdf, buffer_km=buffer_km)
    n_members = member_probs.shape[0]
    uniform = np.full(n_members, 1.0 / n_members, dtype=np.float64)

    per_fold: list[dict[str, float]] = []
    for train_pos, test_pos in splits:
        # No learning: the simple vote ignores train_pos (fixed weights). The CV
        # is kept ONLY so the held-out parcels match the weighted-vote evaluation.
        del train_pos
        blended = helper._blend(member_probs[:, test_pos, :], uniform)
        preds = blended.argmax(axis=-1)
        per_fold.append(EnsembleModel.compute_metrics(labels[test_pos], preds, ignore_index=None))
    metrics = {key: float(np.mean([f[key] for f in per_fold])) for key in per_fold[0]}
    logger.info(
        "simple_vote_oof_cv_done",
        members=members,
        f1_macro=round(metrics["f1_macro"], 4),
        n_subfolds=len(splits),
    )
    return metrics


def _simple_vote_direct(
    members: tuple[str, ...],
    gt_labels: pl.DataFrame,
    *,
    oof_dir: Path,
    random_state: int,
    n_spatial_folds: int,
    buffer_km: float,
) -> dict[str, float]:
    """Score the SIMPLE parcel vote (fixed 1/N) with evaluate() DIRECTO.

    Refits nothing (the simple vote has no parameters) and blends the MISMA terna
    over ALL fold-5 parcels with uniform weights, scoring with the same
    ``evaluate()`` path that produced the 0.7470 Stacking number. This is the
    honest floor of the combination layer measured on the SAME yardstick as the
    weighted vote, the Stacking and the Blending (refit-on-all + predict-on-same).

    Args:
        members: Parcel base learners to average (the Stacking terna).
        gt_labels: Per-parcel GT with ``canonical_parcel_id`` + ``label``.
        oof_dir: OOF directory.
        random_state: Deterministic seed.
        n_spatial_folds: Geographic sub-folds (unused here; kept for a uniform
            constructor signature with the weighted vote).
        buffer_km: Inter-fold buffer (unused here; constructor parity).

    Returns:
        ``{"f1_macro": ..., "accuracy": ...}`` over all fold-5 parcels.
    """
    from ml.ensemble.voting_weighted import WeightedVotingEnsemble

    helper = WeightedVotingEnsemble(
        members,
        oof_dir=oof_dir,
        random_state=random_state,
        n_spatial_folds=n_spatial_folds,
        buffer_km=buffer_km,
    )
    parcel_ids, member_probs = helper._align_members()
    n_members = member_probs.shape[0]
    uniform = np.full(n_members, 1.0 / n_members, dtype=np.float64)
    proba = helper._blend(member_probs, uniform)
    labels = _aligned_labels(parcel_ids, gt_labels)
    metrics = helper.evaluate(y_true=labels, proba=proba, fold=HELD_OUT_FOLD)
    logger.info(
        "simple_vote_direct_done",
        members=members,
        f1_macro=round(metrics["f1_macro"], 4),
        n_parcels=len(parcel_ids),
    )
    return metrics


@app.command()
def run(
    oof_dir: Path = typer.Option(Path("ml/eval/oof"), help="US-031 OOF directory."),
    pastis_root: Path = typer.Option(
        Path("data/PASTIS-R"), help="PASTIS-R root (ground truth + geometry)."
    ),
    out_dir: Path = typer.Option(
        Path("reports/ensemble"), help="Output dir for the comparison table."
    ),
    fold: int = typer.Option(HELD_OUT_FOLD, help="Report fold; MUST be 5 (anti-leakage)."),
    n_spatial_folds: int = typer.Option(5, help="Geographic sub-folds of fold-5."),
    buffer_km: float = typer.Option(1.0, help="Inter-fold exclusion buffer (km)."),
    n_trials_blending: int = typer.Option(50, help="Blending Optuna trials."),
    use_mlflow: bool = typer.Option(True, help="Log the weighted-vote run to MLflow (:5010)."),
    random_state: int = typer.Option(42, help="Deterministic seed."),
) -> None:
    """Compara Voting simple / ponderado / Stacking / Blending sobre PASTIS fold-5.

    Rechaza ``fold != 5`` de inmediato (anti-leakage R-LEAK). Escribe la tabla
    comparativa (parquet + CSV) bajo ``out_dir/metrics`` y, si ``use_mlflow``,
    registra una corrida MLflow del Voting ponderado con los pesos aprendidos.
    """
    if fold != HELD_OUT_FOLD:
        raise typer.BadParameter(
            f"--fold debe ser {HELD_OUT_FOLD} (anti-leakage): fold-4 fue el fold de "
            "seleccion y nunca se reporta."
        )

    from ml.ensemble.blending import BlendingEnsemble
    from ml.ensemble.stacking import StackingEnsemble
    from ml.ensemble.voting_weighted import WeightedVotingEnsemble

    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Same fold-5 parcels, GT and geometry that produced the 0.7470 Stacking.
    patch_ids = _fold5_patch_ids(oof_dir)
    parcel_gt = build_parcel_ground_truth(patch_ids, pastis_root)
    parcel_geoms = build_parcel_geometries(patch_ids, pastis_root)
    geoms_gdf = _geoms_for_blending(parcel_geoms)

    members = PARCEL_BASE_MEMBERS  # (tsvit-pheno, utae, xgb-alphaearth)
    logger.info("weighted_vote_experiment_start", members=members, n_parcels=parcel_gt.height)

    rows: list[dict[str, object]] = []

    # ------------------------------------------------------------------------
    # VARA UNICA = evaluate() DIRECTO (refit del combinador en TODAS las parcelas
    # fold-5, predict sobre las MISMAS). Es la cifra que produjo el 0.7470 OFICIAL
    # del Stacking (run_us040_ensembles.py:730-733), la unica comparable entre las
    # 4 variantes. El bug anterior mezclaba esta vara (Blending) con la spatial-CV
    # conservadora (Stacking/Voting), por eso el Stacking aparecia en 0.536 en vez
    # de 0.747. La spatial-CV se conserva como columna SECUNDARIA (leakage-free)
    # para transparencia, pero NO ordena la tabla.
    # ------------------------------------------------------------------------

    # --- Voting simple (parcela, 1/N) ----------------------------------------
    simple = _simple_vote_direct(
        members,
        parcel_gt,
        oof_dir=oof_dir,
        random_state=random_state,
        n_spatial_folds=n_spatial_folds,
        buffer_km=buffer_km,
    )
    simple_cv = _simple_vote_oof_cv(
        members,
        geoms_gdf,
        parcel_gt,
        oof_dir=oof_dir,
        random_state=random_state,
        n_spatial_folds=n_spatial_folds,
        buffer_km=buffer_km,
    )
    rows.append(
        {
            "model": "Voting simple (parcela, 1/N)",
            "f1_macro": simple["f1_macro"],
            "accuracy": simple["accuracy"],
            "f1_macro_spatialcv": simple_cv["f1_macro"],
            "n_weights": len(members),
            "combiner": "mean_1_over_N",
        }
    )

    # --- Voting ponderado (parcela), N pesos aprendibles ---------------------
    wvote = WeightedVotingEnsemble(
        members,
        oof_dir=oof_dir,
        random_state=random_state,
        n_spatial_folds=n_spatial_folds,
        buffer_km=buffer_km,
    ).fit(geoms_gdf, y_true=parcel_gt)
    # evaluate() DIRECTO: predict_proba() blende member_probs con self._weights
    # (refit sobre TODO fold-5) -> misma vara que el 0.747 del Stacking.
    wvote_proba = wvote.predict_proba()
    wvote_labels = _aligned_labels(wvote._member_ids, parcel_gt)
    wvote_direct = wvote.evaluate(y_true=wvote_labels, proba=wvote_proba, fold=fold)
    rows.append(
        {
            "model": "Voting ponderado (parcela, N pesos)",
            "f1_macro": wvote_direct["f1_macro"],
            "accuracy": wvote_direct["accuracy"],
            "f1_macro_spatialcv": wvote.oof_cv_metrics_["f1_macro"],
            "n_weights": len(members),
            "combiner": "weighted_vote_f1max",
        }
    )
    if use_mlflow:
        try:
            wvote.log_to_mlflow(
                wvote_direct,
                run_name="e1w-weighted-voting",
                params=wvote.mlflow_params(),
            )
        except Exception as exc:  # noqa: BLE001 - logging must never abort the run
            logger.warning("weighted_vote_mlflow_failed_skipping", error=str(exc))

    # --- Stacking (parcela), meta-LogReg 54 pesos: su cifra OOF leakage-free --
    # Stacking keys its spatial sub-folds off a POLARS geometry frame (it runs
    # canonical_parcel_id internally), whereas Blending/WeightedVoting take the
    # GeoDataFrame. Pass the Polars `parcel_geoms` here, not `geoms_gdf`.
    stacking = StackingEnsemble(
        members,
        meta="logreg",
        oof_dir=oof_dir,
        random_state=random_state,
        n_spatial_folds=n_spatial_folds,
        buffer_km=buffer_km,
    ).fit(parcel_geoms, gt_labels=parcel_gt)
    # evaluate() DIRECTO reproduce el 0.7470 oficial (patron US-040:730-733):
    # predict_proba() refita el meta-LogReg en TODO fold-5 y predice sobre lo mismo.
    stack_proba = stacking.predict_proba()
    stack_keys, _, _ = stacking.build_meta_features(gt_labels=None)
    stack_labels = _aligned_labels(stack_keys[_KEY].to_list(), parcel_gt)
    stack_direct = stacking.evaluate(y_true=stack_labels, proba=stack_proba, fold=fold)
    rows.append(
        {
            "model": "Stacking (parcela, meta-LogReg 54 pesos)",
            "f1_macro": stack_direct["f1_macro"],
            "accuracy": stack_direct["accuracy"],
            "f1_macro_spatialcv": stacking.oof_cv_metrics_["f1_macro"],
            "n_weights": len(members) * 18,
            "combiner": "meta_logreg",
        }
    )

    # --- Blending (parcela), N pesos via Optuna sobre holdout unico ----------
    blending = BlendingEnsemble(
        members, n_trials=n_trials_blending, oof_dir=oof_dir, random_state=random_state
    ).fit(geoms_gdf, y_true=parcel_gt, buffer_km=buffer_km)
    blend_proba = blending.predict_proba()
    blend_labels = _aligned_labels(blending._member_ids, parcel_gt)
    blend_metrics = blending.evaluate(y_true=blend_labels, proba=blend_proba, fold=fold)
    rows.append(
        {
            "model": "Blending (parcela, N pesos Optuna)",
            "f1_macro": blend_metrics["f1_macro"],
            "accuracy": blend_metrics["accuracy"],
            # Blending optimiza sobre UN holdout espacial, no spatial-CV: sin cifra.
            "f1_macro_spatialcv": None,
            "n_weights": len(members),
            "combiner": "optuna_simplex",
        }
    )

    table = (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("f1_macro").round(4),
            pl.col("accuracy").round(4),
            pl.col("f1_macro_spatialcv").round(4),
        )
        .sort("f1_macro", descending=True)
    )
    out_csv = metrics_dir / "weighted_voting_pastis.csv"
    out_parquet = metrics_dir / "weighted_voting_pastis.parquet"
    table.write_csv(out_csv)
    table.write_parquet(out_parquet)

    # Per-member learned weights of the weighted vote (final refit on all OOF).
    final_weights = {m: round(float(w), 4) for m, w in zip(members, wvote.weights, strict=True)}
    logger.info(
        "weighted_vote_experiment_done",
        # Vara titular = evaluate() directo (comparable al 0.747 del Stacking).
        weighted_vote_direct_f1=round(wvote_direct["f1_macro"], 4),
        stacking_direct_f1=round(stack_direct["f1_macro"], 4),
        simple_vote_direct_f1=round(simple["f1_macro"], 4),
        blending_direct_f1=round(blend_metrics["f1_macro"], 4),
        # Cifra secundaria leakage-free (spatial-CV) para contraste.
        weighted_vote_spatialcv_f1=round(wvote.oof_cv_metrics_["f1_macro"], 4),
        stacking_spatialcv_f1=round(stacking.oof_cv_metrics_["f1_macro"], 4),
        final_weights=final_weights,
        table=str(out_csv),
    )
    sys.stdout.buffer.write(table.write_csv().encode("utf-8"))
    sys.stdout.write(f"\nPesos finales del voto ponderado: {final_weights}\n")


if __name__ == "__main__":
    app()
