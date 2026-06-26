"""US-079 runner: dense transfer (fine-tune) + Voting-3 + hierarchical eval.

Orchestrates the full US-079 modelling pipeline on the US-078 Italian homologue,
real-training each dense member on the H100 and combining them with the EPIC 6
deployment winner (the weighted Voting). It is idempotent at the member level
(an already fine-tuned member's ``test_softmax.npz`` is reused unless
``--force``), so a crashed run resumes cheaply.

Pipeline
--------
1. Fine-tune each requested dense member (default ``tsvit-pheno``, ``utae``,
   ``tsvit-pheno-fullm``) from its PASTIS checkpoint with the warm-started Italian
   head (:func:`ml.transfer.finetune_italia.run_italia_finetune`). Per-epoch
   checkpoints land under ``checkpoints/transfer/<member>-italia/<run>/`` (F: on
   the VM).
2. Learn the Voting-3 convex weights over the members' dense post-softmax test
   maps with leave-one-spatial-fold-out CV
   (:class:`ml.ensemble.voting_italia.ItaliaVotingEnsemble`).
3. Evaluate the Voting-3 and each member at the FINE + COARSE granularities,
   build the honest discard curve, and compute the transfer delta vs the
   zero-shot French champion (:mod:`ml.eval.transfer_italia_eval`).
4. Log everything to MLflow (``data_version`` + ``code_version``) and write a
   ``report.json`` + the blended dense maps under the run dir.

Usage
-----
    poetry run python scripts/run_transfer_italia.py \
        --members tsvit-pheno utae tsvit-pheno-fullm \
        --test-fold 3 --finetune-epochs 12 --run us079

    # CPU smoke (tiny, no MLflow server): --device cpu --finetune-epochs 1 \
    #   --head-warmup-epochs 0 --no-mlflow
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ITALIA_ROOT = _REPO_ROOT / "data" / "pastis_italia_2018"
_DEFAULT_CKPT_ROOT = _REPO_ROOT / "checkpoints" / "transfer"

#: The dense members of the Voting-3 (TSViT-pheno + U-TAE are the PASTIS champion
#: pair; the Full-M TSViT is the natural third dense learner now that no per-parcel
#: xgb-alphaearth embedding is materialised for Italy).
DEFAULT_MEMBERS: tuple[str, ...] = ("tsvit-pheno", "utae", "tsvit-pheno-fullm")

#: Rough single-GPU VRAM budget per member at 128px (forward+backward, BF16/FP32
#: mixed), measured headroom on the H100 NVL. Used only to advise serial vs
#: parallel launch; never a hard gate.
_VRAM_GB_PER_MEMBER: dict[str, float] = {
    "tsvit-pheno": 6.0,
    "tsvit-pheno-fullm": 14.0,
    "utae": 5.0,
}


def _member_finetune(
    member: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Fine-tune one member (or reuse its dump) and return its summary."""
    from ml.transfer.finetune_italia import DenseFineTuneConfig, run_italia_finetune

    run_dir = _DEFAULT_CKPT_ROOT / f"{member}-italia" / args.run
    summary_path = run_dir / "summary.json"
    if summary_path.is_file() and not args.force:
        logger.info("member_finetune_reused", member=member, summary=str(summary_path))
        return json.loads(summary_path.read_text(encoding="utf-8"))

    config = DenseFineTuneConfig(
        model_kind="tsvit-pheno" if member == "tsvit-pheno-fullm" else member,
        n_timesteps=args.n_timesteps,
        head_warmup_epochs=args.head_warmup_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        min_patches_per_class=args.min_patches_per_class,
        seed=args.seed,
    )
    # The Full-M member shares the tsvit-pheno builder but inits from the Full-M
    # checkpoint; the L4 member inits from tsvit-pheno-v1.
    from ml.transfer.finetune_italia import DEFAULT_PASTIS_CHECKPOINTS

    ckpt = DEFAULT_PASTIS_CHECKPOINTS["tsvit-pheno"]
    if member == "tsvit-pheno-fullm":
        ckpt = _REPO_ROOT / "checkpoints" / "segmentation" / "tsvit-pheno-fullm-v1" / "best.pt"
    elif member == "utae":
        ckpt = DEFAULT_PASTIS_CHECKPOINTS["utae"]

    return run_italia_finetune(
        config,
        italia_root=args.italia_root,
        pastis_checkpoint=ckpt,
        test_fold=args.test_fold,
        ckpt_root=_DEFAULT_CKPT_ROOT,
        run_name=f"{args.run}-{member}",
        device=args.device,
    )


def _load_test_masks_and_folds(
    italia_root: Path, *, test_fold: int, n_timesteps: int
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """Load the held-out test masks + their spatial folds for the eval/vote."""
    from ml.transfer.finetune_italia import load_italia_patches

    patches = load_italia_patches(
        italia_root=italia_root, n_timesteps=n_timesteps, folds=(test_fold,)
    )
    masks = {pid: patches.masks[i] for i, pid in enumerate(patches.patch_ids)}
    folds = {pid: patches.folds[i] for i, pid in enumerate(patches.patch_ids)}
    return masks, folds


def main() -> None:
    """Parse the CLI, run the US-079 pipeline and persist the report."""
    parser = argparse.ArgumentParser(description="US-079 Italian transfer + Voting-3.")
    parser.add_argument("--members", nargs="+", default=list(DEFAULT_MEMBERS))
    parser.add_argument("--italia-root", type=Path, default=_DEFAULT_ITALIA_ROOT)
    parser.add_argument("--test-fold", type=int, default=3)
    parser.add_argument("--n-timesteps", type=int, default=10)
    parser.add_argument("--head-warmup-epochs", type=int, default=2)
    parser.add_argument("--finetune-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--min-patches-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run", default="us079")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help="Re-train even if a dump exists.")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    parser.add_argument("--no-zero-shot", action="store_true", help="Skip the transfer delta.")
    args = parser.parse_args()

    _advise_vram(args.members)

    # 1) Fine-tune every member.
    member_summaries: dict[str, dict[str, object]] = {}
    for member in args.members:
        member_summaries[member] = _member_finetune(member, args)

    masks, folds = _load_test_masks_and_folds(
        args.italia_root, test_fold=args.test_fold, n_timesteps=args.n_timesteps
    )

    # 2) Learn the Voting-3 over the members' dense post-softmax test maps.
    from ml.ensemble.voting_italia import ItaliaVotingEnsemble, load_member_softmax
    from ml.transfer.italia_label_space import build_italia_label_space

    label_space = build_italia_label_space(italia_root=args.italia_root)
    member_preds = {
        member: load_member_softmax(member, Path(summary["softmax_path"]))
        for member, summary in member_summaries.items()
    }
    vote = ItaliaVotingEnsemble(
        members=tuple(args.members),
        num_classes=label_space.num_classes,
        ignore_index=label_space.background_id,
        random_state=args.seed,
    )
    vote_result = vote.fit_predict(member_preds, masks, folds)

    # 3) Evaluate Voting-3 + each member (fine + coarse), discard curve, delta.
    from ml.eval.transfer_italia_eval import (
        best_subset_over_threshold,
        discard_curve,
        evaluate_dense_predictions,
        probs_to_class_map,
        transfer_delta,
    )

    vote_preds = probs_to_class_map(vote_result.blended_probs_by_patch)
    vote_eval = evaluate_dense_predictions(
        "voting-3", vote_preds, masks, label_space=label_space
    )
    member_evals = {
        member: evaluate_dense_predictions(
            member,
            probs_to_class_map(member_preds[member].probs_by_patch),
            masks,
            label_space=label_space,
        )
        for member in args.members
    }
    curve = discard_curve(vote_eval)
    best_subset = best_subset_over_threshold(vote_eval, threshold=0.9)

    delta: dict[str, float] = {}
    if not args.no_zero_shot:
        from ml.transfer.finetune_italia import zero_shot_pastis_predict

        # The zero-shot cota: the champion dense member (tsvit-pheno) un-adapted.
        zs_member = "tsvit-pheno" if "tsvit-pheno" in args.members else args.members[0]
        zs_preds = zero_shot_pastis_predict(
            model_kind="tsvit-pheno" if zs_member == "tsvit-pheno-fullm" else zs_member,
            italia_root=args.italia_root,
            test_fold=args.test_fold,
            n_timesteps=args.n_timesteps,
            device=args.device,
        )
        zs_eval = evaluate_dense_predictions(
            "zero-shot", zs_preds, masks, label_space=label_space
        )
        delta = transfer_delta(vote_eval, zs_eval)
        member_evals["zero-shot"] = zs_eval

    report = {
        "run": args.run,
        "test_fold": args.test_fold,
        "members": list(args.members),
        "voting_weights": vote_result.weight_map(),
        "voting_oof_f1_macro": round(vote_result.oof_f1_macro, 4),
        "voting_oof_miou": round(vote_result.oof_miou, 4),
        "voting_per_fold": vote_result.per_fold,
        "voting_eval": vote_eval.summary(),
        "voting_per_class": vote_eval.per_class,
        "member_eval": {m: ev.summary() for m, ev in member_evals.items()},
        "discard_curve": curve,
        "best_subset_f1_over_0.9": best_subset,
        "transfer_delta": delta,
        "member_summaries": member_summaries,
    }

    out_dir = _DEFAULT_CKPT_ROOT / "voting-italia" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(
        out_dir / "voting_softmax.npz",
        **{str(pid): arr for pid, arr in vote_result.blended_probs_by_patch.items()},
    )
    logger.info(
        "us079_report_written",
        path=str(report_path),
        voting_fine_f1=report["voting_eval"]["fine_f1_macro"],
        best_subset_n=best_subset["n_classes"],
        best_subset_f1=best_subset["macro_f1"],
    )

    if not args.no_mlflow:
        _log_mlflow(args, report, report_path)


def _advise_vram(members: list[str]) -> None:
    """Log the VRAM budget so the operator can decide serial vs parallel launch."""
    total = sum(_VRAM_GB_PER_MEMBER.get(m, 8.0) for m in members)
    logger.info(
        "us079_vram_budget",
        members=members,
        per_member_gb={m: _VRAM_GB_PER_MEMBER.get(m, 8.0) for m in members},
        total_gb_if_parallel=round(total, 1),
        note="check nvidia-smi BEFORE launching; queue serially if the sum exceeds "
        "the free VRAM (a shared H100; never kill the other session).",
    )


def _log_mlflow(args: argparse.Namespace, report: dict, report_path: Path) -> None:
    """Log the US-079 params / metrics / artifacts to the MLflow server."""
    from ml.utils.mlflow_utils import track_experiment

    with track_experiment(
        "us079-transfer-italia",
        run_name=f"voting3-{args.run}-fold{args.test_fold}",
        dvc_path=str(args.italia_root),
    ) as run:
        import mlflow

        for member, weight in report["voting_weights"].items():
            mlflow.log_param(f"weight_{member}", weight)
        mlflow.log_param("members", ",".join(report["members"]))
        mlflow.log_param("test_fold", report["test_fold"])
        ve = report["voting_eval"]
        mlflow.log_metric("voting_fine_f1_macro", ve["fine_f1_macro"])
        mlflow.log_metric("voting_fine_miou", ve["fine_miou"])
        mlflow.log_metric("voting_coarse_f1_macro", ve["coarse_f1_macro"])
        mlflow.log_metric("voting_coarse_miou", ve["coarse_miou"])
        mlflow.log_metric("voting_oof_f1_macro", report["voting_oof_f1_macro"])
        mlflow.log_metric("best_subset_n_classes", report["best_subset_f1_over_0.9"]["n_classes"])
        mlflow.log_metric("best_subset_macro_f1", report["best_subset_f1_over_0.9"]["macro_f1"])
        for key, value in report.get("transfer_delta", {}).items():
            mlflow.log_metric(key, value)
        mlflow.log_artifact(str(report_path))
        logger.info("us079_mlflow_logged", run_id=run.info.run_id)


if __name__ == "__main__":
    main()
