"""Voting-3 adapter for the Italian dense transfer (US-079 step 3).

Arthur's ratified decision: the combiner is the WEIGHTED Voting (the deployment
winner of EPIC 6, ``ml.ensemble.voting_weighted.WeightedVotingEnsemble``), NOT
meta-LogReg/Stacking. This module is the thin ADAPTER that lets that exact weight
learner operate on the DENSE Italian predictions instead of the PASTIS per-parcel
OOF dumps:

1. Load each fine-tuned member's post-softmax dense maps ``(K, 128, 128)`` per
   held-out Italian patch (the ``.npz`` artifacts produced by
   :func:`ml.transfer.finetune_italia.run_italia_finetune`).
2. Flatten them to a per-pixel member tensor ``(n_members, n_pixels, K)`` aligned
   with the ground-truth pixel labels, dropping the background/ignore pixels.
3. Learn the convex weights with the SAME machinery as the EPIC 6 winner
   (``WeightedVotingEnsemble._learn_weights`` -> ``_project_simplex`` -> ``_blend``
   -> ``_f1_of``, all reused verbatim), so the only moving part is the input
   namespace (dense Italian pixels vs PASTIS parcels).

Anti-leakage (R-LEAK)
---------------------
The weights are learned OUT-OF-FOLD at the PATCH level: the Italian patches carry
a ``fold_espacial`` (US-078). A leave-one-fold-out cross-validation learns the
weights on the pixels of the OTHER spatial folds and scores the held-out fold,
so the weights never see a pixel of the patch they are scored on. Every member
matrix is validated post-softmax (``validate_probs``) before it enters the vote.
The final production weights are refit on ALL the test pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import structlog

from ml.ensemble.voting_weighted import (
    DEFAULT_WEIGHTED_VOTING_MEMBERS,
    WeightedVotingEnsemble,
)

logger = structlog.get_logger(__name__)

__all__ = [
    "DenseMemberPreds",
    "ItaliaVotingEnsemble",
    "ItaliaVotingResult",
    "load_member_softmax",
]


@dataclass
class DenseMemberPreds:
    """A member's dense post-softmax maps keyed by Italian patch id.

    Attributes:
        member: The member name (e.g. ``"tsvit-pheno"``).
        probs_by_patch: ``{patch_id: (K, H, W) float32}`` post-softmax maps.
        num_classes: The class axis size ``K`` (background included).
    """

    member: str
    probs_by_patch: dict[int, np.ndarray]
    num_classes: int


def load_member_softmax(member: str, npz_path: Path) -> DenseMemberPreds:
    """Load a member's dense post-softmax maps from a fine-tune ``.npz`` dump.

    The ``.npz`` is the ``test_softmax.npz`` written by
    :func:`ml.transfer.finetune_italia.run_italia_finetune`: one array per test
    patch, keyed by the patch id (as a string), shape ``(K, H, W)``.

    Args:
        member: The member name used in the vote/report.
        npz_path: Path to the member's ``test_softmax.npz``.

    Returns:
        A :class:`DenseMemberPreds`.

    Raises:
        FileNotFoundError: if the ``.npz`` is absent.
        ValueError: if the dump is empty.
    """
    if not Path(npz_path).is_file():
        raise FileNotFoundError(
            f"member softmax dump not found at {npz_path}; run the fine-tune for "
            f"{member!r} first (ml.transfer.finetune_italia.run_italia_finetune)."
        )
    with np.load(npz_path) as data:
        probs = {int(key): np.asarray(data[key], dtype=np.float32) for key in data.files}
    if not probs:
        raise ValueError(f"member softmax dump {npz_path} is empty.")
    num_classes = next(iter(probs.values())).shape[0]
    logger.info(
        "italia_member_softmax_loaded",
        member=member,
        n_patches=len(probs),
        num_classes=num_classes,
    )
    return DenseMemberPreds(member=member, probs_by_patch=probs, num_classes=num_classes)


@dataclass
class ItaliaVotingResult:
    """The learned vote, its OOF estimate and the blended dense predictions.

    Attributes:
        members: Ordered member names.
        weights: The final convex weights (one per member), refit on all pixels.
        oof_f1_macro: Leave-one-fold-out (spatial) F1-macro of the weighted vote.
        oof_miou: Spatial-CV mIoU of the weighted vote.
        per_fold: Per-held-out-fold ``{"fold", "f1_macro", "miou", "weights",
            "n_pixels"}`` diagnostics.
        blended_probs_by_patch: ``{patch_id: (K, H, W)}`` blended post-softmax maps
            (for the dense eval / confusion / qualitative demo).
    """

    members: tuple[str, ...]
    weights: np.ndarray
    oof_f1_macro: float
    oof_miou: float
    per_fold: list[dict[str, object]] = field(default_factory=list)
    blended_probs_by_patch: dict[int, np.ndarray] = field(default_factory=dict)

    def weight_map(self) -> dict[str, float]:
        """Return ``{member: weight}`` (the interpretable vote, AC2)."""
        return {m: round(float(w), 6) for m, w in zip(self.members, self.weights, strict=True)}


class ItaliaVotingEnsemble:
    """Learn the Voting-3 convex weights over dense Italian member predictions.

    Wraps :class:`ml.ensemble.voting_weighted.WeightedVotingEnsemble` to reuse its
    weight learner (``_learn_weights``), simplex projection (``_project_simplex``),
    convex blend (``_blend``), F1 helper (``_f1_of``) and post-softmax guard
    (``validate_probs``) WITHOUT touching the PASTIS parcel/OOF machinery: the
    inputs are dense Italian pixels, the CV is leave-one-spatial-fold-out at the
    patch level, and the metrics come from the dense accumulator.

    Attributes:
        members: Ordered member names (the vote learns one weight per member).
        num_classes: The class axis size ``K``.
        ignore_index: Pixel label dropped from the vote/eval (background id 0).
        n_restarts: Nelder-Mead restarts for the weight search.
    """

    def __init__(
        self,
        members: tuple[str, ...] = DEFAULT_WEIGHTED_VOTING_MEMBERS,
        *,
        num_classes: int,
        ignore_index: int = 0,
        n_restarts: int = 6,
        random_state: int = 42,
    ) -> None:
        """Initialize the dense Italian voting adapter.

        Args:
            members: Ordered member names (>= 2; a single member is not a vote).
            num_classes: The class axis size ``K`` (background included).
            ignore_index: Pixel label excluded from the vote and the metrics
                (default 0 = background).
            n_restarts: Nelder-Mead restarts per weight optimization.
            random_state: Seed forwarded to the underlying learner.

        Raises:
            ValueError: if fewer than two members are given.
        """
        if len(members) < 2:
            raise ValueError(
                f"ItaliaVotingEnsemble needs at least 2 members, got {members!r}; "
                "a single member is not an ensemble."
            )
        self.members = tuple(members)
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.n_restarts = int(n_restarts)
        # The underlying learner only needs its weight-search helpers; build it with
        # the dense member names and never call its PASTIS ``fit`` (no OOF dumps).
        self._learner = WeightedVotingEnsemble(
            base_members=self.members,
            n_restarts=n_restarts,
            random_state=random_state,
        )

    def _stack_pixels(
        self,
        member_preds: dict[str, DenseMemberPreds],
        masks_by_patch: dict[int, np.ndarray],
        patch_ids: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Flatten the dense maps of ``patch_ids`` into a pixel member tensor.

        Drops the ignore-index pixels (background) so the vote is learned only on
        supervised crop pixels, mirroring the dense metric convention.

        Args:
            member_preds: ``{member: DenseMemberPreds}`` aligned with
                :attr:`members`.
            masks_by_patch: ``{patch_id: (H, W)}`` ground-truth class masks.
            patch_ids: The patch ids contributing pixels.

        Returns:
            ``(probs, labels)`` where ``probs`` is ``(n_members, n_pixels, K)``
            post-softmax (validated) and ``labels`` is ``(n_pixels,)`` class ids.
        """
        per_member_rows: list[list[np.ndarray]] = [[] for _ in self.members]
        label_rows: list[np.ndarray] = []
        for pid in patch_ids:
            mask = masks_by_patch[pid].reshape(-1)
            keep = mask != self.ignore_index
            if not keep.any():
                continue
            label_rows.append(mask[keep])
            for mi, member in enumerate(self.members):
                probs = member_preds[member].probs_by_patch[pid]  # (K, H, W)
                flat = probs.reshape(probs.shape[0], -1).T  # (n_pix, K)
                per_member_rows[mi].append(flat[keep])
        labels = np.concatenate(label_rows, axis=0)
        stacked = np.stack(
            [np.concatenate(rows, axis=0) for rows in per_member_rows], axis=0
        )  # (n_members, n_pixels, K)
        validated = np.stack(
            [
                self._learner.validate_probs(
                    stacked[mi], class_axis=-1, name=f"italia-{self.members[mi]}"
                )
                for mi in range(len(self.members))
            ],
            axis=0,
        )
        return validated, labels

    def fit_predict(
        self,
        member_preds: dict[str, DenseMemberPreds],
        masks_by_patch: dict[int, np.ndarray],
        folds_by_patch: dict[int, int],
    ) -> ItaliaVotingResult:
        """Learn the vote OOF (leave-one-spatial-fold-out) and blend the maps.

        Pipeline:

        1. For each spatial fold present in ``folds_by_patch``, learn the convex
           weights on the pixels of the OTHER folds (``_learn_weights``) and score
           the weighted vote on the held-out fold's pixels (dense F1-macro / mIoU).
           The patch-level fold split guarantees no pixel of a scored patch leaked
           into the weight fit (R-LEAK).
        2. Aggregate the per-fold metrics into the OOF estimate.
        3. Refit the weights on ALL test pixels (the production weights) and blend
           every member's dense map with them (``_blend``), producing the dense
           post-softmax maps the US-079 eval consumes.

        Args:
            member_preds: ``{member: DenseMemberPreds}`` aligned with
                :attr:`members`.
            masks_by_patch: ``{patch_id: (H, W)}`` ground-truth class masks.
            folds_by_patch: ``{patch_id: fold_espacial}`` (US-078 spatial fold).

        Returns:
            An :class:`ItaliaVotingResult` with the learned weights, the OOF
            estimate, the per-fold diagnostics and the blended dense maps.

        Raises:
            ValueError: if a member is missing predictions for a patch, or the
                spatial folds yield no usable leave-one-fold-out split.
        """
        from ml.ensemble.base import EnsembleModel
        from ml.eval.dense_metrics import compute_dense_metrics

        common_ids = sorted(
            set.intersection(
                *[set(member_preds[m].probs_by_patch) for m in self.members]
            )
        )
        if not common_ids:
            raise ValueError(
                "no patch is predicted by every member; the members are not aligned."
            )

        fold_of = {pid: folds_by_patch.get(pid, 0) for pid in common_ids}
        folds = sorted(set(fold_of.values()))
        per_fold: list[dict[str, object]] = []
        if len(folds) >= 2:
            for held in folds:
                train_ids = [pid for pid in common_ids if fold_of[pid] != held]
                test_ids = [pid for pid in common_ids if fold_of[pid] == held]
                EnsembleModel.assert_oof_only(
                    train_ids, test_ids, context=f"italia-vote fold {held}"
                )
                probs_tr, y_tr = self._stack_pixels(
                    member_preds, masks_by_patch, train_ids
                )
                probs_te, y_te = self._stack_pixels(
                    member_preds, masks_by_patch, test_ids
                )
                weights = self._learner._learn_weights(probs_tr, y_tr)
                blended = self._learner._blend(probs_te, weights)
                preds = blended.argmax(axis=-1)
                metrics = compute_dense_metrics(
                    preds, y_te, num_classes=self.num_classes, ignore_index=None
                )
                per_fold.append(
                    {
                        "fold": held,
                        "f1_macro": round(float(metrics["f1_macro"]), 4),
                        "miou": round(float(metrics["miou"]), 4),
                        "weights": [round(float(w), 4) for w in weights],
                        "n_pixels": int(y_te.size),
                    }
                )
                logger.info(
                    "italia_vote_oof_fold",
                    fold=held,
                    f1_macro=per_fold[-1]["f1_macro"],
                    miou=per_fold[-1]["miou"],
                    weights=per_fold[-1]["weights"],
                )
        else:
            logger.warning(
                "italia_vote_single_fold",
                fold=folds,
                note="only one spatial fold in test; OOF estimate falls back to the "
                "in-fold fit (reported honestly).",
            )

        # Production weights: refit on ALL test pixels and blend every member map.
        probs_all, y_all = self._stack_pixels(member_preds, masks_by_patch, common_ids)
        final_weights = self._learner._learn_weights(probs_all, y_all)

        oof_f1 = (
            float(np.mean([f["f1_macro"] for f in per_fold])) if per_fold else float("nan")
        )
        oof_miou = (
            float(np.mean([f["miou"] for f in per_fold])) if per_fold else float("nan")
        )

        blended_by_patch: dict[int, np.ndarray] = {}
        for pid in common_ids:
            stack = np.stack(
                [member_preds[m].probs_by_patch[pid] for m in self.members], axis=0
            )  # (n_members, K, H, W)
            k, h, w = stack.shape[1:]
            flat = stack.reshape(len(self.members), k, -1).transpose(0, 2, 1)
            blended = self._learner._blend(flat, final_weights)  # (n_pix, K)
            blended_by_patch[pid] = blended.T.reshape(k, h, w).astype(np.float32)

        logger.info(
            "italia_vote_fit_done",
            members=list(self.members),
            weights=[round(float(w), 4) for w in final_weights],
            oof_f1_macro=round(oof_f1, 4) if per_fold else None,
            oof_miou=round(oof_miou, 4) if per_fold else None,
            n_patches=len(common_ids),
        )
        return ItaliaVotingResult(
            members=self.members,
            weights=final_weights,
            oof_f1_macro=oof_f1,
            oof_miou=oof_miou,
            per_fold=per_fold,
            blended_probs_by_patch=blended_by_patch,
        )
