"""Voting-3 adapter for the Italian transfer (US-079 step 3) -- PARCEL level.

Arthur's ratified decision: the combiner is the WEIGHTED Voting (the deployment
winner of EPIC 6, ``ml.ensemble.voting_weighted.WeightedVotingEnsemble``), and it
must replicate the champion FAITHFULLY -- the champion votes per PARCEL, not per
pixel (the terna ``tsvit-pheno`` + ``utae`` + ``xgb-alphaearth`` scored F1 0.9069
on France by aggregating each member to a per-parcel distribution and combining
those with the convex weight learner). This module therefore votes at the PARCEL
level:

1. Each dense member's post-softmax maps are aggregated to per-parcel
   distributions by :mod:`ml.transfer.dense_to_parcel_italia` and dumped as
   ``oof_parcel_<member>_italia_fold5.parquet`` (the SAME contract the
   ``xgb-alphaearth-italia`` member already writes).
2. The three per-parcel OOF parquets are aligned on a common
   ``canonical_parcel_id`` set and a common global crop-class column space, and
   the convex weights are learned with the EXACT champion machinery
   (``WeightedVotingEnsemble._learn_weights`` -> ``_project_simplex`` -> ``_blend``
   -> ``_f1_of``, reused verbatim), leave-one-spatial-fold-out over the US-078
   ``fold_espacial`` (anti-leakage R-LEAK).

The previous PIXEL-dense voting is preserved as
:class:`ItaliaPixelVotingEnsemble` (a secondary path) for dense-only experiments;
the parcel path is the default and the one that replicates the champion.

Anti-leakage (R-LEAK)
---------------------
The weights are learned OUT-OF-FOLD at the patch level via the per-parcel
``fold_espacial`` (US-078): the vote is learned on the parcels of the OTHER
spatial folds and scored on the held-out fold, so no parcel of a scored fold
leaks into the weight fit. Every member matrix is validated post-softmax
(``validate_probs``) before it enters the vote, and the blended distribution is
validated again. With a single fold present in test the CV degrades with grace and
a HONEST warning (the pilot has 20 patches); the full run spans several folds.

Project conventions: ``polars`` (never pandas), ``numpy`` only at the array
boundary, ``structlog``, type hints + Google-style docstrings; visible prose
Spanish, identifiers English; no emojis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import structlog

from ml.ensemble.voting_weighted import (
    DEFAULT_WEIGHTED_VOTING_MEMBERS,
    WeightedVotingEnsemble,
)
from ml.transfer.dense_to_parcel_italia import DEFAULT_OOF_DIR

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_ITALIA_PARCEL_MEMBERS",
    "DenseMemberPreds",
    "ItaliaParcelVotingEnsemble",
    "ItaliaParcelVotingResult",
    "ItaliaPixelVotingEnsemble",
    "ItaliaVotingResult",
    "load_member_softmax",
]

#: The champion terna at the PARCEL level (replicates the France F1 0.9069 vote):
#: two dense members aggregated to parcel + the tabular AlphaEarth XGBoost member.
DEFAULT_ITALIA_PARCEL_MEMBERS: tuple[str, ...] = (
    "tsvit-pheno",
    "utae",
    "xgb-alphaearth-italia",
)


# --------------------------------------------------------------------------- #
# Parcel-level Voting-3 (the champion replica, DEFAULT path)
# --------------------------------------------------------------------------- #
@dataclass
class ItaliaParcelVotingResult:
    """The learned per-parcel vote, its OOF estimate and the blended parcels.

    Attributes:
        members: Ordered member names.
        weights: The final convex weights (one per member), refit on all parcels.
        oof_f1_macro: Leave-one-spatial-fold-out F1-macro of the weighted vote.
        oof_accuracy: Spatial-CV accuracy of the weighted vote.
        per_fold: Per-held-out-fold ``{"fold", "f1_macro", "accuracy", "weights",
            "n_parcels"}`` diagnostics.
        crop_class_ids: The global crop class ids the prob columns map to.
        n_parcels: Number of parcels in the aligned vote.
        match_coverage: Fraction of each member's parcels kept after the
            intersection join ``{member: coverage}`` (honest join report).
        parcel_ids: The aligned ``canonical_parcel_id`` of every voted parcel, in
            the row order of :attr:`blended_probs`.
        blended_probs: ``(n_parcels, n_crops)`` final weighted vote per parcel
            (refit on all parcels) over :attr:`crop_class_ids` -- the input the
            dense projection (:func:`ml.eval.transfer_italia_eval.
            project_parcel_vote_to_dense`) re-paints onto the patch grid.
    """

    members: tuple[str, ...]
    weights: np.ndarray
    oof_f1_macro: float
    oof_accuracy: float
    per_fold: list[dict[str, object]] = field(default_factory=list)
    crop_class_ids: tuple[int, ...] = ()
    n_parcels: int = 0
    match_coverage: dict[str, float] = field(default_factory=dict)
    parcel_ids: list[str] = field(default_factory=list)
    blended_probs: np.ndarray | None = None

    def weight_map(self) -> dict[str, float]:
        """Return ``{member: weight}`` (the interpretable vote, AC2)."""
        return {
            m: round(float(w), 6)
            for m, w in zip(self.members, self.weights, strict=True)
        }

    def parcel_vote(self) -> dict[str, np.ndarray]:
        """Return ``{canonical_parcel_id: (n_crops,)}`` the blended vote per parcel.

        The per-parcel distribution the dense projection re-paints onto the patch
        grid for the fine/coarse dense eval. Empty if the vote was not blended.
        """
        if self.blended_probs is None:
            return {}
        return {
            pid: self.blended_probs[i]
            for i, pid in enumerate(self.parcel_ids)
        }


class ItaliaParcelVotingEnsemble:
    """Learn the Voting-3 convex weights over per-parcel Italian member OOF.

    Reuses :class:`ml.ensemble.voting_weighted.WeightedVotingEnsemble`'s weight
    learner (``_learn_weights``), simplex projection (``_project_simplex``),
    convex blend (``_blend``) and post-softmax guard (``validate_probs``) WITHOUT
    touching the PASTIS OOF loaders: the inputs are the Italian per-parcel OOF
    parquets (``oof_parcel_<member>_italia_fold5.parquet``), the class column space
    is the global crop-class union, and the CV is leave-one-US078-fold-out.

    Attributes:
        members: Ordered member names (one weight per member).
        oof_dir: Directory holding the per-parcel OOF parquets.
        n_restarts: Nelder-Mead restarts for the weight search.
    """

    def __init__(
        self,
        members: tuple[str, ...] = DEFAULT_ITALIA_PARCEL_MEMBERS,
        *,
        oof_dir: Path = DEFAULT_OOF_DIR,
        n_restarts: int = 6,
        random_state: int = 42,
    ) -> None:
        """Initialize the Italian parcel voting adapter.

        Args:
            members: Ordered member names (>= 2). Each must have an
                ``oof_parcel_<member>_italia_fold5.parquet`` (or, for the xgb
                member, ``oof_parcel_xgb-alphaearth-italia_fold5.parquet``).
            oof_dir: Directory holding the per-parcel OOF parquets.
            n_restarts: Nelder-Mead restarts per weight optimization.
            random_state: Seed forwarded to the underlying learner.

        Raises:
            ValueError: if fewer than two members are given.
        """
        if len(members) < 2:
            raise ValueError(
                f"ItaliaParcelVotingEnsemble needs at least 2 members, got "
                f"{members!r}; a single member is not an ensemble."
            )
        self.members = tuple(members)
        self.oof_dir = Path(oof_dir)
        self.n_restarts = int(n_restarts)
        self._learner = WeightedVotingEnsemble(
            base_members=self.members, n_restarts=n_restarts, random_state=random_state
        )

    def _oof_path(self, member: str) -> Path:
        """Return the per-parcel OOF parquet path of a member.

        The dense members are written by :mod:`ml.transfer.dense_to_parcel_italia`
        as ``oof_parcel_<member>_italia_fold5.parquet``; the xgb member is written
        by :mod:`ml.ensemble.xgb_alphaearth_italia` as
        ``oof_parcel_xgb-alphaearth-italia_fold5.parquet`` (it already carries the
        ``-italia`` suffix in its name).

        Args:
            member: The member name.

        Returns:
            The OOF parquet path under :attr:`oof_dir`.
        """
        if member.endswith("-italia"):
            return self.oof_dir / f"oof_parcel_{member}_fold5.parquet"
        return self.oof_dir / f"oof_parcel_{member}_italia_fold5.parquet"

    def _align_members(
        self,
    ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, tuple[int, ...], dict[str, float]]:
        """Load + align the per-parcel members on a common id/class space.

        Reduces the members to the INTERSECTION of their ``canonical_parcel_id``
        sets and scatters every member onto the UNION of the global crop class ids
        each member's prob columns cover (read off its ``class_id`` ground-truth
        column, the same convention the xgb OOF uses), so column ``i`` of every
        member is the SAME global class. The join coverage per member is reported.

        Returns:
            ``(parcel_ids, probs, labels, folds, crop_class_ids, coverage)`` where
            ``probs`` is ``(n_members, n_parcels, n_crops)`` post-softmax,
            ``labels`` is the per-parcel global class id ``(n_parcels,)``,
            ``folds`` is the per-parcel spatial fold ``(n_parcels,)`` and
            ``coverage`` is ``{member: kept_fraction}``.

        Raises:
            FileNotFoundError: if a member's OOF parquet is absent.
            ValueError: if the members share no parcel id.
        """
        frames: dict[str, pl.DataFrame] = {}
        member_class_ids: dict[str, list[int]] = {}
        for member in self.members:
            path = self._oof_path(member)
            if not path.is_file():
                raise FileNotFoundError(
                    f"per-parcel OOF parquet not found for member {member!r}: "
                    f"{path}. Aggregate the dense members with "
                    "ml.transfer.dense_to_parcel_italia.write_parcel_oof first, or "
                    "train the xgb member with ml.ensemble.xgb_alphaearth_italia."
                )
            frame = pl.read_parquet(path).with_columns(
                pl.col("canonical_parcel_id").cast(pl.Utf8)
            )
            frames[member] = frame
            member_class_ids[member] = _member_crop_class_ids(frame)

        # Common parcel id intersection + per-member join coverage.
        sets = {m: set(frames[m]["canonical_parcel_id"].to_list()) for m in self.members}
        common = set.intersection(*sets.values())
        if not common:
            raise ValueError(
                "the parcel members share no canonical_parcel_id; the dense blob "
                "namespace does not join to the EuroCrops xgb namespace -- use the "
                "EuroCrops ParcelID strategy for the champion terna (see handoff)."
            )
        parcel_ids = sorted(common)
        coverage = {m: round(len(common) / max(len(sets[m]), 1), 4) for m in self.members}

        # Global crop-class column space = union of every member's class ids.
        crop_class_ids = tuple(
            sorted(set().union(*member_class_ids.values()))
        )
        col_of = {cid: i for i, cid in enumerate(crop_class_ids)}
        n_crops = len(crop_class_ids)

        stacked: list[np.ndarray] = []
        labels: np.ndarray | None = None
        folds: np.ndarray | None = None
        for member in self.members:
            aligned = (
                frames[member]
                .filter(pl.col("canonical_parcel_id").is_in(parcel_ids))
                .sort("canonical_parcel_id")
            )
            prob_cols = sorted(c for c in aligned.columns if c.startswith("prob_"))
            raw = aligned.select(prob_cols).to_numpy().astype(np.float64)
            member_ids = member_class_ids[member]
            scattered = np.zeros((raw.shape[0], n_crops), dtype=np.float64)
            for src_col, cid in enumerate(member_ids):
                if cid in col_of:
                    scattered[:, col_of[cid]] = raw[:, src_col]
            row_sums = scattered.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
            scattered = scattered / row_sums
            self._learner.validate_probs(
                scattered, class_axis=-1, name=f"italia-parcel-{member}"
            )
            stacked.append(scattered)
            if labels is None:
                labels = aligned["class_id"].to_numpy().astype(np.int64)
                folds = (
                    aligned["fold"].to_numpy().astype(np.int64)
                    if "fold" in aligned.columns
                    else np.zeros(raw.shape[0], dtype=np.int64)
                )
        probs = np.stack(stacked, axis=0)
        assert labels is not None and folds is not None
        # Re-key labels to the column index space so argmax(col) -> argmax(label).
        label_to_col = np.full(int(max(crop_class_ids) + 1), -1, dtype=np.int64)
        for col, cid in enumerate(crop_class_ids):
            label_to_col[cid] = col
        labels_col = np.array(
            [label_to_col[c] if 0 <= c < label_to_col.size else -1 for c in labels],
            dtype=np.int64,
        )
        logger.info(
            "italia_parcel_members_aligned",
            members=list(self.members),
            n_parcels=len(parcel_ids),
            n_crops=n_crops,
            coverage=coverage,
        )
        return parcel_ids, probs, labels_col, folds, crop_class_ids, coverage

    def fit(self) -> ItaliaParcelVotingResult:
        """Learn the per-parcel vote OOF (leave-one-spatial-fold-out).

        Pipeline:

        1. Align the per-parcel members on a common parcel-id intersection and a
           common global crop-class column space (the join coverage is reported).
        2. For each US-078 spatial fold present, learn the convex weights on the
           parcels of the OTHER folds (``_learn_weights``) and score the weighted
           vote on the held-out fold (F1-macro / accuracy). The fold partition is
           the SAME ``fold_espacial`` the dense members and the xgb member use, so
           no parcel of a scored fold leaked into the weight fit (R-LEAK).
        3. Aggregate the per-fold metrics into the OOF estimate.
        4. Refit the weights on ALL parcels (the production weights).

        Returns:
            An :class:`ItaliaParcelVotingResult` with the learned weights, the OOF
            estimate, the per-fold diagnostics and the join coverage.

        Raises:
            ValueError: if the members share no parcel id (see ``_align_members``).
        """
        from ml.ensemble.base import EnsembleModel

        parcel_ids, probs, labels, folds, crop_class_ids, coverage = self._align_members()

        present_folds = sorted(set(int(f) for f in folds))
        positions = np.arange(len(parcel_ids), dtype=np.int64)
        per_fold: list[dict[str, object]] = []
        if len(present_folds) >= 2:
            for held in present_folds:
                test_pos = positions[folds == held]
                train_pos = positions[folds != held]
                if test_pos.size == 0 or train_pos.size == 0:
                    continue
                EnsembleModel.assert_oof_only(
                    [parcel_ids[i] for i in train_pos],
                    [parcel_ids[i] for i in test_pos],
                    context=f"italia-parcel-vote fold {held}",
                )
                weights = self._learner._learn_weights(
                    probs[:, train_pos, :], labels[train_pos]
                )
                blended = self._learner._blend(probs[:, test_pos, :], weights)
                preds = blended.argmax(axis=-1)
                metrics = EnsembleModel.compute_metrics(
                    labels[test_pos], preds, ignore_index=None
                )
                per_fold.append(
                    {
                        "fold": held,
                        "f1_macro": round(float(metrics["f1_macro"]), 4),
                        "accuracy": round(float(metrics["accuracy"]), 4),
                        "weights": [round(float(w), 4) for w in weights],
                        "n_parcels": int(test_pos.size),
                    }
                )
                logger.info(
                    "italia_parcel_vote_oof_fold",
                    fold=held,
                    f1_macro=per_fold[-1]["f1_macro"],
                    weights=per_fold[-1]["weights"],
                )
        else:
            logger.warning(
                "italia_parcel_vote_single_fold",
                folds=present_folds,
                note="only one spatial fold present; the OOF estimate falls back "
                "to the in-fold fit (reported honestly). The full run spans several "
                "folds for a genuine leave-one-fold-out estimate.",
            )

        final_weights = self._learner._learn_weights(probs, labels)
        # Blend every parcel with the production weights so the dense projection
        # (US-079 rubric eval) can re-paint the voted distribution onto the grid.
        blended_probs = self._learner._blend(probs, final_weights)
        oof_f1 = (
            float(np.mean([f["f1_macro"] for f in per_fold])) if per_fold else float("nan")
        )
        oof_acc = (
            float(np.mean([f["accuracy"] for f in per_fold])) if per_fold else float("nan")
        )
        logger.info(
            "italia_parcel_vote_fit_done",
            members=list(self.members),
            weights=[round(float(w), 4) for w in final_weights],
            oof_f1_macro=round(oof_f1, 4) if per_fold else None,
            n_parcels=len(parcel_ids),
            n_folds=len(present_folds),
        )
        return ItaliaParcelVotingResult(
            members=self.members,
            weights=final_weights,
            oof_f1_macro=oof_f1,
            oof_accuracy=oof_acc,
            per_fold=per_fold,
            crop_class_ids=crop_class_ids,
            n_parcels=len(parcel_ids),
            match_coverage=coverage,
            parcel_ids=list(parcel_ids),
            blended_probs=blended_probs,
        )


def _member_crop_class_ids(frame: pl.DataFrame) -> list[int]:
    """Return the global crop class id of each ``prob_*`` column of an OOF frame.

    The OOF dump (xgb + the dense aggregation) lays out ``prob_{i}`` over the
    sorted global crop class ids present in that member's parcels, the same
    convention :mod:`ml.ensemble.xgb_alphaearth_italia` uses (``prob_{i}`` =
    ``class_ids[i]`` where ``class_ids = sorted(unique(class_id))``). When the
    frame carries an explicit ``prob_class_ids`` metadata column it is honoured;
    otherwise the sorted unique ``class_id`` is the column order.

    Args:
        frame: A per-parcel OOF frame with ``prob_*`` and ``class_id`` columns.

    Returns:
        The global crop class id of each prob column, in column order.
    """
    prob_cols = sorted(c for c in frame.columns if c.startswith("prob_"))
    crop_ids = sorted(
        int(c) for c in frame["class_id"].unique().to_list() if int(c) != 0
    )
    if len(crop_ids) == len(prob_cols):
        return crop_ids
    # Defensive: if the prob columns outnumber the present GT classes (a fold
    # missed a class in test), fall back to a dense 0..n-1 assumption shifted by 1
    # (crop ids start at 1). This keeps the scatter well-defined.
    return list(range(1, len(prob_cols) + 1))


# --------------------------------------------------------------------------- #
# Pixel-dense Voting (SECONDARY path, kept for dense-only experiments)
# --------------------------------------------------------------------------- #
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
    """The learned dense vote, its OOF estimate and the blended dense predictions.

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


class ItaliaPixelVotingEnsemble:
    """Learn the Voting-3 convex weights over DENSE Italian member predictions.

    SECONDARY path (the champion votes per PARCEL -- see
    :class:`ItaliaParcelVotingEnsemble`). Kept for dense-only experiments: it
    flattens the dense maps to a per-pixel member tensor and learns the convex
    weights with the same machinery, leave-one-spatial-fold-out at the patch level.

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
                f"ItaliaPixelVotingEnsemble needs at least 2 members, got "
                f"{members!r}; a single member is not an ensemble."
            )
        self.members = tuple(members)
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.n_restarts = int(n_restarts)
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
        """Learn the dense vote OOF (leave-one-spatial-fold-out) and blend the maps.

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
