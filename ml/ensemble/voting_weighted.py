"""E1-w -- Weighted parcel-level soft-voting ensemble (pendiente PENDIENTE #334).

A learnable-weight generalization of the E1 Voting ensemble, evaluated with the
EXACT anti-leakage protocol of E3 Stacking so the two are an apples-to-apples
comparison of the COMBINATION LAYER alone:

    P_wvote = sum_i w_i * P_member_i      with   w_i >= 0, sum_i w_i = 1

The plain E1 :class:`ml.ensemble.voting.VotingEnsemble` fixes ``w_i = 1/N`` (its
``fit`` is a no-op arithmetic mean). This ensemble instead LEARNS the ``N`` convex
weights -- one scalar per member, not one per (member, class) -- by directly
maximizing F1-macro on out-of-fold spatial sub-folds. The hypothesis under test
(memoria engram #334): the meta-LogReg of Stacking learns ``N x 18 = 54`` weights
and may overfit; a weighted vote learns only ``N`` weights, so it has far less
variance and might generalize better in transfer even if it is worse in-domain.
Measuring it on PASTIS (in-distribution) first tells us whether the COMBINATION
LAYER can close any of the +0.124 F1 gap E1 vs E3 before spending it on transfer.

Why this is a Voting, not a Blending (memoria engram #332). It IS the same convex
combination Blending uses, but the comparison must isolate the WEIGHT CARDINALITY
(``N`` vs ``54``), not the protocol. So this class deliberately differs from
:class:`ml.ensemble.blending.BlendingEnsemble` on two axes:

1. **Protocol = Stacking's, not Blending's.** Stacking estimates its leakage-free
   quality by spatial K-fold CV over fold-5 (train the combiner on the OOF rows of
   the other sub-folds, score the held-out sub-fold, average). Blending instead
   optimizes on a SINGLE spatial holdout. To compare the weighted vote against the
   Stacking number on equal footing, the weights are learned and scored with the
   SAME spatial K-fold CV (:meth:`EnsembleModel.spatial_subfolds`), and the
   reported metric is the aggregated OOF estimate -- the leakage-free number that
   sits next to Stacking's ``oof_cv_metrics_``.
2. **Objective = raw F1-macro, no Optuna, no gap term.** Blending maximizes
   ``f1_val - gap_lambda * |f1_train - f1_val|`` via Optuna/TPE. Here the only
   question is "what does the best convex weighting buy", so the objective is the
   bare F1-macro of the weighted vote, optimized directly with
   :func:`scipy.optimize.minimize` (Nelder-Mead over the simplex logits). This
   removes the gap-penalty confound so the gap to Stacking is attributable to
   weight cardinality, not to a different regularizer.

The members default to the SAME parcel base learners as Stacking
(``tsvit-pheno``, ``utae``, ``xgb-alphaearth``) so the only moving part between
this ensemble and E3 is the combiner (N convex weights vs a 54-weight LogReg).

Anti-leakage (R-LEAK). Inherited wholesale from
:class:`ml.ensemble.blending.BlendingEnsemble` /
:class:`ml.ensemble.base.EnsembleModel`: every member matrix is validated
post-softmax before it enters the vote; the weighted output (a convex combination
of distributions) is validated again; the sub-folds are geographic
(``build_spatial_kfold``), never random; and :meth:`EnsembleModel.assert_oof_only`
hard-fails any train/eval parcel overlap inside the CV.

Project conventions: ``polars`` (never pandas), ``numpy`` only at the array
boundary, ``structlog`` for logging, type hints and Google-style docstrings;
visible prose Spanish, code identifiers English; real PASTIS-R data only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import structlog

from ml.ensemble.blending import BlendingEnsemble

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from collections.abc import Sequence

    import geopandas as gpd
    import polars as pl

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_WEIGHTED_VOTING_MEMBERS",
    "WeightedVotingEnsemble",
]

#: Default parcel base learners: the SAME terna as Stacking/Blending so the only
#: difference between this ensemble and E3 is the combination layer.
DEFAULT_WEIGHTED_VOTING_MEMBERS: tuple[str, ...] = (
    "tsvit-pheno",
    "utae",
    "xgb-alphaearth",
)

#: Number of Nelder-Mead restarts from different simplex corners/centre. The
#: F1-macro surface over the simplex is non-convex and piecewise-constant (argmax
#: of a weighted mean), so a multi-start search avoids a single bad basin.
_N_RESTARTS: int = 6

#: Max Nelder-Mead iterations per restart (the simplex is low-dim: N members).
_MAX_ITER: int = 400


class WeightedVotingEnsemble(BlendingEnsemble):
    """E1-w: convex weighted vote, learned by F1-macro with Stacking's CV.

    Subclasses :class:`ml.ensemble.blending.BlendingEnsemble` to reuse, verbatim,
    the member alignment (:meth:`_align_members`), the label alignment
    (:meth:`_labels_for`), the simplex projection (:meth:`_project_simplex`), the
    convex combination (:meth:`_blend`) and the F1 helper (:meth:`_f1_of`). Only
    :meth:`fit` is overridden: instead of Optuna on a single holdout, it learns the
    ``N`` weights by direct F1-macro maximization (:func:`scipy.optimize.minimize`)
    and estimates the leakage-free quality with the SAME spatial K-fold CV as
    Stacking, exposing it on :attr:`oof_cv_metrics_`.

    Attributes:
        base_members: Ordered parcel base-learner names (default the Stacking
            terna).
        n_spatial_folds: Geographic sub-folds of fold-5 for the OOF CV (default 5).
        buffer_km: Inter-fold exclusion buffer in km for the spatial split.
        oof_cv_metrics_: Aggregated spatial-CV F1-macro/accuracy of the weighted
            vote (the leakage-free estimate that sits next to Stacking's).
        weights: The final convex weights, refit on ALL fold-5 OOF rows (for the
            ``predict_proba`` production path inherited from Blending).
    """

    def __init__(
        self,
        base_members: Sequence[str] = DEFAULT_WEIGHTED_VOTING_MEMBERS,
        *,
        n_spatial_folds: int = 5,
        buffer_km: float = 1.0,
        n_restarts: int = _N_RESTARTS,
        **kw: object,
    ) -> None:
        """Initialize the weighted voting ensemble.

        Args:
            base_members: Ordered parcel base learners to vote over (default the
                Stacking terna ``tsvit-pheno`` + ``utae`` + ``xgb-alphaearth``).
                Each must have a ``oof_parcel_{member}_fold5.parquet`` artifact.
            n_spatial_folds: Number of geographic sub-folds of fold-5 for the OOF
                cross-validation of the weights (default 5; mirrors Stacking).
            buffer_km: Inter-fold exclusion buffer in km (default 1.0).
            n_restarts: Number of Nelder-Mead restarts per weight optimization
                (default :data:`_N_RESTARTS`).
            **kw: Forwarded to :class:`BlendingEnsemble` /
                :class:`ml.ensemble.base.EnsembleModel` (``oof_dir``,
                ``random_state``). ``n_trials`` / ``gap_lambda`` are accepted but
                unused (this ensemble does not run Optuna).

        Raises:
            ValueError: if fewer than two members are given (a single member is
                not an ensemble) or ``n_spatial_folds < 2``.
        """
        # Blending's __init__ needs n_trials/gap_lambda; pass inert defaults (this
        # ensemble overrides fit and never runs the Optuna study).
        kw.setdefault("n_trials", 1)
        super().__init__(base_members, **kw)  # type: ignore[arg-type]
        if len(self.base_members) < 2:
            raise ValueError(
                f"WeightedVotingEnsemble needs at least 2 members, got "
                f"{self.base_members!r}; a single member is not an ensemble."
            )
        if n_spatial_folds < 2:
            raise ValueError(
                f"n_spatial_folds must be >= 2 for a spatial CV; got {n_spatial_folds}."
            )
        self.n_spatial_folds = int(n_spatial_folds)
        self.buffer_km = float(buffer_km)
        self.n_restarts = int(n_restarts)
        self.oof_cv_metrics_: dict[str, float] = {}
        #: Per-sub-fold learned weights (diagnostic: how stable the vote is).
        self.subfold_weights_: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Weight learning (direct F1-macro maximization on the simplex).
    # ------------------------------------------------------------------

    def _learn_weights(
        self, probs: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Learn convex weights maximizing F1-macro of the weighted vote.

        Optimizes the simplex logits with :func:`scipy.optimize.minimize`
        (Nelder-Mead), multi-started from each member corner and the centroid to
        escape the non-convex, piecewise-constant F1 surface. The logits are
        mapped to the simplex by :meth:`_project_simplex` (shared with Blending),
        so every evaluated point is a valid convex weighting.

        Args:
            probs: Member tensor ``(n_members, n_parcels, 18)`` of post-softmax
                rows (the TRAIN side of a sub-fold).
            labels: Aligned class ids ``(n_parcels,)``.

        Returns:
            The convex weights ``(n_members,)`` (``w_i >= 0``, ``sum(w) == 1``)
            with the best training F1-macro found.
        """
        from scipy.optimize import minimize

        n_members = probs.shape[0]

        def neg_f1(raw: np.ndarray) -> float:
            weights = self._project_simplex(raw)
            return -self._f1_of(probs, labels, weights)

        # Start points: each member corner (one-hot logits) + the uniform centroid.
        starts: list[np.ndarray] = []
        for i in range(n_members):
            corner = np.full(n_members, 0.05, dtype=np.float64)
            corner[i] = 1.0
            starts.append(corner)
        starts.append(np.full(n_members, 1.0 / n_members, dtype=np.float64))
        # Extra restarts (if requested) from deterministic jitters of the centroid
        # so the search is reproducible without Math.random/seeded RNG churn.
        for r in range(max(0, self.n_restarts - len(starts))):
            jitter = np.full(n_members, 1.0 / n_members, dtype=np.float64)
            jitter[r % n_members] += 0.3
            starts.append(jitter)

        best_raw: np.ndarray | None = None
        best_neg = np.inf
        for x0 in starts:
            res = minimize(
                neg_f1,
                x0,
                method="Nelder-Mead",
                options={"maxiter": _MAX_ITER, "xatol": 1e-4, "fatol": 1e-4},
            )
            if float(res.fun) < best_neg:
                best_neg = float(res.fun)
                best_raw = np.asarray(res.x, dtype=np.float64)

        assert best_raw is not None  # starts is non-empty
        return self._project_simplex(best_raw)

    # ------------------------------------------------------------------
    # Fit: spatial K-fold CV (Stacking's protocol) + final refit.
    # ------------------------------------------------------------------

    def fit(  # type: ignore[override]
        self,
        parcel_geoms: gpd.GeoDataFrame,
        *,
        y_true: pl.DataFrame,
        buffer_km: float | None = None,
    ) -> WeightedVotingEnsemble:
        """Learn the convex weights with Stacking's spatial K-fold CV.

        Pipeline (every step leakage-guarded, mirroring Stacking):

        1. Align the parcel members and labels (shared Blending machinery).
        2. Partition the fold-5 parcels into geographic sub-folds via
           :meth:`EnsembleModel.spatial_subfolds` (NEVER random).
        3. For each sub-fold ``k``: learn the ``N`` weights on the OOF rows of the
           OTHER sub-folds (by F1-macro maximization) and score the weighted vote
           on ``k``; :meth:`assert_oof_only` hard-fails any train/eval overlap.
           Aggregate the per-sub-fold F1-macro/accuracy into
           :attr:`oof_cv_metrics_` (the leakage-free estimate compared to E3).
        4. Refit the weights on ALL fold-5 OOF rows and cache the aligned member
           probabilities so the inherited :meth:`predict_proba` works.

        Args:
            parcel_geoms: GeoDataFrame of the fold-5 parcels with ``parcel_id``
                (integer surrogate), ``canonical_parcel_id`` (Utf8, matching the
                OOF members) and an active ``geometry`` in EPSG:4326.
            y_true: PASTIS-R ground truth with ``canonical_parcel_id`` + ``label``
                (the OOF dump discards the target).
            buffer_km: Override of the inter-fold buffer (km); defaults to
                :attr:`buffer_km`.

        Returns:
            ``self`` (fitted), with :attr:`oof_cv_metrics_`, :attr:`weights` and
            the cached member probabilities populated.

        Raises:
            ValueError: if the members/labels do not align or the spatial
                sub-folds produce no usable split.
        """
        from ml.ensemble.base import EnsembleModel

        buf = self.buffer_km if buffer_km is None else float(buffer_km)

        parcel_ids, member_probs = self._align_members()
        self._member_ids = parcel_ids
        self._member_probs = member_probs
        labels = self._labels_for(parcel_ids, y_true)

        splits = self._cv_splits(parcel_ids, parcel_geoms, buffer_km=buf)
        keys = list(parcel_ids)

        per_fold: list[dict[str, float]] = []
        self.subfold_weights_ = []
        for fold_idx, (train_pos, test_pos) in enumerate(splits):
            # Anti-leakage HARD GUARD: train and eval parcels must be disjoint.
            EnsembleModel.assert_oof_only(
                [keys[i] for i in train_pos],
                [keys[i] for i in test_pos],
                context=f"weighted-vote sub-fold {fold_idx}",
            )
            weights = self._learn_weights(
                member_probs[:, train_pos, :], labels[train_pos]
            )
            self.subfold_weights_.append(weights)
            blended_test = self._blend(member_probs[:, test_pos, :], weights)
            preds = blended_test.argmax(axis=-1)
            fold_metrics = EnsembleModel.compute_metrics(
                labels[test_pos], preds, ignore_index=None
            )
            per_fold.append(fold_metrics)
            logger.info(
                "weighted_vote_subfold_done",
                fold=f"{fold_idx + 1}/{len(splits)}",
                n_train=int(train_pos.size),
                n_test=int(test_pos.size),
                f1_macro=round(fold_metrics["f1_macro"], 4),
                weights=[round(float(w), 3) for w in weights],
            )

        self.oof_cv_metrics_ = _aggregate_metrics(per_fold)

        # Final refit on ALL fold-5 OOF rows for the production predict_proba.
        self._weights = self._learn_weights(member_probs, labels)
        self.best_params = {
            f"w_{i}": float(w) for i, w in enumerate(self._weights)
        }
        logger.info(
            "weighted_vote_fit_done",
            n_members=len(self.base_members),
            n_parcels=len(parcel_ids),
            n_subfolds=len(splits),
            f1_macro_oof=round(self.oof_cv_metrics_.get("f1_macro", float("nan")), 4),
            weights_final=[round(float(w), 4) for w in self._weights],
        )
        return self

    def _cv_splits(
        self,
        parcel_ids: Sequence[str],
        parcel_geoms: gpd.GeoDataFrame,
        *,
        buffer_km: float,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Map the fold-5 spatial sub-folds onto positional meta-row indices.

        Mirrors :meth:`ml.ensemble.stacking.StackingEnsemble._subfolds_by_canonical_id`
        but keyed on the aligned member order returned by :meth:`_align_members`:
        each sub-fold's ``test_ids`` becomes the held-out positions and the rest
        of the sub-folds become the train positions, so the weights are learned on
        OOF rows the held-out parcels never contributed to.

        Args:
            parcel_ids: Aligned canonical parcel ids (member/label order).
            parcel_geoms: GeoDataFrame of the fold-5 parcels (``parcel_id`` int
                surrogate + ``canonical_parcel_id`` + geometry, EPSG:4326).
            buffer_km: Inter-fold exclusion buffer (km).

        Returns:
            List of ``(train_pos, test_pos)`` positional index arrays, one per
            non-empty spatial sub-fold. Train and test are disjoint by
            construction (the buffer excludes border parcels from both).

        Raises:
            ValueError: if no parcel geometry matches the meta rows or the
                sub-folds produce no usable split.
        """
        geoms = parcel_geoms.copy()
        geoms["canonical_parcel_id"] = geoms["canonical_parcel_id"].astype(str)
        pos_by_key = {pid: i for i, pid in enumerate(parcel_ids)}
        geoms = geoms[geoms["canonical_parcel_id"].isin(set(pos_by_key))]
        if len(geoms) == 0:
            raise ValueError(
                "no parcel geometry matches the aligned member rows; check the "
                "canonical_parcel_id namespace."
            )
        int_to_canonical = dict(
            zip(
                geoms["parcel_id"].astype("int64").tolist(),
                geoms["canonical_parcel_id"].tolist(),
                strict=True,
            )
        )

        assignments = self.spatial_subfolds(
            geoms, n_folds=self.n_spatial_folds, buffer_km=buffer_km
        )
        n_rows = len(parcel_ids)
        all_pos = np.arange(n_rows, dtype=np.int64)
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for fold in assignments:
            test_int = set(fold.test_ids)
            train_int = set(fold.train_ids) | set(fold.val_ids)
            test_pos = np.array(
                sorted(
                    pos_by_key[int_to_canonical[i]]
                    for i in test_int
                    if i in int_to_canonical
                ),
                dtype=np.int64,
            )
            train_pos = np.array(
                sorted(
                    pos_by_key[int_to_canonical[i]]
                    for i in train_int
                    if i in int_to_canonical
                ),
                dtype=np.int64,
            )
            test_pos = test_pos[np.isin(test_pos, all_pos)]
            train_pos = train_pos[np.isin(train_pos, all_pos)]
            if test_pos.size == 0 or train_pos.size == 0:
                continue
            splits.append((train_pos, test_pos))

        if not splits:
            raise ValueError(
                "the spatial sub-folds of fold-5 produced no usable train/test "
                "split; reduce n_spatial_folds or buffer_km."
            )
        return splits

    # ------------------------------------------------------------------
    # MLflow params (extend Blending's with the learned weights).
    # ------------------------------------------------------------------

    def mlflow_params(self) -> dict[str, object]:
        """Return the params logged to MLflow for this weighted-vote run.

        Returns:
            A mapping with the members, the spatial-CV config and the learned
            per-member weights (only once fitted).
        """
        params: dict[str, object] = {
            "members": ",".join(self.base_members),
            "n_spatial_folds": self.n_spatial_folds,
            "buffer_km": self.buffer_km,
            "combiner": "weighted_vote_f1max",
        }
        if self._weights is not None:
            for member, weight in zip(self.base_members, self._weights, strict=True):
                params[f"weight_{member}"] = round(float(weight), 6)
        return params


def _aggregate_metrics(per_fold: list[dict[str, float]]) -> dict[str, float]:
    """Average the per-sub-fold metrics into a single mean estimate.

    Args:
        per_fold: List of ``{"f1_macro": ..., "accuracy": ...}`` dicts.

    Returns:
        ``{"f1_macro": mean, "accuracy": mean}``; ``nan`` if there were no
        sub-folds.
    """
    if not per_fold:
        return {"f1_macro": float("nan"), "accuracy": float("nan")}
    keys = per_fold[0].keys()
    return {key: float(np.mean([fold[key] for fold in per_fold])) for key in keys}
