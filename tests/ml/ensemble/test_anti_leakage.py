"""Consolidated anti-leakage tests for the four ensembles (US-040, R-LEAK).

This is the single most important test module of US-040 (plan Section 9, R-LEAK).
It does NOT re-test each ensemble's mechanics (those live in ``test_voting.py``,
``test_bagging.py``, ``test_stacking.py``, ``test_blending.py``); instead it
asserts the FOUR cross-cutting anti-leakage invariants HOLD TOGETHER across every
ensemble at once, so a regression in any one of them is caught here:

1. **Report fold-5 ONLY, never fold-4.** ``evaluate(fold=4)`` raises a
   ``ValueError`` on every ensemble (fold-4 was the selection fold).
2. **Probabilities, not logits.** Every ensemble validates its inputs/outputs are
   post-softmax (non-negative, sum-to-1); a logits array is rejected.
3. **Meta-learner sees OOF only (stacking).** The meta train and eval parcel sets
   of every spatial sub-fold are disjoint (no base learner contributes a meta
   feature for a parcel it would predict in the held-out sub-fold).
4. **Blending holdout spatially disjoint.** The weights are optimized on a
   geographically disjoint holdout (``build_spatial_kfold``), not a random split.

All data is tiny deterministic synthetic OOF (no checkpoint, no PASTIS-R load):
the readers and the spatial split run end-to-end on format-faithful fixtures.

Project conventions: ``polars`` (never pandas), ``numpy`` only at the array
boundary, type hints and Google-style docstrings; no emojis.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
import pytest
from shapely.geometry import Point

from ml.ensemble.bagging import BaggingEnsemble
from ml.ensemble.base import EnsembleModel
from ml.ensemble.blending import BlendingEnsemble
from ml.ensemble.stacking import StackingEnsemble
from ml.ensemble.voting import VotingEnsemble
from ml.utils.parcel_reconcile import PROB_COLUMNS
from tests.ml.ensemble.fixtures.synthetic_oof import (
    NUM_CLASSES,
    write_pixel_oof,
)

#: Heterogeneous parcel base members (stacking / blending).
_PARCEL_MEMBERS: tuple[str, ...] = ("tsvit-pheno", "utae", "xgb-alphaearth")

#: Dense voting members.
_VOTING_MEMBERS: tuple[str, ...] = ("tsvit-pheno", "utae", "unet")

#: Three well-separated geographic clusters so build_spatial_kfold forms folds.
_CLUSTERS: tuple[tuple[float, float], ...] = ((2.0, 44.0), (3.5, 45.5), (5.0, 47.0))


# ---------------------------------------------------------------------------
# Deterministic synthetic builders (parcel OOF + GT + geometry).
# ---------------------------------------------------------------------------


def _post_softmax(n: int, *, seed: int, signal: np.ndarray | None = None) -> np.ndarray:
    """Build an ``(n, 18)`` post-softmax matrix (optionally with class signal)."""
    rng = np.random.default_rng(seed)
    logits = rng.uniform(-1.0, 1.0, size=(n, NUM_CLASSES))
    if signal is not None:
        logits[np.arange(n), signal] += 3.0
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _write_parcel_members(
    oof_dir: Path, members: tuple[str, ...], ids: list[str], labels: np.ndarray, *, seed: int
) -> None:
    """Write a parcel OOF parquet per member over the SAME parcel id set."""
    for m, member in enumerate(members):
        probs = _post_softmax(len(ids), seed=seed + 100 * (m + 1), signal=labels)
        data: dict[str, object] = {"canonical_parcel_id": ids}
        for c, col in enumerate(PROB_COLUMNS):
            data[col] = probs[:, c].astype(np.float32)
        data["pred_class"] = probs.argmax(axis=1).astype(np.int64)
        data["n_pixels"] = np.full(len(ids), 100, dtype=np.int64)
        pl.DataFrame(data).write_parquet(oof_dir / f"oof_parcel_{member}_fold5.parquet")


def _make_parcel_fixture(
    oof_dir: Path, *, n_parcels: int = 60, seed: int = 0
) -> tuple[list[str], np.ndarray, pl.DataFrame, gpd.GeoDataFrame]:
    """Write parcel OOF for the 3 members + return (ids, labels, gt, geoms).

    Returns:
        Tuple ``(ids, labels, gt_labels, geoms)`` where ``gt_labels`` is a Polars
        ``canonical_parcel_id`` + ``label`` frame and ``geoms`` is a GeoDataFrame
        with an integer ``parcel_id`` surrogate + ``canonical_parcel_id`` + Point
        geometry in three separated clusters.
    """
    oof_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ids = [f"10000_{i:04d}" for i in range(n_parcels)]
    labels = rng.integers(0, 4, size=n_parcels).astype(np.int64)
    _write_parcel_members(oof_dir, _PARCEL_MEMBERS, ids, labels, seed=seed)

    gt_labels = pl.DataFrame({"canonical_parcel_id": ids, "label": labels})

    rows: list[dict[str, object]] = []
    for i, canonical in enumerate(ids):
        cx, cy = _CLUSTERS[i % len(_CLUSTERS)]
        jitter = rng.uniform(-0.04, 0.04, size=2)
        rows.append(
            {
                "parcel_id": i + 1,
                "canonical_parcel_id": canonical,
                "geometry": Point(cx + float(jitter[0]), cy + float(jitter[1])),
            }
        )
    geoms = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    return ids, labels, gt_labels, geoms


def _geoms_polars(geoms: gpd.GeoDataFrame) -> pl.DataFrame:
    """Convert the GeoDataFrame to the WKT Polars frame stacking expects."""
    return pl.DataFrame(
        {
            "canonical_parcel_id": geoms["canonical_parcel_id"].tolist(),
            "geometry": [g.wkt for g in geoms["geometry"]],
        }
    )


# ===========================================================================
# Invariant 1: report fold-5 only, NEVER fold-4 (every ensemble).
# ===========================================================================


def test_report_fold5_not_fold4_voting(tmp_path: Path) -> None:
    """VotingEnsemble.evaluate rejects fold-4 (selection) and accepts fold-5."""
    for member in _VOTING_MEMBERS:
        write_pixel_oof(tmp_path, member, patch_ids=("10000",), seed=1)
    ens = VotingEnsemble(_VOTING_MEMBERS, oof_dir=tmp_path)
    y_true = np.zeros(8 * 8, dtype=np.int64)
    y_pred = np.zeros(8 * 8, dtype=np.int64)
    with pytest.raises(ValueError, match="fold-5-only"):
        ens.evaluate(y_true=y_true, y_pred=y_pred, fold=4)
    # fold-5 is accepted.
    metrics = ens.evaluate(y_true=y_true, y_pred=y_pred, fold=5)
    assert set(metrics) == {"f1_macro", "accuracy"}


def test_report_fold5_not_fold4_all_parcel_ensembles(tmp_path: Path) -> None:
    """Bagging / Stacking / Blending all reject fold-4 via the base evaluate."""
    ids, labels, _, _ = _make_parcel_fixture(tmp_path, n_parcels=12, seed=2)
    proba = _post_softmax(len(ids), seed=999, signal=labels)
    for ens in (
        BaggingEnsemble(oof_dir=tmp_path),
        StackingEnsemble(_PARCEL_MEMBERS, oof_dir=tmp_path),
        BlendingEnsemble(_PARCEL_MEMBERS, oof_dir=tmp_path),
    ):
        with pytest.raises(ValueError, match="fold-5-only"):
            ens.evaluate(y_true=labels, proba=proba, fold=4)
        # fold-5 (default) is accepted.
        metrics = ens.evaluate(y_true=labels, proba=proba, fold=5)
        assert set(metrics) == {"f1_macro", "accuracy"}


def test_held_out_fold_is_five_on_every_class() -> None:
    """The fold constant is 5 on the base and every subclass (no override)."""
    assert EnsembleModel.HELD_OUT_FOLD == 5
    assert VotingEnsemble.HELD_OUT_FOLD == 5
    assert BaggingEnsemble.HELD_OUT_FOLD == 5
    assert StackingEnsemble.HELD_OUT_FOLD == 5
    assert BlendingEnsemble.HELD_OUT_FOLD == 5


# ===========================================================================
# Invariant 2: probabilities, NOT logits (post-softmax everywhere).
# ===========================================================================


def test_logits_rejected_by_validate_probs() -> None:
    """Raw logits (negative, not sum-to-1) are rejected by validate_probs."""
    logits = np.array([[-2.0, 0.5, 3.0], [1.0, 1.0, 1.0]])  # neither row sums to 1
    with pytest.raises(ValueError, match=r"negative|sum to 1|logits"):
        EnsembleModel.validate_probs(logits, class_axis=-1, name="logits")


def test_voting_output_is_post_softmax(tmp_path: Path) -> None:
    """The voting mean is post-softmax (sum-to-1 over the class axis)."""
    for member in _VOTING_MEMBERS:
        write_pixel_oof(tmp_path, member, patch_ids=("10000",), seed=3)
    ens = VotingEnsemble(_VOTING_MEMBERS, oof_dir=tmp_path)
    proba = ens.predict_proba(["10000"])
    sums = proba.sum(axis=0)
    np.testing.assert_allclose(sums, 1.0, atol=1e-5)
    assert (proba >= 0.0).all()


def test_parcel_ensembles_outputs_are_post_softmax(tmp_path: Path) -> None:
    """Stacking + Blending return post-softmax matrices (validated by the base)."""
    _ids, _labels, gt, geoms = _make_parcel_fixture(tmp_path, n_parcels=60, seed=4)

    stacking = StackingEnsemble(_PARCEL_MEMBERS, oof_dir=tmp_path).fit(
        _geoms_polars(geoms), gt_labels=gt
    )
    proba_stack = stacking.predict_proba()
    np.testing.assert_allclose(proba_stack.sum(axis=1), 1.0, atol=1e-5)
    assert (proba_stack >= 0.0).all()

    blending = BlendingEnsemble(_PARCEL_MEMBERS, n_trials=6, oof_dir=tmp_path).fit(geoms, y_true=gt)
    proba_blend = blending.predict_proba()
    np.testing.assert_allclose(proba_blend.sum(axis=1), 1.0, atol=1e-5)
    assert (proba_blend >= 0.0).all()


def test_blend_of_post_softmax_stays_post_softmax(tmp_path: Path) -> None:
    """A convex combination of distributions is itself a distribution."""
    _ids, _labels, gt, geoms = _make_parcel_fixture(tmp_path, n_parcels=60, seed=5)
    blending = BlendingEnsemble(_PARCEL_MEMBERS, n_trials=4, oof_dir=tmp_path).fit(geoms, y_true=gt)
    w = blending.weights
    assert (w >= 0.0).all()
    assert abs(float(w.sum()) - 1.0) < 1e-6


# ===========================================================================
# Invariant 3: meta-learner sees OOF only (stacking) -- disjoint sub-folds.
# ===========================================================================


def test_stacking_meta_subfolds_disjoint(tmp_path: Path) -> None:
    """Every stacking spatial sub-fold has disjoint meta train/eval parcels."""
    _ids, _labels, gt, geoms = _make_parcel_fixture(tmp_path, n_parcels=60, seed=6)
    ens = StackingEnsemble(_PARCEL_MEMBERS, oof_dir=tmp_path)
    keys_df, _x, _y = ens.build_meta_features(gt_labels=gt)
    splits = ens._subfolds_by_canonical_id(_geoms_polars(geoms), keys_df)
    assert splits, "spatial sub-folds must be non-empty"
    for train_pos, test_pos in splits:
        assert set(train_pos.tolist()).isdisjoint(set(test_pos.tolist()))


def test_stacking_assert_oof_only_hard_fails_on_overlap() -> None:
    """assert_oof_only raises when train and eval parcel sets intersect."""
    with pytest.raises(ValueError, match="leakage"):
        EnsembleModel.assert_oof_only(["10000_1", "10000_2"], ["10000_2", "10000_3"])
    # Disjoint sets pass silently.
    EnsembleModel.assert_oof_only(["10000_1"], ["10000_2"])


def test_stacking_meta_features_are_only_oof_probs(tmp_path: Path) -> None:
    """The meta feature matrix width == n_members * 18 (only OOF prob columns)."""
    _ids, _labels, gt, _geoms = _make_parcel_fixture(tmp_path, n_parcels=40, seed=7)
    ens = StackingEnsemble(_PARCEL_MEMBERS, oof_dir=tmp_path)
    _keys, x_meta, y = ens.build_meta_features(gt_labels=gt)
    assert x_meta.shape[1] == len(_PARCEL_MEMBERS) * NUM_CLASSES
    assert y is not None
    # The meta features are all post-softmax sub-blocks (no raw features/logits).
    for m in range(len(_PARCEL_MEMBERS)):
        block = x_meta[:, m * NUM_CLASSES : (m + 1) * NUM_CLASSES]
        np.testing.assert_allclose(block.sum(axis=1), 1.0, atol=1e-4)


# ===========================================================================
# Invariant 4: blending holdout spatially disjoint (build_spatial_kfold).
# ===========================================================================


def test_blending_holdout_spatially_disjoint(tmp_path: Path) -> None:
    """The blending holdout yields disjoint, non-empty train/val indices."""
    _ids, _labels, _gt, geoms = _make_parcel_fixture(tmp_path, n_parcels=60, seed=8)
    ens = BlendingEnsemble(_PARCEL_MEMBERS, n_trials=4, oof_dir=tmp_path)
    parcel_ids, _probs = ens._align_members()
    train_idx, val_idx = ens._spatial_holdout(parcel_ids, geoms, buffer_km=1.0)
    assert train_idx.size > 0
    assert val_idx.size > 0
    assert set(train_idx.tolist()).isdisjoint(set(val_idx.tolist()))


def test_blending_holdout_uses_build_spatial_kfold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The blending holdout is sourced from build_spatial_kfold, never random."""
    _ids, _labels, _gt, geoms = _make_parcel_fixture(tmp_path, n_parcels=60, seed=9)
    ens = BlendingEnsemble(_PARCEL_MEMBERS, n_trials=4, oof_dir=tmp_path)
    parcel_ids, _probs = ens._align_members()

    import ml.features.spatial_split as ss

    calls: list[int] = []
    original = ss.build_spatial_kfold

    def spy(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(ss, "build_spatial_kfold", spy)
    ens._spatial_holdout(parcel_ids, geoms, buffer_km=1.0)
    assert calls, "build_spatial_kfold was not used (a random split would leak)."


def test_stacking_subfolds_use_build_spatial_kfold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stacking sub-folds are sourced from build_spatial_kfold, never random."""
    _ids, _labels, gt, geoms = _make_parcel_fixture(tmp_path, n_parcels=60, seed=10)
    ens = StackingEnsemble(_PARCEL_MEMBERS, oof_dir=tmp_path)
    keys_df, _x, _y = ens.build_meta_features(gt_labels=gt)

    import ml.features.spatial_split as ss

    calls: list[int] = []
    original = ss.build_spatial_kfold

    def spy(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(ss, "build_spatial_kfold", spy)
    ens._subfolds_by_canonical_id(_geoms_polars(geoms), keys_df)
    assert calls, "build_spatial_kfold was not used (a random split would leak)."
