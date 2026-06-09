"""Tests for :mod:`ml.eval.ensemble_figures` (US-040, A5 figures + table).

Each figure factory must return a valid :class:`pathlib.Path` to a written PNG on
tiny deterministic synthetic data (no PASTIS-R load); the comparison table must
have the Selection schema and elect exactly one model; the ROC/PR AUC/AP must be
computed and a logits input must be rejected (anti-leakage).

Project conventions: ``polars`` (never pandas), ``numpy`` only at the array
boundary, type hints; no emojis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from shapely.geometry import Point

from ml.eval.ensemble_figures import (
    build_comparison_table,
    confusion_norm_abs,
    pr_per_class,
    roc_ovr_per_class,
    spatial_residuals,
)

#: Class count of the harness 18-class space.
_NUM_CLASSES: int = 18


# ---------------------------------------------------------------------------
# Deterministic synthetic builders.
# ---------------------------------------------------------------------------


def _make_data(n: int = 80, *, n_present: int = 4, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Build aligned (y_true, proba) where proba carries signal on the true class.

    Args:
        n: Number of samples.
        n_present: Number of distinct classes present in y_true.
        seed: Deterministic seed.

    Returns:
        Tuple ``(y_true, proba)`` with ``y_true`` in ``[0, n_present)`` and
        ``proba`` an ``(n, 18)`` post-softmax matrix bumped on the true class
        (so the AUC/AP are well above chance).
    """
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, n_present, size=n).astype(np.int64)
    logits = rng.uniform(-1.0, 1.0, size=(n, _NUM_CLASSES))
    logits[np.arange(n), y_true] += 2.5
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    proba = exp / exp.sum(axis=1, keepdims=True)
    return y_true, proba


def _geoms(n: int, *, seed: int = 1) -> pl.DataFrame:
    """Build a per-parcel WKT geometry frame aligned to ``n`` predictions."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform([2.0, 44.0], [5.0, 47.0], size=(n, 2))
    return pl.DataFrame(
        {
            "canonical_parcel_id": [f"10000_{i:04d}" for i in range(n)],
            "geometry": [Point(float(x), float(y)).wkt for x, y in coords],
        }
    )


# ---------------------------------------------------------------------------
# Figure 1: confusion (normalized + absolute).
# ---------------------------------------------------------------------------


def test_confusion_norm_abs_writes_png(tmp_path: Path) -> None:
    """confusion_norm_abs returns a valid Path to a non-empty PNG."""
    y_true, proba = _make_data(seed=2)
    out = confusion_norm_abs(
        y_true, proba.argmax(axis=1), out_path=tmp_path / "cm.png", model="E3 Stacking"
    )
    assert isinstance(out, Path)
    assert out.exists()
    assert out.stat().st_size > 0


def test_confusion_accepts_label_names(tmp_path: Path) -> None:
    """A {id: name} label map is accepted (no crash, valid PNG)."""
    y_true, proba = _make_data(seed=3)
    names = {i: f"clase_{i}" for i in range(_NUM_CLASSES)}
    out = confusion_norm_abs(
        y_true, proba.argmax(axis=1), labels=names, out_path=tmp_path / "cm2.png"
    )
    assert out.exists()


# ---------------------------------------------------------------------------
# Figure 2: ROC one-vs-rest (AUC per class + macro).
# ---------------------------------------------------------------------------


def test_roc_ovr_returns_auc_and_png(tmp_path: Path) -> None:
    """roc_ovr_per_class returns a Path + AUCs including a macro key."""
    y_true, proba = _make_data(seed=4)
    out, aucs = roc_ovr_per_class(y_true, proba, out_path=tmp_path / "roc.png")
    assert out.exists()
    assert "macro" in aucs
    # Signal injected on the true class -> AUC well above chance.
    assert aucs["macro"] > 0.7
    # Every per-class AUC is a probability in [0, 1].
    for name, auc in aucs.items():
        assert 0.0 <= auc <= 1.0, name


def test_roc_rejects_logits(tmp_path: Path) -> None:
    """A non-post-softmax (logits) proba is rejected (anti-leakage)."""
    y_true, _ = _make_data(seed=5)
    logits = np.random.default_rng(5).uniform(-3.0, 3.0, size=(y_true.size, _NUM_CLASSES))
    with pytest.raises(ValueError, match=r"sum to 1|negative|logits"):
        roc_ovr_per_class(y_true, logits, out_path=tmp_path / "roc_bad.png")


def test_roc_rejects_misaligned_proba(tmp_path: Path) -> None:
    """A proba with the wrong row count is rejected."""
    y_true, proba = _make_data(n=80, seed=6)
    with pytest.raises(ValueError, match="aligned"):
        roc_ovr_per_class(y_true, proba[:40], out_path=tmp_path / "roc_bad2.png")


# ---------------------------------------------------------------------------
# Figure 3: precision-recall per class.
# ---------------------------------------------------------------------------


def test_pr_per_class_returns_ap_and_png(tmp_path: Path) -> None:
    """pr_per_class returns a Path + APs including a macro key above chance."""
    y_true, proba = _make_data(seed=7)
    out, aps = pr_per_class(y_true, proba, out_path=tmp_path / "pr.png")
    assert out.exists()
    assert "macro" in aps
    assert aps["macro"] > 0.5
    for name, ap in aps.items():
        assert 0.0 <= ap <= 1.0, name


# ---------------------------------------------------------------------------
# Figure 4: spatial residuals over real geometry.
# ---------------------------------------------------------------------------


def test_spatial_residuals_writes_png(tmp_path: Path) -> None:
    """spatial_residuals returns a valid Path to a non-empty PNG."""
    y_true, proba = _make_data(n=60, seed=8)
    geoms = _geoms(60, seed=8)
    out = spatial_residuals(geoms, y_true, proba.argmax(axis=1), out_path=tmp_path / "res.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_spatial_residuals_rejects_misaligned(tmp_path: Path) -> None:
    """Misaligned geometry/labels lengths are rejected."""
    y_true, proba = _make_data(n=60, seed=9)
    geoms = _geoms(40, seed=9)  # wrong length
    with pytest.raises(ValueError, match="aligned"):
        spatial_residuals(geoms, y_true, proba.argmax(axis=1), out_path=tmp_path / "r.png")


def test_spatial_residuals_requires_geometry(tmp_path: Path) -> None:
    """A frame without a geometry column is rejected."""
    y_true, proba = _make_data(n=10, seed=10)
    bad = pl.DataFrame({"canonical_parcel_id": [f"10000_{i}" for i in range(10)]})
    with pytest.raises(ValueError, match="geometry"):
        spatial_residuals(bad, y_true, proba.argmax(axis=1), out_path=tmp_path / "r2.png")


# ---------------------------------------------------------------------------
# Comparison table (Selection criterion).
# ---------------------------------------------------------------------------


def test_build_comparison_table_schema_and_chosen() -> None:
    """The table has the Selection schema and elects exactly one model."""
    results = {
        "TSViT-pheno (individual)": {"f1_macro": 0.6253, "accuracy": 0.80, "inference_time_s": 0.5},
        "E1 Voting (pixel)": {"f1_macro": 0.64, "accuracy": 0.81, "inference_time_s": 1.2},
        "E2 Bagging (parcela)": {"f1_macro": 0.57, "accuracy": 0.74, "inference_time_s": 0.3},
        "E3 Stacking (parcela)": {"f1_macro": 0.66, "accuracy": 0.83, "inference_time_s": 0.4},
        "E4 Blending (parcela)": {"f1_macro": 0.63, "accuracy": 0.80, "inference_time_s": 0.2},
    }
    table = build_comparison_table(results)
    assert table.columns == ["model", "f1_macro", "accuracy", "inference_time_s", "chosen"]
    assert table.height == 5
    # Exactly one chosen, and it is the F1-macro maximizer (Stacking).
    chosen = table.filter(pl.col("chosen"))
    assert chosen.height == 1
    assert chosen["model"][0] == "E3 Stacking (parcela)"
    # Sorted by f1_macro descending.
    f1 = table["f1_macro"].to_list()
    assert f1 == sorted(f1, reverse=True)


def test_comparison_table_tie_breaks_by_inference_time() -> None:
    """On an F1 tie the cheapest (lowest inference_time_s) model is chosen."""
    results = {
        "A": {"f1_macro": 0.66, "accuracy": 0.8, "inference_time_s": 1.0},
        "B": {"f1_macro": 0.66, "accuracy": 0.8, "inference_time_s": 0.2},
    }
    table = build_comparison_table(results)
    chosen = table.filter(pl.col("chosen"))
    assert chosen["model"][0] == "B"


def test_comparison_table_empty_raises() -> None:
    """An empty results mapping is rejected."""
    with pytest.raises(ValueError, match="empty"):
        build_comparison_table({})


def test_comparison_table_missing_f1_raises() -> None:
    """An entry without f1_macro is rejected."""
    with pytest.raises(ValueError, match="f1_macro"):
        build_comparison_table({"A": {"accuracy": 0.8}})
