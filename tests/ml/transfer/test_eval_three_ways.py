"""Mechanics tests for the three-via Italian TL comparison (US-082).

These fixtures are DELIBERATELY tiny and synthetic -- they exist ONLY to verify
the wiring of the A/B/C comparison (label-space collapse, gate counting, table
shape), NOT to produce any headline metric. The real macro-F1 / per-class numbers
come exclusively from the full 1,438-patch extraction run on the H100 (Arthur's
"datos 100% reales" rule). A perfect-prediction fixture is used so the asserted
F1 values are exact and deterministic.
"""

from __future__ import annotations

import numpy as np

from ml.transfer.eval_three_ways import (
    ThreeWayComparison,
    compare_three_ways,
    count_classes_over,
)
from ml.transfer.italia_label_space import ItaliaLabelSpace


def _toy_label_space() -> ItaliaLabelSpace:
    """A 4-class (+ background) toy space: 2 conserved (PASTIS) + 2 new.

    Both conserved leaves map to the SAME PASTIS parent so VIA B collapses them
    into one coarse class -- exactly the "the champion already knew this crop"
    case the via is meant to expose.
    """
    leaves = ("__background__", "soft_wheat", "durum_wheat", "olive", "vineyards")
    return ItaliaLabelSpace(
        leaves=leaves,
        class_ids=(1, 2, 3, 4),
        conserved=("soft_wheat", "durum_wheat"),
        new=("olive", "vineyards"),
        leaf_to_pastis={"soft_wheat": "Cereals", "durum_wheat": "Cereals"},
    )


def _perfect_preds() -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Two 2x2 patches whose prediction equals the ground truth (F1 == 1.0)."""
    mask0 = np.array([[1, 2], [3, 4]], dtype=np.int64)
    mask1 = np.array([[1, 1], [4, 0]], dtype=np.int64)  # 0 = background, ignored
    return {0: mask0.copy(), 1: mask1.copy()}, {0: mask0, 1: mask1}


def test_count_classes_over_gate() -> None:
    rows = [{"f1": 0.9}, {"f1": 0.6}, {"f1": 0.55}, {"f1": 0.81}]
    assert count_classes_over(rows, 0.6) == 3
    assert count_classes_over(rows, 0.8) == 2


def test_via_a_native_scores_all_present_leaves() -> None:
    preds, masks = _perfect_preds()
    cmp = compare_three_ways(
        "toy", preds, masks, label_space=_toy_label_space(), is_full_procedure=False
    )
    assert isinstance(cmp, ThreeWayComparison)
    # Perfect prediction -> macro-F1 == 1.0 over the present crop leaves.
    assert cmp.via_a.via == "A"
    assert cmp.via_a.macro_f1 == 1.0
    # Classes present (background excluded): 1,2,3,4 -> 4 scored.
    assert cmp.via_a.n_classes_scored == 4
    assert cmp.via_a.classes_over["0.6"] == 4


def test_via_b_collapses_conserved_into_one_pastis_bucket() -> None:
    preds, masks = _perfect_preds()
    cmp = compare_three_ways(
        "toy", preds, masks, label_space=_toy_label_space(), is_full_procedure=False
    )
    # soft_wheat + durum_wheat both -> "Cereals", so VIA B has FEWER classes than A.
    assert cmp.via_b.via == "B"
    assert cmp.via_b.n_classes_scored < cmp.via_a.n_classes_scored
    # Still a perfect prediction -> coarse macro-F1 == 1.0.
    assert cmp.via_b.macro_f1 == 1.0


def test_via_c_only_emitted_for_full_procedure() -> None:
    preds, masks = _perfect_preds()
    ls = _toy_label_space()
    no_c = compare_three_ways("toy", preds, masks, label_space=ls, is_full_procedure=False)
    assert no_c.via_c is None
    assert len(no_c.table()) == 2  # A + B only

    with_c = compare_three_ways("toy", preds, masks, label_space=ls, is_full_procedure=True)
    assert with_c.via_c is not None
    assert with_c.via_c.via == "C"
    assert len(with_c.table()) == 3  # A + B + C


def test_table_rows_are_flat_and_carry_gate_counts() -> None:
    preds, masks = _perfect_preds()
    cmp = compare_three_ways(
        "toy", preds, masks, label_space=_toy_label_space(), is_full_procedure=True
    )
    for row in cmp.table():
        assert set(row) >= {
            "via",
            "label_space",
            "macro_f1",
            "n_classes_scored",
            "n_classes_ge_0.6",
            "n_classes_ge_0.8",
        }
