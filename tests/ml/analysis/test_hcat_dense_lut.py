"""Tests del LUT denso 18 clases PASTIS -> 6 grupos HCAT (ml.analysis.hcat_grouping)."""

from __future__ import annotations

from ml.analysis.hcat_grouping import (
    HCAT_L1_GROUP_ORDER,
    PASTIS_CLASS_TO_HCAT_L1,
    hcat6_dense_lut,
)


def test_lut_shape_and_ignore() -> None:
    """El LUT tiene 20 entradas; fondo (0) y void (19) van a ignore_index."""
    lut = hcat6_dense_lut(ignore_index=255)
    assert lut.shape == (20,)
    assert lut[0] == 255
    assert lut[19] == 255
    # Las 18 clases de cultivo mapean al rango contiguo de 6 grupos.
    for class_id in range(1, 19):
        assert 0 <= lut[class_id] <= 5


def test_lut_matches_canonical_mapping() -> None:
    """Cada clase de cultivo cae en el grupo que dicta PASTIS_CLASS_TO_HCAT_L1."""
    lut = hcat6_dense_lut()
    order = {group: idx for idx, group in enumerate(HCAT_L1_GROUP_ORDER)}
    for class_id, group in PASTIS_CLASS_TO_HCAT_L1.items():
        assert lut[class_id] == order[group]


def test_cereals_collapse_to_single_group() -> None:
    """Los ocho cereales hermanos caen en el mismo grupo (el punto del agrupamiento)."""
    lut = hcat6_dense_lut()
    cereal_ids = [2, 3, 4, 6, 10, 11, 17, 18]
    assert len({int(lut[c]) for c in cereal_ids}) == 1
