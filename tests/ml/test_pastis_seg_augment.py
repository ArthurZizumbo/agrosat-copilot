"""Tests de las palancas anti-desbalance del Avance 5 (US-029).

Cubre las tres piezas nuevas que cierran parte de la brecha F1-macro del mejor
modelo (TSViT) sin reentrenar la arquitectura:

- :func:`ml.data.pastis_seg_dataset.apply_synchronized_augment`: augmentacion
  geometrica D4 que DEBE transformar imagen y mascara de forma identica (un
  desalineo silencioso corromperia el entrenamiento).
- :func:`ml.train.train_segmentation._class_weights_from_counts`: derivacion de
  pesos por clase (effective-number / inverse-frequency) que sube las
  minoritarias sin que las clases ausentes capturen el peso maximo.
- :meth:`ml.eval.dense_metrics.DenseConfusionAccumulator.per_class_metrics`:
  tabla per-clase para el error-analysis del model card.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.data.pastis_seg_dataset import apply_synchronized_augment
from ml.eval.dense_metrics import DenseConfusionAccumulator
from ml.train.train_segmentation import _class_weights_from_counts


def test_augment_hflip_sincroniza_imagen_y_mascara() -> None:
    """El flip horizontal mueve imagen y mascara a la vez (eje W)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 4, 5)).astype(np.float32)  # (T, C, H, W)
    y = np.zeros((4, 5), dtype=np.int64)
    y[0, 0] = 7  # marcador en la esquina superior-izquierda

    x_aug, y_aug = apply_synchronized_augment(
        x, y, hflip=True, vflip=False, rot_k=0
    )

    assert y_aug[0, -1] == 7  # la marca viaja a la columna espejo
    np.testing.assert_array_equal(x_aug, x[..., ::-1])
    np.testing.assert_array_equal(y_aug, y[:, ::-1])


def test_augment_rot90_consistente_imagen_y_mascara() -> None:
    """rot90 aplica la MISMA rotacion al plano (H, W) de x y de y."""
    x = np.arange(2 * 1 * 3 * 3, dtype=np.float32).reshape(2, 1, 3, 3)
    y = np.arange(9, dtype=np.int64).reshape(3, 3)

    x_aug, y_aug = apply_synchronized_augment(
        x, y, hflip=False, vflip=False, rot_k=1
    )

    np.testing.assert_array_equal(x_aug, np.rot90(x, k=1, axes=(-2, -1)))
    np.testing.assert_array_equal(y_aug, np.rot90(y, k=1, axes=(-2, -1)))
    assert x_aug.shape[-2:] == y_aug.shape


def test_augment_identidad_no_modifica() -> None:
    """Sin flips ni rotacion, los arrays salen intactos (y contiguos)."""
    x = np.ones((3, 4, 4), dtype=np.float32)
    y = np.zeros((4, 4), dtype=np.int64)

    x_aug, y_aug = apply_synchronized_augment(
        x, y, hflip=False, vflip=False, rot_k=0
    )

    np.testing.assert_array_equal(x_aug, x)
    np.testing.assert_array_equal(y_aug, y)
    assert x_aug.flags["C_CONTIGUOUS"]


def test_class_weights_effective_sube_minoritarias_y_neutraliza_ausentes() -> None:
    """effective-number: minoritaria > mayoritaria; ausente queda neutra (1.0)."""
    counts = np.array([1000, 100, 10, 0], dtype=np.int64)  # clase 3 ausente
    w = _class_weights_from_counts(counts, scheme="effective", beta=0.9999)

    assert w.shape == (4,)
    assert w[2] > w[0]  # la rara pesa mas que la abundante
    assert w[3] == pytest.approx(1.0)  # ausente neutralizada (no peso maximo)
    assert w[3] < w[2]
    assert w.min() >= 0.5 - 1e-6
    assert w.max() <= 4.0 + 1e-6


def test_class_weights_inverse_preserva_orden() -> None:
    """inverse-frequency: a menor frecuencia, mayor peso (dentro del clip)."""
    counts = np.array([400, 200, 100], dtype=np.int64)  # ratios 4:2:1
    w = _class_weights_from_counts(counts, scheme="inverse", beta=0.9999)

    assert w[2] > w[1] > w[0]
    assert w.min() >= 0.5 - 1e-6
    assert w.max() <= 4.0 + 1e-6


def test_class_weights_scheme_invalido() -> None:
    """Un esquema no soportado falla explicitamente."""
    with pytest.raises(ValueError, match="class_balance invalido"):
        _class_weights_from_counts(
            np.array([1, 2, 3], dtype=np.int64), scheme="softmax", beta=0.99
        )


def test_per_class_metrics_synthetic() -> None:
    """per_class_metrics deriva recall/precision/soporte por clase y omite ignore."""
    acc = DenseConfusionAccumulator(num_classes=3, ignore_index=2)
    # clase 0: 8 aciertos + 2 confundidos con 1 (recall 0.8); clase 1: 5 aciertos.
    preds = torch.tensor([0] * 8 + [1] * 2 + [1] * 5)
    target = torch.tensor([0] * 10 + [1] * 5)
    acc.update(preds, target)

    rows = acc.per_class_metrics(class_names={0: "trigo", 1: "maiz"})
    by_id = {int(r["class_id"]): r for r in rows}

    assert set(by_id) == {0, 1}  # la clase 2 (ignore) no aparece
    assert by_id[0]["recall"] == pytest.approx(0.8)
    assert by_id[0]["precision"] == pytest.approx(1.0)
    assert by_id[0]["support"] == 10
    assert by_id[0]["name"] == "trigo"
    assert by_id[1]["recall"] == pytest.approx(1.0)
