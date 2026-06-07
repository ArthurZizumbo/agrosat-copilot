"""Tests de los helpers de espacio de clases del harness (ml.eval.class_remap, US-030).

Golden-value tests del computo puro 20->18 y del remuestreo 128 NEAREST,
siguiendo el patron de ``tests/ml/eval/test_dense_metrics.py``: sin checkpoints,
arrays sinteticos deterministas (seed fija) y asserts exactos por pixel.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.eval.class_remap import (
    HARNESS_IGNORE_INDEX,
    HARNESS_NUM_CLASSES,
    HARNESS_SIZE,
    remap_20_to_18,
    resample_mask_128_nearest,
)
from tests.ml.eval.fixtures.rescore_synthetic import (
    make_18class_pair,
    make_20class_pair,
    make_label_map_256,
)


def test_remap_20_to_18_reindex() -> None:
    """Clases 1..18 -> [0..17] exactas; Background(0) y Void(19) -> ignore_index."""
    labels = np.array(
        [[0, 1, 2, 18, 19], [19, 18, 3, 1, 0]],
        dtype=np.int64,
    )
    out = remap_20_to_18(labels)
    expected = np.array(
        [
            [HARNESS_IGNORE_INDEX, 0, 1, 17, HARNESS_IGNORE_INDEX],
            [HARNESS_IGNORE_INDEX, 17, 2, 0, HARNESS_IGNORE_INDEX],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(out, expected)


def test_remap_20_to_18_all_agronomic_shift() -> None:
    """Un mapa solo agronomico 1..18 se desplaza exactamente -1 en todo pixel."""
    preds, _ = make_20class_pair(background_frac=0.0, void_frac=0.0, seed=7)
    out = remap_20_to_18(preds)
    np.testing.assert_array_equal(out, preds.astype(np.int64) - 1)
    assert out.min() >= 0
    assert out.max() <= HARNESS_NUM_CLASSES - 1


def test_remap_20_to_18_background_void_to_ignore() -> None:
    """Todos los pixeles Background/Void terminan en ignore_index."""
    _, target = make_20class_pair(background_frac=0.2, void_frac=0.2, seed=3)
    out = remap_20_to_18(target)
    bg_void = (target == 0) | (target == 19)
    assert np.all(out[bg_void] == HARNESS_IGNORE_INDEX)
    assert np.all(out[~bg_void] != HARNESS_IGNORE_INDEX)


def test_remap_accepts_torch_tensor() -> None:
    """El helper acepta torch.Tensor y devuelve numpy equivalente."""
    labels = torch.tensor([[1, 19, 0, 18]], dtype=torch.long)
    out = remap_20_to_18(labels)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(
        out,
        np.array([[0, HARNESS_IGNORE_INDEX, HARNESS_IGNORE_INDEX, 17]], dtype=np.int64),
    )


def test_remap_idempotent_on_18() -> None:
    """``remap_20_to_18`` NO se aplica a mapas ya 18-contiguos (modelos nativos 18).

    El harness solo remapea los checkpoints de 20 clases; los nativos de 18
    (DeepLab, TSViT, TSViT-pheno) se acumulan sin remap. Este test documenta
    que el "no-op" correcto sobre un mapa ya en ``[0..17]`` es la identidad
    (no llamar al helper), y que aplicar el helper a un mapa 18-contiguo SI lo
    altera (shift adicional) -> de ahi que el harness deba no invocarlo.
    """
    preds, _ = make_18class_pair(ignore_frac=0.0, error_frac=0.0, seed=1)
    # Identidad: el modelo nativo 18 no pasa por remap.
    no_op = preds.astype(np.int64)
    np.testing.assert_array_equal(no_op, preds.astype(np.int64))
    # Y aplicar el helper a un 18-contiguo desplaza las clases agronomicas,
    # confirmando que aplicarlo seria incorrecto para nativos 18.
    remapped = remap_20_to_18(preds)
    has_agronomic = np.any((preds >= 1) & (preds <= HARNESS_NUM_CLASSES))
    if has_agronomic:
        assert not np.array_equal(remapped, preds.astype(np.int64))


def test_resample_128_nearest_shape() -> None:
    """Salida (128, 128) a partir de una entrada 256."""
    mask = make_label_map_256(size=256, block=16, seed=2)
    out = resample_mask_128_nearest(mask)
    assert out.shape == (HARNESS_SIZE, HARNESS_SIZE)


def test_resample_nearest_no_new_classes() -> None:
    """NEAREST no interpola: set(out) subset set(in)."""
    mask = make_label_map_256(size=256, n_classes=18, block=16, seed=5)
    out = resample_mask_128_nearest(mask)
    assert set(np.unique(out)).issubset(set(np.unique(mask)))


def test_resample_nearest_blocky_exact() -> None:
    """Downsample exacto de un mapa por bloques: cada tile conserva su clase."""
    mask = make_label_map_256(size=256, block=16, seed=9)
    out = resample_mask_128_nearest(mask, size=128)
    # 256/16 = 16 tiles -> en 128 cada tile mide 8 px; recuperamos la rejilla.
    recovered = out[::8, ::8]
    original_tiles = mask[::16, ::16]
    np.testing.assert_array_equal(recovered, original_tiles)


def test_resample_identity_when_already_128() -> None:
    """Una mascara ya (128, 128) se devuelve sin alterar valores."""
    rng = np.random.default_rng(11)
    mask = rng.integers(0, 18, size=(HARNESS_SIZE, HARNESS_SIZE)).astype(np.int64)
    out = resample_mask_128_nearest(mask)
    np.testing.assert_array_equal(out, mask)


def test_resample_accepts_torch_tensor() -> None:
    """El remuestreo acepta torch.Tensor y devuelve numpy int."""
    mask = torch.from_numpy(make_label_map_256(size=256, block=16, seed=4))
    out = resample_mask_128_nearest(mask)
    assert isinstance(out, np.ndarray)
    assert out.shape == (HARNESS_SIZE, HARNESS_SIZE)
    assert np.issubdtype(out.dtype, np.integer)


def test_resample_rejects_non_2d() -> None:
    """Un mapa no 2D dispara ValueError."""
    bad = np.zeros((1, 256, 256), dtype=np.int64)
    with pytest.raises(ValueError, match="2D"):
        resample_mask_128_nearest(bad)
