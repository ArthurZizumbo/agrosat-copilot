"""Tests para ``ml.data.sen4agrinet_adapter.Sen4AgriNetDataset`` (US-075).

Dos bloques, segun la Regla Arthur (cero sinteticos en lo que valida el
contrato real):

1. **Logica pura** (sin netCDF): el encoder FAO-ICC -> macro-HCAT, la
   interpolacion mensual y el tiling espacial. Estos NO leen ningun patch y
   verifican que un crop id conocido cae en su macro esperada (p.ej. ``330
   Grapes -> vineyard``), que el background/fuera-de-nomenclator va a
   ``ignore_index``, y que el rango de ids es contiguo.

2. **Patches reales** (``@pytest.mark.skipif``): si el subset ``data/sen4agrinet/``
   esta descargado, instancia ``Sen4AgriNetDataset`` sobre 1-2 patches REALES y
   verifica shape ``(T,10,H,W)``/``(H,W)``, dtypes, rango de bandas en ``[0,~1.5]``,
   labels en ``[0,num_classes) U {ignore}`` y compatibilidad con el forward de
   TSViT. NUNCA se fabrica un netCDF para falsear estas aserciones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ml.data.sen4agrinet_adapter import (
    FAO_ICC_TO_MACRO,
    IGNORE_INDEX,
    MACRO_GROUP_TO_ID,
    N_MACRO_CLASSES,
    Sen4AgriNetDataset,
    _interpolate_months,
    _tile_indices,
    build_fao_icc_lut,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SUBSET_ROOT = _REPO_ROOT / "data" / "sen4agrinet"
_subset_present = _SUBSET_ROOT.exists() and any(_SUBSET_ROOT.rglob("*.nc"))

_TILE = 128


# ---------------------------------------------------------------------------
# Bloque 1: logica pura del encoder / binning / tiling (sin datos)
# ---------------------------------------------------------------------------


def test_macro_id_space_is_contiguous_and_excludes_void() -> None:
    """Los ids macro son ``[0, N_MACRO_CLASSES)`` contiguos y sin ``void``."""
    ids = sorted(MACRO_GROUP_TO_ID.values())
    assert ids == list(range(N_MACRO_CLASSES))
    assert "void" not in MACRO_GROUP_TO_ID
    assert N_MACRO_CLASSES == 10


def test_fao_icc_lut_known_crops() -> None:
    """Crop ids FAO-ICC conocidos caen en su macro-HCAT esperada (no inventada)."""
    lut = build_fao_icc_lut()
    # 330 Grapes -> vineyard; 120 Maize -> cereals; 435 Rapeseed -> oilseed.
    assert lut[330] == MACRO_GROUP_TO_ID["vineyard"]
    assert lut[120] == MACRO_GROUP_TO_ID["cereals"]
    assert lut[435] == MACRO_GROUP_TO_ID["oilseed_industrial"]
    assert lut[510] == MACRO_GROUP_TO_ID["potato"]
    assert lut[770] == MACRO_GROUP_TO_ID["legumes_fodder"]
    # Every selected FAO-ICC code resolves to a valid contiguous id.
    for code, macro in FAO_ICC_TO_MACRO.items():
        assert lut[code] == MACRO_GROUP_TO_ID[macro]
        assert 0 <= lut[code] < N_MACRO_CLASSES


def test_encode_labels_background_and_unknown_go_to_ignore() -> None:
    """Background (0) y un crop fuera del nomenclator van a ``ignore_index``."""
    ds = Sen4AgriNetDataset.__new__(Sen4AgriNetDataset)  # bypass __init__ (no I/O)
    ds.ignore_index = IGNORE_INDEX
    ds._fao_lut = build_fao_icc_lut()
    labels = np.array([[0, 330], [120, 99999]], dtype=np.int64)  # 99999 = unknown
    out = ds._encode_labels(labels)
    assert out[0, 0] == IGNORE_INDEX  # background
    assert out[1, 1] == IGNORE_INDEX  # out of nomenclature
    assert out[0, 1] == MACRO_GROUP_TO_ID["vineyard"]  # 330
    assert out[1, 0] == MACRO_GROUP_TO_ID["cereals"]  # 120


def test_interpolate_months_fills_gaps_linearly() -> None:
    """Meses NaN se rellenan por interpolacion lineal y extrapolacion."""
    monthly = np.full((12, 1, 1), np.nan, dtype=np.float32)
    monthly[0, 0, 0] = 0.0
    monthly[11, 0, 0] = 11.0
    filled = _interpolate_months(monthly)
    # Linear ramp 0..11 over the 12 months.
    np.testing.assert_allclose(filled[:, 0, 0], np.arange(12, dtype=np.float32))
    assert not np.isnan(filled).any()


def test_interpolate_months_all_nan_pixel_is_zero() -> None:
    """Un pixel sin ningun mes valido queda en 0.0 (honesto, no inventado)."""
    monthly = np.full((12, 1, 1), np.nan, dtype=np.float32)
    filled = _interpolate_months(monthly)
    assert np.all(filled == 0.0)


def test_tile_indices_cover_366_into_128() -> None:
    """El tiling 366 -> 128 produce 3 offsets que cubren el patch completo."""
    offs = _tile_indices(366, 128)
    assert offs == [0, 128, 238]  # ultimo desplazado a 366-128=238
    assert offs[-1] + 128 == 366


def test_tile_indices_smaller_than_tile_returns_zero() -> None:
    """Si el lado es <= tile, solo hay un offset 0."""
    assert _tile_indices(100, 128) == [0]


# ---------------------------------------------------------------------------
# Bloque 2: patches REALES (skip si el subset no esta descargado)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _subset_present, reason="subset Sen4AgriNet no descargado")
def test_item_shape_and_dtype() -> None:
    """``x`` es ``(T,10,128,128)`` float32 y ``y`` es ``(128,128)`` int64."""
    ds = Sen4AgriNetDataset(_SUBSET_ROOT, n_timesteps=10, tile_size=_TILE)
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (10, 10, _TILE, _TILE)
    assert x.dtype == torch.float32
    assert y.shape == (_TILE, _TILE)
    assert y.dtype == torch.int64


@pytest.mark.skipif(not _subset_present, reason="subset Sen4AgriNet no descargado")
def test_band_range() -> None:
    """Las bandas normalizadas ``/10000`` caen en ``[0, ~1.5]`` (tolerante)."""
    ds = Sen4AgriNetDataset(_SUBSET_ROOT, n_timesteps=10, tile_size=_TILE)
    x, _ = ds[0]
    assert float(x.min()) >= 0.0
    assert float(x.max()) <= 1.5


@pytest.mark.skipif(not _subset_present, reason="subset Sen4AgriNet no descargado")
def test_label_encoding_in_range() -> None:
    """Labels en ``[0, num_classes) U {ignore_index}``."""
    ds = Sen4AgriNetDataset(_SUBSET_ROOT, n_timesteps=10, tile_size=_TILE)
    valid = set(range(ds.num_classes)) | {ds.ignore_index}
    # Recorre algunos sub-patches buscando labels reales no-ignore.
    seen_non_ignore = False
    for i in range(min(len(ds), 30)):
        _, y = ds[i]
        uniques = set(np.unique(y.numpy()).tolist())
        assert uniques <= valid
        if uniques - {ds.ignore_index}:
            seen_non_ignore = True
    assert seen_non_ignore, "ningun sub-patch tenia crops del nomenclator"


@pytest.mark.skipif(not _subset_present, reason="subset Sen4AgriNet no descargado")
def test_compatible_with_tsvit_forward() -> None:
    """El ``x`` del adapter pasa por ``build_tsvit`` sin shape mismatch."""
    from ml.models.tsvit_wrapper import build_tsvit

    ds = Sen4AgriNetDataset(_SUBSET_ROOT, n_timesteps=10, tile_size=_TILE)
    x, _ = ds[0]
    model = build_tsvit(
        num_classes=N_MACRO_CLASSES, n_timesteps=10, img_size=_TILE, in_channels=10
    )
    model.eval()
    with torch.no_grad():
        out = model(x.unsqueeze(0))  # (B, T, C, H, W)
    logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape[0] == 1
    assert logits.shape[-2:] == (_TILE, _TILE)
