"""Tests de la config TSViT Full-M y la retro-compatibilidad L4 (US-038).

Corren en CPU, sin red ni dataset real: validan que la factory ``build_tsvit``
expone los hiperparametros de capacidad Full-M (``heads``/``dim_head``/
``mlp_ratio``/``dropout``/``max_doy``), que el forward Full-M produce la forma
densa esperada, que Full-M tiene mas parametros que L4, que los defaults siguen
siendo los de L4 (firma intacta para los notebooks 5a/5b y el harness) y que el
subsampleo temporal no recorta cuando ``n_timesteps >= T`` (R-TLEN).

La capacidad Full-M completa solo se ejercita en el run H100 (mIoU); aqui se
verifica el algebra de formas y conteo de parametros, deterministas en CPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.data.pastis_seg_dataset import _equispaced_indices
from ml.models.tsvit_wrapper import TSVIT_FULLM_CONFIG, TSViT, build_tsvit

_NUM_CLASSES = 18
_IN_CHANNELS = 10


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Fija la semilla global de torch para reproducibilidad de los forwards."""
    torch.manual_seed(0)


def _n_params(model: torch.nn.Module) -> int:
    """Devuelve el numero total de parametros entrenables del modelo."""
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# 6.1 Factory Full-M (AC-3) + retro-compat L4 (R-FIRMA)
# ---------------------------------------------------------------------------


def test_build_tsvit_accepts_fullm_kwargs() -> None:
    """``build_tsvit`` acepta los kwargs Full-M y construye sin error."""
    model = build_tsvit(
        num_classes=_NUM_CLASSES,
        n_timesteps=64,
        img_size=128,
        in_channels=_IN_CHANNELS,
        dim=192,
        depth_temporal=6,
        depth_spatial=6,
        heads=6,
        dim_head=64,
        mlp_ratio=4,
        dropout=0.1,
        max_doy=366,
    )
    assert isinstance(model, TSViT)
    assert model.dim == 192
    assert model.n_timesteps == 64
    # La PE ordinal se dimensiona con n_timesteps (R-TLEN): [1, 64, dim].
    assert model.temporal_pos_ordinal.shape == (1, 64, 192)


def test_build_tsvit_default_is_l4() -> None:
    """``build_tsvit()`` sin kwargs nuevos reproduce la config L4 (firma intacta).

    Los notebooks 5a/5b y el harness invocan la factory con la firma posicional
    historica; los kwargs Full-M son keyword-only con default == L4, de modo que
    no cambian el modelo por defecto.
    """
    model = build_tsvit()
    assert isinstance(model, TSViT)
    assert model.dim == 128
    assert model.n_timesteps == 10
    assert model.num_classes == 18
    # heads=4, dim_head=32 -> inner_dim = 128 en el modulo de atencion temporal.
    attn = model.temporal_transformer.layers[0].attn
    assert attn.heads == 4
    # depth 4+4.
    assert len(model.temporal_transformer.layers) == 4
    assert len(model.spatial_transformer.layers) == 4


def test_fullm_has_more_params() -> None:
    """Full-M tiene mas parametros que L4 (sanity de capacidad)."""
    l4 = build_tsvit(num_classes=_NUM_CLASSES, n_timesteps=10, in_channels=_IN_CHANNELS)
    fullm = build_tsvit(
        num_classes=_NUM_CLASSES,
        in_channels=_IN_CHANNELS,
        **TSVIT_FULLM_CONFIG,
    )
    assert _n_params(fullm) > _n_params(l4)


def test_fullm_config_matches_factory_signature() -> None:
    """La config Full-M se acepta tal cual por la factory (contrato compartido).

    ``TSVIT_FULLM_CONFIG`` es la unica fuente de verdad: la misma usada por el
    orquestador de entrenamiento y por la entrada ``tsvit`` del registry. Si la
    factory no aceptara alguna clave, el run y el re-score divergirian.
    """
    model = build_tsvit(
        num_classes=_NUM_CLASSES,
        in_channels=_IN_CHANNELS,
        **TSVIT_FULLM_CONFIG,
    )
    assert model.dim == TSVIT_FULLM_CONFIG["dim"]
    assert model.n_timesteps == TSVIT_FULLM_CONFIG["n_timesteps"]
    assert len(model.temporal_transformer.layers) == TSVIT_FULLM_CONFIG["depth_temporal"]
    assert len(model.spatial_transformer.layers) == TSVIT_FULLM_CONFIG["depth_spatial"]


# ---------------------------------------------------------------------------
# 6.2 Forward shape (AC-2)
# ---------------------------------------------------------------------------


def test_fullm_forward_shape() -> None:
    """Forward Full-M ``(2, T, 10, 128, 128) -> (2, 18, 128, 128)`` sin NaN.

    ``T=12`` (arbitrario, < n_timesteps=64): el modelo es invariante al numero
    de fechas por el flatten temporal; la PE ordinal indexa ``[:, :T, :]``.
    """
    model = build_tsvit(
        num_classes=_NUM_CLASSES,
        in_channels=_IN_CHANNELS,
        **TSVIT_FULLM_CONFIG,
    ).eval()
    x = torch.randn(2, 12, _IN_CHANNELS, 128, 128)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, _NUM_CLASSES, 128, 128)
    assert torch.isfinite(logits).all()


def test_fullm_forward_visual_proj() -> None:
    """``return_visual_proj=True`` devuelve ``(logits, (2, 384, 128, 128))``.

    La rama visual existe en Full-M (la consume US-039 pheno); aqui solo se
    verifica que la forma semantica es correcta.
    """
    model = build_tsvit(
        num_classes=_NUM_CLASSES,
        in_channels=_IN_CHANNELS,
        **TSVIT_FULLM_CONFIG,
    ).eval()
    x = torch.randn(2, 8, _IN_CHANNELS, 128, 128)
    with torch.no_grad():
        out = model(x, return_visual_proj=True)
    assert isinstance(out, tuple)
    logits, visual = out
    assert logits.shape == (2, _NUM_CLASSES, 128, 128)
    assert visual.shape == (2, 384, 128, 128)


def test_fullm_forward_t_equals_n_timesteps() -> None:
    """Forward con ``T == n_timesteps`` (borde de la PE ordinal, R-TLEN).

    El caso extremo es ``T`` igual al ``n_timesteps`` con que se construyo el
    modelo: la PE ordinal ``[:, :T, :]`` no debe quedar corta ni indexar fuera
    de rango.
    """
    n = TSVIT_FULLM_CONFIG["n_timesteps"]
    model = build_tsvit(
        num_classes=_NUM_CLASSES,
        in_channels=_IN_CHANNELS,
        **TSVIT_FULLM_CONFIG,
    ).eval()
    x = torch.randn(1, n, _IN_CHANNELS, 128, 128)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, _NUM_CLASSES, 128, 128)
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# 6.3 No-recorte temporal (AC-1, R-TLEN)
# ---------------------------------------------------------------------------


def test_equispaced_no_trim_when_n_ge_t() -> None:
    """``_equispaced_indices(61, 64)`` devuelve TODA la serie (T completo)."""
    idx = _equispaced_indices(61, 64)
    np.testing.assert_array_equal(idx, np.arange(61))


def test_equispaced_no_trim_at_t_max() -> None:
    """En el limite ``T == n_timesteps`` tampoco recorta."""
    idx = _equispaced_indices(64, 64)
    np.testing.assert_array_equal(idx, np.arange(64))


def test_equispaced_trims_when_n_lt_t() -> None:
    """``_equispaced_indices(61, 10)`` subsamplea a 10 (sanity del modo L4)."""
    idx = _equispaced_indices(61, 10)
    assert len(idx) == 10
    assert idx[0] == 0
    assert idx[-1] == 60
    # Indices unicos y ascendentes.
    assert np.all(np.diff(idx) > 0)
