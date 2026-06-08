"""Tests del contrato harness <-> checkpoint para el TSViT Full-M (US-038).

Fijan la entrada ``"tsvit"`` del registry (kind, clases, ignore, bandas, path
``alt-tsvit-fullm-v1/best.pt`` y los ``model_kwargs`` de capacidad Full-M) y,
sobre todo, verifican que ``build_model_for_kind`` reconstruye el modelo con la
MISMA capacidad con que se entreno, de modo que ``load_state_dict`` carga el
``best.pt`` Full-M SIN shape mismatch (US-038 R-HARNESS). El smoke de carga usa
un checkpoint mini guardado en un ``tmp_path`` (cero GPU/red/dataset); el test
sobre el ``best.pt`` real se salta si el binario no esta en disco (DVC).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from ml.eval.checkpoint_registry import (
    CHECKPOINT_REGISTRY,
    CheckpointSpec,
    resolve_state_dict,
)
from ml.eval.segmentation_inference import build_model_for_kind
from ml.models.tsvit_wrapper import TSVIT_FULLM_CONFIG, build_tsvit


def test_registry_has_tsvit() -> None:
    """La entrada ``tsvit`` existe con los metadatos Full-M esperados (AC-9)."""
    assert "tsvit" in CHECKPOINT_REGISTRY
    spec = CHECKPOINT_REGISTRY["tsvit"]
    assert spec.model_kind == "tsvit"
    assert spec.native_num_classes == 18
    assert spec.native_ignore_index == 255
    assert spec.in_channels == 10
    assert spec.needs_resize is False
    assert spec.state_key_candidates[0] == "model_state"
    assert spec.path.parts[-2:] == ("alt-tsvit-fullm-v1", "best.pt")


def test_registry_tsvit_carries_fullm_capacity() -> None:
    """``model_kwargs`` lleva la capacidad Full-M (anti R-HARNESS).

    El ``best.pt`` que guarda ``train_segmentation`` NO incrusta la capacidad;
    si el spec no la transportara, el harness reconstruiria un TSViT L4 y
    ``load_state_dict`` fallaria por shape mismatch.
    """
    spec = CHECKPOINT_REGISTRY["tsvit"]
    assert spec.model_kwargs == TSVIT_FULLM_CONFIG
    assert spec.model_kwargs["dim"] == 192
    assert spec.model_kwargs["n_timesteps"] == 64
    assert spec.model_kwargs["depth_temporal"] == 6
    assert spec.model_kwargs["depth_spatial"] == 6


def test_tsvit_pheno_has_no_extra_kwargs() -> None:
    """El TSViT-pheno (L4) no arrastra ``model_kwargs`` (retro-compat)."""
    assert CHECKPOINT_REGISTRY["tsvit-pheno"].model_kwargs == {}


def test_build_model_for_kind_applies_fullm_kwargs() -> None:
    """``build_model_for_kind`` construye Full-M cuando el spec lo declara.

    Sin tocar ``rescore_all_checkpoints``: el builder mezcla ``spec.model_kwargs``
    sobre los defaults L4, de modo que la topologia reconstruida coincide con la
    entrenada (dim/depth/heads/n_timesteps Full-M).
    """
    spec = CHECKPOINT_REGISTRY["tsvit"]
    model = build_model_for_kind(spec, device="cpu")
    assert model.dim == 192  # type: ignore[attr-defined]
    assert model.n_timesteps == 64  # type: ignore[attr-defined]
    assert len(model.temporal_transformer.layers) == 6  # type: ignore[attr-defined]
    assert len(model.spatial_transformer.layers) == 6  # type: ignore[attr-defined]


def test_harness_builds_fullm_matching_synthetic_ckpt(tmp_path: Path) -> None:
    """Smoke R-HARNESS: un ``best.pt`` Full-M mini se recarga sin shape mismatch.

    Reproduce el contrato acoplado con una capacidad reducida (mismo perfil que
    Full-M: dim>L4, depth>L4, n_timesteps grande) para que el test sea barato:
    (1) se entrena-simula guardando ``{"model_state": ...}`` como
    ``train_segmentation``; (2) un ``CheckpointSpec`` con esos ``model_kwargs``
    apunta al binario; (3) ``build_model_for_kind`` + ``load_state_dict`` cargan
    sin claves faltantes/inesperadas y el forward produce ``(B, 18, H, W)``.
    """
    mini_kwargs = {
        "n_timesteps": 64,
        "img_size": 64,
        "patch_size": 8,
        "dim": 96,
        "depth_temporal": 3,
        "depth_spatial": 3,
        "heads": 3,
        "dim_head": 32,
        "mlp_ratio": 4,
        "semantic_dim": 384,
    }
    trained = build_tsvit(num_classes=18, in_channels=10, **mini_kwargs)
    ckpt_path = tmp_path / "best.pt"
    torch.save({"model_state": trained.state_dict()}, ckpt_path)

    spec = CheckpointSpec(
        name="tsvit",
        model_kind="tsvit",
        path=ckpt_path,
        native_num_classes=18,
        native_ignore_index=255,
        in_channels=10,
        needs_resize=False,
        state_key_candidates=("model_state", "model_state_dict"),
        model_kwargs=mini_kwargs,
    )

    rebuilt = build_model_for_kind(spec, device="cpu")
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = resolve_state_dict(loaded, spec)
    missing, unexpected = rebuilt.load_state_dict(state, strict=False)
    # Mismatch de forma -> load_state_dict(strict=False) reporta keys faltantes.
    # Con la capacidad correcta no debe haber ninguna.
    assert missing == []
    assert unexpected == []

    rebuilt.eval()
    x = torch.randn(1, 12, 10, 64, 64)
    with torch.no_grad():
        logits = rebuilt(x)
    assert logits.shape == (1, 18, 64, 64)


def test_harness_rejects_l4_rebuild_for_fullm_ckpt(tmp_path: Path) -> None:
    """Contraprueba: reconstruir en L4 un checkpoint Full-M SI da mismatch.

    Justifica la existencia de ``model_kwargs``: si el spec no transportara la
    capacidad (caso L4 por defecto), ``load_state_dict`` lanza ``RuntimeError``
    por discrepancia de formas (incluso con ``strict=False``, que solo tolera
    keys faltantes/inesperadas, no tensores de distinto tamano). Es el bug que
    ``model_kwargs`` evita.
    """
    import pytest

    mini_kwargs = {
        "n_timesteps": 64,
        "img_size": 64,
        "patch_size": 8,
        "dim": 96,
        "depth_temporal": 3,
        "depth_spatial": 3,
        "heads": 3,
        "dim_head": 32,
        "mlp_ratio": 4,
        "semantic_dim": 384,
    }
    trained = build_tsvit(num_classes=18, in_channels=10, **mini_kwargs)
    ckpt_path = tmp_path / "best.pt"
    torch.save({"model_state": trained.state_dict()}, ckpt_path)

    # Spec SIN model_kwargs -> el builder cae a L4 (dim=128, depth 4+4): topologia
    # distinta de la entrenada (dim=96, depth 3+3).
    l4_spec = CheckpointSpec(
        name="tsvit",
        model_kind="tsvit",
        path=ckpt_path,
        native_num_classes=18,
        native_ignore_index=255,
        in_channels=10,
    )
    l4_model = build_model_for_kind(l4_spec, device="cpu")
    state = resolve_state_dict(
        torch.load(ckpt_path, map_location="cpu", weights_only=False), l4_spec
    )
    # La discrepancia de capacidad (dim 96 vs 128) se manifiesta como un
    # RuntimeError de size mismatch, incluso con strict=False.
    with pytest.raises(RuntimeError, match="size mismatch"):
        l4_model.load_state_dict(state, strict=False)


def test_harness_loads_real_fullm_best_pt() -> None:
    """Carga el ``best.pt`` Full-M real si esta en disco (skip si ausente, DVC).

    El binario vive en F:/checkpoints (VM H100) o se obtiene por DVC; en CI/local
    sin el peso, el test se salta. Cuando existe, verifica end-to-end que el
    harness reconstruye y carga el ``best.pt`` real sin shape mismatch.
    """
    import pytest

    spec = CHECKPOINT_REGISTRY["tsvit"]
    if not spec.path.exists():
        pytest.skip(f"checkpoint Full-M ausente (DVC): {spec.path}")

    model = build_model_for_kind(spec, device="cpu")
    loaded = torch.load(spec.path, map_location="cpu", weights_only=False)
    state = resolve_state_dict(loaded, spec)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert missing == []
    assert unexpected == []


def test_replace_spec_keeps_default_kwargs() -> None:
    """``dataclasses.replace`` sobre un spec sin model_kwargs preserva ``{}``.

    Sanity de que el nuevo campo con ``default_factory`` no rompe el patron de
    construccion de specs en los tests existentes del harness.
    """
    base = CHECKPOINT_REGISTRY["deeplabv3plus"]
    clone = replace(base, name="deeplab-clone")
    assert clone.model_kwargs == {}
