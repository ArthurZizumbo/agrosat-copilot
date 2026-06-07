"""Explicit checkpoint registry for the segmentation re-score harness (US-030).

This module isolates the discovery of the 6 real segmentation checkpoints from
the rest of the harness (:mod:`ml.eval.dense_metrics`). It maps every model to
its exact weights path and the static metadata the loader needs (native class
count, ignore convention, input bands, resolution flag, state_dict key).

The harness NEVER globs ``best.pt`` blindly: duplicated / HCAT variants
(``deeplab-hcat6/``, ``alt-tsvit-*``, ``tsvit-v1/``, stray ``best.pt`` inside
``mlruns/``) would silently shadow the canonical weights and produce wrong
apples-to-apples numbers. Only the 6 entries below are valid (plan
``docs/us-planning/us-030.md`` 3.3, AC-8).

Conventions captured per checkpoint (verified during recon):

- ``unet``: pure ``state_dict`` (no wrapper), 20 classes, 10 bands.
- ``deeplabv3plus``: dict under key ``"model_state"``, 18 classes, 10 bands.
- ``segformer``: HuggingFace directory (``from_pretrained``), 20 labels,
  3 RGB bands, trained at 256 -> ``needs_resize=True``.
- ``utae``: dict under key ``"model_state_dict"`` (+ ``val_miou``), 20 classes,
  10 bands. Keys (``in_conv``/``out_conv``/...) must stay intact: the 20->18
  mapping happens AFTER argmax, never on the state_dict (``ml/AGENTS.md`` R1).
- ``tsvit-pheno``: dict under key ``"model_state"``, 18 classes, 10 bands.
- ``anysat``: pure ``state_dict``, 20 classes, 10 bands. The head is a
  ``LazyConv2d`` -> a dummy forward must materialize it BEFORE
  ``load_state_dict`` (handled by the loader, not here).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "CHECKPOINT_REGISTRY",
    "CheckpointSpec",
    "ModelKind",
    "resolve_state_dict",
]

#: Architecture tags understood by the harness builder/loader. ``tsvit`` and
#: ``tsvit-pheno`` share the TSViT builder but differ in their checkpoint.
ModelKind = Literal[
    "unet",
    "deeplabv3plus",
    "segformer",
    "utae",
    "tsvit",
    "tsvit-pheno",
    "anysat",
]


@dataclass(frozen=True)
class CheckpointSpec:
    """Static descriptor of a trained segmentation checkpoint for the harness.

    Attributes:
        name: Stable model identifier used as the table row key.
        model_kind: Architecture tag that selects the builder/loader path.
        path: Repo-relative path to the weights (``.pt`` file or HF directory).
        native_num_classes: Class count the head was trained with (18 or 20).
        native_ignore_index: Ignore index of the training convention (255 for the
            contiguous 18-class models, 19 for the 20-class PASTIS convention).
        in_channels: Input bands (10 for most, 3 RGB for SegFormer).
        needs_resize: ``True`` if the model was trained at 256 and predictions
            must be resampled to 128 NEAREST before accumulation.
        state_key_candidates: Ordered keys to resolve the ``state_dict`` from the
            loaded object (e.g. ``("model_state", "model_state_dict")``). If none
            match, the loaded object is assumed to be a pure ``state_dict``.
    """

    name: str
    model_kind: ModelKind
    path: Path
    native_num_classes: int
    native_ignore_index: int
    in_channels: int = 10
    needs_resize: bool = False
    state_key_candidates: tuple[str, ...] = ("model_state", "model_state_dict")


#: Repo root resolved from this file (``ml/eval/checkpoint_registry.py`` -> repo).
#: Anchoring to ``__file__`` keeps the registry robust to the caller's cwd: the
#: harness resolves the same materialized checkpoints whether invoked from the
#: repo root, from ``ml/``, or from a notebook.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Absolute root of the materialized segmentation checkpoints.
_CKPT_ROOT = _REPO_ROOT / "checkpoints" / "segmentation"


#: Explicit mapping model -> checkpoint. Never glob ``best.pt`` blindly (AC-8).
#: The 4 models trained with 20 classes (unet, segformer, utae, anysat) get
#: their predictions mapped 20->18 after argmax; the 2 native 18-class models
#: (deeplabv3plus, tsvit-pheno) skip the remap. SegFormer is the only 3-RGB /
#: 256px (needs_resize) entry.
CHECKPOINT_REGISTRY: dict[str, CheckpointSpec] = {
    "unet": CheckpointSpec(
        name="unet",
        model_kind="unet",
        path=_CKPT_ROOT / "unet-aaron" / "unet_pastis.pt",
        native_num_classes=20,
        native_ignore_index=19,
        in_channels=10,
        needs_resize=False,
        state_key_candidates=("model_state", "model_state_dict"),
    ),
    "deeplabv3plus": CheckpointSpec(
        name="deeplabv3plus",
        model_kind="deeplabv3plus",
        path=_CKPT_ROOT / "deeplab-18" / "best.pt",
        native_num_classes=18,
        native_ignore_index=255,
        in_channels=10,
        needs_resize=False,
        state_key_candidates=("model_state", "model_state_dict"),
    ),
    "segformer": CheckpointSpec(
        name="segformer",
        model_kind="segformer",
        path=_CKPT_ROOT / "segformer-isaac" / "hf_model",
        native_num_classes=20,
        native_ignore_index=19,
        in_channels=3,
        needs_resize=True,
        state_key_candidates=("model_state", "model_state_dict"),
    ),
    "utae": CheckpointSpec(
        name="utae",
        model_kind="utae",
        path=_CKPT_ROOT / "utae-isaac" / "best_model.pt",
        native_num_classes=20,
        native_ignore_index=19,
        in_channels=10,
        needs_resize=False,
        state_key_candidates=("model_state_dict", "model_state"),
    ),
    "tsvit-pheno": CheckpointSpec(
        name="tsvit-pheno",
        model_kind="tsvit-pheno",
        path=_CKPT_ROOT / "tsvit-pheno-v1" / "best.pt",
        native_num_classes=18,
        native_ignore_index=255,
        in_channels=10,
        needs_resize=False,
        state_key_candidates=("model_state", "model_state_dict"),
    ),
    "anysat": CheckpointSpec(
        name="anysat",
        model_kind="anysat",
        # NOTE: AnySat's head is a torch ``LazyConv2d``; the loader must run a
        # dummy forward (e.g. a (1, 10, 256, 256) tensor) to materialize the
        # lazy parameters BEFORE calling ``load_state_dict``. The encoder is
        # pulled from ``torch.hub gastruc/anysat`` (needs internet the first
        # time); when absent the harness yields status="missing" gracefully.
        path=_CKPT_ROOT / "anysat-aaron" / "anysat_pastis.pt",
        native_num_classes=20,
        native_ignore_index=19,
        in_channels=10,
        needs_resize=False,
        state_key_candidates=("model_state", "model_state_dict"),
    ),
}


def resolve_state_dict(loaded: object, spec: CheckpointSpec) -> dict:
    """Return the raw ``state_dict``, tolerant to the 3 checkpoint conventions.

    Three conventions coexist across the 6 checkpoints (plan R5):

    1. ``deeplabv3plus`` / ``tsvit-pheno``: dict under key ``"model_state"``.
    2. ``utae``: dict under key ``"model_state_dict"`` (alongside ``val_miou``).
    3. ``unet`` / ``anysat``: a pure ``state_dict`` (no wrapper dict).

    The resolution probes ``spec.state_key_candidates`` in order; if ``loaded``
    is a mapping and one of those keys holds a dict, that inner dict is the
    ``state_dict``. Otherwise ``loaded`` is assumed to already be the
    ``state_dict`` and returned as-is.

    Args:
        loaded: Object returned by ``torch.load`` (a checkpoint dict or a pure
            ``state_dict``).
        spec: Descriptor whose ``state_key_candidates`` drive the probing.

    Returns:
        The raw ``state_dict`` (a ``dict`` of tensor parameters).

    Raises:
        TypeError: if ``loaded`` is neither a mapping nor exposes a usable
            ``state_dict`` under the candidate keys.
    """
    if not isinstance(loaded, dict):
        raise TypeError(
            f"Unsupported checkpoint object for `{spec.name}`: expected a dict "
            f"(checkpoint wrapper or pure state_dict), got {type(loaded)!r}."
        )

    for key in spec.state_key_candidates:
        inner = loaded.get(key)
        if isinstance(inner, dict):
            logger.debug(
                "state_dict_resolved", model=spec.name, key=key, convention="wrapped"
            )
            return inner

    # No wrapper key matched -> `loaded` is already a pure state_dict.
    logger.debug("state_dict_resolved", model=spec.name, convention="pure")
    return loaded
