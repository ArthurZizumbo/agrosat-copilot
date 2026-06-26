"""Real dense transfer: fine-tune TSViT-pheno + U-TAE to the Italian label space.

US-079 modelling. The dense champion members were trained on PASTIS-R France
(18 semantic classes). Here they are TRANSFERRED to the US-078 Mediterranean
homologue (Italy 2018) with the corrected protocol Arthur asked for: a REAL
fine-tune (backbone unfrozen), NOT a frozen feature extractor, with a head sized
to the Italian dense label space whose CONSERVED rows are warm-started from the
PASTIS head (the kept-class flag of :mod:`ml.transfer.italia_label_space`).

What is reused (no reinvention)
-------------------------------
- :func:`ml.transfer.italia_label_space.warm_start_head` + ``ItaliaLabelSpace`` --
  the kept-class flag, generalised from the Baltic version.
- :func:`ml.models.tsvit_wrapper.build_tsvit` / :func:`ml.models.utae.build_utae`
  -- the exact architectures of the PASTIS checkpoints (L4 capacity for
  ``tsvit-pheno-v1``, Isaac's 20-class U-TAE).
- :func:`ml.eval.checkpoint_registry.resolve_state_dict` -- the 3 checkpoint
  conventions (``model_state`` / ``model_state_dict`` / pure).
- :mod:`ml.eval.dense_metrics` -- the pixel-level mIoU / F1-macro accumulator
  (apples-to-apples with the segmentation harness).

Dense, not per-parcel
---------------------
Unlike the Baltic fine-tune (per-parcel, GAP-pooled logits), the Italian
homologue carries PASTIS-style DENSE masks ``TARGET_<id>.npy (128, 128)``, so the
loss is the standard pixel cross-entropy and the prediction is a dense map
``(K, 128, 128)`` -- exactly what the Voting-3 combiner (US-079 step 3) consumes
post-softmax.

Anti-leakage
------------
The train/test split is SPATIAL: it uses the ``fold_espacial`` assigned by the
US-078 builder (adjacent patches share a fold), never a random split. The held-out
fold's patches never contribute a training pixel.

H100 reality (Arthur: train for real)
--------------------------------------
The backbone unfreeze is a real GPU train. Per-epoch checkpoints are written to a
RELATIVE path ``checkpoints/transfer/<model>-italia/<run>/epoch_<NN>.pt`` (+
``best.pt`` + ``last.pt``) so on the VM they land on ``F:`` (the worktree drive),
never ``C:``. Subset / epochs are config parameters so a CPU smoke is cheap and
the full H100 run is one flag away.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from torch import nn

from ml.transfer.italia_label_space import (
    ItaliaLabelSpace,
    build_italia_label_space,
    stratified_pixel_patch_sample,
    warm_start_head,
)

logger = structlog.get_logger(__name__)

__all__ = [
    "DenseFineTuneConfig",
    "ItaliaDensePatches",
    "build_italia_finetune_model",
    "load_italia_patches",
    "run_italia_finetune",
    "zero_shot_pastis_predict",
]

#: Repo root (this file is ``<root>/ml/transfer/finetune_italia.py``).
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Default US-078 homologue dataset root.
DEFAULT_ITALIA_ROOT: Path = _REPO_ROOT / "data" / "pastis_italia_2018"

#: Default PASTIS-R checkpoints for the backbone init (real, on disk / DVC).
DEFAULT_PASTIS_CHECKPOINTS: dict[str, Path] = {
    "tsvit-pheno": _REPO_ROOT
    / "checkpoints"
    / "segmentation"
    / "tsvit-pheno-v1"
    / "best.pt",
    "utae": _REPO_ROOT
    / "checkpoints"
    / "segmentation"
    / "utae-isaac"
    / "best_model.pt",
}

#: Default root for the per-epoch transfer checkpoints (RELATIVE; lands on F: on
#: the VM worktree, never C:).
DEFAULT_CKPT_ROOT: Path = _REPO_ROOT / "checkpoints" / "transfer"

#: PASTIS-R reflectance scale (int16 DN -> reflectance); the US-078 builder writes
#: the same ``x10000`` DN convention, so the dense models read the trained scale.
_S2_SCALE: float = 10000.0
#: Patch side in pixels (= PASTIS).
_PATCH_PX: int = 128


@dataclass
class DenseFineTuneConfig:
    """Hyperparameters for the Italian dense fine-tune.

    Attributes:
        model_kind: ``"tsvit-pheno"`` or ``"utae"``.
        n_timesteps: Equispaced dates subsampled per patch (TSViT-pheno-v1 was
            trained with 10; U-TAE is length-agnostic but a fixed length makes the
            batch stackable).
        head_warmup_epochs: Head-only epochs before unfreezing the backbone.
        finetune_epochs: Backbone+head epochs.
        lr_head: Learning rate for the head (and cls-tokens for TSViT).
        lr_backbone: Smaller learning rate for the pretrained backbone.
        weight_decay: AdamW weight decay.
        batch_size: Patches per step.
        min_patches_per_class: Stratified patch coverage per crop class (``None``
            = use every train patch).
        ignore_background: When ``True`` the background id 0 is excluded from the
            loss and the metrics (the model still has the row, but it is not
            supervised on it).
        seed: RNG seed.
    """

    model_kind: str = "tsvit-pheno"
    n_timesteps: int = 10
    head_warmup_epochs: int = 2
    finetune_epochs: int = 12
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    min_patches_per_class: int | None = None
    ignore_background: bool = True
    seed: int = 42


@dataclass
class ItaliaDensePatches:
    """A loaded split of Italian dense patches.

    Attributes:
        patch_ids: The patch ids in this split (ordered).
        images: Per-patch float32 stacks ``(T_sub, 10, 128, 128)`` (normalised).
        masks: Per-patch int64 dense masks ``(128, 128)`` (0 = background,
            ``[1, K]`` = crop class id).
        doys: Per-patch day-of-year arrays ``(T_sub,)`` int64.
        folds: Per-patch spatial fold id (from US-078 ``fold_espacial``).
    """

    patch_ids: list[int]
    images: list[np.ndarray]
    masks: list[np.ndarray]
    doys: list[np.ndarray]
    folds: list[int]

    def __len__(self) -> int:
        return len(self.patch_ids)

    def present_classes(self) -> list[set[int]]:
        """Per-patch set of present crop class ids (background excluded)."""
        out: list[set[int]] = []
        for mask in self.masks:
            present = {int(c) for c in np.unique(mask) if int(c) != 0}
            out.append(present)
        return out


def _equispaced_indices(n_available: int, n_select: int) -> np.ndarray:
    """Deterministic equispaced indices (same convention as the PASTIS loader).

    Args:
        n_available: Dates available in the patch (T).
        n_select: Dates to keep.

    Returns:
        Sorted unique int indices in ``[0, n_available)``.
    """
    if n_select >= n_available:
        return np.arange(n_available)
    idx = np.linspace(0, n_available - 1, num=n_select)
    return np.unique(np.round(idx).astype(int))


def load_italia_patches(
    *,
    italia_root: Path = DEFAULT_ITALIA_ROOT,
    n_timesteps: int = 10,
    folds: tuple[int, ...] | None = None,
) -> ItaliaDensePatches:
    """Load the US-078 homologue patches as PASTIS-style dense tensors.

    Reads ``DATA_S2/S2_<id>.npy (T, 10, 128, 128)`` (int16 DN, scaled ``/10000``),
    ``ANNOTATIONS/TARGET_<id>.npy (128, 128)`` (int32 class mask) and
    ``ANNOTATIONS/dates_<id>.npy (T,)`` (DOY), subsampling ``n_timesteps``
    equispaced dates per patch (the convention TSViT-pheno-v1 / U-TAE were trained
    with). The spatial fold of each patch comes from the US-078
    ``metadata.parquet`` (``fold_espacial``).

    Args:
        italia_root: The homologue dataset root.
        n_timesteps: Equispaced dates to keep per patch.
        folds: When given, keep only patches whose ``fold_espacial`` is in this
            set; ``None`` keeps every patch on disk.

    Returns:
        An :class:`ItaliaDensePatches` split (ordered by patch id).

    Raises:
        FileNotFoundError: if the dataset root or its ``metadata.parquet`` are
            absent (the US-078 builder must run first).
    """
    import polars as pl

    s2_dir = italia_root / "DATA_S2"
    ann_dir = italia_root / "ANNOTATIONS"
    meta_path = italia_root / "metadata.parquet"
    if not s2_dir.is_dir() or not meta_path.is_file():
        raise FileNotFoundError(
            f"homologue dataset incomplete under {italia_root} (need DATA_S2/ + "
            "metadata.parquet); run scripts/build_italia_pastis.py first."
        )
    meta = pl.read_parquet(meta_path)
    fold_of = dict(
        zip(
            (int(p) for p in meta["patch_id"].to_list()),
            (int(f) for f in meta["fold_espacial"].to_list()),
            strict=True,
        )
    )
    wanted = set(folds) if folds is not None else None

    on_disk = sorted(
        int(p.stem.split("_", 1)[1]) for p in s2_dir.glob("S2_*.npy")
    )
    patch_ids: list[int] = []
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    doys: list[np.ndarray] = []
    patch_folds: list[int] = []
    for pid in on_disk:
        fold = fold_of.get(pid, 0)
        if wanted is not None and fold not in wanted:
            continue
        s2 = np.load(s2_dir / f"S2_{pid}.npy").astype(np.float32) / _S2_SCALE
        mask = np.load(ann_dir / f"TARGET_{pid}.npy").astype(np.int64)
        date_path = ann_dir / f"dates_{pid}.npy"
        doy = (
            np.load(date_path).astype(np.int64)
            if date_path.is_file()
            else np.arange(s2.shape[0], dtype=np.int64)
        )
        idx = _equispaced_indices(s2.shape[0], n_timesteps)
        images.append(np.ascontiguousarray(s2[idx]))
        doys.append(np.ascontiguousarray(doy[idx]))
        masks.append(np.ascontiguousarray(mask))
        patch_ids.append(pid)
        patch_folds.append(fold)
    logger.info(
        "italia_patches_loaded",
        n_patches=len(patch_ids),
        folds=sorted(set(patch_folds)),
        n_timesteps=n_timesteps,
    )
    return ItaliaDensePatches(
        patch_ids=patch_ids,
        images=images,
        masks=masks,
        doys=doys,
        folds=patch_folds,
    )


def build_italia_finetune_model(
    label_space: ItaliaLabelSpace,
    *,
    model_kind: str,
    pastis_checkpoint: Path,
    n_timesteps: int = 10,
    device: str = "cuda",
) -> nn.Module:
    """Build the dense model with an Italian head, backbone init from PASTIS.

    Loads the PASTIS checkpoint into the backbone (shared layers, matched by name
    and shape), attaches a fresh classification head sized for the Italian dense
    label space (``label_space.num_classes`` = background + ``K`` crops), and
    warm-starts the conserved rows of that head from the PASTIS head (the
    kept-class flag). New-class and background rows stay at their random init.

    Per-architecture head:
    - ``utae``: the head is the ``out_conv.2`` 1x1 conv ``(K, 32, 1, 1)``; its rows
      are warm-started from the PASTIS U-TAE ``out_conv.2`` (20 classes, contiguous
      18 after the background/void shift handled by ``SEMANTIC18_CLASS_NAMES``).
    - ``tsvit-pheno``: the per-class "head" is the bank of separable temporal
      cls-tokens ``temporal_cls_tokens (1, K, dim)`` -- one token per class. Those
      rows are warm-started (the shared ``to_seg`` projection is class-agnostic and
      copied verbatim from PASTIS).

    Args:
        label_space: The Italian target label space.
        model_kind: ``"utae"`` or ``"tsvit-pheno"``.
        pastis_checkpoint: Path to the PASTIS checkpoint (.pt).
        n_timesteps: Temporal length for the TSViT ordinal PE (10 for the L4
            ``tsvit-pheno-v1``).
        device: Torch device.

    Returns:
        The model ready to fine-tune (on ``device``).

    Raises:
        ValueError: for an unsupported ``model_kind``.
        FileNotFoundError: if the PASTIS checkpoint is absent.
    """
    import torch

    from ml.data.pastis_filter import SEMANTIC18_CLASS_NAMES
    from ml.eval.checkpoint_registry import CHECKPOINT_REGISTRY, resolve_state_dict

    if not Path(pastis_checkpoint).is_file():
        raise FileNotFoundError(
            f"PASTIS checkpoint not found at {pastis_checkpoint}; pull it with "
            "`dvc pull checkpoints/segmentation/...` or check the path."
        )

    from torch import nn as _nn

    k_new = label_space.num_classes
    pastis_names = dict(SEMANTIC18_CLASS_NAMES)
    model: _nn.Module
    if model_kind == "utae":
        from ml.models.utae import build_utae

        # Isaac's U-TAE head is (20, 32, 1, 1): 18 crops + background + void. The
        # warm-start maps the contiguous semantic-18 ids onto the U-TAE rows 1..18
        # (id 0 = background, 19 = void), so the PASTIS name -> head id used here is
        # the U-TAE native id, i.e. semantic id + 1.
        model = build_utae(num_classes=k_new, input_dim=10)
        head_kind = "conv_head"
        spec_key = "utae"
        # U-TAE head id namespace = semantic id + 1 (id 0 = background, 19 = void).
        pastis_head_ids = {cid + 1: name for cid, name in pastis_names.items()}
    elif model_kind == "tsvit-pheno":
        from ml.models.tsvit_wrapper import build_tsvit

        model = build_tsvit(
            num_classes=k_new,
            n_timesteps=n_timesteps,
            img_size=_PATCH_PX,
            in_channels=10,
            patch_size=8,
            dim=128,
            depth_temporal=4,
            depth_spatial=4,
            semantic_dim=384,
        )
        head_kind = "cls_tokens"
        spec_key = "tsvit-pheno"
        # TSViT cls-token id namespace = the contiguous semantic-18 ids (0..17).
        pastis_head_ids = dict(pastis_names)
    else:
        raise ValueError(f"unsupported model_kind {model_kind!r}")

    spec = CHECKPOINT_REGISTRY[spec_key]
    loaded = torch.load(pastis_checkpoint, map_location="cpu", weights_only=False)
    pastis_state = resolve_state_dict(loaded, spec)
    own = model.state_dict()
    compatible = {
        k: v
        for k, v in pastis_state.items()
        if k in own and own[k].shape == v.shape
    }
    own.update(compatible)
    model.load_state_dict(own, strict=False)
    logger.info(
        "italia_finetune_backbone_init",
        model_kind=model_kind,
        n_loaded=len(compatible),
        n_total=len(own),
        n_classes_new=k_new,
    )

    _warm_start_dense_head(
        model,
        pastis_state,
        label_space=label_space,
        head_kind=head_kind,
        pastis_class_names=pastis_head_ids,
    )

    model.to(device)
    return model


def _warm_start_dense_head(
    model: nn.Module,
    pastis_state: dict,
    *,
    label_space: ItaliaLabelSpace,
    head_kind: str,
    pastis_class_names: dict[int, str],
) -> list[str]:
    """Warm-start the conserved rows of the dense head from the PASTIS head.

    Dispatches on the architecture's head representation:
    - ``conv_head`` (U-TAE ``out_conv.2``): a 1x1 conv ``(K, C, 1, 1)`` flattened
      to ``(K, C)`` for the row copy.
    - ``cls_tokens`` (TSViT ``temporal_cls_tokens``): a ``(1, K, dim)`` bank of
      per-class tokens flattened to ``(K, dim)``.

    Args:
        model: The freshly built Italian model (modified in place).
        pastis_state: The resolved PASTIS ``state_dict``.
        label_space: The Italian label space (conserved rows).
        head_kind: ``"conv_head"`` or ``"cls_tokens"``.
        pastis_class_names: PASTIS head id -> class name (in the SAME id namespace
            as the PASTIS head tensor: U-TAE native 1..18, TSViT semantic 0..17).

    Returns:
        The list of leaves actually warm-started.
    """
    import torch

    if head_kind == "conv_head":
        w_name, b_name = "out_conv.2.weight", "out_conv.2.bias"
    elif head_kind == "cls_tokens":
        w_name, b_name = "temporal_cls_tokens", None
    else:
        raise ValueError(f"unsupported head_kind {head_kind!r}")

    if w_name not in pastis_state:
        logger.warning("warm_start_head_absent", w_name=w_name)
        return []

    own = model.state_dict()
    new_w = own[w_name].detach().cpu().numpy()
    pw = pastis_state[w_name].detach().cpu().numpy()
    # Flatten both heads to a row-per-class matrix ``(K, D)`` for the copy.
    # conv_head: ``(K, C, 1, 1)`` -> the class axis is axis 0.
    # cls_tokens: ``(1, K, dim)`` -> the class axis is axis 1, so squeeze axis 0.
    if head_kind == "conv_head":
        new_w2 = new_w.reshape(new_w.shape[0], -1)
        pw2 = pw.reshape(pw.shape[0], -1)
    else:
        new_w2 = new_w[0]  # (K, dim)
        pw2 = pw[0]  # (18, dim)
    if b_name is not None and b_name in pastis_state:
        new_b = own[b_name].detach().cpu().numpy().copy()
        pb = pastis_state[b_name].detach().cpu().numpy()
    else:
        new_b, pb = None, None

    new_w2, new_b, warmed = warm_start_head(
        new_w2,
        new_b,
        pw2,
        pb,
        label_space=label_space,
        pastis_class_names=pastis_class_names,
    )
    with torch.no_grad():
        if head_kind == "conv_head":
            own[w_name].copy_(torch.from_numpy(new_w2.reshape(new_w.shape)))
        else:
            own[w_name].copy_(torch.from_numpy(new_w2[None]))  # (1, K, dim)
        if b_name is not None and new_b is not None:
            own[b_name].copy_(torch.from_numpy(new_b))
    logger.info("italia_finetune_head_warmstarted", head_kind=head_kind, n_warmed=len(warmed))
    return warmed


def _is_head_param(name: str) -> bool:
    """True for the head / cls-token parameters (warmed in phase 1)."""
    return "out_conv" in name or "to_seg" in name or "temporal_cls_tokens" in name


def _doy_positions(
    doy: np.ndarray, batch: int, *, device: torch.device | str
) -> torch.Tensor:
    """Build a ``(B, T)`` day-of-year tensor for U-TAE positional encoding."""
    import torch

    pos = torch.from_numpy(np.asarray(doy, dtype=np.int64)).to(device)
    return pos.unsqueeze(0).repeat(batch, 1)


def run_italia_finetune(
    config: DenseFineTuneConfig,
    *,
    italia_root: Path = DEFAULT_ITALIA_ROOT,
    pastis_checkpoint: Path | None = None,
    test_fold: int = 3,
    ckpt_root: Path = DEFAULT_CKPT_ROOT,
    run_name: str = "us079",
    device: str = "cuda",
) -> dict[str, object]:
    """Fine-tune one dense member on the Italian homologue and dump test preds.

    Pipeline (every step leakage-guarded):

    1. Build the Italian dense label space (conserved + new) from the US-078
       class table.
    2. Load the patches and SPATIALLY split them by ``fold_espacial`` -- the
       ``test_fold`` patches are held out, the rest train (no random split).
    3. Build the model (PASTIS backbone init + warm head) and train head-only for
       ``head_warmup_epochs``, then unfreeze the backbone for ``finetune_epochs``
       (smaller backbone LR). Per-epoch checkpoints + ``best.pt`` (by train mIoU)
       + ``last.pt`` are written under
       ``ckpt_root/<model>-italia/<run>/epoch_<NN>.pt``.
    4. Predict the dense POST-SOFTMAX probability map ``(K, 128, 128)`` per test
       patch (what the Voting-3 combiner consumes) and the hard mIoU / F1-macro.

    Args:
        config: Dense fine-tune hyperparameters.
        italia_root: The US-078 homologue dataset root.
        pastis_checkpoint: PASTIS checkpoint for the backbone init; ``None`` uses
            the default for ``config.model_kind``.
        test_fold: The spatial fold held out for test.
        ckpt_root: Root for the per-epoch transfer checkpoints (relative -> F: on
            the VM).
        run_name: Run tag for the checkpoint subdirectory.
        device: Torch device.

    Returns:
        A summary dict with the fine/coarse-ready dense predictions, the test
        patch ids, the warm-started classes, the per-epoch checkpoint paths and
        the held-out mIoU / F1-macro. The post-softmax probabilities are saved as
        an ``.npz`` artifact (referenced by path) so they are not held in the dict.
    """
    import torch
    from torch import nn

    from ml.eval.dense_metrics import DenseConfusionAccumulator

    pastis_ckpt = (
        Path(pastis_checkpoint)
        if pastis_checkpoint is not None
        else DEFAULT_PASTIS_CHECKPOINTS[config.model_kind]
    )
    label_space = build_italia_label_space(italia_root=italia_root)

    all_patches = load_italia_patches(
        italia_root=italia_root, n_timesteps=config.n_timesteps
    )
    train_idx = [i for i, f in enumerate(all_patches.folds) if f != test_fold]
    test_idx = [i for i, f in enumerate(all_patches.folds) if f == test_fold]
    if not train_idx or not test_idx:
        raise ValueError(
            f"spatial split by fold_espacial={test_fold} produced an empty side "
            f"(train={len(train_idx)}, test={len(test_idx)}); pick another fold."
        )

    if config.min_patches_per_class is not None:
        present = all_patches.present_classes()
        keep_local = stratified_pixel_patch_sample(
            [present[i] for i in train_idx],
            class_ids=label_space.class_ids,
            min_patches_per_class=config.min_patches_per_class,
            seed=config.seed,
        )
        train_idx = [train_idx[i] for i in keep_local]

    logger.info(
        "italia_finetune_split",
        model_kind=config.model_kind,
        n_train=len(train_idx),
        n_test=len(test_idx),
        test_fold=test_fold,
        n_classes=label_space.num_classes,
    )

    model = build_italia_finetune_model(
        label_space,
        model_kind=config.model_kind,
        pastis_checkpoint=pastis_ckpt,
        n_timesteps=config.n_timesteps,
        device=device,
    )
    ignore_index = label_space.background_id if config.ignore_background else -100
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    run_dir = ckpt_root / f"{config.model_kind}-italia" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    def _forward(xb: torch.Tensor, doy_list: list[np.ndarray]) -> torch.Tensor:
        if config.model_kind == "utae":
            # One DOY vector per sample; the batch shares the equispaced length.
            positions = _doy_positions(doy_list[0], xb.shape[0], device=device)
            out = model(xb, positions)
        else:
            out = model(xb)
        logits: torch.Tensor = out[0] if isinstance(out, tuple) else out
        return logits

    def _epoch(
        idxs: list[int], *, train: bool, opt: torch.optim.Optimizer
    ) -> dict[str, float]:
        model.train(train)
        order = (
            np.random.default_rng(config.seed).permutation(len(idxs))
            if train
            else np.arange(len(idxs))
        )
        acc = DenseConfusionAccumulator(
            label_space.num_classes, ignore_index=ignore_index
        )
        running = 0.0
        for start in range(0, len(order), config.batch_size):
            batch_local = order[start : start + config.batch_size]
            batch = [idxs[i] for i in batch_local]
            xb = torch.from_numpy(
                np.stack([all_patches.images[i] for i in batch])
            ).float().to(device)
            yb = torch.from_numpy(
                np.stack([all_patches.masks[i] for i in batch])
            ).long().to(device)
            doy_list = [all_patches.doys[i] for i in batch]
            logits = _forward(xb, doy_list)  # (B, K, H, W)
            loss = criterion(logits, yb)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            running += float(loss.item()) * len(batch)
            acc.update(logits.argmax(dim=1).detach().cpu(), yb.detach().cpu())
        metrics = acc.compute()
        metrics["loss"] = running / max(len(idxs), 1)
        return metrics

    # Phase 1: head-only warmup (backbone frozen).
    for name, p in model.named_parameters():
        p.requires_grad = _is_head_param(name)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config.lr_head
    )
    for ep in range(config.head_warmup_epochs):
        m = _epoch(train_idx, train=True, opt=opt)
        logger.info(
            "italia_finetune_warmup_epoch",
            epoch=ep,
            loss=round(m["loss"], 4),
            train_miou=round(m["miou"], 4),
        )

    # Phase 2: unfreeze the backbone (smaller LR for it).
    for p in model.parameters():
        p.requires_grad = True
    head_params = [p for n, p in model.named_parameters() if _is_head_param(n)]
    backbone_params = [p for n, p in model.named_parameters() if not _is_head_param(n)]
    opt = torch.optim.AdamW(
        [
            {"params": head_params, "lr": config.lr_head},
            {"params": backbone_params, "lr": config.lr_backbone},
        ],
        weight_decay=config.weight_decay,
    )

    best_miou = -1.0
    epoch_ckpts: list[str] = []
    for ep in range(config.finetune_epochs):
        m = _epoch(train_idx, train=True, opt=opt)
        logger.info(
            "italia_finetune_epoch",
            epoch=ep,
            loss=round(m["loss"], 4),
            train_miou=round(m["miou"], 4),
            train_f1=round(m["f1_macro"], 4),
        )
        ckpt_path = run_dir / f"epoch_{ep:02d}.pt"
        payload = {
            "epoch": ep,
            "model_state": model.state_dict(),
            "config": _config_to_dict(config),
            "label_space_leaves": list(label_space.leaves),
            "train_metrics": m,
        }
        torch.save(payload, ckpt_path)
        torch.save(payload, run_dir / "last.pt")
        epoch_ckpts.append(str(ckpt_path))
        if m["miou"] > best_miou:
            best_miou = m["miou"]
            torch.save(payload, run_dir / "best.pt")

    # Test: dense post-softmax probabilities + hard metrics on the held-out fold.
    model.eval()
    probs_by_patch: dict[int, np.ndarray] = {}
    test_acc = DenseConfusionAccumulator(
        label_space.num_classes, ignore_index=ignore_index
    )
    with torch.no_grad():
        for i in test_idx:
            xb = torch.from_numpy(all_patches.images[i][None]).float().to(device)
            logits = _forward(xb, [all_patches.doys[i]])
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (K,H,W)
            probs_by_patch[all_patches.patch_ids[i]] = probs.astype(np.float32)
            preds = probs.argmax(axis=0)
            test_acc.update(preds, all_patches.masks[i])

    test_metrics = test_acc.compute()
    probs_path = run_dir / "test_softmax.npz"
    # numpy's savez stub types **kwds as bool; the real API takes named arrays.
    named_arrays = {str(pid): arr for pid, arr in probs_by_patch.items()}
    np.savez_compressed(probs_path, **named_arrays)  # type: ignore[arg-type]

    summary = {
        "model_kind": config.model_kind,
        "test_fold": test_fold,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "num_classes": label_space.num_classes,
        "n_conserved": len(label_space.conserved),
        "n_new": len(label_space.new),
        "test_miou": round(float(test_metrics["miou"]), 4),
        "test_f1_macro": round(float(test_metrics["f1_macro"]), 4),
        "test_pixel_accuracy": round(float(test_metrics["pixel_accuracy"]), 4),
        "best_train_miou": round(float(best_miou), 4),
        "test_patch_ids": [all_patches.patch_ids[i] for i in test_idx],
        "softmax_path": str(probs_path),
        "best_ckpt": str(run_dir / "best.pt"),
        "last_ckpt": str(run_dir / "last.pt"),
        "epoch_ckpts": epoch_ckpts,
        "conserved_leaves": list(label_space.conserved),
        "new_leaves": list(label_space.new),
    }
    logger.info(
        "italia_finetune_done",
        **{k: v for k, v in summary.items() if not isinstance(v, list)},
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def zero_shot_pastis_predict(
    *,
    model_kind: str,
    italia_root: Path = DEFAULT_ITALIA_ROOT,
    pastis_checkpoint: Path | None = None,
    test_fold: int = 3,
    n_timesteps: int = 10,
    device: str = "cuda",
) -> dict[int, np.ndarray]:
    """Predict Italian masks with the French champion WITHOUT fine-tuning.

    The cota inferior of the transfer delta (US-079 step 4): the PASTIS checkpoint
    is loaded verbatim (no fine-tune), run over the held-out Italian patches, and
    its native PASTIS predictions are mapped onto the ITALIAN dense id space via
    the conserved crosswalk -- only the classes France actually knows
    (``ItaliaLabelSpace.conserved`` -> their PASTIS name -> the PASTIS head id) get
    their Italian id; every other pixel falls to background (the French champion
    cannot name the new Mediterranean classes). This measures, honestly, how much
    the un-adapted backbone already transfers.

    Args:
        model_kind: ``"tsvit-pheno"`` or ``"utae"``.
        italia_root: The US-078 homologue dataset root.
        pastis_checkpoint: PASTIS checkpoint; ``None`` uses the default for
            ``model_kind``.
        test_fold: The spatial fold held out (same as the fine-tune).
        n_timesteps: Equispaced dates per patch.
        device: Torch device.

    Returns:
        ``{patch_id: (H, W)}`` predicted Italian fine class maps (zero-shot).
    """
    import torch

    from ml.data.pastis_filter import SEMANTIC18_CLASS_NAMES
    from ml.eval.checkpoint_registry import CHECKPOINT_REGISTRY
    from ml.eval.segmentation_inference import load_checkpoint_model, softmax_patch_for_kind

    label_space = build_italia_label_space(italia_root=italia_root)
    spec = CHECKPOINT_REGISTRY[model_kind]
    pastis_ckpt = (
        Path(pastis_checkpoint)
        if pastis_checkpoint is not None
        else DEFAULT_PASTIS_CHECKPOINTS[model_kind]
    )
    # Use the real PASTIS path via the registry (honours its native class count).
    from dataclasses import replace

    spec = replace(spec, path=pastis_ckpt)
    model = load_checkpoint_model(spec, n_timesteps=n_timesteps, device=device)

    # PASTIS native head id -> contiguous semantic-18 id (U-TAE native = id-1).
    pastis_names = dict(SEMANTIC18_CLASS_NAMES)
    name_to_semantic = {name: cid for cid, name in pastis_names.items()}
    # Build a LUT from the model's NATIVE class id to the Italian dense id.
    native_to_italia = np.zeros(spec.native_num_classes, dtype=np.int64)
    for leaf in label_space.conserved:
        pastis_name = label_space.leaf_to_pastis[leaf]
        sem_id = name_to_semantic.get(pastis_name)
        if sem_id is None:
            continue
        native_id = sem_id + 1 if model_kind == "utae" else sem_id
        if 0 <= native_id < spec.native_num_classes:
            native_to_italia[native_id] = label_space.index[leaf]

    patches = load_italia_patches(
        italia_root=italia_root, n_timesteps=n_timesteps, folds=(test_fold,)
    )
    preds_by_patch: dict[int, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for i, pid in enumerate(patches.patch_ids):
            x = torch.from_numpy(patches.images[i]).float()
            probs = softmax_patch_for_kind(model, x, model_kind=model_kind)
            native_pred = probs.argmax(axis=0)  # (H, W) native PASTIS id
            preds_by_patch[pid] = native_to_italia[native_pred].astype(np.int64)
    logger.info(
        "italia_zero_shot_done",
        model_kind=model_kind,
        n_patches=len(preds_by_patch),
        test_fold=test_fold,
    )
    return preds_by_patch


def _config_to_dict(config: DenseFineTuneConfig) -> dict[str, object]:
    """Serialize the config to a JSON-friendly dict (for the checkpoint payload)."""
    return {
        "model_kind": config.model_kind,
        "n_timesteps": config.n_timesteps,
        "head_warmup_epochs": config.head_warmup_epochs,
        "finetune_epochs": config.finetune_epochs,
        "lr_head": config.lr_head,
        "lr_backbone": config.lr_backbone,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "min_patches_per_class": config.min_patches_per_class,
        "ignore_background": config.ignore_background,
        "seed": config.seed,
    }
