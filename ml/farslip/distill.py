"""FarSLIP distillation — losses + trainer (US-017 / US-016b).

Implements the Li et al. 2025 procedure (arXiv:2511.14901):

- :class:`PatchDistillationLoss` (paper §3.2): MSE + cosine between the 196
  patches of the student and those of the teacher, with explicit ``stop-grad``
  on the teacher features to avoid back-prop toward the frozen model.
- :class:`RegionCategoryAlignmentLoss` (paper §3.3): contrastive InfoNCE on the
  student's CLS token vs region x category text prototypes. The prototypes are
  computed ONCE per epoch (text encoder frozen).
- :class:`FarSLIPDistillationTrainer`: orchestrates the AdamW bf16 loop with
  MLflow autolog. Initializes the student from the teacher (``copy.deepcopy``),
  adapts ``patch_embed.proj`` from 3 to 4 channels with init = mean(RGB) for the
  NIR channel (avoids dead-neuron). Hard cap 8 h, warning at 6 h.

Expected VRAM on GCP L4 24 GB: ~22 GB (ViT-B/16 bf16, batch=64, grad_accum=2).
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ml.utils.git_meta import dvc_data_version, git_sha
from ml.utils.seed import propagate_seed

try:
    from transformers import CLIPVisionModel
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "FarSLIP requires transformers>=4.46. Install with `poetry add transformers`."
    ) from exc

_log = structlog.get_logger(__name__)

LossType = Literal["mse", "cosine", "mse_plus_cosine"]
SaveFormat = Literal["safetensors", "pytorch"]


# ---------------------------------------------------------------------------
# Patch-to-patch distillation loss (AC-1, AC-7).
# ---------------------------------------------------------------------------


class PatchDistillationLoss(nn.Module):
    """Patch-to-patch distillation loss (FarSLIP §3.2).

    Combines MSE and/or cosine between the features of the 196 patches of the
    student and the teacher. The teacher is assumed frozen; we apply ``.detach()``
    to guarantee explicit stop-gradient (defensive against caller failures).

    Args:
        loss_type: ``"mse"``, ``"cosine"`` or ``"mse_plus_cosine"`` (default).
        cosine_weight: weight of the cosine term when ``loss_type=="mse_plus_cosine"``.
        normalize: if ``True``, L2-normalizes the features before the computation.
    """

    def __init__(
        self,
        loss_type: LossType = "mse_plus_cosine",
        cosine_weight: float = 0.3,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if loss_type not in ("mse", "cosine", "mse_plus_cosine"):
            raise ValueError(f"invalid loss_type: {loss_type!r}")
        if not 0.0 <= cosine_weight <= 1.0:
            raise ValueError(f"cosine_weight out of [0,1]: {cosine_weight}")
        self.loss_type = loss_type
        self.cosine_weight = cosine_weight
        self.normalize = normalize

    def forward(
        self,
        student_patch_feats: torch.Tensor,
        teacher_patch_feats: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the scalar loss differentiable wrt ``student_patch_feats``.

        Args:
            student_patch_feats: tensor ``(B, P, D)`` (P=196 patches default).
            teacher_patch_feats: tensor ``(B, P, D)`` — will be detached.
            patch_mask: optional bool ``(B, P)`` (True = valid patch).

        Returns:
            Scalar loss tensor with grad active with respect to the student.
        """

        if student_patch_feats.shape != teacher_patch_feats.shape:
            raise ValueError(
                f"shape mismatch student={student_patch_feats.shape} "
                f"teacher={teacher_patch_feats.shape}"
            )
        teacher = teacher_patch_feats.detach()
        student = student_patch_feats

        if self.normalize:
            student = F.normalize(student, p=2, dim=-1)
            teacher = F.normalize(teacher, p=2, dim=-1)

        if patch_mask is not None:
            if patch_mask.shape != student.shape[:2]:
                raise ValueError(
                    f"mask shape mismatch mask={patch_mask.shape} "
                    f"feats={student.shape[:2]}"
                )
            mask = patch_mask.unsqueeze(-1).to(student.dtype)
            n_valid = mask.sum().clamp(min=1.0)
        else:
            mask = None
            n_valid = torch.tensor(
                float(student.shape[0] * student.shape[1]),
                device=student.device,
                dtype=student.dtype,
            )

        mse_term = torch.tensor(0.0, device=student.device, dtype=student.dtype)
        cos_term = torch.tensor(0.0, device=student.device, dtype=student.dtype)

        if self.loss_type in ("mse", "mse_plus_cosine"):
            squared = (student - teacher).pow(2).sum(dim=-1)  # (B, P)
            if mask is not None:
                squared = squared * mask.squeeze(-1)
            mse_term = squared.sum() / n_valid

        if self.loss_type in ("cosine", "mse_plus_cosine"):
            # 1 - cos similarity; both are already L2-norm if self.normalize True.
            if not self.normalize:
                s = F.normalize(student, p=2, dim=-1)
                t = F.normalize(teacher, p=2, dim=-1)
            else:
                s, t = student, teacher
            cos_sim = (s * t).sum(dim=-1)  # (B, P)
            cos_loss = 1.0 - cos_sim
            if mask is not None:
                cos_loss = cos_loss * mask.squeeze(-1)
            cos_term = cos_loss.sum() / n_valid

        if self.loss_type == "mse":
            return mse_term
        if self.loss_type == "cosine":
            return cos_term
        return mse_term + self.cosine_weight * cos_term


# ---------------------------------------------------------------------------
# Region x Category InfoNCE alignment (AC-1, AC-7).
# ---------------------------------------------------------------------------


class RegionCategoryAlignmentLoss(nn.Module):
    """Region-category alignment on the CLS token (FarSLIP §3.3).

    Computes contrastive InfoNCE between the student's CLS and the text
    prototypes ``(n_regions * n_categories, D)`` precomputed by the teacher's text
    encoder (frozen). The positive of each sample is the prototype corresponding
    to its (region_id, category_id) pair.

    Args:
        temperature: softmax temperature (default 0.07, paper §3.3).
        n_regions: number of regions (3 default for Italy).
        n_categories: number of CAP classes (32 default).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        n_regions: int = 3,
        n_categories: int = 32,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive: {temperature}")
        if n_regions < 1 or n_categories < 1:
            raise ValueError("n_regions and n_categories must be >= 1")
        self.temperature = temperature
        self.n_regions = n_regions
        self.n_categories = n_categories

    def forward(
        self,
        student_cls: torch.Tensor,
        text_prototypes: torch.Tensor,
        region_ids: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Computes contrastive InfoNCE.

        Args:
            student_cls: tensor ``(B, D)`` of the student.
            text_prototypes: tensor ``(n_regions * n_categories, D)``; it is
                detached internally (frozen).
            region_ids: long tensor ``(B,)`` with region index.
            category_ids: long tensor ``(B,)`` with category index.

        Returns:
            Scalar loss tensor with grad active with respect to the student.
        """

        if student_cls.dim() != 2:
            raise ValueError(f"student_cls must be (B,D); got {student_cls.shape}")
        if text_prototypes.dim() != 2:
            raise ValueError(
                f"text_prototypes must be (R*C,D); got {text_prototypes.shape}"
            )
        expected_protos = self.n_regions * self.n_categories
        if text_prototypes.shape[0] != expected_protos:
            raise ValueError(
                f"text_prototypes rows={text_prototypes.shape[0]} expected={expected_protos}"
            )
        if region_ids.shape != category_ids.shape:
            raise ValueError("region_ids and category_ids must have the same shape")
        if region_ids.shape[0] != student_cls.shape[0]:
            raise ValueError("inconsistent batch size between student_cls and ids")
        if (region_ids < 0).any() or (region_ids >= self.n_regions).any():
            raise ValueError("region_ids out of range")
        if (category_ids < 0).any() or (category_ids >= self.n_categories).any():
            raise ValueError("category_ids out of range")

        protos = text_prototypes.detach()
        student_n = F.normalize(student_cls, p=2, dim=-1)
        protos_n = F.normalize(protos, p=2, dim=-1)

        # logits: (B, n_regions * n_categories)
        logits = student_n @ protos_n.t() / self.temperature
        # target index: region * n_categories + category
        targets = region_ids.long() * self.n_categories + category_ids.long()
        return F.cross_entropy(logits, targets)


# ---------------------------------------------------------------------------
# Trainer (AC-2, AC-4, AC-5, AC-9).
# ---------------------------------------------------------------------------


@dataclass
class FarSLIPTrainerConfig:
    """Hparams of :class:`FarSLIPDistillationTrainer`.

    Attributes:
        teacher_model_id: HF id of the CLIP teacher.
        dataset_root: path to ``data/farslip_pairs/`` (manifest + crops).
        output_dir: local path of the weights before uploading to GCS.
        gcs_output_uri: optional ``gs://agrosat-models/farslip/{run_name}/``.
        loss_weights: ``{"alpha":1.0, "beta":0.5, "gamma":0.2}`` default.
        n_epochs: AC-4 default 4.
        batch_size: AC-4 default 64 (effective 128 with grad_accum=2).
        grad_accum_steps: AC-4 default 2.
        lr: AC-4 default 1e-5 AdamW.
        weight_decay: 0.01 default.
        warmup_ratio: 0.05 cosine warmup.
        seed: 42 (propagated to torch/np/random + deterministic algos).
        mlflow_run_name: ``"farslip-clip-italy-v1"``.
        device: ``"cuda"`` | ``"cpu"`` | ``"auto"``.
        time_cap_hours: hard cap 8 h (warning at 6 h).
        num_workers: DataLoader workers default 4.
        n_in_channels: 4 (B02 B03 B04 B08).
        n_regions: 3.
        n_categories: 32.
    """

    teacher_model_id: str = "openai/clip-vit-base-patch16"
    dataset_root: Path = Path("data/farslip_pairs")
    output_dir: Path = Path("artifacts/farslip")
    gcs_output_uri: str | None = None
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {"alpha": 1.0, "beta": 0.5, "gamma": 0.2}
    )
    n_epochs: int = 4
    batch_size: int = 64
    grad_accum_steps: int = 2
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    seed: int = 42
    mlflow_run_name: str = "farslip-clip-italy-v1"
    device: str = "auto"
    time_cap_hours: float = 8.0
    warning_hours: float = 6.0
    num_workers: int = 4
    n_in_channels: int = 4
    n_regions: int = 3
    n_categories: int = 32


def _resolve_device(device: str) -> torch.device:
    """Resolves ``"auto"`` -> ``"cuda"`` if available, otherwise ``"cpu"``."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def adapt_patch_embed_to_n_channels(
    vision_model: nn.Module, target_channels: int
) -> None:
    """Adapts the ``patch_embedding`` of a CLIP vision model to ``target_channels``.

    Supports both :class:`CLIPVisionModel` (transformers 5.x, flat: it has
    ``embeddings`` directly) and ``CLIPModel.vision_model`` (with hierarchy). The
    extra channel (NIR) is initialized as the ``mean`` of the 3 RGB to avoid
    dead-neuron (zero init would flatten the NIR signal). Modifies the module
    in-place. Reuses the same bias (there is no bias in the CLIP patch_embed).
    """

    # Resolve ``embeddings`` with fallback (transformers 4.x vs 5.x).
    if hasattr(vision_model, "embeddings"):
        embeddings = vision_model.embeddings  # type: ignore[union-attr]
    elif hasattr(vision_model, "vision_model"):
        embeddings = vision_model.vision_model.embeddings  # type: ignore[union-attr]
    else:
        raise AttributeError("vision_model exposes neither .embeddings nor .vision_model")
    old_proj = embeddings.patch_embedding  # type: ignore[union-attr]
    assert isinstance(old_proj, nn.Conv2d), (
        f"patch_embedding must be Conv2d; got {type(old_proj).__name__}"
    )
    if old_proj.in_channels == target_channels:
        return
    if old_proj.in_channels != 3:
        raise ValueError(
            f"expected patch_embed with 3 input channels, got {old_proj.in_channels}"
        )
    out_ch = old_proj.out_channels
    # Conv2d.kernel_size/stride/padding are tuple[int, int] at runtime although
    # the type-stub publishes tuple[int, ...]. Explicit cast for mypy.
    k: tuple[int, int] = (old_proj.kernel_size[0], old_proj.kernel_size[1])
    stride: tuple[int, int] = (old_proj.stride[0], old_proj.stride[1])
    if isinstance(old_proj.padding, str):
        padding: str | tuple[int, int] = old_proj.padding
    else:
        padding = (old_proj.padding[0], old_proj.padding[1])
    bias_flag = old_proj.bias is not None

    new_proj = nn.Conv2d(
        in_channels=target_channels,
        out_channels=out_ch,
        kernel_size=k,
        stride=stride,
        padding=padding,
        bias=bias_flag,
    )
    with torch.no_grad():
        # copy first 3 channels as-is
        new_proj.weight[:, :3, :, :] = old_proj.weight.detach().clone()
        if target_channels > 3:
            rgb_mean = old_proj.weight.detach().mean(dim=1, keepdim=True)  # (O,1,k,k)
            for ch in range(3, target_channels):
                new_proj.weight[:, ch : ch + 1, :, :] = rgb_mean.clone()
        if bias_flag and old_proj.bias is not None and new_proj.bias is not None:
            new_proj.bias.copy_(old_proj.bias.detach().clone())
    embeddings.patch_embedding = new_proj  # type: ignore[union-attr]
    _log.info(
        "patch_embed adapted",
        from_channels=3,
        to_channels=target_channels,
        init="mean_rgb_on_extra",
    )


class FarSLIPDistillationTrainer:
    """End-to-end FarSLIP distillation trainer.

    Initializes teacher (frozen) and student (deep clone + trainable) from the
    same HF id, adapts patch_embed to ``n_in_channels``, configures AdamW + cosine
    warmup + AMP bf16 + grad accumulation, records MLflow autolog and saves
    weights in safetensors format.

    Args see :class:`FarSLIPTrainerConfig`.
    """

    def __init__(
        self,
        config: FarSLIPTrainerConfig,
        dataset: torch.utils.data.Dataset | None = None,
        text_prototypes: torch.Tensor | None = None,
    ) -> None:
        self.config = config
        self.device = _resolve_device(config.device)
        propagate_seed(config.seed)
        self._load_models()
        self._patch_student_proj()
        self._optim = self._build_optimizer()
        self._scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
        self._dataset = dataset
        # text_prototypes optional: if None, the trainer expects the caller
        # to provide them via :meth:`set_text_prototypes` before :meth:`train`.
        self._text_prototypes = text_prototypes
        self._patch_loss = PatchDistillationLoss()
        self._cls_loss = RegionCategoryAlignmentLoss(
            n_regions=config.n_regions, n_categories=config.n_categories
        )
        config.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ utils

    def _load_models(self) -> None:
        teacher = CLIPVisionModel.from_pretrained(self.config.teacher_model_id)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        # Student starts as an exact copy of the teacher (AC-2)
        student = copy.deepcopy(teacher)
        for p in student.parameters():
            p.requires_grad_(True)
        student.train()
        self.teacher = teacher.to(self.device)  # type: ignore[arg-type]
        self.student = student.to(self.device)  # type: ignore[arg-type]

    def _patch_student_proj(self) -> None:
        """Adapts ONLY the student to ``n_in_channels`` Sentinel-2 bands.

        The teacher is kept with 3 channels (pure RGB, FarSLIP paper §3.2 +
        AC-2: teacher = original ``openai/clip-vit-base-patch16``). The teacher's
        forward receives the first 3 bands of the student via slicing in
        :meth:`_teacher_forward`. This preserves the authentic distillation signal
        of the pretrained CLIP — adapting the teacher as well would contaminate
        the pseudo-label with an untrained NIR projection.
        """
        adapt_patch_embed_to_n_channels(self.student, self.config.n_in_channels)
        # Move back to the device (new layers created on CPU).
        self.student.to(self.device)
        # Teacher stays with 3 channels: its patch_embed is NOT touched.

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            (p for p in self.student.parameters() if p.requires_grad),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    def _build_scheduler(self, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
        warmup_steps = max(1, int(total_steps * self.config.warmup_ratio))

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self._optim, _lr_lambda)

    # ------------------------------------------------------------------ API

    def set_text_prototypes(self, prototypes: torch.Tensor) -> None:
        """Injects precomputed text prototypes ``(R*C, D)``.

        Computed externally to avoid coupling the text encoder to the trainer
        (the text encoder is frozen, running it once per epoch is enough).
        """
        self._text_prototypes = prototypes.to(self.device)

    def step(
        self,
        images: torch.Tensor,
        region_ids: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward + backward of ONE batch (without optimizer.step).

        Returns a dict with loss tensors (not detached) so that the caller decides
        when to do ``backward`` + ``optimizer.step`` (smoke tests use this method
        under the hood).
        """
        if self._text_prototypes is None:
            raise RuntimeError(
                "text_prototypes not initialized. Call set_text_prototypes()."
            )
        images = images.to(self.device)
        region_ids = region_ids.to(self.device)
        category_ids = category_ids.to(self.device)

        student_out = self.student(pixel_values=images, output_hidden_states=False)
        # Teacher stays with 3 channels (pure RGB = B04/B03/B02 = BGR slice).
        # AC-2: we preserve the authentic pretrained CLIP; the student learns
        # to map 4 bands to the same semantics the teacher sees in 3.
        teacher_input = images[:, :3, :, :]
        with torch.no_grad():
            teacher_out = self.teacher(
                pixel_values=teacher_input, output_hidden_states=False
            )

        # CLIPVisionModel last_hidden_state shape: (B, 1+P, D) with CLS at pos 0.
        student_hidden = student_out.last_hidden_state
        teacher_hidden = teacher_out.last_hidden_state
        student_cls = student_hidden[:, 0, :]
        teacher_cls = teacher_hidden[:, 0, :]
        student_patches = student_hidden[:, 1:, :]
        teacher_patches = teacher_hidden[:, 1:, :]

        loss_patch = self._patch_loss(student_patches, teacher_patches)
        loss_cls = self._cls_loss(
            student_cls, self._text_prototypes, region_ids, category_ids
        )
        # auxiliary contrastive image-text-batch: aligns student CLS with teacher CLS
        # (lightweight placeholder for gamma; stabilizes the training)
        cos_aux = 1.0 - F.cosine_similarity(
            F.normalize(student_cls, dim=-1),
            F.normalize(teacher_cls.detach(), dim=-1),
            dim=-1,
        ).mean()

        w = self.config.loss_weights
        total = w["alpha"] * loss_patch + w["beta"] * loss_cls + w["gamma"] * cos_aux
        return {
            "loss_total": total,
            "loss_patch": loss_patch,
            "loss_cls": loss_cls,
            "loss_aux": cos_aux,
        }

    def train(self, dataloader: DataLoader | None = None) -> dict[str, float]:
        """Runs ``n_epochs`` complete with MLflow autolog.

        Args:
            dataloader: optional. If not passed, requires ``self._dataset`` set.

        Returns:
            Dict with ``loss_total`` and the other final metrics (last epoch).
        """
        if dataloader is None:
            if self._dataset is None:
                raise RuntimeError("dataset and dataloader null: nothing to train")
            dataloader = DataLoader(
                self._dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=self.device.type == "cuda",
            )

        total_steps = max(1, len(dataloader) * self.config.n_epochs)
        self._scheduler = self._build_scheduler(total_steps)
        start = time.monotonic()
        warned = False

        # Import mlflow only once (Q14). ``mlflow`` stays ``None`` if the
        # library is not installed — the loop continues without remote logging.
        try:
            import mlflow as _mlflow
        except ImportError as exc:  # pragma: no cover
            _log.warning("mlflow no disponible", error=str(exc))
            _mlflow = None  # type: ignore[assignment]

        run_ctx = None
        if _mlflow is not None:
            try:
                _mlflow.set_experiment("farslip")
                run_ctx = _mlflow.start_run(run_name=self.config.mlflow_run_name)
                _mlflow.set_tags(
                    {
                        "code_version": git_sha(),
                        # data_version = hash of the .dvc file (not local path).
                        # If the dataset is not yet DVC-tracked, returns
                        # ``"<path>@untracked"`` and it is documented in the run.
                        "data_version": dvc_data_version(
                            str(self.config.dataset_root)
                        ),
                        "us": "US-017",
                        "us_alias": "US-016b",
                    }
                )
                _mlflow.log_params(
                    {
                        "teacher_model_id": self.config.teacher_model_id,
                        "n_epochs": self.config.n_epochs,
                        "batch_size": self.config.batch_size,
                        "grad_accum_steps": self.config.grad_accum_steps,
                        "lr": self.config.lr,
                        "weight_decay": self.config.weight_decay,
                        "warmup_ratio": self.config.warmup_ratio,
                        "seed": self.config.seed,
                        "n_in_channels": self.config.n_in_channels,
                        "loss_alpha": self.config.loss_weights["alpha"],
                        "loss_beta": self.config.loss_weights["beta"],
                        "loss_gamma": self.config.loss_weights["gamma"],
                    }
                )
            except RuntimeError as exc:  # pragma: no cover
                _log.warning("mlflow init fallo", error=str(exc))
                run_ctx = None

        last_metrics: dict[str, float] = {}
        global_step = 0
        try:
            for epoch in range(self.config.n_epochs):
                for batch in dataloader:
                    elapsed_h = (time.monotonic() - start) / 3600.0
                    if elapsed_h >= self.config.time_cap_hours:
                        _log.error("hard time cap reached, stopping", elapsed_h=elapsed_h)
                        return last_metrics
                    if not warned and elapsed_h >= self.config.warning_hours:
                        _log.warning("training over warning threshold", elapsed_h=elapsed_h)
                        warned = True

                    images = batch["image"]
                    region_ids = batch["region_id"]
                    category_ids = batch["category_id"]

                    losses = self.step(images, region_ids, category_ids)
                    total = losses["loss_total"] / self.config.grad_accum_steps
                    total.backward()

                    if (global_step + 1) % self.config.grad_accum_steps == 0:
                        self._optim.step()
                        if self._scheduler is not None:
                            self._scheduler.step()
                        self._optim.zero_grad(set_to_none=True)

                    if run_ctx is not None and _mlflow is not None:
                        try:
                            _mlflow.log_metrics(
                                {
                                    k: float(v.detach().cpu().item())
                                    for k, v in losses.items()
                                },
                                step=global_step,
                            )
                        except RuntimeError as exc:  # pragma: no cover
                            _log.debug("mlflow log_metrics fallo", error=str(exc))

                    last_metrics = {
                        k: float(v.detach().cpu().item()) for k, v in losses.items()
                    }
                    global_step += 1

                _log.info("epoch done", epoch=epoch, **last_metrics)
                # Checkpoint per epoch (resilience AC-9 R3)
                self.save_student(format="safetensors", suffix=f"epoch_{epoch}")
        finally:
            if run_ctx is not None and _mlflow is not None:
                try:
                    _mlflow.end_run()
                except RuntimeError as exc:  # pragma: no cover
                    _log.debug("mlflow end_run fallo", error=str(exc))

        return last_metrics

    def save_student(
        self,
        format: SaveFormat = "safetensors",
        suffix: str | None = None,
    ) -> str:
        """Persists the student weights.

        Args:
            format: ``"safetensors"`` (default) or ``"pytorch"``.
            suffix: optional, e.g. ``"epoch_3"``; file suffix.

        Returns:
            Absolute local path of the written file.
        """
        out_dir = self.config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        name = "student"
        if suffix:
            name = f"{name}_{suffix}"
        # ``self.student`` is CLIPVisionModel: the ENTIRE state_dict belongs to
        # the vision encoder (there is no text encoder in this wrapper). The
        # extractor loads directly into ``vision_model`` with ``strict=False`` to
        # tolerate prefix differences between CLIPVisionModel and CLIPModel.
        # We defensively filter `text_*` or `logit_scale` prefixes in case
        # a future iteration introduces a composite wrapper.
        raw_state = self.student.state_dict()
        state_dict = {
            k: v
            for k, v in raw_state.items()
            if not k.startswith(("text_", "logit_scale"))
        }
        if format == "safetensors":
            from safetensors.torch import save_file

            path = out_dir / f"{name}.safetensors"
            # safetensors requires contiguous tensors on CPU
            cpu_state = {k: v.detach().contiguous().cpu() for k, v in state_dict.items()}
            save_file(cpu_state, str(path))
        else:
            path = out_dir / f"{name}.pt"
            torch.save(state_dict, path)
        _log.info("student weights saved", path=str(path), format=format)
        return str(path.resolve())


def build_default_trainer(
    dataset_root: Path = Path("data/farslip_pairs"),
    output_dir: Path = Path("artifacts/farslip"),
    **overrides: Any,
) -> FarSLIPDistillationTrainer:
    """Ergonomic factory with defaults validated during planning."""

    cfg = FarSLIPTrainerConfig(dataset_root=dataset_root, output_dir=output_dir)
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"FarSLIPTrainerConfig has no attribute {k!r}")
        setattr(cfg, k, v)
    return FarSLIPDistillationTrainer(cfg)


__all__ = [
    "FarSLIPDistillationTrainer",
    "FarSLIPTrainerConfig",
    "PatchDistillationLoss",
    "RegionCategoryAlignmentLoss",
    "build_default_trainer",
]
