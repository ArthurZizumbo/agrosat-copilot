"""Destilacion FarSLIP — perdidas + trainer (US-017 / US-016b).

Implementa el procedimiento Li et al. 2025 (arXiv:2511.14901):

- :class:`PatchDistillationLoss` (paper §3.2): MSE + cosine entre los 196
  patches del student y los del teacher, con ``stop-grad`` explicito sobre las
  features teacher para evitar back-prop hacia el modelo congelado.
- :class:`RegionCategoryAlignmentLoss` (paper §3.3): InfoNCE contrastivo sobre
  el token CLS del student vs prototipos textuales region x categoria. Los
  prototipos se calculan UNA vez por epoch (text encoder frozen).
- :class:`FarSLIPDistillationTrainer`: orquesta el loop AdamW bf16 con MLflow
  autolog. Inicializa student desde teacher (``copy.deepcopy``), adapta
  ``patch_embed.proj`` de 3 a 4 canales con init = mean(RGB) para el canal NIR
  (evita dead-neuron). Hard cap 8 h, warning a 6 h.

VRAM esperada en GCP L4 24 GB: ~22 GB (ViT-B/16 bf16, batch=64, grad_accum=2).
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
        "FarSLIP requiere transformers>=4.46. Instala con `poetry add transformers`."
    ) from exc

_log = structlog.get_logger(__name__)

LossType = Literal["mse", "cosine", "mse_plus_cosine"]
SaveFormat = Literal["safetensors", "pytorch"]


# ---------------------------------------------------------------------------
# Patch-to-patch distillation loss (AC-1, AC-7).
# ---------------------------------------------------------------------------


class PatchDistillationLoss(nn.Module):
    """Perdida de destilacion parche-a-parche (FarSLIP §3.2).

    Combina MSE y/o coseno entre las features de los 196 patches del student
    y del teacher. El teacher se asume congelado; aplicamos ``.detach()`` para
    garantizar stop-gradient explicito (defensive contra fallos del caller).

    Args:
        loss_type: ``"mse"``, ``"cosine"`` o ``"mse_plus_cosine"`` (default).
        cosine_weight: peso del termino coseno cuando ``loss_type=="mse_plus_cosine"``.
        normalize: si ``True``, normaliza L2 las features antes del computo.
    """

    def __init__(
        self,
        loss_type: LossType = "mse_plus_cosine",
        cosine_weight: float = 0.3,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if loss_type not in ("mse", "cosine", "mse_plus_cosine"):
            raise ValueError(f"loss_type invalido: {loss_type!r}")
        if not 0.0 <= cosine_weight <= 1.0:
            raise ValueError(f"cosine_weight fuera de [0,1]: {cosine_weight}")
        self.loss_type = loss_type
        self.cosine_weight = cosine_weight
        self.normalize = normalize

    def forward(
        self,
        student_patch_feats: torch.Tensor,
        teacher_patch_feats: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calcula la perdida escalar diferenciable wrt ``student_patch_feats``.

        Args:
            student_patch_feats: tensor ``(B, P, D)`` (P=196 patches default).
            teacher_patch_feats: tensor ``(B, P, D)`` — sera detacheado.
            patch_mask: opcional bool ``(B, P)`` (True = patch valido).

        Returns:
            Loss escalar tensor con grad activo respecto al student.
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
            # 1 - cos similarity; ambos ya estan L2-norm si self.normalize True.
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
    """Alineacion region-categoria sobre el token CLS (FarSLIP §3.3).

    Calcula InfoNCE contrastivo entre el CLS del student y los prototipos
    textuales ``(n_regions * n_categories, D)`` precomputados por el text
    encoder del teacher (frozen). El positivo de cada sample es el prototipo
    correspondiente a su pareja (region_id, category_id).

    Args:
        temperature: temperatura softmax (default 0.07, paper §3.3).
        n_regions: numero de regiones (3 default para Italia).
        n_categories: numero de clases CAP (32 default).
    """

    def __init__(
        self,
        temperature: float = 0.07,
        n_regions: int = 3,
        n_categories: int = 32,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature debe ser positivo: {temperature}")
        if n_regions < 1 or n_categories < 1:
            raise ValueError("n_regions y n_categories deben ser >= 1")
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
        """Calcula InfoNCE contrastivo.

        Args:
            student_cls: tensor ``(B, D)`` del student.
            text_prototypes: tensor ``(n_regions * n_categories, D)``; se
                detachea internamente (frozen).
            region_ids: tensor long ``(B,)`` con region index.
            category_ids: tensor long ``(B,)`` con categoria index.

        Returns:
            Loss escalar tensor con grad activo respecto al student.
        """

        if student_cls.dim() != 2:
            raise ValueError(f"student_cls debe ser (B,D); got {student_cls.shape}")
        if text_prototypes.dim() != 2:
            raise ValueError(
                f"text_prototypes debe ser (R*C,D); got {text_prototypes.shape}"
            )
        expected_protos = self.n_regions * self.n_categories
        if text_prototypes.shape[0] != expected_protos:
            raise ValueError(
                f"text_prototypes filas={text_prototypes.shape[0]} esperado={expected_protos}"
            )
        if region_ids.shape != category_ids.shape:
            raise ValueError("region_ids y category_ids deben tener mismo shape")
        if region_ids.shape[0] != student_cls.shape[0]:
            raise ValueError("batch size inconsistente entre student_cls e ids")
        if (region_ids < 0).any() or (region_ids >= self.n_regions).any():
            raise ValueError("region_ids fuera de rango")
        if (category_ids < 0).any() or (category_ids >= self.n_categories).any():
            raise ValueError("category_ids fuera de rango")

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
    """Hparams de :class:`FarSLIPDistillationTrainer`.

    Attributes:
        teacher_model_id: HF id del CLIP teacher.
        dataset_root: ruta a ``data/farslip_pairs/`` (manifest + crops).
        output_dir: ruta local de pesos antes de subir a GCS.
        gcs_output_uri: ``gs://agrosat-models/farslip/{run_name}/`` opcional.
        loss_weights: ``{"alpha":1.0, "beta":0.5, "gamma":0.2}`` default.
        n_epochs: AC-4 default 4.
        batch_size: AC-4 default 64 (effective 128 con grad_accum=2).
        grad_accum_steps: AC-4 default 2.
        lr: AC-4 default 1e-5 AdamW.
        weight_decay: 0.01 default.
        warmup_ratio: 0.05 cosine warmup.
        seed: 42 (propagado a torch/np/random + deterministic algos).
        mlflow_run_name: ``"farslip-clip-italy-v1"``.
        device: ``"cuda"`` | ``"cpu"`` | ``"auto"``.
        time_cap_hours: hard cap 8 h (warning a 6 h).
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
    """Resuelve ``"auto"`` -> ``"cuda"`` si disponible, sino ``"cpu"``."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def adapt_patch_embed_to_n_channels(
    vision_model: nn.Module, target_channels: int
) -> None:
    """Adapta el ``patch_embedding`` de un CLIP vision model a ``target_channels``.

    Soporta tanto :class:`CLIPVisionModel` (transformers 5.x, plano: tiene
    ``embeddings`` directo) como ``CLIPModel.vision_model`` (con jerarquia).
    El canal extra (NIR) se inicializa como ``mean`` de los 3 RGB para evitar
    dead-neuron (init en ceros aplanaria la senal NIR). Modifica el modulo
    in-place. Reusa el mismo bias (no hay bias en patch_embed CLIP).
    """

    # Resolver ``embeddings`` con fallback (transformers 4.x vs 5.x).
    if hasattr(vision_model, "embeddings"):
        embeddings = vision_model.embeddings  # type: ignore[union-attr]
    elif hasattr(vision_model, "vision_model"):
        embeddings = vision_model.vision_model.embeddings  # type: ignore[union-attr]
    else:
        raise AttributeError("vision_model no expone .embeddings ni .vision_model")
    old_proj = embeddings.patch_embedding  # type: ignore[union-attr]
    assert isinstance(old_proj, nn.Conv2d), (
        f"patch_embedding debe ser Conv2d; got {type(old_proj).__name__}"
    )
    if old_proj.in_channels == target_channels:
        return
    if old_proj.in_channels != 3:
        raise ValueError(
            f"esperado patch_embed con 3 input channels, got {old_proj.in_channels}"
        )
    out_ch = old_proj.out_channels
    # Conv2d.kernel_size/stride/padding son tuple[int, int] en runtime aunque
    # el type-stub publique tuple[int, ...]. Cast explicito para mypy.
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
        # copiar primeros 3 canales tal cual
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
    """Trainer end-to-end de destilacion FarSLIP.

    Inicializa teacher (frozen) y student (clon profundo + trainable) desde el
    mismo HF id, adapta patch_embed a ``n_in_channels``, configura AdamW + cosine
    warmup + AMP bf16 + grad accumulation, registra MLflow autolog y guarda
    pesos en formato safetensors.

    Args ver :class:`FarSLIPTrainerConfig`.
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
        # text_prototypes opcional: si None, el trainer espera que el caller
        # los provea via :meth:`set_text_prototypes` antes de :meth:`train`.
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
        # Student arranca como copia exacta del teacher (AC-2)
        student = copy.deepcopy(teacher)
        for p in student.parameters():
            p.requires_grad_(True)
        student.train()
        self.teacher = teacher.to(self.device)  # type: ignore[arg-type]
        self.student = student.to(self.device)  # type: ignore[arg-type]

    def _patch_student_proj(self) -> None:
        """Adapta SOLO el student a ``n_in_channels`` bandas Sentinel-2.

        El teacher se conserva con 3 canales (RGB puro, paper FarSLIP §3.2 +
        AC-2: teacher = ``openai/clip-vit-base-patch16`` original). El forward
        del teacher recibe las 3 primeras bandas del student via slicing en
        :meth:`_teacher_forward`. Esto preserva la senal de destilacion
        autentica del CLIP pretrained — adaptar tambien el teacher
        contaminaba la pseudo-label con una proyeccion NIR no entrenada.
        """
        adapt_patch_embed_to_n_channels(self.student, self.config.n_in_channels)
        # Re-mover al device (nuevas capas creadas en CPU).
        self.student.to(self.device)
        # Teacher queda con 3 canales: NO se toca su patch_embed.

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
        """Inyecta prototipos textuales precomputados ``(R*C, D)``.

        Calculados externamente para evitar acoplar el text encoder al trainer
        (el text encoder esta frozen, basta correrlo 1x por epoch).
        """
        self._text_prototypes = prototypes.to(self.device)

    def step(
        self,
        images: torch.Tensor,
        region_ids: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward + backward de UN batch (sin optimizer.step).

        Devuelve dict con tensores de perdida (no detacheados) para que el
        caller decida cuando hacer ``backward`` + ``optimizer.step`` (smoke
        tests usan este metodo bajo el capot).
        """
        if self._text_prototypes is None:
            raise RuntimeError(
                "text_prototypes no inicializados. Llama set_text_prototypes()."
            )
        images = images.to(self.device)
        region_ids = region_ids.to(self.device)
        category_ids = category_ids.to(self.device)

        student_out = self.student(pixel_values=images, output_hidden_states=False)
        # Teacher se queda con 3 canales (RGB puro = B04/B03/B02 = BGR slice).
        # AC-2: preservamos el CLIP pretrained autentico; el student aprende
        # a mapear 4 bandas a la misma semantica que el teacher ve en 3.
        teacher_input = images[:, :3, :, :]
        with torch.no_grad():
            teacher_out = self.teacher(
                pixel_values=teacher_input, output_hidden_states=False
            )

        # CLIPVisionModel last_hidden_state shape: (B, 1+P, D) con CLS en pos 0.
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
        # auxiliar contrastive imagen-texto-batch: alinea CLS student con CLS teacher
        # (placeholder ligero para gamma; estabiliza el training)
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
        """Ejecuta ``n_epochs`` completas con MLflow autolog.

        Args:
            dataloader: opcional. Si no se pasa, requiere ``self._dataset`` set.

        Returns:
            Dict con ``loss_total`` y demas metricas finales (epoch ultima).
        """
        if dataloader is None:
            if self._dataset is None:
                raise RuntimeError("dataset y dataloader nulos: nada que entrenar")
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

        # Import mlflow una sola vez (Q14). ``mlflow`` queda en ``None`` si la
        # libreria no esta instalada — el loop sigue sin logging remoto.
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
                        # data_version = hash del .dvc file (no ruta local).
                        # Si el dataset aun no esta DVC-tracked, devuelve
                        # ``"<path>@untracked"`` y se documenta en el run.
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
                # Checkpoint por epoch (resilencia AC-9 R3)
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
        """Persiste pesos del student.

        Args:
            format: ``"safetensors"`` (default) o ``"pytorch"``.
            suffix: opcional, p.ej. ``"epoch_3"``; sufijo del archivo.

        Returns:
            Ruta local absoluta del archivo escrito.
        """
        out_dir = self.config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        name = "student"
        if suffix:
            name = f"{name}_{suffix}"
        # ``self.student`` es CLIPVisionModel: TODO el state_dict pertenece al
        # vision encoder (no hay text encoder en este wrapper). El extractor
        # carga directamente al ``vision_model`` con ``strict=False`` para
        # tolerar diferencias de prefijo entre CLIPVisionModel y CLIPModel.
        # Filtramos defensivamente prefijos `text_*` o `logit_scale` si
        # alguna iteracion futura introduce un wrapper compuesto.
        raw_state = self.student.state_dict()
        state_dict = {
            k: v
            for k, v in raw_state.items()
            if not k.startswith(("text_", "logit_scale"))
        }
        if format == "safetensors":
            from safetensors.torch import save_file

            path = out_dir / f"{name}.safetensors"
            # safetensors requiere tensors contiguos sobre CPU
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
    """Factory ergonomica con defaults validados en planning."""

    cfg = FarSLIPTrainerConfig(dataset_root=dataset_root, output_dir=output_dir)
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"FarSLIPTrainerConfig no tiene atributo {k!r}")
        setattr(cfg, k, v)
    return FarSLIPDistillationTrainer(cfg)


__all__ = [
    "FarSLIPDistillationTrainer",
    "FarSLIPTrainerConfig",
    "PatchDistillationLoss",
    "RegionCategoryAlignmentLoss",
    "build_default_trainer",
]
