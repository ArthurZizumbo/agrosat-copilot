"""Loop de entrenamiento compartido de segmentacion densa PASTIS-R (US-025).

Loop unico que entrena los dos segmentadores de la US-025 sobre el
:class:`ml.data.pastis_seg_dataset.PASTISSegmentationDataset`:

- **DeepLabv3+ MobileNetV3** (modo 2D, ``collapse_time="median"``): entrada
  ``(B, 10, H, W)``; solo head de segmentacion.
- **TSViT recortado** (modo temporal, ``collapse_time=None``): entrada
  ``(B, T, 10, H, W)``; head de segmentacion + (opcional) **rama
  fenologica-contrastiva** (Wen et al. 2025, ec. 15-16) que alinea por
  contraste las features visuales por pixel con el prototipo semantico de la
  clase de cada pixel.

La perdida total es::

    L = L_dice_ce  +  (lambda_contrast * L_contrast   si use_phenology)

donde ``L_dice_ce`` sale de
:func:`ml.models.deeplabv3plus.build_dice_ce_loss` y ``L_contrast`` es el
InfoNCE simetrico pixel<->prototipo de :func:`phenology_contrastive_loss`.

El loop detecta automaticamente la forma de la entrada (4D vs 5D) y, para
TSViT con ``use_phenology=True``, pide al modelo la proyeccion visual con
``return_visual_proj=True``. El registro MLflow reusa
:func:`ml.utils.mlflow_utils.track_experiment` (tags obligatorios
``data_version`` + ``code_version``, regla CLAUDE.md 10) y loguea
``loss``/``miou``/``f1_macro``/``pixel_acc`` por epoch sobre el split de
validacion via :func:`ml.eval.metrics.segmentation_metrics_report`.

Decisiones tecnicas
-------------------

- **Device auto**: ``"cuda"`` si esta disponible, si no ``"cpu"`` (regla
  CLAUDE.md ML: ``_resolve_device`` prioriza CUDA). AMP opcional (``use_amp``)
  solo se activa en CUDA.
- **mIoU/F1 por epoch en val**: se acumula la matriz de confusion densa de
  todo el split (no por-batch) para que la metrica sea exacta a nivel de
  conjunto, no un promedio de promedios.
- **Mejor epoch por mIoU val**: se conserva el ``state_dict`` del mejor epoch
  y se devuelven sus metricas (criterio de seleccion del paper de
  segmentacion: mIoU es la metrica principal de la rubrica).
- **Sin pandas**: ``torch``/``numpy`` solo en el borde del modelo; logging via
  ``structlog``.

Atribucion: la alineacion contrastiva sigue Wen et al. (2025), "Phenology
Description is All You Need!", ISPRS J. Photogrammetry RS 228 (ec. 15-16).
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import numpy as np
import structlog
import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.eval.metrics import (
    _per_class_iou_from_cm,
    dense_confusion_matrix,
)
from ml.models.deeplabv3plus import build_dice_ce_loss
from ml.utils.mlflow_utils import track_experiment

if TYPE_CHECKING:  # pragma: no cover - solo anotaciones de tipo
    from collections.abc import Sequence

    from torch.utils.data import Dataset

logger = structlog.get_logger(__name__)

__all__ = [
    "phenology_contrastive_loss",
    "train_segmentation",
]

#: Experimento MLflow de los segmentadores del EPIC 5.
_EXPERIMENT_NAME = "agrosat-segmentation"

#: Ruta (relativa al repo) del dataset PASTIS-R para resolver el ``data_version``
#: DVC. Si no hay ``.dvc`` file, ``track_experiment`` cae a ``"untracked"``.
_PASTIS_DVC_PATH = "data/PASTIS-R"


# ---------------------------------------------------------------------------
# Loss contrastivo fenologico (Wen et al. 2025, ec. 15-16)
# ---------------------------------------------------------------------------


def phenology_contrastive_loss(
    visual_proj: torch.Tensor,
    target: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    ignore_index: int = 255,
    temperature: float = 0.07,
    max_pixels: int = 4096,
) -> torch.Tensor:
    """InfoNCE simetrico pixel-visual <-> prototipo-de-clase (Wen et al. 2025).

    Alinea cada feature visual por pixel ``visual_proj[:, :, i, j]`` con el
    prototipo semantico de la **clase de ese pixel** (``target[:, i, j]``),
    siguiendo la alineacion contrastiva del paper (ec. 15-16, ``L_cl =
    (L_v + L_s) / 2``). A diferencia de la concatenacion tabular (que degrado
    el baseline), el contraste empuja las features visuales hacia el cluster
    semantico de su clase sin inflar la dimensionalidad del head.

    Implementacion:

    1. Se aplanan los pixeles validos (``target != ignore_index`` y dentro de
       ``[0, num_prototipos)``).
    2. Se submuestrean ``max_pixels`` pixeles (memoria acotada en GPU; el
       contraste no necesita todos los pixeles del batch para una senal de
       gradiente estable).
    3. ``visual_proj`` y ``prototypes`` se L2-normalizan; la matriz de
       similitud ``logits = (v @ p^T) / temperature`` se compara contra la
       etiqueta de clase de cada pixel con CrossEntropy en **ambas
       direcciones** (pixel->prototipo y prototipo->pixel agregado), promediadas.

    Args:
        visual_proj: Proyeccion visual por pixel ``(B, S, H, W)`` con ``S`` =
            dimension del espacio semantico (384) de los prototipos.
        target: Etiquetas por pixel ``(B, H, W)`` int; ``ignore_index`` y
            clases fuera de rango se excluyen.
        prototypes: Matriz de prototipos por clase ``(K, S)`` (uno por clase,
            indexable por la etiqueta de clase).
        ignore_index: Etiqueta de pixeles a ignorar (Background/Void).
        temperature: Temperatura del softmax contrastivo (0.07, estandar
            CLIP/InfoNCE).
        max_pixels: Numero maximo de pixeles validos a usar por llamada
            (submuestreo determinista por ``torch.randperm`` para acotar VRAM).

    Returns:
        Escalar ``torch.Tensor`` con la perdida contrastiva simetrica. Si no
        hay pixeles validos en el batch, devuelve ``0.0`` (tensor con grad).
    """
    device = visual_proj.device
    semantic_dim = visual_proj.shape[1]
    n_proto = prototypes.shape[0]

    protos = prototypes.to(device=device, dtype=visual_proj.dtype)
    protos = nn.functional.normalize(protos, dim=1)  # (K, S)

    # (B, S, H, W) -> (B*H*W, S) y (B, H, W) -> (B*H*W,)
    v_flat = visual_proj.permute(0, 2, 3, 1).reshape(-1, semantic_dim)
    y_flat = target.reshape(-1).long()

    valid = (y_flat != ignore_index) & (y_flat >= 0) & (y_flat < n_proto)
    if not bool(valid.any()):
        # Sin pixeles validos: termino neutro que conserva el grafo de grad.
        return visual_proj.sum() * 0.0

    v_valid = v_flat[valid]
    y_valid = y_flat[valid]

    # Submuestreo determinista para acotar la matriz de similitud en VRAM.
    n_valid = v_valid.shape[0]
    if n_valid > max_pixels:
        gen = torch.Generator(device="cpu").manual_seed(0)
        perm = torch.randperm(n_valid, generator=gen)[:max_pixels].to(device)
        v_valid = v_valid[perm]
        y_valid = y_valid[perm]

    v_valid = nn.functional.normalize(v_valid, dim=1)  # (P, S)

    # Similitud pixel x prototipo -> logits (P, K).
    logits = (v_valid @ protos.t()) / temperature

    # Direccion 1 (visual): cada pixel debe clasificar a su prototipo de clase.
    loss_v = nn.functional.cross_entropy(logits, y_valid)

    # Direccion 2 (semantica): por cada clase presente, el prototipo debe
    # recuperar sus pixeles. Se promedia la similitud prototipo->pixeles de su
    # clase contra todos los pixeles del batch (InfoNCE simetrico del paper).
    present = torch.unique(y_valid)
    proto_logits = (protos[present] @ v_valid.t()) / temperature  # (Kp, P)
    # Target multi-positivo: para cada prototipo presente, los pixeles de su
    # clase son los positivos; se usa la media de log-softmax sobre positivos.
    log_prob = nn.functional.log_softmax(proto_logits, dim=1)  # (Kp, P)
    pos_mask = (y_valid.unsqueeze(0) == present.unsqueeze(1)).to(log_prob.dtype)
    pos_counts = pos_mask.sum(dim=1).clamp_min(1.0)
    loss_s = -(log_prob * pos_mask).sum(dim=1) / pos_counts
    loss_s = loss_s.mean()

    loss: torch.Tensor = 0.5 * (loss_v + loss_s)
    return loss


# ---------------------------------------------------------------------------
# Helpers de device / forward
# ---------------------------------------------------------------------------


def _resolve_device(requested: str) -> torch.device:
    """Resuelve el device priorizando CUDA cuando esta disponible.

    Args:
        requested: ``"cuda"``, ``"cpu"`` o ``"auto"``. ``"cuda"`` sin GPU
            degrada a ``"cpu"`` con un warning estructurado.

    Returns:
        ``torch.device`` resuelto.
    """
    if requested in ("auto", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "cuda":
            logger.warning("cuda_requested_but_unavailable_fallback_cpu")
        return torch.device("cpu")
    return torch.device(requested)


def _forward_model(
    model: nn.Module,
    x: torch.Tensor,
    *,
    return_visual_proj: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Hace ``forward`` del modelo devolviendo logits y proyeccion visual.

    TSViT acepta el kwarg ``return_visual_proj`` y, cuando es ``True``,
    devuelve la tupla ``(logits, visual_proj)``. DeepLabv3+ (smp) no acepta el
    kwarg: se llama de forma estandar y ``visual_proj`` queda ``None``.

    Args:
        model: Segmentador (DeepLabv3+ o TSViT).
        x: Entrada ``(B, C, H, W)`` (2D) o ``(B, T, C, H, W)`` (temporal).
        return_visual_proj: Si ``True`` y el modelo lo soporta, pide la rama
            visual contrastiva.

    Returns:
        Tupla ``(logits (B, K, H, W), visual_proj | None)``.
    """
    if return_visual_proj:
        out = model(x, return_visual_proj=True)
        if isinstance(out, tuple):
            return out[0], out[1]
        # El modelo no honro el flag (caso defensivo): solo logits.
        return out, None
    out = model(x)
    return out, None


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    device: torch.device,
    use_phenology: bool,
    prototypes: torch.Tensor | None,
    lambda_contrast: float,
    ignore_index: int,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    use_amp: bool,
) -> float:
    """Corre una epoca de train (con ``optimizer``) o de eval (sin el).

    Args:
        model: Segmentador.
        loader: DataLoader del split.
        criterion: Perdida de segmentacion (Dice + CE).
        device: Device resuelto.
        use_phenology: Activa la rama contrastiva (solo si el modelo la expone).
        prototypes: Matriz de prototipos ``(K, S)`` o ``None``.
        lambda_contrast: Peso del termino contrastivo.
        ignore_index: Etiqueta ignorada (Background/Void).
        optimizer: Optimizador para train; ``None`` para eval (no backward).
        scaler: ``GradScaler`` de AMP o ``None``.
        use_amp: Si ``True`` usa autocast (solo efectivo en CUDA).

    Returns:
        Loss media de la epoca (escalar Python).
    """
    is_train = optimizer is not None
    model.train(is_train)

    amp_enabled = use_amp and device.type == "cuda"
    total_loss = 0.0
    n_batches = 0

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for x, y in loader:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).long()

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits, visual_proj = _forward_model(
                    model, x, return_visual_proj=use_phenology
                )
                loss = criterion(logits, y)
                if (
                    use_phenology
                    and visual_proj is not None
                    and prototypes is not None
                    and lambda_contrast > 0.0
                ):
                    loss = loss + lambda_contrast * phenology_contrastive_loss(
                        visual_proj, y, prototypes, ignore_index=ignore_index
                    )

            if optimizer is not None:
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += float(loss.detach().item())
            n_batches += 1

    return total_loss / max(1, n_batches)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    use_phenology: bool,
) -> dict[str, float]:
    """Evalua el modelo acumulando la matriz de confusion densa del split.

    Acumula la confusion de todo el split (no por-batch) para que mIoU/F1 sean
    exactos a nivel de conjunto. Reusa los helpers de :mod:`ml.eval.metrics`.

    Args:
        model: Segmentador.
        loader: DataLoader del split de validacion.
        device: Device resuelto.
        num_classes: Numero de clases del logit denso (18 o 6).
        ignore_index: Etiqueta ignorada.
        use_phenology: Si ``True`` se hace forward pidiendo la rama visual
            (se descarta para la metrica; solo importan los logits).

    Returns:
        Diccionario ``{"miou", "f1_macro", "pixel_acc"}``.
    """
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True).float()
            logits, _ = _forward_model(model, x, return_visual_proj=use_phenology)
            preds = logits.argmax(dim=1)
            cm += dense_confusion_matrix(
                preds, y, n_classes=num_classes, ignore_index=ignore_index
            )

    cm_f = cm.astype(np.float64)
    iou = _per_class_iou_from_cm(cm)
    miou = 0.0 if np.all(np.isnan(iou)) else float(np.nanmean(iou))

    tp = np.diag(cm_f)
    fp = cm_f.sum(axis=0) - tp
    fn = cm_f.sum(axis=1) - tp
    denom = 2.0 * tp + fp + fn
    present = (cm_f.sum(axis=1) + cm_f.sum(axis=0)) > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0.0, 2.0 * tp / denom, 0.0)
    f1_macro = 0.0 if not np.any(present) else float(f1[present].mean())

    total = int(cm.sum())
    pixel_acc = 0.0 if total == 0 else float(np.trace(cm)) / float(total)

    return {"miou": miou, "f1_macro": f1_macro, "pixel_acc": pixel_acc}


# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------


def train_segmentation(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    *,
    mlflow_run_name: str,
    epochs: int,
    batch_size: int,
    device: str = "cuda",
    lr: float = 1e-3,
    use_phenology: bool = False,
    prototypes: np.ndarray | Sequence[Sequence[float]] | torch.Tensor | None = None,
    lambda_contrast: float = 0.3,
    num_workers: int = 0,
    use_amp: bool = True,
    ignore_index: int = 255,
    num_classes: int | None = None,
    mlflow_uri: str | None = None,
    dice_weight: float = 1.0,
    ce_weight: float = 1.0,
) -> dict[str, float]:
    """Entrena un segmentador denso PASTIS-R con logging MLflow.

    Loop compartido por DeepLabv3+ (2D) y TSViT (temporal + opcional rama
    fenologica-contrastiva). En cada epoch entrena sobre ``train_ds``, evalua
    en ``val_ds`` y loguea ``loss``/``miou``/``f1_macro``/``pixel_acc`` a
    MLflow (tags ``data_version`` + ``code_version`` via
    :func:`ml.utils.mlflow_utils.track_experiment`). Conserva el mejor epoch
    por mIoU de validacion y devuelve sus metricas.

    Args:
        model: Segmentador construido (``build_deeplabv3plus_mobilenet`` o
            ``build_tsvit``). Para la rama contrastiva debe aceptar
            ``return_visual_proj=True`` (TSViT).
        train_ds: Dataset de entrenamiento (``PASTISSegmentationDataset`` en
            modo 2D o temporal segun el modelo).
        val_ds: Dataset de validacion (folds disjuntos del train).
        mlflow_run_name: Nombre del run MLflow
            (``"alt-deeplabv3plus-mobilenet-v1"`` o ``"alt-tsvit-v1"`` /
            ``"alt-tsvit-pheno-v1"``).
        epochs: Numero de epochs.
        batch_size: Tamano de batch del ``DataLoader``.
        device: ``"cuda"``, ``"cpu"`` o ``"auto"``. CUDA con GPU ausente
            degrada a CPU.
        lr: Learning rate del optimizador AdamW.
        use_phenology: Si ``True`` agrega el termino contrastivo
            ``lambda_contrast * L_contrast`` (requiere ``prototypes`` y un
            modelo que exponga la rama visual).
        prototypes: Matriz de prototipos por clase ``(K, S)`` (numpy, lista o
            tensor). Obligatoria si ``use_phenology=True``.
        lambda_contrast: Peso del termino contrastivo en la suma.
        num_workers: Workers del ``DataLoader`` (0 en Windows/CI para evitar
            el coste de spawn).
        use_amp: Si ``True`` usa mixed-precision autocast (solo efectivo en
            CUDA; en CPU es no-op).
        ignore_index: Etiqueta ignorada (Background/Void).
        num_classes: Numero de clases del logit denso. Si es ``None`` se
            infiere de ``train_ds.num_classes`` (o 18 por defecto).
        mlflow_uri: Override del tracking URI MLflow; ``None`` delega en
            :func:`ml.utils.mlflow_utils.resolve_tracking_uri`.
        dice_weight: Peso del termino Dice en la perdida de segmentacion.
        ce_weight: Peso del termino CrossEntropy en la perdida de segmentacion.

    Returns:
        Diccionario ``{"miou", "f1_macro", "pixel_acc"}`` del **mejor epoch**
        de validacion (por mIoU).

    Raises:
        ValueError: si ``use_phenology=True`` pero no se pasan ``prototypes``,
            o si ``epochs`` no es positivo.
    """
    if epochs <= 0:
        raise ValueError(f"epochs debe ser positivo, recibido {epochs}.")
    if use_phenology and prototypes is None:
        raise ValueError(
            "use_phenology=True requiere `prototypes` (matriz (K, S) por clase)."
        )

    resolved_device = _resolve_device(device)
    resolved_classes = int(
        num_classes
        if num_classes is not None
        else getattr(train_ds, "num_classes", 18)
    )

    model = model.to(resolved_device)

    proto_tensor: torch.Tensor | None = None
    if use_phenology and prototypes is not None:
        proto_tensor = (
            prototypes
            if isinstance(prototypes, torch.Tensor)
            else torch.as_tensor(np.asarray(prototypes), dtype=torch.float32)
        ).to(resolved_device)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=resolved_device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=resolved_device.type == "cuda",
    )

    criterion = build_dice_ce_loss(
        ignore_index=ignore_index,
        n_classes=resolved_classes,
        dice_weight=dice_weight,
        ce_weight=ce_weight,
    ).to(resolved_device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    amp_enabled = use_amp and resolved_device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if amp_enabled else None

    logger.info(
        "train_segmentation_start",
        run_name=mlflow_run_name,
        epochs=epochs,
        batch_size=batch_size,
        device=str(resolved_device),
        num_classes=resolved_classes,
        use_phenology=use_phenology,
        lambda_contrast=lambda_contrast if use_phenology else 0.0,
        amp=amp_enabled,
        n_train=len(train_ds),  # type: ignore[arg-type]
        n_val=len(val_ds),  # type: ignore[arg-type]
    )

    best_metrics: dict[str, float] = {"miou": -1.0, "f1_macro": 0.0, "pixel_acc": 0.0}

    with track_experiment(
        _EXPERIMENT_NAME,
        run_name=mlflow_run_name,
        tracking_uri=mlflow_uri,
        dvc_path=_PASTIS_DVC_PATH,
    ):
        import mlflow

        mlflow.log_params(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "device": str(resolved_device),
                "num_classes": resolved_classes,
                "use_phenology": use_phenology,
                "lambda_contrast": lambda_contrast if use_phenology else 0.0,
                "ignore_index": ignore_index,
                "dice_weight": dice_weight,
                "ce_weight": ce_weight,
                "amp": amp_enabled,
                "optimizer": "AdamW",
            }
        )

        for epoch in range(epochs):
            train_loss = _run_epoch(
                model,
                train_loader,
                criterion=criterion,
                device=resolved_device,
                use_phenology=use_phenology,
                prototypes=proto_tensor,
                lambda_contrast=lambda_contrast,
                ignore_index=ignore_index,
                optimizer=optimizer,
                scaler=scaler,
                use_amp=use_amp,
            )
            val_metrics = _evaluate(
                model,
                val_loader,
                device=resolved_device,
                num_classes=resolved_classes,
                ignore_index=ignore_index,
                use_phenology=use_phenology,
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_miou", val_metrics["miou"], step=epoch)
            mlflow.log_metric("val_f1_macro", val_metrics["f1_macro"], step=epoch)
            mlflow.log_metric("val_pixel_acc", val_metrics["pixel_acc"], step=epoch)

            logger.info(
                "train_segmentation_epoch",
                run_name=mlflow_run_name,
                epoch=epoch + 1,
                train_loss=round(train_loss, 4),
                val_miou=round(val_metrics["miou"], 4),
                val_f1_macro=round(val_metrics["f1_macro"], 4),
                val_pixel_acc=round(val_metrics["pixel_acc"], 4),
            )

            if val_metrics["miou"] > best_metrics["miou"]:
                best_metrics = dict(val_metrics)
                best_metrics["best_epoch"] = float(epoch + 1)

        # mIoU inicial -1.0 indica que ningun epoch corrio (no deberia pasar).
        if best_metrics["miou"] < 0.0:
            best_metrics = {"miou": 0.0, "f1_macro": 0.0, "pixel_acc": 0.0}

        mlflow.log_metric("best_val_miou", best_metrics["miou"])
        mlflow.log_metric("best_val_f1_macro", best_metrics["f1_macro"])
        mlflow.log_metric("best_val_pixel_acc", best_metrics["pixel_acc"])

    logger.info(
        "train_segmentation_done",
        run_name=mlflow_run_name,
        best_miou=round(best_metrics["miou"], 4),
        best_f1_macro=round(best_metrics["f1_macro"], 4),
        best_pixel_acc=round(best_metrics["pixel_acc"], 4),
    )
    return best_metrics


# ---------------------------------------------------------------------------
# Orquestacion CLI: construye dataset + modelo + prototipos y entrena.
# El notebook `notebooks/models/5_*` invoca esta interfaz por subprocess para
# que los runs queden documentados en MLflow sin reimplementar logica.
# ---------------------------------------------------------------------------

#: Folds oficiales PASTIS-R para train/val/test (split canonico del benchmark).
_DEFAULT_TRAIN_FOLDS: tuple[int, ...] = (1, 2, 3)
_DEFAULT_VAL_FOLDS: tuple[int, ...] = (4,)

#: Nombres de run MLflow por defecto segun el modelo.
_DEFAULT_RUN_NAMES: dict[str, str] = {
    "deeplabv3plus": "alt-deeplabv3plus-mobilenet-v1",
    "tsvit": "alt-tsvit-v1",
    "tsvit-pheno": "alt-tsvit-pheno-v1",
}


def build_and_train(
    model_kind: str,
    *,
    train_folds: tuple[int, ...] = _DEFAULT_TRAIN_FOLDS,
    val_folds: tuple[int, ...] = _DEFAULT_VAL_FOLDS,
    epochs: int = 30,
    batch_size: int = 4,
    n_timesteps: int = 10,
    target: str = "semantic18",
    device: str = "auto",
    lr: float = 1e-3,
    lambda_contrast: float = 0.3,
    mlflow_run_name: str | None = None,
    mlflow_uri: str | None = None,
) -> dict[str, float]:
    """Construye dataset + modelo + prototipos y lanza el entrenamiento.

    Orquestador de alto nivel para la CLI: segun ``model_kind`` arma el
    ``PASTISSegmentationDataset`` en el modo correcto (2D para DeepLabv3+,
    temporal para TSViT), instancia el modelo, carga los prototipos
    fenologicos si se pide la rama contrastiva, y delega en
    :func:`train_segmentation`.

    Args:
        model_kind: ``"deeplabv3plus"`` (CNN 2D), ``"tsvit"`` (temporal sin
            fenologia) o ``"tsvit-pheno"`` (temporal con rama contrastiva).
        train_folds: Folds PASTIS-R de entrenamiento.
        val_folds: Folds de validacion (disjuntos del train).
        epochs: Numero de epochs.
        batch_size: Tamano de batch.
        n_timesteps: T submuestreado para los modelos temporales.
        target: ``"semantic18"`` (18 clases) o ``"hcat6"`` (6 grupos HCAT).
        device: ``"auto"``, ``"cuda"`` o ``"cpu"``.
        lr: Learning rate AdamW.
        lambda_contrast: Peso del termino contrastivo (solo tsvit-pheno).
        mlflow_run_name: Override del nombre del run; ``None`` usa el default
            por modelo.
        mlflow_uri: Override del tracking URI MLflow.

    Returns:
        Metricas del mejor epoch de validacion ``{miou, f1_macro, pixel_acc}``.

    Raises:
        ValueError: si ``model_kind`` no es reconocido.
    """
    from ml.data.pastis_seg_dataset import PASTISSegmentationDataset
    from ml.models.deeplabv3plus import build_deeplabv3plus_mobilenet
    from ml.models.pheno_semantic_branch import PhenoSemanticBranch
    from ml.models.tsvit_wrapper import build_tsvit

    if model_kind not in _DEFAULT_RUN_NAMES:
        raise ValueError(
            f"model_kind no reconocido: {model_kind!r}. "
            f"Opciones: {sorted(_DEFAULT_RUN_NAMES)}."
        )

    n_classes = 6 if target == "hcat6" else 18
    use_phenology = model_kind == "tsvit-pheno"
    collapse_time = "median" if model_kind == "deeplabv3plus" else None
    run_name = mlflow_run_name or _DEFAULT_RUN_NAMES[model_kind]

    train_ds = PASTISSegmentationDataset(
        folds=train_folds,
        collapse_time=collapse_time,
        n_timesteps=n_timesteps,
        target=target,
    )
    val_ds = PASTISSegmentationDataset(
        folds=val_folds,
        collapse_time=collapse_time,
        n_timesteps=n_timesteps,
        target=target,
    )

    if model_kind == "deeplabv3plus":
        model: nn.Module = build_deeplabv3plus_mobilenet(
            in_channels=10, classes=n_classes
        )
    else:
        model = build_tsvit(
            num_classes=n_classes,
            n_timesteps=n_timesteps,
            img_size=128,
            in_channels=10,
            semantic_dim=384,
        )

    prototypes = None
    if use_phenology:
        branch = PhenoSemanticBranch(semantic_dim=384)
        prototypes = branch.get_class_prototypes().detach()

    logger.info(
        "build_and_train_start",
        model_kind=model_kind,
        run_name=run_name,
        train_folds=train_folds,
        val_folds=val_folds,
        epochs=epochs,
        use_phenology=use_phenology,
        n_classes=n_classes,
    )

    return train_segmentation(
        model,
        train_ds,
        val_ds,
        mlflow_run_name=run_name,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        lr=lr,
        use_phenology=use_phenology,
        prototypes=prototypes,
        lambda_contrast=lambda_contrast,
        num_classes=n_classes,
        mlflow_uri=mlflow_uri,
    )


def _build_arg_parser() -> argparse.ArgumentParser:  # pragma: no cover
    p = argparse.ArgumentParser(
        description=(
            "Entrena un segmentador denso PASTIS-R (DeepLabv3+ o TSViT con/sin "
            "rama fenologica) y registra el run en MLflow."
        )
    )
    p.add_argument(
        "--model",
        required=True,
        choices=sorted(_DEFAULT_RUN_NAMES),
        help="Arquitectura a entrenar.",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-timesteps", type=int, default=10)
    p.add_argument(
        "--target", choices=("semantic18", "hcat6"), default="semantic18"
    )
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-contrast", type=float, default=0.3)
    p.add_argument("--run-name", default=None)
    p.add_argument("--mlflow-uri", default=None)
    p.add_argument(
        "--train-folds",
        default="1,2,3",
        help="Folds de entrenamiento separados por coma.",
    )
    p.add_argument(
        "--val-folds", default="4", help="Folds de validacion separados por coma."
    )
    return p


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    """Punto de entrada CLI. Invocado por el notebook ``5_*`` via subprocess."""
    args = _build_arg_parser().parse_args(argv)
    train_folds = tuple(int(x) for x in args.train_folds.split(","))
    val_folds = tuple(int(x) for x in args.val_folds.split(","))
    metrics = build_and_train(
        args.model,
        train_folds=train_folds,
        val_folds=val_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_timesteps=args.n_timesteps,
        target=args.target,
        device=args.device,
        lr=args.lr,
        lambda_contrast=args.lambda_contrast,
        mlflow_run_name=args.run_name,
        mlflow_uri=args.mlflow_uri,
    )
    logger.info("cli_done", **{k: round(v, 4) for k, v in metrics.items()})
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    raise SystemExit(main(sys.argv[1:]))
