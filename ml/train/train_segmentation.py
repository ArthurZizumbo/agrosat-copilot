"""CLI Typer para entrenar modelos de segmentacion densa PASTIS-R (EPIC 5, Avance 4).

Orquesta el entrenamiento de las arquitecturas a cargo de Aaron en el reparto del
equipo: **#1 U-Net ResNet-50** (composite temporal 2D) y **#6 AnySat frozen +
linear head** (serie temporal). Comparte el pipeline denso
(:mod:`ml.ingest.pastis_dataset`), las metricas pixel-level
(:mod:`ml.eval.dense_metrics`) y el tracking MLflow (:mod:`ml.utils.mlflow_utils`).

La logica de modelado vive en los factories (:mod:`ml.models.segmentation`,
:mod:`ml.models.anysat_wrapper`); este modulo solo orquesta el loop de
entrenamiento, la evaluacion por epoch y la persistencia de artefactos
(separation of concerns, regla CLAUDE.md 8).

Uso (smoke CPU local)::

    poetry run python -m ml.train.train_segmentation \\
        --model unet --subset 4 --epochs 1 --device cpu

Uso (corrida real en Colab L4)::

    poetry run python -m ml.train.train_segmentation \\
        --model unet --epochs 30 --batch-size 8 --device cuda

Operativo permanente (NO viola el anti-patron ``scripts/_*.py``).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import mlflow
import numpy as np
import polars as pl
import structlog
import torch
import typer
from torch import nn
from torch.utils.data import DataLoader

from ml.analysis.hcat_grouping import hcat6_dense_lut
from ml.eval.dense_metrics import DenseConfusionAccumulator
from ml.eval.metrics import dense_confusion_matrix, dense_metrics_from_cm
from ml.ingest.pastis_dataset import (
    PASTIS_IGNORE_INDEX,
    PASTIS_NUM_CLASSES,
    PASTISDataset,
    load_norm_stats,
    pastis_fold_split,
)
from ml.models.deeplabv3plus import build_dice_ce_loss
from ml.utils.mlflow_utils import track_experiment

if TYPE_CHECKING:  # pragma: no cover - type annotations only
    from collections.abc import Sequence

    from torch.utils.data import Dataset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None

logger = structlog.get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)

# MLflow 3.x emits emojis when closing runs; the Windows console uses cp1252 and
# that causes UnicodeEncodeError. Force UTF-8 (no-op on Linux/macOS/Colab).
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")

_EXPERIMENT_NAME = "agrosat-segmentation"
_DEFAULT_OUTPUT = Path("artifacts/segmentation")
_DEFAULT_COMPARISON = Path("reports/segmentation/model_comparison_avance4_aaron.parquet")
_DEFAULT_ROOT = Path("data/PASTIS-R")
#: Path (relative to the repo) of the PASTIS-R dataset to resolve the ``data_version``
#: DVC in the DeepLab/TSViT trainers (us-025).
_PASTIS_DVC_PATH = "data/PASTIS-R"



def _parse_folds(spec: str) -> tuple[int, ...]:
    """Parsea ``"1,2,3"`` a ``(1, 2, 3)``."""
    return tuple(int(x) for x in spec.split(",") if x.strip())


def _build_model(model_name: str, num_classes: int, target_size: int) -> tuple[nn.Module, str]:
    """Construye el modelo y devuelve ``(modelo, temporal_reduction)``.

    Args:
        model_name: ``unet`` o ``anysat``.
        num_classes: Numero de clases de salida.
        target_size: Lado espacial de los logits.

    Returns:
        Tupla ``(nn.Module, temporal_reduction)`` donde la reduccion temporal es
        ``"median"`` para modelos 2D y ``"none"`` para AnySat.

    Raises:
        typer.BadParameter: si ``model_name`` no esta soportado por este CLI.
    """
    if model_name == "unet":
        from ml.models.segmentation import build_unet

        return build_unet(num_classes), "median"
    if model_name == "anysat":
        from ml.models.anysat_wrapper import AnySatSegmenter

        return AnySatSegmenter(num_classes, target_size=target_size), "none"
    raise typer.BadParameter("`--model` debe ser 'unet' o 'anysat'.")


def _forward(model: nn.Module, model_name: str, batch: dict[str, Any], device: torch.device):
    """Ejecuta el forward adaptado a la firma de cada modelo.

    Args:
        model: Modelo a evaluar.
        model_name: ``unet`` (2D) o ``anysat`` (temporal con fechas).
        batch: Batch del DataLoader con ``image`` y opcionalmente ``dates``.
        device: Dispositivo destino.

    Returns:
        Logits ``(B, num_classes, H, W)``.
    """
    image = batch["image"].to(device)
    if model_name == "anysat":
        dates = batch.get("dates")
        dates = dates.to(device) if dates is not None else None
        return model(image, dates)
    return model(image)


def _make_loader(
    patch_ids: list[str],
    *,
    root: Path,
    reduction: str,
    target_size: int,
    norm: tuple,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool = False,
) -> DataLoader:
    """Construye un ``DataLoader`` sobre un :class:`PASTISDataset`.

    Con GPU conviene ``pin_memory`` (acelera la transferencia a la GPU) y, si hay
    varios workers, ``persistent_workers`` para no recrearlos en cada epoca.
    """
    dataset = PASTISDataset(
        patch_ids,
        root=root,
        target_size=target_size,
        temporal_reduction=reduction,  # type: ignore[arg-type]
        norm=norm,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def _evaluate(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    group_lut: np.ndarray | None = None,
) -> dict[str, float]:
    """Evalua el modelo y devuelve mIoU/F1-macro/pixel-accuracy.

    Si se pasa ``group_lut`` (LUT 18 clases -> 6 grupos HCAT), tambien computa las
    mismas tres metricas sobre los 6 grupos agronomicos (sufijo ``_grouped``),
    para comparabilidad con el baseline. El fondo y el void no entran en esas
    metricas; predecir fondo sobre un pixel de cultivo se penaliza con una clase
    extra "no-cultivo" (id 6) que nunca es objetivo, de modo que el macro promedia
    solo los 6 grupos de cultivo.
    """
    model.eval()
    acc = DenseConfusionAccumulator(num_classes, ignore_index=ignore_index, device=str(device))
    acc_grouped = None
    lut_target = lut_pred = None
    if group_lut is not None:
        acc_grouped = DenseConfusionAccumulator(7, ignore_index=255, device=str(device))
        lut_target = torch.as_tensor(group_lut, device=device)
        _pred_lut = group_lut.copy()
        _pred_lut[_pred_lut == 255] = 6  # predicted background/void -> "non-crop" class
        lut_pred = torch.as_tensor(_pred_lut, device=device)
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc="validacion", leave=False, unit="batch")
    with torch.no_grad():
        for batch in iterator:
            logits = _forward(model, model_name, batch, device)
            preds = logits.argmax(dim=1)
            target = batch["semantic"].to(device)
            acc.update(preds, target)
            if acc_grouped is not None:
                acc_grouped.update(lut_pred[preds.clamp(0, 19)], lut_target[target.clamp(0, 19)])
    flat = acc.compute()
    if acc_grouped is None:
        return flat
    grouped = {f"{k}_grouped": v for k, v in acc_grouped.compute().items()}
    return {**flat, **grouped}


def _upsert_comparison_row(row: dict[str, Any], comparison_path: Path) -> None:
    """Inserta/actualiza la fila de metricas del modelo en el parquet comparativo.

    Lee el parquet existente (si hay), elimina cualquier fila previa del mismo
    ``model`` y escribe la version nueva. Este parquet lo consume el notebook
    integrador ``Avance4.Equipo17.ipynb`` para la tabla comparativa de los 6.

    Args:
        row: Fila de metricas del modelo recien entrenado.
        comparison_path: Ruta del parquet comparativo.
    """
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    new = pl.DataFrame([row])
    if comparison_path.exists():
        existing = pl.read_parquet(comparison_path).filter(pl.col("model") != row["model"])
        new = pl.concat([existing, new], how="vertical_relaxed")
    new.write_parquet(comparison_path)


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    best: dict[str, float],
    config: dict[str, Any],
) -> None:
    """Guarda el estado completo para poder reanudar el entrenamiento.

    Persiste modelo, optimizer, scaler, la ultima epoca completada, las mejores
    metricas y la config (para validar que el checkpoint corresponde a la misma
    corrida). Se sobreescribe cada epoca; en Colab conviene apuntarlo a Drive
    para que sobreviva el reinicio de la sesion.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    # Atomic write: first to a .tmp and then rename, to avoid corrupting the
    # checkpoint if the session is cut off right during the save.
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best": best,
            "config": config,
        },
        tmp,
    )
    tmp.replace(path)


def run_training(
    *,
    model: str = "unet",
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    target_size: int = 256,
    train_folds: str = "1,2,3",
    val_folds: str = "4",
    subset: int = 0,
    device: str = "auto",
    num_workers: int = 0,
    root: Path = _DEFAULT_ROOT,
    output_dir: Path = _DEFAULT_OUTPUT,
    comparison_path: Path = _DEFAULT_COMPARISON,
    mlflow_uri: str = "",
    resume: bool = True,
    checkpoint_every: int = 1,
    on_epoch: Callable[[int, dict[str, float]], None] | None = None,
) -> dict[str, Any]:
    """Entrena un modelo de segmentacion densa y registra metricas en MLflow.

    Funcion reutilizable por el CLI (:func:`main`) y por el notebook Colab, de
    modo que ambos ejecuten exactamente la misma logica de entrenamiento.

    Args:
        model: ``unet`` o ``anysat``.
        epochs: Numero de epocas de entrenamiento.
        batch_size: Tamano de batch.
        lr: Learning rate de AdamW.
        weight_decay: Weight decay de AdamW.
        target_size: Resolucion espacial objetivo (256 por convencion del equipo).
        train_folds: Folds oficiales PASTIS asignados a train (ej. ``"1,2,3"``).
        val_folds: Folds asignados a validacion (ej. ``"4"``).
        subset: Limita el numero de patches por split (dev/CI; 0 = todos).
        device: ``cpu``, ``cuda`` o ``auto``.
        num_workers: Workers del DataLoader (0 recomendado en Windows).
        root: Raiz del dataset PASTIS-R.
        output_dir: Directorio destino de los checkpoints ``.pt``.
        comparison_path: Parquet comparativo que consume el notebook integrador.
        mlflow_uri: Override del tracking URI MLflow (vacio = autoresolucion).
        on_epoch: Callback opcional ``(epoch, metrics)`` invocado tras evaluar
            cada epoca. Lo usa el ajuste fino con Optuna para reportar la metrica
            intermedia y podar trials malos (``optuna.TrialPruned``); si lanza,
            la excepcion se propaga y aborta el entrenamiento de ese trial.

    Returns:
        Diccionario con ``model``, ``miou``, ``f1_macro``, ``pixel_accuracy``,
        ``train_time_s`` y ``checkpoint_path``.

    Raises:
        FileNotFoundError: si la raiz PASTIS-R no existe.
        RuntimeError: si el split de train/val queda vacio.
    """
    if not root.exists():
        raise FileNotFoundError(f"Raiz PASTIS-R no encontrada: {root}")

    dev = _resolve_device(device)
    tr_folds = _parse_folds(train_folds)
    va_folds = _parse_folds(val_folds)
    split = pastis_fold_split(root, train_folds=tr_folds, val_folds=va_folds, test_folds=())
    train_ids, val_ids = split["train"], split["val"]
    if subset > 0:
        train_ids, val_ids = train_ids[:subset], val_ids[: max(1, subset // 2)]
    if not train_ids or not val_ids:
        raise RuntimeError(
            f"Split PASTIS vacio (n_train={len(train_ids)}, n_val={len(val_ids)}). "
            "Verifica los folds y que metadata.geojson tenga el campo Fold."
        )

    seg_model, reduction = _build_model(model, PASTIS_NUM_CLASSES, target_size)
    seg_model = seg_model.to(dev)
    # Normalization with stats from the train folds (no leakage from the val fold).
    norm = load_norm_stats(root, folds=tr_folds)

    pin = dev.type == "cuda"
    train_loader = _make_loader(
        train_ids, root=root, reduction=reduction, target_size=target_size, norm=norm,
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin,
    )
    val_loader = _make_loader(
        val_ids, root=root, reduction=reduction, target_size=target_size, norm=norm,
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin,
    )

    # Materialize Lazy parameters (the AnySat Conv1x1 head uses nn.LazyConv2d,
    # which infers its channels on the first forward) with a real batch BEFORE
    # building the optimizer; otherwise the param count and AdamW fail
    # over UninitializedParameter.
    seg_model.train()
    with torch.no_grad():
        _forward(seg_model, model, next(iter(train_loader)), dev)

    trainable = [p for p in seg_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=PASTIS_IGNORE_INDEX)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Resume: if there is a checkpoint from the same run, continue from the
    # next epoch instead of starting from scratch (key in Colab, ephemeral session).
    resume_ckpt_path = output_dir / f"{model}_ckpt.pt"
    final_model_path = output_dir / f"{model}_pastis.pt"
    ckpt_config = {"model": model, "target_size": target_size, "epochs": epochs}
    start_epoch = 0
    best: dict[str, float] = {"miou": 0.0, "f1_macro": 0.0, "pixel_accuracy": 0.0}
    if resume and resume_ckpt_path.exists():
        try:
            ckpt = torch.load(resume_ckpt_path, map_location=dev)
            if ckpt.get("config") == ckpt_config:
                seg_model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                scaler.load_state_dict(ckpt["scaler_state_dict"])
                best = ckpt["best"]
                start_epoch = ckpt["epoch"] + 1
                logger.info("segmentation_resume", model=model, start_epoch=start_epoch, **best)
            else:
                logger.warning("segmentation_ckpt_config_mismatch", path=str(resume_ckpt_path))
        except Exception as exc:  # noqa: BLE001 - corrupt checkpoint: start from scratch
            logger.warning(
                "segmentation_ckpt_load_failed", path=str(resume_ckpt_path), error=str(exc)
            )

    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in seg_model.parameters())
    logger.info(
        "segmentation_train_start",
        model=model,
        device=str(dev),
        n_train=len(train_ids),
        n_val=len(val_ids),
        epochs=epochs,
        n_trainable=n_trainable,
        n_total=n_total,
    )

    run_name = f"seg-{model}-pastis-v1"
    tracking_override = mlflow_uri or None
    start = time.perf_counter()

    with track_experiment(
        _EXPERIMENT_NAME, run_name=run_name, tracking_uri=tracking_override, dvc_path=str(root)
    ):
        mlflow.set_tag("architecture", model)
        mlflow.log_params(
            {
                "model": model,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "target_size": target_size,
                "train_folds": train_folds,
                "val_folds": val_folds,
                "n_train": len(train_ids),
                "n_val": len(val_ids),
                "n_trainable_params": n_trainable,
                "n_total_params": n_total,
                "device": str(dev),
            }
        )

        # LUT 18 classes -> 6 HCAT groups to also report the grouped metric
        # (comparable with the baseline). See ml.analysis.hcat_grouping.
        group_lut = hcat6_dense_lut()
        # Per-epoch history for the loss/mIoU curves. It is persisted to Drive
        # alongside the comparison parquet and resumed if it already exists (survives cutoffs).
        history_path = comparison_path.with_name(
            comparison_path.name.replace("model_comparison_avance4", "history")
        )
        history: list[dict[str, float]] = []
        if resume and start_epoch > 0 and history_path.exists():
            history = pl.read_parquet(history_path).to_dicts()
        for epoch in range(start_epoch, epochs):
            seg_model.train()
            if model == "anysat":
                # The frozen encoder stays in eval; only the head trains.
                seg_model.encoder.eval()
            epoch_loss = 0.0
            # Per-batch progress bar within the epoch (progress, it/s, loss).
            bar = train_loader
            if tqdm is not None:
                bar = tqdm(
                    train_loader,
                    desc=f"epoca {epoch + 1}/{epochs}",
                    leave=False,
                    unit="batch",
                )
            for step, batch in enumerate(bar, 1):
                target = batch["semantic"].to(dev)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = _forward(seg_model, model, batch, dev)
                    loss = criterion(logits, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += float(loss.detach())
                if tqdm is not None:
                    bar.set_postfix(loss=f"{epoch_loss / step:.3f}")

            metrics = _evaluate(
                seg_model, model, val_loader, dev, PASTIS_NUM_CLASSES, PASTIS_IGNORE_INDEX,
                group_lut=group_lut,
            )
            train_loss = epoch_loss / max(1, len(train_loader))
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            for key, value in metrics.items():
                mlflow.log_metric(f"val_{key}", value, step=epoch)
            if metrics["miou"] >= best["miou"]:
                best = metrics
            logger.info("segmentation_epoch", epoch=epoch, loss=epoch_loss, **metrics)
            # Per-epoch hook (Optuna fine-tuning: reports intermediate metric and prunes).
            # If it raises (TrialPruned), the exception propagates and cuts off this training.
            if on_epoch is not None:
                on_epoch(epoch, metrics)
            # Per-epoch history logging (for the curves) + persistence to Drive.
            history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
            comparison_path.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(history).write_parquet(history_path)
            # Resumable checkpoint every `checkpoint_every` epochs (and on the last one).
            if (epoch + 1) % checkpoint_every == 0 or epoch == epochs - 1:
                _save_checkpoint(
                    resume_ckpt_path, epoch=epoch, model=seg_model, optimizer=optimizer,
                    scaler=scaler, best=best, config=ckpt_config,
                )

        train_time_s = time.perf_counter() - start
        mlflow.log_metric("train_time_s", train_time_s)

        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(seg_model.state_dict(), final_model_path)
        mlflow.log_artifact(str(final_model_path))

        comparison_row = {
            "model": model,
            "miou": best["miou"],
            "f1_macro": best["f1_macro"],
            "pixel_accuracy": best["pixel_accuracy"],
            "miou_grouped": best.get("miou_grouped"),
            "f1_macro_grouped": best.get("f1_macro_grouped"),
            "pixel_accuracy_grouped": best.get("pixel_accuracy_grouped"),
            "train_time_s": train_time_s,
            "epochs": epochs,
            "n_train": len(train_ids),
            "n_val": len(val_ids),
            "n_trainable_params": n_trainable,
            "target_size": target_size,
            "device": str(dev),
        }
        _upsert_comparison_row(comparison_row, comparison_path)

    logger.info("segmentation_train_done", model=model, **best)
    return {
        "model": model,
        "miou": best["miou"],
        "f1_macro": best["f1_macro"],
        "pixel_accuracy": best["pixel_accuracy"],
        "miou_grouped": best.get("miou_grouped"),
        "f1_macro_grouped": best.get("f1_macro_grouped"),
        "pixel_accuracy_grouped": best.get("pixel_accuracy_grouped"),
        "train_time_s": train_time_s,
        "checkpoint_path": str(final_model_path),
    }



# ===========================================================================
# US-025 trainers: DeepLabv3+ (2D) and TSViT (temporal, + phenology branch).
# Own APIs (train_segmentation / build_and_train) that the 5a/5b notebooks
# invoke by subprocess with --model deeplabv3plus|tsvit|tsvit-pheno.
# ===========================================================================

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
        # No valid pixels: neutral term that preserves the grad graph.
        return visual_proj.sum() * 0.0

    v_valid = v_flat[valid]
    y_valid = y_flat[valid]

    # Deterministic subsampling to bound the similarity matrix in VRAM.
    n_valid = v_valid.shape[0]
    if n_valid > max_pixels:
        gen = torch.Generator(device="cpu").manual_seed(0)
        perm = torch.randperm(n_valid, generator=gen)[:max_pixels].to(device)
        v_valid = v_valid[perm]
        y_valid = y_valid[perm]

    v_valid = nn.functional.normalize(v_valid, dim=1)  # (P, S)

    # Pixel x prototype similarity -> logits (P, K).
    logits = (v_valid @ protos.t()) / temperature

    # Direction 1 (visual): each pixel must classify to its class prototype.
    loss_v = nn.functional.cross_entropy(logits, y_valid)

    # Direction 2 (semantic): for each present class, the prototype must
    # recover its pixels. The prototype->pixels similarity of its class is
    # averaged against all pixels in the batch (symmetric InfoNCE from the paper).
    present = torch.unique(y_valid)
    proto_logits = (protos[present] @ v_valid.t()) / temperature  # (Kp, P)
    # Multi-positive target: for each present prototype, the pixels of its
    # class are the positives; the mean of log-softmax over positives is used.
    log_prob = nn.functional.log_softmax(proto_logits, dim=1)  # (Kp, P)
    pos_mask = (y_valid.unsqueeze(0) == present.unsqueeze(1)).to(log_prob.dtype)
    pos_counts = pos_mask.sum(dim=1).clamp_min(1.0)
    loss_s = -(log_prob * pos_mask).sum(dim=1) / pos_counts
    loss_s = loss_s.mean()

    loss: torch.Tensor = 0.5 * (loss_v + loss_s)
    return loss


# ---------------------------------------------------------------------------
# Device / forward helpers
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
        # The model did not honor the flag (defensive case): only logits.
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
                # Gradient clipping (max_norm=1.0): essential for TSViT
                # (transformer) — without it, the gradients explode and the loss
                # diverges to NaN after ~8 epochs. With AMP you must `unscale_`
                # before clipping. DeepLabv3+ (CNN) tolerates not clipping, but
                # applying it to both is safe and stabilizes.
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += float(loss.detach().item())
            n_batches += 1

    return total_loss / max(1, n_batches)


def _evaluate_dense(
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
        Tupla ``(metrics, cm)``: el diccionario completo de metricas
        (``miou``, ``f1_macro``, ``pixel_acc``, ``balanced_acc``,
        ``cohen_kappa``, ``per_class_iou``, ``per_class_f1``) y la matriz de
        confusion densa acumulada del split (para artefactos al final).
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

    return dense_metrics_from_cm(cm), cm


# ---------------------------------------------------------------------------
# Per-epoch checkpointing (resume after interruption).
# ---------------------------------------------------------------------------


def _save_checkpoint_seg(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    best_metrics: dict[str, float],
) -> None:
    """Persiste el estado completo de entrenamiento para reanudar.

    Guarda ``model``/``optimizer``/``scaler`` state_dicts + el ``epoch`` ya
    completado + las mejores metricas, de forma atomica (escribe a ``.tmp`` y
    renombra) para no corromper el checkpoint si el proceso muere a mitad de
    la escritura.

    Args:
        path: Ruta destino del checkpoint (``.pt``).
        epoch: Indice del ultimo epoch COMPLETADO (0-based).
        model: Modelo cuyo state_dict se guarda.
        optimizer: Optimizador AdamW.
        scaler: GradScaler AMP (o ``None`` si no se usa AMP).
        best_metrics: Mejores metricas de validacion hasta ahora.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_metrics": best_metrics,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_checkpoint_seg(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
) -> tuple[int, dict[str, float]]:
    """Carga un checkpoint y restaura el estado de entrenamiento.

    Args:
        path: Ruta del checkpoint ``.pt``.
        model: Modelo a restaurar (in-place).
        optimizer: Optimizador a restaurar (in-place).
        scaler: GradScaler a restaurar (in-place) o ``None``.
        device: Dispositivo destino para mapear los tensores.

    Returns:
        ``(start_epoch, best_metrics)``: el epoch desde el que continuar
        (= ultimo completado + 1) y las mejores metricas previas.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = int(ckpt["epoch"]) + 1
    best = ckpt.get("best_metrics") or {
        "miou": -1.0,
        "f1_macro": 0.0,
        "pixel_acc": 0.0,
    }
    logger.info(
        "checkpoint_resumed",
        path=str(path),
        start_epoch=start_epoch,
        best_miou=round(best.get("miou", -1.0), 4),
    )
    return start_epoch, best


def _log_final_artifacts(
    ckpt_dir: Path,
    *,
    best_cm: np.ndarray,
    best_metrics: dict[str, float],
    num_classes: int,
) -> None:
    """Genera y loguea a MLflow los artefactos finales del mejor epoch.

    Produce dos artefactos del mejor modelo en validacion:
    1. ``confusion_matrix.png``: matriz de confusion normalizada (recall),
       util para ver que clases/grupos confunde el modelo.
    2. ``per_class_metrics.json``: IoU y F1 por clase + las metricas macro.

    Args:
        ckpt_dir: Directorio donde escribir los artefactos antes de subirlos.
        best_cm: Matriz de confusion densa del mejor epoch.
        best_metrics: Diccionario de metricas del mejor epoch (incluye
            ``per_class_iou`` y ``per_class_f1``).
        num_classes: Numero de clases (18 o 6).
    """
    import json

    import matplotlib.pyplot as plt
    import mlflow

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix (figure). Reconstructing y_true/y_pred from the cm by
    # expanding counts would be costly; instead we draw the cm directly.
    cm_f = best_cm.astype(np.float64)
    row_sums = cm_f.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(row_sums > 0.0, cm_f / row_sums, 0.0)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Etiqueta real")
    ax.set_title(f"Matriz de confusion normalizada ({num_classes} clases)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cm_path = ckpt_dir / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    metrics_path = ckpt_dir / "per_class_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "num_classes": num_classes,
                "best_epoch": best_metrics.get("best_epoch"),
                "miou": best_metrics.get("miou"),
                "f1_macro": best_metrics.get("f1_macro"),
                "pixel_acc": best_metrics.get("pixel_acc"),
                "balanced_acc": best_metrics.get("balanced_acc"),
                "cohen_kappa": best_metrics.get("cohen_kappa"),
                "per_class_iou": best_metrics.get("per_class_iou"),
                "per_class_f1": best_metrics.get("per_class_f1"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    mlflow.log_artifact(str(cm_path), artifact_path="eval")
    mlflow.log_artifact(str(metrics_path), artifact_path="eval")
    logger.info("final_artifacts_logged", cm=str(cm_path), metrics=str(metrics_path))


# ---------------------------------------------------------------------------
# Public API
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
    ckpt_dir: str | Path | None = None,
    resume: bool = True,
    warmup_epochs: int = 10,
    lr_min: float = 5e-6,
    patience: int = 0,
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

    # `persistent_workers` avoids re-spawning the workers on each epoch (high
    # cost on Windows with spawn); `prefetch_factor` preloads several batches per
    # worker to overlap the temporal collapse (np.median ~79ms/patch) with the
    # GPU step. They only apply with num_workers > 0.
    loader_kwargs: dict[str, object] = {
        "pin_memory": resolved_device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        **loader_kwargs,
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

    # LR schedule from Tarasiou et al. 2023 (TSViT, §4.1 "Implementation
    # details"): linear warmup 0 -> lr up to `warmup_epochs`, then cosine
    # decay to `lr_min`. The warmup is what stabilizes the transformer (without it,
    # the high LR from step 0 makes the loss diverge to NaN ~epoch 8). Applied
    # per epoch; for DeepLabv3+ (CNN) it also helps but is not critical.
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=max(1, warmup_epochs),
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs - warmup_epochs),
                eta_min=lr_min,
            ),
        ],
        milestones=[max(1, warmup_epochs)],
    )

    # Per-epoch checkpoints: `last.pt` (always) + `best.pt` (best mIoU).
    # They allow resuming after interruption (the L4 VM shut down once).
    resolved_ckpt_dir = (
        Path(ckpt_dir)
        if ckpt_dir is not None
        else Path("checkpoints/segmentation") / mlflow_run_name
    )
    last_ckpt = resolved_ckpt_dir / "last.pt"
    best_ckpt = resolved_ckpt_dir / "best.pt"
    start_epoch = 0
    best_metrics: dict[str, float] = {"miou": -1.0, "f1_macro": 0.0, "pixel_acc": 0.0}
    if resume and last_ckpt.exists():
        start_epoch, best_metrics = _load_checkpoint_seg(
            last_ckpt,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            device=resolved_device,
        )

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
        start_epoch=start_epoch,
        ckpt_dir=str(resolved_ckpt_dir),
        patience=patience,
    )

    # Early stopping state and cm of the best epoch (for final artifacts).
    epochs_no_improve = 0
    best_cm = np.zeros((resolved_classes, resolved_classes), dtype=np.int64)

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

        for epoch in range(start_epoch, epochs):
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
            val_metrics, val_cm = _evaluate_dense(
                model,
                val_loader,
                device=resolved_device,
                num_classes=resolved_classes,
                ignore_index=ignore_index,
                use_phenology=use_phenology,
            )

            current_lr = optimizer.param_groups[0]["lr"]
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_miou", val_metrics["miou"], step=epoch)
            mlflow.log_metric("val_f1_macro", val_metrics["f1_macro"], step=epoch)
            mlflow.log_metric("val_pixel_acc", val_metrics["pixel_acc"], step=epoch)
            mlflow.log_metric("val_balanced_acc", val_metrics["balanced_acc"], step=epoch)
            mlflow.log_metric("val_cohen_kappa", val_metrics["cohen_kappa"], step=epoch)
            mlflow.log_metric("lr", current_lr, step=epoch)

            logger.info(
                "train_segmentation_epoch",
                run_name=mlflow_run_name,
                epoch=epoch + 1,
                lr=round(current_lr, 6),
                train_loss=round(train_loss, 4),
                val_miou=round(val_metrics["miou"], 4),
                val_f1_macro=round(val_metrics["f1_macro"], 4),
                val_pixel_acc=round(val_metrics["pixel_acc"], 4),
                val_balanced_acc=round(val_metrics["balanced_acc"], 4),
                val_cohen_kappa=round(val_metrics["cohen_kappa"], 4),
            )

            is_best = val_metrics["miou"] > best_metrics["miou"]
            if is_best:
                best_metrics = dict(val_metrics)
                best_metrics["best_epoch"] = float(epoch + 1)
                best_cm = val_cm.copy()  # cm of the best epoch (for artifacts)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Per-epoch checkpoint: `last.pt` always (for resume), `best.pt`
            # when the validation mIoU improves (for later inference).
            _save_checkpoint_seg(
                last_ckpt,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                best_metrics=best_metrics,
            )
            if is_best:
                _save_checkpoint_seg(
                    best_ckpt,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    scheduler=scheduler,
                    best_metrics=best_metrics,
                )

            # Advance the LR schedule (warmup -> cosine) at the end of each epoch.
            scheduler.step()

            # Early stopping: cuts off if val_miou does not improve in `patience` epochs
            # (DeepLabv3+ tends to overfit after ~7 epochs). 0 = disabled.
            if patience > 0 and epochs_no_improve >= patience:
                logger.info(
                    "early_stopping",
                    run_name=mlflow_run_name,
                    epoch=epoch + 1,
                    best_epoch=int(best_metrics.get("best_epoch", 0)),
                    patience=patience,
                )
                break

        # Initial mIoU -1.0 indicates no epoch ran (should not happen).
        if best_metrics["miou"] < 0.0:
            best_metrics = {"miou": 0.0, "f1_macro": 0.0, "pixel_acc": 0.0}

        mlflow.log_metric("best_val_miou", best_metrics["miou"])
        mlflow.log_metric("best_val_f1_macro", best_metrics["f1_macro"])
        mlflow.log_metric("best_val_pixel_acc", best_metrics["pixel_acc"])
        mlflow.log_metric("best_val_balanced_acc", best_metrics.get("balanced_acc", 0.0))
        mlflow.log_metric("best_val_cohen_kappa", best_metrics.get("cohen_kappa", 0.0))

        # Best epoch artifacts: confusion matrix (PNG figure) +
        # per-class metrics (JSON). For the notebook and the analysis.
        _log_final_artifacts(
            resolved_ckpt_dir,
            best_cm=best_cm,
            best_metrics=best_metrics,
            num_classes=resolved_classes,
        )

        # Upload the best checkpoint to MLflow as an artifact (for reproducible
        # inference from the run, not only from the local disk).
        if best_ckpt.exists():
            mlflow.log_artifact(str(best_ckpt), artifact_path="checkpoint")

    logger.info(
        "train_segmentation_done",
        run_name=mlflow_run_name,
        best_miou=round(best_metrics["miou"], 4),
        best_f1_macro=round(best_metrics["f1_macro"], 4),
        best_pixel_acc=round(best_metrics["pixel_acc"], 4),
    )
    return best_metrics


# ---------------------------------------------------------------------------
# CLI orchestration: builds dataset + model + prototypes and trains.
# The notebook `notebooks/models/5_*` invokes this interface by subprocess so
# that the runs are documented in MLflow without reimplementing logic.
# ---------------------------------------------------------------------------

#: Official PASTIS-R folds for train/val/test (canonical benchmark split).
_DEFAULT_TRAIN_FOLDS: tuple[int, ...] = (1, 2, 3)
_DEFAULT_VAL_FOLDS: tuple[int, ...] = (4,)

#: Default MLflow run names according to the model.
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
    num_workers: int = 0,
    ckpt_dir: str | Path | None = None,
    resume: bool = True,
    patience: int = 0,
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
        num_workers=num_workers,
        ckpt_dir=ckpt_dir,
        resume=resume,
        patience=patience,
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
    p.add_argument(
        "--ckpt-dir",
        default=None,
        help=(
            "Directorio de checkpoints. Default checkpoints/segmentation/"
            "<run-name>. Guarda last.pt (resume) + best.pt (inferencia) por epoch."
        ),
    )
    p.add_argument(
        "--patience",
        type=int,
        default=0,
        help=(
            "Early stopping: corta si val_miou no mejora en N epochs. "
            "0 = desactivado. DeepLabv3+ sobreajusta tras ~7 epochs."
        ),
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignora last.pt y entrena desde cero (por defecto reanuda).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Workers del DataLoader. El colapso temporal (np.median ~79ms/patch) "
            "es CPU-bound; subir esto satura la GPU. Optimo ~3/4 de los cores "
            "fisicos (ej. 12 en un CPU de 16 cores). 0 = serial (CI/debug)."
        ),
    )
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

def main_legacy(argv: list[str] | None = None) -> int:  # pragma: no cover
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
        num_workers=args.num_workers,
        ckpt_dir=args.ckpt_dir,
        resume=not args.no_resume,
        patience=args.patience,
        mlflow_run_name=args.run_name,
        mlflow_uri=args.mlflow_uri,
    )
    logger.info("cli_done", **{k: round(v, 4) for k, v in metrics.items()})
    return 0
@app.command()
def main(
    model: Annotated[str, typer.Option(help="Modelo: 'unet' (#1) o 'anysat' (#6).")] = "unet",
    epochs: Annotated[int, typer.Option(help="Numero de epocas.")] = 30,
    batch_size: Annotated[int, typer.Option(help="Tamano de batch.")] = 8,
    lr: Annotated[float, typer.Option(help="Learning rate AdamW.")] = 1e-4,
    weight_decay: Annotated[float, typer.Option(help="Weight decay AdamW.")] = 1e-4,
    target_size: Annotated[int, typer.Option(help="Resolucion espacial objetivo.")] = 256,
    train_folds: Annotated[str, typer.Option(help="Folds de train (coma).")] = "1,2,3",
    val_folds: Annotated[str, typer.Option(help="Folds de validacion (coma).")] = "4",
    subset: Annotated[int, typer.Option(help="Limita patches por split (0 = todos).")] = 0,
    device: Annotated[str, typer.Option(help="cpu, cuda o auto.")] = "auto",
    num_workers: Annotated[int, typer.Option(help="Workers del DataLoader.")] = 0,
    root: Annotated[Path, typer.Option(help="Raiz PASTIS-R.")] = _DEFAULT_ROOT,
    output_dir: Annotated[Path, typer.Option(help="Destino de checkpoints.")] = _DEFAULT_OUTPUT,
    comparison_path: Annotated[
        Path, typer.Option(help="Parquet comparativo (lo consume el integrador).")
    ] = _DEFAULT_COMPARISON,
    mlflow_uri: Annotated[str, typer.Option(help="Tracking URI MLflow (vacio = auto).")] = "",
    resume: Annotated[
        bool, typer.Option("--resume/--no-resume", help="Reanudar desde checkpoint si existe.")
    ] = True,
    checkpoint_every: Annotated[
        int, typer.Option(help="Guardar checkpoint cada N epocas.")
    ] = 1,
) -> None:
    """Wrapper CLI de :func:`run_training` (ver su docstring para los argumentos)."""
    try:
        result = run_training(
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            target_size=target_size,
            train_folds=train_folds,
            val_folds=val_folds,
            subset=subset,
            device=device,
            num_workers=num_workers,
            root=root,
            output_dir=output_dir,
            comparison_path=comparison_path,
            mlflow_uri=mlflow_uri,
            resume=resume,
            checkpoint_every=checkpoint_every,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        logger.warning("segmentation_train_skipped", reason=str(exc))
        raise typer.Exit(code=0) from exc

    typer.echo(
        f"[{result['model']}] mIoU={result['miou']:.4f} "
        f"F1-macro={result['f1_macro']:.4f} pixacc={result['pixel_accuracy']:.4f} "
        f"({result['train_time_s']:.1f}s) -> {result['checkpoint_path']}"
    )


if __name__ == "__main__":  # pragma: no cover - CLI dispatcher
    # Two CLIs coexist in this module: the UNet/AnySat Typer (Aaron) and the
    # DeepLab/TSViT argparse (us-025). Routing is by the --model value so
    # that `python -m ml.train.train_segmentation --model X` works for both.
    _US025_MODELS = {"deeplabv3plus", "tsvit", "tsvit-pheno"}
    _argv = sys.argv[1:]
    _model = None
    for _i, _a in enumerate(_argv):
        if _a == "--model" and _i + 1 < len(_argv):
            _model = _argv[_i + 1]
            break
        if _a.startswith("--model="):
            _model = _a.split("=", 1)[1]
            break
    if _model in _US025_MODELS:
        sys.exit(main_legacy(_argv))
    sys.exit(app())
