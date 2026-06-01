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

import sys
import time
from pathlib import Path
from typing import Annotated, Any

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
from ml.ingest.pastis_dataset import (
    PASTIS_IGNORE_INDEX,
    PASTIS_NUM_CLASSES,
    PASTISDataset,
    load_norm_stats,
    pastis_fold_split,
)
from ml.utils.mlflow_utils import track_experiment

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm es opcional
    tqdm = None

logger = structlog.get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)

# MLflow 3.x emite emojis al cerrar runs; la consola Windows usa cp1252 y eso
# provoca UnicodeEncodeError. Forzamos UTF-8 (no-op en Linux/macOS/Colab).
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")

_EXPERIMENT_NAME = "agrosat-segmentation"
_DEFAULT_OUTPUT = Path("artifacts/segmentation")
_DEFAULT_COMPARISON = Path("reports/segmentation/model_comparison_avance4_aaron.parquet")
_DEFAULT_ROOT = Path("data/PASTIS-R")


def _resolve_device(device: str) -> torch.device:
    """Resuelve el dispositivo (``auto`` -> cuda si disponible, sino cpu)."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


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
        _pred_lut[_pred_lut == 255] = 6  # fondo/void predichos -> clase "no-cultivo"
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
    # Escritura atomica: primero a un .tmp y luego rename, para no corromper el
    # checkpoint si la sesion se corta justo durante el guardado.
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
    # Normalizacion con stats de los folds de train (sin leakage del fold de val).
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

    # Materializa parametros Lazy (la cabeza Conv1x1 de AnySat usa nn.LazyConv2d,
    # que infiere sus canales en el primer forward) con un batch real ANTES de
    # construir el optimizer; de lo contrario el conteo de params y AdamW fallan
    # sobre UninitializedParameter.
    seg_model.train()
    with torch.no_grad():
        _forward(seg_model, model, next(iter(train_loader)), dev)

    trainable = [p for p in seg_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=PASTIS_IGNORE_INDEX)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Reanudacion: si hay un checkpoint de la misma corrida, continua desde la
    # epoca siguiente en vez de empezar de cero (clave en Colab, sesion efimera).
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
        except Exception as exc:  # noqa: BLE001 - checkpoint corrupto: arrancar de cero
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

        # LUT 18 clases -> 6 grupos HCAT para reportar tambien la metrica agrupada
        # (comparable con el baseline). Ver ml.analysis.hcat_grouping.
        group_lut = hcat6_dense_lut()
        # Historial por epoca para las curvas de loss/mIoU. Se persiste a Drive
        # junto al parquet comparativo y se reanuda si ya existe (sobrevive cortes).
        history_path = comparison_path.with_name(
            comparison_path.name.replace("model_comparison_avance4", "history")
        )
        history: list[dict[str, float]] = []
        if resume and start_epoch > 0 and history_path.exists():
            history = pl.read_parquet(history_path).to_dicts()
        for epoch in range(start_epoch, epochs):
            seg_model.train()
            if model == "anysat":
                # El encoder congelado permanece en eval; solo la head entrena.
                seg_model.encoder.eval()
            epoch_loss = 0.0
            # Barra de progreso por batch dentro de la epoca (avance, it/s, loss).
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
            # Registro del historial por epoca (para las curvas) + persistencia a Drive.
            history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
            comparison_path.parent.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(history).write_parquet(history_path)
            # Checkpoint reanudable cada `checkpoint_every` epocas (y en la ultima).
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


if __name__ == "__main__":  # pragma: no cover - punto de entrada CLI
    sys.exit(app())
