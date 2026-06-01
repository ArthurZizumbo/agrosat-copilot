"""Inferencia y visualizacion de predicciones de segmentacion densa.

Carga un checkpoint entrenado (``best.pt`` de
:mod:`ml.train.train_segmentation`), predice sobre patches PASTIS-R y genera
la figura comparativa ``Input (RGB) | Ground truth | Prediction`` por patch,
mas las metricas del modelo cargado. Es el modulo que el notebook
``notebooks/models/5*`` invoca para el analisis visual; el notebook llama, no
implementa la logica.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import structlog
import torch

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = structlog.get_logger(__name__)

#: Bandas RGB en los .npy PASTIS-R (orden S2 de 10: B2,B3,B4,...): B4(rojo)=2,
#: B3(verde)=1, B2(azul)=0.
_RGB_BANDS = (2, 1, 0)


def load_segmentation_model(
    checkpoint_path: Path | str,
    *,
    model_kind: Literal["deeplabv3plus", "tsvit", "tsvit-pheno"],
    num_classes: int = 18,
    n_timesteps: int = 10,
    device: str = "auto",
) -> torch.nn.Module:
    """Reconstruye el modelo y carga los pesos del checkpoint.

    Args:
        checkpoint_path: Ruta a ``best.pt`` (state completo del entrenamiento).
        model_kind: Arquitectura para reconstruir la topologia exacta.
        num_classes: Numero de clases del head (18 semantico o 6 HCAT).
        n_timesteps: T submuestreado (solo modelos temporales).
        device: ``"auto"``, ``"cuda"`` o ``"cpu"``.

    Returns:
        Modelo en modo ``eval()`` con los pesos del mejor epoch cargados.
    """
    from ml.models.deeplabv3plus import build_deeplabv3plus_mobilenet
    from ml.models.tsvit_wrapper import build_tsvit

    resolved_device = torch.device(
        "cuda" if (device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    )
    if model_kind == "deeplabv3plus":
        model: torch.nn.Module = build_deeplabv3plus_mobilenet(
            in_channels=10, classes=num_classes
        )
    else:
        model = build_tsvit(
            num_classes=num_classes,
            n_timesteps=n_timesteps,
            img_size=128,
            in_channels=10,
            semantic_dim=384,
        )
    ckpt = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(resolved_device).eval()
    logger.info(
        "segmentation_model_loaded",
        checkpoint=str(checkpoint_path),
        model_kind=model_kind,
        num_classes=num_classes,
        best_epoch=ckpt.get("best_metrics", {}).get("best_epoch"),
        device=str(resolved_device),
    )
    return model


@torch.no_grad()
def predict_patch(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    model_kind: str,
    doy: torch.Tensor | None = None,
) -> np.ndarray:
    """Predice la mascara densa de un patch.

    Args:
        model: Modelo cargado en ``eval()``.
        x: Tensor del patch: ``(10, H, W)`` (2D) o ``(T, 10, H, W)``
            (temporal). Se le anade la dimension batch internamente.
        model_kind: Arquitectura (decide si pasa ``doy``).
        doy: Vector de DOY ``(T,)`` para los modelos temporales.

    Returns:
        Mascara ``(H, W)`` int con la clase predicha por pixel (en el espacio
        ``[0..num_classes-1]``).
    """
    device = next(model.parameters()).device
    xb = x.unsqueeze(0).to(device).float()
    if model_kind == "deeplabv3plus":
        logits = model(xb)
    else:
        doy_b = doy.unsqueeze(0).to(device) if doy is not None else None
        out = model(xb, doy=doy_b)
        logits = out[0] if isinstance(out, tuple) else out
    return logits.argmax(dim=1).squeeze(0).cpu().numpy()


def prediction_figure(
    rgb: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int = 18,
    ignore_index: int = 255,
    titles: tuple[str, str, str] = ("Input (RGB)", "Ground truth", "Prediction"),
) -> Figure:
    """Construye la figura comparativa RGB | ground truth | prediccion.

    Args:
        rgb: Imagen RGB ``(H, W, 3)`` en ``[0, 1]``.
        y_true: Mascara real ``(H, W)`` (clases ``[0..C-1]`` + ``ignore_index``).
        y_pred: Mascara predicha ``(H, W)``.
        num_classes: Numero de clases para el colormap discreto.
        ignore_index: Valor ignorado en ``y_true`` (se pinta neutro).
        titles: Titulos de los tres paneles.

    Returns:
        Figura matplotlib 1x3 lista para ``display(fig)`` en el notebook.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors

    cmap = plt.get_cmap("tab20", num_classes)
    norm = colors.Normalize(vmin=0, vmax=num_classes - 1)

    yt = np.where(y_true == ignore_index, np.nan, y_true.astype(float))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(np.clip(rgb, 0.0, 1.0))
    axes[1].imshow(yt, cmap=cmap, norm=norm, interpolation="nearest")
    axes[2].imshow(y_pred.astype(float), cmap=cmap, norm=norm, interpolation="nearest")
    for ax, title in zip(axes, titles, strict=True):
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    return fig


def rgb_from_patch(x_2d: np.ndarray) -> np.ndarray:
    """Extrae una imagen RGB ``(H, W, 3)`` normalizada de un patch 2D.

    Toma las bandas B4/B3/B2 y reescala por percentiles (2-98) para un
    contraste visual razonable (las reflectancias S2 crudas son oscuras).

    Args:
        x_2d: Patch ``(10, H, W)`` (ya colapsado en tiempo, escala cualquiera).

    Returns:
        Array ``(H, W, 3)`` float en ``[0, 1]``.
    """
    rgb = np.stack([x_2d[b] for b in _RGB_BANDS], axis=-1).astype(np.float32)
    lo, hi = np.nanpercentile(rgb, 2), np.nanpercentile(rgb, 98)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((rgb - lo) / (hi - lo), 0.0, 1.0)


@torch.no_grad()
def evaluate_checkpoint(
    model: torch.nn.Module,
    dataset: object,
    *,
    model_kind: str,
    num_classes: int = 18,
    ignore_index: int = 255,
    max_patches: int | None = None,
) -> tuple[dict[str, object], np.ndarray]:
    """Evalua un checkpoint sobre un split acumulando la matriz de confusion.

    Recorre el ``dataset`` de validacion patch a patch, predice y acumula la
    matriz de confusion densa; al final deriva todas las metricas con
    :func:`ml.eval.metrics.dense_metrics_from_cm` (mIoU, F1-macro, pixel_acc,
    balanced accuracy, Cohen kappa, IoU y F1 por clase). Es el helper que las
    notebooks ``5*`` invocan para reproducir las cifras del entrenamiento sin
    re-entrenar: cargan ``best.pt`` y llaman aqui.

    Args:
        model: Modelo cargado en ``eval()`` (ver :func:`load_segmentation_model`).
        dataset: ``PASTISSegmentationDataset`` del split de validacion.
        model_kind: Arquitectura (decide la firma del forward).
        num_classes: Numero de clases (18 semantico o 6 HCAT).
        ignore_index: Valor ignorado en las etiquetas.
        max_patches: Si se da, limita el numero de patches evaluados (smoke).

    Returns:
        Tupla ``(metrics, cm)``: ``metrics`` es el dict completo de
        ``dense_metrics_from_cm`` y ``cm`` la matriz de confusion
        ``(num_classes, num_classes)`` acumulada.
    """
    from ml.eval.metrics import dense_confusion_matrix, dense_metrics_from_cm

    n = len(dataset)  # type: ignore[arg-type]
    if max_patches is not None:
        n = min(n, max_patches)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for idx in range(n):
        x, y = dataset[idx]  # type: ignore[index]
        pred = predict_patch(model, x, model_kind=model_kind)
        cm += dense_confusion_matrix(
            pred, y.numpy(), n_classes=num_classes, ignore_index=ignore_index
        )
    metrics = dense_metrics_from_cm(cm)
    logger.info(
        "checkpoint_evaluated",
        model_kind=model_kind,
        n_patches=n,
        num_classes=num_classes,
        miou=round(float(metrics["miou"]), 4),
        f1_macro=round(float(metrics["f1_macro"]), 4),
        pixel_acc=round(float(metrics["pixel_acc"]), 4),
    )
    return metrics, cm


def predict_examples(
    model: torch.nn.Module,
    dataset: object,
    *,
    model_kind: str,
    indices: list[int],
    num_classes: int = 18,
    ignore_index: int = 255,
) -> list[Figure]:
    """Genera las figuras RGB|GT|pred para una lista de patches del dataset.

    Helper de alto nivel para el notebook: por cada indice, obtiene el patch,
    predice, arma el RGB y construye la figura comparativa. Para modelos
    temporales colapsa la serie por mediana solo para el panel RGB.

    Args:
        model: Modelo cargado en ``eval()``.
        dataset: ``PASTISSegmentationDataset`` (2D o temporal segun el modelo).
        model_kind: Arquitectura.
        indices: Indices de los patches a visualizar.
        num_classes: Numero de clases.
        ignore_index: Valor ignorado.

    Returns:
        Lista de figuras matplotlib (una por indice).
    """
    figs: list[Figure] = []
    for idx in indices:
        x, y = dataset[idx]  # type: ignore[index]
        x_np = x.numpy()
        if x_np.ndim == 4:  # temporal (T,10,H,W) -> RGB de la mediana temporal
            rgb = rgb_from_patch(np.median(x_np, axis=0))
        else:  # 2D (10,H,W)
            rgb = rgb_from_patch(x_np)
        pred = predict_patch(model, x, model_kind=model_kind)
        figs.append(
            prediction_figure(
                rgb,
                y.numpy(),
                pred,
                num_classes=num_classes,
                ignore_index=ignore_index,
            )
        )
    return figs
