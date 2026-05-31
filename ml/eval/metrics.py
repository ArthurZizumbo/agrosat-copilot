"""Metricas del baseline de clasificacion de cultivos (US-019, EPIC 4) y de
segmentacion densa pixel-level (US-025, EPIC 5).

Modulo reutilizable consumido por el baseline tabular (RF/XGB) y por las
arquitecturas de segmentacion del EPIC 5/6. Expone:

* Nivel parcela (AC-3 US-019): ``compute_baseline_metrics`` con las cinco
  metricas exactas mas dos artefactos (matriz de confusion y reporte).
* Nivel pixel (US-025): ``dense_miou``, ``dense_f1_macro``,
  ``dense_pixel_accuracy`` y ``segmentation_metrics_report``, que aceptan
  tensores ``torch`` o ``numpy``, soportan logits ``(B, C, H, W)`` o
  etiquetas ``(B, H, W)`` e ignoran un ``ignore_index`` (Background/Void).

Decision D6 (plan US-019 2.1): la ``mIoU`` del baseline tabular se calcula
como ``jaccard_score(average="macro")`` a nivel parcela. Es un *proxy* de la
mIoU de segmentacion densa pixel-level del EPIC 5; se documenta como tal para
mantener comparabilidad de tablas entre epicas. Las funciones ``dense_*`` de
este modulo son la mIoU densa real (Jaccard por clase agregado sobre todos
los pixeles validos del lote).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
)

if TYPE_CHECKING:  # pragma: no cover - solo para anotaciones de tipo
    import torch

    DenseArray = np.ndarray | torch.Tensor
else:
    DenseArray = Any

__all__ = [
    "classification_report_text",
    "compute_baseline_metrics",
    "confusion_matrix_figure",
    "dense_confusion_matrix",
    "dense_f1_macro",
    "dense_miou",
    "dense_pixel_accuracy",
    "segmentation_metrics_report",
]


def compute_baseline_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: list[int] | None = None,
) -> dict[str, float]:
    """Calcula las cinco metricas del baseline (criterio AC-3).

    Args:
        y_true: Etiquetas verdaderas, vector ``(n_samples,)`` de enteros.
        y_pred: Etiquetas predichas, vector ``(n_samples,)`` de enteros del
            mismo largo que ``y_true``.
        labels: Conjunto explicito de etiquetas a considerar. Si es
            ``None`` se infiere de la union de clases presentes en
            ``y_true`` e ``y_pred`` (orden ascendente). Pasar el universo
            completo de clases garantiza metricas estables entre folds.

    Returns:
        Diccionario con las claves exactas ``f1_macro``, ``f1_weighted``,
        ``miou``, ``accuracy`` y ``cohen_kappa``, todas ``float``. Las
        cuatro primeras viven en ``[0, 1]``; ``cohen_kappa`` puede ser
        negativo (acuerdo peor que el azar).

    Raises:
        ValueError: si ``y_true`` e ``y_pred`` difieren en longitud o si
            ambos vectores estan vacios.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"`y_true` y `y_pred` deben tener la misma forma; "
            f"recibido {y_true.shape} vs {y_pred.shape}."
        )
    if y_true.size == 0:
        raise ValueError("`y_true` e `y_pred` no pueden estar vacios.")

    if labels is None:
        resolved_labels: list[int] = sorted(
            int(c) for c in np.union1d(y_true, y_pred)
        )
    else:
        resolved_labels = [int(c) for c in labels]

    return {
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                labels=resolved_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true,
                y_pred,
                labels=resolved_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "miou": float(
            jaccard_score(
                y_true,
                y_pred,
                labels=resolved_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_names: dict[int, str] | None = None,
    normalize: bool = True,
) -> Figure:
    """Construye la matriz de confusion como :class:`matplotlib.figure.Figure`.

    Usa el backend ``Agg`` de matplotlib (no interactivo) para que la
    figura sea serializable a PNG en CI y en notebooks ejecutados con
    papermill.

    Args:
        y_true: Etiquetas verdaderas, vector ``(n_samples,)``.
        y_pred: Etiquetas predichas, vector ``(n_samples,)``.
        class_names: Mapa ``{class_id: nombre}`` para rotular ejes. Si es
            ``None`` se usan los enteros de clase como etiqueta.
        normalize: Si ``True`` (default) normaliza cada fila para que
            sume 1.0 (recall por clase); si ``False`` muestra conteos.

    Returns:
        Figura matplotlib lista para ``fig.savefig(...)`` o ``display``.

    Raises:
        ValueError: si ``y_true`` e ``y_pred`` difieren en longitud.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"`y_true` y `y_pred` deben tener la misma forma; "
            f"recibido {y_true.shape} vs {y_pred.shape}."
        )

    labels = sorted(int(c) for c in np.union1d(y_true, y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display_matrix = matrix.astype(np.float64)
    if normalize:
        row_sums = display_matrix.sum(axis=1, keepdims=True)
        # Evita division por cero en clases ausentes del ground truth.
        row_sums[row_sums == 0.0] = 1.0
        display_matrix = display_matrix / row_sums

    tick_labels = [
        (class_names.get(c, str(c)) if class_names else str(c)) for c in labels
    ]

    fig, ax = plt.subplots(figsize=(max(6.0, len(labels) * 0.6),) * 2)
    image = ax.imshow(display_matrix, cmap="Blues", vmin=0.0, vmax=display_matrix.max() or 1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Verdadero")
    ax.set_title(
        "Matriz de confusion " + ("normalizada (recall)" if normalize else "(conteos)")
    )

    text_fmt = "{:.2f}" if normalize else "{:.0f}"
    threshold = display_matrix.max() / 2.0 if display_matrix.size else 0.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = display_matrix[i, j]
            ax.text(
                j,
                i,
                text_fmt.format(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=7,
            )
    fig.tight_layout()
    return fig


def classification_report_text(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_names: dict[int, str] | None = None,
) -> str:
    """Devuelve el reporte de clasificacion de sklearn como texto.

    Args:
        y_true: Etiquetas verdaderas, vector ``(n_samples,)``.
        y_pred: Etiquetas predichas, vector ``(n_samples,)``.
        class_names: Mapa ``{class_id: nombre}`` para usar nombres de
            clase legibles en lugar de enteros.

    Returns:
        Cadena multilinea con precision, recall, F1 y soporte por clase
        mas los promedios macro y weighted, lista para ``log_artifact`` o
        para imprimir en el notebook.

    Raises:
        ValueError: si ``y_true`` e ``y_pred`` difieren en longitud.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"`y_true` y `y_pred` deben tener la misma forma; "
            f"recibido {y_true.shape} vs {y_pred.shape}."
        )

    labels = sorted(int(c) for c in np.union1d(y_true, y_pred))
    target_names = [
        (class_names.get(c, str(c)) if class_names else str(c)) for c in labels
    ]
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        digits=4,
    )


# ---------------------------------------------------------------------------
# Segmentacion densa pixel-level (US-025, EPIC 5)
# ---------------------------------------------------------------------------


def _to_numpy(arr: DenseArray) -> np.ndarray:
    """Convierte un tensor ``torch`` o array ``numpy`` a ``numpy`` sin copia gratis.

    Args:
        arr: Tensor de ``torch`` (en cualquier device) o ``numpy.ndarray``.

    Returns:
        El contenido como ``numpy.ndarray`` en CPU. Si ``arr`` ya es
        ``numpy`` se devuelve tal cual (``np.asarray`` no copia si el dtype
        y la contiguidad ya coinciden).
    """
    if hasattr(arr, "detach"):  # torch.Tensor (evita import duro de torch)
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _to_label_array(arr: DenseArray, *, n_classes: int) -> np.ndarray:
    """Normaliza la entrada a etiquetas enteras ``(N,)`` aplanadas.

    Acepta tanto logits/probabilidades ``(B, C, H, W)`` (se aplica
    ``argmax`` sobre el eje de canal ``C``) como etiquetas duras de
    cualquier forma ``(B, H, W)``, ``(H, W)`` o ya aplanadas.

    La heuristica para detectar logits es: array de punto flotante con un
    eje cuyo tamano coincide con ``n_classes`` en la posicion de canal
    (eje 1 para ``(B, C, H, W)``). Las etiquetas enteras se tratan siempre
    como etiquetas, nunca como logits.

    Args:
        arr: Logits ``(B, C, H, W)`` o etiquetas de forma arbitraria.
        n_classes: Numero de clases ``C`` esperado para reconocer logits.

    Returns:
        Vector ``numpy`` 1-D de etiquetas enteras (``int64``).
    """
    data = _to_numpy(arr)
    is_float = np.issubdtype(data.dtype, np.floating)
    if is_float and data.ndim == 4 and data.shape[1] == n_classes:
        data = data.argmax(axis=1)
    elif is_float and data.ndim == 4:
        # Float 4-D sin canal == n_classes: asumir canal en eje 1 de todos modos.
        data = data.argmax(axis=1)
    return data.reshape(-1).astype(np.int64, copy=False)


def dense_confusion_matrix(
    y_pred: DenseArray,
    y_true: DenseArray,
    *,
    n_classes: int = 18,
    ignore_index: int = 255,
) -> np.ndarray:
    """Construye la matriz de confusion densa ``(n_classes, n_classes)``.

    Agrega sobre todos los pixeles validos del lote. Los pixeles cuyo
    *ground truth* es ``ignore_index`` (Background/Void) se excluyen por
    completo, igual que los pixeles cuya etiqueta verdadera cae fuera de
    ``[0, n_classes)`` (defensa frente a targets mal mapeados).

    Args:
        y_pred: Logits ``(B, C, H, W)`` o etiquetas predichas ``(B, H, W)``.
        y_true: Etiquetas verdaderas (``numpy`` o ``torch``) de la misma
            cantidad de pixeles que ``y_pred`` tras aplanar.
        n_classes: Numero de clases ``C`` (18 para PASTIS-R semantico).
        ignore_index: Valor de etiqueta a ignorar (Background/Void).

    Returns:
        Matriz ``numpy`` ``int64`` de forma ``(n_classes, n_classes)`` con
        ``cm[i, j]`` = pixeles de clase verdadera ``i`` predichos como ``j``.

    Raises:
        ValueError: si ``y_pred`` e ``y_true`` no tienen el mismo numero de
            pixeles tras aplanar.
    """
    pred = _to_label_array(y_pred, n_classes=n_classes)
    true = _to_label_array(y_true, n_classes=n_classes)
    if pred.shape != true.shape:
        raise ValueError(
            f"`y_pred` e `y_true` deben tener el mismo numero de pixeles; "
            f"recibido {pred.shape} vs {true.shape}."
        )

    valid = (true != ignore_index) & (true >= 0) & (true < n_classes)
    true = true[valid]
    pred = pred[valid]
    # Las predicciones fuera de rango se cuelan a la ultima clase para no
    # romper el bincount; en la practica argmax sobre n_classes nunca excede.
    pred = np.clip(pred, 0, n_classes - 1)

    indices = true * n_classes + pred
    counts = np.bincount(indices, minlength=n_classes * n_classes)
    return counts.reshape(n_classes, n_classes).astype(np.int64, copy=False)


def _per_class_iou_from_cm(cm: np.ndarray) -> np.ndarray:
    """IoU (Jaccard) por clase a partir de una matriz de confusion.

    Args:
        cm: Matriz de confusion ``(n_classes, n_classes)`` densa.

    Returns:
        Vector ``(n_classes,)`` con el IoU por clase. Las clases ausentes
        del *ground truth* y de las predicciones (union vacia) reciben
        ``nan`` para excluirse del promedio macro.
    """
    cm = cm.astype(np.float64)
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0.0, intersection / union, np.nan)
    return iou


def dense_miou(
    y_pred: DenseArray,
    y_true: DenseArray,
    *,
    n_classes: int = 18,
    ignore_index: int = 255,
) -> float:
    """Calcula la mIoU (mean Jaccard) densa pixel-level.

    Promedio macro del IoU por clase sobre las clases presentes en la
    union (ground truth o prediccion). Las clases totalmente ausentes se
    excluyen del promedio (no penalizan con cero), siguiendo la convencion
    de PASTIS/U-TAE para folds donde no aparecen todas las clases.

    Args:
        y_pred: Logits ``(B, C, H, W)`` o etiquetas ``(B, H, W)`` (``torch``
            o ``numpy``).
        y_true: Etiquetas verdaderas (``torch`` o ``numpy``).
        n_classes: Numero de clases (18 para PASTIS-R semantico).
        ignore_index: Etiqueta a ignorar (Background/Void).

    Returns:
        mIoU en ``[0, 1]``. Devuelve ``0.0`` si no hay ninguna clase valida
        (todos los pixeles eran ``ignore_index``).
    """
    cm = dense_confusion_matrix(
        y_pred, y_true, n_classes=n_classes, ignore_index=ignore_index
    )
    iou = _per_class_iou_from_cm(cm)
    if np.all(np.isnan(iou)):
        return 0.0
    return float(np.nanmean(iou))


def dense_f1_macro(
    y_pred: DenseArray,
    y_true: DenseArray,
    *,
    n_classes: int = 18,
    ignore_index: int = 255,
) -> float:
    """Calcula el F1-macro denso pixel-level (Dice por clase promediado).

    Equivale al promedio macro del F1 por clase sobre los pixeles validos.
    Las clases ausentes de la union (sin GT ni prediccion) se excluyen del
    promedio, en linea con ``dense_miou``.

    Args:
        y_pred: Logits ``(B, C, H, W)`` o etiquetas ``(B, H, W)``.
        y_true: Etiquetas verdaderas.
        n_classes: Numero de clases.
        ignore_index: Etiqueta a ignorar.

    Returns:
        F1-macro en ``[0, 1]``. Devuelve ``0.0`` si no hay clases validas.
    """
    cm = dense_confusion_matrix(
        y_pred, y_true, n_classes=n_classes, ignore_index=ignore_index
    ).astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2.0 * tp + fp + fn
    present = (cm.sum(axis=1) + cm.sum(axis=0)) > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0.0, 2.0 * tp / denom, 0.0)
    if not np.any(present):
        return 0.0
    return float(f1[present].mean())


def dense_pixel_accuracy(
    y_pred: DenseArray,
    y_true: DenseArray,
    *,
    n_classes: int = 18,
    ignore_index: int = 255,
) -> float:
    """Calcula la exactitud global a nivel pixel (pixeles correctos / validos).

    Args:
        y_pred: Logits ``(B, C, H, W)`` o etiquetas ``(B, H, W)``.
        y_true: Etiquetas verdaderas.
        n_classes: Numero de clases (define el rango valido del target).
        ignore_index: Etiqueta a ignorar (Background/Void).

    Returns:
        Exactitud en ``[0, 1]``. Devuelve ``0.0`` si no hay pixeles validos.
    """
    cm = dense_confusion_matrix(
        y_pred, y_true, n_classes=n_classes, ignore_index=ignore_index
    )
    total = int(cm.sum())
    if total == 0:
        return 0.0
    return float(np.trace(cm)) / float(total)


def segmentation_metrics_report(
    y_pred: DenseArray,
    y_true: DenseArray,
    *,
    n_classes: int = 18,
    ignore_index: int = 255,
) -> dict[str, Any]:
    """Reporte completo de metricas de segmentacion densa en una pasada.

    Construye la matriz de confusion una sola vez y deriva todas las
    metricas, evitando recomputos (DRY/eficiencia). Util para registrar en
    MLflow al cierre de cada epoch/eval del EPIC 5.

    Args:
        y_pred: Logits ``(B, C, H, W)`` o etiquetas ``(B, H, W)`` (``torch``
            o ``numpy``).
        y_true: Etiquetas verdaderas (``torch`` o ``numpy``).
        n_classes: Numero de clases (18 PASTIS-R semantico, 6 HCAT L1).
        ignore_index: Etiqueta a ignorar (Background/Void).

    Returns:
        Diccionario con las claves:

        * ``miou`` (``float``): mean IoU macro sobre clases presentes.
        * ``f1_macro`` (``float``): F1-macro denso.
        * ``pixel_acc`` (``float``): exactitud global pixel-level.
        * ``per_class_iou`` (``list[float | None]``): IoU por clase de ``0``
          a ``n_classes - 1``; ``None`` para clases ausentes de la union.
    """
    cm = dense_confusion_matrix(
        y_pred, y_true, n_classes=n_classes, ignore_index=ignore_index
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

    per_class_iou: list[float | None] = [
        (None if np.isnan(value) else float(value)) for value in iou
    ]

    return {
        "miou": miou,
        "f1_macro": f1_macro,
        "pixel_acc": pixel_acc,
        "per_class_iou": per_class_iou,
    }
