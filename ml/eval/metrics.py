"""Metricas del baseline de clasificacion de cultivos (US-019, EPIC 4).

Modulo reutilizable consumido por el baseline tabular (RF/XGB) y, mas
adelante, por las arquitecturas de segmentacion del EPIC 5/6. Expone las
cinco metricas exactas del criterio de aceptacion AC-3 mas dos artefactos
visuales/textuales (matriz de confusion y reporte de clasificacion).

Decision D6 (plan US-019 2.1): la ``mIoU`` del baseline se calcula como
``jaccard_score(average="macro")`` a nivel parcela. Es un *proxy* de la
mIoU de segmentacion densa pixel-level que llegara en el EPIC 5; se
documenta como tal para mantener comparabilidad de tablas entre epicas.
"""

from __future__ import annotations

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

__all__ = [
    "classification_report_text",
    "compute_baseline_metrics",
    "confusion_matrix_figure",
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
