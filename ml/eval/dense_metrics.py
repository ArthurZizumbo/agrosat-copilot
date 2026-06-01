"""Metricas pixel-level para segmentacion semantica densa (EPIC 5/6).

Complementa :mod:`ml.eval.metrics` (que opera a nivel parcela) con las tres
metricas de segmentacion exigidas por la rubrica del Avance 4: **mIoU**,
**F1-macro** y **pixel-accuracy**, calculadas a nivel pixel sobre mapas 2D.

La implementacion acumula una matriz de confusion ``(C, C)`` en torch puro (sin
dependencia de ``torchmetrics``), lo que permite agregar batches en streaming
durante la validacion y derivar las tres metricas de forma exacta al final. La
clase ``ignore_index`` (void = 19 en PASTIS-R) se excluye tanto de la
acumulacion como del promedio macro.
"""

from __future__ import annotations

import numpy as np
import torch
from matplotlib.figure import Figure

from ml.eval.metrics import confusion_matrix_figure

__all__ = [
    "DenseConfusionAccumulator",
    "compute_dense_metrics",
    "dense_confusion_figure",
]


def _as_long_tensor(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Convierte una entrada numpy/torch a ``torch.Tensor`` ``long`` en CPU."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).long()
    return x.detach().long()


class DenseConfusionAccumulator:
    """Acumulador de matriz de confusion pixel-level para metricas densas.

    Permite ``update`` por batch durante la validacion y ``compute`` al final,
    derivando mIoU, F1-macro y pixel-accuracy de la matriz acumulada. La clase
    ``ignore_index`` se filtra del ground truth antes de acumular.

    Attributes:
        num_classes: Numero de clases del problema.
        ignore_index: Clase a ignorar (no contribuye a la confusion ni al macro).
    """

    def __init__(
        self,
        num_classes: int,
        *,
        ignore_index: int | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """Inicializa el acumulador con una matriz ``(C, C)`` en ceros.

        Args:
            num_classes: Numero de clases ``C``.
            ignore_index: Clase a ignorar (``None`` para no ignorar ninguna).
            device: Dispositivo donde mantener la matriz acumulada.
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self._device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        """Reinicia la matriz de confusion acumulada a ceros."""
        self._confusion = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.int64, device=self._device
        )

    def update(self, preds: torch.Tensor | np.ndarray, target: torch.Tensor | np.ndarray) -> None:
        """Acumula un batch de predicciones contra el ground truth.

        Args:
            preds: Mapa(s) de clases predichas, enteros de cualquier forma.
            target: Mapa(s) de clases verdaderas, misma forma que ``preds``.

        Raises:
            ValueError: si ``preds`` y ``target`` difieren en forma.
        """
        preds_t = _as_long_tensor(preds).to(self._device).reshape(-1)
        target_t = _as_long_tensor(target).to(self._device).reshape(-1)
        if preds_t.shape != target_t.shape:
            raise ValueError(
                f"`preds` y `target` deben tener el mismo numero de pixeles; "
                f"recibido {preds_t.numel()} vs {target_t.numel()}."
            )

        valid = torch.ones_like(target_t, dtype=torch.bool)
        if self.ignore_index is not None:
            valid &= target_t != self.ignore_index
        # Defensivo: descarta pixeles fuera de rango (p.ej. pred==num_classes).
        valid &= (target_t >= 0) & (target_t < self.num_classes)
        valid &= (preds_t >= 0) & (preds_t < self.num_classes)

        t = target_t[valid]
        p = preds_t[valid]
        if t.numel() == 0:
            return
        indices = t * self.num_classes + p
        binned = torch.bincount(indices, minlength=self.num_classes**2)
        self._confusion += binned.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, float]:
        """Deriva mIoU, F1-macro y pixel-accuracy de la matriz acumulada.

        El promedio macro (mIoU y F1) se toma solo sobre las clases presentes en
        el ground truth (soporte > 0), excluyendo ``ignore_index``. Esto evita
        sesgar la metrica hacia abajo por clases ausentes en el split de val.

        Returns:
            Diccionario con ``miou``, ``f1_macro`` y ``pixel_accuracy`` (floats
            en ``[0, 1]``). Si no se acumulo ningun pixel valido, devuelve ceros.
        """
        conf = self._confusion.double()
        total = conf.sum()
        if total <= 0:
            return {"miou": 0.0, "f1_macro": 0.0, "pixel_accuracy": 0.0}

        diag = torch.diag(conf)
        row_sum = conf.sum(dim=1)  # soporte real por clase
        col_sum = conf.sum(dim=0)  # predicciones por clase

        union = row_sum + col_sum - diag
        iou = torch.where(union > 0, diag / union, torch.zeros_like(diag))

        precision = torch.where(col_sum > 0, diag / col_sum, torch.zeros_like(diag))
        recall = torch.where(row_sum > 0, diag / row_sum, torch.zeros_like(diag))
        denom = precision + recall
        f1 = torch.where(denom > 0, 2 * precision * recall / denom, torch.zeros_like(diag))

        present = row_sum > 0
        if self.ignore_index is not None and 0 <= self.ignore_index < self.num_classes:
            present[self.ignore_index] = False

        n_present = int(present.sum().item())
        miou = float(iou[present].mean().item()) if n_present > 0 else 0.0
        f1_macro = float(f1[present].mean().item()) if n_present > 0 else 0.0
        pixel_accuracy = float((diag.sum() / total).item())
        return {"miou": miou, "f1_macro": f1_macro, "pixel_accuracy": pixel_accuracy}

    def per_class_iou(self) -> dict[int, float]:
        """Devuelve el IoU por clase (para el barplot de IoU por clase).

        Returns:
            Diccionario ``{class_id: iou}`` solo para las clases con soporte en el
            ground truth (excluyendo ``ignore_index``). Vacio si no hay pixeles.
        """
        conf = self._confusion.double()
        if conf.sum() <= 0:
            return {}
        diag = torch.diag(conf)
        row_sum = conf.sum(dim=1)
        col_sum = conf.sum(dim=0)
        union = row_sum + col_sum - diag
        iou = torch.where(union > 0, diag / union, torch.zeros_like(diag))
        out: dict[int, float] = {}
        for c in range(self.num_classes):
            if c == self.ignore_index or row_sum[c] <= 0:
                continue
            out[c] = float(iou[c].item())
        return out


def compute_dense_metrics(
    preds: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    *,
    num_classes: int,
    ignore_index: int | None = None,
) -> dict[str, float]:
    """Calcula mIoU + F1-macro + pixel-accuracy en una sola pasada (one-shot).

    Conveniencia sobre :class:`DenseConfusionAccumulator` para evaluar un par
    ``(preds, target)`` completo de una vez (tests, evaluacion final).

    Args:
        preds: Mapa(s) de clases predichas.
        target: Mapa(s) de clases verdaderas.
        num_classes: Numero de clases ``C``.
        ignore_index: Clase a ignorar (default ``None``).

    Returns:
        Diccionario con ``miou``, ``f1_macro`` y ``pixel_accuracy``.
    """
    acc = DenseConfusionAccumulator(num_classes, ignore_index=ignore_index)
    acc.update(preds, target)
    return acc.compute()


def dense_confusion_figure(
    preds: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    *,
    class_names: dict[int, str] | None = None,
    ignore_index: int | None = None,
    normalize: bool = True,
) -> Figure:
    """Matriz de confusion pixel-level reutilizando :func:`confusion_matrix_figure`.

    Aplana los mapas 2D a vectores de pixeles, descarta los pixeles cuyo ground
    truth es ``ignore_index`` y delega el render en el helper ya existente del
    baseline (DRY, mismo estilo visual que las matrices a nivel parcela).

    Args:
        preds: Mapa(s) de clases predichas.
        target: Mapa(s) de clases verdaderas.
        class_names: Mapa ``{class_id: nombre}`` para rotular ejes.
        ignore_index: Clase a excluir del plot (default ``None``).
        normalize: Si ``True`` normaliza por fila (recall por clase).

    Returns:
        Figura matplotlib lista para ``savefig``/``display``.
    """
    p = _as_long_tensor(preds).reshape(-1).cpu().numpy()
    t = _as_long_tensor(target).reshape(-1).cpu().numpy()
    if ignore_index is not None:
        mask = t != ignore_index
        p, t = p[mask], t[mask]
    return confusion_matrix_figure(t, p, class_names=class_names, normalize=normalize)
