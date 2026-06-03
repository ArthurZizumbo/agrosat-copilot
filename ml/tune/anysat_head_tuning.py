"""Ajuste fino de la cabeza lineal de AnySat sobre features del encoder cacheadas.

El encoder de AnySat esta congelado, asi que sus features densas no cambian entre
trials de Optuna. Re-ejecutarlo por trial (decenas de minutos por epoca a batch 1
sobre la serie temporal) es el cuello de botella del tuning. Aqui se precomputan
las features **una sola vez** y cada trial entrena unicamente la cabeza Conv 1x1
sobre ellas (segundos), de modo que >=30 trials corren en minutos en vez de horas.

Reusa las metricas densas (:mod:`ml.eval.dense_metrics`) y el agrupamiento HCAT 18
clases -> 6 grupos (:mod:`ml.analysis.hcat_grouping`) del pipeline principal, para
que el ``miou_grouped`` que optimiza Optuna sea el mismo que reporta el modelo
final (separation of concerns, regla CLAUDE.md 8).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ml.analysis.hcat_grouping import hcat6_dense_lut
from ml.eval.dense_metrics import DenseConfusionAccumulator
from ml.ingest.pastis_dataset import (
    PASTIS_IGNORE_INDEX,
    PASTIS_NUM_CLASSES,
    PASTISDataset,
)

if TYPE_CHECKING:  # pragma: no cover - only for type annotations
    from ml.models.anysat_wrapper import AnySatSegmenter

logger = structlog.get_logger(__name__)

__all__ = ["CachedFeatures", "cache_encoder_features", "train_head"]

# "Non-crop" class for the grouped metrics (background/void predicted over a
# crop pixel): never a target, so the macro averages only the 6 groups.
_NON_CROP_GROUP = 6
_GROUPED_CLASSES = 7


class CachedFeatures:
    """Par ``(features, labels)`` precomputado del encoder congelado, en CPU.

    Attributes:
        features: ``(N, D, h, w)`` float16 en CPU (mapa denso del encoder).
        labels: ``(N, target_size, target_size)`` long en CPU (semantico 0-19).
    """

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features = features
        self.labels = labels

    @property
    def feature_dim(self) -> int:
        """Numero de canales ``D`` de las features densas del encoder."""
        return int(self.features.shape[1])

    def __len__(self) -> int:
        return int(self.features.shape[0])


@torch.no_grad()
def cache_encoder_features(
    model: AnySatSegmenter,
    patch_ids: Sequence[str],
    *,
    root: Path,
    target_size: int,
    norm: tuple,
    device: str = "auto",
    batch_size: int = 4,
    num_workers: int = 0,
) -> CachedFeatures:
    """Ejecuta el encoder congelado una vez por patch y cachea ``(features, labels)``.

    Args:
        model: :class:`~ml.models.anysat_wrapper.AnySatSegmenter` (o compatible con
            ``extract_features(image, dates)``), con el encoder ya cargado.
        patch_ids: Ids de los patches PASTIS a cachear.
        root: Raiz del dataset PASTIS-R.
        target_size: Lado espacial de los labels (debe coincidir con el del modelo).
        norm: Stats de normalizacion por banda (de ``load_norm_stats``).
        device: ``cpu``, ``cuda`` o ``auto``.
        batch_size: Batch para la pasada del encoder (mayor = mas rapido si entra).
        num_workers: Workers del DataLoader.

    Returns:
        :class:`CachedFeatures` con las features ``(N, D, h, w)`` en float16 (CPU) y
        los labels ``(N, target_size, target_size)`` long (CPU).
    """
    dev = _resolve_device(device)
    model.to(dev)
    model.eval()
    dataset = PASTISDataset(
        list(patch_ids),
        root=root,
        target_size=target_size,
        temporal_reduction="none",
        norm=norm,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )
    feats_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    for batch in loader:
        image = batch["image"].to(dev)
        dates = batch.get("dates")
        dates = dates.to(dev) if dates is not None else None
        feats = model.extract_features(image, dates)  # (B, D, h, w)
        feats_chunks.append(feats.detach().to("cpu", dtype=torch.float16))
        label_chunks.append(batch["semantic"].cpu())
    features = torch.cat(feats_chunks, dim=0)
    labels = torch.cat(label_chunks, dim=0)
    logger.info(
        "anysat_features_cached",
        n=int(features.shape[0]),
        feature_dim=int(features.shape[1]),
        feat_hw=tuple(features.shape[2:]),
    )
    return CachedFeatures(features, labels)


def _resolve_device(device: str) -> torch.device:
    """Resuelve el dispositivo (``auto`` -> cuda si disponible, sino cpu)."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _build_group_luts(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Construye los LUT 18 clases -> 6 grupos para target y para prediccion."""
    group_lut = hcat6_dense_lut()
    lut_target = torch.as_tensor(group_lut, device=device)
    pred_lut = group_lut.copy()
    pred_lut[pred_lut == 255] = _NON_CROP_GROUP  # background/void predicted -> "non-crop"
    lut_pred = torch.as_tensor(pred_lut, device=device)
    return lut_target, lut_pred


def _evaluate_head(
    head: nn.Module,
    cached: CachedFeatures,
    *,
    target_size: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    """Evalua la cabeza sobre features cacheadas: mIoU/F1/pixacc planos y agrupados."""
    head.eval()
    acc = DenseConfusionAccumulator(
        PASTIS_NUM_CLASSES, ignore_index=PASTIS_IGNORE_INDEX, device=str(device)
    )
    acc_grouped = DenseConfusionAccumulator(
        _GROUPED_CLASSES, ignore_index=255, device=str(device)
    )
    lut_target, lut_pred = _build_group_luts(device)
    n = len(cached)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            feats = cached.features[i : i + batch_size].to(device, dtype=torch.float32)
            target = cached.labels[i : i + batch_size].to(device)
            logits = F.interpolate(
                head(feats), size=(target_size, target_size), mode="bilinear", align_corners=False
            )
            preds = logits.argmax(dim=1)
            acc.update(preds, target)
            acc_grouped.update(lut_pred[preds.clamp(0, 19)], lut_target[target.clamp(0, 19)])
    flat = acc.compute()
    grouped = {f"{k}_grouped": v for k, v in acc_grouped.compute().items()}
    return {**flat, **grouped}


def train_head(
    train_cache: CachedFeatures,
    val_cache: CachedFeatures,
    *,
    num_classes: int = PASTIS_NUM_CLASSES,
    target_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 8,
    batch_size: int = 8,
    device: str = "auto",
    seed: int = 0,
    on_epoch: Callable[[int, dict[str, float]], None] | None = None,
) -> dict[str, float]:
    """Entrena una cabeza Conv 1x1 sobre features cacheadas y devuelve la mejor metrica.

    Cada epoca entrena la cabeza (AdamW + CrossEntropy con ``ignore_index`` void) y
    evalua mIoU/F1/pixel-accuracy planos y agrupados sobre el cache de validacion. La
    metrica de seleccion es ``miou_grouped`` (comparable con el baseline y el modelo
    final). Es la unidad de trabajo de cada trial de Optuna.

    Args:
        train_cache: Features+labels de train (de :func:`cache_encoder_features`).
        val_cache: Features+labels de validacion.
        num_classes: Numero de clases de salida (20 en PASTIS-R).
        target_size: Lado espacial de los logits/labels.
        lr: Learning rate de AdamW.
        weight_decay: Weight decay de AdamW.
        epochs: Numero de epocas de la cabeza.
        batch_size: Batch sobre las features cacheadas (barato; puede ser alto).
        device: ``cpu``, ``cuda`` o ``auto``.
        seed: Semilla para el orden de los minibatches (reproducibilidad por trial).
        on_epoch: Callback ``(epoch, metrics)`` tras evaluar cada epoca (pruning Optuna).

    Returns:
        Diccionario con las mejores metricas (mayor ``miou_grouped``) observadas.
    """
    dev = _resolve_device(device)
    feature_dim = train_cache.feature_dim
    head = nn.Conv2d(feature_dim, num_classes, kernel_size=1).to(dev)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=PASTIS_IGNORE_INDEX)

    generator = torch.Generator().manual_seed(seed)
    n = len(train_cache)
    best: dict[str, float] = {"miou_grouped": -1.0}
    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(n, generator=generator)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            feats = train_cache.features[idx].to(dev, dtype=torch.float32)
            target = train_cache.labels[idx].to(dev)
            optimizer.zero_grad()
            logits = F.interpolate(
                head(feats), size=(target_size, target_size), mode="bilinear", align_corners=False
            )
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

        metrics = _evaluate_head(
            head, val_cache, target_size=target_size, device=dev, batch_size=batch_size
        )
        if metrics["miou_grouped"] >= best["miou_grouped"]:
            best = metrics
        if on_epoch is not None:
            on_epoch(epoch, metrics)
    return best
