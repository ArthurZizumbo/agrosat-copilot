"""Arquitecturas de segmentacion semantica densa 2D (EPIC 5, Avance 4).

Factory para los modelos de segmentacion basados en composites temporales 2D
(``image (B, 10, H, W) -> logits (B, num_classes, H, W)``). El modelo #1 del
reparto del equipo, **U-Net ResNet-50**, se construye directo desde
``segmentation_models_pytorch`` (encoder ImageNet preentrenado, adaptado a 10
canales Sentinel-2). El factory ``build_segmentation_model`` queda registrable
para que el resto del equipo enganche #2 (DeepLabv3+) y #3 (SegFormer) sobre el
mismo pipeline denso (:mod:`ml.ingest.pastis_dataset`).

Los modelos temporales (#4 U-TAE, #5 TSViT, #6 AnySat) consumen la serie
completa y viven en wrappers dedicados (ver :mod:`ml.models.anysat_wrapper`).
"""

from __future__ import annotations

from collections.abc import Callable

import segmentation_models_pytorch as smp
import torch.nn as nn

__all__ = [
    "SEGMENTATION_BUILDERS",
    "build_deeplabv3plus",
    "build_segmentation_model",
    "build_unet",
]

_S2_CHANNELS = 10


def build_unet(
    num_classes: int,
    *,
    in_channels: int = _S2_CHANNELS,
    encoder_name: str = "resnet50",
    encoder_weights: str | None = "imagenet",
) -> nn.Module:
    """Construye U-Net con encoder ResNet-50 preentrenado en ImageNet (#1).

    El primer conv del encoder se adapta automaticamente de 3 a ``in_channels``
    canales (smp replica/promedia los pesos RGB). Salida sin activacion (logits)
    para usar ``CrossEntropyLoss`` con ``ignore_index``.

    Args:
        num_classes: Numero de clases de salida (20 en PASTIS-R).
        in_channels: Canales de entrada (10 bandas Sentinel-2).
        encoder_name: Backbone smp (default ``resnet50``).
        encoder_weights: Pesos del encoder (``imagenet`` o ``None``).

    Returns:
        ``nn.Module`` que mapea ``(B, in_channels, H, W) -> (B, num_classes, H, W)``.
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )


def build_deeplabv3plus(
    num_classes: int,
    *,
    in_channels: int = _S2_CHANNELS,
    encoder_name: str = "mobilenet_v2",
    encoder_weights: str | None = "imagenet",
) -> nn.Module:
    """Construye DeepLabv3+ ligero (#2) sobre el mismo pipeline denso.

    Provisto para que el integrante a cargo de #2 reutilice el factory sin
    duplicar el pipeline. ``mobilenet_v2`` es el encoder ligero (smp no expone
    ``mobilenet_v3`` para DeepLabv3+; v2 es el equivalente eficiente disponible).

    Args:
        num_classes: Numero de clases de salida.
        in_channels: Canales de entrada.
        encoder_name: Backbone smp.
        encoder_weights: Pesos del encoder.

    Returns:
        ``nn.Module`` de segmentacion densa.
    """
    return smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )


SEGMENTATION_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    "unet": build_unet,
    "deeplabv3plus": build_deeplabv3plus,
}
"""Registro ``nombre -> builder`` de modelos 2D. El equipo agrega entradas aqui."""


def build_segmentation_model(kind: str, num_classes: int, **kwargs: object) -> nn.Module:
    """Construye un modelo de segmentacion 2D por nombre registrado.

    Args:
        kind: Clave en :data:`SEGMENTATION_BUILDERS` (``unet``, ``deeplabv3plus``).
        num_classes: Numero de clases de salida.
        **kwargs: Overrides pasados al builder concreto (``encoder_name``, etc.).

    Returns:
        El ``nn.Module`` construido.

    Raises:
        ValueError: si ``kind`` no esta registrado.
    """
    builder = SEGMENTATION_BUILDERS.get(kind)
    if builder is None:
        valid = ", ".join(sorted(SEGMENTATION_BUILDERS))
        raise ValueError(f"Modelo de segmentacion desconocido: {kind!r}. Validos: {valid}.")
    return builder(num_classes, **kwargs)  # type: ignore[arg-type]
