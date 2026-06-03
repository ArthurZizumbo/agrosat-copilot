"""Factory de DeepLabv3+ (MobileNetV3-Large) para segmentacion densa 10 bandas.

Segmentador CNN base del EPIC 5 (US-025, Tarea 2). Envuelve
``segmentation_models_pytorch`` 0.5 para producir un DeepLabv3+ con encoder
MobileNetV3-Large (timm) adaptado a entrada Sentinel-2 de 10 bandas y salida
densa de 18 clases semanticas PASTIS-R.

El registro nativo de encoders de smp 0.5 NO incluye ``mobilenet_v3_large``
(solo ResNet/EfficientNet/MiT/MobileOne + un subconjunto ``timm-*``). Para
MobileNetV3-Large se usa el prefijo timm-universal ``tu-``: el nombre del
encoder es ``tu-mobilenetv3_large_100``, que delega en ``timm`` y descarga los
pesos ImageNet adaptando la primera convolucion a ``in_channels`` canales.

A diferencia del uso por defecto de smp (entrada RGB, ``in_channels=3``), aqui
``in_channels=10``: smp/timm adaptan la primera convolucion replicando y
promediando los pesos ImageNet sobre los canales extra. Si la inicializacion
ImageNet falla con 10 canales se reintenta con pesos aleatorios
(``encoder_weights=None``) emitiendo un warning estructurado.

El modulo expone tambien una perdida combinada Dice + CrossEntropy
(:func:`build_dice_ce_loss`), patron habitual en segmentacion de cultivos
desbalanceada: Dice estabiliza clases minoritarias y CrossEntropy aporta
gradiente por pixel. Ambas respetan ``ignore_index`` para Background/Void.

Decisiones tecnicas
-------------------

- ``decoder_atrous_rates=(6, 12, 18)`` por el criterio de aceptacion de la
  US-025 (el default de smp es ``(12, 24, 36)``); rates menores favorecen
  parcelas pequenas a 128px.
- ``decoder_aspp_separable=True`` reduce parametros del modulo ASPP
  (convoluciones separables en profundidad), util para entrenar en laptop RTX.
- ``encoder_name="mobilenet_v3_large"`` (timm) por presupuesto de VRAM
  (<8 GB, batch 8, 128px segun plan US-025).

Acreditacion
------------

- segmentation-models-pytorch 0.5 (MIT License), Pavel Iakubovskii.
- Chen et al. ``Encoder-Decoder with Atrous Separable Convolution for
  Semantic Image Segmentation`` (DeepLabv3+). ECCV 2018.
- Howard et al. ``Searching for MobileNetV3``. ICCV 2019.
"""

from __future__ import annotations

from collections.abc import Sequence

import segmentation_models_pytorch as smp
import structlog
import torch
from segmentation_models_pytorch.losses import DiceLoss
from torch import nn

__all__ = [
    "DiceCrossEntropyLoss",
    "build_deeplabv3plus_mobilenet",
    "build_dice_ce_loss",
]

logger = structlog.get_logger(__name__)

# Default ignore index: aligned with PASTISSegmentationDataset
# (Background/Void mapped to 255, outside the range [0..n_classes-1]).
_DEFAULT_IGNORE_INDEX = 255

# Encoder name in smp 0.5: timm-universal prefix ``tu-`` because
# ``mobilenet_v3_large`` is not in the native smp registry. timm exposes it
# as ``mobilenetv3_large_100``.
_ENCODER_NAME = "tu-mobilenetv3_large_100"


def build_deeplabv3plus_mobilenet(
    in_channels: int = 10,
    classes: int = 18,
    atrous_rates: tuple[int, int, int] = (6, 12, 18),
    encoder_weights: str | None = "imagenet",
) -> nn.Module:
    """Construye un DeepLabv3+ con encoder MobileNetV3-Large para 10 bandas.

    Args:
        in_channels: Numero de canales de entrada. Por defecto 10 (bandas
            Sentinel-2 de PASTIS-R). smp/timm adaptan la primera convolucion
            cuando es distinto de 3 (RGB).
        classes: Numero de clases de salida (canales del logit denso). Por
            defecto 18 (clases semanticas PASTIS-R sin Background/Void).
        atrous_rates: Tasas de dilatacion del modulo ASPP (3 enteros). Por
            defecto ``(6, 12, 18)`` segun el criterio de aceptacion US-025.
        encoder_weights: Pesos iniciales del encoder. ``"imagenet"`` (por
            defecto) o ``None`` (aleatorio). Si ``"imagenet"`` falla al
            adaptarse a ``in_channels`` se reintenta con ``None``.

    Returns:
        Modelo ``torch.nn.Module`` que mapea ``(B, in_channels, H, W)`` a
        logits densos ``(B, classes, H, W)``.

    Raises:
        ValueError: Si ``atrous_rates`` no contiene exactamente 3 enteros.
    """
    rates = tuple(atrous_rates)
    if len(rates) != 3:
        raise ValueError(
            f"atrous_rates debe tener 3 valores enteros, recibido {rates!r}"
        )

    try:
        model = smp.DeepLabV3Plus(
            encoder_name=_ENCODER_NAME,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            decoder_atrous_rates=rates,
            decoder_aspp_separable=True,
        )
    except (RuntimeError, ValueError, KeyError) as exc:
        if encoder_weights is None:
            # No fallback possible: random initialization was already requested.
            raise
        logger.warning(
            "deeplabv3plus_imagenet_init_failed",
            in_channels=in_channels,
            classes=classes,
            error=str(exc),
            fallback="encoder_weights=None",
        )
        model = smp.DeepLabV3Plus(
            encoder_name=_ENCODER_NAME,
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
            decoder_atrous_rates=rates,
            decoder_aspp_separable=True,
        )

    logger.info(
        "deeplabv3plus_built",
        encoder=_ENCODER_NAME,
        in_channels=in_channels,
        classes=classes,
        atrous_rates=rates,
        aspp_separable=True,
        encoder_weights=encoder_weights,
    )
    return model


class DiceCrossEntropyLoss(nn.Module):
    """Perdida combinada Dice + CrossEntropy para segmentacion multiclase.

    Suma ponderada de :class:`segmentation_models_pytorch.losses.DiceLoss`
    (modo ``multiclass``, sobre logits) y
    :class:`torch.nn.CrossEntropyLoss`. Ambos terminos ignoran ``ignore_index``
    (Background/Void). CrossEntropy admite pesos por clase opcionales para
    contrarrestar el desbalance de cultivos.

    El termino total es ``dice_weight * dice + ce_weight * ce``. Dice estabiliza
    clases minoritarias (region overlap) y CrossEntropy aporta gradiente por
    pixel; la combinacion es estandar en segmentacion de cultivos.

    Attributes:
        dice: Termino Dice multiclase sobre logits.
        ce: Termino CrossEntropy por pixel.
        dice_weight: Peso del termino Dice.
        ce_weight: Peso del termino CrossEntropy.
    """

    def __init__(
        self,
        ignore_index: int = _DEFAULT_IGNORE_INDEX,
        n_classes: int = 18,
        class_weights: torch.Tensor | Sequence[float] | None = None,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
    ) -> None:
        """Inicializa la perdida combinada.

        Args:
            ignore_index: Etiqueta a ignorar en ambos terminos (Background/Void).
            n_classes: Numero de clases del logit denso. Solo se usa para
                validar la forma de ``class_weights``.
            class_weights: Pesos por clase para CrossEntropy. Tensor o secuencia
                de longitud ``n_classes``, o ``None`` (sin ponderar).
            dice_weight: Peso del termino Dice en la suma.
            ce_weight: Peso del termino CrossEntropy en la suma.

        Raises:
            ValueError: Si ``class_weights`` no tiene longitud ``n_classes``.
        """
        super().__init__()

        weight_tensor: torch.Tensor | None
        if class_weights is None:
            weight_tensor = None
        else:
            weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32)
            if weight_tensor.numel() != n_classes:
                raise ValueError(
                    "class_weights debe tener longitud n_classes "
                    f"({n_classes}), recibido {weight_tensor.numel()}"
                )

        self.dice = DiceLoss(
            mode="multiclass",
            from_logits=True,
            ignore_index=ignore_index,
        )
        # ``weight`` is registered as a buffer inside CrossEntropyLoss and
        # moves with ``.to(device)`` along with the parent module.
        self.ce = nn.CrossEntropyLoss(
            weight=weight_tensor,
            ignore_index=ignore_index,
        )
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calcula la perdida combinada.

        Args:
            logits: Logits densos ``(B, C, H, W)`` (sin softmax).
            target: Etiquetas por pixel ``(B, H, W)`` de tipo entero, con
                ``ignore_index`` en pixeles a omitir.

        Returns:
            Escalar ``torch.Tensor`` con la perdida combinada ponderada.
        """
        target_long = target.long()
        dice_term = self.dice(logits, target_long)
        ce_term = self.ce(logits, target_long)
        # A batch/patch entirely ``ignore_index`` (Background/Void) leaves
        # CrossEntropyLoss averaging over zero valid pixels -> NaN, which would
        # poison training. It is neutralized to zero while preserving the
        # graph (``nan_to_num`` keeps the gradient of the valid paths).
        ce_term = torch.nan_to_num(ce_term, nan=0.0)
        return self.dice_weight * dice_term + self.ce_weight * ce_term


def build_dice_ce_loss(
    ignore_index: int = _DEFAULT_IGNORE_INDEX,
    n_classes: int = 18,
    class_weights: torch.Tensor | Sequence[float] | None = None,
    dice_weight: float = 1.0,
    ce_weight: float = 1.0,
) -> DiceCrossEntropyLoss:
    """Construye la perdida combinada Dice + CrossEntropy ponderada.

    Args:
        ignore_index: Etiqueta a ignorar (Background/Void). Por defecto 255.
        n_classes: Numero de clases del logit denso. Por defecto 18.
        class_weights: Pesos por clase opcionales para CrossEntropy (longitud
            ``n_classes``) o ``None``.
        dice_weight: Peso del termino Dice en la suma.
        ce_weight: Peso del termino CrossEntropy en la suma.

    Returns:
        Instancia de :class:`DiceCrossEntropyLoss` lista para usar.
    """
    return DiceCrossEntropyLoss(
        ignore_index=ignore_index,
        n_classes=n_classes,
        class_weights=class_weights,
        dice_weight=dice_weight,
        ce_weight=ce_weight,
    )
