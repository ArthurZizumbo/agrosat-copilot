"""AnySat frozen + linear head para segmentacion densa (#6, Avance 4).

AnySat (Astruc et al., 2024, IGN — "AnySat: An Earth Observation Model for Any
Resolutions, Scales, and Modalities") es un foundation model multimodal/multi-
temporal que se carga via ``torch.hub`` desde ``gastruc/anysat``. Aqui se usa el
encoder **congelado** como extractor de features densas y se entrena unicamente
una **cabeza lineal** (Conv 1x1) que proyecta esas features a las 20 clases
PASTIS-R y las upsamplea a la resolucion objetivo. Es el setup mas barato del
reparto (encoder congelado -> solo gradientes en la cabeza, ~2-3 h L4).

Diseno defensivo: la integracion con la API exacta de AnySat (``output='dense'``)
queda aislada en :meth:`AnySatSegmenter._encode` y el encoder es **inyectable**,
de modo que los tests corran con un encoder sinteptico sin descargar pesos. La
carga real via ``torch.hub`` y la firma exacta del forward se validan en la celda
dedicada del notebook Colab.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = structlog.get_logger(__name__)

__all__ = ["AnySatSegmenter", "load_anysat_encoder"]

_HUB_REPO = "gastruc/anysat"
_HUB_MODEL = "anysat"


def load_anysat_encoder(
    *,
    repo: str = _HUB_REPO,
    model: str = _HUB_MODEL,
    pretrained: bool = True,
    flash_attn: bool = False,
) -> nn.Module:
    """Carga el encoder AnySat preentrenado via ``torch.hub`` (descarga remota).

    Args:
        repo: Repositorio ``torch.hub`` (default ``gastruc/anysat``).
        model: Entrypoint del hub (default ``anysat``).
        pretrained: Si carga los pesos preentrenados.
        flash_attn: Si habilita FlashAttention (requiere soporte; default False
            para portabilidad en L4/Colab).

    Returns:
        El ``nn.Module`` encoder de AnySat.

    Raises:
        RuntimeError: si la carga via hub falla (sin internet, repo inaccesible).
    """
    try:
        encoder = torch.hub.load(
            repo,
            model,
            pretrained=pretrained,
            flash_attn=flash_attn,
            trust_repo=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"No se pudo cargar AnySat desde torch.hub ({repo}). "
            "Requiere internet y el repo accesible. En Colab ejecutar la celda de "
            "setup de AnySat antes de instanciar AnySatSegmenter."
        ) from exc
    logger.info("anysat_encoder_loaded", repo=repo, pretrained=pretrained)
    return encoder


class AnySatSegmenter(nn.Module):
    """AnySat congelado + cabeza lineal Conv 1x1 para segmentacion densa.

    Forward: ``image (B, T, C, H, W)`` (+ ``dates (B, T)``) -> ``logits
    (B, num_classes, target_size, target_size)``. El encoder se ejecuta sin
    gradientes (congelado); solo la cabeza ``head`` se entrena.

    La cabeza usa ``nn.LazyConv2d`` cuando ``feature_dim`` es desconocido, de modo
    que el numero de canales de las features densas de AnySat se infiere en el
    primer forward (la cabeza se materializa entonces y es entrenable).
    """

    def __init__(
        self,
        num_classes: int,
        *,
        target_size: int = 256,
        patch_size: int = 10,
        modality: str = "s2",
        feature_dim: int | None = None,
        encoder: nn.Module | Callable[..., Any] | None = None,
        freeze: bool = True,
    ) -> None:
        """Inicializa el segmentador.

        Args:
            num_classes: Numero de clases de salida (20 en PASTIS-R).
            target_size: Lado espacial de los logits de salida.
            patch_size: ``patch_size`` que AnySat usa para la granularidad densa.
            modality: Clave de modalidad en el dict de entrada de AnySat (``s2``).
            feature_dim: Dimension de las features densas de AnySat. Si ``None``
                se infiere en el primer forward via ``LazyConv2d``.
            encoder: Encoder inyectable (para tests). Si ``None`` se carga AnySat
                via ``torch.hub`` con :func:`load_anysat_encoder`.
            freeze: Si congela el encoder (default ``True``).
        """
        super().__init__()
        self.num_classes = num_classes
        self.target_size = target_size
        self.patch_size = patch_size
        self.modality = modality
        self._frozen = freeze

        self.encoder = encoder if encoder is not None else load_anysat_encoder()
        if freeze and isinstance(self.encoder, nn.Module):
            self.encoder.requires_grad_(False)
            self.encoder.eval()

        if feature_dim is not None:
            self.head: nn.Module = nn.Conv2d(feature_dim, num_classes, kernel_size=1)
        else:
            self.head = nn.LazyConv2d(num_classes, kernel_size=1)

    def _encode(self, image: torch.Tensor, dates: torch.Tensor | None) -> torch.Tensor:
        """Ejecuta el encoder AnySat y devuelve un mapa de features denso.

        Aisla la firma concreta de AnySat. Construye el dict de modalidades,
        invoca ``output='dense'`` y normaliza la salida a ``(B, D, h, w)``.
        Acepta tambien encoders sinteticos (callables) que ya devuelven
        ``(B, D, h, w)`` directamente (tests).

        Args:
            image: ``(B, T, C, H, W)`` serie temporal Sentinel-2 normalizada.
            dates: ``(B, T)`` dia-del-anio por frame (o ``None``).

        Returns:
            Mapa de features denso ``(B, D, h, w)``.
        """
        data: dict[str, Any] = {self.modality: image}
        if dates is not None:
            data[f"{self.modality}_dates"] = dates

        try:
            feats = self.encoder(data, patch_size=self.patch_size, output="dense")
        except TypeError:
            # Synthetic test encoder: simple signature encoder(image) -> (B, D, h, w).
            return self._to_feature_map(self.encoder(image))

        # AnySat 'dense' returns channels-last (B, H, W, D) at full resolution;
        # it is permuted to channels-first (B, D, H, W), which is what the Conv2d head expects.
        if feats.dim() == 4:
            feats = feats.permute(0, 3, 1, 2).contiguous()
        return self._to_feature_map(feats)

    @staticmethod
    def _to_feature_map(feats: torch.Tensor) -> torch.Tensor:
        """Normaliza la salida del encoder a un mapa espacial ``(B, D, h, w)``.

        AnySat ``output='dense'`` puede devolver ``(B, N, D)`` (tokens) o ya
        ``(B, D, h, w)``. Si llega como tokens cuadrados se reacomoda a mapa.

        Args:
            feats: Salida cruda del encoder.

        Returns:
            Tensor ``(B, D, h, w)``.

        Raises:
            ValueError: si la salida no es interpretable como mapa denso.
        """
        if feats.dim() == 4:
            return feats  # (B, D, h, w)
        if feats.dim() == 3:
            # (B, N, D) -> (B, D, sqrt(N), sqrt(N)) if N is a perfect square.
            b, n, d = feats.shape
            side = round(n**0.5)
            if side * side != n:
                raise ValueError(
                    f"Features densas no cuadradas (N={n}); ajustar patch_size o "
                    "el manejo de tokens en _to_feature_map."
                )
            return feats.transpose(1, 2).reshape(b, d, side, side)
        raise ValueError(f"Forma de features densas no soportada: {tuple(feats.shape)}")

    @torch.no_grad()
    def extract_features(
        self, image: torch.Tensor, dates: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Devuelve el mapa de features denso del encoder congelado ``(B, D, h, w)``.

        Pensado para cachear las features una sola vez y tunear unicamente la cabeza
        lineal sin re-ejecutar el encoder (el cuello de botella en AnySat). No aplica
        la cabeza ni el upsample; el llamador entrena su propia cabeza sobre el cache.

        Args:
            image: ``(B, T, C, H, W)`` serie Sentinel-2 normalizada.
            dates: ``(B, T)`` dia-del-anio por frame (opcional).

        Returns:
            Mapa de features denso ``(B, D, h, w)``.
        """
        return self._encode(image, dates)

    def forward(self, image: torch.Tensor, dates: torch.Tensor | None = None) -> torch.Tensor:
        """Produce logits de segmentacion densos.

        Args:
            image: ``(B, T, C, H, W)`` serie Sentinel-2 normalizada.
            dates: ``(B, T)`` dia-del-anio por frame (opcional).

        Returns:
            Logits ``(B, num_classes, target_size, target_size)``.
        """
        if self._frozen:
            with torch.no_grad():
                feats = self._encode(image, dates)
        else:
            feats = self._encode(image, dates)

        logits = self.head(feats)
        return F.interpolate(
            logits, size=(self.target_size, self.target_size), mode="bilinear", align_corners=False
        )
