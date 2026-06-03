"""Rama semantica + loss contrastivo fenologico para TSViT (Wen et al. 2025).

Implementa la **rama semantica** y la **alineacion contrastiva pixel-visual <->
prototipo-de-clase** del metodo de Wen et al. (2025), "Phenology Description is
All You Need!" (ISPRS J. Photogrammetry RS 228), ecuaciones 15-16. A diferencia
del baseline tabular (que concatenaba el ``pheno_text`` como columnas extra y
degradaba), el paper **no concatena**: alinea por contraste la feature visual de
cada pixel con el prototipo semantico de SU clase (positivo) frente a los demas
17 prototipos (negativos). Ablacion del paper (Tabla 2, zero-shot area 1): solo
patches F1 26.8 -> patches + fenologia F1 53.4.

Componentes:

1. :class:`PhenoSemanticBranch`: carga la matriz de 18 prototipos por clase
   (384-dim) de :func:`ml.features.phenology_class_prototypes.\
   load_class_prototype_embeddings`, los proyecta a un espacio comun
   (``Linear 384 -> semantic_dim``) y los normaliza L2. Expone
   :meth:`get_class_prototypes` que devuelve la matriz ``(num_classes, D)`` lista
   para el contraste. La proyeccion al espacio comun reemplaza la rama de texto
   con GCN del paper (§3.2); el GCN sobre keywords fenologicos queda como TODO
   post-Avance (ver nota al final del modulo).

2. :func:`phenology_contrastive_loss`: InfoNCE simetrico estilo CLIP (ec. 15-16,
   ``L_cl = (L_v + L_s)/2``) entre las features visuales por pixel (de
   :meth:`ml.models.tsvit_wrapper.TSViT.forward` con ``return_visual_proj=True``)
   y los prototipos por clase. Submuestrea pixeles validos para acotar memoria.

Atribucion: metodo de Wen et al. (2025), ISPRS J. Photogrammetry RS 228.
Documentado en ``docs/licenses/DATA_LICENSE.md``.
"""

from __future__ import annotations

from pathlib import Path

import structlog
import torch
import torch.nn.functional as F
from torch import nn

from ml.features.phenology_class_prototypes import (
    load_class_prototype_embeddings,
)

__all__ = [
    "PhenoSemanticBranch",
    "phenology_contrastive_loss",
]

logger = structlog.get_logger(__name__)

#: Dimension of the phenological text prototypes (``all-MiniLM-L6-v2``).
_PROTOTYPE_DIM = 384

#: Maximum number of valid pixels sampled per batch for the contrast. Bounds the
#: similarity matrix (n_sampled x num_classes) and the autograd graph to something
#: that fits in L4 24GB; the sampling is stochastic per step (see Wen §3.3).
_DEFAULT_MAX_PIXELS = 4096


class PhenoSemanticBranch(nn.Module):
    """Rama semantica: prototipos fenologicos por clase proyectados y L2-norm.

    Carga la matriz de prototipos textuales por clase (``num_classes``, 384) y la
    proyecta a un espacio comun de dimension ``semantic_dim`` mediante una capa
    lineal aprendible, normalizando L2 la salida. La feature visual por pixel de
    TSViT (``return_visual_proj=True``) vive en este mismo espacio, lo que
    permite la alineacion contrastiva de la ecuacion 15-16 del paper.

    Los prototipos crudos se registran como buffer (no entrenable, viaja con el
    ``state_dict`` y al dispositivo del modulo); solo la proyeccion es
    aprendible. Esto preserva la semantica del encoder de texto congelado y deja
    que el modelo aprenda unicamente como mapear esa semantica al espacio visual.

    Args:
        semantic_dim: Dimension del espacio comun de alineacion. Debe coincidir
            con ``semantic_dim`` de :class:`ml.models.tsvit_wrapper.TSViT`
            (384 por defecto, igual que los prototipos, lo que hace de la
            proyeccion un refinamiento y no un cambio de dimensionalidad).
        prototype_path: Ruta al parquet de prototipos por clase. Si es ``None``
            usa el default de
            :func:`ml.features.phenology_class_prototypes.\
            load_class_prototype_embeddings`.
        freeze_prototypes: Si ``True`` (defecto) los prototipos crudos son un
            buffer no entrenable; si ``False`` se registran como parametro y se
            afinan junto a la proyeccion.

    Raises:
        ValueError: Si la matriz de prototipos cargada no tiene dimension 384.
    """

    def __init__(
        self,
        semantic_dim: int = _PROTOTYPE_DIM,
        prototype_path: Path | None = None,
        *,
        freeze_prototypes: bool = True,
    ) -> None:
        super().__init__()
        if prototype_path is None:
            prototypes_np, class_ids = load_class_prototype_embeddings()
        else:
            prototypes_np, class_ids = load_class_prototype_embeddings(
                prototype_path
            )
        if prototypes_np.shape[1] != _PROTOTYPE_DIM:
            raise ValueError(
                f"Los prototipos deben ser {_PROTOTYPE_DIM}-dim; se cargo "
                f"shape {prototypes_np.shape}."
            )

        self.num_classes = int(prototypes_np.shape[0])
        self.semantic_dim = semantic_dim
        self.class_ids = list(class_ids)

        prototypes = torch.from_numpy(prototypes_np).float()  # (K, 384)
        if freeze_prototypes:
            self.register_buffer("raw_prototypes", prototypes)
        else:
            self.raw_prototypes = nn.Parameter(prototypes)

        # Projection to the common alignment space (semantic branch of the paper,
        # §3.2; replaces the keyword GCN with a linear layer for simplicity).
        self.proj = nn.Linear(_PROTOTYPE_DIM, semantic_dim)

        logger.info(
            "pheno_semantic_branch_init",
            num_classes=self.num_classes,
            semantic_dim=semantic_dim,
            freeze_prototypes=freeze_prototypes,
        )

    def get_class_prototypes(self) -> torch.Tensor:
        """Devuelve los prototipos proyectados y L2-normalizados.

        Returns:
            Tensor ``(num_classes, semantic_dim)`` float, normalizado por fila
            (norma L2 unitaria), en el dispositivo del modulo. Listo para usarse
            como ``prototypes`` en :func:`phenology_contrastive_loss`.
        """
        projected = self.proj(self.raw_prototypes)  # (K, semantic_dim)
        return F.normalize(projected, p=2, dim=-1)

    def forward(self) -> torch.Tensor:
        """Alias de :meth:`get_class_prototypes` (interfaz ``nn.Module``).

        Returns:
            Prototipos proyectados y L2-normalizados ``(num_classes,
            semantic_dim)``.
        """
        return self.get_class_prototypes()


def phenology_contrastive_loss(
    visual_proj: torch.Tensor,
    target: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    ignore_index: int = 255,
    temperature: float = 0.07,
    max_pixels: int = _DEFAULT_MAX_PIXELS,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Loss contrastivo fenologico InfoNCE simetrico (Wen et al. 2025, ec 15-16).

    Para cada pixel valido (``target != ignore_index``) alinea su feature visual
    con el prototipo de SU clase (positivo) frente a los otros prototipos
    (negativos). Se computan dos terminos InfoNCE simetricos estilo CLIP y se
    promedian (``L_cl = (L_v + L_s)/2``):

    - ``L_v`` (visual->semantico): para cada pixel, softmax sobre las
      similitudes con los ``num_classes`` prototipos; el positivo es el
      prototipo de su clase.
    - ``L_s`` (semantico->visual): para cada clase **presente** en el batch
      muestreado, softmax sobre las similitudes con todos los pixeles
      muestreados; los positivos son los pixeles de esa clase (multi-positivo,
      log-mean-exp de los positivos).

    Las features visuales y los prototipos se L2-normalizan, la similitud es el
    producto punto dividido por ``temperature``. Si los pixeles validos superan
    ``max_pixels`` se submuestrean de forma estocastica para acotar la matriz de
    similitud y el grafo de autograd (memoria de L4).

    Args:
        visual_proj: Features visuales por pixel ``(B, D, H, W)`` (de TSViT con
            ``return_visual_proj=True``).
        target: Clase por pixel ``(B, H, W)`` int. Los valores en ``[0,
            num_classes)`` indexan ``prototypes``; ``ignore_index`` se descarta.
        prototypes: Prototipos por clase ``(num_classes, D)``; idealmente ya
            L2-normalizados (se renormalizan por robustez).
        ignore_index: Valor de ``target`` a ignorar (Background/Void).
        temperature: Temperatura ``tau`` del softmax InfoNCE (0.07, estandar
            CLIP/SimCLR).
        max_pixels: Maximo de pixeles validos a muestrear por llamada.
        generator: ``torch.Generator`` opcional para el muestreo determinista
            (tests/smoke); si ``None`` usa el RNG global.

    Returns:
        Loss escalar ``(L_v + L_s)/2``. Devuelve ``0.0`` (con grafo) si no hay
        pixeles validos o si solo hay una clase presente (contraste indefinido).
    """
    if visual_proj.dim() != 4:
        raise ValueError(
            f"visual_proj debe ser (B, D, H, W); se recibio {tuple(visual_proj.shape)}."
        )
    num_classes, dim = prototypes.shape
    device = visual_proj.device

    # (B, D, H, W) -> (B*H*W, D) and target -> (B*H*W,)
    feats = visual_proj.permute(0, 2, 3, 1).reshape(-1, dim)  # (P, D)
    labels = target.reshape(-1).to(device)  # (P,)

    valid = (labels != ignore_index) & (labels >= 0) & (labels < num_classes)
    if not bool(valid.any()):
        return visual_proj.sum() * 0.0

    feats = feats[valid]
    labels = labels[valid].long()

    # Stochastic subsampling to bound memory.
    n_valid = feats.shape[0]
    if n_valid > max_pixels:
        perm = torch.randperm(n_valid, device=device, generator=generator)
        idx = perm[:max_pixels]
        feats = feats[idx]
        labels = labels[idx]

    # Contrast undefined with a single class present.
    if labels.unique().numel() < 2:
        return visual_proj.sum() * 0.0

    feats = F.normalize(feats, p=2, dim=-1)  # (S, D)
    protos = F.normalize(prototypes.to(device), p=2, dim=-1)  # (K, D)

    # Pixel-prototype logits: (S, K).
    logits = (feats @ protos.t()) / temperature

    # --- L_v: visual -> semantic (classify the pixel to its prototype) ---
    loss_v = F.cross_entropy(logits, labels)

    # --- L_s: semantic -> visual (each present prototype attracts its pixels) -
    # Transpose: (K, S). For each present class, the positives are the
    # pixels of that class; log-mean-exp of the positives is used (multi-positive
    # supervised-InfoNCE style, robust to the variable number of positives).
    logits_s = logits.t()  # (K, S)
    log_prob_s = F.log_softmax(logits_s, dim=1)  # (K, S) over the pixels
    present = torch.unique(labels)
    pos_mask = present.unsqueeze(1) == labels.unsqueeze(0)  # (K_present, S)
    log_prob_present = log_prob_s[present]  # (K_present, S)
    # Mean log-prob over the positive pixels of each present class.
    pos_counts = pos_mask.sum(dim=1).clamp(min=1)  # (K_present,)
    pos_log_prob = (log_prob_present * pos_mask).sum(dim=1) / pos_counts
    loss_s = -pos_log_prob.mean()

    return 0.5 * (loss_v + loss_s)


# TODO(post-Avance): text branch with a GCN over phenological keywords (Wen et al.
# 2025, §3.2). The paper builds a graph over the phenological terms extracted from
# the descriptions (sowing/emergence/peak/senescence/harvest) and propagates with
# a GCN before the projection, instead of the direct linear layer of
# :class:`PhenoSemanticBranch`. Here the linear projection is used for simplicity
# and to keep training viable within the L4 window; the GCN is a paper-fidelity
# improvement that does not block the Avance (deferred risk/benefit trade-off).
