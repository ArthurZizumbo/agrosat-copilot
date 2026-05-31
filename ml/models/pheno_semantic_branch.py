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

#: Dimension de los prototipos de texto fenologico (``all-MiniLM-L6-v2``).
_PROTOTYPE_DIM = 384

#: Maximo de pixeles validos muestreados por batch para el contraste. Acota la
#: matriz de similitud (n_sampled x num_classes) y el grafo de autograd a algo
#: que cabe en L4 24GB; el muestreo es estocastico por step (ver Wen §3.3).
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

        # Proyeccion al espacio comun de alineacion (rama semantica del paper,
        # §3.2; reemplaza el GCN de keywords por una lineal por simplicidad).
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

    # (B, D, H, W) -> (B*H*W, D) y target -> (B*H*W,)
    feats = visual_proj.permute(0, 2, 3, 1).reshape(-1, dim)  # (P, D)
    labels = target.reshape(-1).to(device)  # (P,)

    valid = (labels != ignore_index) & (labels >= 0) & (labels < num_classes)
    if not bool(valid.any()):
        return visual_proj.sum() * 0.0

    feats = feats[valid]
    labels = labels[valid].long()

    # Submuestreo estocastico para acotar memoria.
    n_valid = feats.shape[0]
    if n_valid > max_pixels:
        perm = torch.randperm(n_valid, device=device, generator=generator)
        idx = perm[:max_pixels]
        feats = feats[idx]
        labels = labels[idx]

    # Contraste indefinido con una sola clase presente.
    if labels.unique().numel() < 2:
        return visual_proj.sum() * 0.0

    feats = F.normalize(feats, p=2, dim=-1)  # (S, D)
    protos = F.normalize(prototypes.to(device), p=2, dim=-1)  # (K, D)

    # Logits pixel-prototipo: (S, K).
    logits = (feats @ protos.t()) / temperature

    # --- L_v: visual -> semantico (clasificacion del pixel a su prototipo) ---
    loss_v = F.cross_entropy(logits, labels)

    # --- L_s: semantico -> visual (cada prototipo presente atrae sus pixeles) -
    # Transpuesta: (K, S). Para cada clase presente, los positivos son los
    # pixeles de esa clase; se usa log-mean-exp de los positivos (multi-positivo
    # estilo InfoNCE supervisado, robusto al numero variable de positivos).
    logits_s = logits.t()  # (K, S)
    log_prob_s = F.log_softmax(logits_s, dim=1)  # (K, S) sobre los pixeles
    present = torch.unique(labels)
    pos_mask = present.unsqueeze(1) == labels.unsqueeze(0)  # (K_present, S)
    log_prob_present = log_prob_s[present]  # (K_present, S)
    # Media de log-prob sobre los pixeles positivos de cada clase presente.
    pos_counts = pos_mask.sum(dim=1).clamp(min=1)  # (K_present,)
    pos_log_prob = (log_prob_present * pos_mask).sum(dim=1) / pos_counts
    loss_s = -pos_log_prob.mean()

    return 0.5 * (loss_v + loss_s)


# TODO(post-Avance): rama de texto con GCN sobre keywords fenologicos (Wen et al.
# 2025, §3.2). El paper construye un grafo sobre los terminos fenologicos
# extraidos de las descripciones (siembra/emergencia/pico/senescencia/cosecha) y
# propaga con una GCN antes de la proyeccion, en lugar de la lineal directa de
# :class:`PhenoSemanticBranch`. Aqui se usa la proyeccion lineal por simplicidad
# y por mantener el entrenamiento viable en la ventana L4; el GCN es una mejora
# de fidelidad al paper que no bloquea el Avance (riesgo/beneficio diferido).
