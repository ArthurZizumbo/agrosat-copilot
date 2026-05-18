"""Paquete FarSLIP — destilación parche-a-parche CLIP adaptada a Sentinel-2.

Esta US-017 (alias US-016b en el plan v6) implementa el procedimiento descrito
en Li et al. 2025 (arXiv:2511.14901) para adaptar un CLIP ViT-B/16 al dominio
agrícola con crops 256x256 Sentinel-2 y vocabulario CAP italiano.

Re-exporta las clases publicas del modulo. Importar desde paths internos esta
permitido para tests; los consumidores aguas abajo (US-016 / US-025) deben usar
estos re-exports para sostener un contrato estable.
"""

from __future__ import annotations

from ml.farslip.dataset import FarSLIPDataset, build_farslip_pairs
from ml.farslip.distill import (
    FarSLIPDistillationTrainer,
    PatchDistillationLoss,
    RegionCategoryAlignmentLoss,
    adapt_patch_embed_to_n_channels,
)

__all__ = [
    "FarSLIPDataset",
    "FarSLIPDistillationTrainer",
    "PatchDistillationLoss",
    "RegionCategoryAlignmentLoss",
    "adapt_patch_embed_to_n_channels",
    "build_farslip_pairs",
]
