"""FarSLIP package — CLIP patch-to-patch distillation adapted to Sentinel-2.

This US-017 (alias US-016b in the v6 plan) implements the procedure described in
Li et al. 2025 (arXiv:2511.14901) to adapt a CLIP ViT-B/16 to the agricultural
domain with 256x256 Sentinel-2 crops and Italian CAP vocabulary.

Re-exports the public classes of the module. Importing from internal paths is
allowed for tests; downstream consumers (US-016 / US-025) must use these
re-exports to keep a stable contract.
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
