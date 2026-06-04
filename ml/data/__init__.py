"""PyTorch DataLoaders and datasets for AgroSatCopilot.

This package hosts the ``torch.utils.data.Dataset`` wrappers that adapt the
on-disk artifacts (PASTIS-R, AlphaEarth, etc.) to the training contract of
the dense segmentation models (EPIC 5) and the ensembles (EPIC 6).
"""

from __future__ import annotations

from ml.data.pastis_seg_dataset import PASTISSegmentationDataset

__all__ = ["PASTISSegmentationDataset"]
