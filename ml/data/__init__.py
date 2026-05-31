"""DataLoaders y datasets de PyTorch para AgroSatCopilot.

Este paquete aloja los wrappers ``torch.utils.data.Dataset`` que adaptan los
artefactos en disco (PASTIS-R, AlphaEarth, etc.) al contrato de entrenamiento
de los modelos de segmentacion densa (EPIC 5) y los ensembles (EPIC 6).
"""

from __future__ import annotations

from ml.data.pastis_seg_dataset import PASTISSegmentationDataset

__all__ = ["PASTISSegmentationDataset"]
