"""Feature engineering canónico de AgroSatCopilot (EPIC 3).

Exporta la API pública del módulo :mod:`ml.features.spectral_indices`.
"""

from ml.features.spectral_indices import (
    INDEX_NAMES,
    compute_index,
    compute_index_cached,
    compute_index_ee,
    compute_index_timeseries,
)
from ml.features.temporal_features import (
    DEFAULT_FFT_INDICES,
    DEFAULT_INDICES,
    extract_temporal_features,
)

__all__ = [
    "DEFAULT_FFT_INDICES",
    "DEFAULT_INDICES",
    "INDEX_NAMES",
    "compute_index",
    "compute_index_cached",
    "compute_index_ee",
    "compute_index_timeseries",
    "extract_temporal_features",
]
