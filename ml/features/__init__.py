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

__all__ = [
    "INDEX_NAMES",
    "compute_index",
    "compute_index_cached",
    "compute_index_ee",
    "compute_index_timeseries",
]
