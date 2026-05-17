"""Feature engineering canónico de AgroSatCopilot (EPIC 3).

Exporta la API pública de :mod:`ml.features.spectral_indices` (US-014),
:mod:`ml.features.temporal_features` (US-015), y los módulos de fusión
multisensor + spatial K-fold + scaler de US-016
(:mod:`ml.features.fusion`, :mod:`ml.features.spatial_split`,
:mod:`ml.features.scaler`).
"""

from ml.features.fusion import (
    AE_COLS,
    BLOCK_NAMES,
    EXPECTED_COL_COUNT_NO_FARSLIP,
    EXPECTED_COL_COUNT_WITH_FARSLIP,
    FUSION_STATS,
    build_fused_features,
)
from ml.features.scaler import (
    fit_scaler_on_train,
    load_scaler,
)
from ml.features.spatial_split import (
    FoldAssignment,
    build_spatial_kfold,
)
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
    "AE_COLS",
    "BLOCK_NAMES",
    "DEFAULT_FFT_INDICES",
    "DEFAULT_INDICES",
    "EXPECTED_COL_COUNT_NO_FARSLIP",
    "EXPECTED_COL_COUNT_WITH_FARSLIP",
    "FUSION_STATS",
    "INDEX_NAMES",
    "FoldAssignment",
    "build_fused_features",
    "build_spatial_kfold",
    "compute_index",
    "compute_index_cached",
    "compute_index_ee",
    "compute_index_timeseries",
    "extract_temporal_features",
    "fit_scaler_on_train",
    "load_scaler",
]
