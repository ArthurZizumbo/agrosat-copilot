"""Feature engineering canónico de AgroSatCopilot (EPIC 3).

Exporta la API pública de :mod:`ml.features.spectral_indices` (US-014),
:mod:`ml.features.temporal_features` (US-015), los módulos de fusión
multisensor + spatial K-fold + scaler de US-016
(:mod:`ml.features.fusion`, :mod:`ml.features.spatial_split`,
:mod:`ml.features.scaler`), y la suite de selección/normalización de US-018
(:mod:`ml.features.selection`).
"""

from ml.features.encoding import (
    derive_crop_group_from_class_id,
    derive_season_from_doy,
    encode_onehot,
    encode_ordinal,
    encode_target_mean,
)
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
from ml.features.selection import (
    anova_f_select,
    apply_variance_threshold,
    chi2_select,
    compare_before_after,
    compute_feature_importance,
    discretize_features,
    discretize_ndvi_phenology_domain,
    drop_correlated_features,
    fit_factor_analysis,
    fit_pca,
    fit_umap_2d,
    make_preprocessor,
    select_normalizer,
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
    "anova_f_select",
    "apply_variance_threshold",
    "build_fused_features",
    "build_spatial_kfold",
    "chi2_select",
    "compare_before_after",
    "compute_feature_importance",
    "compute_index",
    "compute_index_cached",
    "compute_index_ee",
    "compute_index_timeseries",
    "derive_crop_group_from_class_id",
    "derive_season_from_doy",
    "discretize_features",
    "discretize_ndvi_phenology_domain",
    "drop_correlated_features",
    "encode_onehot",
    "encode_ordinal",
    "encode_target_mean",
    "extract_temporal_features",
    "fit_factor_analysis",
    "fit_pca",
    "fit_scaler_on_train",
    "fit_umap_2d",
    "load_scaler",
    "make_preprocessor",
    "select_normalizer",
]
