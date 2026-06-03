"""Modulo de analisis exploratorio para US-010/011/012/018."""

from __future__ import annotations

# Correlations, embeddings, and paper methods depend on heavy packages
# (umap, statsmodels, etc.) that are not always installed, for example in a
# segmentation environment on Colab. They are imported tolerantly: the HCAT mapping
# (hcat_grouping) is a lightweight submodule that does not need this.
try:
    from ml.analysis.correlations import (
        SPECTRAL_INDICES_CORE,
        acf_pacf_per_parcel,
        compute_indices_subset,
        correlation_pair,
        dtw_cluster_temporal,
        era5_ndvi_anomaly,
        phenology_peaks,
        vif_table,
    )
    from ml.analysis.embeddings import (
        DIM_COLS,
        compare_alphaearth_vs_ndvi,
        correlation_matrix,
        cross_region_consistency,
        qq_test_dims,
        rf_feature_importance,
        temporal_stability,
        tsne_2d,
        umap_2d,
    )
    from ml.analysis.paper_methods import (
        aggregate_rare_classes,
        boundary_interior_stats,
        boundary_pixel_mask,
        cloud_gap_robustness,
        compute_boundary_ratio,
        confusion_symmetry_analysis,
        phenology_calendar_features,
        temporal_sampling_stats,
    )

    __all__ = [
        "DIM_COLS",
        "SPECTRAL_INDICES_CORE",
        "acf_pacf_per_parcel",
        "aggregate_rare_classes",
        "boundary_interior_stats",
        "boundary_pixel_mask",
        "cloud_gap_robustness",
        "compare_alphaearth_vs_ndvi",
        "compute_boundary_ratio",
        "compute_indices_subset",
        "confusion_symmetry_analysis",
        "correlation_matrix",
        "correlation_pair",
        "cross_region_consistency",
        "dtw_cluster_temporal",
        "era5_ndvi_anomaly",
        "phenology_calendar_features",
        "phenology_peaks",
        "qq_test_dims",
        "rf_feature_importance",
        "temporal_sampling_stats",
        "temporal_stability",
        "tsne_2d",
        "umap_2d",
        "vif_table",
    ]
except ImportError:  # pragma: no cover - optional dependencies absent
    __all__ = []
