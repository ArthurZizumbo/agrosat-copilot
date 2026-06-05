"""Satellite and agronomic data ingestion helpers (EPIC 2).

Re-exports of the samplers used by downstream feature engineering
(US-014/015/016) and the canonical Sentinel-2 bands API (PASTIS).
"""

# The Google Earth Engine samplers depend on heavy packages (earthengine,
# geopandas) that are not always installed, for example in a segmentation
# environment in Colab. They are imported tolerantly: the PASTIS loaders
# (pastis_loader, pastis_dataset) are direct submodules and do not need this.
try:
    from ml.ingest.gee_sampler import (
        ALPHAEARTH_COLLECTION,
        ALPHAEARTH_DIM_COLS,
        DYNAMIC_WORLD_CLASSES,
        DYNAMIC_WORLD_COLLECTION,
        ERA5_COLLECTION,
        S1_COLLECTION,
        SRTM_COLLECTION,
        era5_annual_precip,
        fetch_s2_ndvi_rgb_for_parcel,
        init_ee,
        sample_alphaearth_at_coords,
        sample_alphaearth_roi,
        sample_dynamic_world_at,
        sample_era5_monthly_climate,
        sample_s1_roi_for_parcels,
        sample_s2_roi,
        sample_srtm_terrain,
    )

    __all__ = [
        "ALPHAEARTH_COLLECTION",
        "ALPHAEARTH_DIM_COLS",
        "DYNAMIC_WORLD_CLASSES",
        "DYNAMIC_WORLD_COLLECTION",
        "ERA5_COLLECTION",
        "S1_COLLECTION",
        "SRTM_COLLECTION",
        "era5_annual_precip",
        "fetch_s2_ndvi_rgb_for_parcel",
        "init_ee",
        "sample_alphaearth_at_coords",
        "sample_alphaearth_roi",
        "sample_dynamic_world_at",
        "sample_era5_monthly_climate",
        "sample_s1_roi_for_parcels",
        "sample_s2_roi",
        "sample_srtm_terrain",
    ]
except ImportError:  # pragma: no cover - optional dependencies absent
    __all__ = []
