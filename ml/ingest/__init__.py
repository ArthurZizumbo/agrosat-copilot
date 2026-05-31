"""Helpers de ingestión de datos satelitales y agronómicos (EPIC 2).

Re-exports de los samplers usados por feature engineering downstream
(US-014/015/016) y la API canónica de bandas Sentinel-2 (PASTIS).
"""

# Los samplers de Google Earth Engine dependen de paquetes pesados (earthengine,
# geopandas) que no siempre estan instalados, por ejemplo en un entorno de
# segmentacion en Colab. Se importan de forma tolerante: los loaders de PASTIS
# (pastis_loader, pastis_dataset) son submodulos directos y no necesitan esto.
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
except ImportError:  # pragma: no cover - dependencias opcionales ausentes
    __all__ = []
