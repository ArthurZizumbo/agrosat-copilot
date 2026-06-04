"""Download AlphaEarth 2018 over the 85951 PASTIS-R parcels.

The PASTIS-R growing season crosses two calendar years (first S2 image
~17-sep-2018, last ~oct-2019). The annual AlphaEarth-2019 embedding is already
downloaded (``alphaearth_parcels_parcels_2019_85951.parquet``); this script
adds the 2018 embedding (sowing/emergence) so both can be concatenated
(64+64 dims) and the model sees the full cycle.

Uses the same sampling function that produced 2019
(``sample_alphaearth_for_parcels``) with ``year=2018``, so the cache is named
``alphaearth_parcels_parcels_2018_85951.parquet`` automatically.

Usage:
    python scripts/download_alphaearth_2018_pastis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import structlog

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.ingest.gee_sampler import init_ee, sample_alphaearth_for_parcels  # noqa: E402

log = structlog.get_logger(__name__)

PARCELS_PATH = REPO_ROOT / "data" / "processed" / "pastis_parcels_full.geoparquet"
TARGET_YEAR = 2018
EXPECTED_PARCELS = 85951


def main() -> int:
    parcels = gpd.read_parquet(PARCELS_PATH)
    log.info("parcels_loaded", n=len(parcels), crs=str(parcels.crs))
    if len(parcels) != EXPECTED_PARCELS:
        log.warning("parcels_count_unexpected", got=len(parcels), expected=EXPECTED_PARCELS)

    init_ee()
    log.info("gee_initialized", year=TARGET_YEAR)

    df = sample_alphaearth_for_parcels(parcels, year=TARGET_YEAR, cache_key="parcels")
    log.info("alphaearth_2018_sampled", shape=tuple(df.shape))

    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    nan_pct = df.select(dim_cols).null_count().to_numpy().sum() / (len(df) * len(dim_cols)) * 100
    log.info("alphaearth_2018_quality", n_dim=len(dim_cols), pct_null=round(nan_pct, 3))

    if len(df) == 0:
        log.error("alphaearth_2018_empty_gee_failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
