-- migrate:up
-- US-015: features_parcels — temporal + spectral aggregates per (parcel, year).
-- - alphaearth_embedding nullable: populated by US-016 fusion downstream.
-- - ndvi_stats / phenology JSONB hold the 153 stats + harmonic block; the
--   scalar phenology columns mirror commonly queried metrics for dashboards.
-- - UNIQUE (parcel_id, year): one row per parcel/year. Multi-cycle crops will
--   add a `season` column via rollforward migration in a future US.
-- The `vector` extension is already created by the initial schema migration.
CREATE TABLE IF NOT EXISTS features_parcels (
    id BIGSERIAL PRIMARY KEY,
    parcel_id BIGINT NOT NULL REFERENCES parcels(id) ON DELETE CASCADE,
    year SMALLINT NOT NULL,
    alphaearth_embedding VECTOR(64),
    ndvi_stats JSONB NOT NULL DEFAULT '{}'::jsonb,
    phenology JSONB NOT NULL DEFAULT '{}'::jsonb,
    sog_doy SMALLINT,
    peak_doy SMALLINT,
    peak_value REAL,
    senescence_doy SMALLINT,
    ndvi_auc REAL,
    ndvi_slope_pre_peak REAL,
    ndvi_slope_post_peak REAL,
    maturity_duration_days SMALLINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT features_parcels_parcel_year_uniq UNIQUE (parcel_id, year)
);
CREATE INDEX features_parcels_parcel_id_idx ON features_parcels(parcel_id);
CREATE INDEX features_parcels_year_idx ON features_parcels(year);

-- migrate:down
DROP TABLE IF EXISTS features_parcels CASCADE;
