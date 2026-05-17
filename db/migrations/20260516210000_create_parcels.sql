-- migrate:up
-- US-015: parcels table (minimal superset compatible with future segmentation US).
-- Holds inferred / ingested agricultural parcel polygons. Schema deliberately
-- minimal: downstream features live in `features_parcels` (FK parcel_id).
-- No CHECK constraints on optional columns to keep schema extensible.
CREATE TABLE IF NOT EXISTS parcels (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
    aoi_id BIGINT REFERENCES aois(id) ON DELETE SET NULL,
    geom GEOMETRY(POLYGON, 4326) NOT NULL,
    crop_class TEXT,
    confidence REAL,
    area_ha REAL,
    year SMALLINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX parcels_geom_idx ON parcels USING GIST (geom);
CREATE INDEX parcels_session_id_idx ON parcels(session_id);
CREATE INDEX parcels_year_idx ON parcels(year);

-- migrate:down
DROP TABLE IF EXISTS parcels CASCADE;
