-- migrate:up
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
-- pgstac requires the pgstac extension installed in the container image; see infrastructure/docker/postgres.Dockerfile
-- CREATE EXTENSION IF NOT EXISTS pgstac;

CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    llm_variant TEXT NOT NULL CHECK (llm_variant IN ('gemini', 'qwen35')) DEFAULT 'gemini',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chat_sessions_user_id_idx ON chat_sessions(user_id);

CREATE TABLE IF NOT EXISTS aois (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    geom GEOMETRY(POLYGON, 4326) NOT NULL,
    label TEXT,
    area_ha REAL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS aois_geom_idx ON aois USING GIST (geom);
CREATE INDEX IF NOT EXISTS aois_session_id_idx ON aois(session_id);

-- migrate:down
DROP TABLE IF EXISTS aois CASCADE;
DROP TABLE IF EXISTS chat_sessions CASCADE;
DROP EXTENSION IF EXISTS vector;
DROP EXTENSION IF EXISTS postgis_topology;
DROP EXTENSION IF EXISTS postgis;
