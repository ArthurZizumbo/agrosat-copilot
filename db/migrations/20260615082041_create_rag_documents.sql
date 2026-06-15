-- migrate:up
-- US-046: rag_documents — corpus for the Spatial-RAG "lite" retrieval layer.
-- This lite variant grounds the reasoner on the real PASTIS-R phenology corpus
-- (parcel captions) plus scene metadata. The hybrid pipeline runs in series:
--   1. PostGIS ST_DWithin spatial pre-filter against `geom`,
--   2. pgvector cosine similarity over the AlphaEarth 64-dim `embedding`,
--   3. weighted fusion of spatial proximity and semantic distance.
-- Intentionally uses the already-persisted AlphaEarth 64-dim vector (NOT
-- e5-mistral 4096-dim) and NO ANN index. The full Spatial-RAG (e5-mistral 4096
-- + HNSW + reranking) is FUTURE; the lite layer relies on a flat cosine scan
-- over a small, spatially pre-filtered candidate set.
-- The `vector` and PostGIS extensions are already created by the initial schema.
CREATE TABLE IF NOT EXISTS rag_documents (
    id BIGSERIAL PRIMARY KEY,
    parcel_id TEXT,                       -- composite corpus id (nullable for scene-level docs)
    geom GEOMETRY(GEOMETRY,4326),         -- point or polygon: patch centroid or parcel geometry
    content TEXT NOT NULL,                -- phenology description / scene metadata text
    source TEXT NOT NULL,                 -- 'phenology_caption' | 'scene_meta'
    embedding VECTOR(64),                 -- AlphaEarth 64-dim (lite RAG; 4096 + HNSW is FUTURE)
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- GIST powers the ST_DWithin spatial pre-filter (step 1 of the hybrid pipeline).
CREATE INDEX IF NOT EXISTS rag_documents_geom_idx ON rag_documents USING GIST (geom);
-- BTREE on source to slice the corpus by document kind.
CREATE INDEX IF NOT EXISTS rag_documents_source_idx ON rag_documents(source);
-- NOTE: no HNSW/IVFFlat index over `embedding` on purpose. The lite layer uses a
-- plain cosine scan over a small candidate set already narrowed by ST_DWithin;
-- an ANN index belongs to the FUTURE full Spatial-RAG (e5-mistral 4096-dim).

-- migrate:down
DROP TABLE IF EXISTS rag_documents CASCADE;
