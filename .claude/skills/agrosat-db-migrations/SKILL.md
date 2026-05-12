---
name: agrosat-db-migrations
description: Create reversible SQL migrations using dbmate for AgroSatCopilot PostgreSQL 15 + PostGIS + pgvector + pgstac. Use when adding tables, columns, indexes (GIST for geometry, IVFFlat/HNSW for vectors), extensions, RLS policies, or seed data.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot DB Migrations Skill

Current migrations: `! ls db/migrations/ 2>/dev/null | head -10`

## Rules — NON-NEGOTIABLE

- `dbmate new <name>` for every change; never edit applied migrations
- Both `-- migrate:up` and `-- migrate:down` sections (reversible)
- Extensions in initial migration: `postgis`, `postgis_topology`, `vector`, `pgstac`
- Geometry: `GEOMETRY(POLYGON, 4326)` with `CREATE INDEX ... USING GIST`
- Vector: `VECTOR(N)` with IVFFlat or HNSW
- Foreign keys: explicit `ON DELETE` and `ON UPDATE`
- Timestamps: `TIMESTAMPTZ DEFAULT now()` for `created_at`, `updated_at`
- Multi-tenant: `session_id UUID NOT NULL` + RLS policy

## Initial Schema

```sql
-- migrate:up
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgstac;

CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    llm_variant TEXT NOT NULL CHECK (llm_variant IN ('gemini', 'qwen35')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX chat_sessions_user_id_idx ON chat_sessions(user_id);

CREATE TABLE aois (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    geom GEOMETRY(POLYGON, 4326) NOT NULL,
    label TEXT,
    area_ha REAL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX aois_geom_idx ON aois USING GIST (geom);
CREATE INDEX aois_session_id_idx ON aois(session_id);

-- migrate:down
DROP TABLE IF EXISTS aois CASCADE;
DROP TABLE IF EXISTS chat_sessions CASCADE;
DROP EXTENSION IF EXISTS pgstac;
DROP EXTENSION IF EXISTS vector;
DROP EXTENSION IF EXISTS postgis_topology;
DROP EXTENSION IF EXISTS postgis;
```

## AlphaEarth Tiles

```sql
-- migrate:up
CREATE TABLE alphaearth_tiles (
    id BIGSERIAL PRIMARY KEY,
    roi_name TEXT NOT NULL,
    year INT NOT NULL CHECK (year BETWEEN 2017 AND 2030),
    bbox GEOMETRY(POLYGON, 4326) NOT NULL,
    storage_uri TEXT NOT NULL,
    size_mb REAL,
    download_date TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (roi_name, year)
);
CREATE INDEX alphaearth_tiles_bbox_idx ON alphaearth_tiles USING GIST (bbox);
CREATE INDEX alphaearth_tiles_year_idx ON alphaearth_tiles(year);
```

## pgvector Index (Spatial-RAG)

```sql
CREATE TABLE rag_documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(4096) NOT NULL,
    source TEXT NOT NULL,
    geom GEOMETRY(POLYGON, 4326)
);
CREATE INDEX rag_documents_geom_idx ON rag_documents USING GIST (geom);
CREATE INDEX rag_documents_embedding_idx ON rag_documents
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
```

## RLS Policy

```sql
ALTER TABLE aois ENABLE ROW LEVEL SECURITY;
CREATE POLICY aois_session_isolation ON aois
    USING (session_id::text = current_setting('app.current_session_id', true));
```

```python
# backend/app/core/database.py
await session.execute(
    text("SET LOCAL app.current_session_id = :sid"),
    {"sid": str(session_id)},
)
```

## Comandos

```bash
make db-new name=create_alphaearth_tiles
make db-migrate
make db-rollback
make db-status
dbmate dump
```
