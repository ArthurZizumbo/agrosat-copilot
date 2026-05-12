# Database Sub-Agent — AgroSatCopilot

> Sobreescribe al orquestador root para trabajo de schema y migraciones.

**Rol**: PostgreSQL 15 + PostGIS + pgvector + pgstac, migraciones con dbmate (SQL puro, framework-agnóstico).

## Skills References

- [agrosat-db-migrations](../.claude/skills/agrosat-db-migrations/SKILL.md) — dbmate workflow, índices GIST, RLS
- [agrosat-db-models](../.claude/skills/agrosat-db-models/SKILL.md) — SQLModel + GeoAlchemy2
- [agrosat-security](../.claude/skills/agrosat-security/SKILL.md) — RLS, ACL

## Auto-Invoke

| Acción | Skill |
|--------|-------|
| Crear migración nueva | `agrosat-db-migrations` |
| Crear/modificar SQLModel | `agrosat-db-models` |
| Crear índice GIST sobre geometry | `agrosat-db-migrations` |
| Crear índice IVFFlat/HNSW sobre vector | `agrosat-db-migrations` |
| Habilitar extensión (postgis, pgvector, pgstac) | `agrosat-db-migrations` |
| Crear RLS policy por session_id | `agrosat-security` + `agrosat-db-migrations` |
| Seed data demo | `agrosat-db-migrations` |

## Critical Rules

- **ALWAYS**: `dbmate new <name>` para nuevas migraciones, jamás editar archivos existentes
- **ALWAYS**: Migraciones reversibles con `-- migrate:up` y `-- migrate:down`
- **ALWAYS**: Extensiones declaradas en migración inicial (`CREATE EXTENSION IF NOT EXISTS postgis;`)
- **ALWAYS**: Columnas geometry con SRID explícito (4326 WGS84 por defecto)
- **ALWAYS**: Índice GIST en cada columna geometry/geography
- **ALWAYS**: Índice BTREE en columnas con queries `WHERE column = X`
- **ALWAYS**: Foreign keys con `ON DELETE` y `ON UPDATE` explícitos
- **ALWAYS**: `created_at`, `updated_at` con `TIMESTAMPTZ DEFAULT now()`
- **NEVER**: `SQLModel.metadata.create_all()` en prod
- **NEVER**: Modificar migración ya aplicada — crear `dbmate new` rollforward
- **NEVER**: Raw SQL con string format en backend — siempre parametrizado

## Schema Principal

```sql
-- Sesiones de chat con el agente (multi-tenant key)
chat_sessions (id, user_id, llm_variant, created_at, updated_at)

-- AOI dibujados por el usuario
aois (id, session_id, geom GEOMETRY(POLYGON, 4326), label, created_at)

-- Catálogo STAC interno
stac_collections (id, collection_id, title, description)
stac_items (id, collection_id, bbox GEOGRAPHY, datetime TIMESTAMPTZ, geometry GEOMETRY, properties JSONB, storage_uri)

-- AlphaEarth tiles
alphaearth_tiles (id, roi_name, year, bbox GEOMETRY, storage_uri, size_mb, download_date)

-- Sentinel-2 scenes
sentinel2_scenes (id, scene_id, bbox GEOMETRY, datetime TIMESTAMPTZ, cloud_cover REAL, bands_available TEXT[], storage_uri)

-- Parcelas inferidas con classification + segmentation
parcels (id, session_id, aoi_id, geom GEOMETRY, crop_class TEXT, confidence REAL, area_ha REAL, year INT)

-- Features pre-computadas por parcela
features_parcels (id, parcel_id, year, alphaearth_embedding VECTOR(64), ndvi_stats JSONB, phenology JSONB)

-- Spatial-RAG documentos
rag_documents (id, content TEXT, embedding VECTOR(4096), source TEXT, geom GEOMETRY)

-- Memory persistente del agente (ADK SessionService)
agent_sessions (id, session_id, messages JSONB, state JSONB, updated_at)
```

## Comandos

```bash
make db-migrate             # dbmate up
make db-rollback            # dbmate down
make db-new name=create_xxx # genera 20260511HHMMSS_create_xxx.sql
make db-status              # dbmate status
make db-seed                # python scripts/seed.py
make db-shell               # psql interactiva
```

## QA Checklist DB

- [ ] Migración reversible (`-- migrate:up` y `-- migrate:down`)
- [ ] Índices GIST en geometry columns
- [ ] Índices BTREE en columnas de query frecuente
- [ ] Foreign keys con ON DELETE explícito
- [ ] `created_at` y `updated_at` TIMESTAMPTZ
- [ ] RLS policies por session_id en tablas multi-tenant
- [ ] Seed data demo en migración inicial
- [ ] Tests de integración con DB efímera (testcontainers)
