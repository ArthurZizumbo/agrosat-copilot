# Catálogo de Skills — AgroSatCopilot

> Catálogo completo de las 30 skills `agrosat-*`. Resumen ejecutivo en [`AGENTS.md`](../../AGENTS.md). Detalle de cada skill en su `SKILL.md` correspondiente.

Skills viven en `.claude/skills/<nombre>/SKILL.md` con frontmatter YAML (`name`, `description`, `allowed-tools`). Claude las carga automáticamente por `description` o por invocación manual `/<nombre>`.

## Backend & API (3)

| Skill | Descripción |
|-------|-------------|
| `agrosat-backend-api` | Endpoints FastAPI `/chat` SSE, `/aois`, `/timeseries`, `/stac/search`, `/llm/switch`, `/tiles` |
| `agrosat-backend-services` | Service layer, DI, Pub/Sub workers, integración ADK |
| `agrosat-titiler-cog` | TiTiler dinámico COG overlays NDVI/NDWI, rio-tiler, mosaic JSON |

## Database (2)

| Skill | Descripción |
|-------|-------------|
| `agrosat-db-migrations` | dbmate SQL, extensions (postgis, pgvector, pgstac), índices GIST, RLS |
| `agrosat-db-models` | SQLModel + GeoAlchemy2, `alphaearth_tiles`, `sentinel2_scenes`, `parcels`, `chat_sessions` |

## Frontend (3)

| Skill | Descripción |
|-------|-------------|
| `agrosat-frontend-components` | Vue 3, Nuxt UI Pro, chat streaming, switch A/B LLM, panel resultados |
| `agrosat-frontend-composables` | `useChat()`, `useSSE`, `useMap`, Pinia stores |
| `agrosat-maplibre-geo` | MapLibre GL, deck.gl, draw-polygon AOI, overlays COG, GeoJSON reactivos |

## ML & Modelado (5)

| Skill | Descripción |
|-------|-------------|
| `agrosat-ml-features` | 17 índices espectrales (NDVI, NDWI, EVI, MSAVI2, MCARI, CCCI, NDRE…), fusión Polars |
| `agrosat-ml-baseline` | XGBoost + LightGBM + RF sobre AlphaEarth 64-dim, spatial CV, calibración |
| `agrosat-ml-segmentation` | 6 arquitecturas EPIC 5 (U-Net, DeepLabv3+, SegFormer, U-TAE, TSViT, Swin-UNETR) |
| `agrosat-ml-ensemble` | Voting / Bagging / Stacking / Blending con Optuna |
| `agrosat-ml-evaluation` | mIoU, F1-macro, AgroMind, GeoAnalystBench, GeoBenchX, LLM-as-judge DeepEval |

## Foundation Models, Geospatial & Agente (4)

| Skill | Descripción |
|-------|-------------|
| `agrosat-gee-alphaearth` | GEE + AlphaEarth Foundations v2.1, export COG, service account |
| `agrosat-llm-finetuning` | Gemma 4 26B-MoE LoRA rank 16, Qwen3-VL, QLoRA fallback, vLLM serving |
| `agrosat-google-adk-agent` | Google ADK Plan-and-React, 9 FunctionTools, tracing, Vertex AI Agent Engine |
| `agrosat-spatial-rag` | Spatial-RAG híbrido: PostGIS ST_DWithin + pgvector e5-mistral-7b |

## MLOps & Infra (5)

| Skill | Descripción |
|-------|-------------|
| `agrosat-dagster-mlops` | Assets `alphaearth_annual`, `sentinel2_scenes`, `dinov3_features`, `*_model`, schedules |
| `agrosat-dvc-mlflow` | DVC remote GCS, MLflow 2.16, tags `data_version` + `code_version`, model registry |
| `agrosat-terraform` | Módulos `gcp/` y `azure/`, workspaces dev/staging/prod, backend GCS versionado |
| `agrosat-gcp-services` | Cloud Run, Cloud SQL, GCS, Pub/Sub, Vertex AI, Artifact Registry, Secret Manager |
| `agrosat-azure-h100` | VM `Standard_NC40ads_H100_v5` spot, `azure_h100_start/stop.sh`, auto-shutdown 12h |

## Observabilidad, Seguridad & QA (5)

| Skill | Descripción |
|-------|-------------|
| `agrosat-evidently-drift` | Evidently AI reportes semanales sobre bandas Sentinel-2 y embeddings AlphaEarth |
| `agrosat-finops` | Auditoría Cloud Run + Cloud SQL + Azure H100 spot, scale-to-zero, budget alerts |
| `agrosat-security` | Clerk OAuth, JWT, RBAC, rate limiting, CSP, RLS per-session, audit logging |
| `agrosat-security-audit` | OWASP Top 10, CIS GCP, pre-deploy checklist, gitleaks, cross-tenant isolation |
| `agrosat-testing` | pytest + pytest-asyncio, Playwright E2E `/chat`, mocks Vertex AI + GEE + vLLM |

## Transversal (3)

| Skill | Descripción |
|-------|-------------|
| `agrosat-code-review` | Checklist PR por epica, rúbrica curso, security gates |
| `agrosat-git-workflow` | Conventional Commits `feat(EX): ...`, branches `feature/E{epic}-US-XXX-{slug}` |
| `agrosat-engram-memory` | Engram local SQLite+FTS5 como memoria dev-time entre sesiones (solo dev) |
