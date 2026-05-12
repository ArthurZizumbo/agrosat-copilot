# Mapa Skill → Subagente owner — AgroSatCopilot

> Cada skill tiene 1+ subagente que la opera. Las skills transversales (git, memoria dev) son self-served. Auto-invoke en [`auto-invoke.md`](auto-invoke.md). Subagentes en [`.claude/agents/`](../../.claude/agents/).

| Skill | Owner(s) |
|-------|----------|
| `agrosat-backend-api` | `backend-engineer` |
| `agrosat-backend-services` | `backend-engineer` |
| `agrosat-titiler-cog` | `backend-engineer`, `geo-data-engineer` |
| `agrosat-db-migrations` | `backend-engineer`, `geo-data-engineer` |
| `agrosat-db-models` | `backend-engineer` |
| `agrosat-frontend-components` | `frontend-engineer` |
| `agrosat-frontend-composables` | `frontend-engineer` |
| `agrosat-maplibre-geo` | `frontend-engineer` |
| `agrosat-ml-features` | `geo-data-engineer` |
| `agrosat-ml-baseline` | `ml-engineer` |
| `agrosat-ml-segmentation` | `ml-engineer` |
| `agrosat-ml-ensemble` | `ml-engineer` |
| `agrosat-ml-evaluation` | `ml-engineer`, `agent-engineer`, `paper-writer` |
| `agrosat-gee-alphaearth` | `geo-data-engineer` |
| `agrosat-llm-finetuning` | `ml-engineer`, `agent-engineer`, `paper-writer` |
| `agrosat-google-adk-agent` | `agent-engineer`, `backend-engineer` |
| `agrosat-spatial-rag` | `agent-engineer` |
| `agrosat-dagster-mlops` | `mlops-engineer`, `geo-data-engineer` |
| `agrosat-dvc-mlflow` | `mlops-engineer`, `paper-writer` |
| `agrosat-terraform` | `mlops-engineer`, `finops-auditor` |
| `agrosat-gcp-services` | `mlops-engineer`, `finops-auditor` |
| `agrosat-azure-h100` | `mlops-engineer`, `ml-engineer`, `finops-auditor` |
| `agrosat-evidently-drift` | `mlops-engineer` |
| `agrosat-finops` | `finops-auditor`, `mlops-engineer` |
| `agrosat-security` | `backend-engineer`, `frontend-engineer`, `security-reviewer` |
| `agrosat-security-audit` | `security-reviewer`, `backend-engineer` |
| `agrosat-testing` | `backend-engineer`, `frontend-engineer` |
| `agrosat-code-review` | `security-reviewer` |
| `agrosat-git-workflow` | `security-reviewer`, `backend-engineer`, `frontend-engineer` (transversal) |
| `agrosat-engram-memory` | self-served (dev-time only, cualquier sesión Claude Code) |

## Subagentes (9 totales en `.claude/agents/`)

| Subagente | Cuándo invocarlo |
|-----------|------------------|
| `ml-engineer` | Diseño arquitectura, configs LoRA, validación VRAM H100 |
| `mlops-engineer` | Pipelines Dagster + DVC + MLflow, scripts H100, Terraform |
| `geo-data-engineer` | Ingesta GEE + CDSE + DINOv3, COG, pgstac, índices |
| `backend-engineer` | FastAPI + SSE, workers Pub/Sub, integración ADK, TiTiler |
| `frontend-engineer` | Nuxt 4 + MapLibre + chat streaming + i18n |
| `agent-engineer` | Tools ADK, Spatial-RAG, Plan-and-React, eval AgroMind |
| `finops-auditor` | Auditoría costos cloud, presupuesto H100, scale-to-zero |
| `security-reviewer` | OWASP, CIS GCP, RBAC, audit logging, secret scanning |
| `paper-writer` | Paper Track opcional, IEEE format, GEO-Bench-2, AgroMind-IT/ES |
