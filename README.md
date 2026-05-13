# AgroSatCopilot

> Cuantificación de superficies de cultivo mediante segmentación semántica de imágenes satelitales, Foundation Models (AlphaEarth Foundations) y procesamiento conversacional por LLMs (Gemma 4, Qwen3.5-35B-A3B y Gemini 3.1 Pro).

**Proyecto Integrador MNA** — Tec de Monterrey · Scuola Superiore Sant'Anna (Pisa)
**Sponsor:** Dr. Gerardo Jesús Camacho González (gjcamacho@tec.mx)
**Trimestre:** 20-abr a 3-jul-2026 (10 semanas efectivas + 2 buffer)

## Equipo

- Arthur Jafed Zizumbo Velasco — MLOps / Platform Engineer (lead)
- Carlos Aaron Bocanegra Buitrón — Full-Stack / Backend Lead
- Carlos Isaac Ávila Gutiérrez — ML Engineer / Data Scientist

## Stack v5

| Capa | Tecnología |
|------|-----------|
| Frontend | Nuxt 4 SSR + MapLibre GL + deck.gl + `@ai-sdk/vue` + Pinia + `@nuxtjs/i18n` (it/es/en) + Clerk |
| Backend | FastAPI + Polars + TiTiler + SQLModel + GeoAlchemy2 + structlog |
| Database | PostgreSQL 15 + PostGIS + pgvector + pgstac (migraciones con **dbmate**) |
| ML | PyTorch 2.4 + `transformers` + `peft` LoRA + `segmentation_models.pytorch` + `monai` + vLLM |
| FM EO | AlphaEarth Foundations v2.1 (GEE, gratis) |
| Feature extractor | DINOv3-satellite frozen |
| VLM principal | Gemma 4 26B-MoE LoRA (Apache 2.0) |
| LLM orquestador | Gemini 3.1 Pro (cloud) + Qwen3.5-35B-A3B (vLLM on-prem) — switch A/B |
| Framework agente | **Google ADK** con tracing built-in + deploy Vertex AI Agent Engine |
| MLOps | DVC + MLflow + **Dagster** asset-oriented + Evidently AI drift |
| Infra | Terraform mono-cloud GCP + Azure H100 NVL 96GB spot puntual |

[![CI](https://github.com/arthurzizumbo/agro_sat_copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/arthurzizumbo/agro_sat_copilot/actions/workflows/ci.yml)

## Setup (5 pasos)

```bash
# 1) Clonar
git clone https://github.com/arthurzizumbo/agro_sat_copilot.git
cd agro_sat_copilot

# 2) Configurar env local (editar GCP_PROJECT_ID, CLERK_*, HF_TOKEN, etc.)
cp .env.example .env.local

# 3) Instalar deps (poetry + pnpm)
make bootstrap                  # CPU only (Mac, Win/Linux sin GPU, CI)
# make bootstrap-gpu            # +torch CUDA 13.0 +bitsandbytes (Win/Linux con GPU NVIDIA)
# make bootstrap-gpu-linux      # +flash-attn +vllm (solo Linux, replica cloud)

# 4) Levantar 8 servicios + aplicar migraciones
make dev
make db-migrate

# 5) Abrir UI (puertos por defecto +10 sobre canónico para evitar choques con
#    otros stacks Docker locales; override en .env.local si los conflictos son
#    otros).
open http://localhost:3010     # Nuxt 4 frontend
# API:     http://localhost:8010/docs
# Dagster: http://localhost:3011
# MLflow:  http://localhost:5010
# TiTiler: http://localhost:8011
# Postgres: localhost:55432  ·  Redis: localhost:63790
```

## Quickstart adicional

```bash
# Tests + lint
make check                            # ruff + secrets-scan + nbstripout + i18n-check
make test                             # pytest backend
make verify-structure                 # chequea AC-4 de US-001

# Entrenamiento
make train-l4 epic=E4                 # baselines en L4 spot
make azure-h100-start
make train-h100 window=V3 script=train_gemma4_lora.py
make azure-h100-stop

# Infra
make tf-plan env=dev
make tf-apply env=dev
make deploy-staging                   # solo si entornos reactivados (ver ADR-002)
```

## Decisiones arquitectónicas (ADRs)

- [ADR-001](docs/decisions/ADR-001-no-cookiecutter-externo.md) — Monorepo en lugar de cookiecutter externo
- [ADR-002](docs/decisions/ADR-002-single-env-dev.md) — Único entorno `dev` durante el curso
- [ADR-003](docs/decisions/ADR-003-upstash-redis.md) — Upstash Redis serverless en lugar de GCP Memorystore
- [ADR-004](docs/decisions/ADR-004-poetry-optional-groups-and-aiplatform-ml.md) — Grupos Poetry opcionales + Vertex AI Agent Engine fuera del backend

## Documentación

- [`AGENTS.md`](AGENTS.md) ≡ [`CLAUDE.md`](CLAUDE.md) — orquestador único (espejos): identidad, decisiones irrevocables, calendario, presupuesto, reglas, checklist US
- [`context/RefinamientoPlaneacionAgroSatCopilot_v6_RESUMEN.md`](context/RefinamientoPlaneacionAgroSatCopilot_v6_RESUMEN.md) — resumen operativo del plan
- [`context/RefinamientoPlaneacionAgroSatCopilot_v6.md`](context/RefinamientoPlaneacionAgroSatCopilot_v6.md) — plan SCRUM completo (12 EPICs, US-001 a US-055)
- [`docs/orchestration/`](docs/orchestration/) — catálogo de 30 skills, auto-invoke table, mapa skill↔subagente, comandos Make
- [`docs/licenses/DATA_LICENSE.md`](docs/licenses/DATA_LICENSE.md) — atribuciones de datasets y modelos

## Skills (.claude/skills/)

30 skills especializadas: backend, frontend, ml, agent, infra, mlops, security, qa. Ver [`docs/orchestration/auto-invoke.md`](docs/orchestration/auto-invoke.md).

## Subagentes (.claude/agents/)

9 subagentes profundos: ml-engineer, mlops-engineer, geo-data-engineer, backend-engineer, frontend-engineer, agent-engineer, finops-auditor, security-reviewer, paper-writer.

## Licencia

MIT (código). Datasets bajo sus licencias respectivas; ver `docs/licenses/DATA_LICENSE.md`.
