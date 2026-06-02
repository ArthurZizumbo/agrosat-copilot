# AgroSatCopilot

> **Cuantificación de superficies de cultivo** mediante segmentación semántica de
> imágenes satelitales, Foundation Models (AlphaEarth Foundations) y procesamiento
> conversacional por LLMs (Gemma 4, Qwen3.5-35B-A3B y Gemini 3.1 Pro).

**Proyecto Integrador MNA** — Tec de Monterrey · Scuola Superiore Sant'Anna (Pisa)
**Sponsor:** Dr. Gerardo Jesús Camacho González (gjcamacho@tec.mx)
**Trimestre:** 20-abr a 3-jul-2026 (10 semanas efectivas + 2 buffer)

[![CI](https://github.com/arthurzizumbo/agro_sat_copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/arthurzizumbo/agro_sat_copilot/actions/workflows/ci.yml)
[![Dashboard](https://img.shields.io/badge/Streamlit-dashboard%20live-FF4B4B?logo=streamlit&logoColor=white)](https://share.streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/Code-MIT-green)](#licencia)

---

## El problema, la hipótesis y el diseño

| | |
|---|---|
| **Problema** | Las administraciones agrícolas necesitan saber **cuánta superficie de cada cultivo** hay y **dónde**, a partir de imágenes satelitales, sin censos de campo costosos. La clasificación píxel a píxel de cultivos es difícil: hay decenas de clases, fuerte desbalance (~31x entre la clase mayoritaria y la minoritaria) y nubosidad que corrompe el dato. |
| **Hipótesis** | La señal que distingue un cultivo de otro **vive en su evolución temporal** (el calendario fenológico de crecimiento), no en una sola imagen estática. Los modelos que consumen la serie temporal completa de Sentinel-2 deberían superar a los que ven un compuesto fijo. |
| **Diseño** | Recorrido CRISP-ML(Q) en cuatro avances: **EDA → Feature Engineering → Baseline tabular → Segmentación densa**. En segmentación se enfrentan seis arquitecturas de tres familias (convolucional 2D, transformer jerárquico y encoder temporal). El criterio de ganador es el mayor **mIoU** sobre el split de validación espacial. |

> **Resultado central:** la hipótesis se confirma. Los encoders temporales (TSViT, U-TAE)
> dominan a los baselines densos 2D, y el reencuadre fenológico aporta un margen
> consistente. El detalle, figura a figura, vive en el **dashboard interactivo**.

---

## La historia del proyecto (A1 → A4)

Cada avance parte del anterior y mejora una métrica medible. Esta tabla es el hilo
narrativo; el dashboard la cuenta con figuras y conclusiones por hito.

| Avance | Fase | Qué se hizo | Métrica clave |
|--------|------|-------------|---------------|
| **A1** | EDA | 6 notebooks: calidad del dato, desbalance, separabilidad de embeddings AlphaEarth | RF crudo sobre AlphaEarth: **OOB 0,83–0,89** |
| **A2** | Feature Engineering | 3 notebooks: índices espectrales, features temporales (FFT, fenología), fusión multisensor | Reducción de features hasta **−55,7 %** sin perder señal |
| **A3** | Baseline | Baseline tabular + reencuadre fenológico; ablation de features y descarte de leakage geográfico | XGBoost **F1-macro 0,41** (+0,09 sobre el baseline 0,32) |
| **A4** | Segmentación | 6 arquitecturas densas + ajuste fino Optuna sobre los top-2 | **TSViT-pheno: mIoU 0,625 · F1-macro 0,75 · pixel-acc 0,876** |

**Comparativa del Avance 4 (métricas reales, split de validación espacial):**

| Arquitectura | mIoU | F1-macro | Nota |
|--------------|------|----------|------|
| **TSViT-pheno** (Paper 1) | **0,625** | **0,750** | Ganador — encoder temporal + reencuadre fenológico |
| TSViT | 0,622 | 0,747 | Encoder temporal puro |
| U-TAE | 0,474 | 0,609 | Atención temporal U-Net |
| AnySAT / Swin-UNETR | 0,446 | 0,572 | Backbone multimodal ligero |
| DeepLabv3+ | 0,271 | 0,386 | Baseline denso 2D |
| U-Net | 0,242 | 0,346 | Baseline denso 2D |
| SegFormer-B2 | 0,232 | 0,342 | Corrida sobre 3 bandas RGB (no comparable de igual a igual) |

> **Lectura honesta:** el mejor mIoU flat (0,625) queda **bajo el target de rúbrica 0,70**.
> La brecha la arrastran las clases minoritarias (legumbres, viñedos); la pixel-accuracy
> del ganador es 0,876 y el mIoU agrupado por familia agronómica sube. El cierre pasa por
> *loss* ponderada y más épocas en H100, no por cambiar de familia de modelo.

### Dashboard interactivo

El recorrido completo —línea de tiempo, EDA, FE, baseline y segmentación, con narrativa
por figura y métricas reales— vive en un dashboard Streamlit:

```bash
poetry run streamlit run app/eda_dashboard.py    # local
# o desplegado en share.streamlit.io (ver deploy/streamlit/)
```

---

## Equipo

<table>
  <tr>
    <td align="center" width="33%">
      <img src="img/ArthurZizumbo.png" width="120" style="border-radius:50%"><br>
      <strong>Arthur Zizumbo</strong><br>
      MLOps / Platform Lead<br>
      <sub>Terraform · CI/CD · DVC · MLflow · Dagster · FinOps</sub>
    </td>
    <td align="center" width="33%">
      <img src="img/AaronBocanegra.jpg" width="120" style="border-radius:50%"><br>
      <strong>Aaron Bocanegra</strong><br>
      Full-Stack / Backend Lead<br>
      <sub>FastAPI · TiTiler · Nuxt 4 · endpoints ADK · seguridad</sub>
    </td>
    <td align="center" width="33%">
      <img src="img/IsaacAvila.jpg" width="120" style="border-radius:50%"><br>
      <strong>Isaac Ávila</strong><br>
      ML Engineer / Data Scientist<br>
      <sub>Modelos · fine-tune Gemma 4 + Qwen3-VL · AlphaEarth · Polars</sub>
    </td>
  </tr>
</table>

---

## Estructura del repositorio

```
agro_sat_copilot/
├── app/                  # Dashboard Streamlit (paquete app/dashboard/ + entry point shim)
│   └── dashboard/        #   theme · loaders · components · layout · spatial · timeline · registry · sections/
├── backend/              # FastAPI + SQLModel + TiTiler + SSE + Pub/Sub workers
├── frontend/             # Nuxt 4 SSR + MapLibre + deck.gl + @ai-sdk/vue + i18n (it/es/en)
├── ml/                   # Pipeline ML: ingesta, features, baseline, 6 segmentaciones, ensambles
│   ├── report/           #   contenido editorial del dashboard (avance1..4_content, narrativas)
│   ├── models/           #   arquitecturas custom (TSViT, U-TAE, Swin-UNETR)
│   └── agent/            #   agente conversacional Google ADK + 9 tools geoespaciales
├── dagster_project/      # Assets, jobs, schedules; lineage DVC <-> MLflow
├── db/                   # Migraciones dbmate · PostGIS · pgvector · pgstac · RLS por sesión
├── infrastructure/       # Terraform GCP + Azure H100 + Cloud Build
├── notebooks/            # Avances del curso (EDA, FE, baseline, segmentación) ejecutados con outputs
├── reports/              # Figuras y métricas de cada avance (segmentación, baseline, ...)
├── paper/                # Paper Track opcional + figuras del dashboard
├── deploy/streamlit/     # Configuración de despliegue del dashboard en Streamlit Cloud
└── docs/                 # Orquestación, ADRs, US resueltas, licencias, rúbricas
```

---

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
make check                            # ruff + secrets-scan + i18n-check (notebooks NO se strippean, se commitean con outputs)
make notebooks-check                  # papermill end-to-end opcional - valida que los .ipynb sigan ejecutables
make test                             # pytest backend
make verify-structure                 # chequea AC-4 de US-001

# Dashboard del curso (línea de tiempo + 4 avances)
poetry run streamlit run app/eda_dashboard.py

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
- [`context/RefinamientoPlaneacionAgroSatCopilot_v6.md`](context/RefinamientoPlaneacionAgroSatCopilot_v6.md) — plan SCRUM completo (12 EPICs, US-001 a US-056)
- [`docs/orchestration/`](docs/orchestration/) — catálogo de 30 skills, auto-invoke table, mapa skill↔subagente, comandos Make
- [`docs/us-resolved/`](docs/us-resolved/) — bitácora de User Stories cerradas (la historia detallada del proyecto)
- [`docs/licenses/DATA_LICENSE.md`](docs/licenses/DATA_LICENSE.md) — atribuciones de datasets y modelos

## Skills y subagentes

30 skills `agrosat-*` (`.claude/skills/`) y 9 subagentes profundos (`.claude/agents/`):
ml-engineer, mlops-engineer, geo-data-engineer, backend-engineer, frontend-engineer,
agent-engineer, finops-auditor, security-reviewer, paper-writer. Ver
[`docs/orchestration/auto-invoke.md`](docs/orchestration/auto-invoke.md).

## Licencia

MIT (código). Datasets bajo sus licencias respectivas; ver
[`docs/licenses/DATA_LICENSE.md`](docs/licenses/DATA_LICENSE.md).
