# Plan de Proyecto: AgroSatCopilot — v8

**Cuantificación de Superficies de Cultivo mediante Segmentación Semántica de Imágenes Satelitales, AlphaEarth Foundations, FarSLIP contrastivo-fenológico y procesamiento conversacional por LLMs (Gemini 3.5 Flash + Qwen3.5-35B-A3B)**

---

**Documento:** Plan SCRUM completo y vigente del Proyecto Integrador — MNA (Maestría en Inteligencia Artificial Aplicada). Este es **el plan que se ejecuta**; reemplaza por completo a v6 y v7 (conservados como histórico).
**Trimestre:** 20 de abril a 3 de julio de 2026.
**Corte de proyecto:** 7-jun-2026. 
**Ratificado por:** [ADR-009](../docs/decisions/ADR-009-h100-reactivacion-pivote-farslip-alcance-v8.md) (reactivación H100 + pivote FarSLIP). Calendario: [ADR-008](../docs/decisions/ADR-008-rediseno-calendario-presentacion-27jun.md).

### Equipo

| Integrante | Rol |
|------------|-----|
| Arthur Jafed Zizumbo Velasco | MLOps / Platform Engineer (lead) |
| Carlos Aaron Bocanegra Buitrón | Full-Stack / Backend Lead |
| Carlos Isaac Ávila Gutiérrez | ML Engineer / Data Scientist |

### Cómputo (corte 7-jun-2026)

- **H100 NVL 96GB — DISPONIBLE** (VM `gjcamacho-gpuh1` del sponsor, túnel Cloudflare, entorno micromamba `agrosat`, repo en `F:\projects\agrosat-copilot`, disco F: 3.8 TB). Una sola GPU → coordinar `nvidia-smi`. El cuello de botella ahora es **tiempo** (~3 semanas a la presentación), no cómputo.
- **GCP L4 24GB** + RTX local del equipo para desarrollo y jobs ligeros.

---

## 0. Cambios vs v6 (trazabilidad — por qué este es un nuevo plan, no una suma)

Este v8 **quita** lo descartado y lo **reemplaza** con lo vigente. No es v6 + apéndice. Resumen de los cambios estructurales:

| Área | v6 (descartado) | v8 (vigente) | Razón |
|------|-----------------|--------------|-------|
| Modelo final (EPIC 6) | Gemma 4 26B-MoE LoRA como modelo final | **Ensambles** (4 base + E-a/E-b/E-c). Gemma 4 LoRA → **diferido** (US-050) | Experts MoE 3D fused rompen QLoRA; AgroMind es eval-only (fine-tune = leakage). Sin tiempo para 3-5 días de debug |
| Modelo 6 (EPIC 5) | Swin-UNETR | **AnySat** (sustitución formalizada) | Swin-UNETR nunca se entrenó; AnySat ya ocupa la 6.ª silla (mIoU 0.4459) |
| SegFormer (EPIC 5) | SegFormer-B2 + cabezal open-vocabulary FarSLIP | **SegFormer-B0, 3 bandas RGB** (realidad de lo corrido) | El cabezal open-vocab no se implementó así; FarSLIP se reorienta a contrastivo-fenológico |
| FarSLIP | Ablación negativa (perdía 0.163 vs 0.233) | **Camino principal** (directiva sponsor): contrastivo con descripciones fenológicas Gemini Flash, incremental 4→18, filtro 3:1 | El gap era `train.py:184` (prototipos `torch.randn`), no el método |
| LLM orquestador (EPIC 7) | Gemini 3.1 Pro (2M ctx) + Qwen3.5 vLLM + Qwen3-VL LoRA | **Gemini 2.5 Pro GA (1M ctx)** reasoner + **Qwen3.5-35B-A3B vLLM** variante-B. Qwen3-VL LoRA → fuera | Corrección factual (1M no 2M); Qwen3-VL LoRA no aporta vs Gemini en 3 semanas |
| FM EO | "AlphaEarth Foundations v2.1" | **AlphaEarth `SATELLITE_EMBEDDING/V1/ANNUAL` data v1.1** (CC-BY-4.0) | El asset público de GEE es V1/v1.1; no existe "v2.1" |
| Alcance regional | Solo Francia (PASTIS-R) | **+ Multi-región transfer**: Sen4AgriNet Catalonia, EuroCropsML few-shot, WorldCereal/Harmonized (tropical, futuro) | Cubrir la brecha "funciona fuera de Francia" con transfer learning real |
| Ventanas H100 | V1-V6 con fechas viejas | Calendario único [ADR-008](../docs/decisions/ADR-008-rediseno-calendario-presentacion-27jun.md); H100 24/7 del sponsor | La VM del sponsor está disponible 24/7, no por ventanas |

**US descartadas del cuerpo vigente** (su intención migró o se diferió): el cabezal open-vocabulary de SegFormer; Swin-UNETR como entregable; Gemma 4 LoRA como modelo final; Qwen3-VL LoRA. Donde una US v6 ya se entregó, se marca **RESUELTA** con 1 línea y se conserva su texto.

---

## Backlog Priorizado de Ejecución

Numeración = identidad estable; el orden de EJECUCIÓN vive aquí y en el Roadmap (§Roadmap de Sprints), no en los números. El proyecto salta entre EPICs por prioridad.

| Prioridad | US | Título | EPIC | Bloquea/Depende | Sprint |
|-----------|----|--------|------|------------------|--------|
| P1 | US-032 | Filtro 3:1 Meadow per-patch | E5 | bloquea FarSLIP-pheno | S7 |
| P2 | US-033 | Prototipos fenología (Gemini Flash) | E5 | dep US-032 | S7 |
| P3 | US-034 | Fix `torch.randn` (prototipos reales) | E5 | dep US-033 | S7 |
| P4 | US-035 | Ablación de bandas FarSLIP (H100) | E5 | dep US-034 | S7 |
| P5 | US-036 | Incremental 4→18 clases | E5 | dep US-035 | S7 |
| P6 | US-037 | Eval FarSLIP-pheno vs AlphaEarth | E5 | dep US-036 | S7 |
| P7 | US-030 | Harness único de métrica | E5 | bloquea ensambles | S7 |
| P8 | US-031 | Regenerar softmax/OOF | E5 | dep US-030 | S7 |
| P9 | US-040 | 4 ensambles base (rúbrica A5) | E6 | dep US-031 | S7 (Avance 5) |
| P10 | US-041 | Ensamble E-a TSViT-pheno+FarSLIP | E6 | dep US-037, US-039 | S7 (Avance 5) |
| P11 | US-038/039 | TSViT full retrain H100 | E5 | dep US-030 | S8 |
| P12 | US-042 | Ensamble E-b +AlphaEarth | E6 | dep US-041 | S8 |
| P13 | US-051 | RLS multi-tenant | E8 | bloquea endpoints | S8 |
| P14 | US-045 | 9 FunctionTools ADK | E7 | dep US-051 | S8 |
| P15 | US-052 | /chat SSE | E8 | dep US-045 | S8 |
| P16 | US-047 | Agente factory (Gemini/Qwen) | E7 | dep US-045 | S8 |
| P17 | US-057 | ChatPanel.vue + useChat | E9 | dep US-052 | S9 |
| P18 | US-058 | MapView.vue + useMap | E9 | dep US-052 | S9 |

---

## 1. Resumen Ejecutivo

**AgroSatCopilot** es una plataforma SaaS conversacional open-source que permite a agrónomos y gestores agrícolas interactuar en lenguaje natural (italiano, español, inglés) con imágenes satelitales multimodales para análisis de cultivos. Combina:

1. **AlphaEarth Foundations** (`SATELLITE_EMBEDDING/V1/ANNUAL`, data v1.1) — embeddings 64-dim/píxel/año, gratis en GEE, global incluido México. Backbone de features. No se entrena un FM propio.
2. **Segmentación semántica densa** — 6 arquitecturas (U-Net, DeepLabv3+, SegFormer-B0, U-TAE, TSViT-pheno, AnySat) sobre PASTIS-R. Mejor individual: **TSViT-pheno (mIoU 0.6253 / F1 0.7500)**.
3. **FarSLIP contrastivo-fenológico** — adaptación CLIP que alinea imágenes con descripciones fenológicas generadas por Gemini Flash (directiva del sponsor). Alimenta el ensamble E-a.
4. **Ensambles** (modelo final) — 4 base (Voting/Bagging/Stacking/Blending) + 3 incrementales (E-a TSViT-pheno+FarSLIP → E-b +AlphaEarth → E-c +contexto geoespacial).
5. **Capa conversacional** — agente Google ADK con patrón perceiver-reasoner (Be My Eyes): los modelos perciben, **Gemini 2.5 Pro razona** (no clasifica píxeles). Variante-B on-prem: Qwen3.5-35B-A3B vLLM.
6. **Transferencia multi-región** — recipe train-Francia → extend-elsewhere (Sen4AgriNet Catalonia denso + EuroCropsML few-shot), armonizado vía HCAT v3. Demo metodológico de "recibir clases nuevas" (incl. México aguacate/guayaba, cualitativo).

El LLM no inventa los números: los lee del mapa segmentado y del ensamble.

### Método central

Segmentación semántica densa píxel-por-píxel como núcleo de la cuantificación de superficies. El ensamble final (nivel parcela) reduce el error respecto al mejor individual y alimenta las respuestas del copiloto.

### Calendario (ADR-008)

A4 ✓31-may · **A5 7-jun → mié 10-jun** (extendido por el profesor) · A6 14-jun · A7 21-jun · **Presentación 27-jun** · buffer/Paper Track 28-jun→3-jul.

---

## 2. Stack Tecnológico (vigente)

| Capa | Tecnología | Nota v8 |
|------|-----------|---------|
| Backend | FastAPI 3.12 + Polars 1.x, SQLModel + GeoAlchemy2, asyncpg | router→service→model |
| Frontend | Nuxt 4 SSR, MapLibre + deck.gl, @nuxtjs/i18n (it/es/en), Tailwind v4, Pinia | |
| DB | PostgreSQL 15 + PostGIS + pgvector, dbmate | RLS pendiente (US-051) |
| FM EO | AlphaEarth `SATELLITE_EMBEDDING/V1/ANNUAL` v1.1 (GEE, CC-BY-4.0) | global incl. México |
| Segmentación | smp (U-Net, DeepLabv3+), U-TAE, TSViT, AnySat | Swin-UNETR → AnySat |
| Contrastivo | FarSLIP (CLIP ViT-B/16) + descripciones fenológicas Gemini Flash | directiva sponsor |
| Ensambles | XGBoost, Optuna, OOF spatial-CV | `ml/ensemble/` (a construir) |
| LLM cloud | Gemini 2.5 Pro (GA, 1M ctx) | reasoner |
| LLM on-prem | Qwen3.5-35B-A3B vLLM (GPTQ-Int4, H100) | variante-B |
| Agente | Google ADK (perceiver-reasoner) | `google-adk` fuera del lock (conflicto genai 2.x) |
| MLOps | DVC (GCS) + MLflow (Docker :5010) + Dagster + Terraform GCP/Azure | nivel maestría |
| Datasets transfer | Sen4AgriNet (CC-BY-SA-4.0), EuroCropsML (CC-BY-SA-4.0), WorldCereal RDM | multi-región |

Detalle de arquitectura, FinOps y benchmarks: ver las épicas correspondientes abajo y [`docs/STATUS.md`](../docs/STATUS.md).

---

## 3. Mapa de Épicas

| EPIC | Tema | Avance | Estado v8 |
|------|------|--------|-----------|
| E0 | Infra, Cookiecutter, MLOps base | A0 | 🟢 mayormente cerrada |
| E1 | Ingesta (AlphaEarth, Sentinel) | A1 | 🟢 + ingesta multi-región nueva |
| E2 | EDA + Transferencia multi-región | A1/A6 | 🟢 EDA cerrado + US transfer nuevas |
| E3 | Feature Engineering + FarSLIP distill | A2 | 🟢 cerrada |
| E4 | Baseline AlphaEarth + XGBoost | A3 | 🟢 cerrada (XGB 0.6535 HCAT) |
| E5 | Segmentación + FarSLIP-fenológico | A4/A5 | 🟠 6 modelos listos + FarSLIP-pheno en curso |
| E6 | Modelo Final — Ensambles | A5 | 🔴 `ml/ensemble/` vacío — crítico |
| E7 | Agente Conversacional ADK | A6 | 🔴 esqueleto |
| E8 | Backend API + Tiling | A6 | 🟠 esqueleto |
| E9 | Frontend Web + Mapa + Chat | A6 | 🟠 esqueleto |
| E10 | Observabilidad, Drift, FinOps, Seguridad | A7 | 🟠 parcial (RLS pendiente) |
| E11 | Paper Track (opcional) | post | ⚪ opcional |
| E12 | Transferencia Multi-Región (datasets + few-shot) | A6/A7 | 🟠 nuevo — Sen4AgriNet/EuroCropsML demoable |

Las US completas de cada épica siguen a continuación, en formato **Como/quiero/para que + Criterios de Aceptación + Tareas técnicas + Estimación**. Las US ya entregadas se marcan **RESUELTA**.

---

## EPIC 0: Infraestructura, Cookiecutter y MLOps Base {#epic-0}

**Objetivo.** Establecer la estructura base del proyecto, el entorno reproducible local, la infraestructura mono-cloud GCP más la VM H100 (ahora disponible vía sponsor), y el pipeline MLOps antes de iniciar el trabajo con datos. Permite que los tres desarrolladores tengan paridad absoluta desde el primer commit y que cada experimento quede versionado y trazable.

**Alineado con.** Avance 0 (26 abril 2026) — Propuesta del proyecto.

**Estrategia.** Maximizar la reutilización del stack MLOps del proyecto previo del equipo (DVC, MLflow, GitHub Actions, Terraform) e incorporar Dagster asset-oriented y dbmate para reducir la curva de aprendizaje y el presupuesto de story points.

**Estado v8 (corte 7-jun-2026).** Esta épica está **mayoritariamente CERRADA**: monorepo, Docker Compose, CI/CD, quality gates, Terraform GCP + Azure H100, DVC, MLflow y Dagster en operación nivel maestría (ver [`docs/STATUS.md`](../docs/STATUS.md)). Dos correcciones de realidad se incorporan a las US existentes: (1) la **H100 NVL 96GB ya está disponible** (VM `gjcamacho-gpuh1`, entorno micromamba `agrosat`, repo en `F:\projects\agrosat-copilot`), por lo que el módulo Azure pasa de "parked" a operativo; (2) el **MLflow lineage real vive en el server Docker `:5010`**, no en `./mlruns`. Se añaden dos US nuevas v8 de capa plataforma — observabilidad de chat (US-065) y doc operativa de FinOps (US-067) — para completar la base antes de la presentación.

**Puntos totales de la épica: 14** (10 base v6 ya entregados + 4 nuevos v8).

---

### US-001 — Cookiecutter template del monorepo

> **Estado: RESUELTA** (docs/us-resolved/us-001.md). Monorepo generado y en uso por los 3 devs; estructura `backend/ frontend/ ml/ infrastructure/ notebooks/ data/ docs/ paper/ scripts/ db/migrations/` consolidada, Poetry + pnpm + Makefile operativos.

**Como** equipo de 3 desarrolladores,
- **quiero** un template cookiecutter que genere la estructura completa del monorepo AgroSatCopilot con un único comando,
- **para que** cualquier módulo nuevo se cree de forma consistente y el onboarding de cualquier colaborador externo sea inmediato.

**Criterios de Aceptación:**

- El comando `cookiecutter gh:agrosatcopilot/cookiecutter-agrosat` genera el proyecto completo en menos de dos minutos en macOS, Linux y WSL2.
- El template solicita interactivamente: `project_name`, `gcp_project_id`, `azure_subscription_id`, `region` (por defecto `us-central1`, default real en `infrastructure/terraform/modules/gcp/variables.tf`), `db_name`, `team_lead_email`.
- La estructura de directorios generada es: `backend/`, `frontend/`, `ml/`, `infrastructure/`, `notebooks/`, `data/`, `docs/`, `paper/`, `scripts/`, `.github/workflows/`, `db/migrations/`.
- Incluye `pyproject.toml` con Poetry y grupos `dev`, `test`, `ml`, `geo`, `paper`; `package.json` con pnpm; `Makefile` con comandos estandarizados (`make dev`, `make db-migrate`, `make train-l4`, `make train-h100`); `.env.example` con todas las variables requeridas documentadas.
- Incluye Dockerfiles multi-stage para backend y frontend, `docker-compose.yml` para desarrollo local, `cloudbuild.yaml`, módulos Terraform base para GCP y Azure, `dagster.yaml`, y configuración inicial `dbmate` en `db/migrations/`.

**Tareas técnicas:**

- [x] Crear repositorio `cookiecutter-agrosat` en GitHub con licencia MIT
- [x] Implementar templates Jinja2 para todos los archivos de configuración
- [x] Escribir hook `post_gen_project.py` que ejecuta `poetry install`, `pnpm install` y `git init`
- [x] Pipeline de validación en GitHub Actions con matrix de sistemas operativos (Ubuntu, macOS)
- [x] Documentar el uso del template en el README del repositorio

**Estimación:** 2 puntos (~1 día).

---

### US-002 — Entorno Docker Compose multiservicio

> **Estado: RESUELTA** (sin doc formal; verificado en repo). `make dev` levanta el stack multiservicio; PostGIS+pgvector, MLflow UI, Dagster UI y seed dbmate operativos. Nota v8: el MLflow de lineage real corre en el server Docker `:5010` (no `./mlruns`); los runs por subprocess pueden quedar `RUNNING` y deben cerrarse explícitamente.

**Como** desarrollador del equipo,
- **quiero** un entorno local reproducible levantado con `make dev`,
- **para que** los tres miembros del equipo trabajemos sobre exactamente los mismos componentes y versiones, y para que CI/CD tenga la misma especificación.

**Criterios de Aceptación:**

- El comando `make dev` levanta simultáneamente ocho servicios: FastAPI (puerto 8000), Nuxt 4 (3000), PostgreSQL con PostGIS y pgvector (5432), Redis (6379), TiTiler (8001), MLflow UI (5010), **Dagster UI (3001)** y Ollama local (11434) para pruebas de LLM pequeños (Gemma 4 E4B).
- `poetry install --with dev,test,ml,geo` completa sin conflictos de dependencias (validado con `poetry check`).
- Hot-reload funciona en FastAPI (vía uvicorn `--reload`) y Nuxt 4 (vía Vite HMR) dentro de los contenedores Docker.
- Las variables de entorno se cargan desde `.env.local` con validación Pydantic Settings en startup.
- PostgreSQL ejecuta seed automático la primera vez que se levanta usando **dbmate** (`dbmate up`): tablas base, datos demo de 1 parcela en Toscana.
- Healthchecks configurados en todos los servicios con retries exponenciales.

**Tareas técnicas:**

- [x] Escribir `docker-compose.yml` con los ocho servicios y red bridge compartida
- [x] Configurar Dockerfile multi-stage backend con builder (compila wheels) y runtime (slim)
- [x] Configurar Dockerfile frontend Nuxt 4 con cache de pnpm
- [x] Migración inicial `db/migrations/20260511213942_initial_schema.sql` con tablas base y parcela demo
- [x] Documentar troubleshooting común (puerto ocupado, rate limit Docker Hub)

**Estimación:** 2 puntos (~1 día).

---

### US-003 — Infraestructura GCP + Azure H100 con Terraform

> **Estado: RESUELTA** (sin doc formal; verificado en `infrastructure/terraform/`). Módulos GCP y Azure aplicados; backend de estado GCS versionado. Nota v8: la H100 NVL 96GB **ya está disponible** (VM `gjcamacho-gpuh1` del sponsor, acceso por túnel Cloudflare) — el módulo Azure pasa de "parked" a operativo. FinOps activos: Cloud SQL dev con `activation_policy=NEVER` (var `db_activation_policy` evita drift) y disco `farslip-data` reducido 250→125 GB vía snapshot.

**Como** MLOps Engineer,
- **quiero** la infraestructura declarada en Terraform para GCP primario y la VM H100 en Azure,
- **para que** el entorno de staging y producción sea reproducible y para que encender o apagar la VM H100 sea trivial.

**Criterios de Aceptación:**

- Módulo `terraform/gcp/` provisiona: Cloud Run services (api, frontend, tiling, inference-worker), Cloud SQL PostgreSQL 15 con extensiones PostGIS y pgvector, GCS buckets (data, artifacts, dvc-remote), Cloud Pub/Sub topics (`inference-jobs`, `inference-results`), Secret Manager con 6 secretos base, Artifact Registry para imágenes Docker, Cloud CDN, IAM roles mínimos necesarios.
- Módulo `terraform/azure/` provisiona: VM `Standard_NC40ads_H100_v5` con H100 NVL 96GB on-demand + variante spot, Azure Blob Storage Hot, VNet privada, NSG que sólo permite SSH desde IPs de los 3 devs. **(Realidad v8: la H100 operativa es la VM `gjcamacho-gpuh1` del sponsor, accedida por túnel Cloudflare sin tocar el NSG; el módulo Azure queda como referencia reproducible.)**
- Workspaces de Terraform separados: `dev`, `staging`, `prod`.
- Scripts `make azure-h100-start` y `make azure-h100-stop` automatizan el encendido/apagado de la VM H100 con timer de auto-shutdown configurable (por defecto 12 h).
- `terraform plan` y `terraform apply` ejecutan desde el pipeline Cloud Build con back-end de estado GCS versionado.

**Tareas técnicas:**

- [x] Escribir módulos Terraform con variables parametrizadas y outputs
- [x] Backend de estado en bucket `gs://agrosat-tfstate` con versionado activado
- [x] Scripts Bash `scripts/azure_h100_start.sh` y `scripts/azure_h100_stop.sh`
- [x] Tests con `terraform validate` en GitHub Actions
- [x] Documentar en `docs/operations/terraform-bootstrap.md` el flujo para aprovisionar y destruir
- [x] Var `db_activation_policy` (default `NEVER`) en módulo GCP para apagar Cloud SQL dev sin drift (FinOps)

**Estimación:** 2 puntos (~1 día).

---

### US-004 — DVC + MLflow + Dagster + dbmate MLOps base

> **Estado: RESUELTA** (sin doc formal; verificado en `dagster_project/` y `ml/utils/`). DVC remoto GCS, MLflow server (PostgreSQL backend + GCS artifacts), Dagster assets y dbmate operativos. Nota v8 crítica: **el lineage real vive en el server MLflow Docker `:5010`**, no en `./mlruns`; los runs lanzados por subprocess deben cerrarse o quedan `RUNNING`. Tag obligatorio `data_version` (hash DVC) + `code_version` (sha git) en todo entrenamiento E4/E5/E6.

**Como** equipo,
- **quiero** versionado de datos, tracking de experimentos, orquestación asset-oriented y migraciones de base de datos desde el primer commit,
- **para que** cualquier experimento reportado en los avances del curso sea ejecutable por un tercero a partir del repositorio.

**Criterios de Aceptación:**

- **DVC 3.48** inicializado con remote `gcs://agrosat-dvc-remote` y autenticación vía service account.
- **Dagster 1.9+** desplegado con assets declarativos: `alphaearth_annual`, `sentinel2_scenes`, `dinov3_features`, `spectral_indices`, `parcel_features`, `baseline_model`, `alt_models`, `final_vlm`, `ensemble`, `drift_check`. Cada asset con dependencias explícitas y lineage visible en Dagster UI. **(Realidad v8: assets activos incluyen `farslip`, `farslip_pipeline`, `features`, `phenology_models`, `sentinel2_crops` y `health`.)**
- **MLflow 2.16** server con tracking store PostgreSQL y artifact store GCS; URL accesible para el equipo (server Docker en `:5010`). Integración Dagster→MLflow vía recurso `dagster_project/resources/mlflow.py`.
- **dbmate** configurado en `db/migrations/`, con scripts `make db-migrate` (`dbmate up`) y `make db-rollback` (`dbmate down`). Migración inicial crea tablas base.
- Todos los scripts de entrenamiento del EPIC 4, 5, 6 registrarán automáticamente en MLflow: parámetros, métricas cada epoch, artefactos (checkpoints, matrices de confusión, curvas ROC), tags (`data_version` con el hash DVC y `code_version` con el sha git).

**Tareas técnicas:**

- [x] Inicializar DVC y configurar remote con service account
- [x] Escribir `dagster_project/assets/*.py` con definiciones de los assets principales
- [x] Desplegar MLflow server con `mlflow server --backend-store-uri postgresql://...` (Docker `:5010`)
- [x] Configurar dbmate con migración inicial
- [x] Template `ml/utils/mlflow_utils.py` con decorador `@track_experiment`

**Estimación:** 2 puntos (~1 día).

---

### US-005 — Pipeline CI/CD con GitHub Actions y Cloud Build

> **Estado: RESUELTA** (sin doc formal; verificado en `.github/workflows/ci.yml`). CI replica `make check` (ruff + secrets-scan + i18n-check) en cada PR; quality gates sin pre-commit, notebooks validados con papermill (con outputs preservados). El deploy completo a Cloud Run de los 4 servicios queda parcialmente diferido (el backend de negocio aún se construye en E7).

**Como** equipo,
- **quiero** un pipeline automatizado que valide y despliegue cada cambio,
- **para que** cualquier merge a `main` llegue a staging con smoke tests en menos de 10 minutos y sin intervención manual.

**Criterios de Aceptación:**

- Cada push a `develop` dispara: instalación de dependencias Poetry, linting con `ruff check`, formateo con `ruff format --check`, tipado con `mypy`, tests unitarios con `pytest`, verificación de cobertura ≥70% backend con `pytest-cov`, `dvc status` para detectar archivos sin versionar.
- Cada push a `main` dispara además: build de las imágenes Docker multi-stage, push a Artifact Registry con tag `sha-{git-sha}` y `latest`, aplicación de migraciones de base de datos con `dbmate up`, deploy a Cloud Run de los cuatro servicios (api, frontend, tiling, inference-worker), smoke tests contra `/healthz` de cada servicio, Playwright end-to-end test básico en staging que valida el flujo de chat con un query fijo. **(Realidad v8: smoke tests cubren `/healthz`+`/readyz`; el E2E de chat se activa al cerrar US-052/US-053 en E7/E8.)**
- El pipeline falla si la cobertura de tests cae por debajo del umbral o si los smoke tests no pasan.
- Los secretos utilizados (API keys de Gemini, Copernicus CDSE, HuggingFace tokens) se leen desde GitHub Secrets y se inyectan a Cloud Run desde Secret Manager.

**Tareas técnicas:**

- [x] Workflow `.github/workflows/ci.yml` para `develop`/`main` que replica `make check`
- [x] `cloudbuild.yaml` con substituciones parametrizadas
- [x] `make notebooks-check` (papermill end-to-end) en CI, preservando outputs
- [ ] Test E2E Playwright `tests/e2e/chat_smoke.spec.ts` (se activa al cerrar US-052/US-053)
- [x] Badge de estado del pipeline en el README del proyecto

**Estimación:** 2 puntos (~1 día).

---

**Subtotal EPIC 0: 14 story points** (10 base v6 entregados + 4 nuevos v8: US-065 observabilidad de chat 3 SP + US-067 FinOps doc 1 SP).

---

## EPIC 1: Ingesta de Datos — AlphaEarth, Sentinel, DINOv3, Multi-región {#epic-1}

**Objetivo.** Automatizar la descarga, preprocesamiento, conversión a Cloud-Optimized GeoTIFF y catalogación STAC de las fuentes de datos públicas necesarias para el proyecto, cubriendo las regiones piloto, el benchmark de control francés PASTIS y, **nuevo en v8**, las fuentes multi-región (Sen4AgriNet Catalonia, EuroCropsML, WorldCereal/Harmonized Global Crops) que habilitan la historia de transfer learning fuera de Francia. Toda fuente se armoniza vía la taxonomía jerárquica **HCAT v3**.

**Alineado con.** Avance 0 (entendimiento de los datos), Avance 1 (disponibilidad para EDA) y, en v8, Avance 6/7 (transferencia multi-región).

**Regiones de interés:** control PASTIS en Francia metropolitana (camino denso ya operativo); Cataluña 31TCG (Sen4AgriNet, demo Franco-Ibérica W3); Estonia/Portugal/Letonia (EuroCropsML, few-shot); África/Brasil tropical (WorldCereal/HGC, FUTURE/paper). Las regiones italianas piloto (Pianura Padana, Toscana, Apulia) quedan diferidas frente al camino multi-región franco-ibérico, que es el que la directiva del sponsor y la rúbrica de transfer premian a 3 semanas de la presentación.

**Correcciones de realidad v8 (verificadas 7-jun-2026):**
- **AlphaEarth = `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, data v1.1** (NO "v2.1"), global incluido México, licencia **CC-BY-4.0**. Ya implementado en `ml/ingest/gee_sampler.py`.
- **DINOv3-satellite NO existe en código.** `ml/extractors/` contiene únicamente `farslip_extractor.py` y `farslip_official_extractor.py`. El extractor DINOv3 se marca como **planificado/diferido** con honestidad — el feature-extractor self-supervised real del proyecto es FarSLIP, no DINOv3.

**Puntos totales de la épica: 12 (base v6) + 30 (multi-región v8) = 42 story points.**

---

### US-006 — Pipeline de ingesta de AlphaEarth Foundations desde GEE

> **Estado: AVANZADA (no formalmente cerrada).** Implementada en `ml/ingest/gee_sampler.py` consultando `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` (data v1.1, CC-BY-4.0). El asset Dagster `alphaearth_annual` y la tabla PostGIS quedan como deuda residual; no hay `docs/us-resolved/us-006.md`.

**Como** ML Engineer,
- **quiero** descargar los embeddings AlphaEarth Foundations 64-dim para las regiones de interés del proyecto,
- **para que** sean la fuente principal de features del pipeline de modelado sin necesidad de entrenar un foundation model propio.

**Criterios de Aceptación:**

- Se define un archivo `config/rois.yaml` con las geometrías (PASTIS control + regiones multi-región) en formato GeoJSON, con metadatos `name`, `bbox`, `crs` (EPSG:4326) y `preferred_crs_projection` (EPSG:32631 para la franja franco-ibérica).
- El script de ingesta (`ml/ingest/gee_sampler.py`) consulta la colección **`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`** (data **v1.1**, NO "V2/v2.1") para los años 2017 a 2025 para cada ROI.
- Los exports se lanzan vía `ee.batch.Export.image.toCloudStorage()` con destino `gs://agrosat-data/alphaearth/{roi_name}/{year}.tif`, formato COG, compresión DEFLATE, nodata declarado.
- La tabla PostGIS `alphaearth_tiles` registra cada archivo descargado con columnas `id`, `roi_name`, `year`, `bbox` (geometría), `storage_uri`, `size_mb`, `download_date`.
- El **asset Dagster `alphaearth_annual`** maneja reintentos con backoff exponencial en caso de rate limit y registra eventos en MLflow. El lineage es visible en Dagster UI.
- Documentación en `docs/data/alphaearth.md` incluye la atribución obligatoria a Google (**CC-BY-4.0**, no "GEE ToS sin licencia abierta") y referencia a los GEE Terms of Service.

**Tareas técnicas:**

- [x] Service account con rol Earth Engine Resource Writer
- [x] Credenciales JSON en Secret Manager
- [x] Script de muestreo/descarga AlphaEarth (`ml/ingest/gee_sampler.py`, colección V1/ANNUAL)
- [ ] Migración `dbmate new create_alphaearth_tiles` + RLS por tenant
- [ ] Definir asset Dagster en `dagster_project/assets/alphaearth.py`
- [x] Documentar atribución y licencia (corregir a CC-BY-4.0 en `docs/licenses/DATA_LICENSE.md`)

**Licencia / legal:** AlphaEarth Satellite Embedding V1/ANNUAL data v1.1 — **CC-BY-4.0** (atribución a Google DeepMind). Uso académico y comercial permitido con atribución. Google Earth Engine Terms of Service (https://earthengine.google.com/terms/).

**Estimación:** 3 puntos (~1.5 días).

---

### US-007 — Pipeline de descarga Sentinel-2 L2A vía CDSE

**Como** ML Engineer,
- **quiero** descargar escenas Sentinel-2 L2A completas para consultas que requieran resolución fina,
- **para que** el agente pueda invocar visualizaciones reales de las bandas espectrales cuando el usuario pida detalle visual o el VLM necesite procesar la imagen cruda.

**Criterios de Aceptación:**

- Query STAC vía `pystac-client` contra el endpoint CDSE (`https://catalogue.dataspace.copernicus.eu/stac`).
- Filtros configurables: `eo:cloud_cover<30`, rango temporal, bandas B02/B03/B04/B05/B06/B07/B08/B8A/B11/B12 y SCL.
- Descarga concurrente controlada (máximo 4 conexiones simultáneas para respetar rate limits CDSE) con backoff exponencial en caso de 429.
- Conversión automática a COG con `rio cogeo create --profile deflate` y tiling interno 512×512.
- Almacenamiento en `gs://agrosat-data/raw/s2/{roi}/{date}/B{nn}.tif`.
- Tabla PostGIS `sentinel2_scenes` con `scene_id`, `bbox`, `datetime`, `cloud_cover`, `bands_available`, `storage_uri`.
- **Asset Dagster `sentinel2_scenes`** con reintentos y dependencia declarada de `alphaearth_annual` (misma ROI).

**Tareas técnicas:**

- [ ] Credenciales CDSE (registro gratuito en dataspace.copernicus.eu) en Secret Manager
- [ ] Script `scripts/download_s2.py` con CLI Typer
- [ ] Asset Dagster `sentinel2_scenes` con reintentos
- [ ] Dockerfile del worker con `gdal`, `rasterio`, `sentinelhub-py`, `pystac-client`
- [ ] Migración `dbmate new create_sentinel2_scenes` con índices GIST espaciales

**Licencia / legal:** Copernicus Open Access License. Atribución: "Contains modified Copernicus Sentinel data [year]".

**Estimación:** 3 puntos (~1.5 días).

---

### US-008 — Setup DINOv3-satellite (DIFERIDO) y descarga de PASTIS-R / Dynamic World

> **Estado: PARCIAL — DINOv3 DIFERIDO.** El extractor self-supervised real del proyecto es **FarSLIP** (`ml/extractors/farslip_extractor.py`, `farslip_official_extractor.py`), NO DINOv3: no existe ningún `dinov3*.py` en el repo. PASTIS-R sí está descargado y es la base de toda la segmentación (E5). La parte DINOv3 de esta US se reclasifica honestamente como **planificada/diferida** (no bloquea ningún entregable v8; el camino contrastivo lo cubre FarSLIP-fenológico, US-033..037 de E5).

**Como** ML Engineer,
- **quiero** PASTIS-R y los datasets benchmark auxiliares descargados a DVC, y —de forma diferida— DINOv3-satellite disponible como feature extractor frozen,
- **para que** los pipelines de EDA, baseline, segmentación y modelos alternativos puedan consumir estas fuentes sin esperas en tiempo de experimento.

**Criterios de Aceptación:**

- PASTIS-R descargado desde HuggingFace Datasets a `data/raw/pastis/` y versionado con DVC. **(Cumplido — base de E5.)**
- **DINOv3 (DIFERIDO, no bloqueante):** si se reactiva, checkpoint `facebook/dinov3-vitl16-pretrain-sat493m` cacheado en `gs://agrosat-models/dinov3/` tras aceptar licencia, y módulo `ml/extractors/dinov3_extractor.py` con clase `DINOv3Extractor` y método `.extract(image: np.ndarray) -> torch.Tensor` (features 1024-dim, ViT-L/16). **Hoy NO existe; el extractor frozen del proyecto es FarSLIP.**
- Dynamic World subset (regiones de interés, 2022-2025) descargado vía GEE export a `gs://agrosat-data/dynamic_world/`.
- USGS Spectral Library descargada a `data/reference/spectral_library.parquet`.
- Parcelas administrativas (GSAA / LPIS) de las regiones piloto versionadas en `data/reference/`.

**Tareas técnicas:**

- [x] Descarga + versionado DVC de PASTIS-R
- [ ] (DIFERIDO) Wrapper DINOv3 con caching en Redis (hash de imagen → features) — `ml/extractors/dinov3_extractor.py`
- [ ] (DIFERIDO) Script batch `extract_dinov3_all.py` para pre-computar sobre todos los tiles
- [ ] Scripts separados para cada dataset auxiliar (Dynamic World, USGS, GSAA)
- [ ] Tests unitarios del extractor con fixtures pequeños (cuando se implemente)

**Licencia / legal:** DINOv3 Meta AI Research License (verificar exactos en HuggingFace antes de extender a producto comercial); PASTIS-R CC-BY-SA; Dynamic World CC-BY-4.0; USGS Spectral Library Public Domain.

**Estimación:** 3 puntos (~1.5 días) si se reactiva DINOv3; 1 punto residual sin DINOv3 (auxiliares).

---

### US-009 — Catálogo STAC interno con pgstac

**Como** equipo,
- **quiero** un catálogo STAC interno queryable sobre todas las fuentes ingresadas,
- **para que** tanto los scripts de EDA como el agente LLM puedan localizar la escena o el embedding correctos mediante una sola consulta HTTP.

**Criterios de Aceptación:**

- Extensión PostGIS `pgstac` instalada en la base de datos.
- Ingest automático desde Dagster al finalizar cada asset de descarga.
- Endpoint FastAPI `GET /stac/search` con filtros `bbox`, `datetime`, `collection`, `query` siguiendo la especificación STAC API (cruza con US-053 de E8, `/stac/search` del producto).
- Índice GIST sobre `geometry` y BTREE sobre `datetime` para latencia <100 ms en queries típicas.
- Documentación OpenAPI 3.1 auto-generada.

**Tareas técnicas:**

- [ ] Instalar pgstac y crear collections (`alphaearth`, `sentinel-2-l2a`, `sentinel-1-grd`, `dynamic-world`, `sen4agrinet-catalonia`)
- [ ] Endpoint FastAPI con validación Pydantic
- [ ] Tests de integración con fixtures de escenas mock

**Estimación:** 3 puntos (~1.5 días).

---

**Subtotal EPIC 1: 12 story points** (US-006..009, ingesta base). La transferencia multi-región (HCAT, Sen4AgriNet, EuroCropsML, México, WorldCereal, licencias) vive ahora en **EPIC 12** (US-074..079).

---

## EPIC 2: Análisis Exploratorio de Datos y Transferencia Multi-Región {#epic-2}

**Objetivo.** Producir un análisis exploratorio riguroso que responda las diez preguntas guía de la rúbrica del Avance 1 (CERRADO) y, en su evolución v8, **abrir el dataset más allá de Francia**: armonizar el espacio de etiquetas PASTIS hacia una taxonomía global (HCAT v3), reconciliar el filtro de dominancia per-patch que pide el sponsor, e ingerir subconjuntos de datasets multi-región (Sen4AgriNet, EuroCropsML, WorldCereal) para demostrar transfer learning real. El EDA deja de ser solo "Data Understanding" del Avance 1 y se convierte en la **base de datos del copiloto replicable a otras zonas**.

**Alineado con.** Avance 1 (3-may-2026, EDA CERRADO) + Avance 6 (14-jun) y Avance 7 (21-jun) para la historia de transferencia multi-región (§7 del v8).

**Entregable.** Repositorio GitHub con los notebooks EDA ejecutados (Avance 1) + crosswalk taxonómico documentado en `docs/data/hcat_crosswalk.md` + adapters de ingestión de datasets multi-región versionados con DVC + atribuciones de licencia actualizadas.

**Puntos totales de la épica: 14 (v6, CERRADO) + 30 (v8 multi-región) = 44 story points.** Motor principal de DataFrames: **Polars 1.x**.

> **Nota de realidad (7-jun-2026):** el bloque EDA original (US-010..013) está **RESUELTO** y entregado en el Avance 1, con dos notebooks extra de EDA cross-dataset (`02d_eda_breizhcrops.ipynb`, `02e_eda_metodos_paper.ipynb`). El objetivo de transferencia multi-región del §7 del v8 (crosswalk HCAT, Sen4AgriNet, EuroCropsML, México, WorldCereal) se materializa ahora en **EPIC 12** (US-074..079), y el filtro 3:1 que lo conecta con FarSLIP vive en **EPIC 5** (US-032).

---

### US-010 — Notebook EDA univariado sobre Sentinel-2 crudo

> **Estado: RESUELTA** (entregado en Avance 1; `notebooks/eda/02a_eda_sentinel2.ipynb` ejecutado con outputs poblados). Caracterización estadística de las 10 bandas Sentinel-2 e índices espectrales para las regiones italianas, con conclusiones que alimentaron el Feature Engineering del EPIC 3.

**Como** Data Scientist,
- **quiero** un notebook que caracterice estadísticamente cada banda Sentinel-2 y cada índice espectral derivado para las tres regiones italianas,
- **para que** el Avance 1 cubra exhaustivamente las diez preguntas de la rúbrica del curso.

**Criterios de Aceptación (mapeados 1:1 con rúbrica Avance 1):**

- Análisis de valores faltantes usando la capa SCL (Scene Classification Layer) y detección de patrones de ausencia por región y temporada.
- Estadísticas resumidas por banda (**computadas con Polars**): media, desviación estándar, mínimo, máximo, percentiles 5/25/50/75/95.
- Detección de outliers por banda con IQR y con Isolation Forest sobre muestras de 100k píxeles estratificados por clase.
- Cardinalidad de variables categóricas (clases de cultivo en PASTIS y Dynamic World).
- Análisis de distribuciones por banda (histogramas, pruebas Shapiro-Wilk, tests de normalidad) y evaluación de necesidad de transformaciones Box-Cox o Yeo-Johnson.
- Identificación de tendencias temporales: curva NDVI mensual promedio 2022-2025 por clase de cultivo y por región.
- Evaluación de si las imágenes requieren normalización para visualización (stretch 2-98 percentil, ejemplos visuales).
- Conclusiones concretas que justifiquen las decisiones del EPIC 3 de Feature Engineering.

**Tareas técnicas:**

- [x] Notebook `notebooks/eda/02a_eda_sentinel2.ipynb` con ejecución secuencial
- [x] Muestreo estratificado con Polars (evitar OOM sobre 180 GB de Sentinel-2)
- [x] Visualizaciones con matplotlib, folium (mapas interactivos) y plotly
- [x] Sección final "Conclusiones y decisiones para FE"

**Estimación:** 5 puntos (~2.5 días).

---

### US-011 — Notebook EDA sobre embeddings AlphaEarth

> **Estado: RESUELTA** (docs/us-resolved/us-011.md). `notebooks/eda/02b_eda_alphaearth.ipynb` ejecutado: caracterización de las 64 dimensiones AlphaEarth, t-SNE/UMAP por clase y top-10 dimensiones discriminativas. **Corrección de realidad v8:** el asset GEE es `SATELLITE_EMBEDDING/V1/ANNUAL` data **v1.1** (NO "v2.1"), global incluido México, licencia **CC-BY-4.0**.

**Como** Data Scientist,
- **quiero** caracterizar las 64 dimensiones de los embeddings AlphaEarth para las tres regiones italianas,
- **para que** entienda qué información semántica llevan y cuáles dimensiones son más discriminativas para el tipo de cultivo.

**Criterios de Aceptación:**

- Visualización 2D de los embeddings con t-SNE y UMAP, coloreada por clase de cultivo según GSAA italiano.
- Matriz de correlación entre las 64 dimensiones (heatmap) para detectar redundancia.
- Distribución por dimensión (histogramas, QQ plots) para verificar si vienen pre-normalizadas por DeepMind.
- Análisis de estabilidad temporal del embedding de una misma parcela entre 2022 y 2025.
- Identificación preliminar de las diez dimensiones más discriminativas usando feature importance de Random Forest contra labels GSAA.
- Comparativa visual entre AlphaEarth embedding y NDVI clásico para la misma parcela.

**Tareas técnicas:**

- [x] Notebook `notebooks/eda/02b_eda_alphaearth.ipynb` secuencial
- [x] Muestreo estratificado de 100k píxeles por región y clase
- [x] Parches de visualización reutilizables

**Estimación:** 4 puntos (~2 días).

---

### US-012 — Análisis bivariado, multivariado y temporal

> **Estado: RESUELTA** (docs/us-resolved/us-012.md). `notebooks/eda/02c_eda_bivariado_temporal.ipynb` + `02c_eda_pastis.ipynb` ejecutados: correlaciones Pearson/Spearman, VIF, fenología y clusterización temporal DTW. **Ampliación de realidad v8:** se sumaron `02d_eda_breizhcrops.ipynb` (EDA cross-dataset BreizhCrops, precursor del crosswalk US-074) y `02e_eda_metodos_paper.ipynb`, que el v6 no contemplaba.

**Como** Data Scientist,
- **quiero** cuantificar las correlaciones entre bandas, índices espectrales y labels, más un análisis de fenología,
- **para que** las variables redundantes se identifiquen antes del EPIC 3 y la separabilidad temporal de los cultivos quede documentada.

**Criterios de Aceptación:**

- Matrices de correlación Pearson y Spearman entre las 10 bandas Sentinel-2 y los 17 índices espectrales (computadas con Polars).
- Análisis VIF (Variance Inflation Factor) para detectar multicolinealidad.
- Gráficos de pares (pairplot seaborn) por clase de cultivo.
- Análisis bivariado categórico: tipo de cultivo vs pico de NDVI, distribución de timing de pico por clase.
- Análisis temporal: ACF/PACF del NDVI por parcela, clusterización temporal con DTW (`tslearn`), identificación de mono-cultivo vs doble ciclo.
- Detección de anomalías temporales (años secos vs normales) cruzando con ERA5.

**Tareas técnicas:**

- [x] Notebook `notebooks/eda/02c_eda_bivariado_temporal.ipynb`
- [x] Funciones utilitarias en `ml/analysis/correlations.py`
- [x] Gráficos exportados como PNG de alta resolución para el anexo
- [x] (Extra v8) `notebooks/eda/02d_eda_breizhcrops.ipynb` y `02e_eda_metodos_paper.ipynb` para EDA cross-dataset

**Estimación:** 3 puntos (~1.5 días).

---

### US-013 — Dashboard Streamlit de EDA y reporte PDF

> **Estado: RESUELTA** (docs/us-resolved/us-013.md). Cerrada Fase A el 2026-05-16; **scope real 5 puntos** (cambio de scope mid-fase + design system premium + deuda técnica). Dashboard Streamlit de 6 tabs + export PDF WeasyPrint + notebook integrador `notebooks/eda/Avance1.Equipo17.ipynb`. Fase B (FastAPI + Nuxt + ECharts) diferida al backlog.

**Como** equipo,
- **quiero** un dashboard ejecutable y un reporte PDF que resuman el EDA,
- **para que** la rúbrica del Avance 1 valore las conclusiones claramente y el sponsor pueda revisar el trabajo sin abrir notebooks.

**Criterios de Aceptación:**

- Dashboard Streamlit `app/eda_dashboard.py` con seis tabs: univariado Sentinel-2, AlphaEarth, bivariado, temporal, espacial (mapa folium), conclusiones. **Rehecho como 5 fichas notebook + 1 tab mapa espacial** (ver cambio de scope en handoff).
- Exportación PDF de las conclusiones vía `weasyprint` o `reportlab` para anexar al Avance 1.
- Conclusiones explícitas y mapeadas al contexto CRISP-ML(Q) Data Understanding.

**Tareas técnicas:**

- [x] Dashboard Streamlit con navegación (6 tabs, design system Data-Dense Dashboard, KPI cards, narrativa por figura)
- [x] Función `export_report_pdf()` (`ml/report/export_pdf.py` CLI Typer + Jinja2 + WeasyPrint, requiere GTK3 en Windows)
- [x] Integración con notebooks vía papermill (notebook integrador `notebooks/eda/Avance1.Equipo17.ipynb` generado por `scripts/build_avance1_notebook.py`, ejecutable con `make eda-notebook-avance1`)

**Entregables adicionales (no en plan original)**:
- Módulo DRY `ml/report/notebook_content.py` + `figure_narratives.py` consumido por los 3 canales (dashboard, PDF, notebook)
- Subset compacto `data/reference/pastis_tiles_dissolved.geojson` (506 KB) para mapa folium sin DVC
- Setup Streamlit Community Cloud aislado en `deploy/streamlit/`
- Fase B (FastAPI + Nuxt + ECharts) documentada en `docs/product-backlog/eda-dashboard-fase-b-nuxt.md`

**Cierre**: ver [`docs/us-resolved/us-013.md`](../docs/us-resolved/us-013.md).

**Estimación:** 2 puntos (~1 día). **Real: 5 puntos** (cambio de scope mid-fase + design system premium + 4 bugs Fase A v2 + 3 Mayores deuda técnica resueltos antes del PR).

---

**Subtotal EPIC 2: 14 story points** (US-010..013, EDA — CERRADO/Avance 1). La transferencia multi-región se trasladó a **EPIC 12** (US-074..079); el filtro 3:1 (insumo de FarSLIP) vive en **EPIC 5** (US-032).

---

## EPIC 3: Feature Engineering e Índices Espectrales {#epic-3}

> **Estado de la épica (corte 7-jun-2026): COMPLETADA en el Avance 2.** Las cinco US (US-014..018) están cerradas o avanzadas; el banco de features tabular (189-dim) + temporal + multisensor alimentó el baseline (EPIC 4) y la segmentación (EPIC 5). La rama FarSLIP (US-017) dejó de ser una ablación negativa y pasó a **camino principal por directiva del sponsor**: su bug de prototipos de texto (`ml/farslip/train.py:~184` usa `torch.randn`) y su fine-tune fenológico-contrastivo se ejecutan ahora en el EPIC 5 (US-029..033) sobre la H100 NVL 96GB ya disponible. Esta épica conserva su alcance histórico de Avance 2 sin nuevas US v8.

**Objetivo.** Convertir los datos crudos en features listos para modelado, cubriendo los cuatro criterios de la rúbrica del Avance 2 (Construcción 30 pts, Normalización 30 pts, Selección/Extracción 30 pts, Conclusiones 10 pts).

**Alineado con.** Avance 2 (17 de mayo de 2026).

**Puntos totales de la épica: 18** (14 baseline + 4 SP de US-017 FarSLIP).

---

### US-014 — Biblioteca de 17 índices espectrales con justificación agronómica

> **Estado: RESUELTA** ([`docs/us-resolved/us-014.md`](../docs/us-resolved/us-014.md)). Cerrada 2026-05-16. 17 índices vectorizados con justificación agronómica documentada, tests con valores de referencia y tabla académica en `docs/spectral_indices.md`.

**Como** equipo,
- **quiero** calcular al menos 17 índices espectrales estándar sobre Sentinel-2 con justificación documentada,
- **para que** el criterio "Construcción" de la rúbrica del Avance 2 (30 pts) quede cubierto con profundidad.

**Criterios de Aceptación:**

- Implementación vectorizada con `eemont` (sobre Google Earth Engine) y `spyndex` de los siguientes índices con sus justificaciones agronómicas documentadas en docstring: **NDVI** (vigor), **NDWI** (contenido de agua en hoja), **NDMI** (humedad de canopy), **EVI** (vigor mejorado en canopy denso), **SAVI** (vigor ajustado por suelo), **MSAVI2** (versión mejorada SAVI), **NBR** (detección de estrés por fuego/sequía), **MCARI** (clorofila en canopy), **CCCI** (clorofila corregida por canopy), **LAI** (Leaf Area Index), **FAPAR** (fracción de radiación absorbida), **PSRI** (senescencia), **NDCI** (clorofila en ambientes acuáticos agrícolas), **GCVI** (green chlorophyll), **RENDVI** (Red-Edge NDVI), **NDRE** (Red-Edge NDVI para cultivos densos), **TSAVI** (SAVI transformado).
- Módulo `ml/features/spectral_indices.py` con API consistente: cada índice es una función que acepta un `xarray.DataArray` con bandas como dimensión y devuelve un `xarray.DataArray` con el índice computado.
- Soporte para cálculo sobre series temporales (axis=time) con reduce.
- Cache en Redis con clave `{scene_id}:{index_name}` para evitar recómputo.
- Tests unitarios con valores de referencia conocidos (e.g., NDVI de píxel de bosque caducifolio en junio debe estar entre 0.7 y 0.9).

**Tareas técnicas:**

- [x] Función `compute_index(da: xr.DataArray, index: str) -> xr.DataArray`
- [x] Tabla de referencias académicas por índice en `docs/spectral_indices.md`
- [x] Tests con fixtures sintéticos y fixtures reales de una parcela demo

**Estimación:** 4 puntos (~2 días).

---

### US-015 — Features temporales agregados por parcela

> **Estado: RESUELTA** ([`docs/us-resolved/us-015.md`](../docs/us-resolved/us-015.md)). Cerrada 2026-05-17. Estadísticos temporales, armónicos FFT y features fenológicos por parcela (`ml/features/temporal_features.py`) persistidos en `features_parcels`; insumo del baseline tabular del EPIC 4.

**Como** ML Engineer,
- **quiero** features temporales agregados a nivel parcela (armónicos FFT, percentiles, fenología),
- **para que** los modelos baseline del EPIC 4 puedan capturar dinámica temporal sin necesidad de arquitecturas temporales explícitas.

**Criterios de Aceptación:**

- Estadísticos temporales por índice espectral a lo largo del ciclo vegetativo (**computados con Polars LazyFrame**): media, std, min, max, percentiles 5/25/50/75/95.
- Descomposición harmónica (FFT) con las primeras tres componentes de frecuencia (amplitud y fase).
- Features fenológicos derivados: fecha de inicio del verdor (día en que NDVI cruza 0.3 ascendente), fecha de pico, valor del pico, fecha de senescencia, integral AUC del NDVI sobre el ciclo completo.
- Features derivativos: pendiente NDVI pre-pico, pendiente post-pico, duración del período de madurez.
- Todos los features disponibles en tabla `features_parcels` en PostgreSQL con UNIQUE `(parcel_id, year)`.

**Tareas técnicas:**

- [x] Función `extract_temporal_features(parcel_timeseries: xr.DataArray) -> pl.DataFrame`
- [x] Migración `dbmate new create_features_parcels` (+ `create_parcels` como precondición del FK)
- [x] Tests contra parcela demo con curva NDVI conocida

**Estimación:** 3 puntos (~1.5 días).

---

### US-016 — Fusión multisensor a nivel parcela

> **Estado: RESUELTA** ([`docs/us-resolved/us-016.md`](../docs/us-resolved/us-016.md)). Vector multisensor 189-dim por parcela (`ml/features/fusion.py`) + splits espaciales K-fold + scaler global. Versionado DVC de scaler/splits diferido a sub-US **US-016.1** (backlog). El banco FarSLIP 512-dim se materializa en EPIC 5 (US-033..037, ahora con prototipos fenológicos reales, no aleatorios).

**Como** ML Engineer,
- **quiero** un vector de features combinado por parcela,
- **para que** los modelos de EPIC 4-6 consuman una tabla única con features heterogéneos ya alineados.

**Criterios de Aceptación:**

- Vector combinado con las siguientes componentes por parcela: 64 dimensiones AlphaEarth (media sobre la parcela; asset GEE `SATELLITE_EMBEDDING/V1/ANNUAL` v1.1, CC-BY-4.0), 17 índices espectrales con sus estadísticos temporales (5 stats × 17 = 85 features), backscatter Sentinel-1 VV y VH con sus stats temporales (5 × 2 = 10 features), elevación media, pendiente media y orientación dominante desde SRTM DEM, temperatura media mensual y precipitación acumulada mensual desde ERA5 (24 features), geometría: superficie en ha, perímetro en m, elongación (3 features).
- **Banco FarSLIP fine-grained:** 512 dimensiones de embeddings producidos por la rama de destilación parche-a-parche descrita en US-017. **Nota v8:** el banco se genera de forma definitiva en EPIC 5 (US-033..037) una vez corregido el bug de prototipos de texto y aplicada la directiva fenológica del sponsor; el consumo tabular de este banco en EPIC 6 se materializa vía el ensamble dual-head E-a (US-041) más que como columnas planas.
- Shape final aproximado: 64 + 85 + 10 + 3 + 24 + 3 + 512 (FarSLIP opcional) = **189 features tabulares clásicos por parcela + 512-dim FarSLIP**. Los modelos pueden consumir el vector completo o sólo el subset tabular según ablation declarada en el notebook.
- Normalización z-score global con estadísticos guardados en `artifacts/scaler_v1.pkl`.
- Split train/val/test estratificado espacialmente (K=5 folds por regiones no contiguas) guardado en `data/splits/`.
- **Fusión implementada con Polars `LazyFrame`** para eficiencia en memoria.

**Tareas técnicas:**

- [x] Script `scripts/build_parcel_features.py` con asset Dagster (CLI Typer + 3 assets Dagster `parcel_features_fused`, `parcel_splits_spatial_kfold`, `parcel_features_scaler` compartiendo `ml/features/fusion.py` — DRY)
- [x] Spatial train-test split con `geopandas` y tessellation (`ml/features/spatial_split.py` con tessellation H3 res 5 + KMeans + buffer 1 km)
- [ ] Guardar scaler y splits con versionado DVC (deferido a sub-US **US-016.1** — bucket `gs://agrosat-dvc-remote` pendiente; backlog en `docs/product-backlog/us-016-1-dvc-multisensor-outputs.md`)

**Estimación:** 3 puntos (~1.5 días).

---

### US-017 — Destilación FarSLIP parche-a-parche sobre crops Sentinel-2

> **Estado: RESUELTA (infraestructura) — evolucionada a camino principal en EPIC 5.** La infra de destilación está completa y fiel a Li et al. 2025: [`ml/farslip/distill.py`](../ml/farslip/distill.py) (716 LOC, `PatchDistillationLoss` §3.2 + `RegionCategoryAlignmentLoss` InfoNCE §3.3) y checkpoints reales `FarSLIP1/2_ViT-B-16.pt`. **Hallazgo crítico v8:** el contrastivo se alineaba contra **ruido** — [`ml/farslip/train.py`](../ml/farslip/train.py) (~línea 184) inicializa `text_prototypes` con `torch.randn(...)` (prototipos frozen aleatorios sin vocabulario semántico). Por eso FarSLIP perdía vs AlphaEarth (0.163 vs 0.233). La **directiva del sponsor** ordena llenar exactamente ese hueco con descripciones fenológicas (Wen et al. 2025) generadas por Gemini Flash → contrastivo → fine-tune incremental 4→18 clases con filtro 3:1 Meadow per-patch → ensamble TSViT-pheno + FarSLIP. Ese trabajo se ejecuta en las nuevas US **US-033..037** (EPIC 5) y desemboca en el ensamble **E-a (US-041, EPIC 6)**. Ver §4 del [plan v8](RefinamientoPlaneacionAgroSatCopilot_v8.md).

**Como** ML Engineer,
- **quiero** entrenar una rama de adaptación CLIP siguiendo la técnica FarSLIP (Li et al., 2025) sobre crops Sentinel-2 de las tres regiones italianas,
- **para que** el pipeline disponga de embeddings fine-grained de 512 dimensiones que mejoren la cuantificación sub-parcela y alimenten tanto el banco de features de US-016 como el ensamble dual-head con TSViT en EPIC 5/6.

**Criterios de Aceptación:**

- Implementación de la pérdida de destilación parche-a-parche y de la alineación región-categoría basada en token CLS, siguiendo el procedimiento del paper (arXiv:2511.14901). **[Completado en `ml/farslip/distill.py`.]**
- Backbone teacher: CLIP ViT-B/16 pretrained; student: ViT-B/16 con las mismas dimensiones, inicializado desde teacher y fine-tuneado sobre crops Sentinel-2 256×256 px etiquetados con texto agronómico generado a partir de las clases CAP italianas.
- Dataset interno `data/farslip_pairs/` con al menos 30,000 pares imagen-texto cubriendo Pianura Padana, Toscana y Puglia. **[v8: el etiquetado de texto evoluciona de clases CAP genéricas a descripciones fenológicas reales por clase/parcela vía Gemini Flash — `ml/features/phenology_description.py` + `ml/features/phenology_class_prototypes.py` — para corregir el `torch.randn`. Ver US-029.]**
- Entrenamiento sobre GPU con MLflow run `farslip-clip-italy-v1`. **[v8: la H100 NVL 96GB del sponsor está disponible (VM `gjcamacho-gpuh1`, env micromamba `agrosat`, `F:\projects\agrosat-copilot`); la ablación de bandas y el incremental 4→18 corren ahí (US-031/032), con L4 spot como fallback.]**
- Outputs: pesos student en `gs://agrosat-models/farslip/`, módulo `ml/extractors/farslip_extractor.py` (o equivalente `ml/farslip/extract_embeddings.py`) con extractor de embeddings.
- Métrica de calidad de la adaptación: cerrar el gap previo (0.163 vs AlphaEarth 0.233) con embeddings FarSLIP-pheno; silhouette/cluster por clase reportado honesto. **[v8: claim ajustado — objetivo es superar el 0.163 propio y acercarse a 0.233 en su espacio, NO prometer mejora open-vocabulary de +5 pp sin medición apples-to-apples. Validación en US-037 con el harness único de métrica US-030.]**

**Tareas técnicas:**

- [x] Reproducir la lógica de destilación parche-a-parche del paper en `ml/farslip/distill.py`
- [x] Asset / pipeline de entrenamiento FarSLIP con dependencia de crops Sentinel-2 (`ml/farslip/dataset.py`, `train.py`, `extract_embeddings.py`)
- [x] Suite de tests unitarios para la pérdida y para la alineación CLS
- [ ] **(→ US-033/US-034, EPIC 5)** Reemplazar `torch.randn` en `train.py:~184` por prototipos fenológicos reales (Gemini Flash → MiniLM 384-dim → reproyección frozen 384→512) vía `set_text_prototypes()`
- [ ] **(→ US-031, EPIC 5)** Ablación de bandas (RGB vs falso-color NIR-R-G vs 4-band-pheno) en H100
- [ ] **(→ US-032, EPIC 5)** Incremental n-clase 4→18 con curriculum + filtro 3:1 Meadow per-patch

**Estimación:** 4 puntos (~2 días). *(El trabajo residual del fix fenológico + ablación + incremental se re-estima en EPIC 5: US-033=3, US-034=5, US-031=5, US-032=5, US-037=3 SP.)*

---

### US-018 — Selección, extracción y normalización

> **Estado: RESUELTA** ([`docs/us-resolved/us-018.md`](../docs/us-resolved/us-018.md)). Cerrada 2026-05-21. Filtrado (VarianceThreshold, correlación, chi², ANOVA F), extracción (PCA 95%, Factor Analysis, UMAP) y normalización justificada en `ml/features/selection.py` + `encoding.py`; entregable integrador `Avance2.Equipo17.ipynb` con sección de conclusiones CRISP-ML(Q).

**Como** ML Engineer,
- **quiero** aplicar métodos de filtrado y extracción con justificación empírica,
- **para que** el criterio "Selección/extracción" y "Normalización" de la rúbrica del Avance 2 (30+30 pts) quede cubierto.

**Criterios de Aceptación:**

- Métodos de filtrado ejecutados y documentados: VarianceThreshold (elimina features con varianza <0.01), correlación (remueve un feature de cada par con |r|>0.95), chi-cuadrado para categóricas, ANOVA F-score para numéricas.
- Métodos de extracción ejecutados: PCA con análisis de varianza explicada (objetivo 95%), Factor Analysis para firmas espectrales, UMAP 2D para visualización.
- Feature importance de Random Forest y XGBoost entrenados sobre todos los features como complemento.
- Tabla comparativa antes/después con métricas F1-macro y mIoU cross-validadas con split espacial.
- Transformaciones numéricas justificadas: StandardScaler para modelos lineales/SVM, MinMax para redes neuronales, Yeo-Johnson para variables sesgadas (como NDVI que puede ser negativo), log-transform para LAI y biomasa.
- Sección "Conclusiones CRISP-ML(Q) Data Preparation" al final del notebook.

**Tareas técnicas:**

- [x] Notebook secuencial de feature engineering (`notebooks/feature_engineering/03b_fe_spectral_temporal_pastis.ipynb` + `03a` + `03c`; entregable integrador `Avance2.Equipo17.ipynb`)
- [x] Funciones reutilizables en `ml/features/selection.py` (11 funciones publicas) + `ml/features/encoding.py` (codificacion categorica)
- [x] Reporte tabular antes/despues (`reports/feature_selection/before_after.csv` + `.md`)

**Estimación:** 4 puntos (~2 días).

---

**Subtotal EPIC 3: 18 story points** (14 originales + 4 de US-017 FarSLIP). **Estado: 18/18 SP entregados en Avance 2** (US-016.1 DVC y el fine-tune fenológico de FarSLIP se rastrean fuera de esta épica: US-016.1 en backlog, US-033..037 en EPIC 5).

---

## EPIC 4: Baseline — AlphaEarth + XGBoost/RF {#epic-4}

> **Estado: ÉPICA RESUELTA Y REENCUADRADA** (Avance 3, 20-may-2026). Todas las US (US-019, US-020, US-021, US-022, US-022-b, US-022-c, US-023-preview) están cerradas en `docs/us-resolved/`. El baseline tabular cumplió la rúbrica del A3 y, en el corte v8 (6-jun), se reencuadró con el **hallazgo de producto 6 familias HCAT**: XGBoost salta de **0.4365 (18 clases) a 0.6535 (6 familias HCAT)**, cruzando la meta F1-macro ≥ 0.60. El baseline tabular XGBoost-AlphaEarth es ahora el **base learner tabular del ensamble E-b** (US-042, EPIC 6) y la **semilla del transfer few-shot multi-región** (US-076 EuroCropsML, EPIC 12). No se reabre cómputo en esta épica.

**Objetivo.** Construir un baseline sólido sobre features tabulares AlphaEarth + índices espectrales con Random Forest y XGBoost, cubriendo los cinco criterios de la rúbrica del Avance 3 (Algoritmo 40 pts, Características 20 pts, Sub/sobreajuste 10 pts, Métrica 20 pts, Desempeño 10 pts).

**Alineado con.** Avance 3 (20 de mayo de 2026).

**Hipótesis clave.** Dado que AlphaEarth ya encapsula información multisensor en 64 dimensiones compactas, un baseline tabular sobre estas dimensiones debe superar en F1-macro a baselines clásicos que usan únicamente bandas Sentinel-2 crudas, alcanzando la meta de F1-macro ≥ 0.60 sobre PASTIS-R. **Confirmada en v8 a nivel familia de cultivo:** el desbalance long-tail de las 18 clases PASTIS (Meadow ~45 %) ocultaba la señal; al colapsar a 6 familias HCAT el baseline cruza 0.60 (0.6535). El cuello de botella era el espacio de etiquetas, no las features.

**Corrección de realidad v8 (verificada 7-jun-2026).** AlphaEarth = asset GEE `SATELLITE_EMBEDDING/V1/ANNUAL` **data v1.1** (NO "v2.1"), cobertura global incluido México, licencia **CC-BY-4.0** (no usar el ID v2.1 en docs). La H100 NVL 96GB del sponsor está disponible, pero esta épica es **CPU/RTX 4070 only** — no consume horas H100.

**Puntos totales de la épica: 10** (+ 17 SP de US-022-b + 14 SP de US-023-preview, ambas transversales E4/E5, ya contabilizadas en su propia secuenciación).

---

### US-019 — Random Forest y XGBoost sobre features combinados

> **Estado: RESUELTA** (docs/us-resolved/us-019.md). RF y XGBoost entrenados sobre AlphaEarth 64-dim; XGBoost full = 0.4094 (18 clases) en A3 y reencuadre v8 a **0.6535 con 6 familias HCAT**, cruzando la meta ≥ 0.60. Runs MLflow `baseline-rf-alphaearth-v1` y `baseline-xgb-alphaearth-v1` registrados.

**Como** ML Engineer,
- **quiero** entrenar Random Forest y XGBoost sobre el vector de features del EPIC 3,
- **para que** el criterio "Algoritmo" (40 pts) de la rúbrica quede justificado con elección interpretable, robusta a outliers, bajo costo computacional y con feature importance nativa.

**Criterios de Aceptación:**

- Entrenamiento en GCP L4 spot / RTX 4070 local con scikit-learn (RandomForestClassifier) y XGBoost 2.1 (XGBClassifier).
- Justificación documentada en el notebook: AlphaEarth ya codifica información multisensor; RF/XGB sobre estos 64-dim es un baseline fuerte, interpretable y computacionalmente barato; sirve como lower bound para evaluar viabilidad.
- Métricas reportadas: F1-macro (principal), F1-weighted, mIoU (para segmentación a nivel píxel), accuracy, Cohen's kappa.
- Hyperparameter tuning ligero con GridSearchCV (5-fold spatial CV).
- Desempeño mínimo declarado: F1-macro ≥ 0.60 sobre PASTIS-R. Si no se alcanza con 18 clases, el notebook documenta las causas probables (long-tail Meadow ~45 %) y las decisiones para el EPIC 5; **el hallazgo v8 demuestra que a 6 familias HCAT se alcanza 0.6535**.
- Modelos finales registrados en MLflow con runs `baseline-rf-alphaearth-v1` y `baseline-xgb-alphaearth-v1`.

**Tareas técnicas:**

- [x] Script `ml/train/train_baseline.py` con CLI
- [x] MLflow autologging para RF y XGB
- [x] Serialización de modelos con joblib en MLflow artifacts

**Estimación:** 3 puntos (~1.5 días).

---

### US-020 — Feature importance y análisis SHAP

> **Estado: RESUELTA** (docs/us-resolved/us-020.md). Feature importance nativa (Gini/Gain) + SHAP `TreeExplainer` sobre top-20; confirmado empíricamente que las dimensiones AlphaEarth dominan y que `geom_*`/ERA5/SRTM son redundantes (delta F1 = 0.0).

**Como** ML Engineer,
- **quiero** identificar y visualizar los features más relevantes,
- **para que** el criterio "Características importantes" (20 pts) de la rúbrica quede justificado con interpretación y representación visual.

**Criterios de Aceptación:**

- Feature importance nativa de Random Forest (Gini) y XGBoost (Gain).
- Análisis SHAP (explainable AI) sobre top 20 features globalmente con `shap.TreeExplainer`.
- SHAP dependency plots para los cinco features más importantes.
- Identificación explícita de cuáles dimensiones AlphaEarth dominan (dato relevante para el Paper Track).
- Conclusiones que validen (o refuten) las decisiones de Feature Engineering del EPIC 3.

**Tareas técnicas:**

- [x] Notebook con SHAP waterfall y summary plots
- [x] Guardar gráficos como PNG de alta resolución
- [x] Sección de conclusiones con recomendación de ajustes a FE si aplica

**Estimación:** 2 puntos (~1 día).

---

### US-021 — Curvas de aprendizaje, validación y análisis de sub/sobreajuste

> **Estado: RESUELTA** (docs/us-resolved/us-021.md). Curvas de aprendizaje y validación + spatial CV 5-fold (mismo splitter reutilizado luego en US-022-b/US-030); diagnóstico de gap train-val documentado.

**Como** ML Engineer,
- **quiero** diagnosticar sub y sobreajuste con visualizaciones,
- **para que** el criterio "Sub/sobreajuste" (10 pts) de la rúbrica quede cubierto con evidencia gráfica.

**Criterios de Aceptación:**

- Curva de aprendizaje (accuracy train/val vs número de muestras de entrenamiento).
- Curva de validación (accuracy vs hiperparámetros críticos: `max_depth` para RF, `n_estimators` y `learning_rate` para XGB).
- Cross-validation 5-fold estratificado espacial (splits por regiones no contiguas para evitar data leakage geográfico).
- Diagnóstico explícito del gap train-val: si >10% se documenta como sobreajuste; si accuracy train y val ambos bajos se documenta como subajuste.

**Tareas técnicas:**

- [x] Funciones `plot_learning_curve` y `plot_validation_curve`
- [x] Documentación del criterio de spatial CV

**Estimación:** 2 puntos (~1 día).

---

### US-022 — Notebook secuencial y comparativa vs Sentinel-2 crudo

> **Estado: RESUELTA** (docs/us-resolved/us-022.md). `notebooks/baseline/04_baseline.ipynb` ejecutable end-to-end (papermill CI verde); comparativa AlphaEarth puro vs Sentinel-2 crudo vs vector combinado; F1-macro v1 = 0.32 (18 clases, cumple rúbrica A3). Reubicado a `notebooks/baseline/` en US-023-preview.

**Como** equipo,
- **quiero** un notebook `notebooks/baseline/04_baseline.ipynb` ejecutable de principio a fin más una comparativa AlphaEarth vs Sentinel-2 crudo,
- **para que** el criterio de libreta secuencial de la rúbrica se cumpla y el valor incremental de AlphaEarth quede documentado empíricamente.

**Criterios de Aceptación:**

- Notebook secuencial que ejecuta todas las celdas sin intervención manual.
- Tabla comparativa: RF+XGB sobre (a) AlphaEarth 64-dim puro, (b) Sentinel-2 crudo (10 bandas medias), (c) vector combinado completo del EPIC 3.
- Métrica principal F1-macro + otras dos métricas relevantes + tiempo de entrenamiento.
- Discusión de resultados y conclusiones para el EPIC 5.

**Tareas técnicas:**

- [x] Ejecución secuencial validada con papermill en CI
- [x] Exportación de resultados a tabla LaTeX para uso futuro en Paper Track

**Estimación:** 3 puntos (~1.5 días).

---

**Subtotal EPIC 4 (US-019..022): 10 story points.**

---

### US-022-b — Deuda técnica FarSLIP + infra GCP L4 + reencuadre fenológico (post-A3, transversal E4/E5) {#us-022b}

> **Estado: RESUELTA** (docs/us-resolved/us-022b.md · cierre de ciclo en docs/us-resolved/us-022-c.md). Reencuadre fenológico (ADR-006 Aceptada): se confirmó que el modelo aprende fenología y no geografía (descartar `geom_*`/ERA5/SRTM no degrada, delta = 0.0); XGBoost full = 0.4094 (+0.089 vs baseline 0.32). Infra GCP L4 (`ml-train` Dockerfile + Terraform SAs + Cloud Run MLflow scale-to-zero) operativa.

**Status:** cerrada 2026-05-23 · [docs/us-resolved/us-022b.md](../docs/us-resolved/us-022b.md) · cierre completo del ciclo en [us-022-c](../docs/us-resolved/us-022-c.md).

**Motivación:** [ADR-006 Aceptada](../docs/decisions/ADR-006-reencuadre-baseline-fenologico.md) — el baseline tabular RF/XGB (US-022 F1-macro 0.32) cumple la rúbrica del A3 pero NO es el baseline conceptualmente correcto para la clasificación de cultivos (problema fenológico-temporal). Wen et al. (2025) confirma la vía vía descripción fenológica + LLM. Restricción hardware del momento: cuota GCP `NVIDIA_L4_GPUS=1`, sin A100/H100 (A100=0, A2_CPUS=0 en 7 regiones). **Nota v8:** esta restricción ya no aplica — la H100 NVL 96GB del sponsor está disponible (VM `gjcamacho-gpuh1`); la deuda FarSLIP de esta US se ejecuta finalmente como camino principal en EPIC 5 (US-033..037), no como ablación negativa.

**Entregables (17 SP, 5 sub-US):**
- 022b-A infra GCP L4 (`ml-train` Dockerfile + Terraform SAs + Cloud Run MLflow scale-to-zero + bucket versioned).
- 022b-B FarSLIP Fase 4 (US-017 AC-3/4/6/8) → transferida a `us-022-c` P1 (GCP dedicada).
- 022b-C reencuadre FE: ablation 5+ conjuntos, TempCNN + InceptionTime portados nativos a `ml/models/temporal.py`, clustering sin coordenadas.
- 022b-D rama semántica: Gemini 3.5 Flash + sentence-transformers + bloque `pheno_text_*` via LEFT JOIN en `fusion.py`.
- 022b-E ADR-006 Aceptada + esta referencia.

**Hipótesis confirmadas empíricamente:**
- C-2 (Dr. Camacho): descartar `geom_*` no degrada F1 (delta = 0.0); el modelo aprende fenología, no geografía.
- C-2 extendida: quitar ERA5 + SRTM tampoco degrada (delta = 0.0); AlphaEarth ya los codifica.
- C-4: XGBoost full = 0.4094 (+0.089 vs baseline 0.32). Gate mínimo pasado.

**Cobertura del diff:** 87 % (1148 stmts, 147 miss) · 117/117 tests passing.

**Cómputo:** 0 h H100, 0 h Vertex AI directo en esta US; smoke local 22-may + 23-may en GPU local. Gemini smoke real ~$0.001 USD.

---

### US-023-preview — Correcciones al baseline previo a EPIC 5 (post-A3, transversal E4/E5) {#us-023-preview}

> **Estado: RESUELTA** (plan en docs/us-planning/us-023-preview.md · ejecución en docs/manual-test/us-023-preview-execution.md · auditoría v2 en docs/audit/us-023-preview-v2-audit.md). Las 9 observaciones cerradas: rutas/builders saneados (P1), FarSLIP materializado en path canónico con fix naming (P2), ablation `geom_only` aislada (P3), módulo `spectral_signature.py` + 16 tests (P5), QA notebooks (P6), baseline v2 con 3 modelos sobre conjunto ganador (P8) y categoría Baseline en dashboard Streamlit (P9). P4 (Gemini Flash pheno_text) quedó como SKIP-DOC honesto por `GEMINI_API_KEY` ausente en la corrida solo-dev — deuda absorbida por la rama FarSLIP-fenológica de EPIC 5 (US-033).

**Status:** resuelta · plan canónico en [`docs/us-planning/us-023-preview.md`](../docs/us-planning/us-023-preview.md) · handoff en [`docs/us-handoff/us-023-preview.md`](../docs/us-handoff/us-023-preview.md).

**Motivación:** auditoría del 25-may detecta 9 observaciones sobre `notebooks/baseline/04_baseline.ipynb` y `notebooks/baseline/05_reencuadre_fenologico.ipynb` (movidos desde `notebooks/feature_engineering/`): rutas y builders desactualizados, FarSLIP no materializado en el path canónico (`data/farslip/embeddings_italy.parquet`) por lo que la ablation omite `with_farslip` y `farslip_only` (bug adicional: discrepancia de naming `farslip_emb_XXX` en parquet vs `farslip_XXX` esperado en `fusion.py:582`), falta comparativa visual aislada `full` vs `no_geom`, falta ablation cuantitativa del bloque `pheno_text_*` (Gemini Flash 3.5) sobre subset ≥ 1 000 parcelas, falta evaluar un descriptor compacto de firma espectral, falta validar estándar [`notebooks/CLAUDE.md`](../notebooks/CLAUDE.md), falta reentrenar los **3 modelos baseline (XGBoost + TempCNN + InceptionTime)** sobre el conjunto ganador post-ablation, y el dashboard Streamlit no expone los resultados del baseline. Esta US **no entrega Avance nuevo** — sanea el baseline post-A3 para que EPIC 5 (US-023 U-Net en adelante) arranque sobre conjuntos de features y modelos ya validados.

**Como** ML Engineer,
- **quiero** cerrar las 9 observaciones (P1 rutas + P2 FarSLIP en path canónico con fix naming + P3 ablation `geom_only` aislada + P4 ablation real `pheno_text` Gemini Flash 3.5 + P5 descriptor de firma espectral + P6 cumplimiento `notebooks/CLAUDE.md` + P8 baseline v2 con 3 modelos + P9 categoría "Baseline" en dashboard Streamlit),
- **para que** el conjunto de features ganador, los 3 modelos baseline reentrenados y los resultados visuales queden cuantificados y publicados antes de iniciar el modelado denso, reduciendo iteración en EPIC 5 y alimentando el stacking de EPIC 6.

**Criterios de Aceptación:**

- P1: Builders (`scripts/build_baseline_notebook.py`, `scripts/build_reencuadre_notebook.py`) y `Makefile` apuntan a `notebooks/baseline/*.ipynb`; `make notebooks-check` exit 0; `notebooks/CLAUDE.md` §"Estructura Canónica" actualizada.
- P2: `data/farslip/embeddings_italy.parquet` materializado (promoción de v2 = extracción real epoch_2, commit `0f01255`) + renombrado de columnas `farslip_emb_XXX → farslip_XXX` para alinear con contrato `fusion.py:582` + patch defensivo en `_build_farslip_block` para aceptar ambos prefijos + DVC tracked + tag git `farslip-embeddings-italy-v1` creado (gate B-4 US-022-c cerrado retroactivamente); ablation reporta `with_farslip` y `farslip_only` con delta vs `full` documentado y MLflow run `baseline-farslip-ablation-v1` con tags `data_version` + `code_version`.
- P3: Plot aislado `ablation_geom_comparison.png` con 2 barras (`full` vs `no_geom`) + nuevo conjunto `geom_only` en la ablation (gate F1-macro < 0.10) + interpretación de leakage espacial en el notebook.
- P4: Bloque `pheno_text_*` (384-dim sentence-transformers) ampliado a subset balanceado ≥ 1 000 parcelas; ablation reporta `with_pheno_text` y `pheno_text_only`; costo Gemini Flash 3.5 ≤ $5 USD documentado en `docs/l4_log.md`; MLflow run `baseline-pheno-text-ablation-v1`.
- P5: `ml/features/spectral_signature.py` con `SpectralSignatureFeatures(BaseEstimator, TransformerMixin)` y descriptor Red Edge Position (Frampton et al. 2013) por defecto; integrado en `ml/features/fusion.py` como bloque opcional con LEFT JOIN; ablation reporta `with_spectral_signature` y `spectral_signature_only`; 6+ tests pytest con cobertura ≥ 80 %.
- P6: Las 2 libretas pasan el QA Checklist completo de `notebooks/CLAUDE.md` (16 ítems: imports + autoreload, Polars, idioma strings vs identificadores, `display()` sobre `print()`, sin emojis decorativos, conclusiones sin US-XXX/EPIC/AC-X, paths via `pathlib`, etc.).
- P7: Plan v6 referencia US-023-preview en EPIC 4 + secuenciación S6.
- **P8 (baseline v2):** `notebooks/baseline/04_baseline.ipynb` v2 reentrena los 3 modelos del A3 (XGBoost + TempCNN + InceptionTime) sobre el conjunto de features ganador post-ablation P2/P3/P4/P5, con spatial CV 5-fold (mismo splitter US-022b), 4 fixes ML preservados (class_weights, weighted_sampler, lr_scheduler warmup+cosine, early_stopping); tabla `model_comparison_v2.parquet` con 3 modelos × 6 métricas (F1-macro, F1-weighted, mIoU, accuracy, kappa, train_time_s); plot `model_comparison_v2.png` con deltas vs v1; 3 MLflow runs (`baseline-v2-xgb`, `baseline-v2-tempcnn`, `baseline-v2-inceptiontime`) con tags `data_version` + `code_version`; DVC tag `fused-features-italy-v2`; tabla LaTeX `baseline_v2_comparison.tex` exportada a `paper/tables/us-023-preview/`; decisión "modelo ganador v2" documentada (F1-macro → F1-weighted → mIoU como tiebreak); wall clock ≤ 90 min RTX 4070.
- **P9 (dashboard Streamlit):** nueva categoría `_SECTION_BASELINE = "Baseline (US-023-preview)"` agregada al selector en [`app/eda_dashboard.py`](../app/eda_dashboard.py); función `_render_baseline_section()` con 5 tabs: (1) Ablation de features (7-10 conjuntos), (2) Leakage geográfico (`geom_only` vs `full`), (3) Bloques opcionales (FarSLIP + Gemini + firma espectral + decisiones), (4) Modelos baseline v2 (3 modelos + comparativa v1 vs v2 + ganador), (5) Conclusiones (H-1..H-4 + Lo que sigue en EPIC 5); reusa helpers `_render_section_divider`/`_render_figures_section`/`_render_tables_section`; lazy loading con `st.cache_data` sobre `pl.read_parquet(...)`; graceful degradation si algún artefacto no existe (`st.warning("ejecuta make reencuadre-notebook-full && make baseline-v2-full")`); smoke test en `tests/app/test_eda_dashboard_baseline_section.py`; sin nuevas dependencias en `pyproject.toml`.
- Cobertura ML del diff ≥ 75 %; `make check` limpio; PR a `develop` con Conventional Commit `feat(E4): US-023-preview …`.

**Tareas técnicas:**

- [x] P1 rutas + builders + Makefile + `notebooks/CLAUDE.md` (cerrado 2026-05-25)
- [x] P2 promover FarSLIP v2 al path canónico + rename cols + patch fusion.py + gate B-4 cerrado retroactivamente (cerrado 2026-05-25; DVC tag y MLflow run pendientes de gate B-2 training)
- [x] P3 plot `ablation_geom_comparison.png` + conjunto `geom_only` + narrativa "Por qué descartar `geom_*`" (cerrado 2026-05-25)
- [SKIP-DOC] P4 Gemini Flash 3.5 — skip honesto documentado en `docs/l4_log.md`: GEMINI_API_KEY no configurada en entorno solo-dev de esta corrida; bloque 216-parcelas US-022-c sigue siendo la referencia. **Deuda absorbida por la rama FarSLIP-fenológica de EPIC 5 (US-033..037) en v8**, donde los prototipos de fenología por clase se generan con Gemini Flash y se inyectan en el loss contrastivo.
- [x] P5 módulo `spectral_signature.py` + 16 tests pytest (cobertura efectiva en la clase >= 80%) + integración fusion.py + bloque opcional + ablation `with_spectral_signature`/`spectral_signature_only` (cerrado 2026-05-25)
- [x] P6 QA `notebooks/CLAUDE.md` sobre `04_baseline.ipynb` + `05_reencuadre_fenologico.ipynb` (cerrado 2026-05-25 — papermill smoke + full ejecutados)
- [x] P7 entrada US-023-preview en plan v6 + secuenciación S6 (2026-05-25)
- [x] P8 baseline v2 con 3 modelos sobre conjunto ganador + builder de celdas v2 (RUN_BASELINE_V2 toggle) + target `make baseline-v2-full` (cerrado 2026-05-25; corrida real de 90 min lanzada con `make baseline-v2-full`)
- [x] P9 categoría "Baseline" en `app/eda_dashboard.py` + 5 tabs + smoke test (cerrado por el frontend-engineer)
- [x] `docs/us-resolved/us-023-preview.md` / ejecución registrada en `docs/manual-test/us-023-preview-execution.md`

**Estimación:** 14 puntos (~5-6 días distribuidos en S6).

**Hipótesis validadas:**

- H-1 (FarSLIP): incluir embeddings FarSLIP 512-dim mejora F1-macro ≥ +0.02 sobre `full` → promover al baseline; si delta ∈ [-0.02, +0.02] → base learner del stacking EPIC 6; si peor → descartar con justificación. **Resultado v8:** FarSLIP perdió como bloque tabular (0.163 vs AlphaEarth 0.233) por el bug `torch.randn` en los prototipos de texto; reabierto como camino contrastivo-fenológico denso en EPIC 5 (US-033..037) por directiva del sponsor.
- H-2 (`pheno_text`): la rama semántica mejora F1-macro ≥ +0.01 → promover; si no → deuda. **Resultado:** no ayudó como clasificador tabular; validado luego por el patrón "Be My Eyes" (Huang et al. 2025) — el texto sirve para comunicar al reasoner, no para clasificar píxeles.
- H-3 (firma espectral): descriptor Red Edge Position aporta ≥ +0.01 F1-macro → promover; si no → deuda documentada.
- H-4 (leakage `geom_*`): F1-macro de `geom_only` < 0.10 confirma que las 3 columnas son solo proxy de región. **Confirmada.**

**ADR de referencia:** [ADR-006 Aceptada](../docs/decisions/ADR-006-reencuadre-baseline-fenologico.md).

**Hallazgo de producto v8 (6 familias HCAT).** El reencuadre del espacio de etiquetas a **6 familias HCAT** (colapso del long-tail de 18 clases PASTIS, ~45 % Meadow) eleva el XGBoost-AlphaEarth de **0.4365 → 0.6535 F1-macro**, cruzando la meta ≥ 0.60 (ver [`docs/STATUS.md`](../docs/STATUS.md) §"Hallazgo de producto 6 familias HCAT"). Este es el insumo del **crosswalk taxonómico PASTIS-18 → HCAT v3 (US-074, EPIC 12)** y del **ensamble E-b (US-042, EPIC 6)**, donde el baseline tabular XGBoost-AlphaEarth entra al stacking a nivel parcela.

**Cómputo:** 0 h H100, 0 h Vertex AI, 0 h L4. Trabajo local (CPU + RTX 4070) + 1 llamada cloud (Gemini Flash 3.5, ≤ $5 USD). Wall clock P8 baseline v2 (3 modelos): ≤ 90 min en RTX 4070 batch=128.

**Artefactos finales:** 6 MLflow runs (3 ablations + 3 baseline v2), DVC tags (`farslip-embeddings-italy-v1`, `phenology-text-italy-v1`, `spectral-signature-italy-v1`, `fused-features-italy-v2`), 2 notebooks `notebooks/baseline/*.ipynb` con outputs poblados, figuras `paper/figures/us-023-preview/*.png`, tabla LaTeX `paper/tables/us-023-preview/baseline_v2_comparison.tex`, categoría "Baseline" en dashboard Streamlit con 5 tabs, gate B-4 de US-022-c cerrado retroactivamente.

---

**Subtotal EPIC 4: 10 story points** (US-019..022) **+ 31 SP transversales** (US-022-b 17 SP + US-023-preview 14 SP, contabilizados en su propia secuenciación E4/E5). **Toda la épica está RESUELTA al corte v8 (6-jun-2026).**

---

## EPIC 5: Modelos Alternativos — Seis Arquitecturas + FarSLIP-fenológico {#epic-5}

**Objetivo.** Construir seis modelos individuales diversos de segmentación densa (mínimo requerido por la rúbrica del Avance 4), normalizarlos bajo un único harness de métrica, compararlos apples-to-apples y ejecutar la directiva del sponsor: convertir FarSLIP de ablación negativa a camino principal mediante alineación contrastiva con descripciones fenológicas reales (Gemini Flash), re-entrenar TSViT full en H100 y dejar listos los insumos (softmax/OOF) para los ensambles de EPIC 6.

**Alineado con.** Avance 4 (31-may-2026, cerrado) — seis segmentadores. Avance 5 (mié 10-jun-2026) — FarSLIP-fenológico + tabla normalizada. Avance 6 (14-jun) — TSViT full retrain. Rúbrica A4: Comparativa 60 pts + Ajuste fino 30 pts + Modelo individual final 10 pts.

**Arquitecturas seleccionadas (realidad verificada 7-jun-2026):**

1. U-Net con backbone ResNet-50 pretrained (CNN clásica, spatial only) — entrenada, mIoU 0.2423.
2. DeepLabv3+ con backbone MobileNetV3 (CNN eficiente con ASPP) — entrenada, mIoU 0.2709.
3. SegFormer-**B0** (Transformer de segmentación, **3 bandas RGB**, no B2 ni 10 bandas) — entrenada por Isaac, mIoU 0.2325.
4. U-TAE — Temporal Attention Encoder (baseline temporal de referencia, 20 clases) — entrenada, mIoU 0.4742; portada a `ml/models/utae.py`.
5. TSViT — Vision Transformer factorizado temporal-espacial (Paper 1 del profesor) — entrenada (recortada en L4), mIoU 0.6215; variante **TSViT-pheno** mIoU **0.6253 / F1 0.7500** (mejor modelo individual).
6. **AnySat — SUSTITUYE a Swin-UNETR** (este último nunca se entrenó). FM multi-modal/multi-resolución; entrenada, mIoU 0.4459. Formalizado como decisión: AnySat ocupa la 6.ª silla "Transformer moderno SITS" de la rúbrica.

> **Cambio de realidad clave (v8).** La H100 NVL 96GB del sponsor (VM `gjcamacho-gpuh1`, entorno micromamba `agrosat`, repo en `F:\projects\agrosat-copilot`) ya está disponible, reabriendo TSViT full retrain. El cuello de botella ya no es VRAM sino **tiempo (~3 semanas a la presentación 27-jun)**. La directiva del sponsor (junta) **ordena** llevar FarSLIP a camino principal: contrastivo con descripciones fenológicas Gemini Flash → fine-tune, incremental 4→18 clases, filtro 3:1 Meadow per-patch, y ensamble TSViT-pheno + FarSLIP.

> **Numeración v8 (final).** Los seis modelos resueltos del Avance 4 son US-023 (U-Net), US-024 (DeepLabv3+), US-025 (SegFormer-B0), US-026 (U-TAE), US-027 (TSViT) y US-028 (AnySat); US-029 es la comparativa/ajuste fino del A4. El trabajo FarSLIP-fenológico + métrica es US-030 (harness), US-031 (softmax/OOF), US-032 (filtro 3:1), US-033 (prototipos fenología), US-034 (fix `torch.randn`), US-035 (ablación de bandas), US-036 (incremental 4→18), US-037 (eval FarSLIP-pheno), US-038 (TSViT full retrain) y US-039 (TSViT-pheno full retrain). El orden de ejecución vive en el Backlog Priorizado, no en los números.

---

### US-023 — Modelo 1: U-Net con ResNet-50

> **Estado: RESUELTA** (implementada en `ml/models/segmentation.py`; entregada en Avance 4). U-Net ResNet-50 entrenada sobre patches Sentinel-2 spatial-only; mIoU fold-4 = 0.2423 (baseline CNN spatial-only confirmado). Pendiente: re-score bajo el harness único de US-030.

**Como** ML Engineer,
- **quiero** entrenar U-Net sobre patches 256×256 de una imagen Sentinel-2 sin dimensión temporal,
- **para que** dispongamos de un baseline denso CNN spatial-only contra el cual comparar arquitecturas temporales.

**Criterios de Aceptación:**

- Backbone ResNet-50 pretrained en ImageNet; head U-Net con skip connections.
- Loss combined CrossEntropy + Dice con pesos {0.5, 0.5}.
- Entrenamiento en GCP L4 con batch 8 y Automatic Mixed Precision BF16.
- Métricas reportadas: mIoU, pixel accuracy, F1 por clase.
- Run MLflow `alt-unet-resnet50-v1`.

**Tareas técnicas:**

- [x] Script `ml/train/train_unet.py` usando `segmentation_models.pytorch`
- [x] Pipeline de datos con `WebDataset` para streaming de patches
- [x] Early stopping con patience 5

**Estimación:** 3 puntos (~1.5 días).

---

### US-024 — Modelo 2: DeepLabv3+ con MobileNetV3

> **Estado: RESUELTA** (implementada en `ml/models/deeplabv3plus.py`; entregada en Avance 4). DeepLabv3+ entrenada; mIoU fold-4 = 0.2709. Pendiente: re-score bajo el harness único de US-030.

**Como** ML Engineer,
- **quiero** entrenar DeepLabv3+ como alternativa eficiente,
- **para que** tengamos una CNN con ASPP (Atrous Spatial Pyramid Pooling) en la comparativa.

**Criterios de Aceptación:**

- Backbone MobileNetV3-Large pretrained; head DeepLabv3+ con ASPP rates {6, 12, 18}.
- Mismo pipeline de datos y loss que US-023.
- Run MLflow `alt-deeplabv3plus-mobilenet-v1`.

**Tareas técnicas:**

- [x] Reusar pipeline de datos de US-023
- [x] Configurar backbone desde `segmentation_models.pytorch`

**Estimación:** 2 puntos (~1 día).

---

### US-025 — Modelo 3: SegFormer-B0 (3 bandas RGB) y trazabilidad FarSLIP open-vocabulary

> **Estado: RESUELTA con corrección de realidad** (SegFormer entrenado por Isaac; DeepLab + TSViT + TSViT-pheno cierran el set de Avance 4). La variante entregada es **SegFormer-B0 sobre 3 bandas RGB** (no B2 sobre 10 bandas como planteaba v6). El cabezal open-vocabulary FarSLIP NO se acopló dentro de SegFormer; FarSLIP se aborda como rama propia y camino principal en US-033..US-041 (directiva del sponsor). mIoU fold-4 SegFormer-B0 = 0.2325.

**Como** ML Engineer,
- **quiero** un representante Transformer de segmentación spatial-only en la comparativa,
- **para que** la tabla incluya arquitecturas CNN y Transformer bajo idéntico protocolo de evaluación.

**Criterios de Aceptación:**

- Variante **SegFormer-B0** pretrained, head adaptado al espacio de clases PASTIS contiguo.
- **Realidad documentada:** entrada de **3 bandas RGB** (B04-B03-B02), no 10 bandas; esto explica parte de la brecha de mIoU frente a U-Net/DeepLab y es exactamente el tipo de asimetría que el harness de US-030 debe neutralizar al re-scoring.
- El acoplamiento FarSLIP open-vocabulary originalmente planteado aquí se **desacopla** y migra a la rama FarSLIP propia (US-033..US-041); la trazabilidad con el extractor FarSLIP de US-017 se preserva como antecedente.
- Run MLflow `alt-segformer-b0-rgb-v1`.

**Tareas técnicas:**

- [x] Cargar SegFormer desde `transformers.SegformerForSemanticSegmentation`
- [x] Adaptar head al espacio de clases PASTIS
- [ ] Documentar en notebook la diferencia 3-band RGB vs 10-band (caveat de comparabilidad para US-030)

**Estimación:** 3 puntos (~1.5 días).

---

### US-026 — Modelo 4: U-TAE

> **Estado: RESUELTA** (entregada en Avance 4; portada a `ml/models/utae.py`, warm-start Optuna documentado en memoria del proyecto). U-TAE entrenado sobre series temporales PASTIS; mIoU fold-4 = 0.4742 (20 clases). Pendiente: re-score 18-clase contiguo bajo el harness de US-030.

**Como** ML Engineer,
- **quiero** entrenar U-TAE sobre las series temporales Sentinel-2,
- **para que** el baseline temporal de referencia de PASTIS esté en la comparativa.

**Criterios de Aceptación:**

- Implementación oficial de VSainteuf (`utae-paps`) integrada en el repo.
- Input: T=20 observaciones × 10 bandas × H × W.
- Positional encoding temporal absoluto.
- Entrenamiento en **H100 ventana V2** (ahora disponible; en la práctica se completó en L4 con warm-start).
- Run MLflow `alt-utae-v1`.

**Tareas técnicas:**

- [x] Clonar repo oficial y adaptarlo al pipeline del proyecto
- [x] Configurar dataloader para secuencias temporales PASTIS-R

**Estimación:** 3 puntos (~1.5 días).

---

### US-027 — Modelo 5: TSViT (Paper 1 del profesor)

> **Estado: RESUELTA (entrenamiento recortado L4)** (entregada en Avance 4; `ml/models/tsvit_wrapper.py` + rama fenológica en `ml/models/pheno_semantic_branch.py`). TSViT mIoU fold-4 = 0.6215; **TSViT-pheno = 0.6253 (mejor individual)**. El re-entrenamiento full en H100 se formaliza como US-034/US-035.

**Como** ML Engineer,
- **quiero** replicar TSViT con el encoder temporal-espacial factorizado y múltiples cls tokens,
- **para que** implementemos directamente la propuesta del Paper 1 del profesor como contribución al benchmark y como componente del ensemble final.

**Criterios de Aceptación:**

- Reproducción fiel de Tarasiou et al. 2023: temporal encoder → spatial encoder factorizado.
- Múltiples cls tokens (K clases PASTIS) separables entre encoders.
- Positional encoding temporal por fecha real de adquisición (tabla aprendida).
- Entrenamiento en **H100 ventana V2** (entregado recortado en L4: T=10, batch limitado; full retrain → US-034).
- Métricas alineadas con el paper (≥ estado del arte en PASTIS, dentro del techo de saturación ~70%).
- Run MLflow `alt-tsvit-v1`.

**Tareas técnicas:**

- [x] Clonar repo oficial del paper y adaptar a pipeline del proyecto
- [x] Verificar reproducción contra el número reportado en el paper
- [x] Integrar en la comparativa

**Estimación:** 5 puntos (~2.5 días).

---

### US-028 — Modelo 6: AnySat (sustituye Swin-UNETR)

> **Estado: RESUELTA con sustitución formalizada** (`ml/models/anysat_wrapper.py`). **Swin-UNETR NUNCA se entrenó**; AnySat lo sustituye como representante Transformer/FM moderno 2024-2025 para SITS y ocupa la 6.ª silla de la rúbrica. AnySat mIoU fold-4 = 0.4459. Decisión a sincronizar en `CLAUDE.md`/`AGENTS.md` (Decisiones Irrevocables, EPIC 5).

**Como** ML Engineer,
- **quiero** un representante Transformer/FM moderno (2024-2025) para series temporales satelitales,
- **para que** la comparativa incluya estado del arte reciente sin depender de Swin-UNETR (no entrenado).

**Criterios de Aceptación:**

- **AnySat** (FM multi-modal/multi-resolución) integrado y entrenado sobre PASTIS-R como sustituto formal de Swin-UNETR.
- Entrenamiento en H100/L4; checkpoint y métricas registrados.
- Run MLflow `alt-anysat-v1`.
- Sustitución documentada en `CLAUDE.md`/`AGENTS.md` y en el notebook de comparativa.

**Tareas técnicas:**

- [x] Integrar AnySat como `ml/models/anysat_wrapper.py`
- [x] Adaptar input a series temporales PASTIS-R
- [ ] Sincronizar la sustitución AnySat↔Swin-UNETR en los espejos `CLAUDE.md`/`AGENTS.md`

**Estimación:** 2 puntos (~1 día).

---

### US-029 — Comparativa, ajuste fino top-2 y modelo individual final

> **Estado: RESUELTA (versión Avance 4)** (notebook `notebooks/05_alt_models.ipynb` con tabla comparativa y ajuste fino Optuna entregados). **Deuda crítica heredada:** la tabla mezcla 3 pipelines, 18 vs 20 clases y resoluciones 64/128/256 → el orden no es apples-to-apples. Su normalización se eleva a US-030 (harness único) como prerequisito de todo ensamble.

**Como** equipo,
- **quiero** una tabla comparativa ordenada por métrica principal y un ajuste fino bayesiano de los dos mejores modelos,
- **para que** los criterios "Comparativa" (60 pts), "Ajuste fino" (30 pts) y "Modelo individual final" (10 pts) de la rúbrica del Avance 4 queden cubiertos.

**Criterios de Aceptación:**

- Tabla comparativa con columnas: modelo, F1-macro, F1-weighted, mIoU, accuracy, tiempo de entrenamiento (min), tiempo de inferencia (ms/imagen), número de parámetros.
- Ajuste fino con Optuna (≥30 trials) sobre los dos mejores según F1-macro (TSViT-pheno y TSViT).
- Selección justificada del modelo individual final con trade-offs documentados.
- Notebook `notebooks/05_alt_models.ipynb` secuencial.
- **Caveat registrado:** la comparabilidad real se difiere a US-030 (re-score fold-5 con harness único).

**Tareas técnicas:**

- [x] Script `ml/tune/optuna_tune.py` con storage persistente en PostgreSQL
- [x] Tabla comparativa auto-generada desde MLflow API
- [x] Sección de conclusiones con recomendación para EPIC 6

**Estimación:** 2 puntos (~1 día).

---

### US-030 — Harness único de métrica de segmentación (re-score apples-to-apples)

**Como** ML lead que defiende la comparativa ante el jurado,
- **quiero** una sola función que re-evalúe todos los checkpoints `best.pt` con idéntica definición de métrica, esquema de clases, resolución y convención de void,
- **para que** el orden de modelos (hoy en parte artefacto de definición: 18 vs 20 clases, 64/128/256 px, present-only vs torchmetrics macro, SegFormer-B0 a 3 bandas) sea defendible y sirva de base a todos los ensambles.

**Criterios de Aceptación:**

- Una sola función re-evalúa todos los `best.pt` de los 6 modelos sobre **fold-5 (held-out)**, no fold-4 (que fue selección).
- Esquema **18-clase contiguo**, misma definición torchmetrics mIoU / F1-macro, **resolución 128 NEAREST**, misma convención de void/ignore, 10 bandas (donde aplique).
- Reutiliza `dense_confusion_matrix` + `dense_metrics_from_cm` de US-025 (`ml/eval/dense_metrics.py`); regenera las filas hardcodeadas de la tabla A4.
- Es **prerequisito declarado de toda US de ensamble** (US-031, US-040, US-041, US-042).
- Sin fuga de datos: folds oficiales PASTIS respetados; normalización solo sobre train.

**Tareas técnicas:**

- [ ] Función `rescore_all_checkpoints()` en `ml/eval/dense_metrics.py` que itere los 6 `best.pt`
- [ ] Mapeo 20→18 clases contiguo para U-TAE; remuestreo 128px NEAREST homogéneo
- [ ] Regenerar tabla comparativa (CSV + figura) reportando fold-5
- [ ] Tests pytest en `tests/ml/eval/` que fijen la definición de métrica (golden values)

**Estimación:** 5 puntos (~2.5 días).

---

### US-031 — Regenerar softmax / OOF de segmentación desde `best.pt`

**Como** ingeniero de ensambles,
- **quiero** persistir los logits softmax por-píxel y las predicciones OOF (con CV espacial) de cada modelo a partir de su `best.pt`,
- **para que** los ensambles de EPIC 6 dispongan de insumos reproducibles (hoy `ml/ensemble/` está vacío y no existe softmax/OOF guardado para segmentación).

**Criterios de Aceptación:**

- Para cada modelo se persisten softmax por-píxel + predicciones OOF con **spatial-CV** (folds oficiales PASTIS) en `ml/eval/oof/*.parquet`.
- Convención de clases idéntica a US-030 (18-clase contiguo, void homogéneo).
- Los artefactos OOF son la entrada de Voting/Bagging/Stacking/Blending (US-040) y de los ensambles incrementales E-a/E-b (US-041/US-042).
- Tamaño y formato documentados (parquet por modelo/fold); regenerable por script.

**Tareas técnicas:**

- [ ] `ml/eval/oof/dump_oof.py` que reuse el harness de US-030 para inferencia OOF
- [ ] Guardar softmax post-softmax (no logits) por convención anti-fuga del ensamble
- [ ] Validar reconciliación píxel↔parcela para los miembros tabulares

**Estimación:** 3 puntos (~1.5 días).

---

### US-032 — Reconciliar filtro 3:1 Meadow per-patch + `n_classes` con `pastis_filter.py` de Isaac

**Como** equipo de datos ejecutando la directiva del sponsor,
- **quiero** extender el `PastisFilter` de Isaac (hoy modo coverage-threshold) con un modo `dominance_ratio` per-patch parametrizado por `n_classes`,
- **para que** el filtro 3:1 aclarado por el sponsor (mantener patch si Meadow ≤ 3× la 2.ª clase EN ESE PATCH) alimente el protocolo incremental sin romper los splits existentes.

**Criterios de Aceptación:**

- `PastisFilter` (en branch `origin/user/iavila/pastis-preparation-for-farslip`, base correcta) extendido con modo **`dominance_ratio`**: mantener patch si `Meadow_px ≤ ratio × (2.ª clase más grande EN ESE PATCH)`, default ratio=3.
- Función **parametrizada por `n_classes`** (acepta el conjunto de clases objetivo dinámicamente, directiva sponsor).
- Modo `coverage` / `min_coverage` legacy **preservado** (no romper `pastis_farslip_filtered_splits.json`).
- Histograma por-patch vía `np.bincount(target[0].ravel())` sobre `TARGET_*.npy (3,128,128)` uint8, canal 0 (0=Background, 1-18 cultivos, 19=Void).
- **NO** mezclar con `abocanegra/semana-5` (84 commits detrás, 11 conflictos).
- Tests verde en `tests/ml/data/`.

**Tareas técnicas:**

- [ ] Añadir `mode="dominance_ratio"` y `ratio` a `ml/data/pastis_filter.py`
- [ ] Parametrizar `n_classes` / `target_classes`
- [ ] Tests del filtro 3:1 contra un patch de referencia conocido

**Estimación:** 5 puntos (~2.5 días).

---

### US-033 — Prototipos de fenología reales (18 clases) vía Gemini Flash → embeddings

**Como** científico de datos cumpliendo la directiva del sponsor,
- **quiero** generar descripciones fenológicas reales por clase (Gemini Flash) y codificarlas a embeddings,
- **para que** FarSLIP alinee su pérdida contrastiva contra un vocabulario semántico verdadero y no contra ruido aleatorio.

**Criterios de Aceptación:**

- `generate_class_prototypes()` produce `phenology_class_prototypes_pastis.parquet` (18 filas: `class_id`, descripción ES, `emb_000..383`).
- Reutiliza `ml/features/phenology_description.py` (Wen et al. 2025, Gemini Flash, temperature=0, cache).
- Codificación MiniLM 384-dim (sentence-transformers frozen).
- Cache determinista SHA256; costo ≈ $0.0018, dentro de ~$115/mes.
- Texto por parcela disponible como variante (descripción individual según curva NDVI real, cacheada por `parcel_id`).

**Tareas técnicas:**

- [ ] `ml/features/phenology_class_prototypes.py` con loop 18 clases
- [ ] `encode_descriptions()` MiniLM 384-dim
- [ ] Persistir parquet + cache SHA256; test de determinismo

**Estimación:** 3 puntos (~1.5 días).

---

### US-034 — Fix `torch.randn` → inyectar prototipos fenológicos en `RegionCategoryAlignmentLoss`

**Como** ingeniero ML que arregla el gap exacto que FarSLIP perdía,
- **quiero** reemplazar la inicialización aleatoria de `text_prototypes` (`ml/farslip/train.py:184` usa `torch.randn`) por la carga de los prototipos fenológicos reales,
- **para que** la pérdida contrastiva InfoNCE alinee contra semántica verdadera y FarSLIP supere el 0.163 previo.

**Criterios de Aceptación:**

- `ml/farslip/train.py` (~177-186) reemplaza `torch.randn(...)` por carga de `phenology_class_prototypes_pastis.parquet` → `np.tile(emb,(n_regions,1))` → `(n_regions·n_categories, 384)` → `trainer.set_text_prototypes()`.
- `set_text_prototypes()` aplica reproyección **frozen ortogonal 384→512** (`_proto_to_clip_proj`); encoder de texto frozen (`_load_text_encoder`).
- Entrenamiento corre sin OOM en H100; la loss contrastiva baja monótono.
- Caveat registrado: la reproyección 384→512 es aproximación cruda (solución limpia = CLIP text encoder nativo, post-entrega).

**Tareas técnicas:**

- [ ] Reemplazar bloque `torch.randn` en `train.py` por inyección de prototipos
- [ ] Implementar `_proto_to_clip_proj` (ortogonal frozen) y `set_text_prototypes()` en `ml/farslip/distill.py`
- [ ] Run MLflow `farslip-pheno-fix-v1` con curva de loss

**Estimación:** 5 puntos (~2.5 días).

---

### US-035 — Ablación de bandas FarSLIP (RGB vs falso-color NIR-R-G vs 4-band) en H100

**Como** equipo que quiere un argumento de selección que la rúbrica premia,
- **quiero** correr tres variantes de bandas de entrada de FarSLIP (rgb / nir_rgb / 4band-pheno),
- **para que** se cuantifique el aporte de la señal NIR y de la fenología real sobre la calidad de embeddings.

**Criterios de Aceptación:**

- 3 corridas en MLflow: `baseline-rgb` (B04-B03-B02, protos random), `baseline-nir` (B08-B04-B03 falso color, protos random), **`4band-pheno`** (B02-B03-B04-B08, fenología real).
- Tabla que cuantifica el aporte de NIR/fenología (apuesta honesta: `4band-pheno` gana).
- `band_selection: Literal["rgb","nir_rgb","4band"]` + `select_and_reorder_bands()` (índices B02=0, B03=1, B04=2, B08=3).
- Camino 4-band adapta `patch_embed.proj` de 3→4 canales inicializando el canal NIR como `mean(RGB)` (evita neurona muerta).
- Confianza ~95%; ~10 h H100; fallback L4 disponible.

**Tareas técnicas:**

- [ ] Añadir `band_selection` al CLI de `ml/farslip/train.py` y a `distill.py`
- [ ] Adaptación 3→4 canales del `patch_embed.proj`
- [ ] Unit test de indexado de bandas vs parcela de referencia (mitiga mismatch 4 vs 10 bandas)

**Estimación:** 5 puntos (~2.5 días).

---

### US-036 — Protocolo incremental n-clase 4→18 con curriculum + filtro 3:1

**Como** equipo siguiendo la prueba incremental ordenada por el sponsor,
- **quiero** entrenar FarSLIP en dos etapas (4 clases filtradas 3:1 → 18 clases full, init desde Stage-1),
- **para que** se demuestre el aprendizaje incremental de clases con curriculum sin colapsar la convergencia.

**Criterios de Aceptación:**

- **Stage-1:** 4 clases [Meadow=1, Soft winter wheat=2, Corn=3, Grapevine=8], patches filtrados 3:1 (US-032), 2 epochs, 4 prototipos.
- **Stage-2:** 18 clases full, init desde Stage-1 (`load_state_dict(strict=False)`), 2 epochs, 18 prototipos.
- Checkpoints y métricas por stage; convergencia documentada.
- **Fallback honesto:** si Stage-2 degrada, entrenar 18-clase desde cero; validar con POC 2-epoch antes de comprometer schedule.
- Confianza ~70% (convergencia Stage-1 incierta con set pequeño ~500-700 patches/fold).

**Tareas técnicas:**

- [ ] `scripts/train_incremental.py` con `create_incremental_dataset(n_classes)`
- [ ] Lógica Stage-1→Stage-2 init `strict=False`
- [ ] POC 2-epoch de validación de convergencia antes del run full

**Estimación:** 5 puntos (~2.5 días).

---

### US-037 — Evaluación FarSLIP-pheno vs AlphaEarth (cierra gap 0.163 vs 0.233)

**Como** equipo que rinde cuentas honestas al sponsor,
- **quiero** evaluar los embeddings FarSLIP-pheno contra el baseline AlphaEarth y reportar el cierre del gap,
- **para que** quede documentado si FarSLIP-pheno supera el 0.163 previo y cuánto se acerca a AlphaEarth (0.233) en su espacio.

**Criterios de Aceptación:**

- Embeddings FarSLIP-pheno superan el **0.163** previo; objetivo: acercarse/superar AlphaEarth **0.233** en su espacio de evaluación.
- Silhouette / separabilidad de cluster por clase reportadas en notebook (reusar `ml/eval/embedding_separability.py`).
- Reporte honesto (sin sobre-afirmar); métricas fold-4/5.
- Insumo directo del ensamble E-a (US-041).

**Tareas técnicas:**

- [ ] `scripts/farslip_eval_phenology.py` (nuevo) + sección de notebook
- [ ] Silhouette por clase + tabla comparativa MLflow
- [ ] Conclusión honesta sobre el cierre del gap

**Estimación:** 3 puntos (~1.5 días).

---

### US-038 — TSViT full-config retrain H100 (T completo, 128px nativo, Full-M, warmup+cosine)

**Como** ML lead aprovechando la H100 ya disponible,
- **quiero** re-entrenar TSViT con configuración full (hoy entrenado recortado en L4),
- **para que** el mejor modelo individual suba de mIoU y aporte un componente más fuerte al ensamble.

**Criterios de Aceptación:**

- Config full: T completo de la serie PASTIS, **128px nativo**, variante **Full-M**, batch grande (≈20), 40 epochs, warmup + cosine con hiperparámetros oficiales de Tarasiou 2023.
- MLflow con `data_version` + `code_version` (lineage en server Docker :5010).
- **Target realista mIoU fold-4 ≥ 0.65 (objetivo 0.68-0.72, +5.5 a +9.5 pp).** Honestidad: PASTIS-R satura ~70% (Tarasiou reportó 65.1%); **no prometer 0.75+**.
- VRAM: Full-M 128px batch 20 cabe holgado en 96GB.
- Prioridad H100 estricta: (1) FarSLIP-pheno → (2) este retrain → (3) ensambles OOF → (4) Qwen serving.

**Tareas técnicas:**

- [ ] Config full en el wrapper TSViT (`ml/models/tsvit_wrapper.py`)
- [ ] Schedule warmup+cosine; checkpointing + auto-resume en H100
- [ ] Run MLflow `alt-tsvit-fullm-v1`; re-score con harness US-030

**Estimación:** 8 puntos (~4 días).

---

### US-039 — TSViT-pheno full-config retrain H100 (rama contrastiva λ=0.3)

**Como** equipo que valida honestamente el aporte fenológico,
- **quiero** re-entrenar TSViT-pheno con la misma config full más la rama contrastiva fenológica,
- **para que** se compare contra TSViT base sin sobre-afirmar el efecto de la fenología.

**Criterios de Aceptación:**

- Misma config full que US-034 + rama fenológica (`ml/models/pheno_semantic_branch.py`), λ=0.3.
- Comparación honesta vs TSViT base: **esperar Δ ~0.3%, NO 5%** (en supervisado el modelo ya aprende las firmas de cultivo; el +0.004 actual es ruido).
- MLflow con `data_version` + `code_version`.
- Mejor checkpoint pasa al ensamble E-a (US-041).

**Tareas técnicas:**

- [ ] Activar rama contrastiva en el retrain full con λ=0.3
- [ ] Run MLflow `alt-tsvit-pheno-fullm-v1`
- [ ] Reporte honesto del Δ vs base + re-score US-030

**Estimación:** 5 puntos (~2.5 días).

---

**Subtotal EPIC 5: 68 story points** (21 baseline v6 — US-023..US-025 + alias v6 US-023v6-c..f — ya entregados; + 47 nuevos v8: US-030 harness 5, US-031 OOF 3, US-032 filtro 3:1 5, US-033 prototipos 3, US-034 fix torch.randn 5, US-035 ablación bandas 5, US-036 incremental 4→18 5, US-037 eval FarSLIP-pheno 3, US-038 TSViT full 8, US-039 TSViT-pheno full 5).

---

## EPIC 6: Modelo Final — Ensambles {#epic-6}

**Objetivo.** Construir el modelo final de máximo rendimiento mediante **ensambles** que combinan los mejores segmentadores del EPIC 5 (TSViT-pheno mIoU 0.6253 / F1 0.7500, TSViT base, U-TAE), el baseline tabular AlphaEarth+XGB del EPIC 4 y la rama contrastiva-fenológica FarSLIP (directiva del sponsor). El v8 **descarta el fine-tuning de Gemma 4 26B-MoE como modelo final** (diferido a ADR-009 future por bloqueo técnico de QLoRA sobre experts MoE 3D fused + leakage de AgroMind eval-only — ver EPIC 10 / §8 del plan v8) y lo reemplaza por **siete ensambles**: los 4 de rúbrica (Voting/Bagging/Stacking/Blending) más 3 incrementales **E-a** (TSViT-pheno + FarSLIP), **E-b** (+AlphaEarth) y **E-c** (geo-context, diseño-only). El reasoner LLM del producto pasa a ser Gemini 2.5-pro frozen (patrón "Be My Eyes": perceiver = nuestros modelos, reasoner = LLM frozen), no un VLM fine-tuned.

**Alineado con.** Avance 5 (mié 10-jun-2026) — notebook secuencial en GitHub. Rúbrica: Ensambles 60 pts + Selección 20 pts + Gráficos 20 pts. E-b + reconciliación píxel↔parcela hacia Avance 6 (14-jun); E-c documentado como trabajo futuro (ADR-010).

**Deuda crítica prerequisito (EPIC 5).** Ningún ensamble es defendible sin **US-030** (harness único de métrica, re-score apples-to-apples en fold-5) y **US-031** (regenerar softmax/OOF desde cada `best.pt`). La tabla A4 actual mezcla 3 pipelines, 18 vs 20 clases, resolución 64/128/256 — el orden TSViT≫U-TAE≫SegFormer es en parte artefacto de definición. `ml/ensemble/` está **vacío** (solo `__init__.py`) y no existe `ml/eval/oof/` a 7-jun.

**Puntos totales de la épica: 20.**

---

### US-040 — Cuatro ensambles base de rúbrica (Voting/Bagging/Stacking/Blending)

**Como** equipo de ML,
- **quiero** implementar los cuatro ensambles base de la rúbrica reusando los softmax/OOF normalizados (US-031) de los mejores segmentadores y del baseline tabular,
- **para que** el criterio "Ensambles" (60 pts) + "Selección" (20 pts) + "Gráficos" (20 pts) del Avance 5 quede cubierto con estrategias homogéneas y heterogéneas defendibles ante el jurado.

**Criterios de Aceptación:**

- Módulo poblado `ml/ensemble/{voting,bagging,stacking,blending}.py` (hoy solo existe `__init__.py` vacío).
- **Ensamble 1 — Voting homogéneo (píxel):** soft-vote por promedio de **probabilidades post-softmax** de TSViT-pheno + TSViT + U-TAE, a nivel píxel. Majority/argmax sobre el promedio; no se promedian logits.
- **Ensamble 2 — Bagging tabular (parcela):** XGBoost-AlphaEarth, N bootstraps distintos del training set + tuning Optuna; promedio de probabilidades de parcela.
- **Ensamble 3 — Stacking heterogéneo (parcela):** base learners `{TSViT-pheno↓parcela, U-TAE↓parcela, XGBoost-AlphaEarth}` → meta-learner (XGBoost / regresión logística) entrenado **solo** sobre predicciones OOF con CV **espacial** (folds oficiales PASTIS). Swin-UNETR descartado (nunca entrenado); Gemma 4 diferido.
- **Ensamble 4 — Blending Optuna (parcela):** los mismos 3 base learners combinados por pesos simplex optimizados con Optuna sobre un holdout **espacialmente disjunto** (minimiza gap F1 train-val).
- **Reconciliación píxel↔parcela documentada:** la conversión de un modelo denso a predicción de parcela usa la moda/promedio de probabilidades dentro de la geometría de parcela (asunción pureza PASTIS ~98%; caveat de píxeles mixtos de margen).
- **Reglas anti-fuga (NON-NEGOTIABLE):** promediar probabilidades (no logits); meta-learner solo sobre OOF con CV espacial; blending sobre holdout disjunto; **reportar en fold-5 (test held-out), NO fold-4** (que fue selección).
- Tabla comparativa: mejor individual (TSViT-pheno 0.6253) vs los 4 ensambles + tiempo de inferencia + modelo elegido con justificación escrita (criterio "Selección").
- **≥4 gráficas interpretadas** (absorbe US-037): matriz de confusión normalizada+absoluta, ROC one-vs-rest con AUC por clase + macro, PR por clase, residuos espaciales sobre geometría real; cada una con ≥1 párrafo de interpretación; UMAP de embeddings opcional.
- Runs MLflow individuales por ensamble con `data_version` + `code_version` (lineage en server Docker :5010, no `./mlruns`).

**Tareas técnicas:**

- [ ] Implementar `ml/ensemble/voting.py`, `bagging.py`, `stacking.py`, `blending.py` con una clase base común `EnsembleModel` (DRY)
- [ ] Cargar softmax/OOF persistidos por US-031 desde `ml/eval/oof/*.parquet` (regenerar si faltan — directorio inexistente a 7-jun)
- [ ] Helper de reducción denso→parcela (`pixel_to_parcel_probs`) en `ml/utils/`
- [ ] Optuna study para Bagging y Blending; persistir best params
- [ ] Tabla comparativa + figuras vía `ml/eval/plots.py` / `avance4_figures.py`
- [ ] Tests pytest en `tests/ml/ensemble/` (anti-fuga: verificar que el meta-learner solo ve OOF; CV espacial; report fold-5)
- [ ] Runs MLflow por ensamble + tag de modelo elegido

**Licencia / legal:** AlphaEarth `SATELLITE_EMBEDDING/V1/ANNUAL` v1.1 CC-BY-4.0 (no "v2.1"); PASTIS-R licencia original.

**Estimación:** 8 puntos (~4 días).

---

### US-041 — Ensamble E-a: TSViT-pheno + FarSLIP (fusión dual-head)

**Como** equipo de ML,
- **quiero** un ensamble incremental que fusione el mejor segmentador denso (TSViT-pheno) con la rama contrastiva-fenológica FarSLIP corregida (directiva del sponsor),
- **para que** el Avance 5 demuestre el camino principal ordenado por el sponsor (FarSLIP-fenológico → ensamble) con FarSLIP como vía positiva, no como ablación negativa.

**Criterios de Aceptación:**

- Depende de **US-037 (EPIC 5)** (FarSLIP-pheno evaluado, supera el 0.163 previo) y **US-039 (EPIC 5)** (TSViT-pheno full-config retrain en H100). El gap que hacía perder a FarSLIP era el `torch.randn` de `ml/farslip/train.py:~184` (prototipos de texto aleatorios); la directiva del sponsor (descripciones fenológicas Gemini Flash → contrastivo) lo llena exactamente.
- FarSLIP emite **1 embedding 512-dim por parcela** → se construyen **18 prototipos de clase** (media de embeddings FarSLIP por clase) → cosine per-pixel por **broadcast espacial** del embedding de parcela → mapa `(1,18,H,W)`.
- Fusión con el softmax denso de TSViT-pheno vía `DualHeadFusionHead` con coeficiente **α aprendible**, entrenado con OOF spatial-CV.
- Asunción de pureza de parcela PASTIS ~98% documentada como caveat (se rompe en píxeles mixtos de margen); la reproyección frozen 384→512 es aproximación cruda (solución limpia = CLIP text encoder nativo, post-entrega).
- Métrica reportada en **fold-5 held-out** con el harness único (US-030); comparar honesto E-a vs TSViT-pheno solo (la rama fenológica aporta ~0.3%, no 5% — no sobre-afirmar).
- Confianza de ingeniería: fix + ablación de bandas ~95% (~10 h H100); E-a depende de que la convergencia FarSLIP-pheno se materialice.

**Tareas técnicas:**

- [ ] `ml/ensemble/dual_head_fusion.py` con `DualHeadFusionHead` (α aprendible)
- [ ] `build_class_prototypes()` (media de embeddings FarSLIP-pheno por clase) + broadcast cosine per-pixel
- [ ] Reproyección frozen ortogonal 384→512 reutilizada de `ml/farslip/distill.py` (`_proto_to_clip_proj`)
- [ ] Loop OOF spatial-CV para entrenar α; persistir en `ml/eval/oof/`
- [ ] Unit test de indexado de bandas vs parcela referencia (mitiga mismatch 4 vs 10 bandas)
- [ ] Run MLflow `ensemble-Ea-tsvit-pheno-farslip` con métrica fold-5

**Licencia / legal:** FarSLIP checkpoints (Li et al. 2025, arXiv:2511.14901); descripciones fenológicas vía Gemini Flash.

**Estimación:** 5 puntos (~2.5 días).

---

### US-042 — Ensamble E-b: E-a + AlphaEarth (XGBoost)

**Como** equipo de ML,
- **quiero** extender el ensamble E-a incorporando el espacio de embeddings AlphaEarth (XGBoost a nivel parcela) como tercer miembro del stacking,
- **para que** el Avance 6 demuestre si el embedding FM tabular aporta señal complementaria sobre el camino denso + contrastivo, con reconciliación honesta píxel↔parcela.

**Criterios de Aceptación:**

- Depende de **US-041** (E-a operativo).
- XGBoost-AlphaEarth (parcela, AlphaEarth 64-dim `SATELLITE_EMBEDDING/V1/ANNUAL` v1.1) entra al stacking junto a E-a.
- Reconciliación parcela↔píxel explícita: el resultado denso de E-a se reduce a parcela (moda/promedio de probabilidades) para alinearse con el espacio tabular de XGBoost; documentar la asunción y su error.
- Reporte **honesto**: ¿E-b supera a E-a? El baseline XGB cruza 0.60 solo al colapsar a 6 familias HCAT (0.6535) — en 18 clases es 0.4365; explicitar a qué label space se reporta cada cifra.
- Métrica en **fold-5 held-out** con harness único (US-030); MLflow con `data_version` + `code_version`.

**Tareas técnicas:**

- [ ] Integrar XGBoost-AlphaEarth como base learner adicional en `ml/ensemble/stacking.py`
- [ ] Helper de reconciliación denso→parcela compartido (DRY con US-040)
- [ ] OOF spatial-CV con los 3 miembros; meta-learner solo sobre OOF
- [ ] Tabla comparativa E-a vs E-b + interpretación
- [ ] Run MLflow `ensemble-Eb-plus-alphaearth`

**Licencia / legal:** AlphaEarth `SATELLITE_EMBEDDING/V1/ANNUAL` v1.1 CC-BY-4.0, global incl. México.

**Estimación:** 5 puntos (~2.5 días).

---

### US-043 — Ensamble E-c: geo-context + tools (DISEÑO ONLY, ADR-010)

**Como** equipo de ML,
- **quiero** un sketch arquitectónico del ensamble geo-contextual que añada clima (ERA5), elevación (SRTM), vecindad espacial y un refinamiento estructurado (CRF/GNN),
- **para que** quede documentado como trabajo futuro creíble (Paper Track / post-presentación) sin consumir cómputo ni schedule en el horizonte de 3 semanas.

**Criterios de Aceptación:**

- Depende de **US-042** (E-b como punto de partida conceptual).
- **DISEÑO ONLY:** sin entrenamiento, sin runs H100, sin código de producción. Entregable = documento de arquitectura.
- Sketch describe: features ERA5 (clima) + SRTM (elevación) + vecindad de parcela + capa de refinamiento CRF o GNN sobre la salida de E-b.
- Estimación de esfuerzo realista para el futuro (4-6 semanas) y dependencias (jobs GEE zonal).
- Documentado en **ADR-010** como FUTURE; enmarcado honesto (no se promete mejora cuantitativa). (Nota: el texto original decía "ADR-009", pero ADR-009 ya estaba ocupado por la reactivación H100 / alcance v8; el sketch E-c vive en [ADR-010](../docs/decisions/ADR-010-ensamble-ec-geocontext-future.md).)

**Tareas técnicas:**

- [x] Documento `docs/decisions/ADR-010-ensamble-ec-geocontext-future.md` con el sketch E-c (diagrama + features + flujo CRF/GNN)
- [x] Enlazar desde la sección "FUTURE" del plan v8 (este bloque). Nota: el enlace pedido "desde la skill `agrosat-ml-ensemble`" se descarta deliberadamente — las skills son instrucciones operativas reutilizables, no registro de decisiones de una US; el sketch E-c vive en ADR-010 y en `docs/us-planning/us-043.md`.

**Licencia / legal:** ERA5 (Copernicus CDS), SRTM (USGS/NASA) — a documentar cuando se ingiera (futuro).

**Estimación:** 2 puntos (~1 día, documental).

---

**Subtotal EPIC 6: 20 story points** (US-040 8 + US-041 5 + US-042 5 + US-043 2).

---

## EPIC 7: Agente Conversacional con Google ADK {#epic-7}

**Objetivo.** Construir el agente conversacional que es la esencia del producto: combina los modelos perceiver del equipo (TSViT-pheno, FarSLIP-pheno, AlphaEarth+XGBoost de los EPICs 5 y 6) con un orquestador-reasoner LLM en dos variantes (Gemini 2.5-pro GA en la nube y Qwen3.5-35B-A3B self-hosted en la H100 NVL 96GB), nueve FunctionTools geoespaciales y un loop de function calling trazable. El patrón es **Be My Eyes** (Huang et al. 2025): el perceiver son nuestros modelos de segmentacion/clasificacion, el reasoner es el LLM **frozen** que razona sobre el *texto* que el perceiver emite — el LLM **no** es clasificador de pixeles, el clasificador final es el ensamble del EPIC 6.

**Alineado con.** Avances 6 (app MVP) y 7 (evaluacion LLM + observabilidad). Evaluacion formal con benchmarks AgroMind y GeoAnalystBench.

**Cambio de realidad v8 (verificado 7-jun-2026).** Tres correcciones de raiz reescriben este EPIC respecto al v6:

1. **`ml/agent/` esta vacio** (0 de 9 tools; solo `__init__.py` en `ml/agent/` y `ml/agent/tools/`). Todas las US de este EPIC son **construccion desde cero**, ninguna esta resuelta.
2. **`google-adk` salio del lock** — no es dependencia directa en `poetry.lock` (solo aparece como extra opcional de otro paquete) porque colisiona con `google-genai 2.x` ya instalado (`google-genai >=1.66,<3.0`). Se difiere Vertex AI Agent Engine; el agente se construye con el **SDK local `google-genai`** y su capa nativa de function calling, manteniendo la abstraccion ADK-like (factory + tool registry + stream) en codigo propio para no perder la opcion de portar a ADK mas adelante.
3. **Reasoner = Gemini 2.5-pro GA** (no "Gemini 3.1 Pro"; contexto **1M**, no 2M). **Gemma 4 26B LoRA se difiere** (experts MoE 3D fused bloquean QLoRA; AgroMind es eval-only → fine-tune = leakage; ver ADR-009). La variante on-prem es **Qwen3.5-35B-A3B vLLM** (GPTQ-Int4, single-GPU, sin `--tensor-parallel-size`).

**Puntos totales de la epica: 31** (US-045 8 + US-046 6 + US-047 5 + US-048 5 + US-049 5 + US-050 2). El tracing se reduce a `structlog` + tags MLflow (EPIC 11) y latencia chat p95; no hay observabilidad custom del agente. Estimaciones realistas para el horizonte de ~3 semanas a la presentacion (27-jun).

---

### US-045 — Nueve FunctionTools geoespaciales con schemas Pydantic (SDK `google-genai`)

**Como** equipo,
- **quiero** nueve tools ejecutables como FunctionTools del SDK `google-genai`, cada una con `ToolInput`/`ToolOutput` Pydantic validados y logging estructurado,
- **para que** el agente tenga ejecucion real verificable (no alucinaciones), sea testeable unitariamente, y respete RLS por `session_id` en cada query a la base.

**Criterios de Aceptacion:**

- Modulo `ml/agent/tools.py` (mas helpers en `ml/agent/tools/`) con las nueve tools, cada una con schema Pydantic `ToolInput`/`ToolOutput` y `structlog.get_logger()`:
  - `list_parcels(aoi: GeoJSON | None, session_id: str) -> ParcelList` — parcelas de la sesion (session-scoped, RLS via `SET LOCAL`).
  - `get_parcel_timeseries(parcel_id: str, start: date, end: date, index: str) -> TimeSeries` — serie NDVI/NDWI/EVI por parcela desde DB.
  - `get_aoi_stats(aoi: GeoJSON, year: int) -> AoiStats` — estadisticos zonales (area, cultivo dominante via XGBoost-AlphaEarth del EPIC 6).
  - `search_stac(bbox: BBox, datetime_range: str, cloud_cover_max: float) -> SceneList` — query STAC pgstac.
  - `get_tiles(scene_id: str, index: Literal["ndvi","ndwi","evi","rgb"]) -> TileUrl` — delega a TiTiler, devuelve URL de tiles COG.
  - `classify_new_parcel(aoi: GeoJSON) -> ClassificationResult` — invoca el modelo/ensamble final del EPIC 6 (perceiver).
  - `add_aoi(aoi: GeoJSON, name: str, session_id: str) -> AoiRef` — persiste AOI dibujada (RLS por owner).
  - `compare_models(parcel_id: str, models: list[str]) -> ModelComparison` — compara predicciones de modelos del EPIC 5/6 sobre una parcela.
  - `explain_prediction(parcel_id: str) -> Explanation` — emite descripcion estructurada (perceiver → texto) que el reasoner consume estilo Be My Eyes; incluye descripcion fenologica (`phenology_descriptor` Wen et al. 2025).
- Cada tool registrable como `FunctionDeclaration` para el loop de function calling de `google-genai`.
- RLS verificado: las tools que tocan DB ejecutan `SET LOCAL` con el subject del JWT (depende del RLS multi-tenant del EPIC 11).

**Tareas tecnicas:**

- [ ] Modulo `ml/agent/tools.py` + un helper por tool en `ml/agent/tools/`
- [ ] Schemas Pydantic `ToolInput`/`ToolOutput` por tool; export a `FunctionDeclaration`
- [ ] Tests unitarios por tool con fixtures deterministicos y mocks (GEE, TiTiler, DB)
- [ ] Documentacion auto-generada desde los schemas Pydantic
- [ ] Las cinco tools de la demo (`list_parcels`, `get_parcel_timeseries`, `get_aoi_stats`, `classify_new_parcel`, `explain_prediction`) en `/chat` sincrono; las cuatro restantes background/diferidas

**Estimacion:** 8 puntos (~4 dias).

---

### US-046 — Capa perceiver-reasoner: descripcion estructurada de los modelos (patron Be My Eyes)

> **Nota de evolucion v8.** El v6 fundia esta logica en un agente ADK monolitico con Spatial-RAG pgvector. El v8 la reduce a su esencia demoable: el **perceiver** son los modelos del equipo (TSViT-pheno, FarSLIP-pheno, AlphaEarth+XGBoost) que emiten **descripciones estructuradas textuales** (no clasifican el LLM); el **reasoner** Gemini frozen razona sobre ese texto. El marco Be My Eyes (Huang et al. 2025, Tabla 4) sigue vigente: el valor del perceiver esta en *comunicar bien*, no en recuperar para clasificar — lo cual valida el resultado negativo propio del equipo (`pheno_text` aporto solo +0.0016 como miembro del ensamble, Avance5).
>
> **Revision 14-jun-2026: el Spatial-RAG vuelve en variante _lite_ (IN), reposicionado.** No sirve al perceiver-como-clasificador (ese eje quedo cerrado en negativo en el Avance5), sino al **reasoner como grounding conversacional** para reducir alucinaciones — un eje ortogonal que el delta=0.0 de clasificacion NO toca, asi que no contradice a Be My Eyes. Se construye en su version barata (corpus fenologico ya generado por `ml/features/phenology_description.py`; vector AlphaEarth 64-dim ya persistido en `features_parcels`; sin `e5-mistral-7b` ni GPU), **detras de un flag `rag_enabled`** (default off) y como **tool diferida**. El A/B (hallucination rate ±RAG) se **mide en la evaluacion del copiloto** (AgroMind/GeoAnalystBench), no aqui. El Spatial-RAG _completo_ (e5-mistral 4096-dim + HNSW + cross-encoder reranking) sigue **FUTURE**.

**Como** equipo,
- **quiero** una capa perceiver que envuelva los modelos del EPIC 5/6 y emita descripciones estructuradas (cultivo, fenologia, vigor, confianza) que el reasoner Gemini consume, mas un Spatial-RAG _lite_ opcional que aterrice el razonamiento en parcelas vecinas reales,
- **para que** el razonamiento sea trazable y auditable, se pueda intercambiar la variante de reasoner (Gemini ↔ Qwen) sin tocar el perceiver, la arquitectura tenga respaldo academico (Be My Eyes), y el grounding local refuerce la historia de soberania de datos (Qwen on-prem).

**Criterios de Aceptacion:**

- `ml/agent/perceiver.py` que envuelve los modelos perceiver (TSViT-pheno / FarSLIP-pheno / AlphaEarth+XGBoost) y emite descripciones estructuradas textuales por parcela/AOI (incluye descripcion fenologica Wen et al. 2025).
- El reasoner consume **el texto** del perceiver, no la imagen — el LLM no es clasificador de pixeles.
- Evento SSE `perceiver_observation` adicional en `/chat` para mostrar la observacion del perceiver antes de la respuesta final.
- La tool `explain_prediction` es el punto de entrada del perceiver al loop del agente.
- **Spatial-RAG _lite_ detras de flag `rag_enabled` (default off)** — sirve al reasoner (grounding), no al clasificador:
  - Corpus = descripciones fenologicas a escala (reutiliza `ml/features/phenology_description.py`) + metadatos de escena, ingeridos en tabla `rag_documents` (geom + embedding).
  - Vector = embedding AlphaEarth 64-dim ya persistido en `features_parcels` (sin `e5-mistral-7b`, sin GPU); `e5-small` en CPU como fallback texto-nativo.
  - Pipeline hibrido en serie: `ST_DWithin` (PostGIS) -> pgvector cosine -> fusion ponderada (codigo de referencia en skill `agrosat-spatial-rag`).
  - Expuesto como **tool diferida** `retrieve_context` (NO entre las 5 tools sincronas de la demo del agente).
  - Con `rag_enabled=false` el reasoner opera sin grounding (degradacion elegante); el flag habilita el A/B medido en la evaluacion del copiloto.
- **Spatial-RAG _completo_ (e5-mistral 4096-dim + HNSW + cross-encoder reranking): sigue OUT (FUTURE)** — el _lite_ no lo sustituye ni lo bloquea.

**Tareas tecnicas:**

- [ ] `ml/agent/perceiver.py` que envuelve los modelos y emite descripciones estructuradas
- [ ] Evento SSE `perceiver_observation` en `/chat`
- [ ] Conectar `explain_prediction` como entrada del perceiver
- [ ] Test que verifica que el reasoner razona sobre el texto del perceiver (no sobre logits)
- [ ] (RAG lite) Generar el corpus fenologico a escala de las parcelas del fold-5 con `ml/features/phenology_description.py`
- [ ] (RAG lite) Migracion `dbmate` de la tabla `rag_documents` (indice GIST sobre geom + columna embedding pgvector; vector = AlphaEarth 64-dim)
- [ ] (RAG lite) `ml/agent/rag.py`: pipeline hibrido `ST_DWithin` + pgvector cosine + fusion ponderada (adaptar skill `agrosat-spatial-rag` a 64-dim sin GPU)
- [ ] (RAG lite) Tool diferida `retrieve_context` detras de flag `rag_enabled`; hook opcional en el reasoner
- [ ] (RAG lite) Exponer la variante ±RAG al harness de evaluacion (`agent_bench.py`) para el A/B de hallucination
- [ ] (RAG lite) Test de aislamiento: `rag_enabled=false` no toca el loop; `=true` inyecta el contexto recuperado

**Estimacion:** 6 puntos (~3 dias): 3 SP del perceiver base + 3 SP del Spatial-RAG _lite_ (corpus a escala + migracion `rag_documents` + pipeline hibrido + tool diferida con flag). El fine-tune del perceiver y el Spatial-RAG _completo_ (e5-mistral + HNSW + reranking) siguen FUTURE.

---

### US-047 — Agente `ml/agent/agent.py` (factory + stream_response, SDK local)

**Como** equipo,
- **quiero** un factory `create_agent()` que arme el agente reasoner Gemini 2.5-pro con system prompt de analista agronomico y el registro de las nueve tools, y exponga `stream_response()`,
- **para que** el `ChatService` consuma un agente desacoplado, intercambiable de variante (Gemini ↔ Qwen) sin reescribir el loop, y portable a ADK/Agent Engine en el futuro.

**Criterios de Aceptacion:**

- `ml/agent/agent.py` con `create_agent(model: str, tools: list, instruction: str)` que devuelve un agente con las nueve FunctionTools geoespaciales registradas.
- `instruction` = system prompt de analista (rol: interpreta firmas fenologicas, explica predicciones del ensamble, nunca clasifica pixeles el mismo — Be My Eyes).
- `stream_response(messages, session_id)` emite los eventos del loop (`tool_call`, `tool_result`, `text_delta`, `done`) que `ChatService` reenvia por SSE.
- SDK local `google-genai` (sin Agent Engine, sin `google-adk` por conflicto de version); diseno modular para portar a ADK post-presentacion.
- La variante de modelo se inyecta (Gemini 2.5-pro por defecto; `gemini-2.5-flash` o Qwen via `/llm/switch`) sin tocar el factory.

**Tareas tecnicas:**

- [ ] `ml/agent/agent.py` con `create_agent()` + `stream_response()`
- [ ] System prompt de analista agronomico (estilo Be My Eyes: reasoner razona sobre texto del perceiver)
- [ ] Registro de las nueve tools geoespaciales
- [ ] Tests de integracion con queries canonicas del guion de demo

**Estimacion:** 5 puntos (~2.5 dias).

---

### US-048 — Variante B: Qwen3.5-35B-A3B self-hosted en H100 NVL 96GB con vLLM (GPTQ-Int4)

> **Nota de evolucion v8.** La H100 NVL 96GB **ya esta disponible** (VM `gjcamacho-gpuh1`, micromamba `agrosat`, `F:\projects\agrosat-copilot`), por lo que esta variante vuelve a ser viable. El cambio respecto a v6: serving **GPTQ-Int4 single-GPU sin `--tensor-parallel-size`** (no BF16 ~70 GB), y **sin LoRA fine-tune** (el fine-tune de trazas queda FUTURE; la H100 prioriza FarSLIP → TSViT → ensambles → serving Qwen, en ese orden estricto). El cuello no es VRAM, es **tiempo** (~3 semanas).

**Como** equipo,
- **quiero** desplegar **Qwen3.5-35B-A3B** (MoE 35B totales / 3B activos, contexto nativo 128K, Apache 2.0) en la H100 NVL 96GB con vLLM cuantizado GPTQ-Int4 como orquestador open-source on-premise,
- **para que** el copiloto sea 100% desplegable en infraestructura propia del usuario (cooperativas agricolas italianas que no pueden exportar datos a Google Cloud) y demos la historia de **soberania de datos** frente a Google Earth AI.

**Criterios de Aceptacion:**

- Modelo Qwen3.5-35B-A3B en variante **GPTQ-Int4** servido con `vllm serve` en single-GPU (sin `--tensor-parallel-size`), `--enable-prefix-caching` para tool calls repetidos, continuous batching.
- Endpoint `/v1/chat/completions` compatible con la API OpenAI, intercambiable con Gemini desde el mismo cliente (el factory del agente acepta backend OpenAI-compatible).
- Tras `POST /llm/switch` a Qwen, las queries subsecuentes de `/chat` responden por Qwen.
- Latencia objetivo: p50 < 2 s / p95 < 5 s en query simple de un turno; p95 < 15 s en multi-turno con 3-5 tool calls.
- Script `scripts/serve_qwen35.sh` que inicia vLLM, verifica health y publica el endpoint.
- Despliegue durante ventana H100 con orden estricto de prioridad (FarSLIP → TSViT → ensambles → este serving); **sin** LoRA fine-tune (diferido).

**Tareas tecnicas:**

- [ ] Script de descarga via `huggingface_hub.snapshot_download` (variante GPTQ-Int4) con cache en disco F: / Azure Blob
- [ ] `scripts/serve_qwen35.sh` con parametros calibrados (single-GPU, prefix-caching)
- [ ] Smoke test post-launch contra query canonica
- [ ] Benchmark de latencia vs batch size y context length reportado en MLflow
- [ ] Documentacion de arranque/apagado en `docs/serving/qwen35.md`
- [ ] (FUTURE) LoRA fine-tune de trazas de tool calls — diferido a ADR-009

**Licencia / legal:** Apache 2.0 via HuggingFace. Uso academico y comercial permitido. Atribucion recomendada a Alibaba Qwen Team.

**Estimacion:** 5 puntos (~1 dia de serving + integracion `/llm/switch` y benchmark; sin fine-tune).

---

### US-049 — Evaluacion del copiloto en AgroMind y GeoAnalystBench (Gemini, Qwen, Gemma-base)

**Como** equipo,
- **quiero** evaluar las variantes del reasoner (Gemini 2.5-pro cloud, Qwen3.5-35B on-prem, y Gemma 4 base como referencia) en benchmarks estandar **como evaluacion, NO como fine-tune**,
- **para que** los Avances 6 y 7 (y eventualmente el Paper Track) tengan metricas cuantitativas comparables, sin incurrir en leakage.

**Criterios de Aceptacion:**

- AgroMind (subset de 500 pares) evaluado con cada variante; metricas: exact match, F1-SQuAD, BERTScore, tool-call accuracy, hallucination rate (LLM-as-judge con DeepEval / Gemini como juez).
- GeoAnalystBench evaluado en modo plan-and-react.
- **Tres modelos como benchmark, NUNCA fine-tune** (AgroMind es eval-only → fine-tune = leakage; ver ADR-009).
- Tabla comparativa con error bars sobre 3 corridas; analisis de latencia y costo por query.
- Targets de rubrica: AgroMind >= 0.75 (Gemini 2.5-pro), >= 0.70 (Qwen3.5-35B); GeoAnalystBench pass rate >= 0.65.
- MLflow con tags `data_version` + `code_version`; lineage en el server Docker :5010.

**Tareas tecnicas:**

- [ ] Harness `ml/eval/agent_bench.py` (AgroMind + GeoAnalystBench, LLM-as-judge DeepEval)
- [ ] Ejecucion en ventana H100 compartida con el serving de Qwen3.5
- [ ] Reporte HTML con comparativa A/B/base + error bars
- [ ] Registro MLflow con `data_version` + `code_version`

**Estimacion:** 5 puntos (~2.5 dias).

---

### US-050 — Gemma 4 26B LoRA — SOLO documentar como ADR-009 (FUTURE)

> **Nota de evolucion v8.** El v6 contemplaba a Gemma 4 26B-MoE LoRA como perceiver fine-tuned (decision irrevocable). El v8 lo **difiere** tras verificacion en HF (jun-2026): no se entrena antes del 27-jun. La H100 va a FarSLIP / TSViT / ensambles / serving Qwen, que la rubrica **si** califica.

**Como** equipo,
- **quiero** documentar formalmente en ADR-009 por que Gemma 4 26B LoRA queda fuera del alcance de la presentacion, con la correccion tecnica del skill,
- **para que** la decision sea trazable y reactivable post-presentacion sin re-investigar.

**Criterios de Aceptacion:**

- ADR-009 corrige el skill `agrosat-llm-finetuning`: id real `google/gemma-4-26B-A4B-it` (el id `gemma-4-26b-it` NO existe); experts MoE como tensores 3D fused `nn.Parameter` (`Gemma4TextExperts`) → bitsandbytes no cuantiza el layout 3D → **QLoRA bloqueado**; `target_modules=[gate/up/down_proj]` no matchea nada (~0.91% params entrenables); la via real `target_parameters=["experts.gate_up_proj","experts.down_proj"]` (PEFT >=0.17) es bleeding-edge.
- ADR-009 documenta que **AgroMind es eval-only** (~28,482 QA, sin train split) → "fine-tune sobre AgroMind" = leakage; AgroMind-IT/ES propio (500 pares) tambien es eval.
- ADR-009 ratifica: reasoner = Gemini 2.5-pro; variante on-prem = Qwen3.5-35B-A3B vLLM; Gemma LoRA = FUTURE (expert-LoRA con SFT sintetico propio).
- **No se entrena Gemma 4** en este horizonte.

**Tareas tecnicas:**

- [ ] Redactar `docs/decisions/ADR-009-*.md` con las correcciones tecnicas verificadas
- [ ] Actualizar el skill `agrosat-llm-finetuning` con id real, `target_parameters` y AgroMind=eval-only
- [ ] Marcar Gemma 4 LoRA como FUTURE en CLAUDE.md / AGENTS.md (Decisiones Irrevocables: viable con H100 pero diferido)

**Estimacion:** 2 puntos (~1 dia; solo documentacion, sin entrenamiento).

---

**Subtotal EPIC 7: 31 story points** (US-045 8 + US-046 6 + US-047 5 + US-048 5 + US-049 5 + US-050 2). Ninguna US del EPIC esta RESUELTA: `ml/agent/` esta vacio (0 de 9 tools), backend solo expone `/healthz`+`/readyz`. Spatial-RAG _completo_ (e5-mistral + HNSW + reranking), Gemma 4 LoRA, Agent Engine y el fine-tune del perceiver/Qwen quedan FUTURE; el Spatial-RAG _lite_ entra en el alcance de la capa perceiver-reasoner.

---

## EPIC 8: Backend API + Worker Pub/Sub + Tiling {#epic-8}

**Objetivo.** Exponer la plataforma como API REST FastAPI documentada (OpenAPI 3.1), servir tiles dinámicos COG para el frontend, aplicar aislamiento multi-tenant con RLS PostgreSQL y procesar inferencias pesadas de forma asíncrona. En el v8 el backend es el **camino crítico del producto MVP** (track paralelo de Aaron): el estado real al 7-jun es esqueleto con solo `/healthz`+`/readyz` (0 de 7 endpoints de negocio) y RLS sin aplicar. El alcance comprometido para la demo (27-jun) son los endpoints síncronos session-scoped (`/chat` SSE, `/aois`, `/timeseries`, `/tiles`, `/stac/search`, `/llm/switch`) sobre RLS aplicado; la cola async Pub/Sub `/jobs` real y Clerk OAuth2 productivo quedan diferidos (OUT — el MVP usa `user_id` demo).

**Alineado con.** Avance 6 — despliegue + app MVP. Camino crítico §2 (C9: backend US-051..056 + agente US-045..050). Realidad del producto §9.1.

**Puntos totales de la épica: 24.**

---

### US-051 — RLS multi-tenant aplicado (migración dbmate) — bloqueante

**Como** equipo de plataforma,
- **quiero** políticas Row-Level Security aplicadas por `session_id`/subject JWT sobre las tablas multi-tenant,
- **para que** ningún usuario lea ni escriba datos de otra sesión y se cumpla la regla global de aislamiento (NON-NEGOTIABLE #3) antes de exponer cualquier endpoint de negocio.

**Criterios de Aceptación:**

- Migración `db/migrations/00X_rls_multi_tenant.sql` (vía `dbmate new`/`dbmate up`, jamás `metadata.create_all`) que habilita `ROW LEVEL SECURITY` y crea `CREATE POLICY` en `chat_sessions`, `aois`/`parcels` y tablas derivadas con columna de propietario.
- Las policies filtran por `current_setting('app.current_session', true)` (o subject JWT), seteado con `SET LOCAL` por request en el pool de conexión.
- El rol de aplicación NO tiene `BYPASSRLS`; el rol de migración sí (separación).
- Test de aislamiento cross-session en verde: la sesión A no ve filas de la sesión B (lectura, update y delete).
- Las 3 migraciones existentes (`initial_schema`, `create_parcels`, `create_features_parcels`) no se modifican — la RLS entra como migración rollforward.

**Tareas técnicas:**

- [ ] `dbmate new rls_multi_tenant`; `ALTER TABLE ... ENABLE ROW LEVEL SECURITY` + `FORCE`
- [ ] `CREATE POLICY tenant_isolation ON <tabla> USING (session_id = current_setting('app.current_session')::uuid)` (SELECT/INSERT/UPDATE/DELETE)
- [ ] Dependencia FastAPI / event listener SQLAlchemy que emite `SET LOCAL app.current_session = :sid` por request
- [ ] Test `pytest` de aislamiento cross-session (docker-compose local) antes de cualquier deploy; rollback (`down`) probado
- [ ] Verificar en docker-compose local que la migración aplica limpia y revierte

**Estimación:** 3 puntos (~1.5 días). Bloqueante de todos los endpoints de negocio session-scoped (US-052/US-053/US-054).

---

### US-052 — `POST /chat` SSE con function calling (ChatService + ToolExecutor)

**Como** usuario de la plataforma,
- **quiero** un endpoint `/chat` que transmita la respuesta del agente por SSE invocando las tools geoespaciales,
- **para que** la conversación sea fluida (texto incremental) y el LLM razone sobre los resultados de mis modelos (patrón "Be My Eyes": perceiver = nuestros modelos, reasoner = Gemini 2.5-pro frozen).

**Criterios de Aceptación:**

- `POST /chat` emite eventos SSE tipados: `text_delta`, `tool_result`, `done` (y `error`).
- Implementado con el SDK `google-genai` (instalado `^2.6.0`); `google-adk` NO está en `poetry.lock` → se usa solo `google-genai`, difiriendo Vertex Agent Engine.
- Loop tool→reasoner: el ChatService ejecuta el `ToolExecutor` (9 FunctionTools de US-045) y reinyecta `tool_result` al razonador; el LLM no clasifica píxeles (el clasificador es el ensamble E6).
- Razonador por defecto `gemini-2.5-pro` (GA, 1M ctx — NO 2M, NO el hardcode previo `gemini-3.1-pro`).
- Rate limit 10 req/min por sesión; guard de autorización por sesión (regla anti-patrón: nunca `/chat` sin auth ni rate limit).
- La lógica vive en `services` (`ChatService`), no en el router (Separation of Concerns); el router solo recibe y stremea.
- Latencia p95 < 3 s en query simple, < 15 s en multi-step (verificado en US-059).

**Tareas técnicas:**

- [ ] Router `backend/app/api/chat.py` → delega a `ChatService.stream_response()`
- [ ] `ChatService` + `ToolExecutor` en `backend/app/services/`; cliente `google-genai`
- [ ] Generador async SSE (`text/event-stream`) con eventos `text_delta`/`tool_result`/`done`/`error`
- [ ] Rate-limit 10/min (Redis) + dependencia de sesión + `SET LOCAL` RLS
- [ ] Tests `httpx.AsyncClient` con Gemini mockeado + tools stub; e2e de un round-trip

**Estimación:** 8 puntos (~4 días).

---

### US-053 — `/aois` CRUD (session-scoped, RLS) + `/timeseries` + `/tiles` proxy + `/stac/search`

**Como** frontend / cliente de la API,
- **quiero** los endpoints de datos geoespaciales scoped por sesión (CRUD de AOIs, serie temporal de índices, proxy de tiles y búsqueda STAC),
- **para que** el mapa y los paneles consuman datos del usuario sin fugas entre tenants.

**Criterios de Aceptación:**

- `POST/GET/DELETE /aois` — CRUD de AOIs desde GeoJSON, **session-scoped** verificado contra RLS (US-051); cada query filtra por owner.
- `GET /aois/{id}/timeseries?index={NDVI|NDWI|NDMI}` — serie temporal desde la DB (parcelas/features), validada por propiedad.
- `GET /tiles/{z}/{x}/{y}.png` — delega a TiTiler montado en la imagen backend (US-055); sin servicio separado.
- `GET /stac/search` — consulta el catálogo STAC (pgstac, EPIC 1) con filtros bbox/datetime/collection.
- Los 4 endpoints validan autorización por sesión; RLS por owner verificado con test cross-session.
- Validación Pydantic de entrada/salida; tipos de respuesta GeoJSON-compatibles.

**Tareas técnicas:**

- [ ] Routers `aois.py`, `timeseries.py`, `tiles.py`, `stac.py` → delegan a services (`AoiService`, `TimeseriesService`, `StacService`)
- [ ] Modelos SQLModel + GeoAlchemy2 para AOIs (geometry GIST); query `/timeseries` sobre features
- [ ] `/tiles` reusa el `TilerFactory` montado (US-055); `/stac/search` sobre pgstac
- [ ] `SET LOCAL app.current_session` por request en cada endpoint; tests de aislamiento por owner
- [ ] Tests de integración `httpx.AsyncClient` (CRUD + 403 cross-session)

**Estimación:** 5 puntos (~2.5 días).

---

### US-054 — `POST /llm/switch` A/B (Gemini Pro ↔ Flash ↔ Qwen)

**Como** presentador de la demo,
- **quiero** un endpoint que cambie en caliente el modelo del orquestador entre Gemini 2.5-pro, Gemini Flash y Qwen3.5-35B-A3B (on-prem vLLM),
- **para que** se demuestre el trade-off costo/latencia (Pro vs Flash) y la soberanía de datos (cloud vs on-prem) sin reiniciar la sesión.

**Criterios de Aceptación:**

- `POST /llm/switch` setea `chat_sessions.llm_model` (persistido) y los `/chat` subsecuentes de esa sesión usan el nuevo modelo.
- Valores válidos: `gemini-2.5-pro`, `gemini-flash`, `qwen3.5-35b-a3b` (variante-B on-prem servida por vLLM, US-048).
- Latencia y tokens loggeados por switch (insumo FinOps / observabilidad US-059).
- Guard de autorización por sesión + rate limit (regla anti-patrón: `/llm/switch` nunca sin auth).
- Demo verificable: misma query Pro→Flash responde ~3.5 s vs ~1.2 s (trade-off visible, §9.3).

**Tareas técnicas:**

- [ ] Router `backend/app/api/llm.py` → `LLMSwitchService` que actualiza `chat_sessions.llm_model`
- [ ] El `ChatService` lee `llm_model` de la sesión y enruta a Gemini (google-genai) o al endpoint OpenAI-compatible de vLLM (Qwen)
- [ ] Logging estructurado de modelo/latencia/tokens por request
- [ ] Tests: switch persiste y `/chat` posterior usa el nuevo backend (mock por modelo)

**Estimación:** 3 puntos (~1.5 días).

---

### US-055 — TiTiler para tiling COG dinámico

> **Estado: RESUELTA parcialmente (montada en la imagen backend, sin Dockerfile dedicado).** En el v8 NO se crea `titiler.Dockerfile` ni servicio Cloud Run separado: TiTiler se monta como sub-app/router dentro de la **misma imagen backend** (FastAPI), reduciendo superficie de deploy y costo (un solo Cloud Run). El endpoint público al frontend es `/tiles/...` (US-053) que delega internamente a los factories de `titiler.core`.

**Como** frontend,
- **quiero** tiles PNG/WebP generados on-the-fly desde COGs en GCS,
- **para que** el mapa MapLibre muestre overlays NDVI/NDWI sin pre-renderizar todas las combinaciones.

**Criterios de Aceptación:**

- TiTiler montado como router dentro de la imagen backend (no servicio independiente, no `titiler.Dockerfile`), con GDAL configurado para leer COGs de GCS (`vsigs`/`CPL_VSIL` + service account).
- Endpoint `/cog/tiles/{z}/{x}/{y}.png?url={cog_url}&expression=(B8-B4)/(B8+B4)&rescale=-1,1&colormap=RdYlGn` funcional para NDVI/NDWI/NDMI.
- Cache Redis 15 min por tile (clave hash del endpoint completo).
- CORS configurado para dominio frontend; headers de cache HTTP.

**Tareas técnicas:**

- [ ] Añadir `titiler-core` + `rio-tiler` al backend vía `poetry add` (no Dockerfile separado)
- [ ] Montar `TilerFactory` como sub-router en `backend/app/api/tiles.py`; configurar GDAL/GCS env
- [ ] Deploy en la imagen backend Cloud Run con min=0 (scale-to-zero)
- [ ] Configurar CORS, cache headers y cache Redis 15 min

**Estimación:** 3 puntos (~1.5 días).

---

### US-056 — Worker de inferencia con cola Pub/Sub

> **Estado: OUT (diferida post-presentación, ADR-009).** En el v8 el MVP corre la inferencia **síncrona** (parcelas PASTIS pre-cargadas + XGBoost-AlphaEarth tabular rápido); la cola async Pub/Sub + Cloud Run GPU L4 + DLQ no se construye en las 3 semanas al deadline (§1 OUT, §9.2). Se conserva el diseño completo como trabajo Full/futuro. La regla NON-NEGOTIABLE de "inferencia >2 s vía Pub/Sub" se respeta en producción Full; el MVP demo solo invoca modelos ligeros síncronos y NO demuestra cola real (§9.3 "NO demostrar cola Pub/Sub real").

**Como** equipo,
- **quiero** un worker Cloud Run GPU L4 que consume mensajes Pub/Sub para inferencias pesadas,
- **para que** el API FastAPI no bloquee al usuario y la escalabilidad sea horizontal.

**Criterios de Aceptación (diferidos a Full):**

- Worker escucha topic `inference-jobs` con schema `{aoi_geojson, model_id, params}`.
- Resultados persistidos en GCS y notificación publicada en topic `inference-results`.
- Reintentos automáticos con DLQ (dead letter queue) tras 3 fallos.
- Logging estructurado con `job_id` trazable.
- El frontend recibe notificación vía SSE cuando job completa.

**Tareas técnicas (diferidas a Full / ADR-009):**

- [ ] Worker `ml/workers/inference_worker.py` con subscripción Pub/Sub
- [ ] Dockerfile con GPU L4 runtime
- [ ] DLQ topic + alerta Cloud Monitoring
- [ ] (MVP) Documentar la decisión "síncrono en MVP" y la interfaz `JobsService` stub que en Full publicará a Pub/Sub

**Estimación:** 2 puntos (~1 día) — DIFERIDOS (no cuentan en el comprometido de A6).

---

**Subtotal EPIC 8: 24 story points** (US-051 RLS 3 + US-052 /chat SSE 8 + US-053 /aois+/timeseries+/tiles+/stac 5 + US-054 /llm/switch 3 + US-055 TiTiler 3 + US-056 worker Pub/Sub 2 difer.; 22 comprometidos + 2 SP de US-056 diferidos del comprometido de A6).

---

## EPIC 9: Frontend Web + Mapa + Chat Bilingüe {#epic-9}

**Objetivo.** Construir la interfaz web impactante para la presentación final (27-jun), con i18n italiano/español/inglés nativo, mapa MapLibre con AOIs y panel de chat conversacional sobre SSE. El v8 reduce el alcance de la épica al **MVP demoable en 3 semanas** (ChatPanel + MapView sobre `index.vue`, US-057 y US-058), y difiere overlays/timeline interactivos y el switch A/B a su forma mínima.

**Alineado con.** Avance 6 (14-jun) + Avance 7 (21-jun) + Presentación Final (27-jun).

**Estado de partida real (verificado 7-jun-2026).** El frontend solo tiene el esqueleto: `frontend/app.vue`, `frontend/pages/index.vue` (portada con `useI18n()` + `t('app.name')`/`t('app.tagline')`), `nuxt.config.ts` (Nuxt 4 SSR, `@nuxt/ui-pro` + `@nuxtjs/i18n` + `@pinia/nuxt`, locales it/es/en con `strategy: prefix_except_default`, `defaultLocale: it`), `tailwind.config.ts` (Tailwind v4) y los tres `i18n/locales/{it,es,en}.json`. **No existen aún** `components/`, `composables/`, `stores/`, `layouts/` ni `middleware/`. La validación de paridad i18n se ejecuta vía `scripts/i18n_check.mjs` (`make i18n-check`). El backend solo expone `/healthz`+`/readyz`, por lo que el MVP de chat depende del track de Aaron (US-052 `/chat` SSE) y del agente ADK (US-047).

**Puntos totales de la épica: 10.**

---

### US-057 — `ChatPanel.vue` + `useChat.ts` + Pinia `stores/chat.ts` (MVP) {#us-053}

**Como** usuario del copiloto,
- **quiero** un panel de chat que renderice mensajes por rol, markdown y tarjetas de tool, consumiendo el stream SSE del backend con reconexión robusta,
- **para que** pueda conversar con el agente geoespacial y ver de forma transparente qué herramientas ejecuta.

**Criterios de Aceptación:**

- `ChatPanel.vue` renderiza mensajes role-based (user/assistant/tool) con markdown (tablas, bloques de código) y **tool cards** colapsables que muestran input/output JSON.
- `composables/useChat.ts` consume `POST /chat` SSE: parsea eventos `text_delta` / `tool_result` / `done`; reconexión con retry + backoff exponencial; manejo de error y estado de carga (skeleton/typing indicator).
- `stores/chat.ts` (Pinia, `pinia-plugin-persistedstate`) mantiene historial de mensajes de la sesión activa, el `session_id` y el `llm_model` seleccionado; estado compartido (no `reactive()` exportado).
- SSR-safe: el cliente SSE solo se inicializa con `import.meta.client`; el composable nunca toca `window` en server.
- i18n: todo texto visible (placeholders, botones, errores, etiquetas de tool) en `i18n/locales/{it,es,en}.json` en sync (`make i18n-check` verde); el `locale` activo se envía en el payload para condicionar el idioma del reasoner.
- Dark mode soportado vía `@nuxtjs/color-mode` (ya presente); foco visible y `aria-label` en input/botón (WCAG AA básico).
- Se integra en `pages/index.vue` como panel derecho del layout 2-panel (junto a `MapView.vue` de US-058).
- Tests Vitest del parser SSE (mensajes parciales, evento `done`, error de red) con cobertura frontend ≥50 %.

**Tareas técnicas:**

- [ ] Crear `frontend/composables/useChat.ts` (cliente SSE con `fetch`+`ReadableStream`, parser `text_delta/tool_result/done`, retry backoff)
- [ ] Crear `frontend/stores/chat.ts` (Pinia + persistedstate: mensajes, `session_id`, `llm_model`)
- [ ] Crear `frontend/components/chat/ChatPanel.vue` (lista de mensajes, input, send, typing indicator)
- [ ] Crear `frontend/components/chat/ChatMessage.vue` (markdown render + role styling) y `ToolCallBox.vue` (JSON colapsable)
- [ ] Poblar namespaces i18n `chat.*` y `error.*` en las tres locales (en sync)
- [ ] Tests Vitest del parser SSE + render role-based
- [ ] Integrar el panel en `pages/index.vue` (columna derecha del split 2-panel)

**Estimación:** 5 puntos (~2.5 días).

---

### US-058 — `MapView.vue` + `useMap.ts` + Pinia `stores/map.ts` (MapLibre + AOI, MVP) {#us-054}

**Como** usuario del copiloto,
- **quiero** un mapa MapLibre con un basemap satelital, AOIs GeoJSON seleccionables y selección de parcela que inyecta contexto al chat,
- **para que** pueda situar geográficamente el análisis y pedir explicaciones sobre una parcela concreta (ej. parcela pre-cargada en Francia para la demo).

**Criterios de Aceptación:**

- `MapView.vue` monta MapLibre GL JS con un basemap satelital (Esri World Imagery / fuente OSS — sin Mapbox por regla §3.5) y capa base Sentinel-2; SSR-safe (`import.meta.client`) con `cleanup` en `onBeforeUnmount` (regla del sub-agente frontend).
- AOIs GeoJSON renderizados como capa seleccionable (outline + highlight); al hacer click en una parcela se publica su `parcel_id`/contexto al `stores/chat.ts` para enlazar mapa↔chat (guion de demo §9.3).
- `composables/useMap.ts` encapsula creación/destrucción del mapa, `addSource`/`addLayer` de AOIs y los handlers de selección; `stores/map.ts` (Pinia) mantiene la AOI activa y el bbox visible.
- MVP = base layer + AOI outline seleccionable. **Draw-polygon (`maplibre-gl-draw`), parcel boundaries densos y overlay de segmentación son Full/FUTURE** (no comprometidos para el 27-jun).
- Layout 2-panel: `MapView.vue` ocupa el panel izquierdo de `pages/index.vue`, `ChatPanel.vue` (US-057) el derecho; responsive básico (apila en viewport estrecho).
- i18n: etiquetas/leyendas/tooltips visibles en `i18n/locales/{it,es,en}.json` en sync (`make i18n-check` verde).
- Tests Vitest del store de mapa y del wiring de selección (mock de MapLibre); cobertura frontend ≥50 %.

**Tareas técnicas:**

- [ ] `pnpm add maplibre-gl` (y typings) — sin npm/yarn
- [ ] Crear `frontend/composables/useMap.ts` (init/destroy MapLibre, sources/layers AOI, handlers de click)
- [ ] Crear `frontend/stores/map.ts` (Pinia: AOI activa, bbox, parcela seleccionada)
- [ ] Crear `frontend/components/map/MapView.vue` (basemap satelital + capa AOI seleccionable + cleanup)
- [ ] Wiring selección parcela → `stores/chat.ts` (contexto al chat)
- [ ] Poblar namespace i18n `map.*` en las tres locales (en sync)
- [ ] Tests Vitest del store de mapa + selección
- [ ] Integrar el mapa en `pages/index.vue` (columna izquierda del split 2-panel)

**Estimación:** 5 puntos (~2.5 días).

---

**Subtotal EPIC 9: 10 story points** (US-057 ChatPanel 5 + US-058 MapView 5). La i18n it/es/en y el switch A/B en su forma mínima se absorben en los componentes de US-057/US-058 (el endpoint `/llm/switch` es US-054, EPIC 8).

## EPIC 10: Observabilidad, Drift, FinOps, Seguridad y Documentación {#epic-10}

**Objetivo.** Cubrir los cuestionamientos de viabilidad de producción, análisis costo-beneficio y riesgos exigidos por los Avances 6 y 7 del curso, más los aspectos de seguridad, serving de LLM on-prem, evaluación de modelos de lenguaje y documentación reproducible del proyecto. En el v8 esta épica también absorbe el **serving de Qwen3.5-35B vLLM (variante-B on-prem)**, la **evaluación LLM AgroMind/GeoAnalystBench**, la **observabilidad del chat (latencia p95, lineage MLflow)** y las **atribuciones de licencia de los datasets multi-región** introducidos por la historia de transferencia.

**Alineado con.** Avance 6 (14-jun-2026) y Avance 7 (21-jun-2026); presentación final 27-jun-2026 ([ADR-008](../docs/decisions/ADR-008-rediseno-calendario-presentacion-27jun.md)).

**Puntos totales de la épica: 13.** El tracing built-in de Google ADK absorbe la observabilidad del agente, por lo que la observabilidad se concentra en métricas técnicas del sistema (latencia chat, lineage), drift de datos, FinOps y seguridad. El v8 añade el bloque de serving/evaluación LLM y la observabilidad de chat, y formaliza la observabilidad técnica, el drift de datos y las licencias multi-región.

> **Nota de realidad (corte 7-jun-2026).** La H100 NVL 96GB del sponsor (`gjcamacho-gpuh1`, env micromamba `agrosat`, `F:\projects\agrosat-copilot`) está disponible, por lo que el serving de Qwen3.5 vLLM (US-048) es viable; el cuello de botella es el tiempo (~3 semanas a presentación), no el cómputo. El fine-tune de **Gemma 4 26B LoRA queda OUT** y se documenta como ADR-009 future (US-050): los experts MoE son tensores 3D fused que bloquean QLoRA, y AgroMind es eval-only (fine-tunear sobre él = leakage). El reasoner del agente es **Gemini 2.5-pro GA (1M ctx, no 2M)**; la variante on-prem es **Qwen3.5-35B-A3B vLLM (GPTQ-Int4, single-GPU)**.

---

### US-059 — Dashboard de observabilidad con Prometheus y Grafana

**Como** operador,
- **quiero** métricas técnicas en tiempo real del sistema,
- **para que** cualquier anomalía (latencia, error rate, GPU util) sea visible para el equipo.

**Criterios de Aceptación:**

- Métricas exportadas por FastAPI con `prometheus-client`: latencia p50/p95/p99, RPS, error rate por endpoint, GPU utilization del worker (L4 spot o H100 cuando sirve Qwen), tool-call success rate del agente (integrado con **ADK / google-genai tracing**), hallucination rate estimada (LLM-as-judge muestra ~5% de queries).
- Dashboards Grafana con tres paneles: API, worker ML, data pipeline.
- Alertas configuradas (vía email o Cloud Monitoring): p99 latencia > 3 s, GPU OOM, error rate > 5%.
- **Realidad v8:** dado que en el MVP el backend solo expone `/healthz`+`/readyz` hasta que aterricen US-052/051, esta US se reduce al *scaffolding* de instrumentación (decorador de métricas + dashboards plantilla) y la versión completa (paneles poblados con tráfico real) se reporta junto con la observabilidad de chat de US-065. El subconjunto crítico para A7 (latencia chat p95) vive en US-065.

**Tareas técnicas:**

- [ ] Instrumentación FastAPI con `prometheus-client` (middleware de latencia + contadores por endpoint)
- [ ] Dashboards Grafana en `infrastructure/grafana/` (plantillas API / worker ML / pipeline)
- [ ] Alertas Cloud Monitoring (p99 > 3 s, OOM, error rate > 5%)

**Estimación:** 2 puntos (~1 día).

---

### US-060 — Drift detection con Evidently AI

**Como** ML Engineer,
- **quiero** detectar drift en bandas Sentinel-2 y en predicciones del modelo,
- **para que** el equipo sepa cuándo reentrenar en el futuro.

**Criterios de Aceptación:**

- Drift de distribución de bandas Sentinel-2 (KS test) y AlphaEarth embeddings (MMD). **Nota v8:** AlphaEarth = asset GEE `SATELLITE_EMBEDDING/V1/ANNUAL` (data v1.1, 64-dim, global incl. México, CC-BY-4.0) — corregir cualquier referencia muerta a "v2.1".
- Drift de distribución de clases predichas (Chi-cuadrado), usando el espacio de clases normalizado de US-030 (18-clase contiguo) y, cuando aplique, las macro-clases HCAT (US-074) del pipeline multi-región.
- Reporte HTML semanal automático publicado en `gs://agrosat-reports/drift/`.
- Alerta si drift score > 0.3.
- **Integrado como asset Dagster `drift_check`** que corre semanalmente con dependencia de los assets de ingesta.

**Tareas técnicas:**

- [ ] Pipeline Evidently en `ml/monitoring/drift.py` (el directorio `ml/monitoring/` no existe aún — crearlo)
- [ ] Asset Dagster `drift_check` con schedule semanal
- [ ] Notificación por email si drift score > umbral

**Estimación:** 2 puntos (~1 día).

---

### US-061 — Análisis costo-beneficio para Avances 6 y 7

**Como** equipo,
- **quiero** tablas de costos y beneficios cuantificables,
- **para que** el criterio "Costos" (20 pts), "Beneficios" (20 pts) e "Implementación" (30 pts) de las rúbricas de Avance 6 y 7 queden cubiertos.

**Criterios de Aceptación:**

- Tabla de costos por fase CRISP-ML(Q) reales del proyecto + proyección 12 meses, con cifras v8 verificadas: adquisición de datos ($0, fuentes públicas), training (**H100 prestada por el sponsor 24/7, no se cobra**; L4 spot con auto-shutdown para jobs ligeros), serving (~$115/mes con scale-to-zero), Gemini API (centavos — descripciones fenológicas FarSLIP + razonador chat, dentro de ~$115/mes), Qwen3.5 vLLM self-hosted en H100 (incremental ≈$0 sobre la VM prestada en ventanas), GCP acumulado a la fecha ~$0.30-0.49.
- Aclarar que el "Trial credit for GenAI App Builder" ($17,178) es de Vertex AI Search/Agent Builder y **NO** cubre la Gemini API de generación de texto (SKU distinta) — no se necesita.
- Tabla de beneficios cuantificables para cliente tipo 500 ha: horas ahorradas de agrónomo/mes, % ahorro de agua con detección de estrés hídrico, ahorro de insumos por fertilización focalizada, reducción de tiempo de detección de plagas.
- Beneficios intangibles: trazabilidad para cumplimiento CAP europeo, reducción de riesgo regulatorio, imagen sostenibilidad, **soberanía de datos** (variante on-prem Qwen3.5 vía US-048).
- ROI break-even estimado en mes 3 para cliente tipo.

**Tareas técnicas:**

- [ ] Documento `docs/business/costo_beneficio.md`
- [ ] Tablas en Excel + export a LaTeX para paper

**Estimación:** 1 punto (~0.5 días).

---

### US-062 — Análisis de riesgos categorizados para Avance 7

**Como** equipo,
- **quiero** análisis exhaustivo de riesgos por categoría,
- **para que** el criterio "Riesgos" (20 pts) de la rúbrica del Avance 7 quede cubierto.

**Criterios de Aceptación:**

- Cuatro categorías de riesgos según rúbrica del curso: datos (disponibilidad CDSE, calidad de labels, cobertura de nubes, **transferibilidad espacial limitada de AlphaEarth — arXiv:2601.00857**), ataques (adversarial attacks en modelos, DDoS API), confianza (hallucinations, sesgos regionales, falsas alarmas, **no sobre-afirmar F1≥0.80 en México sin ground-truth**), cumplimiento (GDPR, licencias multi-región CC-BY-SA-4.0/ODbL, políticas Copernicus).
- Cada riesgo con probabilidad (Alta/Media/Baja), impacto (Alto/Medio/Bajo) y mitigación concreta y accionable. **Realidad v8:** incorporar los riesgos de ejecución del §11 del v8 (H100 una sola GPU → cola; RLS migración falla → data leak; OOF loop 10-20h; FarSLIP band mismatch 4 vs 10; Gemini rate-limit; incremental Stage-1 no converge).

**Tareas técnicas:**

- [ ] Documento `docs/risks/riesgos.md`
- [ ] Matriz probabilidad × impacto visual

**Estimación:** 1 punto (~0.5 días).

---

### US-063 — Análisis comparativo de proveedores cloud

**Como** equipo,
- **quiero** justificar la elección multi-cloud con análisis comparativo,
- **para que** el criterio "Implementación" (30 pts) del Avance 6 quede cubierto (rúbrica exige mínimo 2 proveedores).

**Criterios de Aceptación:**

- Comparativa GCP vs Azure (mínimo rúbrica) + opcionalmente AWS e IBM Cloud con al menos cinco factores: precio GPU H100 on-demand y spot, ecosistema de Earth Observation (GCP Earth Engine vs Azure Planetary Computer vs AWS Open Data), latencia hacia Europa (target Italia), soporte de pipelines MLOps (Vertex AI Pipelines, Azure ML, SageMaker), disponibilidad de partnerships académicos.
- Decisión justificada: **GCP primario + H100 NVL 96GB del sponsor (Azure/on-prem) para training intensivo y serving Qwen vLLM**. Reflejar la realidad v8: la H100 está disponible (no parked) y la variante on-prem aporta la narrativa de soberanía de datos.

**Tareas técnicas:**

- [ ] Documento `docs/cloud/comparativa_proveedores.md`

**Estimación:** 1 punto (~0.5 días).

---

### US-064 — Seguridad y documentación final

**Como** equipo,
- **quiero** mejores prácticas de seguridad implementadas y documentación consolidada,
- **para que** el sistema sea production-ready y el tercero pueda reproducirlo.

**Criterios de Aceptación:**

- HTTPS obligatorio con Cloud Load Balancer y certificados managed.
- JWT con rotación y refresh tokens; en el MVP demo (27-jun) se usa `user_id` demo hardcoded y Clerk OAuth2 queda como Full/post-presentación.
- **RLS PostgreSQL por tenant aplicado** vía migración dbmate — ver US-051 (bloqueante; hoy `aois`/`parcels`/`features_parcels`/`chat_sessions` están **sin RLS** según `docs/STATUS.md`).
- Secretos nunca en git. **Realidad v8:** el proyecto **NO usa `.pre-commit-config.yaml`** (regla irrevocable) — el secrets-scan vive en `make secrets-scan` (gitleaks) y en CI, **no** en un hook `detect-secrets`. Corregir esta referencia muerta del v6.
- Revisión OWASP Top 10 documentada en `docs/security.md` (el archivo no existe aún — crearlo).
- Penetration test manual básico (nikto, nmap) antes de presentación.
- Model Cards publicadas en `docs/model_cards/` para el **modelo final ensemble (E6)**, **TSViT-pheno** y **FarSLIP-pheno**. **Realidad v8:** la Model Card de **Gemma 4 fine-tuned se elimina** (Gemma LoRA es OUT → ADR-009 future, US-050); en su lugar se documenta Qwen3.5-35B-A3B vLLM serving (variante-B).
- Data Sheets por dataset en `docs/data_sheets/` (incluye PASTIS-R, AlphaEarth V1, y los multi-región Sen4AgriNet/EuroCropsML).
- ADRs en `docs/decisions/`, incluido **ADR-009** (reactivación H100 + pivote FarSLIP del sponsor + alcance v8 + Gemma LoRA diferido).
- **Glosario técnico** en `docs/glosario.md` con estandarización de términos IT/ES/EN.
- README reproducible con instrucciones de setup, running y testing end-to-end.

**Tareas técnicas:**

- [ ] Configurar Cloud Load Balancer + cert managed
- [ ] Implementar JWT refresh en FastAPI
- [ ] Escribir Model Card del ensemble final + TSViT-pheno + FarSLIP-pheno (template HuggingFace)
- [ ] Documentar ADRs (Google ADK/google-genai, Dagster, dbmate, Nuxt 4 SSR, **ADR-009 H100+FarSLIP+v8**)

**Estimación:** 1 punto (~0.5 días).

---

### US-065 — Observabilidad de chat: structlog + MLflow tags + latencia p95

**Como** operador,
- **quiero** métricas estructuradas de la capa conversacional y lineage de experimentos,
- **para que** se verifiquen los SLOs de latencia del chat y todo run quede trazable.

**Criterios de Aceptación:**

- Métricas de chat en logs estructurados (`structlog.get_logger()`, nunca `print()`): latencia por turno con objetivos **p95 < 3 s simple** y **< 15 s multi-step**, conteo de tool-calls por mensaje, tokens y modelo activo (Gemini Pro/Flash/Qwen) para FinOps.
- Lineage de experimentos en MLflow (server **Docker :5010**, no `./mlruns`) con tags `data_version` + `code_version`; documentar el gotcha de los dos almacenes (runs por subprocess quedan RUNNING contra el server equivocado).
- Dependencia funcional de US-052 (`/chat` SSE) para tener tráfico real que medir; mientras tanto, instrumentación lista y validada con un flujo sintético.
- Integración con el scaffolding de Prometheus/Grafana de US-059 (el panel "latencia chat p95" se puebla aquí).

**Tareas técnicas:**

- [ ] Middleware/instrumentación structlog de latencia y tokens por turno de chat
- [ ] Helper de logging de tool-calls y modelo activo (FinOps)
- [ ] Verificar tags MLflow `data_version`+`code_version` contra el server Docker :5010
- [ ] Panel Grafana "latencia chat p95" alimentado por las métricas de US-059

**Estimación:** 3 puntos (~1.5 días).

---

### US-066 — Atribuciones de licencia para nuevos datasets (multi-región)

**Como** equipo,
- **quiero** registrar las atribuciones de licencia de los datasets introducidos por la historia de transferencia multi-región,
- **para que** se cumpla la regla de atribuciones (CLAUDE.md §14) y el criterio de cumplimiento de la rúbrica.

**Criterios de Aceptación:**

- `docs/licenses/DATA_LICENSE.md` actualizado con: **Sen4AgriNet** (CC-BY-SA-4.0; subset Catalonia 31TCG vía DVC/GCS; HF `paren8esis/S4A`), **EuroCropsML** (CC-BY-SA-4.0, Zenodo DOI 10.5281/zenodo.15095445, `pip install eurocropsml`), y **AlphaEarth V1/ANNUAL v1.1** (CC-BY-4.0, asset GEE `SATELLITE_EMBEDDING/V1/ANNUAL`, global incl. México) — **corregir la entrada actual que dice "AlphaEarth Foundations v2.1"**.
- Si se ingiere WorldCereal RDM / Harmonized Global Crops (FUTURE/paper), agregar su atribución (RDM CC-BY por-colección; HGC HF `torchgeo/harmonized_global_crops` CC-BY-SA-4.0) y la nota de compatibilidad de licencias share-alike para el espacio unificado HCAT v3.
- Atribución requerida documentada para figuras/reportes derivados de cada región.

**Tareas técnicas:**

- [ ] Añadir entradas Sen4AgriNet + EuroCropsML en `docs/licenses/DATA_LICENSE.md`
- [ ] Corregir la entrada AlphaEarth (v2.1 → V1/ANNUAL v1.1, CC-BY-4.0)
- [ ] (Si aplica) entrada WorldCereal RDM + Harmonized Global Crops para el paper/FUTURE

**Estimación:** 1 punto (~0.5 días).

---

### US-067 — Documentación operativa de FinOps (`docs/operations/finops.md`)

**Como** MLOps / Platform lead,
- **quiero** una guía operativa única de FinOps que consolide los procedimientos de control de costo ya aplicados en el proyecto,
- **para que** el equipo y el jurado puedan verificar que el gasto cloud se mantiene en el objetivo (~$115 USD/mes operativo) y que las palancas de ahorro son reproducibles.

**Criterios de Aceptación:**

- Existe `docs/operations/finops.md` con: presupuesto objetivo (~$115/mes operativo + entrenamiento puntual), estado de gasto a la fecha (GCP ~$0.30–0.49 acumulado) y la nota de que la **H100 es prestada por el sponsor (24/7, no apagar)**.
- Documenta las palancas ya aplicadas: Cloud SQL dev con `activation_policy=NEVER` (var `db_activation_policy` en Terraform evita drift), shrink de disco `farslip-data` 250→125 GB vía snapshot→disco→rsync→import TF, daemon de auto-shutdown de la VM L4 por idle (Pub/Sub, no GPU), y `scale-to-zero` en Cloud Run.
- Aclara el caveat de créditos: el "Trial credit for GenAI App Builder" ($17,178) es de Vertex AI Search/Agent Builder y **NO** cubre la SKU de generación de texto de la Gemini API.
- Enlaza los scripts operativos permanentes (`scripts/azure_h100_*.sh`, `scripts/cost_audit.sh`) y la skill `agrosat-finops`; consume las métricas de latencia/tokens de US-059 para estimar costo por modelo en el switch A/B (US-054).

**Tareas técnicas:**

- [ ] Redactar `docs/operations/finops.md` consolidando procedimientos de las notas de memoria (Cloud SQL dev, disk shrink, L4 idle daemon)
- [ ] Tabla de presupuesto objetivo vs gasto real (GCP + Gemini API + H100 sponsor) con corte de fecha
- [ ] Sección de palancas de ahorro reproducibles con enlaces a `scripts/` y vars Terraform
- [ ] Nota de caveat de créditos Vertex AI Agent Builder vs Gemini API generación
- [ ] Cruzar costo por modelo (Pro/Flash/Qwen) con latencia/tokens de US-059 para el dashboard de FinOps del switch A/B

**Estimación:** 1 punto (~0.5 día).

---

**Subtotal EPIC 10: 13 story points** (US-059 2 + US-060 2 + US-061 1 + US-062 1 + US-063 1 + US-064 1 + US-065 3 + US-066 1 + US-067 1).

## EPIC 11: Paper Track — Semanas post-Presentación (Opcional) {#epic-11}

**Objetivo.** Redactar y submittear el paper a venue académico, ejecutado en la ventana buffer del 28 de junio al 3 de julio de 2026 post-presentación (con continuación asincrónica si se requiere). **NO afecta entregables del curso.** El v8 amplía el alcance del paper de "copiloto sobre Francia" a **experto multi-región con transfer learning demostrable + FarSLIP contrastivo-fenológico (directiva del sponsor)**, con métricas normalizadas a un harness único (ver §10 del plan v8).

**Alineado con.** Esta épica es completamente externa al Proyecto Integrador. Se ejecuta después de la Presentación Final del 27 de junio (calendario [ADR-008](../docs/decisions/ADR-008-rediseno-calendario-presentacion-27jun.md): A4 ✓31-may · A5 mié 10-jun · A6 14-jun · A7 21-jun · **Presentación 27-jun** · buffer/Paper 28-jun→3-jul), o asincrónicamente post-clase si se requiere más tiempo.

**Reality check (7-jun-2026).** La H100 NVL 96GB del sponsor (VM `gjcamacho-gpuh1`, env micromamba `agrosat`, `F:\projects\agrosat-copilot`) está disponible 24/7, lo que reabre la evaluación LLM de variante on-prem (Qwen3.5-35B-A3B vLLM) y el re-entrenamiento full de TSViT como insumos del paper. El cuello de botella es **tiempo** (~3 semanas hasta la presentación), no cómputo: por eso el Paper Track sigue siendo **opcional y post-presentación**. Correcciones factuales que el paper debe reflejar: AlphaEarth es **GEE asset `SATELLITE_EMBEDDING/V1/ANNUAL` data v1.1, CC-BY-4.0** (NO "v2.1"); Gemini 2.5-pro (GA, 1M ctx, NO 2M); **AgroMind es eval-only** (~28,482 QA sin train split) → cualquier fine-tune sobre AgroMind sería leakage; Swin-UNETR nunca se entrenó y **AnySat lo sustituye** (mIoU fold-4 0.4459); SegFormer corrió en variante **B0 con 3 bandas RGB** (no B2 10-banda); Gemma 4 26B LoRA queda como **future (ADR-009)**, no se entrena.

**Capacidad estimada:** 3 devs × 8 h/semana (dedicación reducida post-curso) × 2 semanas = 48 horas ≈ 20 SP realistas; con dedicación extra de miembros individuales part-time la capacidad sube a ~28 SP. El bloque multi-región (EPIC 12) es el más ambicioso y se difiere a continuación asincrónica si no entra.

**Puntos totales de la épica: 35.**

---

### US-068 — Construcción de benchmark AgroMind-IT/ES (500 pares)

**Como** equipo,
- **quiero** construir y publicar un benchmark bilingüe italiano/español con 500 pares Q&A agrícolas,
- **para que** sea contribución académica original publicable con DOI.

**Criterios de Aceptación:**

- 250 pares en italiano + 250 en español cubriendo las diez familias de preguntas del catálogo del copiloto (clasificación, cuantificación, vigor, estrés hídrico, fenología, comparación, anomalías, metadata, intersecciones, explicabilidad).
- Seed inicial generado sintéticamente con **Gemini 2.5-pro** (GA, 1M ctx, GA — el plan corrige el hardcode "gemini-3.1-pro") sobre imágenes reales Sentinel-2 de Italia.
- Revisión manual por hablantes nativos: italiano por reviewer de Scuola Sant'Anna (vía sponsor), español por miembro del equipo.
- El benchmark es estrictamente **eval-only**: ningún par entra a entrenamiento de ningún modelo (alineado con AgroMind original, que también es eval-only sin train split; entrenar sobre él sería leakage).
- Publicación en Zenodo con DOI y licencia CC-BY-4.0.
- Esquema JSONL compatible con AgroMind original para facilitar re-uso.

**Tareas técnicas:**

- [ ] Script de generación sintética con Gemini 2.5-pro
- [ ] Interfaz Streamlit para revisión humana
- [ ] Upload a Zenodo con metadata completa
- [ ] Declarar explícitamente el split eval-only en el README del dataset

**Estimación:** 6 puntos (~5 días).

---

### US-069 — Evaluación comparativa en GEO-Bench-2, AgroMind y AgroMind-IT/ES

**Como** equipo,
- **quiero** evaluar rigurosamente las dos variantes de reasoner (Gemini 2.5-pro cloud y Qwen3.5-35B-A3B on-prem) en tres benchmarks,
- **para que** la tabla de resultados del paper tenga error bars estadísticamente significativos.

**Criterios de Aceptación:**

- GEO-Bench-2 sobre las tasks agrícolas relevantes (≥3 de las 19 disponibles).
- AgroMind subset 1000 pares (eval-only; sin re-entrenamiento sobre él).
- AgroMind-IT/ES 500 pares (US-068).
- Variantes evaluadas: **Gemini 2.5-pro** (reasoner cloud, GA, 1M ctx) y **Qwen3.5-35B-A3B vLLM** (variante on-prem, GPTQ-Int4 single-GPU en H100, sin `--tensor-parallel-size`). **Gemma 4 26B se evalúa solo como base sin fine-tune** si entra (su LoRA está OUT — experts MoE 3D fused bloquean QLoRA; ver §8 del plan v8 / ADR-009).
- Métricas por variante: accuracy, F1, BERTScore, tool-call accuracy, hallucination rate, latencia p50/p95, costo por query.
- Tres corridas independientes con error bars y test Wilcoxon signed-rank para comparación pareada.
- Targets de referencia (no sobre-afirmar): AgroMind ≥0.75 Gemini / ≥0.70 Qwen.

**Tareas técnicas:**

- [ ] Harness extendido `ml/eval/paper_bench.py`
- [ ] Ejecución en ventana H100 post-presentación (reutilizando el serving Qwen3.5 de la variante-B)
- [ ] Exportación tabla LaTeX
- [ ] Logging MLflow con tags `data_version` + `code_version` (lineage en el server Docker :5010)

**Estimación:** 6 puntos (~5 días).

---

### US-070 — Figuras y tablas reproducibles del paper

**Como** equipo,
- **quiero** las figuras y tablas del paper generadas desde notebooks Python reproducibles,
- **para que** reviewers y lectores puedan regenerar cada resultado.

**Criterios de Aceptación:**

- Ocho figuras clave: arquitectura, mapas AOI Italia, UMAP AlphaEarth, **curvas de entrenamiento de modelos de segmentación (TSViT/TSViT-pheno full-config H100, NO Gemma 4 — su fine-tune está OUT)**, ejemplos conversacionales (IT/ES/EN), matriz de confusión, barplot de benchmarks, mapa de error espacial.
- Cinco tablas clave: comparativa de FMs (AlphaEarth V1/ANNUAL v1.1), modelos individuales EPIC 5 **re-scoreados con el harness único de US-030 en fold-5** (apples-to-apples, 18-clase contiguo, 128px NEAREST; corrige la mezcla previa 18-vs-20 cls / 64-128-256 res / SegFormer-B0 3-banda), ensambles EPIC 6 (4 de rúbrica + E-a/E-b incrementales), benchmark LLMs, ablación de tools.
- Una tabla/figura adicional de **ablación de bandas FarSLIP** (rgb vs nir-rgb falso-color vs 4band-pheno) que cuantifique el aporte NIR/fenología (directiva del sponsor).
- Cada figura/tabla generada desde `paper/notebooks/*.ipynb` con seed fijo y datos versionados en DVC; reutilizar el corpus ya poblado en `paper/figures/` y `paper/tables/` (us-010..us-023-preview, reencuadre_fenologico, breizhcrops).

**Tareas técnicas:**

- [ ] Plantillas matplotlib con estilo científico (CVPR/ISPRS)
- [ ] Notebooks reproducibles con `papermill`
- [ ] Exportación SVG + PNG de alta resolución
- [ ] Regenerar la fila de tabla de segmentación desde el harness de US-030 (sin cifras hardcodeadas)

**Estimación:** 6 puntos (~5 días).

---

### US-071 — Redacción, revisión y submission

**Como** equipo,
- **quiero** redactar el paper en LaTeX, revisarlo con el sponsor y enviarlo a venue,
- **para que** el trabajo trascienda el curso.

**Criterios de Aceptación:**

- Paper 10-15 páginas en Overleaf, template Remote Sensing MDPI (prioridad) o ISPRS Journal (ambicioso).
- Estructura: Abstract (250 palabras), Introduction, Related Work, Method, Experiments, Results, Discussion, Conclusion, References, Appendix.
- Tesis del paper alineada con el v8: **segmentación semántica (TSViT-pheno mejor individual, mIoU fold-4 0.6253 / F1 0.7500) + FarSLIP contrastivo-fenológico + AlphaEarth + capa conversacional LLM (patrón "Be My Eyes": perceiver = modelos, reasoner = Gemini frozen) con transferencia multi-región**.
- Claims defendibles únicamente (§11 plan v8): NO "VLM fine-tuned supera a Gemini" (capa LLM = comunicación/explicación + soberanía de datos Qwen on-prem); NO "F1≥0.80 en México" (demo metodológico zero-shot cualitativo); NO "zero-shot funciona fuera de Francia" (few-shot finetune con curva k-shot, citando "Harvesting AlphaEarth" arXiv:2601.00857 sobre transferibilidad espacial limitada); NO "TSViT 0.75+ mIoU" (PASTIS-R satura ~70%, target Full-M 0.68-0.72; rama fenológica aporta ~0.3%, no 5%).
- Atribuciones correctas: AlphaEarth Foundations `SATELLITE_EMBEDDING/V1/ANNUAL` v1.1 CC-BY-4.0, PASTIS-R, Sen4AgriNet CC-BY-SA-4.0, EuroCropsML CC-BY-SA-4.0, FarSLIP (Li et al. 2025), Gemma 4 / Qwen Apache 2.0.
- Revisión por Dr. Camacho antes de submission.
- Submission a arXiv cs.CV como pre-print (garantiza prioridad temporal).
- Submission a uno de los venues priorizados en orden: Remote Sensing MDPI (rolling), CVPR EarthVision Workshop 2026 si el deadline lo permite, ISPRS Journal.
- Repositorio GitHub público con README reproducible y licencia Apache 2.0.

**Tareas técnicas:**

- [ ] Overleaf project con template MDPI
- [ ] Revisión ortográfica + gramática en inglés con Grammarly
- [ ] Respuesta a revisores (iterativa post-submission)
- [ ] Verificar cada cifra del manuscrito contra MLflow/DVC (reproducibilidad sin números a mano)

**Estimación:** 10 puntos (~8 días).

---

### US-072 — Sección de paper: FarSLIP contrastivo-fenológico como camino principal (directiva sponsor)

> **Nota v8:** US nueva. Eleva el pivote FarSLIP de "ablación negativa" a contribución metodológica central del paper.

**Como** equipo,
- **quiero** documentar en el paper el pipeline FarSLIP-fenológico end-to-end (fix del bug `torch.randn`, prototipos de fenología reales vía Gemini Flash, ablación de bandas, protocolo incremental 4→18 con filtro 3:1 per-patch, y la fusión TSViT-pheno + FarSLIP del ensamble E-a),
- **para que** el resultado cumpla la directiva del sponsor y aporte una novedad publicable (alineación contrastiva sobre descripciones fenológicas reales, no ruido).

**Criterios de Aceptación:**

- Sección de método que explica el gap exacto: `ml/farslip/train.py:~184` inicializaba `text_prototypes` con `torch.randn` (InfoNCE contra ruido, por eso FarSLIP perdía 0.163 vs 0.233 de AlphaEarth); el fix inyecta prototipos fenológicos reales (Gemini Flash → embeddings MiniLM 384-dim, reproyección frozen 384→512).
- Reporte honesto del filtro **3:1 Meadow per-patch** (mantener patch si Meadow ≤ 3× la 2ª clase EN ESE PATCH, modo `dominance_ratio`), distinto del filtro `coverage` legacy de Isaac (`ml/data/pastis_filter.py`).
- Resultados de la ablación de bandas (rgb / nir-rgb falso-color / 4band-pheno) y del protocolo incremental 4→18 (Stage-1 → Stage-2 init `strict=False`), con convergencia documentada o fallback honesto a 18-clase desde cero.
- Marco académico "Be My Eyes" (Huang et al. 2025, arXiv:2511.19417): el LLM NO es clasificador de píxeles; el clasificador final es el ensamble E6. Esto valida el resultado negativo del equipo (pheno_text no ayuda como clasificador supervisado, ~0.3% no 5%) y lo enmarca correctamente.
- Citas a FarSLIP (Li et al. 2025, arXiv:2511.14901) y Phenology Description (Wen et al. 2025, ISPRS J. 228).

**Tareas técnicas:**

- [ ] Redactar subsección Method (pipeline F1-F5 del §4 del plan v8)
- [ ] Figura de fusión dual-head (FarSLIP 512-dim/parcela → 18 prototipos → cosine per-pixel por broadcast → α aprendible)
- [ ] Tabla de ablación de bandas + tabla incremental por stage (MLflow)
- [ ] Caveats explícitos (reproyección 384→512 cruda; ~500-700 patches/fold; pureza de parcela ~98%)

**Estimación:** 4 puntos (~3 días).

---

### US-073 — Sección de paper: transferencia multi-región (Sen4AgriNet + EuroCropsML, HCAT v3)

> **Nota v8:** US nueva. Convierte "funciona solo en Francia" en "experto multi-región con transfer learning demostrable". Es la US más ambiciosa de la épica; se difiere a continuación asincrónica si no entra en la ventana buffer.

**Como** equipo,
- **quiero** documentar la historia de transferencia Franco-Ibérica (Sen4AgriNet Catalonia denso + EuroCropsML few-shot tabular) armonizada a HCAT v3,
- **para que** el paper demuestre que el modelo recibe clases nuevas desde otros datasets y no queda restringido a Francia.

**Criterios de Aceptación:**

- Crosswalk taxonómico PASTIS-18 → HCAT v3 (códigos 10-dígitos) colapsado a ~10-20 macro-clases, documentado en `docs/data/hcat_crosswalk.md`.
- Camino **denso**: finetune de TSViT/U-TAE sobre subset Sen4AgriNet Catalonia (tiles 31TCG 2019/2020 + 1-2 tiles FR vía DVC/GCS, NO 10 TB) → modelo "Franco-Ibérico" que añade olivo/viñedo/sorgo/trigo-duro; reporte de Δ mIoU del domain gap zero-shot vs few-shot FR→ES.
- Camino **few-shot tabular**: AlphaEarth 64-dim + XGBoost sobre EuroCropsML (`pip install eurocropsml`, splits k=1/5/10/20/100/200/500-shot pre-codificados); curva k-shot Francia→Estonia/Portugal.
- Demo México zero-shot **cualitativo** (1-2 ejemplos aguacate/guayaba, alineación fenología-texto), enmarcado como metodología, sin claim de F1.
- WorldCereal RDM + Harmonized Global Crops (clases tropicales: arroz, soya, caña, algodón) declarado explícitamente como **FUTURE/escala** del paper, no demostrado en esta ventana.
- Caveat citado: "Harvesting AlphaEarth" (arXiv:2601.00857) — transferibilidad espacial limitada; presupuestar few-shot finetune, no prometer zero-shot mágico.

**Tareas técnicas:**

- [ ] Sección Experiments multi-región con el recipe train-Francia → extend-elsewhere (§7 del plan v8)
- [ ] Figura UMAP FR/ES en clusters separados + curvas NDVI desfasadas (domain gap)
- [ ] Tabla curva k-shot EuroCropsML
- [ ] Atribuciones en `DATA_LICENSE.md`: Sen4AgriNet CC-BY-SA-4.0, EuroCropsML CC-BY-SA-4.0, AlphaEarth V1 CC-BY-4.0

**Estimación:** 3 puntos (~2-3 días, diferible).

---

**Subtotal EPIC 11: 35 story points.**

---

## EPIC 12: Transferencia Multi-Región — Datasets y Few-Shot {#epic-12}

**Objetivo.** Consolidar la historia de transferencia más allá de Francia: crosswalk taxonómico PASTIS-18 → HCAT v3, ingestión densa Sen4AgriNet (Catalonia), few-shot EuroCropsML, demo cualitativa México y escala tropical WorldCereal/Harmonized (futuro). Reutiliza el harness de segmentación (US-030) y el baseline AlphaEarth+XGB.

**Alineado con.** Avances 6/7 (historia multi-región) + Paper Track (§EPIC 11). Estas US se consolidaron aquí desde EPIC 1/EPIC 2 para tener una sola fuente de verdad por dataset.

### US-074 — Crosswalk taxonómico PASTIS-18 → HCAT v3 (label space unificado)

**Como** ML Engineer,
- **quiero** una tabla de mapeo de las 18 clases PASTIS a la taxonomía jerárquica HCAT v3 (códigos 10-dígitos) colapsable a ~10-20 macro-clases,
- **para que** los datasets multi-región (Sen4AgriNet FAO-ICC, EuroCropsML HCAT) compartan un espacio de etiquetas común y el transfer learning entre regiones sea posible sin conflicto de gradiente.

**Criterios de Aceptación:**

- Tabla de mapeo PASTIS-18 → códigos HCAT-leaf (10 dígitos, jerarquía 4 niveles, prefijo "33") documentada en `docs/data/hcat_crosswalk.md`, reutilizando el crosswalk existente EuroCrops↔HCAT.
- Colapso a ~10-20 macro-clases vía el nivel de grupo de la jerarquía (mitiga el long-tail ~45% grassland del PASTIS).
- Convención void/background homogénea y documentada entre PASTIS, Sen4AgriNet y EuroCropsML.
- Estrategia para clases disjuntas (etiquetada en dataset A, background en B): protocolo partial-label / null-class (estilo UniSeg: BCE class-independent + cross-dataset relation loss) descrito como recomendación, sin implementar el loss en esta US.
- Reaprovechar los hallazgos del EDA cross-dataset (`02d_eda_breizhcrops.ipynb`) y de la rama `origin/feature/E2-breizhcrossdataset` como insumo del mapeo.

**Tareas técnicas:**

- [ ] `docs/data/hcat_crosswalk.md` con tabla PASTIS-18 → HCAT-leaf + colapso a macro-clases
- [ ] Tabla auxiliar `data/reference/hcat_crosswalk.parquet` (versionada, ligera) consumible por los adapters
- [ ] Revisar `origin/feature/E2-breizhcrops-crossdataset` para reutilizar lógica de mapeo previa
- [ ] Documentar la convención void/background unificada y la recomendación partial-label

**Estimación:** 5 puntos (~2.5 días). **IN — prerequisito de toda la transferencia multi-región, Avance 6/W3.**

---

### US-075 — Sen4AgriNet Catalonia: ingestión subset + adapter al pipeline de segmentación

**Como** ML Engineer,
- **quiero** ingerir un subconjunto de Sen4AgriNet (tiles de Catalonia 31TCG 2019/2020 + 1-2 tiles de Francia) y un adapter netCDF→tensores compatible con U-TAE/TSViT,
- **para que** se pueda finetunear el modelo denso entrenado en PASTIS-Francia sobre datos ibéricos y reportar el domain gap Franco-Ibérico de forma cuantitativa.

**Criterios de Aceptación:**

- Subset Sen4AgriNet versionado vía DVC/GCS: tiles Catalonia 31TCG (2019/2020) + 1-2 tiles FR — **NO** bajar los ~10 TB completos (subset ~objetivo manejable, referencia 281 GB del split completo).
- Adapter `netCDF → tensor (T,C,H,W)` compatible con el dataloader denso existente, replicando el binning mensual + `linear_encoder` del `patches_dataset.py` oficial; mismo paradigma denso LPIS→máscaras que PASTIS.
- Etiquetas remapeadas al label space HCAT unificado de US-074 (añade olivo, sorgo, trigo duro, arroz, algodón sobre PASTIS-FR).
- Escenario de transfer reportado: finetune denso FR→Catalonia (1 escenario), con Δ mIoU zero-shot vs few-shot como medida del domain gap (alineado con §7.3 del v8).
- Caveat honesto documentado: AlphaEarth tiene transferibilidad espacial limitada (arXiv:2601.00857) → presupuestar few-shot finetune, NO prometer zero-shot.

**Tareas técnicas:**

- [ ] Script de descarga selectiva del subset Sen4AgriNet desde HF `paren8esis/S4A` + `dvc add` del subset
- [ ] Adapter `ml/data/sen4agrinet_adapter.py` (netCDF→tensor, binning mensual, linear_encoder, remapeo HCAT)
- [ ] Tests del adapter sobre 1-2 patches de referencia (shape, rango de bandas, encoding de labels)
- [ ] Notebook/celdas de finetune denso FR→Catalonia + reporte Δ mIoU del domain gap
- [ ] Lineage MLflow con `data_version` (DVC) + `code_version`

**Estimación:** 8 puntos (~4 días). **IN — diferenciador multi-región, Avance 6/Avance 7 W3 si hay holgura.**

---

### US-076 — EuroCropsML few-shot transfer (Francia → Estonia/Portugal, k-shot)

**Como** ML Engineer,
- **quiero** correr el protocolo few-shot transnacional pre-codificado de EuroCropsML sobre el baseline tabular AlphaEarth+XGB,
- **para que** la historia de transferencia tenga una curva k-shot cuantitativa (cuántas muestras locales se necesitan para cerrar el domain gap), sin comprometer el camino crítico del Avance 5.

**Criterios de Aceptación:**

- `pip install eurocropsml` (dependencia vía `poetry add`); splits y protocolo k=1/5/10/20/100/200/500-shot listos.
- Curva few-shot del baseline AlphaEarth 64-dim + XGBoost sobre EuroCropsML (Francia→Estonia y/o Latvia+Portugal→Estonia), reportando F1 por valor de k (referencia del paper: 0.66 vs 0.57 a 500-shot).
- Espacio de etiquetas alineado a HCAT v3 (176 clases hoja) vía el crosswalk de US-074; subset ~4.8 GB versionado.
- Resultado enmarcado como evidencia del domain gap medido, no como claim de exactitud zero-shot.

**Tareas técnicas:**

- [ ] `poetry add eurocropsml` + descarga de splits (Zenodo DOI 10.5281/zenodo.15095445)
- [ ] Pipeline few-shot `ml/transfer/eurocropsml_fewshot.py` (loop sobre k, XGB sobre AlphaEarth)
- [ ] Gráfico curva k-shot exportado para el anexo + tabla por país
- [ ] Lineage MLflow con `data_version` + `code_version`

**Estimación:** 5 puntos (~2.5 días). **FUTURE / W3 si hay holgura — no en el camino crítico de A5.**

---

### US-077 — Demo México aguacate/guayaba (zero-shot cualitativo, NO métrica)

**Como** equipo,
- **quiero** 1-2 ejemplos cualitativos de aplicación del pipeline a cultivos mexicanos (aguacate/guayaba) usando AlphaEarth global,
- **para que** la presentación muestre la *metodología* zero-shot replicable a otras zonas, sin sobre-afirmar exactitud que no tenemos ground-truth para validar.

**Criterios de Aceptación:**

- 1-2 ejemplos cualitativos: centroide de una zona aguacatera (Michoacán) + alineación fenología-texto sobre la curva NDVI real, usando AlphaEarth V1/ANNUAL (global, incluye México, CC-BY-4.0).
- Enmarcado explícito como *metodología zero-shot cualitativa*, **sin claim de F1** (no hay ground-truth curado para México — ver §11.2 del v8).
- Reutiliza el pipeline de descripciones fenológicas (`ml/features/phenology_description.py`, Gemini Flash) y el espacio HCAT de US-074.
- Caveat documentado: F1≥0.80 mexicano validado queda FUTURE (requiere muestras curadas, fuera de scope de la presentación).

**Tareas técnicas:**

- [ ] Celdas de notebook con extracción AlphaEarth zonal para 1-2 AOIs mexicanas
- [ ] Alineación fenología-texto sobre la curva NDVI (cualitativa) + figuras para la demo
- [ ] Texto explícito de caveat metodológico (no métrico) en el notebook y en el guion de demo

**Estimación:** 3 puntos (~1.5 días). **IN (cualitativo) — refuerza el guion de presentación, Avance 6/W3.**

---

### US-078 — Multi-dataset a escala (WorldCereal África+Brasil, Harmonized Global Crops)

**Como** ML Engineer / investigador,
- **quiero** ingerir etiquetas tropicales/smallholder de WorldCereal RDM + Harmonized Global Crops y cruzarlas con AlphaEarth zonal,
- **para que** el proyecto demuestre escalabilidad a clases tropicales reales (arroz, soya, caña, algodón, sorgo) como trabajo de Paper Track post-presentación.

**Criterios de Aceptación:**

- Ingestión vía WorldCereal RDM API (GeoParquet, sin login, CC-BY por-colección) + Harmonized Global Crops (HF `torchgeo/harmonized_global_crops`, TorchGeo-ready, splits cross-region, CC-BY-SA-4.0).
- Clases tropicales nuevas mapeadas a HCAT v3 (US-074): arroz, soya, caña, sorgo, algodón, mijo, café, cacao, palma.
- Cruce de etiquetas vector con AlphaEarth zonal; protocolo train-Francia→finetune-elsewhere con Δ F1 reportado.
- Enmarcado como FUTURE/Paper Track: requiere jobs GEE zonales de días + curación taxonómica; no bloquea ningún Avance del curso.

**Tareas técnicas:**

- [ ] Cliente WorldCereal RDM API (GeoParquet) + ingestión Harmonized Global Crops vía TorchGeo
- [ ] Mapeo de clases tropicales a HCAT v3 (extiende `docs/data/hcat_crosswalk.md`)
- [ ] Jobs GEE de extracción AlphaEarth zonal para AOIs tropicales (África/Brasil)
- [ ] Reporte train-FR→finetune-elsewhere con Δ F1 por región

**Estimación:** 8 puntos (~4 días). **FUTURE / Paper Track (28-jun → 3-jul) — fuera del scope comprometido del MVP.**

---

### US-079 — Atribuciones de licencia para nuevos datasets multi-región

**Como** MLOps lead,
- **quiero** registrar en `docs/licenses/DATA_LICENSE.md` las atribuciones de los datasets multi-región y la corrección de la versión de AlphaEarth,
- **para que** el proyecto cumpla las licencias CC-BY-SA y CC-BY de cada fuente antes de redistribuir cualquier subset o derivado.

**Criterios de Aceptación:**

- `DATA_LICENSE.md` actualizado con: Sen4AgriNet (CC-BY-SA-4.0, HF `paren8esis/S4A`; capas LPIS pueden implicar ODbL — verificar), EuroCropsML (CC-BY-SA-4.0, Zenodo DOI 10.5281/zenodo.15095445), AlphaEarth **V1/ANNUAL v1.1 (CC-BY-4.0)** corrigiendo la mención previa "v2.1", Harmonized Global Crops (CC-BY-SA-4.0) y WorldCereal RDM (CC-BY por-colección).
- Cada entrada con cita (paper/DOI), URL de la fuente y obligaciones de atribución/share-alike.
- Sincronizado con la nota de §12.3 del v8 (corregir "AlphaEarth v2.1→V1/1.1" en CLAUDE.md/AGENTS.md).

**Tareas técnicas:**

- [ ] Editar `docs/licenses/DATA_LICENSE.md` con las 5 entradas + corrección AlphaEarth
- [ ] Añadir DOIs y URLs verificadas de cada dataset
- [ ] Cross-check con el checklist de cierre de US (atribución licencia obligatoria)

**Estimación:** 1 punto (~0.5 días). **IN — depende de los datasets multi-región (US-074..078), Avance 6/W3.**

---

**Subtotal EPIC 12: 30 story points** (US-074 HCAT 5 + US-075 Sen4AgriNet 8 + US-076 EuroCropsML 5 + US-077 México 3 + US-078 WorldCereal 8 + US-079 licencias 1). De ese total, US-076 y US-078 (13 SP) son FUTURE/diferidos.

> **US asíncronas post-v8 (EPIC 12, post-presentación — creadas después de ratificar este plan):**
> - **US-080** — Refinador FarSLIP open-set de segunda etapa (`ml/agent/refine.py`). Andamiaje; ejecución GPU diferida.
> - **US-081 — RESUELTA** ([docs/us-resolved/us-081.md](../docs/us-resolved/us-081.md), 2026-07-02). Copiloto sobre el campeón desplegado **Voting-3 v2** (12 clases, `france-12`): default `xgb→voting3` con degradación limpia, `/chat` SSE con out-of-vocab handoff, tercer backend on-prem **Qwen3.6-VL** (`qwen-vl`) + UI de selector/gráfica, y 4 mejoras (hedge A/B, open-set, routing availability-aware, calibración). QA Fase 6 `ok`. Uplift real france-12 0.7480→**0.8992** (n=14,688). Bloqueos residuales: deploy Cloud Run + scorecard LLM real (creds).
> - **US-082 — CERRADA (andamiaje + QA), KPIs de métrica PENDIENTES de VM/GEE** ([docs/us-resolved/us-082.md](../docs/us-resolved/us-082.md), 2026-07-02). Diagnóstico de la causa raíz del TL Italia 0.13 (piloto del 1 %, no bug) + andamiaje de re-extracción full-1438 / EDA / 3-vías / separabilidad + consolidación del TL en el paper (es+en) + corrección de fechas DE4 en la presentación (56→~41). La extracción full + re-entreno (KPI-1/2/3) sigue bloqueada por VM H100/GEE.

---

## Roadmap de Sprints (vigente, ADR-008)

```
S6  (25-31 may): E5 6 segmentadores                          → Avance 4 ✓31-may
S7  (1-10 jun):  E5 normalización métricas + FarSLIP-pheno
                 + E6 ensambles (4 base + E-a)                → Avance 5 mié 10-jun
S8  (8-14 jun):  E6 E-b + TSViT full H100 + E2 transfer
                 Franco-Ibérico + E7/E8 backend+agente MVP    → Avance 6 dom 14-jun
S9  (15-21 jun): E9 frontend + E10 observabilidad/RLS
                 + multi-región + eval LLM                    → Avance 7 dom 21-jun
S10 (22-27 jun): Pulido + dry-runs + grabar demo             → Presentación dom 27-jun
S11 (28-jun-3-jul): Buffer + Paper Track opcional
```

**Orden estricto de la H100 (1 GPU):** FarSLIP-pheno ablación → TSViT full retrain → ensambles OOF → Qwen serving. **No** Gemma LoRA antes del 27-jun.

---

## 4. Gestión de Riesgos

| Riesgo | Prob | Mitigación |
|--------|------|------------|
| H100 una sola GPU, cola consume días | Alta | Orden estricto; fallback L4 para ablación de bandas |
| `ml/ensemble/` vacío a días de A5 | Alta | US-030/031 (harness + OOF) primero; 4 ensambles base = MVP |
| Tabla 6-modelos no apples-to-apples | Alta | US-030: re-score con un solo harness, reportar fold-5 |
| RLS migración falla → data leak | Media | Test en docker-compose local antes de exponer endpoints |
| Incremental FarSLIP 4→18 no converge | Media | POC 2-epoch antes de full; fallback 18-desde-cero |
| Gemma 4 LoRA (experts 3D) | — | Diferido (US-050); usar Gemini + Qwen vLLM |
| Transfer México sin ground-truth | — | Demo metodológico cualitativo, no F1 validado |

---

## 5. Criterios de Éxito del MVP (realistas)

| Métrica | Objetivo v8 | Estado |
|---------|-------------|--------|
| Baseline F1-macro (6 familias HCAT) | ≥0.60 | ✅ 0.6535 |
| Segmentación mIoU (TSViT-pheno) | ~0.625; full H100 0.68-0.72 | 🟠 0.6253, retrain en curso |
| Ensamble parcela ≥ mejor individual | tabla honesta fold-5 | 🔴 a construir |
| FarSLIP-pheno > 0.163 previo | cerrar gap vs AlphaEarth | 🟠 en curso |
| Transfer Franco-Ibérico | curva few-shot FR→ES (Δ mIoU) | 🟠 demoable |
| Latencia chat p95 | <3s simple, <15s multi-step | 🔴 sin chat aún |
| Cobertura tests | ≥70% backend, ≥50% frontend | 🟠 ML ✅, negocio ❌ |

**No reclamar sin evidencia:** F1≥0.80 mexicano validado (sin ground-truth), "VLM propio supera a Gemini" (sin training honesto), scores AgroMind/GeoAnalystBench sin correrlos.

---

## 6. Alineación con Rúbricas

- **Avance 5 (10-jun):** Ensambles /60 (4 base + E-a, homogéneo Y heterogéneo) · Selección /20 (tabla mejor-individual + ensambles, fold-5, ≥2 métricas + tiempos) · Gráficos /20 (≥4 interpretados). Ver EPIC 6.
- Rúbricas oficiales: [`docs/general/Rubricas Integrador.html`](../docs/general/Rubricas%20Integrador.html). Verificar antes de cada entrega.

---

*Plan vigente v8. El cuerpo lee como lo que se ejecuta; lo descartado se documentó en §0 (Cambios vs v6) y no se arrastra en las US. Mantenedor: Arthur Zizumbo (MLOps lead). Última actualización: 7-jun-2026.*

