# AgroSatCopilot — Orquestador de Agentes IA

**Proyecto**: SaaS conversacional open-source para análisis satelital agrícola.
**Stack**: FastAPI + Polars | Nuxt 4 SSR | PostgreSQL 15 + PostGIS + pgvector | Google ADK | Gemma 4 26B-MoE LoRA | Dagster + dbmate + DVC + MLflow | Terraform GCP + Azure H100 NVL 96GB.
**Curso**: MNA Tec de Monterrey · 20-abr → 3-jul-2026 · sponsor Dr. Camacho (gjcamacho@tec.mx).

> Orquestador único. [`CLAUDE.md`](CLAUDE.md) es espejo idéntico (Claude Code lo lee; Codex/Cursor/otros leen `AGENTS.md`). Sub-agentes por directorio sobreescriben en su scope.

## Referencias canónicas

| Documento | Propósito |
|-----------|-----------|
| [`context/RefinamientoPlaneacionAgroSatCopilot_v6_RESUMEN.md`](context/RefinamientoPlaneacionAgroSatCopilot_v6_RESUMEN.md) | Plan operativo destilado (172 líneas): EPICs, SP, calendario, riesgos, métricas |
| [`context/RefinamientoPlaneacionAgroSatCopilot_v6.md`](context/RefinamientoPlaneacionAgroSatCopilot_v6.md) | Plan SCRUM completo con US-001 a US-056 |
| [`docs/orchestration/skills-catalog.md`](docs/orchestration/skills-catalog.md) | Catálogo de las 30 skills `agrosat-*` |
| [`docs/orchestration/auto-invoke.md`](docs/orchestration/auto-invoke.md) | Qué skill cargar antes de cada acción |
| [`docs/orchestration/skill-owners.md`](docs/orchestration/skill-owners.md) | Mapa skill → subagente owner + 9 subagentes |
| [`docs/orchestration/commands.md`](docs/orchestration/commands.md) | Lista completa de targets `make` |
| [`docs/general/Rubricas Integrador.html`](docs/general/Rubricas%20Integrador.html) | Rúbricas oficiales — verificar antes de cada entregable |
| [`docs/licenses/DATA_LICENSE.md`](docs/licenses/DATA_LICENSE.md) | Atribuciones AlphaEarth, Sentinel, DINOv3, Gemma 4, Qwen, PASTIS, AgroMind |

## Equipo

| Integrante | Rol |
|------------|-----|
| Arthur Zizumbo | MLOps / Platform Lead — Terraform, CI/CD, DVC, MLflow, Dagster, FinOps |
| Aaron Bocanegra | Full-Stack / Backend Lead — FastAPI, TiTiler, Nuxt 4, endpoints ADK, seguridad |
| Isaac Ávila | ML / Data Scientist — modelos, fine-tune Gemma 4 + Qwen3-VL, AlphaEarth, Polars |

Capacidad: 12 h/sem × 3 devs × 10 sem ≈ **150 SP** para MVP.

## Routing por Directorio

| Directorio | Sub-AGENTS.md | Especialidad |
|------------|---------------|--------------|
| `backend/` | [backend/AGENTS.md](backend/AGENTS.md) | FastAPI, SQLModel, TiTiler, SSE, Pub/Sub workers |
| `frontend/` | [frontend/AGENTS.md](frontend/AGENTS.md) | Nuxt 4 SSR, MapLibre + deck.gl, @ai-sdk/vue, i18n |
| `ml/` | [ml/AGENTS.md](ml/AGENTS.md) | Segmentación, fine-tune LoRA, AlphaEarth, DINOv3 |
| `ml/agent/` | [ml/agent/AGENTS.md](ml/agent/AGENTS.md) | Google ADK, 9 tools geoespaciales, Spatial-RAG |
| `dagster_project/` | [dagster_project/AGENTS.md](dagster_project/AGENTS.md) | Assets, jobs, schedules, DVC ↔ MLflow lineage |
| `db/` | [db/AGENTS.md](db/AGENTS.md) | dbmate, PostGIS, pgvector, pgstac, RLS por sesión |
| `infrastructure/` | [infrastructure/AGENTS.md](infrastructure/AGENTS.md) | Terraform GCP + Azure H100, Cloud Build |
| `notebooks/` | [notebooks/AGENTS.md](notebooks/AGENTS.md) | Avances curso, EDA Polars, papermill |
| `paper/` | [paper/AGENTS.md](paper/AGENTS.md) | Paper Track opcional, GEO-Bench-2, AgroMind-IT/ES |

## Decisiones Irrevocables (NO cambiar sin equipo)

**Stack ML** (HF verificado 24-abr-2026):

| Capa | Modelo | Nota clave |
|------|--------|------------|
| FM EO | AlphaEarth Foundations v2.1 | GEE gratis · NO entrenar FM propio |
| Feature self-sup | DINOv3-satellite | `facebook/dinov3-vitl16-pretrain-sat493m` frozen |
| VLM principal | Gemma 4 26B-MoE | Apache 2.0 · LoRA rank 16 BF16 · ~82 GB H100 |
| VLM comparativo | Qwen3-VL-30B-A3B | MoE 30B/3B · 256K ctx |
| LLM cloud | Gemini 3.1 Pro | Vertex AI · 2M ctx |
| LLM on-prem | Qwen3.5-35B-A3B | vLLM en H100 · sin `-Instruct` |
| Framework agente | Google ADK | Tracing built-in + Vertex AI Agent Engine |

**Arquitecturas obligatorias por rúbrica**:

- EPIC 5 (6 segmentación): U-Net · DeepLabv3+ · SegFormer-B2 · U-TAE · **TSViT (Paper 1)** · Swin-UNETR.
- EPIC 6 (4 ensambles): Voting top-3 · Bagging XGB+AlphaEarth · Stacking (+ Gemma 4) · Blending Optuna.

**Stack plataforma**:

- Backend: FastAPI 3.12, TiTiler, PostgreSQL 15 + PostGIS + pgvector + pgstac, Redis, Polars 1.x, dbmate.
- Frontend: Nuxt 4 SSR (no PWA, no Tauri), MapLibre + deck.gl (OSS), `@ai-sdk/vue`, Nuxt UI Pro, Tailwind v4, `@nuxtjs/i18n` (it/es/en), Pinia, Clerk OAuth.
- MLOps: DVC 3.48+ GCS, MLflow 2.16, Dagster 1.9+ asset-oriented, GitHub Actions + Cloud Build, Terraform 1.9+, Evidently AI.
- Infra: GCP primaria + Azure 1×H100 NVL 96GB spot + Azure Blob.

**Descartados (no reactivar)**: Prithvi-EO-2.0, MiniMax-M2.7, Kimi K2.6, Llama 3.3-70B QLoRA, LangGraph, Prefect, Alembic, DuckDB principal, PWA+Tauri.

## Calendario Inamovible

Avance 0 (26-abr PDF) · Avance 1 EDA (3-may) · Avance 2 FE (17-may) · Avance 3 Baseline (20-may) · Avance 4 Modelos (24-may) · Avance 5 Final+Ensambles (31-may) · Avance 6 Conclusiones (7-jun PDF) · Avance 7 Resumen (14-jun PDF) · **Presentación final dom 21-jun** · Buffer + Paper Track opcional 22-jun→3-jul.

Sprints semanales + gates en [`context/RefinamientoPlaneacionAgroSatCopilot_v6_RESUMEN.md`](context/RefinamientoPlaneacionAgroSatCopilot_v6_RESUMEN.md) §6 y §12.

## Presupuesto Cómputo (80 h H100 + ~50 h L4 spot)

| Ventana | Fecha | Uso | VRAM |
|---------|-------|-----|------|
| V1 | 18-20 may (8 h) | Baselines + preliminar TSViT | ~40 GB |
| V2 | 25-27 may (12 h) | U-TAE + TSViT + Swin-UNETR | ~60 GB |
| V3 | 28-30 may (24 h) | **Gemma 4 26B-MoE LoRA** | ~82 GB |
| V4 | 1-3 jun (12 h) | Qwen3-VL LoRA + ensambles | ~92 GB |
| V5 | 5-7 jun (16 h) | Qwen3.5-35B-A3B vLLM + LoRA traces | ~91 GB |
| V6 | 18-20 jun (8 h) | Warm vLLM demo | ~91 GB |

Training único $262 spot — $602 on-demand USD. Operativo **~$115 USD/mes** con scale-to-zero.

## Reglas Globales NON-NEGOTIABLE

1. **Decisiones irrevocables**: NO se cambian sin consultar al equipo.
2. **Idioma**: código en inglés; docstrings/docs en español neutro; UI it+es+en simultáneo vía `@nuxtjs/i18n`.
3. **Multi-tenant por `session_id`** (no `wedding_id`). Toda query filtra por sesión/usuario.
4. **Sin emojis** en código, comentarios, prints, commits ni logs estructurados.
5. **Secrets**: jamás hardcodear. `.env.local` en dev, Secret Manager (GCP) o Key Vault (Azure) en prod.
6. **Dependencias**: `poetry add` (Python), `pnpm add` (frontend). Nunca npm/yarn/pip ad-hoc.
7. **DRY**: función usada 2+ veces → `backend/app/utils/`, `ml/utils/` o `frontend/composables/`.
8. **Separation of Concerns**: router recibe → service procesa → model persiste. Tools ADK en `ml/agent/tools/`, nunca en routers.
9. **Inferencia pesada (>2 s) o entrenamiento**: Pub/Sub + Cloud Run GPU L4 worker, nunca síncrono.
10. **Versionado**: DVC para datos, MLflow para experimentos (tags `data_version` + `code_version`), Conventional Commits con scope de epica (`feat(E6): ...`).
11. **Migraciones**: solo `dbmate up` / `dbmate new`. Jamás `SQLModel.metadata.create_all()` en prod ni modificar migraciones aplicadas.
12. **Reproducibilidad notebooks**: notebooks se commitean **ejecutados end-to-end con todas sus salidas** (tablas HTML, figuras PNG inline, plots interactivos) para entregable visual del curso. Papermill end-to-end en CI valida que sigan ejecutables. **Sin `.pre-commit-config.yaml`** ni `nbstripout` en quality gates — el `.ipynb` es un artefacto reproducible, no fuente "limpia".
13. **i18n obligatorio**: todo texto visible en `frontend/i18n/locales/{it,es,en}.json` simultáneamente.
14. **Atribuciones licencia**: documentadas en [`docs/licenses/DATA_LICENSE.md`](docs/licenses/DATA_LICENSE.md).

## Quality Gates (sin pre-commit)

```bash
make check       # lint + secrets-scan + i18n-check (obligatorio antes de PR)
make lint        # ruff + mypy + pnpm lint
make secrets-scan ; make i18n-check
make notebooks-check  # papermill end-to-end (notebooks ejecutables, con outputs preservados)
```

CI replica `make check` en cada PR a `develop` y `main`. Comandos completos en [`docs/orchestration/commands.md`](docs/orchestration/commands.md).

## Anti-Patrones (NUNCA)

- Entrenar un Foundation Model propio — AlphaEarth ya lo provee.
- Llamar Vertex AI / vLLM directamente desde router — siempre vía service o tool ADK.
- Lógica de negocio en routers o componentes Vue — delegar a services / composables / Pinia.
- Inferencia de modelos en el frontend — todo va por `/chat` SSE al backend.
- Subir GeoTIFF/COG/pesos al repo Git — siempre DVC + GCS.
- Encender VM H100 sin `make azure-h100-start` y sin auto-shutdown.
- Modificar migraciones dbmate ya aplicadas — crear `dbmate new` rollforward.
- Strings hardcodeados sin `t('key')` — i18n bloquea merge.
- Traducción en un solo idioma sin los otros dos.
- `print()` en producción — usar `structlog.get_logger()`.
- Endpoints `/chat`, `/aois`, `/llm/switch` sin guard de auth ni rate limit.
- Agregar dependencias fuera del stack aprobado.
- Reintroducir `.pre-commit-config.yaml` — quality gates viven en Makefile + CI.
- Reintroducir `nbstripout` en quality gates — los notebooks se commitean **con sus outputs poblados** (entregable visual). Strip solo on-demand si Isaac lo pide explicitamente para un commit puntual.
- Crear scripts ad-hoc `scripts/_*.py` para smoke / debug / diagnose — la validacion va en `tests/ml/` (pytest) o inline en el notebook con `display()`. `scripts/` solo aloja operativos permanentes (`azure_h100_*.sh`, `cost_audit.sh`, `verify_structure.sh`, etc.).

## Autonomía de Claude

**Sí (autónomo)**: nombres de funciones/variables, refactorizar preservando interfaces, tests/docstrings/docs inline, helper scripts en `scripts/`, optimizar queries con profiling, proponer (no implementar) cambios al plan en `docs/decisions/ADR-XXX.md`.

**NO (consultar humano)**: cambiar decisiones irrevocables, calendario, presupuesto cloud, dependencias del stack aprobado, contrato de tools ADK (Pydantic schema) sin coordinar frontend.

## Estilo de Respuesta

- **Antes del primer tool call**: una frase con el plan (≤20 palabras).
- **Tareas con >3 tool calls o >30 s**: usar `TodoWrite` al inicio.
- **Decisiones con trade-offs reales** (>1 enfoque viable): opciones en 1 línea cada una + pedir decisión.
- **Silencio >60 s** en multi-step: prohibido. Confirmar estado y siguiente paso.
- **Prevención de loops**: si fallo 3 veces consecutivas la misma acción, detener, explicar error en 2 líneas, pedir intervención.

**Respuestas finales**:

- Triviales (fix, rename, 1-2 edits): ≤4 líneas, sin preámbulo ni resumen.
- Con >3 archivos o research: 2-4 líneas + links `[nombre](ruta)`.
- Código > prosa: el diff es la respuesta.

**Eficiencia**: tool calls independientes en paralelo · `offset`/`limit` en Read · Grep antes que Read · no crear archivos intermedios salvo pedido · solo lo preguntado.

**Prohibido**: "Perfecto, voy a...", "Listo, he...", narrar tool calls, repetir tool results en prosa, markdown decorativo en respuestas cortas.

## Decision Tree Operativo

```
¿Nueva US?
  1. Buscar US-XXX en context/RefinamientoPlaneacionAgroSatCopilot_v6_RESUMEN.md → §13 navegación
  2. Verificar criterios aceptación en plan completo
  3. Crear rama feature/E{epic}-US-XXX-{slug}
  4. Leer AGENTS.md del subdirectorio relevante
  5. Cargar skill(s) según docs/orchestration/auto-invoke.md

¿Tipo de tarea? → cargar skill via docs/orchestration/auto-invoke.md
¿Necesito profundidad o paralelización? → invocar subagente via docs/orchestration/skill-owners.md
```

## Checklist Cierre de US

- [ ] Rama `feature/E{epic}-US-XXX-{slug}` mergeada vía PR a `develop`
- [ ] Conventional Commit `feat(EX): ...`
- [ ] Código en inglés; docstrings Google style en español con type hints
- [ ] Tests cobertura ≥70 % backend, ≥50 % frontend
- [ ] `make check` limpio (lint + secrets + i18n)
- [ ] Si la US incluye notebook: ejecutado end-to-end con papermill + commiteado con outputs poblados (HTML tables + PNG inline)
- [ ] Migraciones con `dbmate up` si tocó schema
- [ ] Secrets vía `.env.local` / Secret Manager
- [ ] Si entrenó modelo: MLflow con `data_version` + `code_version`
- [ ] Si generó data: DVC (`dvc add` + commit del `.dvc` file)
- [ ] i18n: `it.json`, `es.json`, `en.json` sincronizadas
- [ ] Atribución licencia en `docs/licenses/DATA_LICENSE.md` si nuevo dataset/modelo
- [ ] `docs/us-resolved/us-XXX.md` al cerrar la US completa
- [ ] Rúbrica del Avance verificada en `docs/general/Rubricas Integrador.html`

## Métricas Éxito (mínimos)

- Baseline F1-macro ≥ 0.60 (AlphaEarth + XGBoost).
- Modelo final F1-macro ≥ 0.80 (Gemma 4 LoRA + ensambles).
- mIoU ≥ 0.70 segmentación densa.
- Latencia chat p95 < 3 s simple, < 15 s multi-step.
- AgroMind ≥ 0.70 (Qwen3.5-35B-A3B), ≥ 0.75 (Gemini 3.1 Pro).
- GeoAnalystBench pass rate ≥ 0.65.
- Cobertura tests ≥ 70 % backend, ≥ 50 % frontend.

Detalle producto + MLOps en [`context/RefinamientoPlaneacionAgroSatCopilot_v6_RESUMEN.md`](context/RefinamientoPlaneacionAgroSatCopilot_v6_RESUMEN.md) §9.

## Contacto

- ML/MLOps → Arthur Zizumbo.
- Backend/Frontend → Aaron Bocanegra.
- Modelos/Datasets/EDA → Isaac Ávila.
- Académicas/plan/recursos → Sponsor Dr. Camacho (gjcamacho@tec.mx).
- Rúbricas → [`docs/general/Rubricas Integrador.html`](docs/general/Rubricas%20Integrador.html).

---

**Última actualización**: 11-may-2026 — compactado a referencias modulares en `docs/orchestration/` + resumen v6 destilado.
**Mantenedor**: Arthur Zizumbo (MLOps lead).
