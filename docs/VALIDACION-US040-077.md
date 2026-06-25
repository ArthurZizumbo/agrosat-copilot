# Tablero maestro de validacion — US-040 a US-077

> Pasada de validacion autonoma (2026-06-24/25) de todas las US no validadas
> desde la US-040. Una rama por EPIC, encadenadas
> (`E6 -> E7 -> E8 -> E9 -> E10 -> E12-FINAL`). Cada EPIC cierra con
> us-resolved + manual-test + handoff actualizado + blocker. Estado POST-validacion
> (no el del descubrimiento inicial).

## Resumen ejecutivo

| EPIC | US | Estado | Evidencia clave | Commit |
|------|----|--------|-----------------|--------|
| **E6** Ensambles | 040-043 | ✅ cerrado | Champion Stacking-5 F1 0.749; E-a/E-b fallo conceptual documentado honestamente | `678cb45` |
| **E7** Agente | 045-050 | ✅ cerrado | **Re-cableo perceiver -> champion: +11 pp accuracy real (H100, 13.481 parcelas)** | `efe757f` |
| **E8** Backend | 051-056 | ✅ cerrado | **146 tests** (29 RLS con Postgres real) + fix mypy/ruff | `91123b9` |
| **E9** Frontend | 057-058 | ✅ cerrado | 53 vitest + typecheck + i18n(it/es/en) + lint + build SSR | `197e884` |
| **E10** Obs/Docs | 059-067 | ✅ cerrado | metrics 5 tests + drift 11; FinOps/model cards cruzados con CSVs reales | `063504c` |
| **E12** Multi-region | 074-077 | ✅ cerrado | Transfer FR->Catalonia +0.2468 mIoU; demo Mexico GEE real | (rama FINAL) |

**El hallazgo central** (reportado por Arthur, confirmado y resuelto): el agente
conversacional percibia con el clasificador **base** (xgb-alphaearth), no con el
ensamble **campeon** (Stacking-5, US-043). Se re-cablo al campeon restringido a las
nueve clases mejor resueltas (`france-9`). Impacto medido sobre 13.481 parcelas
reales fold-5: **accuracy 0.831 -> 0.941 (+11.0 pp)**, macro-F1 **0.687 -> 0.901**,
neto **+1.490 parcelas** corregidas.

## Detalle por US

### EPIC 6 — Ensambles (rama `validacion/E6-ensambles`)

| US | Tema | Estado | Metrica / evidencia |
|----|------|--------|---------------------|
| US-040 | Ensambles base (voting/bagging/stacking/blending) | ✅ ok | Stacking F1 **0.7470** (champion base), notebook Avance5 completo |
| US-041 | E-a fusion dual-head | ⚠️ negativo documentado | F1 **0.2694** (-0.407 vs TSViT): fusion ingenua 4-vs-18 clases. Experimento honesto |
| US-042 | E-b stacking + AlphaEarth | ⚠️ suboptimo documentado | F1 **0.3395** (hereda base E-a rota) |
| US-043 | Stacking-5 +FarSLIP | ✅ campeon | F1 **0.7486** (18 clases) / ~0.912 (france-9). Champion final |

> Notebook `notebooks/final_model/Avance5.Equipo17.ipynb` ya documentaba los 7
> ensambles; no se rehizo. Blocker: `docs/blockers/epic6-notas.md`.

### EPIC 7 — Agente conversacional (rama `validacion/E7-agente`)

| US | Tema | Estado | Evidencia |
|----|------|--------|-----------|
| US-045 | 9 FunctionTools geoespaciales | ✅ ok | 130 tests agente verdes |
| US-046 | Perceiver Be My Eyes + Spatial-RAG | ✅ ok | **Re-cableo champion** (+11 pp); rag 10 + retrieve_context 6 |
| US-047 | Agent factory + stream_response | ✅ ok | 33 tests (agent 17 + backends 16) |
| US-048 | Serving Qwen on-prem H100 | ✅ ok | llama.cpp end-to-end, /health 200, ~24.8 tok/s |
| US-049 | Eval copiloto (AgroMind + GeoAnalystBench) | ✅ ok | Harness real; grounded_crop usa stub por diseno (aislado del re-cableo) |
| US-050 | Gemma 4 LoRA FUTURE (ADR-011) | ✅ ok | Documental |

> Nuevo modulo `ml/eval/perceiver_champion_eval.py` + resultado
> `reports/agent_bench/perceiver_champion_eval.json`. Notebook
> `Avance6.Demo.Copiloto.Equipo17.ipynb` actualizado al champion. Regresion verde
> (130 agente + 47 backend). Blocker: `docs/blockers/epic7-notas.md`.

### EPIC 8 — Backend FastAPI (rama `validacion/E8-backend`)

| US | Tema | Estado | Evidencia |
|----|------|--------|-----------|
| US-051 | RLS multi-tenant | ✅ ok | 9/9 aislamiento con rol no-superuser, Postgres real |
| US-052 | POST /chat SSE endurecido | ✅ ok | rate-limit + auth-guard + RLS; 13+12 tests |
| US-053 | 4 endpoints geoespaciales | ✅ ok | 25 tests (conteo corregido vs handoff); fix ruff |
| US-054 | POST /llm/switch | ✅ ok | 34 tests; validado vs Qwen real :8002 |
| US-055 | TiTiler COG tiles | ✅ ok | SSRF mitigado (allowlist); COG real |
| US-056 | Worker Pub/Sub | ✅ ok | Scaffolding honesto (sync funciona, pubsub diferido) |

> Total **146 tests backend** (77 unit + 40 integracion + 29 testcontainers +
> 25 ML). Fix mypy (`metrics.py`) + ruff. Blocker: `docs/blockers/epic8-notas.md`.

### EPIC 9 — Frontend Nuxt 4 (rama `validacion/E9-frontend`)

| US | Tema | Estado | Evidencia |
|----|------|--------|-----------|
| US-057 | ChatPanel/ChatDock | ✅ ok | markdown + tool cards + persist + retry + locale |
| US-058 | MapView | ✅ ok | MapLibre + Esri keyless + draw AOI + link parcela->chat |

> Gates verdes: **vitest 53/7**, typecheck 0 err, i18n parity (it/es/en), eslint
> limpio, build SSR 17.5 MB. Fix de honestidad (`chat.ts` comentario fantasma) +
> `frontend/CLAUDE.md` actualizado. **Validacion E2E en vivo (Playwright) pendiente
> de `backend/.env.local`** — paso manual documentado. Blocker:
> `docs/blockers/epic9-notas.md`.

### EPIC 10 — Observabilidad, Docs, FinOps (rama `validacion/E10-observabilidad`)

| US | Tema | Estado | Evidencia |
|----|------|--------|-----------|
| US-059 | Metricas Prometheus | ✅ ok | 5 tests metrics + 3 dashboards + alert_rules |
| US-060 | Observabilidad/drift | ✅ ok | drift 11 tests, schedule semanal |
| US-061 | Tablas/figuras Avance 7 | ✅ ok | costo_beneficio .md/.tex/.csv |
| US-062 | Riesgos 4 categorias | ✅ ok | riesgos.md (matriz 3x3) |
| US-063 | Comparativa proveedores | ✅ ok | GCP vs Azure, precios con fuente |
| US-064 | Security + model cards | ✅ ok | OWASP A01-A10 + 4 model cards trazadas a CSVs |
| US-065 | Token usage observability | ⚠️ warn | tokens Qwen None sin include_usage; deps externas |
| US-066 | Licencias de datos | ✅ ok | DATA_LICENSE.md |
| US-067 | Drift + FinOps | ✅ ok | finops.md (4 palancas, cifras cruzadas) |

> Blocker: `docs/blockers/epic10-notas.md` (B1-B19 previos + seccion validacion).

### EPIC 12 — Transferencia Multi-Region (rama `validacion/E12-multiregion-FINAL`)

| US | Tema | Estado | Metrica real |
|----|------|--------|--------------|
| US-074 | Crosswalk HCAT (PASTIS->HCAT v3) | ✅ ok | Notebook 02f con outputs |
| US-075 | Finetune FR->Catalonia (Sen4AgriNet) | ✅ ok | **mIoU 0.0000 -> 0.2468** few-shot (k=10, 40 ep) |
| US-076 | EuroCropsML few-shot | ✅ ok | 706.683 .npz reales (EE+LV+PT, 4.15 GB) |
| US-077 | Demo Mexico aguacate/guayaba | ✅ ok | GEE real, zero-shot cualitativo, 2 figuras |

> Blocker: `docs/blockers/epic12-notas.md`.

## Blockers transversales (lo que necesita Arthur)

| ID | Que | Severidad | Accion |
|----|-----|-----------|--------|
| B-E9-1 | Validacion E2E Playwright requiere `backend/.env.local` (ausente) | MEDIA | Configurar .env + sembrar DB; correr flujos manual-test |
| B-E7-2 | 23 fallos pre-existentes en `tests/ml/eval/` (regex i18n + fixtures) | MEDIA | Alinear `match=` al mensaje ingles; versionar fixtures JSONL |
| B-E10-V1 | US-065 tokens Qwen None sin `include_usage` | BAJA | Anadir include_usage a la peticion streaming Qwen |
| B-E6-1 | test_us043_orchestrator espera tsvit-pheno, codigo usa fullm | BAJA | Actualizar 1 assert |
| B-E12 | US-075 MLflow en fallback file (server :5010 caido) + checkpoint/subset solo en VM | BAJA | re-registrar run en :5010; dvc pull o dejar en VM. (US-076 DVC YA resuelto, commit 4714106) |

## Como se valido cada capa

- **ML (E6, E12)**: notebooks con outputs reales + CSVs de metricas + OOF fold-5.
- **Agente (E7)**: tests + eval real en H100 (perceiver champion vs baseline).
- **Backend (E8)**: 146 tests incluidos RLS con Postgres real (testcontainers).
- **Frontend (E9)**: vitest + typecheck + i18n + lint + build (E2E pendiente .env).
- **Docs/Obs (E10)**: verificacion de entregables en disco + tests de metrics/drift.

## Infra usada

- VM H100 NVL 96 GB del sponsor (tunel Cloudflare; keep-alive cada 4 min).
- Postgres local :5432 + Docker para testcontainers (RLS real).
- MLflow :5010 (algunos runs en fallback file por server caido — deuda E12).
