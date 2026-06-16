# Reporte QA — EPIC 7 (US-045 a US-050), agente conversacional

**Fecha:** 2026-06-15 · **Rama:** `feature/E7-agente-conversacional` · **Alcance:** US-045..050.
**Metodo:** handoff -> diff por commit -> `make check` -> tests+cobertura -> agrosat-security-audit +
agrosat-code-review + agrosat-google-adk-agent -> criterios de aceptacion vs codigo real.

> Manual-test por US: [045](us-045.md) · [046](us-046.md) · [047](us-047.md) · [048](us-048.md) ·
> [049](us-049.md) · [050](us-050.md).

---

## 1. Gate de calidad (transversal)

| Gate | Resultado | Nota |
|------|-----------|------|
| `ruff check` backend + ml | [x] PASS | All checks passed |
| `ruff format --check` | [x] PASS **tras fix** | 2 archivos reformateados (ver Issue #1) |
| `i18n-check` (it/es/en) | [x] PASS | parity OK; las US E7 no tocan locales |
| Hardcoded secrets | [x] CLEAN | grep skill agrosat-security-audit -> 0 (gitleaks no instalado local; ver Issue #2) |
| SQL injection | [x] CLEAN | 100% parametrizado `$1..$n`; `set_config(...,$1,true)`; 0 f-string SQL |
| RLS por `session_id` | [x] OK | `WHERE session_id=$1` en cada tool DB + `SET LOCAL` en `db.py` |
| `print()` en codigo | [x] 0 | structlog en todo `ml/agent` |
| `pnpm lint` (frontend) | [ ] FAIL **pre-existente** | ESLint v9 sin `eslint.config.js`; NO de estas US (ver Issue #3) |

**Tests:** 144 verdes — `tests/ml/agent/` 98 + `tests/ml/eval/{agent_bench,agent_metrics}` 46.

| Modulo | Cobertura | Modulo | Cobertura |
|--------|-----------|--------|-----------|
| ml.agent (paquete) | ~86% | ml.eval.agent_metrics | 97% |
| agent.py | 97% | ml.eval.agent_report | 97% |
| schemas.py | 99% | ml.eval.agent_bench | 77% |
| perceiver.py / rag.py | 100% / 92% | backends.py | 67% (*) |
| tools (parcels/aoi/add/compare/tiles/retrieve) | 100% | classify.py | 39% (*) |

(*) Paths que requieren LLM/GPU/DB reales (no unit-testeables); logica de borde si cubierta.

---

## 2. Criterios de aceptacion vs estado (resumen)

| US | Criterios cumplidos | Pendiente |
|----|---------------------|-----------|
| US-045 9 FunctionTools | [x] 6/6 (schemas, FunctionDeclaration, RLS, 5+4 split, tests, auto-doc) | flujos DB real = manual |
| US-046 perceiver + RAG lite | [x] 8/8 (Be My Eyes, SSE, flag rag_enabled, migracion, ingest real) | pgvector vivo + SSE demo = manual |
| US-047 agente factory | [x] 6/6 (create_agent, prompt, stream, SDK local, switch, tests) | streaming real + /llm/switch = manual |
| US-048 Qwen on-prem | [x] 7/7 (serving llama.cpp REAL, OpenAI-compat, latencia, doc) | re-arranque H100 = manual/GPU |
| US-049 eval AgroMind+Geo | [x] harness 5/5; corrida real **EN CURSO** | tabla rubrica + MLflow run al cerrar |
| US-050 Gemma LoRA FUTURE | [x] 4/4 (ADR-011, skill, espejos) | revision humana de hechos HF/PEFT |

---

## 3. Issues / hallazgos

| # | Sev | US | Hallazgo | Estado |
|---|-----|----|---------| ------|
| 1 | media | 046/047 | `chat_service.py` + `test_chat_sse.py` fallaban `ruff format --check` -> rompian `make check` | **CORREGIDO** (`ruff format`) |
| 2 | info | transversal | `gitleaks` no instalado en la maquina local -> secrets-scan via grep manual del skill (clean) | mitigado |
| 3 | media | ninguna (repo) | `pnpm lint` falla: ESLint v9 exige `eslint.config.js` (config legacy sin migrar) | **pre-existente**, fuera de E7; abrir item frontend |
| 4 | baja | 049 | subset AgroMind 494/500 multimodal -> Qwen (text-only) evalua solo 6; ampliar `make_subset(is_multimodal=False)` | documentado (TODO) |
| 5 | baja | 045/047 | `classify.py` 39% / `backends.py` 67% cobertura (paths LLM/GPU/DB reales) | caveat honesto, no bloqueante |

Sin hallazgos criticos/altos. Ningun secreto, ninguna inyeccion SQL, ningun fallo de aislamiento.

---

## 4. Mejoras de excelencia (por US, post-presentacion)

- **US-045:** subir cobertura de `classify.py` con un fixture de XGBoost ligero + DB en memoria (factory de embedding fake) para cubrir el path posterior 18-clase sin GPU. Hacer configurables los umbrales de vigor de `explain.py` (hoy hardcodeados 0.7/0.4).
- **US-046:** anadir un test de integracion real contra Postgres+pgvector (docker-compose efimero) para `spatial_rag`; hoy todo es mock. Medir el A/B ±RAG de hallucination dentro de US-049 y publicarlo.
- **US-047:** subir `backends.py` >70% con un doble de streaming que ejercite el path `generate_content`+stream de Gemini y el reasoning-fallback de Ollama. Hacer `MAX_TURNS` configurable via Settings.
- **US-048:** automatizar el re-arranque + smoke + benchmark en un solo script idempotente que registre latencias en MLflow sin intervencion manual; desbloquear vLLM si el sponsor habilita nested-virt (mejor throughput/batching que llama.cpp).
- **US-049:** ampliar el subset textual de AgroMind para que Qwen evalue >6 items; correr 3 seeds (no 1) para error bars reales una vez el budget lo permita; cachear respuestas del juez para abaratar la hallucination_rate.
- **US-050:** cuando se reactive, validar `target_parameters` con una corrida LoRA minima de humo sobre `gemma-4-26B-A4B-it` y medir el overhead por-experto antes de comprometer SFT.

---

## 5. Archivos auditados

**US-045:** `ml/agent/{schemas,db,context}.py`, `ml/agent/tools/{__init__,parcels,timeseries,aoi_stats,classify,explain,compare,stac,tiles,add_aoi}.py`, `scripts/gen_tools_doc.py`, `tests/ml/agent/{test_schemas,test_db_tools,test_ml_tools,test_function_declarations,test_rls_isolation,test_deferred_tools}.py`.
**US-046:** `ml/agent/{perceiver,rag}.py`, `ml/agent/tools/retrieve_context.py`, `db/migrations/20260615082041_create_rag_documents.sql`, `scripts/ingest_rag_documents.py`, `backend/app/{api/chat,services/chat_service,core/config}.py`, `tests/ml/agent/{test_perceiver,test_rag,test_retrieve_context}.py`.
**US-047:** `ml/agent/{agent,backends,prompts,events}.py`, `tests/ml/agent/{test_agent,test_backends}.py`, `backend/tests/unit/test_chat_sse.py`.
**US-048:** `ml/agent/backends.py` (VLLMOpenAIBackend), `scripts/{serve_qwen35.sh,download_qwen35.py,benchmark_qwen35.py,serve_qwen_llamacpp.bat,download_qwen_gguf.py,setup_llamacpp_vm.ps1}`, `docs/serving/qwen35.md`, `tests/ml/agent/test_qwen_benchmark.py`.
**US-049:** `ml/eval/{agent_bench,agent_metrics,agent_report}.py`, `scripts/{run_us049_eval,download_agromind_images}.py`, `tests/ml/eval/{test_agent_bench,test_agent_metrics}.py`.
**US-050:** `docs/decisions/ADR-011-gemma4-lora-future.md`, `.claude/skills/agrosat-llm-finetuning/SKILL.md`, `CLAUDE.md`, `AGENTS.md`.
