# Hallazgos — revision adversarial profunda E7 (US-045..049)

**Fecha:** 2026-06-15 · **Metodo:** workflow multi-agente (5 revisores + verificacion
adversarial por hallazgo; cada hallazgo se intento REFUTAR leyendo el codigo real).
**Resultado:** 24 hallazgos -> **17 confirmados**, 7 refutados, 0 inciertos.

> **Estado a 2026-06-16: 0 hallazgos pendientes.** Los 17 confirmados estan
> CORREGIDOS y verificados contra el codigo actual. Este archivo conserva solo el
> changelog de lo resuelto (los corregidos se dejan como registro de auditoria, no
> como tareas). Si una revision futura abre hallazgos nuevos, se anaden en una
> seccion "Pendientes" al inicio.

---

## Pendientes

Ninguno. Todos los hallazgos confirmados quedaron corregidos (detalle abajo).

---

## Resueltos (changelog de auditoria)

### 1a pasada — 6 fixes seguros con tests (commit `fix(E7): correctness de revision adversarial`)

| # | Sev | US | Archivo | Fix | Verificado |
|---|-----|----|---------|-----|------------|
| 1 | **critico** | US-048 | `backends.py` `_messages_from_contents` | Mensajes `role=tool` ahora llevan `tool_call_id` (sin el, vLLM/Qwen 400 rompia TODO turno multi-turn con tool). | ids unicos por `(turno,indice)`, test de regresion |
| 8 | alto | US-048 | `backends.py` | id de `tool_call` ya no colapsa a `fc.name`: ids unicos por `(turno,indice)`. | dos tools en un turno no colisionan |
| 11 | medio | US-046 | `rag.py` `_fuse_and_rank` | `cosine_distance` pgvector es `[0,2]`; se clampa `max(0, 1-cos)` a `[0,1]`. | |
| 12 | medio | US-046 | `rag.py` `_to_pgvector_literal` | NaN/Inf -> `0.0` (pgvector los rechaza). Espeja `classify.py`. | |
| 16 | bajo | US-048 | `benchmark_qwen35.py` | valida `--n >= 1` (antes `--n 0` -> StatisticsError). | |
| 15 | bajo | US-047 | `backends.py` `OllamaBackend` | docstring corregido: `tools` se **ignora** (no "forwarded"). | |

### 2a pasada — B-1..B-10 (workflow `fix-adversarial-findings-e7` + integracion, 177 tests)

| # | Sev | US | Archivo | Fix aplicado | Verificado en codigo |
|---|-----|----|---------|--------------|----------------------|
| B-1 | alto | US-045 | `compare.py` | `compare_models` filtra por `session_id`: gate `_parcel_belongs_to_session` (`SELECT 1 FROM parcels WHERE id=$1 AND session_id=$2`) antes de leer OOF. | `_parcel_belongs_to_session` presente, llamado en `run` |
| B-2 | alto | US-045 | `classify.py` | `classify_new_parcel` resuelve el embedding por interseccion con el AOI (`ST_Intersects(p.geom, ST_SetSRID(ST_GeomFromGeoJSON($3),4326))`); sin interseccion -> `_needs_gee_result()`. | `ST_Intersects`/`ST_GeomFromGeoJSON` presentes |
| B-3 | alto | US-045 | `timeseries.py` | `get_parcel_timeseries` ya NO fabrica fechas: `_spread_dates` eliminado; solo anclas fenologicas reales (peak con DOY real). | 0 ocurrencias de `_spread_dates` |
| B-4 | alto | US-047 | `backends.py` | `GeminiBackend` genera UNA sola vez por turno de texto (re-emite el `response` ya obtenido via `_chunks_from_response`), sin doble generacion. | single-generation en `generate_stream` |
| B-5 | alto | US-049 | `agent_metrics.py` | parser AgroMind acepta el set REAL de letras del item (`item.options.keys()`), no capado a `[A-D]`. | `valid_letters` derivado de `item.options` |
| B-6 | critico(eval) | US-049 | `agent_bench.py` | prompt AgroMind **adaptativo** por `answer_type` (no hardcodea "A,B,C,D"); items abiertos piden respuesta directa. | `_build_agromind_prompt` ramificado |
| B-7 | critico(eval) | US-049 | `agent_bench.py` | **Resuelto de forma definitiva**: un escaneo del corpus completo (28482 items) confirma que AgroMind es **100% visual** (0 items textuales). Ampliar el subset textual de Qwen era imposible por diseno; en su lugar Qwen-texto se reporta **N/A** (no 0.0) cuando `n_skipped>0 and n_evaluated<_MIN_AGROMIND_N`, y la comparacion multimodal on-prem la hace la variante `qwen36-vl` (Qwen3.6-35B-A3B por llama.cpp+mmproj). | `_MIN_AGROMIND_N` + variante `qwen36-vl` |
| B-8 | medio | US-045 | `timeseries.py` | colapso de ventanas cortas resuelto al adoptar B-3 (sin `_spread_dates`). | cubierto por B-3 |
| B-9 | medio | US-047 | `agent.py` | texto intercalado con tool calls ya NO se descarta: se emite `TextDeltaEvent` aunque haya tool calls, e incluido en el `Content` reconstruido. | `TextDeltaEvent` emitido en el camino con tools |
| B-10 | bajo | US-049 | `agent_bench.py` | `_split_workflow_and_code` concatena pre-fence + post-fence (no descarta el workflow tras el bloque de codigo). | `post` (post-fence) concatenado al workflow |

> Cross-file regression cazada en la 2a pasada: B-2 cambio la firma de
> `_fetch_parcel_embedding` -> rompia `perceiver`; fix = `aoi` opcional
> (`ST_Intersects` con aoi / most-recent sin aoi). LECCION: los agentes de workflow
> editan en aislamiento; SIEMPRE correr la suite completa de integracion tras un workflow.

### Harness de eval blindado (mismo ciclo)

Tras 2 CUELGUES REALES por falta de timeout (1o tunel cloudflared muerto en fase Qwen,
2o socket Gemini): genai `HttpOptions(120s)` + `AsyncOpenAI(180s)` + `asyncio.wait_for`
por item -> el except por-item existente la salta; **checkpoint por variante (Gemini
primero) + `--resume`** -> Gemini se calcula UNA vez y se guarda, nunca se re-paga; log
por item + `PYTHONUNBUFFERED`.

---

## Refutados (7) — NO son bugs

El verificador refuto 7 hallazgos (falsos positivos: el codigo ya manejaba el caso, o el
revisor malinterpreto). No requieren accion.
