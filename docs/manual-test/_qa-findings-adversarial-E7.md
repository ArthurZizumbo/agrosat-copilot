# Hallazgos — revision adversarial profunda E7 (US-045..049)

**Fecha:** 2026-06-15 · **Metodo:** workflow multi-agente (5 revisores + verificacion
adversarial por hallazgo; cada hallazgo se intento REFUTAR leyendo el codigo real).
**Resultado:** 24 hallazgos -> **17 confirmados**, 7 refutados, 0 inciertos.

De los 17 confirmados: **6 corregidos** en el commit `fix(E7): correctness de revision
adversarial` (con tests de regresion, 147 verdes); **11 documentados** abajo porque cambian
contrato publico / semantica de negocio / UX de streaming, o requieren decision de producto.

---

## A. Corregidos (commit de fix, con tests)

| # | Sev | US | Archivo | Fix |
|---|-----|----|---------|-----|
| 1 | **critico** | US-048 | `backends.py` `_messages_from_contents` | Mensajes `role=tool` ahora llevan `tool_call_id`; sin el, vLLM/Qwen devuelve 400 y **rompia todo turno multi-turn con herramientas** (el caso normal del agente Plan-and-React on-prem). |
| 8 | alto | US-048 | `backends.py` | id de `tool_call` ya no colapsa a `fc.name`: ids unicos por `(turno,indice)` -> dos llamadas a la misma tool en un turno no colisionan. |
| 11 | medio | US-046 | `rag.py` `_fuse_and_rank` | `cosine_distance` de pgvector es `[0,2]`; se clampa `max(0, 1-cos)` para que el score respete `[0,1]`. |
| 12 | medio | US-046 | `rag.py` `_to_pgvector_literal` | NaN/Inf -> `0.0` (pgvector los rechaza; un solo `dim_k` malo abortaba el batch). Espeja `classify.py`. |
| 16 | bajo | US-048 | `benchmark_qwen35.py` | valida `--n >= 1` (antes `--n 0` -> StatisticsError). |
| 15 | bajo | US-047 | `backends.py` `OllamaBackend` | docstring corregido: `tools` se **ignora** (no "forwarded"). |

> El #1 es el de mayor impacto: justo la ruta on-prem que se monto en US-048 estaba rota
> para cualquier consulta que dispare una tool. Ahora con test de regresion.

---

## B. Documentados (requieren decision de producto / cambio de contrato)

> No auto-corregidos para no alterar semantica de negocio o UX sin tu visto bueno. Cada uno
> esta **verificado contra el codigo real** (no es falso positivo). Prioridad sugerida arriba.

### B-1 · ALTO · US-045 · `compare_models` ignora `session_id` (multi-tenant)
`ml/agent/tools/compare.py` — `CompareModelsInput.session_id` se declara pero **nunca se usa**.
El tool lee los OOF globales por `parcel_id` sin verificar pertenencia a la sesion. Viola la
regla NON-NEGOTIABLE "toda query filtra por session_id". Es tool **deferred** (no en las 5 demo).
**Fix:** antes de leer OOF, `SELECT 1 FROM parcels WHERE id=$1 AND session_id=$2` dentro de
`session_scoped_conn`; si no existe, devolver comparacion vacia. Requiere actualizar los 3 tests
de `compare_models` (hoy pasan `parcel_id` sin setup de pertenencia).

### B-2 · ALTO · US-045 · `classify_new_parcel` ignora la geometria del AOI
`ml/agent/tools/classify.py:233-242` — `_fetch_parcel_embedding` toma el embedding de la
parcela **mas reciente** de la sesion (`ORDER BY updated_at DESC LIMIT 1`), sin relacion espacial
con `inp.aoi`. El usuario dibuja el AOI X y recibe la clase de la parcela Y, con confianza alta.
**Fix:** resolver por interseccion: `JOIN parcels p ... WHERE ST_Intersects(p.geom,
ST_SetSRID(ST_GeomFromGeoJSON($3),4326))`; si nada intersecta, devolver `_needs_gee_result()`.
Requiere actualizar el test que mockea `_fetch_parcel_embedding`.

### B-3 · ALTO · US-045 · `get_parcel_timeseries` fabrica las FECHAS
`ml/agent/tools/timeseries.py` — `_spread_dates` reparte los percentiles p05..p95 sobre fechas
equiespaciadas `[start,end]` en orden creciente, produciendo una curva NDVI ascendente cuyas
fechas no corresponden a ninguna observacion. El valor es real; la **fecha** esta inventada
(regla `ml/agent/CLAUDE.md` prohibe "fechas sin origen en un tool call"). El unico ancla honesta
es el peak (solo NDVI). **Fix:** o devolver solo anclas fenologicas reales (sog/peak/senescence
via `_doy_to_date`), o re-etiquetar el contrato de salida como resumen distribucional (p05..p95),
no `dates`/`values`. Cambia el contrato `TimeSeries` -> decision de producto.

### B-4 · ALTO · US-047 · `GeminiBackend` genera dos veces por turno de texto
`ml/agent/backends.py:226-247` — en el camino sin tool calls, `_generate` (no-stream) corre una
generacion completa para detectar function_calls, y luego `_stream_text` corre una **segunda**
generacion completa. Doble costo/latencia por cada respuesta textual, y muestreo inconsistente
(el texto mostrado puede diferir del evaluado). **Fix:** emitir el texto del `response` ya obtenido
via `_chunks_from_response(response)` en vez de re-llamar `_stream_text`. Tradeoff: pierde el
streaming token-a-token (pasa a texto en bloque) -> decision de UX.

### B-5 · ALTO · US-049 · parser AgroMind capado a `[A-D]`
`ml/eval/agent_metrics.py:102-107` — las regex de letra solo matchean `[A-D]`, pero 47 golds son
`E-I` y hay items con >4 opciones (hasta J). Una respuesta correcta `F` envuelta en prosa
("The answer is F") puntua 0 (la bare `F` se rescata por el fallback de texto). **Fix:** derivar
las letras validas de `item.options.keys()` tanto en el prompt como en el parser.

### B-6 · CRITICO(eval) · US-049 · prompt AgroMind hardcodea "A, B, C o D"
`ml/eval/agent_bench.py:416-437` — el prompt dice "elige UNA sola letra (A,B,C,D)" para los 500
items, pero ~245/494 tienen gold fuera de A-D (E-I, numerico, texto libre). Para los ~174 items
sin opciones (numerico/texto), el prompt exige una letra inexistente -> deprime la metrica de esa
mitad. **Importante:** la corrida en curso usa el prompt viejo; al interpretar los numeros de
AgroMind, contar con que la mitad no-A-D esta sesgada a la baja. **Fix:** prompt adaptativo (si
`options` vacio -> pedir respuesta directa; si >4 labels -> listar el set real) + scoring por
`f1_squad`/normalizado para items abiertos. (Verificador degrado la severidad del original
"critico" porque `exact_match` SI cae al match normalizado, no es "imposible de puntuar".)

### B-7 · CRITICO(eval) · US-049 · score Qwen AgroMind sobre 6 items numericos mal formateados
`ml/eval/agent_bench.py` — Qwen (text-only) solo evalua los 6 items textuales del subset, que
resultan ser todos `options:[]` con gold numerico ('0','10','2',...). El prompt les pide una letra
A-D. La celda de rubrica `(qwen, AgroMind, exact_match) >= 0.70` se mide sobre esos 6 -> n
demasiado chico y mal formateado. **Fix:** ya documentado en handoff US-049 (ampliar subset
textual con `is_multimodal=False`); ademas exponer `n_evaluated` prominente en el reporte para que
la celda no se lea como score comparable.

### B-8 · MEDIO · US-045 · `_spread_dates` colapsa con ventanas cortas
`ml/agent/tools/timeseries.py:116-132` — con ventana de 0-3 dias, `round(step*i)` genera fechas
duplicadas que la dedup por dict colapsa: 5 percentiles -> 1-3 puntos, perdiendo datos en silencio
(`start==end` -> 1 punto = p95). **Fix:** detectar colision y fallar/avisar o forzar paso minimo
de 1 dia. (Se resuelve naturalmente si se adopta B-3.)

### B-9 · MEDIO · US-047 · texto intercalado con tool calls se descarta
`ml/agent/agent.py:236-261` — si un turno emite texto Y function calls, el texto se bufferiza pero
solo se emite cuando `not tool_calls`; el `Content` reconstruido omite el texto. El razonamiento
del modelo desaparece del SSE y del historial -> degrada coherencia multiturno (Gemini y Qwen
emiten texto+tool_call juntos a menudo). **Fix:** emitir `text_parts` como `TextDeltaEvent` aunque
haya tool calls, e incluir las text parts en `_model_function_call_content`. Cambia el orden de
eventos SSE -> validar con `chat_service`/frontend.

### B-10 · BAJO · US-049 · splitter workflow pierde texto tras el bloque de codigo
`ml/eval/agent_bench.py:690-718` — `_split_workflow_and_code` toma `answer[:first_fence]`; si el
modelo pone el workflow DESPUES del codigo, se descarta -> sim 0 y falso fallo de pass-rate.
**Fix:** concatenar pre-fence + post-fence (sin el bloque de codigo) para el workflow.

---

## Refutados (7) — NO son bugs

El verificador refuto 7 hallazgos (falsos positivos: el codigo ya manejaba el caso, o el revisor
malinterpreto). No se listan por brevedad; ninguno requiere accion.

---

## Resumen accionable

- **Ya aplicado:** los 6 fixes seguros (incl. el critico de Qwen tool_call_id).
- **Decidir y aplicar (alto):** B-1 (multi-tenant compare_models), B-2 (classify AOI), B-3
  (timeseries fechas), B-4 (Gemini doble generacion).
- **Para la eval (alto):** B-5/B-6/B-7 afectan como se leen los numeros de AgroMind de la corrida
  en curso; conviene re-correr AgroMind con el prompt/parser adaptativos antes del entregable final.
- **Mejora menor:** B-8, B-9, B-10.
