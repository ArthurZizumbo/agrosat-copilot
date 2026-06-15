# ADR-011 — Arquitectura del sistema conversacional (patrón Be My Eyes, transporte desacoplado)

- **Estado:** Aceptado (MVP 27-jun). Implementación en curso.
- **Fecha:** 2026-06-15
- **Contexto previo:** [ADR-008](ADR-008-rediseno-calendario-presentacion-27jun.md) (calendario), [ADR-009](ADR-009-h100-reactivacion-pivote-farslip-alcance-v8.md) (alcance v8). EPIC 7/8/9 del [plan v8](../../context/RefinamientoPlaneacionAgroSatCopilot_v8.md).
- **Scope:** `ml/agent/`, `backend/`, `frontend/`.

## Decisión

### 1. Patrón multi-agente: Be My Eyes asimétrico e iterativo

Dos agentes con **delegación asimétrica de percepción** (no debate simétrico):

- **Orquestador ("cerebro")** — Gemini 2.5 Pro (variante-A) o Qwen3.5-35B-A3B vLLM (variante-B). Conversa, planifica (Plan-and-React), no "ve" el raster. Decide qué preguntar.
- **Agente Visión ("ojos")** — corre los modelos sobre la escena y devuelve `Finding[]` con citas. El orquestador lo invoca como tool, **iterativamente** (puede repreguntar: NDVI de otra parcela, comparar tendencia). Esto mejora el single-shot de Be My Eyes original.

**Ojos híbridos (ML ahora, VLM después):** el Agente Visión expone una interfaz estable `analyze(aoi, question) -> Finding[]`. Sub-tools del MVP = stack ML entrenado (`classify_parcel` XGBoost+AlphaEarth, `compute_ndvi`, futuro `segment_scene`). Post-presentación se enchufa `describe_scene_vlm` (Gemma/Qwen-VL) **sin tocar el orquestador**.

### 2. Sin Google ADK

`google-adk` está fuera del lock (choca con `genai` 2.x). Se usa el SDK `google-genai` directo con function calling nativo. Diseño modular para portar a ADK/Agent Engine post-presentación. La variante-B (Qwen3.5 vLLM) se consume vía API OpenAI-compatible.

### 3. Transporte desacoplado: dispatch + WebSocket (no llamada síncrona bloqueante)

El front **no** espera bloqueado. Flujo:

```
POST /chat {session_id, message, llm_variant?}  ->  202 {job_id, ws_url}   (responde al instante)
        │
        ▼  background task in-process (asyncio.create_task)
   ChatService.run(job_id) consume run_chat(...) -> AsyncIterator[AgentEvent]
        │  cada evento -> JobRegistry[job_id].publish(event)
        ▼
WS /ws/chat/{session_id}?job_id=...   <-  stream JSON de AgentEvent (+ backlog en reconexión)
```

- **Background task in-process** sustituye a la Cloud Function / Pub/Sub (OUT en v8 por tiempo). Mismo desacople, sin infra de cola. Reemplazable por Pub/Sub post-presentación sin tocar el front.
- **`JobRegistry`** (en memoria, por proceso): `job_id -> asyncio.Queue + buffer de eventos + estado`. El buffer permite que un WS que se conecta tarde reciba el backlog. Suficiente para demo single-instance.
- **Transport-agnostic:** el core emite `AgentEvent`; encima va el adaptador WS (preferente) y un adaptador SSE (`GET /chat/{job_id}/events`) como fallback trivial. Nada del core depende del transporte.

### 4. Hexagonal: el agente no importa el backend

`ml/agent/` depende solo de Protocols (`ml/agent/ports.py`): `ParcelReader`, `ChatMemory`. El backend implementa adaptadores SQLModel y los inyecta vía `AgentDeps`. Así el agente se testea con fakes y no arrastra asyncpg/ORM.

## Contratos (fuente de verdad para las 3 capas)

### Esquema de eventos — `ml/agent/events.py`

Unión discriminada por `type`: `plan_created` · `tool_call` · `tool_result` · `token` (opcional) · `final_answer` · `error` · `done`. Toda cifra en `final_answer.citations` enlaza a un `tool_call_id`.

Ejemplo de stream (wire JSON, una línea por evento):

```json
{"type":"plan_created","steps":["Listar parcelas del AOI","Clasificar cultivo","Calcular NDVI","Sintetizar"]}
{"type":"tool_call","call_id":"c1","tool":"classify_parcel","args":{"aoi_id":1},"agent":"vision"}
{"type":"tool_result","call_id":"c1","tool":"classify_parcel","ok":true,"summary":"3 parcelas clasificadas","duration_ms":142,"findings":[{"parcel_id":10,"crop_class":"Meadow","confidence":0.91,"area_ha":4.2,"citation":{"tool_call_id":"c1","source":"XGBoost+AlphaEarth","parcel_id":10}}]}
{"type":"final_answer","text":"El AOI tiene 3 parcelas: predomina pradera (4.2 ha, conf. 0.91)...","citations":[{"tool_call_id":"c1","source":"XGBoost+AlphaEarth","parcel_id":10}]}
{"type":"done","job_id":"job_abc"}
```

### Entry-point del agente — `ml/agent/orchestrator.py`

```python
async def run_chat(
    *,
    session_id: str,
    user_message: str,
    llm_variant: Literal["gemini", "qwen35"],
    deps: AgentDeps,
) -> AsyncIterator[AgentEvent]: ...
```

El backend NO construye prompts ni llama al LLM: solo invoca `run_chat` y reenvía eventos.

### Backend — endpoints MVP

| Método | Ruta | Propósito |
|--------|------|-----------|
| POST | `/sessions` | Crear sesión (devuelve `session_id`, `user_id` demo). |
| POST | `/chat` | Despachar job; `202 {job_id, ws_url}`. |
| WS | `/ws/chat/{session_id}` | Stream de `AgentEvent` (param `job_id` opcional). |
| GET | `/chat/{job_id}/events` | Fallback SSE del mismo stream. |
| GET/POST | `/aois` | Listar/crear AOIs de la sesión (GeoJSON). |
| POST | `/llm/switch` | Cambiar `llm_variant` de la sesión (gemini/qwen35). |

Todo endpoint filtra por `session_id` (`_check_session_owner`). RLS por sesión se endurece en US-051 (fase posterior).

### Frontend — contrato

`useChat` (composable): `POST /chat` -> abre WS `/ws/chat/{session_id}` -> reduce eventos a estado del `chatStore` (Pinia). `ChatPanel.vue` renderiza plan/tool_calls/final_answer; `MapView.vue` (MapLibre) pinta parcelas de los `Finding`. Textos vía `t('key')` en `it/es/en` simultáneo.

## Alternativas descartadas

- **Llamada síncrona bloqueante back->agente:** mala UX en consultas con varias tool calls.
- **Pub/Sub + Cloud Function real:** correcto a futuro pero OUT en v8 (sin tiempo al 27-jun); el background task in-process da el mismo desacople para la demo.
- **Dual-loop simétrico (dos LLM dialogando):** es paradigma de *debate*, no Be My Eyes; más caro en tokens/latencia/build. Diferido.

## Consecuencias

- (+) Front libre tras `POST /chat`; UX fluida con eventos en vivo.
- (+) Core del agente testeable sin DB ni red; portable a ADK.
- (+) Ojo VLM se añade sin reescribir el orquestador.
- (−) `JobRegistry` en memoria no sobrevive a reinicios ni escala multi-instancia (aceptable para demo; Pub/Sub lo resuelve después).
