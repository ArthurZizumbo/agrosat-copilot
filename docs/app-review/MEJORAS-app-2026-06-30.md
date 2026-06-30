# AgroSatCopilot — Revisión de la aplicación y mejoras propuestas

**Fecha**: 2026-06-30 · **Alcance**: backend (FastAPI), agente conversacional (`ml/agent`), frontend (Nuxt 4 SSR) · **Tipo**: auditoría de solo lectura, sin cambios de código.

> Esta es **solo la lista de mejoras** (para validación de Arthur antes de aplicar). Cada
> ítem indica qué, por qué, archivos afectados y esfuerzo estimado (S = horas, M = 1-2 días,
> L = varios días). Hallazgo transversal: **la app está más completa de lo que sus propias
> guías `CLAUDE.md`/`AGENTS.md` afirman** — varias dicen "ESQUELETO/SKELETON" sobre código
> ya implementado. Esta revisión refleja el código real verificado, no las guías.

---

## 0. Resumen ejecutivo

La aplicación es funcional de extremo a extremo: backend con RLS multi-tenant real por
`session_id`, tiling COG con TiTiler + cache Redis, 10 FunctionTools geoespaciales reales,
chat SSE con retry/backoff, e i18n trilingüe (it/es/en) sincronizada (91 claves idénticas).
El código es **honesto**: pgstac, NDWI/NDMI y Pub/Sub degradan explícitamente en vez de
fabricar datos.

Los gaps de mayor impacto:

1. **El switch A/B de LLM no llega al backend** pese a que el endpoint `/llm/switch` existe y es real.
2. **Documentación que llama "Google ADK" a un loop de function-calling custom** y marca como "esqueleto" código ya implementado.
3. **`/readyz` es un stub** que siempre devuelve 200.
4. **`@ai-sdk/vue` instalado sin un solo import** (dependencia muerta).

---

## 1. Inventario factual (qué existe y funciona)

### 1.1 Backend FastAPI (`backend/app/`)

App factory en `backend/app/main.py`: routers montados, CORS endurecido, rate-limit por
sesión (slowapi), middleware Prometheus, manejadores de error TiTiler. Config tipada en
`backend/app/core/config.py` (`extra="forbid"`, rechaza defaults de dev en staging/prod).

| Método | Ruta | Archivo | Estado |
|---|---|---|---|
| GET | `/healthz` | `api/health.py` | Real |
| GET | `/readyz` | `api/health.py` | **Stub** (200 fijo, no chequea Postgres/Redis) |
| GET | `/metrics` | `api/metrics.py` | Real (Prometheus) |
| POST | `/chat` (SSE) | `api/chat.py` | Real, rate-limit 10/min/sesión |
| POST/GET/PATCH/DELETE | `/sessions`, `/sessions/{id}/messages` | `api/sessions.py` | Real (RLS, transcript server-side US-080) |
| POST | `/llm/switch` | `api/llm.py` | Real, rate-limit 5/min, UPDATE RLS |
| POST/GET/DELETE | `/aois`, `/aois/{id}` | `api/aois.py` | Real (PostGIS, área geodésica) |
| GET | `/aois/{id}/timeseries` | `api/timeseries.py` | Real pero limitado (NDVI 1 punto; NDWI/NDMI serie vacía honesta) |
| GET | `/stac/search` | `api/stac.py` | Degradación honesta (pgstac no desplegado → colección vacía) |
| GET | `/tiles/{z}/{x}/{y}.png`, `/cog/**` | `api/tiles.py`, `services/cog_tiler.py` | Real (rio-tiler, cache Redis 15 min, guard SSRF) |

**Capa de servicios**: `chat_service.py` integra de verdad con el agente (`ml/agent`):
resuelve variante LLM desde DB, corre perceiver → grounding → reasoner, reenvía eventos como
SSE. `jobs_service.py` corre modelos CPU-light inline; el modo Pub/Sub lanza
`NotImplementedError` explícito (US-056 diferido).

**Auth / multi-tenant**: RLS por `session_id` real y *enforced* (rol `agrosat_app`
NOBYPASSRLS, `set_config('app.current_session', ...)` por request, binds sin inyección, guard
`verify_session()` fail-closed → 403). **Clerk/JWT no implementado**: sesiones anónimas vía
header `X-User-ID`; la aislación la da RLS, no la autenticación.

**Tests**: 19 archivos, ~144 funciones, objetivo `--cov-fail-under=70`. Integración con
testcontainers PostGIS (RLS isolation, aois, timeseries, chat, tiles/STAC).

### 1.2 Agente conversacional (`ml/agent/`)

**Hallazgo factual**: **NO usa el SDK de Google ADK** (`google-adk` fue removido del lock por
choque con `google-genai` 2.x). Es un **loop manual de function-calling** en `agent.py` sobre
`google-genai`, con `MAX_TURNS=8`, inyección de `session_id` (no confía en el modelo),
validación Pydantic estricta y stream de eventos tipados.

10 FunctionTools, todas reales (golpean DB/PostGIS/modelos): `list_parcels`,
`get_parcel_timeseries`, `get_aoi_stats`, `search_stac`, `get_tiles`, `classify_new_parcel`
(sirve el campeón Voting-3 france-10 F1 0.9069 desde OOF cacheado), `add_aoi`,
`compare_models`, `explain_prediction` (descriptor fenológico Wen et al. 2025),
`retrieve_context` (Spatial-RAG lite, gateado por `RAG_ENABLED`).

**Backends LLM** (`backends.py` + `llm_routing.py`): `GeminiBackend`, `VLLMOpenAIBackend`
(Qwen on-prem), `OllamaBackend` (Gemma/Qwen-VL). 4 variantes con fallback availability-aware
(probe TCP 2 s, degrada a gemini).

**Spatial-RAG** (`rag.py`): real nivel "lite" — `ST_DWithin` + pgvector cosine + fusión
ponderada. e5-mistral/HNSW deliberadamente diferido.

### 1.3 Frontend Nuxt 4 (`frontend/`)

Componentes reales: header/sidebar/statusbar; chat (`ChatDock`, `MessageBubble` con
marked+dompurify, `ToolActivity`, `Composer`, `ChatTabs`, `LlmSwitch`, `PlanStepper`,
`FindingCard`); mapa (`MapCanvas`, `DrawToolbar`, `BasemapSwitcher`, `CropLegend`).

Composables: `useChat.ts` (SSE sobre POST con parser propio + retry/backoff), `useMap.ts`,
`useAoi.ts`, `useSession(s).ts`. Stores Pinia: `chat` (persistido), `map`, `sessions`.

**Mapa**: MapLibre GL real (`maplibre-gl@5.24`). **deck.gl removido** (US-058, confirmado sin uso).
**i18n**: las 3 locales sincronizadas (91 leaf-keys idénticas). 18 componentes usan `t()`.
**Tests**: 7 unit vitest + 3 specs Playwright E2E (requieren backend + sesión sembrada).

---

## 2. Mejoras propuestas (priorizadas)

### Prioridad ALTA

| # | Mejora | Por qué | Archivos | Esfuerzo |
|---|---|---|---|---|
| 1 | **Cablear el switch A/B de LLM al backend**: `useChat.switchLlm()` solo setea un flag local; debe hacer `POST /llm/switch` (ya existe y es real) y habilitar el control en `ChatDock` (hoy `server-fixed`). | Es objetivo de rúbrica; backend listo, frontend lo ignora → el usuario cree que cambia de modelo y no pasa nada. | `frontend/composables/useChat.ts:390`, `frontend/components/chat/ChatDock.vue:104`, `LlmSwitch.vue` | S |
| 2 | **Corregir nomenclatura "Google ADK" en docs**: el código es un loop custom sobre `google-genai`, no el SDK ADK. | Discrepancia factual que confunde a auditores/rúbrica. | `CLAUDE.md`, `AGENTS.md`, `ml/agent/CLAUDE.md` | S |
| 3 | **Sincronizar guías de carpeta obsoletas**: `backend/CLAUDE.md` dice "SKELETON" y `ml/agent/CLAUDE.md` "ESQUELETO PURO", ambos falsos. | Las guías dirigen al orquestador y mienten sobre el estado real. | `backend/{CLAUDE,AGENTS}.md`, `ml/agent/{CLAUDE,AGENTS}.md` | S |
| 4 | **Implementar `/readyz` real**: chequear Postgres (`SELECT 1`) + Redis antes de 200. | Es la readiness probe de Cloud Run; un stub enruta tráfico a instancias sin DB. | `backend/app/api/health.py` | S |

### Prioridad MEDIA

| # | Mejora | Por qué | Archivos | Esfuerzo |
|---|---|---|---|---|
| 5 | **Decidir sobre `@ai-sdk/vue`** (dep muerta): adoptarla para el streaming o removerla. | Instalada sin un solo import; infla bundle/lockfile y contradice la doc de stack. | `frontend/package.json` | S/M |
| 6 | **Verificar `parcel_id`/`locale` end-to-end**: `useChat` los envía pero el backend con `extra="forbid"` daría 422 si no están declarados en `ChatRequest`. | Riesgo de 422 silencioso o feature i18n/parcel-link a medias. | `backend/app/api/chat.py`, `frontend/composables/useChat.ts:198` | S |
| 7 | **Desplegar pgstac o documentar el gap**: `/stac/search` degrada a vacío porque `CREATE EXTENSION pgstac` está comentado. | Capacidad anunciada que hoy no devuelve nada. | `db/migrations/*`, `services/stac_service.py` | M |
| 8 | **Completar timeseries NDWI/NDMI**: hoy NDVI = 1 punto y NDWI/NDMI vacíos. | Honesto pero pobre para la UX de gráfica temporal. | `services/timeseries_service.py`, `ml/agent/tools/timeseries.py` | M-L |
| 9 | **Tests E2E Playwright en CI**: añadir fixture + seed para correr los 3 specs automáticamente. | El flujo /chat SSE y el switch i18n no tienen verificación E2E automatizada. | `frontend/tests/e2e/*`, CI workflow | M |

### Prioridad BAJA

| # | Mejora | Por qué | Archivos | Esfuerzo |
|---|---|---|---|---|
| 10 | **Auth Clerk/JWT**: hoy multi-tenant es anónimo por `X-User-ID`; RLS aísla por `session_id` pero un cliente puede enviar otro `X-User-ID` y listar sesiones ajenas vía la función `SECURITY DEFINER`. | Aceptable para demo, pero es un hueco de autorización real (no solo de aislación). | `api/sessions.py`, `deps.py`, `core/config.py` | L |
| 11 | **Cablear worker Pub/Sub para `/jobs` async**: el modo Pub/Sub lanza `NotImplementedError` y el worker no está. | Solo se corren modelos CPU-light inline; los GPU pesados no tienen ruta async. | `services/jobs_service.py`, `ml/workers/` | L |
| 12 | **Eliminar `tailwind.config.ts` legacy**: la verdad del tema es `@theme` en `main.css`. | Reduce confusión. | `frontend/tailwind.config.ts` | S |
| 13 | **Spatial-RAG full** (post-MVP): subir de lite (AlphaEarth 64-dim + ST_DWithin) a e5-mistral 4096-dim + HNSW + reranking. | La versión lite no escala y el embedding de query es aproximado. | `ml/agent/rag.py` | L |

---

## 3. Notas de honestidad (para defender en presentación)

El código degrada explícitamente en lugar de fabricar datos, lo cual es **defendible y
valioso** ante un evaluador:

- `timeseries` devuelve solo las anclas fenológicas reales que existen, no una serie inventada.
- `/stac/search` devuelve `FeatureCollection` vacía cuando pgstac no está desplegado, no resultados falsos.
- `jobs_service` modo Pub/Sub lanza `NotImplementedError` en vez de simular un job.
- El agente inyecta `session_id` server-side (no confía en el modelo) y valida cada argumento de tool con Pydantic.

Esta disciplina es coherente con la regla del proyecto: **datos reales, cero placeholders en producción**.
