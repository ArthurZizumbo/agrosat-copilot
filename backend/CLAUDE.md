# Backend Sub-Agent — AgroSatCopilot

> Sobreescribe al orquestador root cuando haya conflicto en contexto backend.

**Rol**: API REST + Tiling COG + Workers asíncronos. Integra el agente Google ADK del módulo `ml/agent/` vía service layer.

## Skills References

- [agrosat-backend-api](../.claude/skills/agrosat-backend-api/SKILL.md) — Endpoints FastAPI, routers por epica
- [agrosat-backend-services](../.claude/skills/agrosat-backend-services/SKILL.md) — Service layer, workers Pub/Sub
- [agrosat-titiler-cog](../.claude/skills/agrosat-titiler-cog/SKILL.md) — TiTiler dinámico, COG overlays
- [agrosat-google-adk-agent](../.claude/skills/agrosat-google-adk-agent/SKILL.md) — Integración del agente ADK
- [agrosat-db-migrations](../.claude/skills/agrosat-db-migrations/SKILL.md) — dbmate workflow
- [agrosat-db-models](../.claude/skills/agrosat-db-models/SKILL.md) — SQLModel + GeoAlchemy2
- [agrosat-security](../.claude/skills/agrosat-security/SKILL.md) — Clerk OAuth, RBAC, rate limit, CSP
- [agrosat-security-audit](../.claude/skills/agrosat-security-audit/SKILL.md) — OWASP, CIS GCP
- [agrosat-testing](../.claude/skills/agrosat-testing/SKILL.md) — pytest, mocks Vertex AI + GEE + vLLM
- [agrosat-code-review](../.claude/skills/agrosat-code-review/SKILL.md) — PR checklist

## Auto-Invoke

| Acción | Skill |
|--------|-------|
| Crear/modificar endpoint FastAPI | `agrosat-backend-api` |
| Crear/modificar service class | `agrosat-backend-services` |
| Crear endpoint `/chat` SSE con ADK | `agrosat-backend-api` + `agrosat-google-adk-agent` |
| Servir tiles COG via TiTiler | `agrosat-titiler-cog` |
| Crear Pub/Sub worker (inference-jobs) | `agrosat-backend-services` |
| Crear migración dbmate | `agrosat-db-migrations` |
| Crear/modificar modelo SQLModel + GeoAlchemy2 | `agrosat-db-models` |
| Agregar Clerk auth + RBAC | `agrosat-security` |
| Configurar rate limit en endpoint | `agrosat-security` |
| Validar GeoJSON / coordenadas en input | `agrosat-security` |
| Audit logging | `agrosat-security` |
| Testing endpoint con mocks ADK | `agrosat-testing` |

## Critical Rules

- **ALWAYS**: Validar `session_id` en cada endpoint con `_check_session_owner()`
- **ALWAYS**: Usar `Depends(get_current_user)` con Clerk JWT en endpoints protegidos
- **ALWAYS**: Return Pydantic response models, jamás `SQLModel` crudo
- **ALWAYS**: GeoJSON validado con Pydantic `Feature` schema antes de pasar a service
- **ALWAYS**: Endpoint `/chat` retorna `StreamingResponse` con `Content-Type: text/event-stream`
- **ALWAYS**: Llamadas a Gemini / vLLM / GEE viven en service layer, jamás en router
- **ALWAYS**: Inferencia >2 s → Pub/Sub `inference-jobs` topic, jamás síncrono en request handler
- **ALWAYS**: Structured logging con `structlog`, jamás `print()`
- **ALWAYS**: Usar `poetry add` para dependencias
- **NEVER**: Lógica de negocio en routers (delegar a services)
- **NEVER**: Llamar Vertex AI / vLLM directo desde router
- **NEVER**: `SQLModel.metadata.create_all()` en producción
- **NEVER**: Modificar migraciones aplicadas — crear `dbmate new` forward
- **NEVER**: Raw SQL con string formatting — siempre parametrizado
- **NEVER**: Aceptar GeoTIFF / archivos pesados sin validar MIME + tamaño

## Project Structure

```
backend/
├── app/
│   ├── api/              # Routers por epica (chat.py, aois.py, timeseries.py, stac.py, llm_switch.py, tiles.py)
│   ├── models/           # SQLModel + GeoAlchemy2 (geometry, geography columns)
│   ├── services/         # Business logic (chat_service.py, aoi_service.py, stac_service.py)
│   ├── workers/          # Cloud Run jobs disparados por Pub/Sub (inference_worker.py, drift_worker.py)
│   ├── core/             # config.py, security.py, database.py, sse.py, adk_client.py
│   ├── middleware/       # security_headers.py, rate_limit.py, audit.py, request_id.py
│   └── utils/            # Reusable (gcs.py, redis.py, geo.py, sanitize.py)
├── tests/                # unit/, integration/, e2e/, fixtures/
├── pyproject.toml        # Poetry: grupos dev, test, ml (subset), geo
└── Dockerfile            # Multi-stage: builder (wheels) + slim runtime
```

## Decision Trees

```
¿Latencia esperada del endpoint?
  < 2 s    → Síncrono en handler
  > 2 s    → Pub/Sub topic `inference-jobs` → Cloud Run worker GPU L4
            Frontend hace polling `/jobs/{id}/status` o se suscribe a SSE

¿Tipo de respuesta del agente?
  Single message   → JSON normal
  Streaming chat   → SSE (StreamingResponse) con eventos:
                     plan_created | tool_call | tool_result | final_answer

¿Visualización de raster?
  Estática (PNG)       → backend genera y sirve PNG cached en GCS/Redis
  Interactiva en mapa  → TiTiler /tiles/{z/x/y}.png?url={cog_uri}&rescale=...

¿Función reutilizable?
  Usada 1x         → Mantener en service
  Usada 2x+        → Mover a app/utils/

¿Donde corre la lógica de un tool ADK?
  Definición + schema  → ml/agent/tools/*.py
  Invocación desde API → backend service llama a `agrosat_agent.run(query)`
```

## Endpoints clave (por epica)

```
GET    /healthz                        → smoke test
POST   /chat                           → SSE streaming, body: {query, aoi_geojson, llm_variant, session_id}
GET    /aois/{id}                      → AOI + classification cached
POST   /aois                           → crear AOI (GeoJSON) + lanzar pipeline async
GET    /aois/{id}/segmentation         → máscara polígonos JSON o COG URL
GET    /timeseries?aoi=...&index=NDVI  → serie temporal con Polars LazyFrame
GET    /stac/search                    → STAC API spec (bbox, datetime, collection)
GET    /tiles/{z}/{x}/{y}.png          → TiTiler dinámico, query params url, rescale, colormap
POST   /llm/switch                     → cambia variante A (Gemini) ↔ B (Qwen3.5)
GET    /jobs/{id}                      → estado de Pub/Sub worker job
GET    /llm/health                     → estado vLLM Qwen3.5 + Gemini
```

## Commands

```bash
make test                   # pytest backend con coverage (≥70%)
make test-unit              # solo unit tests, sin DB
make lint                   # ruff check + ruff format --check + mypy
make format                 # ruff format
make migrate                # dbmate up
make migrate-down           # dbmate down
make db-new name=xxx        # dbmate new create_xxx_table
make seed                   # python scripts/seed.py
poetry add <pkg>
poetry add --group dev <pkg>
poetry run pip-audit        # CVE scan
```

## QA Checklist Backend

- [ ] Todos los endpoints tienen `Depends(get_current_user)` con Clerk JWT
- [ ] `session_id` validado en cada query / mutation
- [ ] GeoJSON validado con Pydantic antes de pasar a service
- [ ] Services con docstrings Google style + type hints
- [ ] Inferencia ML pesada delegada a Pub/Sub workers
- [ ] `/chat` retorna SSE con eventos ADK (`plan_created`, `tool_call`, `tool_result`, `final_answer`)
- [ ] TiTiler endpoints con cache Redis (ttl 1h para tiles, 24h para mosaicJSON)
- [ ] Rate limit configurado en `/chat` (10 req/min) y `/llm/switch` (5 req/min)
- [ ] Audit logging en operaciones mutativas
- [ ] Tests unitarios cobertura ≥70%
- [ ] Tests E2E Playwright para flujo `/chat` con AOI fija
- [ ] `make lint` limpio
- [ ] Mocks Vertex AI + vLLM + GEE en tests (no llamadas reales)
- [ ] Migraciones con dbmate, índices GIST en columnas geometry
