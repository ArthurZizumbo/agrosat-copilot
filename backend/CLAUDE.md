# Backend Sub-Agent — AgroSatCopilot

> Scope `backend/`. Sobreescribe al orquestador root solo en conflicto local. NO repite las reglas NON-NEGOTIABLE: ver [`../CLAUDE.md`](../CLAUDE.md) (idioma, secrets, multi-tenant por `session_id`, router->service->model, inferencia pesada via Pub/Sub).

**Rol**: API REST + Tiling COG + workers asincronos. Integrara el agente Google ADK de `ml/agent/` via service layer.

## Estado

SKELETON. Solo existe el bootstrap; casi todo `app/` es `__init__.py` vacio.

- Reales: `app/main.py` (factory `create_app()` + `lifespan` + CORS endurecido), `app/core/config.py` (`Settings` + `get_settings()`), `app/core/logging.py`, `app/api/health.py` (`/healthz`, `/readyz`).
- Vacios (placeholders): `app/api/`, `app/models/`, `app/services/`, `app/workers/`, `app/middleware/`, `app/utils/`.
- NO existen aun (no asumir, crear via skill al abrir su US): routers `/chat`, `/aois`, `/timeseries`, `/stac`, `/llm/switch`, `/jobs`; `_check_session_owner`; middleware de rate-limit/auth; `adk_client`; modelos SQLModel; capa de servicios.

## Comandos

```bash
make lint               # ruff check + ruff format --check + mypy app/
make test               # pytest --cov=app --cov-fail-under=70
make test-unit          # pytest tests/unit -v (vacio hoy)
make test-integration   # pytest tests/integration -v (requiere Docker)
make db-seed            # poetry run python scripts/seed.py
make format             # ruff format .
```

Un solo test: `cd backend && poetry run pytest tests/integration/test_seed_smoke.py -q`

## Stack local

- FastAPI `^0.136` + Pydantic `^2.13` / pydantic-settings; respuestas tipadas con `BaseModel`.
- `structlog` `^25.5` para logging estructurado.
- SQLModel `^0.0.38` + GeoAlchemy2 `^0.20` + asyncpg `^0.31` — capa de datos **async** (geometry/geography columns).
- Config via `get_settings()` con `@lru_cache(maxsize=1)`; `Settings` tiene `extra="forbid"` (toda var de `.env.local` debe declararse) y rechaza defaults de dev si `env != dev`.
- Imports absolutos desde `backend.app...` (ej. `from backend.app.core.config import get_settings`).
- Arranque: `poetry run uvicorn backend.app.main:app --reload --port 8000`.

## Convenciones (✅/❌)

- ✅ `structlog.get_logger()` para todo log — ❌ `print()` en codigo de app.
- ✅ `async`/`await` end-to-end; lifecycle via `@asynccontextmanager lifespan` (no eventos `on_event` deprecados).
- ✅ Flujo router -> service -> model; la logica de negocio vive en `app/services/`, nunca en routers.
- ✅ Retornar Pydantic response models, jamas `SQLModel` crudo al cliente.
- ✅ Llamadas a Vertex/Gemini, vLLM o GEE viven en service layer.
- ❌ Leer `os.environ` directo en router/service — siempre `get_settings()`.
- ❌ Agregar variable a `.env.local` sin declararla en `Settings` (rompe el arranque por `extra="forbid"`).

## No tocar

- Migraciones aplicadas en `db/migrations/*.sql` (ej. `20260511213942_initial_schema.sql`) — solo `dbmate new` rollforward.
- `.env.local` (secrets locales) — nunca commitear ni hardcodear.
- Pins congelados en `pyproject.toml`: `torch 2.11.0+cu130` (index `pytorch-cu130`), `pyarrow ^23` (mlflow no soporta 24), `vllm`/`flash-attn` con marker `sys_platform == 'linux'` (no instalan en Win). No cambiarlos sin coordinar con el equipo.

## Tests

- Real: `tests/integration/test_seed_smoke.py` — levanta PostGIS efimero con `testcontainers`, aplica el bloque `migrate:up` de la migracion inicial y corre `scripts/seed.py` dos veces (exito + idempotencia). Se auto-skipea sin Docker. Marca `pytest.mark.integration`.
- `tests/unit/` y `tests/e2e/` estan vacios — crearlos al cerrar cada US.
- Mockear Vertex AI / vLLM / GEE / Clerk via `pytest-httpx`; Redis via `fakeredis`. Nunca llamadas reales en tests.
- Cobertura objetivo `--cov-fail-under=70` (root NON-NEGOTIABLE).

## Skills

| Accion | Skill |
|--------|-------|
| Endpoint/router FastAPI | `agrosat-backend-api` |
| Service class / worker Pub/Sub | `agrosat-backend-services` |
| `/chat` SSE con agente ADK | `agrosat-backend-api` + `agrosat-google-adk-agent` |
| Tiles COG via TiTiler | `agrosat-titiler-cog` |
| Migracion dbmate | `agrosat-db-migrations` |
| Modelo SQLModel + GeoAlchemy2 | `agrosat-db-models` |
| Clerk auth / rate limit / validacion GeoJSON | `agrosat-security` |
| Tests con mocks | `agrosat-testing` |
| Review pre-PR | `agrosat-code-review` |
