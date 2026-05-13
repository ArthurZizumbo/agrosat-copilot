# ADR-004 — Grupos Poetry opcionales + Vertex AI Agent Engine fuera del backend

**Status**: Propuesta · pendiente visto bueno equipo
**Fecha**: 2026-05-12
**Decisores**: Arthur Zizumbo (MLOps Lead), Aaron Bocanegra, Isaac Avila
**US relacionada**: [US-001](../us-planning/us-001.md) (descubrimiento durante manual-test 5ª/6ª pasada), impacta US-008 (deploy backend), US-010 (inference-worker), US-013 (agent ADK)
**Avance**: A0 (2026-04-26)

---

## Contexto

Durante la 6ª pasada de QA del manual-test de US-001 se ejecutó por primera
vez `docker compose build api` (backend FastAPI) y se observó que la imagen
instalaba **417 paquetes Python** en el stage `dev`, incluyendo:

- Stack CUDA completo: `nvidia-cublas`, `nvidia-cusparse`, `nvidia-nvjitlink`,
  `nvidia-cuda-cupti`, `nvidia-cuda-nvrtc`, `nvidia-cuda-runtime`, `nvidia-cufft`,
  `nvidia-cufile`, `nvidia-curand`, `nvidia-cusolver`, `nvidia-nvtx` (~3 GB).
- `torch 2.11.0+cu130`, `transformers`, `peft`, `monai`, `segmentation-models-pytorch`.
- `mlflow`, `dvc[gs]`, `polars`, `optuna`, `xgboost`, `lightgbm`, `sentence-transformers`.
- `google-adk`, `litellm`.

El backend FastAPI **no necesita ninguno de esos paquetes**: solo expone
endpoints REST, valida Pydantic, despacha jobs a Pub/Sub, y consulta Postgres.
La inferencia ML pesada vive en `inference-worker` (Cloud Run GPU L4 / Azure H100)
y la orquestación de pipelines en `dagster`.

### Causa raíz #1 — `google-cloud-aiplatform[reasoning_engines]` en grupo `main`

`pyproject.toml` declaraba en `[tool.poetry.dependencies]`:

```toml
google-cloud-aiplatform = { extras = ["reasoning_engines"], version = "^1.152.0" }
```

El extra `reasoning_engines` arrastra **transitivamente** todo el SDK de Vertex
AI Agent Engine, que incluye `aiohttp`, `cloudpickle`, `opentelemetry-*`,
`google-cloud-iam/logging/trace`, `pandas`, etc. Esto se hizo porque el agente
Google ADK necesita Vertex Agent Engine para deploy. Pero el agente vive en
`ml/agent/`, no en handlers FastAPI, por lo que **el backend nunca lo usa
en runtime**.

### Causa raíz #2 — Grupos `ml,geo,dagster,paper` no marcados como opcionales

Por default, `poetry install` instala TODOS los grupos no-opcionales del
`pyproject.toml`. Sin `optional = true`, los grupos `ml`, `geo`, `dagster`,
`paper` se instalaban siempre — incluso al correr `poetry install --with
dev,test --no-root` en el `backend.Dockerfile`. El flag `--with` solo agrega
opcionales; no excluye no-opcionales.

## Decisión

**A — Mover `google-cloud-aiplatform[reasoning_engines]` del grupo `main` al
grupo `ml`**.

```toml
# Antes (grupo main, llega al backend)
[tool.poetry.dependencies]
google-cloud-aiplatform = { extras = ["reasoning_engines"], version = "^1.152.0" }

# Después (grupo ml, solo inference-worker / dagster / dev local)
[tool.poetry.group.ml.dependencies]
google-cloud-aiplatform = { extras = ["reasoning_engines"], version = "^1.152.0" }
```

El backend mantiene únicamente: `google-cloud-pubsub`, `google-cloud-storage`,
`google-cloud-secret-manager`, `earthengine-api`. Suficiente para despachar
jobs, firmar URLs GCS, leer secretos y consultar Earth Engine para validación
de geometrías.

**B — Marcar `ml`, `geo`, `dagster`, `paper` como `optional = true`**.

```toml
[tool.poetry.group.ml]
optional = true

[tool.poetry.group.ml.dependencies]
transformers = "^5.8.0"
peft = "^0.19.1"
# ...

[tool.poetry.group.geo]
optional = true
# ...
```

Resultado: `poetry install` (sin `--with`) instala **solo `main`** (~80 deps),
suficiente para el backend Dockerfile. `make bootstrap` sigue trayendo todo
porque ya invoca `--with dev,test,ml,geo,dagster,paper`.

## Consecuencias

### Positivas

| Métrica | Antes | Después | Reducción |
|---------|-------|---------|-----------|
| Backend Dockerfile install (paquetes) | 417 | ~94 | **-78 %** |
| Backend Docker image (estimada) | ~5 GB | ~800 MB - 1 GB | **-80 %** |
| Cloud Run cold start backend | ~12-18 s | ~3-5 s | **-70 %** |
| Artifact Registry storage backend | ~5 GB × N revisions | ~1 GB × N revisions | -80 % costo |
| Tiempo de `docker build api` (sin cache) | ~10 min | ~2 min | -80 % |
| `nvidia-*` (CUDA) en backend | 11 paquetes | **0** | -100 % |
| `torch+cu130` en backend | sí (~2.5 GB) | NO | -100 % |

### Neutras

- `make bootstrap` sigue funcionando idéntico — ya pasa `--with` explícito.
- `make bootstrap-gpu` y `make bootstrap-gpu-linux` también idénticos.
- `inference-worker.Dockerfile` ya pasa `--with ml,ml-gpu,ml-gpu-linux,geo` →
  sigue trayendo Vertex AI Agent Engine como antes.
- `dagster.Dockerfile` pasa `--only main,dagster` → no afectado (el agente ADK
  no se invoca desde dagster, solo desde handlers backend o worker).

### Negativas / a vigilar

- Si en el futuro se agrega un endpoint backend que invoque Vertex AI Agent
  Engine **directamente en el handler** (sin pasar por Pub/Sub al worker), el
  import fallará en runtime. La solución correcta sigue siendo despachar el
  job al `inference-worker`, no llamar Vertex desde backend.
- Devs que ejecuten `poetry install` sin `--with` recibirán un venv mínimo
  (sin ML). Documentado en `Makefile` (`bootstrap` vs `bootstrap-gpu` vs
  `bootstrap-gpu-linux`) y en `README.md` §Setup.

## Alternativas consideradas

1. **Usar `[tool.poetry.extras]` en lugar de grupos opcionales**: poetry extras
   son invocables vía `poetry install -E reasoning`, pero requieren marcar las
   deps con `optional = true` individualmente (en lugar de a nivel de grupo) y
   son más opacas para devs. Descartado.
2. **Crear un `pyproject.toml` separado para backend**: rompe el monorepo
   Python (decisión irrevocable v6 §3). Descartado.
3. **Quitar el extra `[reasoning_engines]` de aiplatform**: factible, pero el
   agente ADK requiere ese extra para `agent_engines.create()`. Mantener el
   extra, mover el paquete al grupo correcto.
4. **Build args en Dockerfile para desactivar grupos via Poetry CLI**: más
   complejo y específico al Dockerfile. Los grupos opcionales son la forma
   canónica documentada en Poetry 2.x.

## Implementación

Cambios aplicados en una sola pasada del 2026-05-12:

```
pyproject.toml:
  - L43-48: comentario "GCP solo Pub/Sub + GCS + Secret Manager"; quitado google-cloud-aiplatform
  - L88-90: nuevo bloque "Vertex AI / Agent" en [tool.poetry.group.ml.dependencies]
  - L74,108,121,127: + "[tool.poetry.group.X] optional = true" en ml, geo, paper, dagster
  - "main" final: 18 deps (fastapi, uvicorn, pydantic, sqlmodel, sqlalchemy,
    asyncpg, geoalchemy2, pgvector, redis, structlog, slowapi, python-magic,
    clerk-sdk, titiler-core, rio-tiler, google-cloud-pubsub, google-cloud-storage,
    google-cloud-secret-manager, earthengine-api)

poetry.lock: regenerado, 487 paquetes totales (suma de todos los grupos).
```

Validación:

| Comando | Antes | Después |
|---------|-------|---------|
| `poetry install --with dev,test --no-root --dry-run` | 417 ops | **94 ops** |
| `poetry install --with dev,test,ml,geo,dagster,paper --no-root --dry-run` | 417 ops | 322 ops |
| Grep `nvidia\|torch\|mlflow\|dvc` en backend dry-run | 11+ matches | **0** |

## Referencias

- [Poetry 2.x dependency groups documentation](https://python-poetry.org/docs/managing-dependencies/#optional-groups)
- `pyproject.toml` líneas 25-49 (grupo main), 73-91 (grupo ml + Vertex AI)
- `infrastructure/docker/backend.Dockerfile` (consume `--with dev,test --no-root`)
- `infrastructure/docker/inference-worker.Dockerfile` (consume `--with ml,ml-gpu,ml-gpu-linux,geo`)
- `Makefile` targets `bootstrap`, `bootstrap-gpu`, `bootstrap-gpu-linux`
- Manual-test US-001 6ª pasada (origen del descubrimiento)
- ADR-002 (single-env-dev) — esta optimización aplica a `dev`; staging/prod heredan
- ADR-003 (Upstash Redis) — patrón similar de "no traer infra cara al backend si
  no se usa en runtime"
