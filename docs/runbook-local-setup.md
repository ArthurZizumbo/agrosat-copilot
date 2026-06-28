# Runbook — Levantar AgroSatCopilot en local

Guía paso a paso para que cualquier integrante levante el sistema completo (backend + frontend + chat conversacional con clasificación de cultivos) en su máquina, evitando los problemas que ya diagnosticamos. Probado en **Windows 11 + NVIDIA GPU**; las notas para CPU/Linux están señaladas.

> TL;DR de arquitectura recomendada en Windows:
> - **PostgreSQL + Redis** → en Docker.
> - **Backend (FastAPI)** → en **local con Poetry** (NO en Docker: la imagen `api` excluye el grupo `ml`/`torch` a propósito y no puede correr el chat).
> - **Frontend (Nuxt)** → en **local con `pnpm dev`** (la imagen Docker del frontend está rota).
> - **Gemini + Earth Engine** → vía **ADC de gcloud** (Vertex AI), sin API keys.

---

## 0. Prerrequisitos (instalar una sola vez)

| Herramienta | Versión | Notas |
|---|---|---|
| Docker Desktop | reciente | Para `postgres`, `redis`. |
| Python | **3.12** | El lock fija 3.12. Recomendado `pyenv-win`. |
| Poetry | **2.2.1** | `pipx install poetry==2.2.1`. |
| Node | **20+** | Para el frontend. |
| pnpm | **10+** | `corepack enable && corepack prepare pnpm@10 --activate`. Nunca npm/yarn. |
| gcloud CLI | reciente | Google Cloud SDK (incluye `bq`). |
| dbmate | reciente | Migraciones SQL. (O usar los `make db-*`.) |
| GPU NVIDIA + driver | CUDA **13.x** | Para `torch 2.11.0+cu130`. Sin GPU: ver [Apéndice B](#apéndice-b--sin-gpu-cpu). |
| DVC | (viene con `poetry install`) | Para descargar los datos versionados. |

Accesos Google necesarios (pídelos si no los tienes):
- Cuenta con acceso al proyecto GCP **`agrosat-copilot`**.
- **Vertex AI API** y **Earth Engine** habilitados en ese proyecto.

---

## 1. Clonar y configurar `.env.local`

```bash
git clone <repo-url> agrosat-copilot
cd agrosat-copilot
```

Crea **`.env.local`** en la raíz (es el archivo de configuración del backend; `pydantic-settings` con `extra="forbid"`, así que **toda** variable debe estar declarada en `backend/app/core/config.py`). Contenido mínimo que funciona:

```dotenv
# Postgres en Docker, puerto host 55432 (5432 suele estar ocupado por un PG nativo).
DATABASE_URL=postgresql+asyncpg://agrosat:agrosat@localhost:55432/agrosat

# Redis en Docker. Usar 6381 (ver paso 4: arrancar con REDIS_HOST_PORT=6381).
REDIS_URL=redis://localhost:6381/0

# Earth Engine / Vertex usan ADC (no service-account JSON en dev). Solo el proyecto.
GEE_PROJECT_ID=agrosat-copilot

# Reasoner Gemini por Vertex AI usando ADC. gemini-2.5-pro está en Vertex para
# este proyecto (gemini-3.5-flash da 404). NO se necesita API key de AI Studio.
GEMINI_MODEL=gemini-2.5-pro
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=agrosat-copilot
GOOGLE_CLOUD_LOCATION=us-central1

# El frontend corre en :3001 (3000 suele estar ocupado por otro proyecto).
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001
```

> **Importante:** `REDIS_HOST_PORT` NO va en `.env.local` (no es un campo de `Settings`; rompería el arranque por `extra="forbid"`). Es una variable de Docker Compose que se pasa al levantar (paso 4).

---

## 2. Dependencias del backend (Python)

Instala **todos los grupos** que el chat necesita en una sola pasada (esto evita el goteo de "No module named X"):

```bash
poetry install --with dev,test,ml,ml-gpu,geo
```

- `ml` → clasificadores, mlflow, polars, transformers, etc.
- `ml-gpu` → **torch 2.11.0+cu130** (Windows/Linux con GPU NVIDIA).
- `geo` → rasterio, earthengine-api, spyndex, eemont, shapely, h3, etc.

> Por qué tantos grupos: el tool de clasificación importa `ml.train`, que arrastra **todo** el stack de modelado (smp, monai, spyndex…). El backend no arranca el `/chat` sin ese stack completo.

### 2.1 Gotcha torch/torchvision (CRÍTICO en GPU)

El lock fija `torch ...+cu130` pero **no** fija `torchvision` desde el índice cu130, así que `pip`/`poetry` puede dejar una `torchvision +cpu` incompatible → error en runtime: `operator torchvision::nms does not exist`.

Verifica y, si no casan, instala la torchvision del índice cu130:

```bash
poetry run python -c "import torch,torchvision; print(torch.__version__, torchvision.__version__)"
# Si torchvision termina en +cpu (no +cu130):
poetry run pip install "torchvision==0.26.0+cu130" --extra-index-url https://download.pytorch.org/whl/cu130
```

### 2.2 Gotcha polars sin binario

Si `polars.__version__` sale vacío o ves `UserWarning: Polars binary is missing!` (rompe mlflow con `Invalid version: ''`), reinstala polars a la versión del lock (trae el runtime nativo):

```bash
poetry run pip install --force-reinstall --no-cache-dir "polars==1.40.1"
poetry run python -c "import polars as pl; print('polars', pl.__version__)"  # debe imprimir 1.40.1
```

### 2.3 Verificación rápida del entorno

```bash
poetry run python -c "import importlib.util as u; mods=['torch','torchvision','xgboost','h3','mlflow','polars','ee','rasterio','spyndex','eemont','segmentation_models_pytorch','monai','sentence_transformers','google.genai','litellm']; m=[x for x in mods if not u.find_spec(x)]; print('FALTAN:', m or 'ninguna')"
```

Debe imprimir `FALTAN: ninguna`.

---

## 3. Dependencias del frontend

```bash
cd frontend
pnpm install
cd ..
```

---

## 4. Infra local: Postgres + Redis (Docker)

```bash
REDIS_HOST_PORT=6381 docker compose --env-file .env.local up -d postgres redis
docker compose --env-file .env.local ps   # ambos "healthy"
```

- Postgres queda en `localhost:55432`, Redis en `localhost:6381` (casa con `.env.local`).
- **No** levantes los servicios `api`/`frontend` de Docker (ver TL;DR): corren en local.

---

## 5. Datos versionados (DVC)

El clasificador ajusta XGBoost sobre `data/features/features_fused_pastis.parquet` (124 MB) y lee los OOF de stacking/voting. Descárgalos:

```bash
dvc pull
# o, si solo quieres lo mínimo del clasificador:
dvc pull data/features/features_fused_pastis.parquet.dvc
```

---

## 6. Migraciones y datos demo

```bash
dbmate up                                   # crea chat_sessions, aois, parcels, features_parcels, RLS...
poetry run python scripts/seed.py           # sesión demo + AOI Tuscany
poetry run python scripts/seed_demo_parcels.py   # parcelas demo dentro del AOI
```

---

## 7. Autenticar Google (Vertex reasoner + Earth Engine)

El reasoner (Gemini por Vertex) y el muestreo AlphaEarth (Earth Engine) usan **ADC**:

```bash
gcloud config set project agrosat-copilot
gcloud auth application-default login        # abre navegador; usa tu cuenta con acceso al proyecto
gcloud auth application-default set-quota-project agrosat-copilot
```

> Si tienes varias cuentas: `gcloud config set account <tu-correo>` y repite el `application-default login` con `--account=<tu-correo>`.

---

## 8. Levantar backend y frontend

### Backend (local, SIN `--reload`)

```bash
poetry run uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

- El primer arranque tarda ~60 s (importa torch + stack ml).
- **NO uses `--reload` en Windows**: deja procesos huérfanos que retienen el puerto 8000 con código viejo (ver [Troubleshooting](#9-troubleshooting)).

### Frontend (local)

En otra terminal:

```bash
cd frontend
NUXT_PUBLIC_API_BASE_URL=http://localhost:8000 pnpm exec nuxt dev --port 3001
```

URLs:
- Frontend: **http://localhost:3001**
- Backend docs: **http://localhost:8000/docs** · health: `/healthz`, `/readyz`

---

## 9. Registrar la sesión del chat (necesario una vez)

El frontend **mintea** su `session_id` (UUID) en el cliente y lo guarda en una cookie `agrosat-session-id`. **No existe `POST /sessions`**, así que esa sesión no está en la BD y el `/chat` responde **403** (`chat_session_forbidden`) hasta que la insertes.

1. Abre el frontend (http://localhost:3001), abre DevTools → Application → Cookies → copia el valor de `agrosat-session-id` (un UUID). Alternativa: míralo en los logs del backend tras intentar chatear (`session_id=...`).
2. Inserta la fila (como superusuario `agrosat`, que bypasea RLS):

```bash
docker compose --env-file .env.local exec -T postgres psql -U agrosat -d agrosat -c \
"INSERT INTO chat_sessions (id, user_id, llm_model) VALUES ('<TU-UUID>', 'demo@agrosat.dev', 'gemini') ON CONFLICT (id) DO NOTHING;"
```

A partir de ahí el chat funciona (la cookie persiste 1 año). Repite solo si cambias de navegador/borras cookies.

---

## 10. Verificación end-to-end (smoke)

```bash
# Salud
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/readyz   # 200

# Chat con un AOI dibujado (ejemplo: Tuscany). Debe responder con un cultivo,
# no con "needs_gee_sampling".
curl -s -N -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: <TU-UUID>" \
  -d '{"messages":[{"role":"user","content":"que cultivo hay en esta zona?"}],"aoi":{"type":"Polygon","coordinates":[[[11.10,43.30],[11.11,43.30],[11.11,43.31],[11.10,43.31],[11.10,43.30]]]},"year":2019,"locale":"es"}'
```

En los logs del backend deberías ver `classify_new_parcel_embedding_resolved source=gee` (buscó en BD, no encontró parcela persistida y **descargó** el embedding de Earth Engine).

---

## 9. Troubleshooting (problemas que ya resolvimos)

| Síntoma | Causa | Solución |
|---|---|---|
| `ModuleNotFoundError: No module named '<x>'` al arrancar el back (`torch`, `polars`, `mlflow`, `xgboost`, `h3`, `spyndex`, `segmentation_models_pytorch`…) | El `.venv` no tiene el grupo `ml`/`geo` completo | `poetry install --with dev,test,ml,ml-gpu,geo` (paso 2). Si poetry falla por red, reintenta o instala los faltantes con `poetry run pip install <pkg>`. |
| `operator torchvision::nms does not exist` | `torchvision +cpu` no casa con `torch +cu130` | Paso [2.1](#21-gotcha-torchtorchvision-crítico-en-gpu). |
| `packaging.version.InvalidVersion: Invalid version: ''` / `Polars binary is missing!` | polars sin binario nativo | Paso [2.2](#22-gotcha-polars-sin-binario). |
| `/chat` → **403** `chat_session_forbidden` | La sesión del front no está en `chat_sessions` | Paso [9](#9-registrar-la-sesión-del-chat-necesario-una-vez). |
| Chat responde "dibuja el área" / perceiver `crop_class=needs_gee_sampling` | El AOI no intersecta una parcela persistida **y** el muestreo GEE no pudo correr | Verifica ADC (paso 7), `GEE_PROJECT_ID`, y que `dvc pull` haya traído el parquet de features (paso 5). |
| `/chat` → `No API key was provided` | El reasoner cayó en modo AI Studio sin key | Activa Vertex en `.env.local` (`GOOGLE_GENAI_USE_VERTEXAI=true` + project/location) y completa el ADC (paso 7). Reinicia el back. |
| Cambios en `.env.local`/código no surten efecto; el puerto 8000 sigue con comportamiento viejo | Worker **huérfano** de `uvicorn --reload` reteniendo el socket | No uses `--reload`. Para matarlo: `Get-NetTCPConnection -LocalPort 8000 -State Listen \| % { taskkill /F /T /PID $_.OwningProcess }` y relanza. |
| `api` en Docker crashea con `ModuleNotFoundError: torch` | La imagen `api` instala solo `dev,test` (sin `ml`) por diseño | Corre el backend en **local** (paso 8), no en Docker. |
| Redis no conecta (rate-limit) | Compose mapea Redis a `63790` por defecto, pero `.env.local` usa `6381` | Levanta con `REDIS_HOST_PORT=6381 docker compose ... up -d` (paso 4). |
| Frontend Docker no buildea (`COPY --from=deps /pnpm` not found) | Bug en `frontend.Dockerfile` (cache-mount no persistente) + `node_modules` win32 | Corre el frontend en **local** con `pnpm dev` (paso 8). |
| `/chat` devuelve "texto vacío" en la UI | Falso positivo: el `text_delta` sí trae texto; suele ser el reasoner pidiendo dibujar el AOI porque el perceiver dio `needs_gee_sampling` | Revisa la causa real arriba (ADC/GEE/sesión). |

---

## Apéndice A — Cambios en el working tree pendientes de commit

Para que el flujo "buscar → si no está, descargar de GEE" y el resto funcione, estos cambios deben estar en la rama (si haces `git pull` y no los ves, pídelos / commitéalos):

- `ml/ingest/gee_sampler.py` → nueva `sample_alphaearth_aoi_mean(...)` (muestreo AlphaEarth de un AOI vía Earth Engine).
- `ml/agent/tools/classify.py` → fallback que, si no hay embedding persistido que intersecte el AOI, **descarga** el embedding de GEE antes de caer a `needs_gee_sampling`.
- `docker-compose.yml` → `command` del `api` corregido a `uvicorn backend.app.main:app` (solo relevante si algún día se corre el api en Docker con el stack ml).

## Apéndice B — Sin GPU (CPU)

Si no tienes GPU NVIDIA, omite `ml-gpu` y usa torch CPU:

```bash
poetry install --with dev,test,ml,geo
poetry run pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

El chat funciona igual (la clasificación tabular XGBoost es CPU); solo afecta a entrenamientos pesados.

## Apéndice C — Hacer reproducible lo que hoy parchamos con pip

Lo ideal es que el equipo deje el entorno consistente desde el lock para no depender de los `pip install` manuales:

- Considerar **fijar `torchvision==0.26.0+cu130`** en el grupo `ml-gpu` (source `pytorch-cu130`) del `pyproject.toml`, para que `poetry install` traiga la torchvision correcta.
- Revisar el pin de `polars` (que el wheel traiga `polars-runtime-*`).
- Tras ajustar, `poetry lock` + commit, y todos hacen `poetry install --with dev,test,ml,ml-gpu,geo` sin parches.
