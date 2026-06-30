# Arranque de la demo AgroSatCopilot — guía operativa

Automatización probada para levantar **toda la app** de cara a la presentación, con
validación por healthchecks. Los scripts viven en `scripts/demo_up.ps1` (levantar) y
`scripts/demo_down.ps1` (bajar).

> **Probado el 2026-06-30** en Windows local (Docker Desktop 29.5) contra el Postgres
> dockerizado y el backend nativo. El frontend y el on-prem se arrancan con la misma
> automatización; los bugs de Docker que se sortean están documentados abajo.

---

## TL;DR

```powershell
# Levantar todo (datos Docker + backend/frontend nativos):
pwsh scripts/demo_up.ps1

# Levantar además el LLM on-prem (Qwen) por túnel a la VM H100:
pwsh scripts/demo_up.ps1 -WithVM

# Bajar todo:
pwsh scripts/demo_down.ps1
```

Al terminar, el script imprime un resumen con el estado de cada servicio y las URLs.

---

## Arquitectura de la demo (y por qué es híbrida)

La demo NO corre 100% en Docker ni 100% en la VM. Es **híbrida por necesidad**, y cada
decisión está respaldada por una limitación real verificada:

| Servicio | Dónde corre | Por qué |
|---|---|---|
| Postgres + PostGIS + pgvector | **Docker local** | El contenedor `agrosat-postgres:15-3.4-pgvector` funciona perfecto (healthy 38 h en la prueba). |
| Redis | **Docker local** | Contenedor `redis:7-alpine`, healthy inmediato. |
| TiTiler (tiling COG) | **Docker local** | Imagen oficial `developmentseed/titiler`, sin build propio. |
| Backend FastAPI | **Nativo (Poetry)** | El contenedor `api` falla con `ModuleNotFoundError: No module named 'app'` (bug DOCKER-API). Nativo arranca limpio y responde `healthz` 200 contra el Postgres dockerizado. |
| Frontend Nuxt | **Nativo (pnpm dev)** | El contenedor `frontend` falla en `COPY --from=deps /pnpm /pnpm: not found` (bug DOCKER-FRONT). Nativo con `pnpm dev` levanta normal. |
| LLM on-prem (Qwen) | **VM H100 (nativo) + túnel SSH** | La VM no tiene Docker (virtualización anidada off). Qwen corre con `llama-server.exe` nativo en `:8002`; el script abre un túnel SSH local. |

Esta topología es además **coherente con el discurso de soberanía de datos** del proyecto:
los modelos pesados viven en la H100 propia y se exponen por endpoint OpenAI-compatible,
sin que el dato salga del perímetro.

---

## Por qué la VM no hospeda la app completa

Se verificó el inventario nativo de la VM (`F:\worktrees\us082\_vm_inventory.ps1`):
la VM **solo** tiene Ollama (Gemma), micromamba+Python (`agrosat`) y las tareas de serving
de Qwen. **No tiene Postgres, Redis ni Node/pnpm instalados**, y el frontend ni siquiera
tiene `node_modules`. Instalar Postgres+PostGIS+pgvector, Redis y Node nativos en la VM
Windows del sponsor sería invasivo y frágil. La VM está optimizada para **serving de LLMs
y entrenamiento en la H100**, no para hospedar la pila web — por eso la app full-stack se
demuestra desde local (donde Docker sí está) y la VM solo aporta el LLM on-prem.

---

## Requisitos

- **Docker Desktop** corriendo (para postgres/redis/titiler).
- **Poetry** con el env del backend instalado (`cd backend && poetry install`).
- **pnpm** + `frontend/node_modules` (el script corre `pnpm install` si falta).
- **dbmate** en el PATH (para migraciones).
- `.env.local` en la raíz (con `DATABASE_URL`, `DBMATE_DATABASE_URL`, `REDIS_URL`, claves
  GEE/Gemini, etc.).
- Para `-WithVM`: túnel cloudflared a la VM activo + llave `~/.ssh/agrosat_h100` + la tarea
  `qwen_serve` corriendo en la VM (ver `docs/serving/encender-modelos-onprem.md`).

---

## Qué hace el script, paso a paso

0. **Retira** los contenedores rotos `api`/`frontend` si quedaron de un `make dev` previo
   (ocupan los puertos 8010/3010 y chocan con los nativos).
1. **Datos**: `docker compose up -d postgres redis titiler` y espera a que Postgres reporte
   `healthy` (vía `docker inspect`).
2. **Migraciones**: `dbmate up` con dos correcciones — `--env-file .env.local` (el repo no
   usa `.env`) y la URL `DBMATE_DATABASE_URL` (sin el driver `+asyncpg` que dbmate rechaza).
3. **Backend nativo**: `poetry run uvicorn app.main:app` desde `backend/`, con
   `-WorkingDirectory` explícito; espera `healthz` 200.
4. **Frontend nativo**: `pnpm dev` desde `frontend/`; espera respuesta del puerto 3010.
5. **On-prem (opcional)**: túnel `ssh -N -L 8002:127.0.0.1:8002`; espera `/health` del Qwen
   (tarda ~40 s en cargar el GGUF).

Cada paso es **idempotente**: si un puerto ya está en uso, reutiliza el servicio en lugar
de duplicarlo.

---

## Bugs reales encontrados al probar (candidatos a arreglar después)

Estos se descubrieron ejecutando la automatización, no en teoría. Están en la lista de
mejoras `docs/app-review/MEJORAS-app-2026-06-30.md`; aquí se sortean para que la demo
levante:

- **DOCKER-API**: `infrastructure/docker/backend.Dockerfile` (target dev) deja el módulo
  `app` fuera del PYTHONPATH del contenedor → `ModuleNotFoundError: No module named 'app'`.
  Sorteo: backend nativo.
- **DOCKER-FRONT**: `infrastructure/docker/frontend.Dockerfile` referencia `COPY --from=deps
  /pnpm /pnpm`, pero el stage `deps` no produce `/pnpm` → build falla. Sorteo: frontend nativo.
- **DBMATE-URL**: el `DATABASE_URL` de la app trae `postgresql+asyncpg`, que dbmate no
  soporta. Sorteo: usar `DBMATE_DATABASE_URL`.

---

## Validación manual rápida (tras `demo_up.ps1`)

```bash
curl http://127.0.0.1:8010/healthz      # backend -> {"status":"ok",...}
curl http://127.0.0.1:8011/healthz      # titiler -> 200
# frontend: abrir http://localhost:3010 en el navegador
# con -WithVM:
curl http://127.0.0.1:8002/health       # Qwen on-prem -> 200 (tras ~40s de carga)
```
