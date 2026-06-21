#!/usr/bin/env bash
#
# bootstrap_cloud.sh -- Levanta el stack completo de AgroSatCopilot en un entorno
# con Docker (laptop, VM cloud genérica o cualquier host con `docker compose`).
#
# A diferencia de la VM H100 del sponsor (Windows nativo sin Docker -> usar
# scripts/bootstrap_sponsor_h100.ps1), aquí SÍ hay Docker, así que se orquestan
# los 8 servicios de docker-compose.yml (postgres+pgvector, redis, api, frontend,
# titiler, mlflow, dagster, ollama), se aplican las migraciones dbmate, se siembra
# la base y se valida con un smoke check de /healthz.
#
# Modos:
#   (default)        levanta el stack con docker compose y deja todo listo.
#   --deploy-gcp     en vez de compose, despliega a Cloud Run vía Terraform +
#                    Cloud Build (requiere gcloud autenticado y vars TF).
#   --no-seed        no ejecuta el seed (solo migraciones).
#   --down           detiene y limpia el stack (docker compose down -v).
#
# Uso:
#   bash scripts/bootstrap_cloud.sh
#   bash scripts/bootstrap_cloud.sh --no-seed
#   bash scripts/bootstrap_cloud.sh --deploy-gcp
#   bash scripts/bootstrap_cloud.sh --down
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log()  { printf '\033[0;36m[bootstrap-cloud]\033[0m %s\n' "$*"; }
warn() { printf '\033[0;33m[bootstrap-cloud] WARN:\033[0m %s\n' "$*"; }
die()  { printf '\033[0;31m[bootstrap-cloud] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

MODE="up"
RUN_SEED=1
for arg in "$@"; do
  case "$arg" in
    --deploy-gcp) MODE="deploy-gcp" ;;
    --down)       MODE="down" ;;
    --no-seed)    RUN_SEED=0 ;;
    *) die "argumento desconocido: $arg" ;;
  esac
done

# .env.local es obligatorio: docker-compose lo usa como env_file y el backend
# tiene extra=forbid. Si falta, se parte de .env.example como punto de partida.
if [[ ! -f .env.local ]]; then
  if [[ -f .env.example ]]; then
    warn ".env.local ausente; copiando .env.example -> .env.local (edita los secretos antes de prod)."
    cp .env.example .env.local
  else
    die ".env.local y .env.example ausentes; no puedo arrancar (extra=forbid)."
  fi
fi

# --- Modo: down --------------------------------------------------------------
if [[ "$MODE" == "down" ]]; then
  log "Deteniendo y limpiando el stack (docker compose down -v)..."
  docker compose --env-file .env.local down -v
  log "Stack detenido."
  exit 0
fi

# --- Modo: deploy-gcp --------------------------------------------------------
if [[ "$MODE" == "deploy-gcp" ]]; then
  log "Despliegue a GCP (Cloud Run vía Terraform + Cloud Build)."
  command -v gcloud >/dev/null 2>&1 || die "gcloud no esta instalado."
  command -v terraform >/dev/null 2>&1 || die "terraform no esta instalado."
  gcloud auth print-access-token >/dev/null 2>&1 || die "gcloud no esta autenticado (gcloud auth login)."

  TF_DIR="infrastructure/terraform/environments/dev"
  [[ -d "$TF_DIR" ]] || die "no encuentro $TF_DIR"

  log "terraform init + plan en $TF_DIR (revisa el plan antes de aplicar)..."
  ( cd "$TF_DIR" && terraform init -input=false && terraform plan -input=false )
  warn "Plan generado. Para APLICAR realmente ejecuta:"
  warn "  ( cd $TF_DIR && terraform apply )"
  warn "El build/push de imágenes vive en Cloud Build (cloudbuild.yaml); ver infrastructure/AGENTS.md."
  log "deploy-gcp: fase de plan completada (apply manual por seguridad)."
  exit 0
fi

# --- Modo: up (docker compose) -----------------------------------------------
command -v docker >/dev/null 2>&1 || die "docker no esta instalado (en la VM H100 usa scripts/bootstrap_sponsor_h100.ps1)."
docker compose version >/dev/null 2>&1 || die "docker compose v2 no disponible."

log "Levantando los 8 servicios (docker compose up -d --build)..."
docker compose --env-file .env.local up -d --build

# Espera a que postgres este healthy antes de migrar.
log "Esperando a que postgres este healthy..."
for _ in $(seq 1 30); do
  state="$(docker compose --env-file .env.local ps postgres --format '{{.Health}}' 2>/dev/null || echo '')"
  [[ "$state" == "healthy" ]] && break
  sleep 2
done
[[ "${state:-}" == "healthy" ]] || warn "postgres no reporto healthy; las migraciones pueden fallar."

# --- Migraciones dbmate ------------------------------------------------------
# El puerto host de postgres sale de .env.local (POSTGRES_HOST_PORT, default 55432).
PG_PORT="$(grep -E '^POSTGRES_HOST_PORT=' .env.local | cut -d= -f2 | tr -d '\r' || true)"
PG_PORT="${PG_PORT:-55432}"
export DATABASE_URL="postgres://agrosat:agrosat@localhost:${PG_PORT}/agrosat?sslmode=disable"

if command -v dbmate >/dev/null 2>&1; then
  log "Aplicando migraciones dbmate (puerto ${PG_PORT})..."
  dbmate up
  dbmate status
else
  warn "dbmate no esta en PATH; aplicando migraciones dentro del contenedor api..."
  docker compose --env-file .env.local exec -T api sh -lc 'dbmate up && dbmate status' \
    || warn "No se pudieron aplicar migraciones automaticamente; corre 'dbmate up' a mano."
fi

# --- Seed --------------------------------------------------------------------
if [[ "$RUN_SEED" -eq 1 ]]; then
  log "Sembrando la base (scripts/seed.py, idempotente)..."
  docker compose --env-file .env.local exec -T api sh -lc 'python scripts/seed.py' \
    || warn "El seed fallo o no aplica; continua (revisa logs del servicio api)."
fi

# --- Smoke check -------------------------------------------------------------
API_PORT="$(grep -E '^API_HOST_PORT=' .env.local | cut -d= -f2 | tr -d '\r' || true)"
API_PORT="${API_PORT:-8010}"
log "Smoke check de /healthz (puerto ${API_PORT})..."
api_ok=0
for _ in $(seq 1 30); do
  if curl -fsS "http://localhost:${API_PORT}/healthz" >/dev/null 2>&1; then api_ok=1; break; fi
  sleep 2
done
[[ "$api_ok" -eq 1 ]] && log "Backend OK -> http://localhost:${API_PORT}" || warn "Backend no respondio /healthz; revisa 'docker compose logs api'."

# --- Resumen -----------------------------------------------------------------
FE_PORT="$(grep -E '^FRONTEND_HOST_PORT=' .env.local | cut -d= -f2 | tr -d '\r' || echo 3010)"
ML_PORT="$(grep -E '^MLFLOW_HOST_PORT=' .env.local | cut -d= -f2 | tr -d '\r' || echo 5010)"
DG_PORT="$(grep -E '^DAGSTER_HOST_PORT=' .env.local | cut -d= -f2 | tr -d '\r' || echo 3011)"
echo
log "Stack levantado. Endpoints:"
echo "  - API      : http://localhost:${API_PORT:-8010}  (/docs, /healthz, /metrics)"
echo "  - Frontend : http://localhost:${FE_PORT:-3010}"
echo "  - MLflow   : http://localhost:${ML_PORT:-5010}"
echo "  - Dagster  : http://localhost:${DG_PORT:-3011}"
echo
log "Para detener: bash scripts/bootstrap_cloud.sh --down"
log "Para desplegar a GCP: bash scripts/bootstrap_cloud.sh --deploy-gcp"
