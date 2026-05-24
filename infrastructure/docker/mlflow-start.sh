#!/bin/sh
# AgroSatCopilot - MLflow Cloud Run entrypoint wrapper (US-022-c P1 etapa 3a fix).
#
# Construye --backend-store-uri en runtime usando el password Postgres desde la
# env var DB_PASSWORD (inyectada por Cloud Run --set-secrets desde Secret Manager
# agrosat-db-password). Evita el anti-patron de embeber el password en los
# `args` del Cloud Run (visible en `gcloud run services describe`).
#
# Variables requeridas (Cloud Run env):
#   DB_PASSWORD            - Postgres password (via --set-secrets desde Secret Manager)
#   CLOUDSQL_CONNECTION    - <project>:<region>:<instance> del Cloud SQL postgres
#   ARTIFACT_ROOT          - gs://bucket/path para artefactos MLflow
#
# Si alguna var falta, abort con mensaje claro y exit 1 (sin printear DB_PASSWORD).
set -eu

if [ -z "${DB_PASSWORD:-}" ]; then
  echo "ERROR mlflow-start: DB_PASSWORD env var no definida. Verificar --set-secrets en Cloud Run." >&2
  exit 1
fi
if [ -z "${CLOUDSQL_CONNECTION:-}" ]; then
  echo "ERROR mlflow-start: CLOUDSQL_CONNECTION env var no definida." >&2
  exit 1
fi
if [ -z "${ARTIFACT_ROOT:-}" ]; then
  echo "ERROR mlflow-start: ARTIFACT_ROOT env var no definida." >&2
  exit 1
fi

BACKEND_URI="postgresql://agrosat:${DB_PASSWORD}@/mlflow?host=/cloudsql/${CLOUDSQL_CONNECTION}"

# NO echo de BACKEND_URI (contiene password) — solo confirmar que arranca.
echo "mlflow-start: launching server on :5000 (artifact_root=${ARTIFACT_ROOT})"

exec mlflow server \
  --backend-store-uri "${BACKEND_URI}" \
  --default-artifact-root "${ARTIFACT_ROOT}" \
  --host 0.0.0.0 \
  --port 5000 \
  --workers 2 \
  --serve-artifacts
