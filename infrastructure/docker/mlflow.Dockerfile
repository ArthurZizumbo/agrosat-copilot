# Servidor MLflow local con el driver Postgres para el backend store.
# La imagen oficial ghcr.io/mlflow/mlflow no incluye psycopg2; el backend
# store en PostgreSQL (DB dedicada `mlflow`) lo requiere.
FROM ghcr.io/mlflow/mlflow:v3.1.0

RUN pip install --no-cache-dir psycopg2-binary

# US-022-c P1 etapa 3a fix (2026-05-24): wrapper que construye --backend-store-uri
# en runtime desde env var DB_PASSWORD (inyectada por Cloud Run --set-secrets desde
# Secret Manager agrosat-db-password). Evita el anti-patron de embeber el password
# en `args` del Cloud Run service (visible en `gcloud run services describe`).
COPY infrastructure/docker/mlflow-start.sh /usr/local/bin/mlflow-start
RUN chmod +x /usr/local/bin/mlflow-start

ENTRYPOINT ["/usr/local/bin/mlflow-start"]
