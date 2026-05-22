# Servidor MLflow local con el driver Postgres para el backend store.
# La imagen oficial ghcr.io/mlflow/mlflow no incluye psycopg2; el backend
# store en PostgreSQL (DB dedicada `mlflow`) lo requiere.
FROM ghcr.io/mlflow/mlflow:v3.1.0

RUN pip install --no-cache-dir psycopg2-binary
