-- Crea la base de datos dedicada `mlflow` para el backend store de MLflow.
-- Se ejecuta una sola vez, al inicializar el cluster Postgres (entrypoint initdb).
-- La DB `mlflow` aloja las ~6 tablas internas de MLflow (runs, metrics,
-- params, registered_models, ...), separadas del schema de negocio en `agrosat`.
CREATE DATABASE mlflow OWNER agrosat;
