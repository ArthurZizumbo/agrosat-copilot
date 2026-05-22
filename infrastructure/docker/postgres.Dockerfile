FROM postgis/postgis:15-3.4

RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-15-pgvector \
    && rm -rf /var/lib/apt/lists/*

# Crea la base de datos dedicada `mlflow` para el backend store de MLflow,
# separada de la DB `agrosat` de la aplicacion (evita mezclar el schema
# interno de MLflow con las migraciones dbmate del negocio).
COPY infrastructure/docker/init-mlflow-db.sql /docker-entrypoint-initdb.d/10-init-mlflow-db.sql
