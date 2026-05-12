---
name: agrosat-gcp-services
description: Configure Google Cloud Platform services for AgroSatCopilot — Cloud Run (api, frontend, tiling, inference-worker), Cloud SQL with PostGIS+pgvector, Cloud Pub/Sub, Vertex AI Agent Engine, Artifact Registry, Secret Manager, GCS, IAM with least privilege.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot GCP Services Skill

## Rules — NON-NEGOTIABLE

- Cloud Run scale-to-zero (min=0)
- Cloud SQL backups + PITR en prod
- Pub/Sub topics: `inference-jobs`, `inference-results`, `drift-events`
- Secret Manager para credenciales (GEE service account, HF token, Vertex AI key)
- IAM roles específicos, jamás `roles/owner` o `roles/editor`
- Artifact Registry para imágenes Docker `europe-west1-docker.pkg.dev/agrosat-prod/services/*`

## Servicios

| Servicio | Recurso | Notas |
|----------|---------|-------|
| `agrosat-api` | Cloud Run | FastAPI, 512 MB, 1 vCPU, min=0 max=10 |
| `agrosat-frontend` | Cloud Run | Nuxt 4 SSR, 256 MB, min=0 max=10 |
| `agrosat-tiling` | Cloud Run | TiTiler, 512 MB, min=0 max=5 |
| `agrosat-inference-worker` | Cloud Run GPU | L4 24GB, min=0 max=3 |
| `agrosat-postgres` | Cloud SQL | db-f1-micro (dev), db-custom-2-7680 (prod), PostGIS + pgvector |
| `agrosat-redis` | Memorystore | Basic 1 GB |
| `agrosat-data` | GCS | Datasets COG + Parquet |
| `agrosat-artifacts` | GCS | MLflow artifact store |
| `agrosat-dvc-remote` | GCS | DVC remote |
| `agrosat-tfstate` | GCS | Terraform state |
| `inference-jobs` | Pub/Sub | Disparador del worker GPU L4 |
| `agrosat-secrets` | Secret Manager | 6+ secrets |
| `agrosat-services` | Artifact Registry | Docker images |

## Cloud Run Deploy

```bash
# Backend
gcloud run deploy agrosat-api \
  --image europe-west1-docker.pkg.dev/agrosat-prod/services/api:sha-$(git rev-parse --short HEAD) \
  --region europe-west1 \
  --min-instances 0 --max-instances 10 \
  --memory 512Mi --cpu 1 \
  --set-env-vars "ENV=prod,LLM_VARIANT_DEFAULT=gemini" \
  --set-secrets "DATABASE_URL=db-url:latest,CLERK_SECRET=clerk-secret:latest,GEE_SA=gee-sa:latest" \
  --service-account agrosat-api@agrosat-prod.iam.gserviceaccount.com
```

## Cloud SQL Setup PostGIS + pgvector

```bash
gcloud sql instances create agrosat-prod \
  --database-version POSTGRES_15 \
  --tier db-custom-2-7680 \
  --region europe-west1 \
  --backup --enable-point-in-time-recovery

gcloud sql users create agrosat --instance=agrosat-prod --password=...

# Habilitar extensiones (después de conectar)
psql -c "CREATE EXTENSION IF NOT EXISTS postgis;"
psql -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql -c "CREATE EXTENSION IF NOT EXISTS pgstac;"
```

## Pub/Sub Topics

```bash
gcloud pubsub topics create inference-jobs
gcloud pubsub topics create inference-results
gcloud pubsub topics create drift-events

gcloud pubsub subscriptions create inference-jobs-worker \
  --topic inference-jobs \
  --push-endpoint https://agrosat-inference-worker-xxx.run.app \
  --ack-deadline 600
```

## Secret Manager

```bash
echo -n "$(cat gee-service-account.json)" | gcloud secrets create gee-sa --data-file=-
echo -n "${HF_TOKEN}" | gcloud secrets create huggingface-token --data-file=-
echo -n "${CLERK_SECRET}" | gcloud secrets create clerk-secret --data-file=-

# Grant access
gcloud secrets add-iam-policy-binding gee-sa \
  --member=serviceAccount:agrosat-api@agrosat-prod.iam.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor
```

## Vertex AI Agent Engine

```python
import vertexai
from vertexai.preview.reasoning_engines import ReasoningEngine

vertexai.init(project="agrosat-prod", location="us-central1")

remote = ReasoningEngine.create(
    agent=agrosat_agent,
    requirements=[
        "google-adk",
        "google-cloud-aiplatform[reasoning_engines]",
    ],
    display_name="agrosat-copilot-prod",
)
```

## IAM Service Accounts

| SA | Role | Para |
|----|------|------|
| `agrosat-api@` | `roles/cloudsql.client`, `roles/pubsub.publisher`, `roles/secretmanager.secretAccessor` | Cloud Run api |
| `agrosat-worker@` | `roles/storage.objectAdmin`, `roles/aiplatform.user` | Inference worker |
| `agrosat-dagster@` | `roles/storage.admin`, `roles/cloudsql.client` | Dagster orchestrator |
| `agrosat-gee@` | Earth Engine Resource Writer | GEE exports |

## QA Checklist

- [ ] Todos los servicios con SA específico
- [ ] Cloud Run min=0
- [ ] Cloud SQL backups + PITR
- [ ] Extensions postgis+vector+pgstac instaladas
- [ ] Secrets en Secret Manager
- [ ] IAM least privilege
- [ ] Artifact Registry con vulnerability scanning
