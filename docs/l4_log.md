# L4 usage log

Registro de uso de GPU L4 spot en GCP Vertex AI custom-jobs. Equivalente al
`h100_log.md` planificado para Azure H100 (US-024). Sirve para:

- Auditoria FinOps (gate A-6 de US-022b-A): nunca dejar L4 colgada.
- Trazar SP consumidos vs presupuesto (10-12 h L4 para US-022b completo, ADR-006).
- Reconstruir reproducibilidad: cada entrada referencia el run MLflow y el SHA
  de la imagen `ml-train` usada.

## Convenciones

- `Job ID`: ultimo segmento de `gcloud ai custom-jobs describe NAME --region=R --format='value(name)'`.
- `Display name`: `train-<EPIC>-<US>[-smoke]`.
- `Image SHA`: `${SHORT_SHA}` con el que Cloud Build construyo `ml-train`.
- `Duracion`: minutos efectivos (no incluye queue ni cold start MLflow).
- `Costo estimado`: g2-standard-8 spot L4 ~$0.30 USD/h on-demand,
  ~$0.10 USD/h spot (price aproximado may-2026). Verificar en
  `gcloud billing` mensualmente.
- `MLflow run`: link relativo o run_id si el server esta accesible.

## Entradas

### 2026-05-22 — US-022b-A AC-5 smoke (planificado, no ejecutado todavia)

| Campo | Valor |
|-------|-------|
| Job ID | (pendiente) |
| Display name | `train-E5-US-022b-smoke` |
| Image SHA | (pendiente — primer build de ml-train) |
| Image URI | `us-central1-docker.pkg.dev/agrosat-copilot/agrosat/ml-train:latest` |
| Config | `ml/configs/l4_smoke.yaml` (timeoutSeconds=1200, strategy=SPOT) |
| Duracion objetivo | < 10 min |
| Costo estimado | < $0.20 USD |
| Estado | **BLOQUEADO POR BILLING/CREDENCIALES** — la infra esta como codigo en `feature/E5-US-022b`. Ejecutar `make train-l4-smoke` una vez Arthur tenga `gcloud auth application-default login` + billing activo en `agrosat-copilot`. |
| MLflow run | (pendiente — experiment `us-022b-a-smoke`, run name `smoke-l4-tempcnn`) |
| Notas | Smoke trivial: 1 forward+backward de `breizhcrops.models.TempCNN` sobre tensor random (B=64, T=24, F=10, C=5). Valida cadena Cloud Build -> Artifact Registry -> Vertex AI L4 -> MLflow Cloud Run -> GCS artifacts. |

### Pasos para ejecutar el smoke (cuando billing este disponible)

```bash
# 1) Build & push de ml-train (esto consume Cloud Build minutos)
gcloud builds submit --config=infrastructure/cloudbuild.yaml \
  --substitutions=_ENV=dev,_PROJECT_ID=agrosat-copilot,_REGION=us-central1

# 2) Provisionar MLflow Cloud Run + SA + buckets via Terraform
cd infrastructure/terraform/environments/dev
terraform init
terraform plan -out tfplan
terraform apply tfplan
export MLFLOW_TRACKING_URI=$(terraform output -raw mlflow_tracking_uri)
cd -

# 3) Smoke A-5
make train-l4-smoke

# 4) Verificar resultado
gcloud ai custom-jobs list --region=us-central1 \
  --filter='displayName:train-E5-US-022b-smoke' \
  --format='table(displayName,state,endTime)'

# 5) Confirmar A-6 (no hay jobs colgados)
make cost-audit
```

Esperado en la fila del job: `state=JOB_STATE_SUCCEEDED`.

### Siguientes entradas (placeholders para US-022b-B/C)

- 2026-05-26 — US-022b-B FarSLIP training (8 h hard cap, ~$0.80 USD).
- 2026-05-28 — US-022b-C TempCNN + InceptionTime spatial CV (~3 h, ~$0.30 USD).
