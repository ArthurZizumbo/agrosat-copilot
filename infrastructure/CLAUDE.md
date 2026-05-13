# Infrastructure Sub-Agent — AgroSatCopilot

> Sobreescribe al orquestador root para trabajo de infra.

**Rol**: Terraform 1.9+ mono-cloud GCP primario + Azure H100 puntual. CI/CD con GitHub Actions + Cloud Build. Scale-to-zero, FinOps target ~$115 USD/mes operativo.

## Skills References

- [agrosat-terraform](../.claude/skills/agrosat-terraform/SKILL.md) — Módulos, workspaces, backend state
- [agrosat-gcp-services](../.claude/skills/agrosat-gcp-services/SKILL.md) — Cloud Run, Cloud SQL, GCS, Pub/Sub, Vertex AI
- [agrosat-azure-h100](../.claude/skills/agrosat-azure-h100/SKILL.md) — VM Standard_NC40ads_H100_v5, scripts start/stop
- [agrosat-finops](../.claude/skills/agrosat-finops/SKILL.md) — Auditoría de costos
- [agrosat-security](../.claude/skills/agrosat-security/SKILL.md) — Secret Manager, Key Vault, IAM mínimo
- [agrosat-security-audit](../.claude/skills/agrosat-security-audit/SKILL.md) — CIS GCP Benchmarks

## Auto-Invoke

| Acción | Skill |
|--------|-------|
| Crear módulo Terraform GCP | `agrosat-terraform` + `agrosat-gcp-services` |
| Crear módulo Terraform Azure | `agrosat-terraform` + `agrosat-azure-h100` |
| Configurar Cloud Run service | `agrosat-gcp-services` |
| Configurar Cloud SQL PostGIS + pgvector | `agrosat-gcp-services` |
| Configurar Pub/Sub topic + subscription | `agrosat-gcp-services` |
| Configurar Vertex AI Agent Engine | `agrosat-gcp-services` |
| Crear VM Azure H100 spot | `agrosat-azure-h100` |
| Configurar Secret Manager / Key Vault | `agrosat-security` |
| Configurar IAM bindings | `agrosat-security` |
| Auditoría costo cloud | `agrosat-finops` |
| Pipeline GitHub Actions deploy | `agrosat-terraform` |
| Cloudbuild.yaml | `agrosat-gcp-services` |
| CIS GCP Benchmarks | `agrosat-security-audit` |

## Critical Rules

- **ALWAYS**: Workspaces Terraform separados `dev`, `staging`, `prod`
- **ALWAYS**: Backend state en `gs://agrosat-tfstate` con versionado activado
- **ALWAYS**: Variables sensibles vía `terraform.tfvars` (gitignored) o Secret Manager
- **ALWAYS**: `terraform plan` antes de `apply`, revisión por humano
- **ALWAYS**: Cloud Run con `min_instances=0` (scale-to-zero)
- **ALWAYS**: Cloud SQL con backups automáticos + point-in-time recovery
- **ALWAYS**: VM Azure H100 spot con auto-shutdown timer (default 12h)
- **ALWAYS**: NSG / firewall rules: SSH solo desde IPs de los 3 devs
- **ALWAYS**: IAM principle of least privilege (roles específicos, no `roles/owner`)
- **NEVER**: `terraform apply` sin plan revisado
- **NEVER**: Hardcodear ningún secreto en .tf files
- **NEVER**: `0.0.0.0/0` en firewall salvo Cloud Run público (HTTPS 443)
- **NEVER**: Encender H100 manualmente sin script (gastos descontrolados)
- **NEVER**: Mezclar recursos de envs distintos en un workspace

## Project Structure

```
infrastructure/
├── terraform/
│   ├── modules/
│   │   ├── gcp/
│   │   │   ├── cloud_run/       # api, frontend, tiling, inference-worker
│   │   │   ├── cloud_sql/       # PostgreSQL 15 + PostGIS + pgvector
│   │   │   ├── pubsub/          # inference-jobs, inference-results
│   │   │   ├── gcs/             # data, artifacts, dvc-remote, tfstate
│   │   │   ├── secret_manager/  # 6 secretos base
│   │   │   ├── vertex_ai/       # Agent Engine binding
│   │   │   ├── artifact_registry/
│   │   │   └── iam/
│   │   ├── azure/
│   │   │   ├── h100_vm/         # Standard_NC40ads_H100_v5 spot + on-demand
│   │   │   ├── blob_storage/    # Checkpoints LoRA
│   │   │   ├── vnet/
│   │   │   └── key_vault/
│   │   └── vertex/              # Reusable Vertex AI bindings
│   └── environments/
│       ├── dev/
│       │   ├── main.tf
│       │   ├── variables.tf
│       │   ├── backend.tf       # bucket=agrosat-tfstate, prefix=dev
│       │   └── terraform.tfvars # GITIGNORED
│       ├── staging/
│       └── prod/
├── docker/
│   ├── backend.Dockerfile
│   ├── frontend.Dockerfile
│   ├── inference-worker.Dockerfile
│   └── titiler.Dockerfile
├── cloudbuild.yaml             # Build + push + migrate + deploy
└── docker-compose.yml          # Dev local con 8 servicios
```

## Servicios GCP

| Servicio | Min instances | Max | Notas |
|----------|---------------|-----|-------|
| Cloud Run `api` | 0 | 10 | FastAPI, 512 MB, 1 vCPU |
| Cloud Run `frontend` | 0 | 10 | Nuxt 4 SSR, 256 MB |
| Cloud Run `tiling` | 0 | 5 | TiTiler, 512 MB |
| Cloud Run `inference-worker` GPU L4 | 0 | 3 | inferencia ML pesada |
| Cloud SQL | — | — | db-f1-micro, 20 GB, PITR |
| Upstash Redis (externo) | — | — | Pay-as-you-go serverless · ADR-003 · secretos en Secret Manager (`agrosat-upstash-rest-url`, `agrosat-upstash-rest-token`) |

## Comandos

```bash
make tf-plan env=dev          # terraform plan
make tf-apply env=dev         # terraform apply (con confirmación)
make tf-destroy env=dev       # cuidado, solo dev
make azure-h100-start         # enciende H100 spot
make azure-h100-stop          # apaga H100
make azure-h100-status        # estado actual + auto-shutdown timer
make cloud-build              # gcloud builds submit
make deploy-staging
make deploy-prod              # solo desde main
make cost-audit               # gcloud + az CLI: costos último mes
```

## QA Checklist Infra

- [ ] Workspace correcto (dev/staging/prod)
- [ ] `terraform plan` revisado por humano
- [ ] Backend state en GCS versionado
- [ ] Secret Manager / Key Vault para todos los secretos
- [ ] Cloud Run min_instances=0 (scale-to-zero)
- [ ] Cloud SQL con backups + PITR
- [ ] H100 VM con auto-shutdown
- [ ] IAM principle of least privilege
- [ ] CIS GCP Benchmarks pasados
- [ ] Costo estimado dentro de presupuesto ($115/mes operativo)
- [ ] Logs centralizados en Cloud Logging
- [ ] Smoke tests post-deploy
