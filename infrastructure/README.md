# Infrastructure — AgroSatCopilot

Capa IaC del monorepo: Dockerfiles, Cloud Build pipeline y módulos Terraform GCP + Azure.

## Layout

```
infrastructure/
├── docker/
│   ├── backend.Dockerfile          # multi-stage: builder + dev + runtime
│   ├── frontend.Dockerfile         # multi-stage: deps + dev + build + runtime
│   ├── dagster.Dockerfile          # orchestrator (poetry group `dagster`)
│   └── inference-worker.Dockerfile # CUDA base + poetry group `ml` (GPU L4 worker)
├── cloudbuild.yaml             # build + push + migrate + deploy 4 servicios
└── terraform/
    ├── modules/
    │   ├── gcp/                # Cloud Run x4, Cloud SQL, GCS, Pub/Sub, Secret Manager, AR, IAM
    │   └── azure/              # H100 VM spot, Blob, VNet+NSG, Key Vault, auto-shutdown
    └── environments/
        ├── dev/                # único entorno activo (ADR-002)
        ├── staging/            # out of scope (README)
        └── prod/               # out of scope (README)
```

## Quickstart

### Prerequisitos

- Terraform `>= 1.9`
- `gcloud` autenticado (`gcloud auth application-default login`)
- `az` autenticado (`az login`)
- Bucket `gs://agrosat-tfstate` pre-creado con versionado:

  ```bash
  gsutil mb -l europe-west1 -b on gs://agrosat-tfstate
  gsutil versioning set on gs://agrosat-tfstate
  ```

### Plan / Apply (entorno `dev`)

```bash
cd infrastructure/terraform/environments/dev

# Primera vez: copia y completa los placeholders (Azure subscription, SSH key, CIDRs).
# terraform.tfvars está gitignored.
cp ../../../../docs/templates/terraform.tfvars.example terraform.tfvars  # si existe el template

terraform init
terraform fmt -recursive ../..
terraform validate
terraform plan -out=tfplan
terraform apply tfplan
```

### Validación sin credenciales (CI)

```bash
cd infrastructure/terraform/environments/dev
terraform init -backend=false
terraform validate
```

### Formateo

```bash
terraform fmt -recursive infrastructure/terraform/
```

## Decisiones irrevocables aplicadas

- **GCP region**: `europe-west1` (proximidad Italia).
- **Cloud Run `min_instances=0`** en todos los servicios (scale-to-zero).
- **Cloud SQL** con `backup_configuration.enabled=true` + PITR.
- **Azure H100**: `Standard_NC40ads_H100_v5` spot por defecto, auto-shutdown 23:00 UTC.
- **IAM**: ningún SA runtime tiene `roles/owner`; bindings mínimos por servicio.
- **Secret Manager / Key Vault**: 6 secretos GCP + Key Vault Azure; ningún plaintext en `.tf`.
- **NSG**: SSH whitelist por CIDR (max 5 IPs de devs), regla DenyAll en prioridad 4096.
- **Backend state**: `gs://agrosat-tfstate/dev` con versionado de objetos.

## Cost target

Operativo dev: ~$115 USD/mes (scale-to-zero + db-f1-micro + Spot H100 puntual). Ver `make cost-audit`.

## Refs

- [ADR-001 — no-cookiecutter-externo](../docs/decisions/ADR-001-no-cookiecutter-externo.md)
- [ADR-002 — single-env-dev](../docs/decisions/ADR-002-single-env-dev.md)
- [infrastructure/CLAUDE.md](CLAUDE.md)
