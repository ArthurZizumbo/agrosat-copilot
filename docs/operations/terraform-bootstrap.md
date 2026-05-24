# Terraform bootstrap — bucket `agrosat-tfstate`

**Estado**: operativo desde 2026-05-23 (US-022-c P4.3)
**Owner**: Arthur Zizumbo (MLOps lead)
**Re-importado por**: cualquier dev con `roles/storage.admin` temporal en
`agrosat-copilot`. En estado estable, el rol vive solo en el grupo MLOps.
**Backend declarado**: [`infrastructure/terraform/environments/dev/backend.tf`](../../infrastructure/terraform/environments/dev/backend.tf)
**Recurso HCL**: [`infrastructure/terraform/modules/gcp/main.tf`](../../infrastructure/terraform/modules/gcp/main.tf) lineas 392-414 (`google_storage_bucket.tfstate`).

---

## 1. Paradoja bootstrap

Terraform persiste su state en `gs://agrosat-tfstate/dev/default.tfstate`
(backend `gcs` configurado en `backend.tf`). Pero el bucket que aloja el state
es a la vez un recurso administrado por el modulo GCP — un Terraform fresco
necesita el bucket existente PARA escribir su state, pero declarar el bucket
en HCL deberia crearlo con el mismo `terraform apply`.

Esta paradoja se resuelve en dos pasos disjuntos:

1. **Pre-creacion manual del bucket** (una sola vez por proyecto GCP):
   ```bash
   gsutil mb -l us-central1 -b on -p agrosat-copilot gs://agrosat-tfstate
   gsutil versioning set on gs://agrosat-tfstate
   ```
2. **Importar el bucket al state** despues del primer `terraform init` para
   que el HCL lo reconozca como managed:
   ```bash
   cd infrastructure/terraform/environments/dev
   terraform import 'module.gcp.google_storage_bucket.tfstate' agrosat-tfstate
   ```

Despues del import, `terraform plan` debe reportar **0 cambios** sobre el
bucket. Si hay drift, ver §4.

---

## 2. Procedimiento step-by-step (re-import en checkout limpio)

Aplicable cuando un dev nuevo clona el repo y necesita anclar su state local al
bucket remoto.

```bash
# 1) Autenticar gcloud con cuenta que tenga roles/storage.admin temporal
gcloud auth login
gcloud config set project agrosat-copilot
gcloud auth application-default login

# 2) Verificar que el bucket existe (si no, abortar — solo el owner del
#    proyecto puede crearlo desde cero, ver §1).
gsutil ls gs://agrosat-tfstate/ 2>&1 || \
  echo "ERROR: bucket no existe; coordina con Arthur (gjcamacho@tec.mx) antes de seguir."

# 3) Init del backend gcs (descarga providers y configura el state remoto)
cd infrastructure/terraform/environments/dev
terraform init -upgrade

# 4) IMPORTAR el bucket al state — comando canonico
terraform import 'module.gcp.google_storage_bucket.tfstate' agrosat-tfstate

# 5) Validar que no hay drift
terraform plan
# Esperado: "Plan: 0 to add, 0 to change, 0 to destroy."
# Si reporta drift en `lifecycle_rule`, ver §4 abajo.
```

---

## 3. Comando exacto de import (referencia rapida)

```bash
terraform import 'module.gcp.google_storage_bucket.tfstate' agrosat-tfstate
```

- El path del recurso (`module.gcp.google_storage_bucket.tfstate`) debe envolverse
  en comillas simples en bash/PowerShell para evitar expansion de `[]`.
- El segundo argumento (`agrosat-tfstate`) es el `id` del bucket en GCP — el
  nombre canonico, no la URI `gs://...`.
- Si `terraform plan` posterior reporta destroy del bucket, hay riesgo de
  perdida del state remoto: NO aplicar; abrir issue + escalar al MLOps lead.

---

## 4. Drift conocido en `lifecycle_rule`

El HCL declara:

```hcl
lifecycle_rule {
  condition {
    num_newer_versions = 30
  }
  action {
    type = "Delete"
  }
}
```

Si el bucket fisico tiene una politica de retencion distinta (p.ej. 10 versions
por una creacion manual con `gsutil`), `terraform plan` reportara drift de
`condition.num_newer_versions`. Dos opciones:

1. **Adoptar el HCL como fuente de verdad** (recomendado): aplicar el plan; el
   bucket se reconfigura a 30 versiones. Riesgo: si habia ya 31+ versiones
   antiguas, la siguiente compaction GCS borrara las mas viejas.
2. **Ajustar el HCL para reflejar lo del bucket fisico**: editar `main.tf`
   lineas 405-411 y subir num_newer_versions/age al valor real. Documentar el
   cambio en este archivo.

**Decision para US-022-c P4.3 (2026-05-23)**: el bucket fue creado con
`versioning ON` y sin lifecycle rules antes del import; el HCL le impone 30
versiones via primer `terraform apply` post-import. No hay drift residual en
`terraform plan`.

---

## 5. Quien tiene grant `roles/storage.admin` temporal

| Persona | Rol | Vigencia |
|---------|-----|----------|
| Arthur Zizumbo (artzizumbo@gmail.com) | `roles/storage.admin` | permanente (MLOps lead) |
| Aaron Bocanegra | `roles/storage.objectViewer` (lectura state) | permanente |
| Isaac Avila | `roles/storage.objectViewer` (lectura state) | permanente |

Para re-importar el bucket, los devs sin `storage.admin` deben pedir grant
temporal a Arthur con duracion <= 24 h via Cloud Console -> IAM -> Add member,
o usar la SA `agrosat-ci-sa@agrosat-copilot.iam.gserviceaccount.com` (ya tiene
`roles/storage.objectAdmin` segun el modulo GCP).

---

## 6. Referencias

- [`infrastructure/terraform/environments/dev/backend.tf`](../../infrastructure/terraform/environments/dev/backend.tf) — configuracion del backend GCS.
- [`infrastructure/terraform/modules/gcp/main.tf`](../../infrastructure/terraform/modules/gcp/main.tf) lineas 392-414 — declaracion HCL del bucket.
- [`docs/us-planning/us-022-c.md`](../us-planning/us-022-c.md) seccion 6.4 — bloque P4.3.
- [Terraform docs — gcs backend](https://developer.hashicorp.com/terraform/language/backend/gcs)
- [Terraform docs — import command](https://developer.hashicorp.com/terraform/cli/commands/import)
