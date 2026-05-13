# `staging/` — Out of scope académico

Conforme a [ADR-002 — single-env-dev](../../../../docs/decisions/ADR-002-single-env-dev.md), durante el curso MNA (20-abr-2026 → 3-jul-2026) sólo se mantiene activo el entorno `dev`.

**No editar archivos aquí**. Reintroducir este entorno requiere:

1. Revertir ADR-002 mediante un nuevo ADR.
2. Replicar la estructura de `environments/dev/` (`main.tf`, `backend.tf`, `variables.tf`, `outputs.tf`, `terraform.tfvars`).
3. Cambiar `prefix = "dev"` por `prefix = "staging"` en `backend.tf`.
4. Validar costos con `agrosat-finops` antes de aplicar.

Motivos para reintroducir:

- Decisión de publicar AgroSatCopilot como SaaS post-curso.
- Necesidad de validar despliegues antes de tocar `prod`.
- Requisito explícito del sponsor (Dr. Camacho).
