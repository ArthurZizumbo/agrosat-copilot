# `prod/` — Out of scope académico

Conforme a [ADR-002 — single-env-dev](../../../../docs/decisions/ADR-002-single-env-dev.md), durante el curso MNA (20-abr-2026 → 3-jul-2026) sólo se mantiene activo el entorno `dev`.

**No editar archivos aquí**. Reintroducir este entorno requiere:

1. Revertir ADR-002 mediante un nuevo ADR.
2. Crear primero el entorno `staging/` y validar al menos un release end-to-end.
3. Replicar la estructura de `environments/dev/` con valores hardening:
   - `deletion_protection = true` en Cloud SQL
   - `availability_type = "REGIONAL"`
   - `cloudrun_min_instances >= 1` para servicios críticos (revisar FinOps)
   - SSH whitelist reducida a IPs estáticas verificadas
4. Cambiar `prefix = "dev"` por `prefix = "prod"` en `backend.tf`.
5. Aprobación explícita del equipo + sponsor antes de `terraform apply`.

Motivos para reintroducir:

- Lanzamiento SaaS post-curso.
- Pilotos con usuarios reales (cooperativas agrícolas Toscana).
