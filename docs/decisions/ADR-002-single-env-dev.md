# ADR-002 — Un solo entorno Terraform `dev` durante el curso

**Status**: Propuesta · pendiente visto bueno equipo
**Fecha**: 2026-05-11
**Decisores**: Arthur Zizumbo (MLOps Lead), Aaron Bocanegra, Isaac Avila
**US relacionada**: [US-001](../us-planning/us-001.md)
**Avance**: A0 (2026-04-26)

---

## Contexto

El plan v6 menciona tres entornos Terraform (`dev`, `staging`, `prod`) y la
estructura inicial del repo crea `infrastructure/terraform/environments/{dev,staging,prod}/`
con `.gitkeep`.

El proyecto integrador:

- Dura 10 semanas efectivas (20-abr → 21-jun-2026) + 2 buffer.
- Presupuesto operativo objetivo: **~$115 USD/mes** total (Cloud Run scale-to-zero
  + 1 Cloud SQL + storage). Un entorno por env tripilcaria minimo el costo de
  Cloud SQL y de IPs reservadas.
- No se planea release a producción real durante el curso: la demo final corre
  contra el entorno único; el Paper Track opcional post-curso ya queda fuera
  del scope evaluado.

Mantener `staging/` y `prod/` vacíos con `.gitkeep` invita a futuras US a
poblarlos, generando inconsistencias entre el código de la app y las
variantes de cada env.

## Decisión

1. **Único entorno activo: `dev`**. Se materializa en
   `infrastructure/terraform/environments/dev/` con `main.tf`, `variables.tf`,
   `backend.tf`, `outputs.tf` y `terraform.tfvars` (gitignored).
2. `environments/staging/` y `environments/prod/` se mantienen como
   directorios con un `README.md` que indica "Out of scope académico — ver
   ADR-002. Reintroducir solo si se decide publicar SaaS post-curso".
3. El módulo GCP acepta una variable `environment` (default `dev`) que
   permitiria reactivar staging/prod sin reescribir el modulo, solo el wrapper
   de entorno.
4. `cloudbuild.yaml` y `.github/workflows/ci.yml` quedan parametrizados con
   `_ENV` / `environment` substitution para soportar staging/prod a futuro,
   pero en CI solo se valida `dev`.

## Consecuencias

### Positivas

- −2/3 del costo operativo (1 Cloud SQL vs 3, 1 Redis vs 3, etc.).
- Menos pipelines CI/CD que validar — `make tf-validate env=dev` suficiente.
- Foco en entregables del calendario (avances semanales) en vez de en
  hardening de prod.

### Negativas

- Si se decide publicar SaaS post-curso requiere reintroducir entornos.
  Mitigación: módulos GCP y Azure ya parametrizados por `environment`,
  resta solo crear `environments/{staging,prod}/main.tf` con tfvars distintos.
- No se ejercita el flujo "promote from staging to prod" durante el curso.
  Aceptable: el flujo CI/CD multi-env queda documentado pero no testeado.

### Neutras

- US-005 (CI/CD), US-052 (deploy staging), US-054 (deploy prod) del plan v6
  quedan reinterpretadas a deploy único `dev`. Cuando se reactiven entornos
  adicionales, se generara una US de seguimiento.

## Alternativas consideradas

| Alternativa | Razón rechazo |
|-------------|---------------|
| Mantener dev + staging + prod con presupuesto tripiclado | Excede $115 USD/mes objetivo (R FinOps) |
| Solo dev + prod (sin staging) | Sigue duplicando Cloud SQL; no aporta vs solo dev para curso académico |
| Workspaces Terraform (`terraform workspace`) en vez de directorios | Más opaco para revisor; directorios separados son explícitos |

## Referencias

- Plan v6 §10 (FinOps) — [`context/RefinamientoPlaneacionAgroSatCopilot_v6.md`](../../context/RefinamientoPlaneacionAgroSatCopilot_v6.md)
- US-001 plan §11 — [`docs/us-planning/us-001.md`](../us-planning/us-001.md)
- ADR-001 — [`ADR-001-no-cookiecutter-externo.md`](ADR-001-no-cookiecutter-externo.md)
