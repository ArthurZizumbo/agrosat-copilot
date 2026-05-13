# ADR-001 — No publicar repo cookiecutter externo

**Status**: Propuesta · pendiente visto bueno equipo + Dr. Camacho
**Fecha**: 2026-05-11
**Decisores**: Arthur Zizumbo (MLOps Lead), Aaron Bocanegra, Isaac Avila
**US relacionada**: [US-001](../us-planning/us-001.md)
**Avance**: A0 (2026-04-26)

---

## Contexto

El plan v6 §EPIC 0 (US-001) pide generar la estructura del monorepo mediante un
template `cookiecutter` publicado en `gh:agrosatcopilot/cookiecutter-agrosat`,
con prompts interactivos para `project_name`, `gcp_project_id`,
`azure_subscription_id`, `region`, `db_name`, `team_lead_email`.

La realidad del proyecto:

- Curso académico de 10 semanas con 3 desarrolladores.
- No se planea spinear N proyectos similares: el repo es el artefacto.
- Variables como `gcp_project_id` y `azure_subscription_id` ya viven (o vivirán)
  en `terraform.tfvars` y `.env.local` — pedirlas también en cookiecutter
  dispersaria la configuracion en tres sitios y obligaria a mantener un mapeo
  manual entre ellos.
- Publicar un repo `cookiecutter-agrosat` externo requiere mantenerlo
  sincronizado con la estructura real del monorepo, doblando la superficie de
  cambio en cada US futura que modifique la arquitectura.

## Decisión

1. **No publicar** un repo cookiecutter externo.
2. El monorepo `agro_sat_copilot` actua como **single source of truth** de la
   estructura del proyecto.
3. Onboarding se reduce a:

   ```bash
   git clone <repo>
   cp .env.example .env.local   # editar variables reales
   make bootstrap                # poetry + pnpm install
   make dev                      # docker compose up (8 servicios)
   make db-migrate               # dbmate up
   ```

4. Las variables que un cookiecutter hubiese pedido por prompt quedan:
   - `project_name` → inferido del directorio (hardcoded `agrosatcopilot`).
   - `gcp_project_id` → `.env.local` + `infrastructure/terraform/environments/dev/terraform.tfvars`.
   - `azure_subscription_id` → idem.
   - `region` → hardcoded `europe-west1` (proximidad Italia; justificado en módulo Terraform GCP).
   - `db_name` → hardcoded `agrosat` (consistente con `docker-compose.yml`).
   - `team_lead_email` → `pyproject.toml` (`[tool.poetry] authors`) + `.github/CODEOWNERS`.

## Consecuencias

### Positivas

- −2 SP de mantenimiento (no hay repo cookiecutter sincronizando).
- Onboarding más directo: 1 clone vs 1 cookiecutter + 1 clone + 2 prompts.
- Una sola fuente de verdad para la estructura (el monorepo actual).
- Coherente con presupuesto académico y calendario inamovible
  (presentación 21-jun-2026).

### Negativas

- Pierde capacidad de "spinear un proyecto similar" en 30 segundos.
  Mitigación: si post-curso se quisiera SaaS multi-tenant, se puede generar
  el cookiecutter externo a partir del monorepo existente en ~1 día.
- Si Dr. Camacho exige cookiecutter literal por rúbrica, ver R-1 del plan
  (§9). Mitigación: presentar esta ADR como evidencia de decisión
  arquitectónica fundamentada (criterio reproducibilidad sigue cumpliéndose).

### Neutras

- Los criterios de aceptación de US-001 se reinterpretan: AC-1 a AC-18
  validan **el monorepo clonado**, no la generación cookiecutter.

## Alternativas consideradas

| Alternativa | Razón rechazo |
|-------------|---------------|
| Publicar cookiecutter externo + mantenerlo en CI | +2 SP de mantenimiento por sprint; doble fuente de verdad |
| Cookiecutter local (`templates/cookiecutter-agrosat/` dentro del repo) | El monorepo ya cumple la función; duplicar la estructura no aporta |
| Copier en vez de cookiecutter | Mismo problema: dos artefactos a sincronizar |

## Referencias

- Plan v6 §EPIC 0 — [`context/RefinamientoPlaneacionAgroSatCopilot_v6.md`](../../context/RefinamientoPlaneacionAgroSatCopilot_v6.md)
- US-001 plan — [`docs/us-planning/us-001.md`](../us-planning/us-001.md) §11
- Rúbrica A0 — [`docs/general/Rubricas Integrador.html`](../general/Rubricas%20Integrador.html)
