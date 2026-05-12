# Frontend AGENTS — AgroSatCopilot

Ver [`CLAUDE.md`](CLAUDE.md) hermano. Este archivo lista subagentes Task delegables.

## Subagentes invocables

| Subagente | Archivo | Cuándo |
|-----------|---------|--------|
| `frontend-engineer` | `../.claude/agents/frontend-engineer.md` | Diseñar página + componentes + composables + store completo para una US |
| `security-reviewer` | `../.claude/agents/security-reviewer.md` | Auditar role guards, CSP, exposición de secrets en runtimeConfig |

## Skills clave (ver CLAUDE.md hermano)

- `agrosat-frontend-components`
- `agrosat-frontend-composables`
- `agrosat-maplibre-geo`
- `agrosat-security`
- `agrosat-testing`

## Atajos de Decisión

```
Texto visible           → t('key') obligatorio en it+es+en
Estado compartido       → Pinia store
Map overlay             → MapLibre raster source → TiTiler URL
Chat streaming          → useChat() / useSSE
Auth                    → Clerk middleware auth.ts
Test E2E flujo crítico  → Playwright /tests/e2e/
```
