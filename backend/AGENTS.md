# Backend AGENTS — AgroSatCopilot

Ver [`CLAUDE.md`](CLAUDE.md) hermano para auto-invoke completo. Este archivo lista los **subagentes Task** que delegan trabajo profundo.

## Subagentes invocables

| Subagente | Archivo | Cuándo |
|-----------|---------|--------|
| `backend-engineer` | `../.claude/agents/backend-engineer.md` | Diseñar router + service + worker completo de una US, integrar ADK |
| `security-reviewer` | `../.claude/agents/security-reviewer.md` | OWASP en endpoints `/chat`, `/aois`, `/llm/switch`, audit JWT |
| `mlops-engineer` | `../.claude/agents/mlops-engineer.md` | Integración Pub/Sub ↔ Cloud Run GPU worker, scripts deploy |

## Skills clave (ver CLAUDE.md hermano)

- `agrosat-backend-api`
- `agrosat-backend-services`
- `agrosat-titiler-cog`
- `agrosat-google-adk-agent`
- `agrosat-db-migrations` / `agrosat-db-models`
- `agrosat-security` / `agrosat-security-audit`
- `agrosat-testing`
- `agrosat-code-review`

## Atajos de Decisión

```
Endpoint <2 s        → handler síncrono
Endpoint >2 s        → Pub/Sub + worker
Chat conversacional  → SSE (StreamingResponse + adk_client)
Tile raster          → TiTiler endpoint
Auth                 → Clerk JWT + require_roles()
Data tenant          → filtrar por session_id
DB schema change     → dbmate new + dbmate up
```
