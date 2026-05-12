# Database AGENTS — AgroSatCopilot

Ver [`CLAUDE.md`](CLAUDE.md) hermano.

## Subagentes invocables

| Subagente | Archivo | Cuándo |
|-----------|---------|--------|
| `backend-engineer` | `../.claude/agents/backend-engineer.md` | Diseñar schema + migración + modelo SQLModel para una US |
| `geo-data-engineer` | `../.claude/agents/geo-data-engineer.md` | Schema espacial pgstac + GIST + RLS |
| `security-reviewer` | `../.claude/agents/security-reviewer.md` | Auditar RLS por session_id, secret scanning en seeds |

## Skills clave

- `agrosat-db-migrations`
- `agrosat-db-models`
- `agrosat-security`
