# Agent AGENTS — AgroSatCopilot

Ver [`CLAUDE.md`](CLAUDE.md) hermano. Subagentes Task delegables.

## Subagentes invocables

| Subagente | Archivo | Cuándo |
|-----------|---------|--------|
| `agent-engineer` | `../../.claude/agents/agent-engineer.md` | Diseñar tool nuevo, planner, RAG, eval contra benchmarks |
| `ml-engineer` | `../../.claude/agents/ml-engineer.md` | Si el tool requiere modelo nuevo (clasificador, segmentación) |
| `backend-engineer` | `../../.claude/agents/backend-engineer.md` | Integración del agente con el endpoint `/chat` SSE |

## Skills clave

- `agrosat-google-adk-agent`
- `agrosat-spatial-rag`
- `agrosat-llm-finetuning` (para backends)
- `agrosat-ml-evaluation`
