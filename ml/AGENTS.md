# ML AGENTS — AgroSatCopilot

Ver [`CLAUDE.md`](CLAUDE.md) hermano. Este archivo lista subagentes Task delegables.

## Subagentes invocables

| Subagente | Archivo | Cuándo |
|-----------|---------|--------|
| `ml-engineer` | `../.claude/agents/ml-engineer.md` | Diseñar arquitectura, calcular VRAM, configurar LoRA, validar SOTA |
| `geo-data-engineer` | `../.claude/agents/geo-data-engineer.md` | Pipeline ingesta GEE + CDSE + DINOv3 + pgstac |
| `mlops-engineer` | `../.claude/agents/mlops-engineer.md` | Asset Dagster + DVC + MLflow + script H100 reproducible |
| `agent-engineer` | `../.claude/agents/agent-engineer.md` | Cuando el trabajo cae en `ml/agent/` |

## Skills clave (ver CLAUDE.md hermano)

- `agrosat-gee-alphaearth`
- `agrosat-ml-features`, `agrosat-ml-baseline`, `agrosat-ml-segmentation`, `agrosat-ml-ensemble`
- `agrosat-llm-finetuning`, `agrosat-azure-h100`
- `agrosat-ml-evaluation`
- `agrosat-dvc-mlflow`, `agrosat-dagster-mlops`

## Atajos de Decisión

```
Dato satelital nuevo      → agrosat-gee-alphaearth (AlphaEarth) / agrosat-ml-features (Sentinel)
Modelo tabular            → agrosat-ml-baseline
Modelo segmentación       → agrosat-ml-segmentation
Fine-tune VLM             → agrosat-llm-finetuning + agrosat-azure-h100
Ensamble                  → agrosat-ml-ensemble
Eval                      → agrosat-ml-evaluation
```
