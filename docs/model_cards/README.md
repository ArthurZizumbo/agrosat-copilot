# Model Cards — AgroSatCopilot

Documentacion de trazabilidad de los modelos **realmente entrenados o servidos**
en el proyecto, en formato de Model Card estilo HuggingFace. En espanol neutro,
sin emojis. **Toda metrica citada proviene de un artefacto real del repo**
(JSON / CSV de resultados, run MLflow); si un dato no se pudo verificar se anota
en [docs/blockers/epic10-notas.md](../blockers/epic10-notas.md), nunca se inventa
(regla de datos reales de Arthur).

## Indice

| Card | Modelo | Tipo | Estado |
|---|---|---|---|
| [ensemble-final-e6.md](ensemble-final-e6.md) | Ensemble final (E6) | Stacking / Blending sobre miembros base | Entrenado |
| [tsvit-pheno.md](tsvit-pheno.md) | TSViT-pheno | Segmentacion semantica temporal | Entrenado |
| [farslip-pheno.md](farslip-pheno.md) | FarSLIP-pheno | Vision-language fenologico (Li et al. 2025) | Entrenado |
| [qwen35-vllm-serving.md](qwen35-vllm-serving.md) | Qwen3.5-35B-A3B vLLM | Serving on-prem del reasoner (variante B) | Servido (no fine-tune) |

> **La Model Card de Gemma 4 fine-tuned NO se incluye.** Gemma 4 LoRA esta fuera
> de alcance ([ADR-011](../decisions/ADR-011-gemma4-lora-future.md), future). En
> su lugar se documenta el serving on-prem de Qwen3.5-35B-A3B (variante B del
> switch A/B del reasoner).

## Plantilla (secciones de cada card)

1. **Model Details** — nombre, arquitectura, version, licencia, autor del componente.
2. **Intended Use** — uso previsto y usos fuera de alcance.
3. **Training Data** — datasets, licencias, espacio de etiquetas.
4. **Evaluation** — protocolo de evaluacion (fold, split, metrica).
5. **Metrics** — cifras reales con la ruta exacta del artefacto fuente.
6. **Limitations & Ethical Considerations** — limites conocidos, sesgos, honestidad de claims.
7. **Licenses & Attribution** — licencias y atribuciones obligatorias.
8. **Reproducibility / MLflow** — tags `data_version` + `code_version` y el gotcha de los dos almacenes MLflow.

## Gotcha comun de reproducibilidad (todas las cards)

El lineage de MLflow vive en el **server Docker en `:5010`**, NO en `./mlruns`.
Un run lanzado por subprocess contra el server equivocado queda en estado
`RUNNING` y sus tags no aparecen en la UI. Para auditar tags
`data_version` / `code_version` levantar `make mlflow-up` y consultar el server
`:5010`; `track_experiment` (`ml/utils/mlflow_utils.py`) resuelve el URI con
fallback `:5010 -> file:./mlruns` y escribe ambos tags en todo run.
