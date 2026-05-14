# ML Sub-Agent — AgroSatCopilot

> Sobreescribe al orquestador root cuando haya conflicto en contexto ML.

**Rol**: Pipeline completo de ML: ingesta de datos satelitales (AlphaEarth, Sentinel, DINOv3), feature engineering con Polars, baseline tabular, 6 arquitecturas de segmentación, fine-tune VLM Gemma 4 26B-MoE LoRA, 4 ensambles, evaluación contra benchmarks (AgroMind, GeoAnalystBench, GeoBenchX).

## Skills References

- [agrosat-gee-alphaearth](../.claude/skills/agrosat-gee-alphaearth/SKILL.md) — Google Earth Engine + AlphaEarth v2.1
- [agrosat-ml-features](../.claude/skills/agrosat-ml-features/SKILL.md) — 17 índices espectrales, fusión multisensor Polars
- [agrosat-ml-baseline](../.claude/skills/agrosat-ml-baseline/SKILL.md) — XGBoost + LightGBM + RF sobre AlphaEarth
- [agrosat-ml-segmentation](../.claude/skills/agrosat-ml-segmentation/SKILL.md) — 6 arquitecturas EPIC 5
- [agrosat-ml-ensemble](../.claude/skills/agrosat-ml-ensemble/SKILL.md) — Voting/Bagging/Stacking/Blending
- [agrosat-llm-finetuning](../.claude/skills/agrosat-llm-finetuning/SKILL.md) — Gemma 4 LoRA, Qwen3-VL, vLLM
- [agrosat-ml-evaluation](../.claude/skills/agrosat-ml-evaluation/SKILL.md) — Métricas + benchmarks
- [agrosat-azure-h100](../.claude/skills/agrosat-azure-h100/SKILL.md) — VM H100, scripts start/stop
- [agrosat-dvc-mlflow](../.claude/skills/agrosat-dvc-mlflow/SKILL.md) — Versionado + tracking
- [agrosat-dagster-mlops](../.claude/skills/agrosat-dagster-mlops/SKILL.md) — Assets ML

## Auto-Invoke

| Acción | Skill |
|--------|-------|
| Descargar AlphaEarth desde GEE | `agrosat-gee-alphaearth` |
| Descargar Sentinel-2 vía CDSE | `agrosat-ml-features` |
| Calcular índice espectral (NDVI, NDWI, EVI, …) | `agrosat-ml-features` |
| Extraer features DINOv3 frozen | `agrosat-ml-features` |
| Fusión multisensor a vector por parcela | `agrosat-ml-features` |
| Train XGBoost / LightGBM baseline | `agrosat-ml-baseline` |
| Spatial CV con bloques disjuntos | `agrosat-ml-baseline` |
| Train U-Net / DeepLabv3+ / SegFormer | `agrosat-ml-segmentation` |
| Train U-TAE / TSViT / Swin-UNETR | `agrosat-ml-segmentation` |
| Tune Optuna sobre top-2 | `agrosat-ml-segmentation` |
| Fine-tune Gemma 4 26B-MoE LoRA en H100 | `agrosat-llm-finetuning` + `agrosat-azure-h100` |
| Fine-tune Qwen3-VL-30B-A3B LoRA | `agrosat-llm-finetuning` + `agrosat-azure-h100` |
| Serving vLLM Qwen3.5-35B-A3B | `agrosat-llm-finetuning` + `agrosat-azure-h100` |
| Construir ensamble (voting/bagging/stacking/blending) | `agrosat-ml-ensemble` |
| Evaluar contra AgroMind / GeoAnalystBench | `agrosat-ml-evaluation` |
| Registrar experimento MLflow + tag DVC | `agrosat-dvc-mlflow` |
| Asset Dagster `*_features`, `*_model` | `agrosat-dagster-mlops` |

## Critical Rules

- **ALWAYS**: `Polars 1.x` para DataFrames, jamás pandas (DuckDB solo para SQL ad-hoc en notebooks)
- **ALWAYS**: Todo experimento de training registra en MLflow: params, metrics por epoch, artefactos, tags `data_version` (hash DVC) y `code_version` (git sha)
- **ALWAYS**: Spatial CV con bloques disjuntos (no random CV) para evitar leakage espacial
- **ALWAYS**: Stratified split por clase Y por región (Pianura/Toscana/Apulia/PASTIS)
- **ALWAYS**: Checkpoint cada 30 min a Azure Blob durante training H100
- **ALWAYS**: LoRA rank 16 BF16 con FSDP + FlashAttention-2 + gradient checkpointing en H100
- **ALWAYS**: Validar VRAM presupuesto ANTES de lanzar training (Gemma 4: ~82 GB / 96; Qwen3.5: ~91 GB / 96)
- **ALWAYS**: Notebooks se commitean **con outputs poblados** (ejecutados end-to-end via papermill). Reproducibilidad se valida con `make notebooks-check`. NO usar `nbstripout` salvo on-demand.
- **ALWAYS**: Documentar licencia de cada dataset/modelo usado en `docs/licenses/`
- **NEVER**: Entrenar Foundation Model propio (AlphaEarth ya lo provee)
- **NEVER**: Modificar pesos DINOv3 (extractor frozen)
- **NEVER**: Random train/val split en datos espaciales (leakage garantizado)
- **NEVER**: Lanzar job H100 sin auto-shutdown timer
- **NEVER**: Subir checkpoints a Git (DVC + GCS/Azure Blob)
- **NEVER**: `pip install` directo — usar `poetry add --group ml`
- **NEVER**: Usar pandas cuando Polars puede hacerlo (10× más rápido en estos volúmenes)

## Project Structure

```
ml/
├── ingest/               # Scripts CLI: download_alphaearth.py, download_s2.py, download_pastis.py
├── extractors/           # DINOv3 wrapper, AlphaEarth tile loader
├── features/             # spectral_indices.py (17 índices), temporal.py (FFT, fenología), selection.py, fusion.py
├── analysis/             # correlations.py (Pearson/Spearman/VIF), eda_helpers.py
├── models/               # Arquitecturas custom: tsvit_wrapper.py, utae_wrapper.py, swin_unetr.py, gemma4_segmenter.py
├── train/                # train_baseline.py, train_segmentation.py, train_gemma4_lora.py, train_qwen3vl_lora.py
├── tune/                 # optuna_tune.py con PostgreSQL storage
├── ensemble/             # voting.py, bagging.py, stacking.py, blending.py
├── eval/                 # metrics.py (mIoU, F1-macro), plots.py, benchmarks/{agromind.py, geoanalyst.py}
├── agent/                # Google ADK agent (ver ml/agent/CLAUDE.md)
└── utils/                # mlflow_utils.py (@track_experiment), dvc_utils.py, spatial_cv.py, geo_io.py
```

## Decision Trees

```
¿Tipo de modelo a entrenar?
  Tabular sobre embeddings  → train_baseline.py + XGBoost (L4)
  Segmentación denso 2D     → train_segmentation.py + smp_lib (L4 dev, H100 final)
  Segmentación temporal     → train_segmentation.py + utae/tsvit/swin_unetr (H100)
  VLM con LoRA              → train_gemma4_lora.py / train_qwen3vl_lora.py (H100)
  LLM serving               → vLLM con configs ml/serving/qwen35.yaml (H100)

¿GPU a usar?
  Dev / debug              → laptop (4060/4080) o L4 spot
  Baselines + DINOv3       → L4 24GB
  Segmentación temporal    → H100 96GB ventana V2
  Gemma 4 26B-MoE LoRA     → H100 96GB ventana V3 (~24h en 3 noches)
  Qwen3-VL LoRA comparativa → H100 96GB ventana V4
  Qwen3.5-35B-A3B serving  → H100 96GB ventana V5

¿Cross-validation?
  Tabular IID              → KFold estratificado
  Datos espaciales         → SpatialBlockCV con buffers entre folds
  Series temporales        → TimeSeriesSplit por año

¿Métrica principal?
  Segmentación             → mIoU + F1-macro
  Clasificación parcela    → F1-macro + Cohen kappa
  VLM Q&A                  → AgroMind score + LLM-as-judge DeepEval
  Agente multi-step        → GeoAnalystBench pass rate
```

## Comandos

```bash
make train-l4 epic=E4 us=US-020          # spot L4 baselines
make train-h100 window=V3 script=train_gemma4_lora.py
make azure-h100-start                     # enciende VM spot
make azure-h100-stop                      # apaga (también auto-shutdown 12h)
make eval-agromind variant=gemini         # eval contra AgroMind
make eval-agromind variant=qwen35
make dvc-push                             # sube data versionada a gs://agrosat-dvc-remote
make mlflow-ui                            # MLflow UI puerto 5000
poetry add --group ml <pkg>
```

## Presupuesto VRAM validado (1×H100 NVL 96GB)

| Modelo | Pesos BF16 | LoRA + optim | KV cache | Activations | Overhead | Total |
|--------|-----------|--------------|----------|-------------|----------|-------|
| Gemma 4 26B-MoE LoRA | ~52 GB | ~1.5 GB | ~8 GB (b=2, ctx=32K) | ~15 GB (grad ckpt) | ~5 GB | **~82 GB** |
| Qwen3-VL-30B-A3B LoRA | ~60 GB | ~1.5 GB | ~10 GB | ~15 GB | ~5 GB | **~92 GB** (margen 4) |
| Qwen3.5-35B-A3B serving | ~70 GB | — | ~13 GB (ctx=64K) | — | ~8 GB | **~91 GB** |

Cualquier cambio de modelo o config requiere recalcular el presupuesto. Si excede 94 GB, abortar.

## QA Checklist ML

- [ ] Experimento registrado en MLflow con tags `data_version` + `code_version`
- [ ] Spatial CV usada en datos espaciales (no random)
- [ ] Stratified split por clase Y por región
- [ ] Datos versionados en DVC (.dvc files commiteados)
- [ ] Notebook ejecutable secuencialmente con papermill
- [ ] Notebook ejecutado end-to-end con papermill + commiteado con outputs poblados (HTML tables + PNG inline)
- [ ] Tests unitarios para extractors, features y métricas
- [ ] Plots interpretados (matriz confusión, ROC, PR, residuos espaciales)
- [ ] Atribución de licencia en `docs/licenses/DATA_LICENSE.md`
- [ ] Si tocó H100: log de uso de ventana en `docs/h100_log.md`
- [ ] Presupuesto VRAM validado antes de lanzar
- [ ] Auto-shutdown configurado en VM H100
