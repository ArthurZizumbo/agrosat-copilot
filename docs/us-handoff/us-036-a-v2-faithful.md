# US-036-a v2 — Handoff (REDISENO FarSLIP fiel al paper)

**Status**: `planning` (plan completo escrito; CERO codigo escrito; run H100 NO lanzado)
**Titulo**: REDISENO de FarSLIP FIEL al paper Li et al. 2025 (arXiv:2511.14901) — region-category multi-objeto + MPCL + caption global Gemma

> **PLANNING (2026-06-08)**. Plan detallado de 11 secciones en
> [docs/us-planning/us-036-a-v2-faithful.md](../us-planning/us-036-a-v2-faithful.md). Esta US v2 **SUPERA** a
> US-036-a v1 (`implemented`): v1 usa **1 etiqueta dominante por patch** (`dominant_class`) + contraste de
> **1 positivo** (`RegionCategoryAlignmentLoss` = `F.cross_entropy`), supervision empobrecida -> el modelo solo
> aprendio ~4 clases. v2 implementa el FarSLIP del paper: **multiples pares region-category por imagen** (via
> `ParcelIDs`) con **Multi-Positive Contrastive Loss (MPCL, ec. 4)** + **caption global rica `L_glo`** (ec. 1-2,
> InfoNCE) generada por **Gemma 4 31B multimodal local**. v1 NO se borra (queda como modo de ablacion).

## Por que el rediseno (problema detectado por el usuario)

- v1 colapsa cada patch a su clase dominante (filtro 3:1 + `dominant_class`) -> tira la riqueza multi-parcela.
- v1 `RegionCategoryAlignmentLoss.forward` usa `F.cross_entropy(logits, targets)` = **1 positivo** contra N
  prototipos. Con varias regiones de la misma categoria, v1 trata las extra como NEGATIVOS (error semantico que
  el paper resuelve con MPCL: positivos mutuos).
- El paper combina `L_glo` (caption global) + `L_loc` (region-category MPCL); v1 no tiene `L_glo` real (solo un
  `loss_aux` coseno placeholder).

## Decisiones ya acordadas con el usuario (INPUT, no re-decidir)

1. **Datos SOLO PASTIS-R frances real** (`data/PASTIS-R`); imagen = composite pico-NDVI 4-banda realzado. Cero italiano/sintetico.
2. **`L_glo`**: caption global por patch via **Gemma 4 31B local (Ollama `gemma4:31b-it-q8_0`)**, API `/api/chat`
   con **`"think": false`** (CRITICO: sin esto razona infinito -> timeout; con esto ~3.1 s/caption). Imagen
   composite realzado p2-p98 a **896 px** PNG (128 px no sirve para el proyector de vision de Gemma).
3. **Input a Gemma sin fuga**: imagen RGB 896 px + clases presentes + composicion espacial + conteo/area +
   tile MGRS + fecha + fenologia tipica (US-033). **PROHIBIDO**: NDVI numerico del patch, AlphaEarth, la
   etiqueta literal.
4. **`L_loc`**: cada parcela (`ParcelIDs_<pid>.npy`) -> su categoria PASTIS; multiples por patch; texto =
   prototipo US-033 o plantilla "imagen satelital de {cultivo}"; MPCL multi-positivo.
5. **Loss**: reescribir `RegionCategoryAlignmentLoss` (`ml/farslip/distill.py:168`) a MPCL (ec. 4); mantener
   `L_glo` (ec. 1-2). Combinar `L_glo + lambda_loc * L_loc` (paper Tabla 3 = mejor config).
6. **AlphaEarth NO se usa aqui** (va al ensamble E-b US-042).

## Estado de la implementacion

**CERO codigo escrito.** Esta entrega es SOLO planning (este handoff + el plan). Lo que YA EXISTE y se REUSA
(no se reescribe):

- `ml/farslip/distill.py`: `FarSLIPDistillationTrainer`, `PatchDistillationLoss` (`L_dis`, NO se toca),
  `RegionCategoryAlignmentLoss` (se reescribe a MPCL manteniendo firma), `set_text_prototypes` (reproyecta
  384->768), `adapt_patch_embed_to_n_channels` (3->4), `save_student`.
- `ml/farslip/pastis_pair_dataset.py`: `peak_ndvi_composite` (composite pico-NDVI 4-banda, REUSAR), `active_classes`.
- `ml/ingest/pastis_loader.py`: `load_pastis_patch` (S2 + semantic + instance + dates + fold + tile).
- `ml/data/pastis_filter.py`: `PastisFilter` (`mode="dominance_ratio"`).
- `ml/features/phenology_class_prototypes.py`: `load_class_prototype_embeddings` (18 prototipos MiniLM-384 Gemini real, US-033, DVC — solo LEER).
- `data/cache/gee/alphaearth_pastis_parcels_2019_85951_enriched.parquet`: existe pero **NO se usa** en este rediseno.

## Write-set (NUEVO en v2)

**Crear**:
- `ml/farslip/caption_generator.py` (cliente Gemma `/api/chat` think=false, prompt builder anti-fuga, realce p2-p98 -> PNG 896)
- `ml/farslip/caption_cache.py` (materializa `data/farslip/pastis_captions.parquet`, `audit_captions`, `load_captions`)
- `ml/farslip/region_category_dataset.py` (`RegionCategoryPairDataset`, `extract_regions`, `collate_region_batch`)
- `ml/farslip/mpcl_loss.py` (`MultiPositiveRegionCategoryLoss` ec.4, `GlobalImageTextLoss` ec.1-2)
- `scripts/run_us036a_v2_farslip_faithful.py` (orquestador Typer: `generate-captions` + `train`)
- `tests/ml/farslip/test_mpcl_loss.py`, `test_region_category_dataset.py`, `test_caption_generator.py`, `test_us036a_v2_orchestrator.py`
- `docs/us-planning/us-036-a-v2-faithful.md`, `docs/us-handoff/us-036-a-v2-faithful.md` (esta entrega)

**Modificar (minimo)**:
- `ml/farslip/distill.py` (MPCL en `RegionCategoryAlignmentLoss` manteniendo firma + back-compat `|P(i)|=1`==CE; integrar `GlobalImageTextLoss` y `caption_glo`; campos config `lambda_loc`, `supervision`). Flag `supervision="dominant"` preserva el path v1.

**NO tocar**: `PatchDistillationLoss`, `save_student`, `set_text_prototypes`, `adapt_patch_embed_to_n_channels`,
`cap_vocabulary.yaml`, `weights_uri` de `farslip_extractor`, pesos frozen (teacher CLIP, MiniLM), el parquet
US-033 + `.dvc`, `ml/farslip/incremental_curriculum.py` y `scripts/run_us036a_farslip_full_incremental.py` (v1).

## Sub-tareas PARALELAS (write-sets disjuntos)

| # | Sub-tarea | Write-set | Paralela | Depende |
|---|-----------|-----------|----------|---------|
| **T1** | Generador captions Gemma | `caption_generator.py`, `caption_cache.py`, `test_caption_generator.py` | **SI** | US-033 (leer) |
| **T2** | Dataset region-category | `region_category_dataset.py`, `test_region_category_dataset.py` | **SI** | composite US-036 (existe) |
| **T3** | MPCL loss + L_glo | `mpcl_loss.py`, `test_mpcl_loss.py` | **SI** | paper ec.1-4 |
| **T4** | Integracion trainer + orquestador | `distill.py` (MOD), `run_us036a_v2_farslip_faithful.py`, `test_us036a_v2_orchestrator.py` | **NO** | T1+T2+T3 |

> **Paralelizables ahora: T1, T2, T3** (3 agentes ml, ramas/worktrees separados, write-sets disjuntos; `distill.py`
> lo toca SOLO T4 -> sin colision). **T4 secuencial** (rebase tras merge de T1-T3).

## Dominios tocados

- [x] **ml** (4 modulos nuevos + `distill.py` MOD + orquestador + tests)
- [x] **H100** (Fase A captions Gemma ~2 h + Fase B re-entrenamiento ~0.5-1 h)
- [ ] backend - [ ] frontend - [ ] agent - [ ] db - [ ] dagster - [ ] infra
- [x] docs (solo estos dos `.md`)

Skills: `agrosat-ml-segmentation` (FarSLIP/distill), `agrosat-llm-finetuning`, `agrosat-ml-evaluation`, `agrosat-dvc-mlflow`, `agrosat-azure-h100`, `agrosat-testing`.

## H100 / MLOps (resumen)

- `nvidia-smi` ANTES; parar daemon idle-shutdown. 1 GPU 96 GB.
- **Fase A**: captions Gemma (`/api/chat`, think=false, 896 px) ~2433 patches x ~3.1 s ~ **2 h**; `resume=True`;
  parquet -> `dvc add`+push. Gemma q8_0 ~33 GB.
- **Fase B**: re-entrenamiento `farslip-faithful-v2` (ViT-B/16 BF16 < 16 GB) ~0.5-1 h; M=30-40 epochs.
- **Fase C**: eval F1/IoU por clase + tabla v1 vs v2; best `checkpoints/farslip/faithful_v2/best.safetensors`
  -> `dvc add`+`dvc push` desde la VM (ADC OK, memoria `session-creds-h100-gcp-available`).
- VRAM pico si A+B concurren `< 50 GB / 96 GB`.
- MLflow `:5010` experiment `farslip`, run `farslip-faithful-v2`, tags `data_version`+`code_version`, params
  (`lambda_loc`, `tau=0.07`, `caption_model`, `prompt_version`, `supervision`), metricas `loss_glo`/`loss_loc`/
  `loss_total` + F1/IoU por clase. **Run CERRADO** (gotcha subprocess RUNNING).

## Riesgos principales

1. **R-MPCL-WRONG (ALTA)**: MPCL mal implementada -> peor que v1. Mitig: test golden `|P(i)|=1`==CE, gradiente al centroide, `logsumexp` estable, excluir el ancla de sus positivos.
2. **R-CAPTION-LEAK (ALTA)**: fuga circular en captions -> metricas mienten. Mitig: prompt prohibe NDVI/AlphaEarth/etiqueta; `audit_captions` regex; muestreo manual 20.
3. **R-GEMMA-SLOW (MEDIA)**: sin `think=false` timeout infinito; Ollama caido. Mitig: think=false OBLIGATORIO, `resume=True`, retries.
4. **R-REGION-CROP (ALTA)**: PASTIS no tiene crops por region; CLS de patch compartido por sus regiones = desviacion del paper. Mitig: documentada (plan §2.3, §10); v2.1 RoIAlign en backlog.
5. **R-NO-IMPROVE (MEDIA)**: si v2 no mejora -> **reporte honesto** (patron ADR-007 §3); FarSLIP base learner del stacking E-b. Tabla v1 vs v2 explicita. NO maquillar.
6. **R-DISTILL-BREAK (MEDIA)**: modificar `distill.py` rompe v1. Mitig: flag `supervision` (v1 disponible), firma preservada, test `test_v1_path_still_works`.

## Criterio de aceptacion (resumen)

Rediseno FIEL al paper Li et al. 2025 sobre **PASTIS-R frances real**: dataset **region-category multi-objeto**
(N>1 regiones/patch via `ParcelIDs`, `mean_regions_per_patch > 1`), loss **MPCL multi-positivo** (ec. 4, `P(i)`
= regiones de la misma categoria; equivalente a CE cuando `|P(i)|=1`) **combinada con `L_glo`** (ec. 1-2,
InfoNCE CLS imagen <-> CLS **caption global Gemma 4 31B** generada con `/api/chat` think=false a 896 px,
**sin fuga** auditada); composite pico-NDVI 4-banda; `V_i^r`=CLS (§4.3); captions cacheadas a parquet+DVC;
anti-leakage train/eval por folds PASTIS oficiales; eval F1/IoU por clase + **tabla honesta v1 vs v2** (reporte
aunque no mejore); un run MLflow `:5010` `farslip-faithful-v2` con `data_version`+`code_version`+params+metricas
CERRADO; best -> DVC (insumo US-037); write-set disjunto que NO rompe el path v1; AlphaEarth NO usado;
`make check` limpio + tests ml verdes (cobertura >=70%, Gemma/trainer/PASTIS mockeados).

**Epic**: E5 (Modelos Alternativos / Comparativa) - **Avance**: A5 (ensamble) / S7
**Sprint**: S7 - **Owner**: Arthur Zizumbo (ML/MLOps) - **Prioridad**: **P0 — MAXIMA del lote** (FarSLIP camino principal; PRIMERA en cola H100)
**Rama**: `feature/E5-US-036-a-v2-farslip-faithful-mpcl`
**Plan**: [docs/us-planning/us-036-a-v2-faithful.md](../us-planning/us-036-a-v2-faithful.md)
**Depende de**: US-036-a v1 (`implemented`, se reusa/supera), US-033 (`implemented`, parquet pheno solo leer), Gemma 4 31B local (Ollama VM H100).
**Insumo de**: US-037 (eval downstream FarSLIP v2 vs AlphaEarth 0.233 / FarSLIP-v1 0.163).

## Proximo paso

1. Lanzar **T1, T2, T3 en paralelo** (3 agentes ml, write-sets disjuntos) con sus tests (mocks: cero GPU/red/PASTIS/Gemma).
2. **T4** tras merge de T1-T3: integrar en `distill.py` (flag `supervision`, `L_glo`+`L_loc`) + orquestador; `make check` + tests verdes.
3. **H100**: `nvidia-smi`; Fase A captions (`dvc add`+push del parquet); Fase B re-entrenamiento; Fase C eval por clase + tabla v1 vs v2; best -> DVC; cerrar run MLflow.
4. Anotar ADR-007 (salto de fidelidad); actualizar este handoff a `implemented` con resultados reales y el reporte honesto v1 vs v2.
