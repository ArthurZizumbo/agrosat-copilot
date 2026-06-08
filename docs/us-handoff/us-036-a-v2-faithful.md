# US-036-a v2 — Handoff (REDISENO FarSLIP fiel al paper)

**Status**: `code-complete` (T1+T2+T3+T4 implementados y verdes en CI; run H100 NO lanzado — pendiente Fase A captions + Fase B/C re-entrenamiento)
**Titulo**: REDISENO de FarSLIP FIEL al paper Li et al. 2025 (arXiv:2511.14901) — region-category multi-objeto + MPCL + caption global Gemma

> **PLANNING (2026-06-08)**. Plan detallado de 11 secciones en
> [docs/us-planning/us-036-a-v2-faithful.md](../us-planning/us-036-a-v2-faithful.md). Esta US v2 **SUPERA** a
> US-036-a v1 (`implemented`): v1 usa **1 etiqueta dominante por patch** (`dominant_class`) + contraste de
> **1 positivo** (`RegionCategoryAlignmentLoss` = `F.cross_entropy`), supervision empobrecida -> el modelo solo
> aprendio ~4 clases. v2 implementa el FarSLIP del paper: **multiples pares region-category por imagen** (via
> `ParcelIDs`) con **Multi-Positive Contrastive Loss (MPCL, ec. 4)** + **caption global rica `L_glo`** (ec. 1-2,
> InfoNCE) generada por **Gemma 4 31B multimodal local**. v1 NO se borra (queda como modo de ablacion).

> **T4 INTEGRACION (2026-06-08)**. T1/T2/T3 (commit 6c8978a) ya estaban; T4 integra los tres en el trainer +
> orquestador. `ml/farslip/distill.py` gana un modo aditivo `supervision="region_category"` (default `"dominant"`
> = v1 intacto) con `step_faithful_v2` (MPCL `L_loc` + InfoNCE `L_glo`), `set_category_prototypes` (mapeo PASTIS
> 1..18 -> [0,C) + lift 384->768 reusando el de v1) y `set_caption_encoder`/`_encode_captions`. El orquestador
> `scripts/run_us036a_v2_farslip_faithful.py` (Typer, comando `train`, flag `--supervision faithful_v2|dominant_v1`)
> carga captions, arma `RegionCategoryPairDataset(train)`, entrena, evalua F1/IoU por clase en el fold held-out,
> tabla honesta v1 vs v2 y un run MLflow `:5010` `farslip-faithful-v2` CERRADO con `data_version` (PASTIS-R +
> parquet captions) + `code_version`. **29 tests T4 verdes + suite `tests/ml/farslip/` sin regresion** (v1 sigue
> verde). El run real en H100 sigue pendiente.

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

**T1+T2+T3+T4 implementados (code-complete).** Lo nuevo de T4 (esta entrega):

- `ml/farslip/distill.py` (MOD, aditivo): `SupervisionMode` literal; campos config `supervision` (default
  `"dominant"`), `lambda_loc` (1.0), `temperature` (0.07), `use_global_caption_loss` (True). Modulos v2
  `MultiPositiveRegionCategoryLoss` + `GlobalImageTextLoss` instanciados siempre (baratos). Metodos nuevos:
  `set_category_prototypes(prototypes, pastis_class_ids)` (lift 384->768 reusando `_proto_to_clip_proj` + guarda el
  mapeo PASTIS id -> [0,C)); `set_caption_encoder`/`_encode_captions` (caption CLS al espacio CLS); `_map_region_cat_ids`
  (PASTIS crudo -> indice de banco, falla si id ausente); `step_faithful_v2(images, region_cat_ids, region_to_patch,
  caption_cls)` (gather `student_cls[region_to_patch]` -> `L_loc` MPCL; `L_glo` InfoNCE; `combine_losses`);
  `_forward_batch` despacha v1/v2 segun `supervision`. **La firma publica de `RegionCategoryAlignmentLoss` y de
  `step`/`set_text_prototypes`/`save_student` NO cambia** (back-compat v1 + tests existentes verdes).
- `scripts/run_us036a_v2_farslip_faithful.py` (NUEVO, Typer comando `train`): `run_faithful_v2(...)`,
  `eval_per_class_v2(...)` (patch-CLS vs banco de categorias, comparable a v1), `_v1_vs_v2_table_rows`,
  `_log_faithful_run` (run MLflow CERRADO), guards `_validate_pastis_root`/`assert_disjoint_folds`/
  `_require_captions_for_dataset`. Flag `--supervision faithful_v2` (default) vs `dominant_v1` (ablacion).
- `tests/ml/farslip/test_us036a_v2_orchestrator.py` (NUEVO, 29 tests, mocks CPU, cero GPU/red/PASTIS/Gemma).

Lo que YA EXISTE y se REUSA (no se reescribe):

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

1. ~~T1, T2, T3 en paralelo~~ **HECHO** (commit 6c8978a).
2. ~~T4 integracion en `distill.py` + orquestador~~ **HECHO** (esta entrega; 29 tests verdes + suite farslip sin regresion).
3. **H100 (pendiente)**: `nvidia-smi` ANTES; parar daemon idle-shutdown.
   - **Fase A — captions Gemma** (genera el parquet, `dvc add`+push). El orquestador de training NO genera
     captions; usa `ml/farslip/caption_cache.generate_captions_parquet` (T1) con el `GemmaCaptionClient`.
   - **Fase B/C — re-entrenamiento fiel + eval + best -> DVC** (comando exacto abajo).
4. Tras el run: anotar ADR-007 (salto de fidelidad 1-positivo -> MPCL + caption global), actualizar este handoff a
   `implemented` con `best`/`stop` reales, la tabla honesta v1 vs v2 (reporte aunque no mejore) y el muestreo de
   20 captions auditadas; cerrar el run MLflow (verificar FINISHED, no RUNNING).

## Comando exacto del re-entrenamiento fiel en H100 (cuando las captions esten listas)

```bash
# Fase A (una vez): materializa data/farslip/pastis_captions.parquet via Gemma local (T1), luego dvc add+push.
#   (lo provee T1: caption_cache.generate_captions_parquet con GemmaCaptionClient /api/chat think=false 896px)
# dvc add data/farslip/pastis_captions.parquet && dvc push

# Fase B/C: re-entrenamiento fiel (MPCL + L_glo) + eval por clase + run MLflow CERRADO.
poetry run python -m scripts.run_us036a_v2_farslip_faithful train \
    --run-name farslip-faithful-v2 \
    --supervision faithful_v2 \
    --lambda-loc 1.0 --temperature 0.07 \
    --n-epochs 30 --batch-size 64 --lr 1e-5 --seed 42 \
    --folds 1,2,3 --val-folds 4 \
    --pastis-root data/PASTIS-R \
    --captions-path data/farslip/pastis_captions.parquet \
    --output-dir checkpoints/farslip/faithful_v2 \
    --time-cap-hours 8.0 --mlflow-uri http://localhost:5010

# Ablacion v1 (mismo flag, path dominante): --supervision dominant_v1
# Ablacion L_glo:                          --no-global-loss
# Best -> DVC tras el run:
# dvc add checkpoints/farslip/faithful_v2/best.safetensors && dvc push
```

## Ajustes de firma (T4)

- `FarSLIPTrainerConfig`: campos NUEVOS `supervision` (`"dominant"` default), `lambda_loc`, `temperature`,
  `use_global_caption_loss`. **Aditivos, defaults compatibles** -> v1 (`run_us036a_farslip_full_incremental.py`)
  sigue funcionando sin tocarse (su `extra_params` y firma intactos).
- `FarSLIPDistillationTrainer`: metodos NUEVOS `set_category_prototypes`, `set_caption_encoder`, `step_faithful_v2`
  (y privados `_encode_captions`, `_map_region_cat_ids`, `_forward_batch`). `RegionCategoryAlignmentLoss`,
  `step`, `set_text_prototypes`, `save_student`, `adapt_patch_embed_to_n_channels` **sin cambios de firma**.
- El orquestador v2 entrena pasando un `DataLoader` con `collate_fn=collate_region_batch` (T2) a `trainer.train`,
  que en modo `region_category` consume `images`/`region_cat_ids`/`region_to_patch`/`caption_cls`. Nota: el batch
  v2 trae `captions` (list[str]) ya codificadas aguas arriba; si se quiere `L_glo` en el run real hay que poblar
  `batch["caption_cls"]` (pre-encodear con MiniLM o el text-encoder via `set_caption_encoder`) — hoy el collate de
  T2 no inyecta `caption_cls`, asi que en su ausencia `L_glo=0` y el run optimiza solo `L_loc` (ablacion). Ver
  "Pendiente para el run real con L_glo" abajo.

## Pendiente para activar L_glo en el run real (no bloquea T4)

El `collate_region_batch` (T2) entrega `captions` como `list[str]` pero no `caption_cls`. Para que `L_glo` sea
real en el run productivo hay dos opciones (ambas sin tocar T2/T3, ambas pre-encodean los CLS de caption UNA vez
fuera del loop, plan §7): (a) un `collate_fn` envoltorio que adjunte `caption_cls` pre-encodeado por patch_id, o
(b) `trainer.set_caption_encoder(encoder)` y encodear las captions del batch dentro de un wrapper del loader.
Mientras tanto el trainer degrada con elegancia: sin `caption_cls`, `step_faithful_v2` pone `L_glo=0` y entrena
`L_loc` (MPCL) — el corazon del rediseno — sin romperse. Documentado como riesgo controlado; el cierre debe
elegir (a)/(b) y reportarlo.
