# US-036-a v2 — REDISENO de FarSLIP FIEL al paper Li et al. 2025 (region-category multi-objeto + MPCL + L_glo)

**Status**: planning · **Epic**: E5 (Modelos Alternativos / Comparativa) · **Avance del curso**: A5 (ensamble) / S7
**Sprint**: S7 · **Owner**: Arthur Zizumbo (ML/MLOps) · **Prioridad**: **P0 — MAXIMA del lote** (FarSLIP es el camino principal; PRIMERA en cola H100)
**Rama sugerida**: `feature/E5-US-036-a-v2-farslip-faithful-mpcl`
**Plan vigente**: `context/RefinamientoPlaneacionAgroSatCopilot_v8.md` (FarSLIP, ADR-009)
**Paper-faro**: Li et al. (2025), "FarSLIP: Discovering Effective CLIP Adaptation for Fine-Grained Remote Sensing Understanding", arXiv:2511.14901 (secciones 3.3, 4.1, 4.3).
**ADR relacionado**: [`docs/decisions/ADR-007-farslip-fidelity-paper.md`](../decisions/ADR-007-farslip-fidelity-paper.md) (contrato de fidelidad/desviaciones — esta US lo ACTUALIZA hacia mayor fidelidad).

**Depende de**:
- **US-036-a v1** (`implemented`): orquestador incremental por cardinalidad (`scripts/run_us036a_farslip_full_incremental.py`, `ml/farslip/incremental_curriculum.py`) y dataset por patch (`ml/farslip/pastis_pair_dataset.py`). **Esta US v2 NO lo borra**: lo SUPERA con la supervision region-category multi-objeto y conserva el esqueleto del curriculum como modo de ablacion.
- **US-033** (`implemented`): `data/features/phenology_class_prototypes_pastis.parquet` (18 prototipos MiniLM-384, descripciones ES via Gemini 3.5 Flash REAL, `class_id` directos 1..18). Solo se LEE/FILTRA, jamas se regenera.
- Infraestructura reutilizable existente: `ml/ingest/pastis_loader.py`, `ml/data/pastis_filter.py` (`PastisFilter` `mode="dominance_ratio"`), `ml/farslip/distill.py` (`FarSLIPDistillationTrainer`, `set_text_prototypes`, `adapt_patch_embed_to_n_channels`, `save_student`), `ml/farslip/pastis_pair_dataset.py` (`peak_ndvi_composite`, `load_pastis_patch`).
- Gemma 4 31B multimodal local via Ollama (`gemma4:31b-it-q8_0`) en la VM H100 (metodo validado: `/api/chat` con `"think": false`).

**Insumo de**: **US-037** (eval downstream FarSLIP vs AlphaEarth 0.233 / FarSLIP-v1 0.163). El checkpoint best del rediseno alimenta esa evaluacion.

---

> **PROBLEMA QUE RESUELVE (detectado por el usuario)**. El FarSLIP v1 (US-036-a v1) usa **1 ETIQUETA por
> imagen** (clase dominante del patch via `dominant_class`) -> supervision empobrecida: el contraste es un
> `F.cross_entropy` de **1 positivo** contra N prototipos, y el dataset descarta toda la riqueza multi-parcela
> del patch (filtro 3:1 + `dominant_class` colapsan ~2433 patches a una sola etiqueta cada uno). Resultado
> empirico: el modelo solo aprendio ~4 clases. El paper FarSLIP (Li et al. 2025) **NO** opera asi: cada
> imagen tiene (a) una **caption global rica `L_glo`** (ec. 1-2, InfoNCE imagen-texto) y (b) **MULTIPLES pares
> region-category `L_loc`** (ec. 3-4, uno por objeto/region) con **Multi-Positive Contrastive Loss (MPCL)**:
> todas las regiones de la misma categoria son positivos mutuos. Hay que rehacerlo fiel al paper.

> **DECISIONES YA ACORDADAS (INPUT, no re-decidir)**:
> 1. **Datos: SOLO PASTIS-R frances real** (`data/PASTIS-R`). Imagen del par = composite del pico NDVI por
>    patch (RGB+NIR 4 bandas) realzado. Cero italiano/sintetico/placeholder.
> 2. **`L_glo` (caption global por patch)**: generada por **Gemma 4 31B multimodal LOCAL** (Ollama,
>    `gemma4:31b-it-q8_0`). Metodo VALIDADO: API HTTP `/api/chat` con `"think": false` (CRITICO: sin
>    `think=false` el modelo razona infinito -> timeout; con `think=false` da ~3.1 s/caption multimodal). NO
>    `/api/generate`, NO CLI. Imagen: composite pico-NDVI realzado (percentil 2-98 por canal) a **896 px** PNG
>    (128 px NO funciona, demasiado pequeno para el proyector de vision de Gemma).
> 3. **Input a Gemma por caption (sin data leakage)**: imagen RGB 896 px + clases reales presentes (de la
>    mascara) + composicion espacial (norte/centro/sur) + conteo/fragmentacion (n parcelas, area) + contexto
>    geografico (tile MGRS + fecha del composite, de `metadata.geojson`) + fenologia TIPICA del cultivo
>    (conocimiento de clase, US-033). **PROHIBIDO en el texto**: valores numericos de NDVI/indices calculados
>    del patch input (FUGA CIRCULAR); AlphaEarth (va al ensamble E-b US-042, NO al texto); y NO filtrar la
>    etiqueta como "respuesta" (la caption describe, no dice "la clase es X").
> 4. **`L_loc` (region-category)**: cada parcela del patch (via `ANNOTATIONS/ParcelIDs_<pid>.npy`) -> su
>    categoria PASTIS real. MULTIPLES por patch. Texto de cada region = prototipo/plantilla de SU categoria
>    (reusar prototipos US-033 o plantilla "imagen satelital de {cultivo}"). MPCL multi-positivo.
> 5. **Loss**: reescribir `RegionCategoryAlignmentLoss` (`ml/farslip/distill.py:168`, hoy `F.cross_entropy` =
>    1 positivo) a **MPCL** (ec. 4: para cada region i, los positivos `P(i)` son TODAS las regiones que
>    comparten su categoria; log-sum sobre positivos). Mantener `L_glo` (InfoNCE global imagen-caption, ec.
>    1-2). El paper (Tabla 3) confirma que `L_glo + L_loc` juntas dan el mejor resultado.
> 6. **AlphaEarth NO se usa aqui** (decidido: va al ensamble E-b US-042).

---

## 0. Ecuaciones del paper (transcritas, fuente de verdad del rediseno)

Pagina 4 del PDF. `S(.,.)` = similitud coseno; `tau` = temperatura aprendible; `[CLS]` = token CLS.

**`L_glo` (global image-text alignment, InfoNCE, ec. 1-2)** — para un batch de N pares imagen-caption:

```
L_glo = (1/2) (L_{I->T} + L_{T->I})
L_{I->T} = -(1/N) sum_{i=1..N} log [ exp(S(V_i, T_i)/tau) / sum_{j=1..N} exp(S(V_i, T_j)/tau) ]
```

donde `V_i`, `T_i` son los embeddings `[CLS]` de imagen y texto (caption global) del i-esimo par. `L_{T->I}`
es simetrico. El texto que el paper alinea aqui es la **caption corta** de su dataset (en nuestro caso, la
caption global de Gemma).

**`L_loc` (region-category, Multi-Positive CL = MPCL, ec. 3-4)** — para un batch de M pares region-category:

```
L_loc = (1/2) (L_{R->C} + L_{C->R})
L_{R->C} = (1/M) sum_{i=1..M} [ -(1/|P(i)|) sum_{j in P(i)} log ( exp(S(V_i^r, T_j^c)/tau) / sum_{k=1..M} exp(S(V_i^r, T_k^c)/tau) ) ]
```

donde `V_i^r` es el embedding visual de la i-esima REGION (paper §4.3 Takeaway-1: **CLS token**, NO RoI ni
pooled patches — el CLS preserva la coherencia semantica de CLIP); `T_j^c` es el embedding textual `[CLS]` de
la j-esima categoria; y `P(i)` = conjunto de indices que **comparten la misma categoria** que la region i
(positivos mutuos). `L_{C->R}` es simetrico (categoria -> regiones). Esto es Supervised Contrastive /
MPCL (refs [21,47] del paper). **Clave: con 1 positivo por ancla, MPCL degenera a InfoNCE de 1 positivo =
exactamente la `RegionCategoryAlignmentLoss` v1** (que es por que v1 es un caso particular pobre).

**`L_dis` (patch-to-patch self-distillation, ec. 5)** — fuera de alcance del rediseno (es stage-2 RS5M del
paper; nuestro `PatchDistillationLoss` ya lo implementa y NO se toca). El paper (Tabla 3) muestra que
`L_glo + L_loc` ya es el mejor sin `L_dis`; `L_dis` aporta marginal y requiere image-caption a gran escala que
no tenemos.

**Combinacion (objetivo del rediseno, alineado a Tabla 3 fila `L_glo + L_loc`)**:

```
L_total = L_glo + lambda_loc * L_loc        (lambda_loc por defecto 1.0; ablacion en Seccion 1.2)
```

---

## 1. Criterios de aceptacion verificables

### 1.1 Criterios funcionales (binarios)

| # | Criterio | Verificacion |
|---|----------|--------------|
| AC-1 | **Region-category MULTI-OBJETO (N>1 por patch)**: cada item del dataset expone `>= 1` par region-category derivado de `ANNOTATIONS/ParcelIDs_<pid>.npy` (una entrada por parcela/instancia agronomica con clase en 1..18), NO una unica etiqueta dominante. El promedio de regiones por patch es `> 1` sobre PASTIS-R. | Log del dataset: `mean_regions_per_patch > 1.0`, `n_total_regions >> n_patches`. Test: para un patch sintetico con 3 parcelas de 2 clases distintas, el dataset devuelve 3 region-category (no 1). |
| AC-2 | **MPCL multi-positivo (ec. 4)**: `MultiPositiveRegionCategoryLoss` implementa `P(i)` = todas las regiones del batch que comparten categoria con la region i; el log-sum corre sobre `P(i)`; con `\|P(i)\|=1` reproduce numericamente la InfoNCE de 1 positivo (equivalencia con la v1 verificada en test). | Test: (a) con 1 positivo por categoria el loss == `F.cross_entropy` (v1) a `< 1e-5`; (b) con 3 positivos de la misma categoria, el loss difiere de tratar 2 como negativos (caso degenerado v1) y es menor cuando los 3 estan alineados. |
| AC-3 | **`L_glo` (InfoNCE imagen-caption, ec. 1-2) presente y combinada**: el trainer optimiza `L_total = L_glo + lambda_loc * L_loc`. `L_glo` alinea el CLS visual del composite con el CLS textual de la **caption global de Gemma** (encoder de texto frozen). | Log/MLflow: metricas separadas `loss_glo`, `loss_loc`, `loss_total`; param `lambda_loc`. Test: ablacion programatica `lambda_loc=0` -> `loss_total == loss_glo`. |
| AC-4 | **Captions Gemma SIN fuga circular**: las captions globales se generan con Gemma 4 31B local (`/api/chat`, `think=false`, imagen 896 px) y NO contienen valores numericos de NDVI/indices calculados del patch, NO mencionan AlphaEarth, NO emiten la etiqueta como "la clase es X". | Auditoria de captions (Seccion 6): regex anti-leakage sobre el parquet de captions (cero matches de patrones NDVI/`alphaearth`/`la clase es`); muestreo manual de 20 captions documentado en el handoff. |
| AC-5 | **Captions cacheadas a parquet + DVC**: `data/farslip/pastis_captions.parquet` (`patch_id`, `caption_glo`, `model`, `prompt_version`, `tile`, `composite_date`, `present_class_ids`, `gen_seconds`) materializado y `dvc add`+push; el training LEE el parquet, no re-llama a Gemma. | `.dvc` committeado; el orquestador de training falla rapido si falta una caption para un patch del split (no genera silenciosamente). |
| AC-6 | **Imagen del par = composite pico-NDVI 4-banda realzado**: el composite reusa `peak_ndvi_composite` (US-036, B02/B03/B04/B08), con realce percentil 2-98 por canal SOLO para el PNG que ve Gemma; el tensor de training mantiene su normalizacion [0,1]. | Test: el composite de training es identico a v1 (`peak_ndvi_composite`); el PNG-896 es una vista derivada, no altera el tensor. |
| AC-7 | **`V_i^r` = CLS token (paper §4.3 Takeaway-1)**, NO RoI ni pooled patches. La region visual se obtiene pasando el patch por el student y tomando el CLS; la categoria de la region viene de la mascara `ParcelIDs`+`semantic`. (Caveat documentado: PASTIS es un solo patch por imagen, no crops independientes por region — ver R-REGION-CROP en Seccion 9). | Codigo: `region_visual = student_cls`; test de la firma. Documentado en handoff como desviacion controlada. |
| AC-8 | **Anti-leakage train/eval (spatial CV oficial)**: train sobre folds PASTIS `(1,2,3)`, eval held-out fold `(4)` (o `(5)`); el orquestador RECHAZA con ValueError si los folds de eval solapan los de train. Las captions de eval se generan igual (mismo prompt) pero NO entran al contraste de train. | Test: folds solapados -> ValueError. Log: `train_folds`, `val_folds` disjuntos. |
| AC-9 | **Eval por clase (F1/IoU) + comparacion honesta vs v1**: se reporta F1/IoU por clase (clasificacion del par via CLS<->prototipo, igual que v1 para comparabilidad) sobre el held-out, y una tabla `v1 (1-etiqueta) vs v2 (region-category+MPCL+L_glo)`. Se reporta aunque NO mejore (Seccion 9 R-NO-IMPROVE). | Artefacto MLflow: tabla por clase + delta vs v1. `n_classes_well_resolved` (F1>=0.50). |
| AC-10 | **MLflow `:5010` experiment `farslip`**: un run del rediseno (`farslip-faithful-v2`) con `data_version` (DVC del PASTIS-R + del parquet de captions) + `code_version` (git SHA) + params (`lambda_loc`, `tau`, `n_categories`, `n_in_channels=4`, `caption_model`, `prompt_version`) + metricas `loss_glo`/`loss_loc`/`loss_total` por step + F1/IoU por clase. Run CERRADO. | UI `:5010`; tags presentes; run FINISHED (gotcha subprocess RUNNING). |
| AC-11 | **Run real en H100**: `nvidia-smi` ANTES, parar daemon idle-shutdown; (a) generacion de captions Gemma (~2 h, coexiste con/precede al training), (b) re-entrenamiento FarSLIP; checkpoint best -> `dvc add`+`dvc push` desde la VM (ADC OK). | Log `nvidia-smi`; checkpoint en `checkpoints/farslip/faithful_v2/best.safetensors`; `.dvc` committeado; `dvc push` confirmado. |
| AC-12 | **`make check` limpio + tests verdes** (type hints, docstrings Google-style ingles, prosa ES, sin emojis, structlog, Polars no pandas; Gemma client y trainer pesado mockeados en CI). | `make check` (ruff + secrets + i18n) + `pytest tests/ml/farslip/` verde. |
| AC-13 | **Fidelidad documentada (ADR-007 actualizado)**: la tabla "que del paper implementamos vs desviaciones" (Seccion 10) refleja el salto de 1-positivo -> MPCL multi-positivo + caption global multimodal real. | Seccion 10 de este plan + nota en ADR-007 (en el cierre, no en planning). |

### 1.2 Metricas verificables (golden values / umbrales)

Tests SIN GPU/red/Gemma/PASTIS real (todo mockeado). Defaults parametrizables:

- **MPCL equivalencia (golden)**: para un batch con exactamente 1 region por categoria (`|P(i)|=1`),
  `MultiPositiveRegionCategoryLoss(logits, cat_ids)` == `F.cross_entropy(logits, cat_ids)` a `atol=1e-5`.
  Esto PRUEBA que MPCL generaliza la v1 (no la contradice).
- **MPCL multi-positivo (golden)**: para un batch con categorias repetidas, el gradiente respecto al ancla
  empuja hacia el centroide de SUS positivos (verificable con un caso analitico de 2 categorias x 3 regiones).
- **`mean_regions_per_patch` (umbral)**: `> 1.0` sobre PASTIS-R (esperado 2-6 segun fragmentacion; la mediana
  de parcelas por patch PASTIS es > 1 por construccion del dataset panoptico).
- **Caption anti-leakage (golden)**: cero matches de `(?i)ndvi\s*[=:]\s*[-+]?\d`, `(?i)alphaearth`,
  `(?i)satellite\s+embedding`, `(?i)la\s+clase\s+es\b`, `(?i)the\s+class\s+is\b` en `caption_glo`.
- **Caption latencia (sanity, no test CI)**: `~3.1 s/caption` con `think=false`; `~2433 patches ~ 2 h` total.
  Sin `think=false` -> timeout (NO usar).
- **Imagen Gemma (golden)**: PNG de lado `896`; realce percentil 2-98 por canal; 3 canales RGB (B04,B03,B02
  como R,G,B). El tensor de training sigue siendo 4-banda [0,1] (sin realce).
- **`tau` y `lambda_loc` (defaults)**: `tau=0.07` (paper §3.3), `lambda_loc=1.0` (paper Tabla 3 combina sin
  reponderar). Ablacion `lambda_loc in {0.0, 0.5, 1.0}` como experimentos MLflow separados.
- **VRAM (estimado)**: ViT-B/16 batch=64 BF16 `< 16 GB`; Gemma 31B q8_0 via Ollama `~33 GB` (coexisten en
  96 GB). Si concurren: pico `< 50 GB`.
- **Spatial CV (golden)**: `set(train_folds) & set(val_folds) == set()` o ValueError.

---

## 2. Arquitectura del rediseno

### 2.1 Flujo end-to-end

```
  data/PASTIS-R/ (frances real)                         data/features/phenology_class_prototypes_pastis.parquet
  DATA_S2/S2_<pid>.npy (T,10,128,128)                   (18 filas, class_id 1..18, emb_000..383, Gemini real, DVC)
  ANNOTATIONS/TARGET_<pid>.npy (3,128,128) [canal0=semantic]            |
  ANNOTATIONS/ParcelIDs_<pid>.npy (instancias de parcela)               | load_class_prototype_embeddings()
  metadata.geojson (TILE MGRS, Fold, dates-S2)                          | (LEER, NO regenerar)
            |                                                            v
            |  peak_ndvi_composite (US-036, B02/B03/B04/B08) ----+   T_c^j: prototipo textual por categoria
            |                                                     |   (US-033 MiniLM-384 -> reproyectado 768,
            v                                                     |    o plantilla "imagen satelital de {cultivo}")
   composite 4-banda [0,1] (tensor de training)                  |
            |                                                     |
            +--> realce p2-p98 -> PNG 896 RGB ---> [GEMMA 4 31B local Ollama /api/chat think=false]
            |        (NO toca el tensor)                |  prompt: clases presentes + composicion N/C/S +
            |                                           |  n parcelas/area + tile MGRS + fecha + fenologia tipica
            |                                           |  PROHIBIDO: NDVI numerico, AlphaEarth, "la clase es X"
            |                                           v
            |                              caption_glo  --> data/farslip/pastis_captions.parquet (DVC, cacheado)
            |                                           |
            v                                           v
  ============================== TRAINER (1 run, NO incremental por defecto) ==============================
   por batch de B patches:
     student(composite 4-banda) -> CLS visual V (B,768) + patch tokens
     # L_glo (ec.1-2): InfoNCE entre V (CLS imagen) y T_glo (CLS de caption_glo, text encoder frozen)
     L_glo = info_nce_symmetric(V, T_glo, tau)
     # L_loc (ec.3-4, MPCL): por cada patch, sus N regiones (ParcelIDs) -> categoria
     #   V_i^r = CLS visual del patch de la region i (paper §4.3: CLS, no RoI)
     #   T_j^c = prototipo textual de la categoria j
     #   P(i) = regiones del batch que comparten categoria con i  (multi-positivo)
     L_loc = mpcl_symmetric(region_cls, region_cat_ids, category_protos, tau)
     L_total = L_glo + lambda_loc * L_loc
     backward; AdamW BF16; grad_accum; cosine warmup
   eval_per_class(student, val_fold) -> F1/IoU por clase (CLS<->prototipo) ; tabla v1 vs v2
   best -> checkpoints/farslip/faithful_v2/best.safetensors -> DVC
```

### 2.2 Region-category: como se extraen N regiones por patch

1. `ParcelIDs_<pid>.npy` (H,W): id de instancia por pixel (0 = sin parcela). `TARGET_<pid>.npy[0]` (H,W):
   clase semantica por pixel.
2. Por cada `parcel_id != 0`: la categoria de la region = clase mayoritaria de `semantic` dentro de esa
   instancia (las parcelas PASTIS son monocultivo, asi que la mayoritaria es la verdadera; se descarta la
   instancia si su clase mayoritaria es Background 0 o Void 19).
3. Cada region produce un par `(patch_id, region_id, category_id PASTIS)`. **Multiples por patch**.
4. Filtro de area minima (parametrizable, p.ej. `>= 16 px`) para descartar slivers que no aportan senal y
   estabilizan el MPCL.
5. La categoria se indexa contra el banco de prototipos `category_protos` (subset del parquet US-033 por
   `class_id`, orden canonico). `n_regions` del paper aqui es conceptual (cada parcela es una "region"); NO
   hay `n_regions=3` geografico italiano.

### 2.3 `V_i^r` = CLS (decision del paper) y su caveat en PASTIS

Paper §4.3 (Tabla 1, Takeaway-1): el CLS token supera a RoI-embedding y pooled-patches para region-category
en RS, porque preserva la coherencia semantica de CLIP. **Desviacion controlada (R-REGION-CROP)**: en MGRS-200k
cada region es un objeto con bbox; en PASTIS un patch tiene varias parcelas pero NO crops independientes por
parcela. Opcion adoptada (v2.0): **todas las regiones de un patch comparten el CLS del patch completo**, y la
senal multi-objeto entra por (a) `L_glo` con caption rica multi-clase y (b) el MPCL que agrupa regiones de la
misma categoria ACROSS patches. Opcion futura (v2.1, backlog): RoIAlign por bbox de parcela para un CLS por
region (mas fiel pero mas costoso y con el riesgo RoI<CLS que el propio paper reporta). Se documenta la
eleccion y su impacto en la Seccion 10.

---

## 3. Plan de implementacion — archivos EXACTOS a crear/modificar

### 3.1 Crear

| Archivo | Contenido | GPU/red |
|---------|-----------|---------|
| `ml/farslip/caption_generator.py` | Cliente Gemma 4 local: `GemmaCaptionClient` (HTTP `/api/chat`, `think=false`, timeout, retries), `build_caption_prompt(...)` (ensambla clases presentes + composicion espacial + conteo/area + tile/fecha + fenologia tipica, SIN NDVI numerico/AlphaEarth/etiqueta), `composite_to_png896(composite4)` (realce p2-p98 -> PNG 896 RGB base64), `generate_caption(...)`. Logica pura (prompt builder, realce) testeable sin red; cliente HTTP mockeable. | red (Ollama VM) |
| `ml/farslip/caption_cache.py` | Materializacion del parquet de captions: `generate_captions_parquet(pastis_root, out_path, folds, client)`, `load_captions(path) -> dict[patch_id, caption]`, esquema y validacion anti-leakage (`audit_captions`). Polars. | red (via client) |
| `ml/farslip/region_category_dataset.py` | `RegionCategoryPairDataset(Dataset)`: por patch, extrae N regiones (`ParcelIDs`+`semantic`), construye el batch region-category + caption global; `extract_regions(parcel_ids, semantic, min_area)`, `collate_region_batch(...)` (custom collate que aplana regiones de B patches a un batch plano con `region_cat_ids`, `patch_index`). Reusa `peak_ndvi_composite`, `load_pastis_patch`, `PastisFilter`. | disco (PASTIS) |
| `ml/farslip/mpcl_loss.py` | `MultiPositiveRegionCategoryLoss(nn.Module)` (ec. 4, log-sum sobre `P(i)`, simetrico R<->C), `GlobalImageTextLoss(nn.Module)` (ec. 1-2, InfoNCE simetrico). Logica pura, testeable en CPU. | no |
| `scripts/run_us036a_v2_farslip_faithful.py` | Orquestador productivo Typer: (1) `generate-captions` (materializa parquet via Gemma), (2) `train` (carga captions, construye `RegionCategoryPairDataset`, instancia el trainer con `L_glo + L_loc`, entrena, eval por clase, MLflow `:5010`, best -> ruta DVC). Importa lo de US-036/US-033, no lo reescribe. | GPU (H100) |
| `tests/ml/farslip/test_mpcl_loss.py` | MPCL equivalencia 1-positivo == CE; multi-positivo correcto; `L_glo` simetrica; ablacion `lambda_loc=0`. | no |
| `tests/ml/farslip/test_region_category_dataset.py` | N regiones por patch (>1); categoria = mayoritaria por instancia; filtro area; collate aplana correcto; rechazo de folds solapados. | no |
| `tests/ml/farslip/test_caption_generator.py` | Prompt builder sin fuga (regex); realce p2-p98 deterministico; PNG 896; cliente Gemma mockeado (`think=false`, `/api/chat`); audit_captions detecta fuga inyectada. | no |

### 3.2 Modificar (minimo, sin romper firmas mas de lo necesario)

| Archivo | Cambio | Riesgo |
|---------|--------|--------|
| `ml/farslip/distill.py` | (a) Reescribir `RegionCategoryAlignmentLoss.forward` para usar MPCL (delegar a `MultiPositiveRegionCategoryLoss`) **manteniendo la firma publica** y la equivalencia con CE cuando `\|P(i)\|=1` (back-compat con v1 + tests existentes). (b) Anadir al trainer `set_global_text_loss`/integrar `GlobalImageTextLoss` y el consumo de `caption_glo` en `step()` (el `loss_aux` placeholder cos se reemplaza por `L_glo` real). (c) Nuevo campo de config `lambda_loc`, `use_global_caption_loss`. | MEDIO: `step()` y `train()` esperan `batch["image"]/region_id/category_id`; el batch nuevo trae regiones aplanadas + caption embeddings. Mitigacion: mantener el path v1 (1-etiqueta) detras de un flag `supervision="region_category"` (default v2) vs `"dominant"` (v1) para no romper el orquestador v1 ni sus tests. |

> **No tocar**: `PatchDistillationLoss` (`L_dis`), `adapt_patch_embed_to_n_channels`, `save_student`,
> `set_text_prototypes` (se reusa para los prototipos de categoria), `cap_vocabulary.yaml`, `weights_uri`,
> pesos frozen (teacher CLIP, MiniLM), el parquet US-033 + su `.dvc`, `ml/farslip/incremental_curriculum.py`
> y `scripts/run_us036a_farslip_full_incremental.py` (v1 queda como modo de ablacion).

---

## 4. Interfaces publicas (firmas con type hints)

```python
# ml/farslip/caption_generator.py
class GemmaCaptionClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma4:31b-it-q8_0",
        timeout_s: float = 60.0,
        max_retries: int = 2,
    ) -> None: ...
    def caption(self, prompt: str, image_png_b64: str) -> tuple[str, float]:
        """POST /api/chat con think=false e images=[b64]. Returns (caption, gen_seconds)."""

def composite_to_png896(composite4: np.ndarray, side: int = 896) -> bytes:
    """(4,H,W) [0,1] -> PNG RGB (B04,B03,B02) realzado percentil 2-98 por canal, lado `side`."""

def build_caption_prompt(
    present_class_names: list[str],
    spatial_composition: str,        # "norte/centro/sur" resumido
    n_parcels: int,
    total_area_px: int,
    tile_mgrs: str,
    composite_date: str,             # YYYYMMDD del t* del composite
    typical_phenology: dict[str, str],  # class_name -> descripcion tipica (US-033)
) -> str:
    """Prompt en espanol que pide DESCRIBIR la escena. Sin NDVI numerico, sin AlphaEarth, sin la etiqueta."""

# ml/farslip/caption_cache.py
CAPTIONS_SCHEMA: dict[str, pl.DataType]  # patch_id, caption_glo, model, prompt_version, tile, composite_date, present_class_ids, gen_seconds

def generate_captions_parquet(
    pastis_root: Path,
    out_path: Path,
    folds: Sequence[int],
    client: GemmaCaptionClient,
    prototype_path: Path | None = None,
    prompt_version: str = "v2",
    resume: bool = True,                 # no regenera captions ya en el parquet
) -> Path: ...

def load_captions(path: Path) -> dict[str, str]: ...

def audit_captions(path: Path) -> dict[str, int]:
    """Cuenta matches de patrones de fuga (NDVI numerico, alphaearth, 'la clase es'). 0 = limpio."""

# ml/farslip/region_category_dataset.py
def extract_regions(
    parcel_ids: np.ndarray,             # (H,W) instancia por pixel
    semantic: np.ndarray,               # (H,W) clase por pixel
    active_class_ids: tuple[int, ...],
    min_area_px: int = 16,
) -> list[tuple[int, int]]:
    """Devuelve [(parcel_instance_id, category_id PASTIS)] por region valida del patch."""

class RegionCategoryPairDataset(Dataset):
    def __init__(
        self,
        captions: dict[str, str],
        root: Path = _DEFAULT_PASTIS_ROOT,
        folds: Sequence[int] = (1, 2, 3),
        active_class_ids: tuple[int, ...] = tuple(range(1, 19)),
        min_area_px: int = 16,
        resize_to: int = 224,
        seed: int = 42,
    ) -> None: ...
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """{'image': (4,224,224), 'caption': str, 'region_cat_ids': LongTensor (N,)}."""

def collate_region_batch(items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Aplana B patches: images (B,4,H,W); captions (B,); region_cat_ids (sum_N,);
       region_to_patch (sum_N,) indice de patch por region."""

# ml/farslip/mpcl_loss.py
class MultiPositiveRegionCategoryLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None: ...
    def forward(
        self,
        region_visual: torch.Tensor,    # (R, D) CLS visual por region
        category_text: torch.Tensor,    # (C, D) prototipo textual por categoria
        region_cat_ids: torch.Tensor,   # (R,) categoria de cada region (indice en [0,C))
    ) -> torch.Tensor:
        """L_loc ec.4 simetrico R<->C; P(i)=regiones que comparten categoria. |P(i)|=1 -> InfoNCE."""

class GlobalImageTextLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None: ...
    def forward(
        self,
        image_cls: torch.Tensor,        # (B, D)
        caption_cls: torch.Tensor,      # (B, D) text encoder frozen
    ) -> torch.Tensor:
        """L_glo ec.1-2 InfoNCE simetrico imagen-caption."""
```

**Como se baten los positivos (MPCL)**: dada `region_cat_ids (R,)`, la matriz de positividad es
`pos[i,j] = (region_cat_ids[i] == cat_of_text[j])` para el termino R->C, y simetrico para C->R. El log-sum-exp
se calcula con `logits = region_visual_n @ category_text_n.T / tau`; para cada ancla i se promedia
`-log( sum_{j in P(i)} exp(logit_ij) / sum_k exp(logit_ik) )` (estable via `logsumexp`). Si `category_text`
tiene 1 fila por categoria y varias regiones comparten categoria, `P(i)` apunta a la MISMA columna -> se
maneja como SupCon clasico (positivos = regiones con misma etiqueta, no columnas duplicadas); la
implementacion construye el set de positivos sobre el eje de regiones cuando se contrasta region<->region, y
sobre el eje de categorias cuando es region<->texto. **Esto se especifica y se testea explicitamente** porque
es el punto donde v1 fallaba (trataba todo positivo extra como negativo via `cross_entropy`).

---

## 5. Dominios + SUB-TAREAS PARALELAS (write-sets disjuntos)

### Dominios tocados

- [x] **ml** (`ml/farslip/caption_generator.py`, `caption_cache.py`, `region_category_dataset.py`, `mpcl_loss.py` NUEVOS; `distill.py` MODIFICADO; `scripts/run_us036a_v2_farslip_faithful.py` NUEVO; tests)
- [x] **H100** (generacion captions Gemma local + re-entrenamiento FarSLIP)
- [ ] backend - [ ] frontend - [ ] agent - [ ] db - [ ] dagster - [ ] infra
- [x] docs (solo estos dos `.md` de planning/handoff)

### Sub-tareas con write-sets DISJUNTOS (paralelizables)

| # | Sub-tarea | Write-set (DISJUNTO) | Paralela? | Depende de |
|---|-----------|----------------------|-----------|------------|
| **T1** | **Generador de captions Gemma** | `ml/farslip/caption_generator.py`, `ml/farslip/caption_cache.py`, `tests/ml/farslip/test_caption_generator.py` | **SI** (independiente del loss y del dataset) | US-033 (parquet pheno, solo leer) |
| **T2** | **Dataset region-category** | `ml/farslip/region_category_dataset.py`, `tests/ml/farslip/test_region_category_dataset.py` | **SI** (consume `peak_ndvi_composite` ya existente; el caption es un `dict` inyectado) | US-036 composite (existe) |
| **T3** | **MPCL loss + L_glo** | `ml/farslip/mpcl_loss.py`, `tests/ml/farslip/test_mpcl_loss.py` | **SI** (logica pura CPU, sin dataset ni Gemma) | paper ec. 3-4, 1-2 |
| **T4** | **Integracion en trainer + orquestador** | `ml/farslip/distill.py` (MOD), `scripts/run_us036a_v2_farslip_faithful.py`, `tests/ml/farslip/test_us036a_v2_orchestrator.py` | **NO** (integra T1+T2+T3) | T1, T2, T3 |

> **Paralelizables a la vez: T1, T2, T3** (3 agentes ml en ramas/worktrees separados, write-sets disjuntos).
> **T4 es secuencial** (rebase tras merge de T1-T3). El unico archivo COMPARTIDO es `distill.py`, tocado SOLO
> por T4 -> sin conflicto entre T1/T2/T3.

---

## 6. Plan de tests

### 6.1 MPCL (T3) — el corazon del rediseno

- `test_mpcl_equals_cross_entropy_when_single_positive`: batch con 1 region por categoria -> MPCL == `F.cross_entropy(logits, cat_ids)` (`atol=1e-5`). **Prueba que MPCL generaliza v1** (caso `|P(i)|=1`).
- `test_mpcl_multipositive_differs_from_v1`: batch con 3 regiones de la misma categoria + 3 de otra ->
  el loss MPCL NO es igual a tratar 2 de los 3 como negativos (lo que hacia `cross_entropy` de v1); con
  logits alineados a su categoria, MPCL multi-positivo da loss menor que el caso degenerado.
- `test_mpcl_gradient_points_to_positive_centroid`: caso analitico 2 cat x 3 regiones; el gradiente del ancla
  empuja hacia el centroide de SUS positivos (no hacia los negativos).
- `test_mpcl_symmetric_r_c`: `L_{R->C}` y `L_{C->R}` ambos presentes; el loss total es su media.
- `test_global_image_text_loss_symmetric`: `L_glo` simetrica imagen<->texto; con CLS identicos diagonales ->
  loss minimo.
- `test_lambda_loc_zero_ablation`: `lambda_loc=0` -> `loss_total == loss_glo` exacto.

### 6.2 Dataset region-category (T2)

- `test_extract_regions_multi_object`: patch sintetico con 3 parcelas (2 clases) -> 3 region-category (N>1).
- `test_region_category_uses_majority_class`: instancia con pixeles mixtos -> categoria = mayoritaria;
  instancia mayoritaria Background/Void -> descartada.
- `test_min_area_filter`: parcela `< min_area_px` excluida.
- `test_collate_flattens_regions`: B=2 patches con (2,3) regiones -> batch plano de 5 regiones + `region_to_patch` correcto.
- `test_dataset_rejects_overlapping_folds`: `folds` train y val solapados -> ValueError (anti-leakage).
- `test_dataset_requires_caption`: patch sin caption en el dict -> error explicito (no silencioso).

### 6.3 Captions Gemma (T1)

- `test_prompt_builder_no_leakage`: el prompt construido NO contiene NDVI numerico, ni `alphaearth`, ni
  `la clase es`/`the class is` (regex). Incluye clases presentes, composicion, conteo, tile, fenologia.
- `test_composite_to_png896_deterministic`: realce p2-p98 deterministico; salida PNG lado 896; canales RGB.
- `test_gemma_client_uses_chat_think_false`: cliente mockeado -> POST a `/api/chat` con body
  `{"think": false, "messages":[{... "images":[b64]}]}` (NO `/api/generate`).
- `test_audit_captions_detects_injected_leak`: parquet con una caption que contiene `NDVI=0.8` -> audit
  reporta `>= 1` match.
- `test_caption_cache_resume`: con `resume=True`, patches ya en el parquet NO se re-generan (cliente no
  invocado para ellos).

### 6.4 Anti-leakage train/eval + integracion (T4)

- `test_orchestrator_rejects_train_eval_fold_overlap`: ValueError si `val_folds` solapa `train_folds`.
- `test_orchestrator_fails_if_caption_missing_for_split_patch`: falta caption de un patch del split -> falla rapido.
- `test_step_combines_glo_and_loc`: con trainer y encoders mockeados, `step()` devuelve `loss_glo`, `loss_loc`,
  `loss_total = loss_glo + lambda_loc*loss_loc`.
- `test_v1_path_still_works`: `supervision="dominant"` reproduce el comportamiento v1 (no rompe US-036-a v1).
- `test_mlflow_run_closed`: run FINISHED (gotcha RUNNING).

Todos los tests: trainer pesado, encoders CLIP/MiniLM, Gemma y PASTIS real **mockeados** (cero GPU/red/disco real). Cobertura objetivo `>= 70%`.

---

## 7. H100 — presupuesto y orden

`nvidia-smi` ANTES; parar el daemon idle-shutdown (memoria `l4-vm-idle-shutdown-daemon` / VM H100); 1 GPU 96 GB.

### Orden de ejecucion

1. **Fase A — Captions Gemma** (`generate-captions`). Gemma 31B q8_0 ya servido por Ollama (~33 GB VRAM).
   `~2433 patches x ~3.1 s/caption (think=false) ~ 2 h` (cota; `resume=True` permite reanudar). Coexiste con
   el training del student (ViT-B/16 < 16 GB) pero por simplicidad se corre PRIMERO (materializa el parquet de
   captions, `dvc add`+push) y luego el training LEE el parquet sin Gemma. Si el presupuesto aprieta, Fase A y
   B pueden concurrir (pico VRAM `< 50 GB`).
2. **Fase B — Re-entrenamiento FarSLIP v2** (`train`). ViT-B/16 batch=64 BF16. PASTIS ~2433 patches; con
   region-category el batch efectivo de regiones es `~B * mean_regions_per_patch` (mas senal por step).
   Estimacion: `~0.6-1.0 min/epoch`; M=30-40 epochs -> `~0.5-1.0 h`. `time_cap_hours=8.0` red de seguridad.
3. **Fase C — Eval por clase + best -> DVC**. F1/IoU por clase sobre held-out fold; tabla v1 vs v2;
   `dvc add checkpoints/farslip/faithful_v2/best.safetensors` + `dvc push` desde la VM (ADC OK, memoria
   `session-creds-h100-gcp-available`).

### Presupuesto VRAM (validacion)

| Componente | Precision | Batch | VRAM aprox |
|------------|-----------|-------|------------|
| Gemma 4 31B q8_0 (Ollama) | q8_0 | 1 img | ~33 GB |
| Student ViT-B/16 + teacher RGB | BF16 | 64 patches | < 16 GB |
| Text encoder MiniLM (frozen, captions pre-encodeables) | FP32 | - | < 2 GB |
| **Pico si concurren A+B** | - | - | **< 50 GB / 96 GB** OK |

> Optimizacion: pre-encodear los CLS textuales de las captions y de los prototipos de categoria UNA vez
> (encoders frozen) -> el training no carga el text encoder en el loop (ahorra VRAM y tiempo). El paper
> recalcula prototipos 1x/epoch; aqui las captions/categorias son fijas, asi que se cachean.

### Horas totales

`Fase A ~2 h + Fase B ~0.5-1 h + eval/DVC ~0.5 h ~ 3-3.5 h`, holgadamente dentro de una ventana H100.

---

## 8. data_version (DVC) + code_version (MLflow :5010)

- **`data_version`**: hash de los `.dvc` de `data/PASTIS-R` (entrada) + `data/farslip/pastis_captions.parquet`
  (NUEVO artefacto, `dvc add`+push tras Fase A) + referencia al `.dvc` de
  `data/features/phenology_class_prototypes_pastis.parquet` (US-033, solo leido). El run MLflow tagea el
  `data_version` compuesto (via `dvc_data_version` de `ml/utils/git_meta.py`).
- **`code_version`**: git SHA (`git_sha()`), igual que el resto del repo.
- **MLflow**: server Docker `:5010`, experiment `farslip`, run `farslip-faithful-v2`. Params: `lambda_loc`,
  `tau`, `n_categories`, `n_in_channels=4`, `caption_model=gemma4:31b-it-q8_0`, `prompt_version`,
  `supervision=region_category`, `mean_regions_per_patch`. Metricas: `loss_glo`, `loss_loc`, `loss_total` por
  step; `f1_class_*`, `iou_class_*`, `macro_f1`, `n_classes_well_resolved`. Artefactos: tabla por clase, tabla
  delta v1 vs v2, muestreo de 20 captions auditadas. **Run CERRADO** (gotcha subprocess RUNNING: verificar
  FINISHED). Si `:5010` cae en la VM: MLflow nativo SQLite (memoria `vm-h100-dvc-pull-401-no-adc`) o ingestar
  desde log (patron US-034/035).
- **Checkpoints**: `checkpoints/farslip/faithful_v2/best.safetensors` (+ por epoch como red de resiliencia).
  Best -> `dvc add` + `dvc push`.

---

## 9. Riesgos y mitigaciones

| # | Riesgo | Prob/Imp | Mitigacion |
|---|--------|----------|------------|
| **R-MPCL-WRONG** | MPCL mal implementada (off-by-one en `P(i)`, doble conteo del ancla, no simetrico, inestabilidad numerica) -> el rediseno no aprende o aprende peor que v1. | ALTA/ALTA | Test golden de equivalencia con CE en `\|P(i)\|=1`; test de gradiente al centroide; `logsumexp` estable; excluir el ancla de sus propios positivos en el termino region<->region; revisar contra ec. 4 linea a linea (Seccion 0). |
| **R-CAPTION-LEAK** | Captions con fuga circular (NDVI numerico, AlphaEarth, etiqueta literal) -> el contraste se vuelve trivial y las metricas mienten. | ALTA/ALTA | Prompt builder prohibe explicitamente esos campos; `audit_captions` con regex sobre el parquet (AC-4, test); muestreo manual de 20 captions documentado; el prompt pide DESCRIBIR, no clasificar. |
| **R-GEMMA-SLOW** | Generador Gemma lento o falla (sin `think=false` -> timeout infinito; Ollama caido; OOM con 896 px). | MEDIA/ALTA | `think=false` OBLIGATORIO (validado ~3.1 s); `resume=True` reanuda; retries + timeout en el cliente; 896 px confirmado (128 px no sirve); fallback: generar en lotes nocturnos. |
| **R-REGION-CROP** | PASTIS no tiene crops por region; usar el CLS del patch para todas sus regiones se desvia del paper (que usa CLS de crop/RoI por region). | ALTA/MEDIA | Documentado como desviacion controlada (Seccion 2.3, Seccion 10); la senal multi-objeto entra por caption `L_glo` + MPCL cross-patch; v2.1 con RoIAlign queda en backlog. Reportar honestamente. |
| **R-CARDINALITY** | Cardinalidad de region-category desbalanceada (Meadow domina; clases cola con 1-2 regiones en todo el fold) -> MPCL con `\|P(i)\|=1` para clases raras = sin ganancia multi-positivo ahi. | MEDIA/MEDIA | MPCL degrada elegante a InfoNCE de 1 positivo para esas clases (no rompe); reportar `mean_\|P(i)\|` por clase; opcional class-balanced sampling del batch. |
| **R-NO-IMPROVE** | El rediseno NO mejora (o empeora) vs v1 / vs AlphaEarth 0.233. | MEDIA/MEDIA | **Reporte honesto** (ADR-007 §3, patron del proyecto): si v2 < v1, se documenta como resultado valido (region-category no transfiere en low-resource PASTIS); FarSLIP queda como base learner del stacking E-b (US-042). Tabla v1 vs v2 explicita. NO maquillar. |
| **R-DISTILL-BREAK** | Modificar `distill.py` rompe el path v1 / sus tests / el extractor. | MEDIA/ALTA | Flag `supervision` (default v2, v1 disponible); mantener firma publica de `RegionCategoryAlignmentLoss`; test `test_v1_path_still_works`; NO tocar `save_student`/`set_text_prototypes`/`weights_uri`. |
| **R-MLFLOW-RUNNING** | Run queda `RUNNING` (subprocess) en `:5010`. | MEDIA/BAJA | `try/finally` cierra el run; test `test_mlflow_run_closed`; verificar FINISHED tras el run (gotcha ml/AGENTS.md). |
| **R-DVC-401** | `dvc push` falla 401 en la VM sin ADC. | MEDIA/MEDIA | ADC OK en esta sesion (memoria `session-creds-h100-gcp-available`); si falla, push desde local o scp (memoria `vm-h100-dvc-pull-401-no-adc`). |
| **R-PROTO-REGEN** | Regenerar el parquet US-033 a ciegas (rompe lineage). | BAJA/ALTA | Solo LEER/FILTRAR por `class_id`; NO Gemini; respetar el `.dvc`. |

---

## 10. Mapeo rubrica + fidelidad al paper

### 10.1 Tabla de fidelidad (que del paper implementamos vs desviaciones)

| Componente paper (Li et al. 2025) | v1 (US-036-a) | **v2 (este rediseno)** | Fidelidad |
|-----------------------------------|---------------|------------------------|-----------|
| `L_glo` InfoNCE imagen-caption (ec. 1-2) | Ausente (solo `loss_aux` cos placeholder) | **InfoNCE simetrico CLS imagen <-> CLS caption global Gemma** | **ALTA** (nuevo) |
| `L_loc` region-category MPCL (ec. 3-4) | `F.cross_entropy` = **1 positivo** | **MPCL multi-positivo: `P(i)`=regiones de la misma categoria** | **ALTA** (corregido) |
| Region multi-objeto por imagen | **1 etiqueta dominante/patch** | **N regiones/patch via `ParcelIDs`** (mean > 1) | **ALTA** (corregido) |
| Caption global rica | Plantilla/prototipo unico | **Caption multimodal real (Gemma 4 31B, imagen 896 px)** | **ALTA** (nuevo) |
| `V_i^r` = CLS token (§4.3 Takeaway-1) | CLS | **CLS** (no RoI) | ALTA |
| `L_dis` patch self-distillation (ec. 5) | Implementado (`PatchDistillationLoss`) | Disponible, NO central (paper: `L_glo+L_loc` ya es lo mejor) | MEDIA |
| `tau=0.07` | 0.07 | 0.07 | ALTA |
| Combinacion `L_glo + L_loc` (Tabla 3) | No (solo `L_loc` 1-pos) | **Si** (`L_total = L_glo + lambda_loc*L_loc`) | ALTA |
| **Desviacion: dataset** | MGRS-200k -> **PASTIS-R real** | igual | controlada (ADR-007) |
| **Desviacion: bandas** | 3 RGB -> **4 RGB+NIR** | igual | controlada (ADR-007) |
| **Desviacion: 1 crop por region** | MGRS bbox -> **patch unico PASTIS** | CLS de patch compartido por sus regiones (R-REGION-CROP) | controlada, documentada |
| **Desviacion: caption** | InternVL3 (paper) -> **Gemma 4 31B local** | igual | controlada (modelo on-prem del proyecto) |

### 10.2 Mapeo rubrica del curso

- **EPIC 5 (Modelos alternativos/comparativa)**: FarSLIP v2 es un modelo de comparacion vs AlphaEarth y vs
  v1; aporta la metrica honesta v1 vs v2 (mejora por fidelidad al paper) — premia "reimplementacion fiel de
  SOTA + analisis de desviaciones".
- **EPIC 6 (ensambles)**: el checkpoint best alimenta el stacking E-b (US-042) como base learner (ADR-007 §3,
  patron "vale +1% al ensamble aunque no gane solo").
- **Paper Track opcional**: la tabla de fidelidad 10.1 es evidencia directa de reimplementacion verificable
  linea-a-linea contra las ecuaciones del paper (Seccion 0).
- **Rubrica MLOps**: MLflow `:5010` con `data_version`+`code_version`, DVC del checkpoint y del parquet de
  captions, spatial CV oficial (folds PASTIS), tests con cobertura.

---

## 11. Checklist de cierre

- [ ] **T1-T3 en paralelo** (write-sets disjuntos), luego **T4** (integracion): codigo funcional, cero stubs/TODOs.
- [ ] `MultiPositiveRegionCategoryLoss` (ec. 4) + `GlobalImageTextLoss` (ec. 1-2) con tests golden (equivalencia CE en `|P(i)|=1`, multi-positivo, simetria, ablacion `lambda_loc=0`).
- [ ] `RegionCategoryPairDataset` con `mean_regions_per_patch > 1` verificado sobre PASTIS-R real; tests N>1, mayoritaria, area, collate, anti-fold-overlap.
- [ ] `GemmaCaptionClient` (`/api/chat`, `think=false`, 896 px) + `audit_captions` limpio (cero fuga); muestreo de 20 captions documentado en el handoff.
- [ ] `distill.py` modificado SIN romper el path v1 (`supervision="dominant"` sigue verde); firma de `RegionCategoryAlignmentLoss` preservada.
- [ ] **Fase A (H100)**: `data/farslip/pastis_captions.parquet` materializado, auditado, `dvc add`+`dvc push`.
- [ ] **Fase B (H100)**: re-entrenamiento `farslip-faithful-v2`, `loss_glo`/`loss_loc`/`loss_total` convergen; `nvidia-smi` previo logueado.
- [ ] **Fase C**: eval F1/IoU por clase + tabla v1 vs v2 (reporte honesto aunque no mejore); best -> `dvc add`+push.
- [ ] MLflow `:5010` run `farslip-faithful-v2` con `data_version`+`code_version`+params+metricas, **CERRADO** (FINISHED, no RUNNING).
- [ ] `make check` limpio (ruff + secrets + i18n) + `pytest tests/ml/farslip/` verde, cobertura `>= 70%`.
- [ ] ADR-007 anotado (en el cierre) con el salto de fidelidad 1-positivo -> MPCL + caption global multimodal.
- [ ] Handoff `docs/us-handoff/us-036-a-v2-faithful.md` actualizado a `implemented` con la tabla de resultados y `stop`/`best` reales.
- [ ] AlphaEarth NO usado en este pipeline (confirmado: va a E-b US-042).
