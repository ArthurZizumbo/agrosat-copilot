# Backlog · US-022-c — Cierre fenologico: FarSLIP en GCP + fix XGBoost + full CUDA + DVC pendientes + Gemini real

**Origen**: post US-022b (2026-05-23) consolidacion en un solo PB de lo
pendiente para cerrar el ciclo del reencuadre fenologico end-to-end. Agrupa
trabajo de tres backlog anteriores que ya se englobaron aqui
(`us-022b.1..4`, `datasets-grandes-dvc-eurocrops`, `us-016-1-dvc-multisensor-outputs`,
bloque A de `us-019-1-terraform`). El bloque B Azure del antiguo us-019-1
queda **fuera del scope** de esta US y vive en
[`us-019-2-azure-pendiente.md`](us-019-2-azure-pendiente.md).

**Status**: backlog · priorizado
**Epic**: E3/E4/E5 (transversal — datos + features + modelos + ensembles)
**Sprint**: S7 (1-jun → 7-jun)
**SP estimados**: **13 SP** (P1=5, P2=1, P3=3, P4=3, P5=1)
**Owner**: Arthur (P1, P4, P5 infra) + Isaac (P2, P3 ML)

---

## Indice de prioridades

| P | Sub-bloque | SP | Duracion | Costo cloud | Bloqueante de |
|---|------------|----|----|-------------|---------------|
| **P1** | FarSLIP materializacion en GCP L4 dedicada | 5 | 1.5 dias | ~$2 USD spot | ablation con `with_farslip`, EPIC 6 stacking |
| **P2** | Fix bug XGBoost NaN en `model_comparison.parquet` | 1 | 30 min | $0 | full CUDA (P3) — fix antes |
| **P3** | Re-lanzar FULL CUDA con los 4 fixes ML aplicados | 3 | 1.5-2h | $0 (GPU local) | cierre US-022b → PR a `develop` |
| **P4** | DVC versionado pendiente (3 datasets + 3 outputs US-016 + import tfstate) | 3 | 4-6h | $0 (bucket ya existe) | reproducibilidad cross-machine, lineage MLflow |
| **P5** | Rama semantica Gemini 3.5 Flash real sobre subset estratificado | 1 | 0.5 dia | ~$0.20 USD | resultado D-4 del plan US-022b |

> **Azure (V1-V6 H100)**: fuera de scope. Permanece en
> [`us-019-2-azure-pendiente.md`](us-019-2-azure-pendiente.md). Solo se
> retoma si se decide ejecutar una ventana H100 antes del 21-jun.

---

## P1 — FarSLIP materializacion en GCP L4 dedicada

**Que**: ejecutar el pipeline FarSLIP end-to-end sobre una maquina GCP L4
dedicada para materializar `data/farslip/embeddings_italy.parquet` (85951 x
512). Cierra la deuda de US-017 Fase 4 + sub-US 022b-B.

**Por que es P1**: el bloque `with_farslip` de la ablation no existe sin este
parquet; el stacking ensemble de EPIC 6 lo necesita como base learner; el
usuario decidio (2026-05-23) ejecutarlo en GCP dedicada para **mayor
exactitud** que en la GPU local del equipo.

**Pre-requisitos GCP** (verificados 2026-05-23):
- [x] Billing activo en `agrosat-copilot`
- [x] Cuota L4 = 1 + PREEMPTIBLE_L4 = 1 en us-central1
- [x] APIs habilitadas: aiplatform, artifactregistry, cloudbuild, run, sqladmin
- [x] Imagen `ml-train` constructible (Dockerfile listo)
- [x] SAs `mlflow` + `ml_train_run` declaradas en Terraform
- [x] Cloud Run MLflow `agrosat-mlflow-dev` declarable scale-to-zero
- [ ] **Falta**: `terraform apply` para provisionar bucket + Cloud SQL + Cloud Run

**Etapas** (3 horas humano + ~6h L4 spot):

```bash
# Etapa 1 — Provisionar infra GCP
cd infrastructure/terraform/environments/dev
terraform init -upgrade
terraform plan -out=plan.out
terraform apply plan.out
export MLFLOW_TRACKING_URI=$(terraform output -raw mlflow_tracking_uri)

# Etapa 2 — Build + push imagen ml-train
gcloud builds submit --config infrastructure/cloudbuild.yaml \
  --substitutions=SHORT_SHA=$(git rev-parse --short HEAD)

# Etapa 3 — Smoke A-5 del Vertex AI custom-job (<$0.20)
make train-l4-smoke

# Etapa 4 — Build dataset FarSLIP (US-017 deuda)
make farslip-dataset-build

# Etapa 5 — Training student real en L4 spot (~6h, ~$1.7 USD)
make train-l4 epic=E3 us=US-017 script=ml/farslip/train_student.py

# Etapa 6 — Extraer embeddings + materializar parquet
make farslip-extract-embeddings
dvc add data/farslip/embeddings_italy.parquet
dvc push
git add data/farslip/embeddings_italy.parquet.dvc
git commit -m "feat(E3): materializa farslip_embeddings_italy_v1 (85951 x 512)"

# Etapa 7 — Eval mIoU PASTIS-R (gate B-3)
make farslip-eval-pastis
```

**Criterios de aceptacion** (heredados US-017 + US-022b):

| AC | Criterio | Como verificar |
|----|----------|----------------|
| B-1 | Dataset >= 30k pares + balance min/max >= 0.20 | `make farslip-dataset-check` exit 0 |
| B-2 | MLflow run `farslip-clip-italy-v1` con `val_clip_acc` por epoch + hard cap 8h | UI MLflow + log L4 en `docs/l4_log.md` |
| B-3 | mIoU PASTIS-R, gate `mIoU_farslip - mIoU_remoteclip >= +0.05` | notebook `notebooks/features/04_farslip_eval_pastis.ipynb` |
| B-4 | `data/farslip/embeddings_italy.parquet` con 85951 x 512 + DVC tag | `dvc list . data/farslip/` + `git tag farslip-embeddings-italy-v1` |
| B-5 | DVC push + MLflow Registry `@Production` | `dvc remote ls` muestra blob + UI MLflow `Models/farslip-clip-italy-v1@Production` |

**Riesgos**:
- Spot L4 se interrumpe → checkpoint cada 30 min + `restartJobOnWorkerRestart: false`.
- mIoU `< +0.05` → resultado negativo honesto, la US cierra igual (R6 del plan).
- Costo > $10 USD → hard timeout 4h por job + `make cost-audit` pre/post.

---

## P2 — Fix bug XGBoost NaN en `model_comparison.parquet`

**Que**: en la corrida FULL CUDA inicial de US-022b (2026-05-22) la tabla
`reports/baseline/reencuadre_fenologico/model_comparison.parquet` quedo con
la fila `xgboost` con NaN en todas las metricas, aunque el dato real existe
en `ablation_table.parquet` (F1-macro 0.4094).

**Causa**: en `scripts/build_reencuadre_notebook.py` linea ~395, el lookup
falla por el match interno `'xgb'` vs `'xgboost'` y `next()` devuelve `None`.

**Fix** (30 min, un solo edit en el builder + reconstruir notebook):

```python
# ANTES (buggy):
model_results['xgboost'] = next(
    (r for r in ablation_results if r.feature_set == winner_set and r.model_kind == 'xgb'),
    None,
)

# DESPUES:
xgb_winner = next(
    (r for r in ablation_results if r.feature_set == winner_set and r.model_kind == 'xgb'),
    None,
)
if xgb_winner is None:
    xgb_results = [r for r in ablation_results if r.model_kind == 'xgb']
    if xgb_results:
        xgb_winner = max(xgb_results, key=lambda r: r.f1_macro)
if xgb_winner is not None:
    model_results['xgboost'] = xgb_winner
```

Tambien agregar al notebook un assert defensivo:

```python
assert not comparison_table['f1_macro'].is_null().any(), "xgboost NaN bug"
```

**Por que P2 antes de P3**: si arrancamos el full CUDA (P3) con este bug
sin arreglar, perdemos 45-60 min de wall clock y la grafica
`model_comparison.png` final queda con 1 barra missing. Arreglar primero el
builder cuesta 30 min y garantiza que el full CUDA produce la tabla
comparativa completa con 3 modelos.

**Criterios de aceptacion**:

| AC | Criterio | Como verificar |
|----|----------|----------------|
| AC-1 | `model_comparison.parquet` no tiene NaN en xgboost | `polars.read_parquet(...).filter(pl.col('model')=='xgboost')` muestra f1_macro real |
| AC-2 | Grafica `model_comparison.png` muestra 3 barras | comparar visual con la corrida anterior |
| AC-3 | Smoke sin regresion | `make reencuadre-notebook-check` (800 parcelas) sigue verde |

---

## P3 — Re-lanzar FULL CUDA con los 4 fixes ML aplicados

**Que**: ejecutar `make reencuadre-notebook-full` sobre 85951 parcelas con
los 4 fixes ML ya implementados en `ml/train/phenology_models.py`
(2026-05-23). Es el cierre numerico de la US-022b-C (criterios C-3 y C-4 del
plan canonico).

**Por que es P3 (no P1)**: depende de P2 (fix XGBoost) para que la grafica
quede limpia, y depende de P4 (DVC del subset feature_selection) para
reproducibilidad del input. Pero NO depende de P1 FarSLIP (la ablation
omite `with_farslip` graceful).

**Comando**:

```bash
make reencuadre-notebook-full
# Equivalente a:
# poetry run papermill notebooks/feature_engineering/05_reencuadre_fenologico.ipynb \
#   notebooks/feature_engineering/05_reencuadre_fenologico.ipynb \
#   -p MAX_SAMPLES 0 -p K_FOLDS 5 -p BUFFER_KM 1.0 \
#   -p TEMPORAL_EPOCHS 200 -p TEMPORAL_BATCH_SIZE 256 \
#   -p DEVICE auto -p RUN_SEMANTIC_BRANCH False --no-progress-bar
```

Early stopping con `patience=20` probablemente corte el training en epoch
50-100 (~5-15 min por modelo en GPU local). Wall clock total ~45-60 min.

**Resultado esperado (paper-based)**:

| Modelo | F1-macro esperado | Referencia |
|--------|-------------------|------------|
| XGBoost (winner set) | **0.4094** | ya verificado, no cambia |
| TempCNN | **0.25-0.40** | Pelletier 2019 sobre series limpias |
| InceptionTime | **0.30-0.45** | Fawaz 2020, mejor que TempCNN en BreizhCrops |

**Criterios de aceptacion**:

| AC | Criterio | Como verificar |
|----|----------|----------------|
| AC-1 | Notebook ejecutado sin errores con outputs poblados | conteo PNG embebidos == 7, errores == 0 |
| AC-2 | TempCNN o InceptionTime supera el baseline tabular 0.4094 | `model_comparison.parquet` con `f1_macro > 0.4094` para al menos uno |
| AC-3 | Early stopping triggered en ambos modelos | logs muestran `temporal_early_stop` con `best_val_f1 > 0` |
| AC-4 | Checkpoints persistidos | `models/checkpoints/phenology/*.pt` con 2-4 archivos `{kind}_{sha}_f1_{score}_seed{N}.pt` |
| AC-5 | MLflow run registrado con tags `data_version` + `code_version` | UI MLflow |
| AC-6 | Handoff actualizado con tabla de resultados reales del FULL CUDA | `docs/us-handoff/us-022b.md` §"Resultados FULL CUDA" reemplaza la version del 22-may |

**Riesgos**:
- OOM en GPU local con batch=256 → bajar a batch=128.
- Best epoch al final del fold (early stopping no se dispara) → aumentar `patience` a 30.
- F1 sigue bajo despues de los fixes (~0.10-0.20) → investigar causa raiz (FE
  FFT sintetica vs serie diaria real). Posible nueva US.

---

## P4 — DVC versionado pendiente (housekeeping cross-machine)

**Que**: cerrar la deuda DVC acumulada en 3 frentes que estaban bloqueados
por el bucket `gs://agrosat-dvc-remote`. **El bucket ya esta provisionado y
operativo** (verificado 2026-05-23: `poetry run dvc remote list` reporta
`gcs-remote gs://agrosat-dvc-remote (default)`). Los tres frentes ya son
**directamente ejecutables**.

### P4.1 — US-016 outputs multisensor (AC-14 deferido)

Versionar los 3 outputs canonicos de US-016 contra el remoto:

| Artefacto | Tag git | Tamano |
|-----------|---------|--------|
| `data/features/features_fused_v1.parquet` (Italia, 189 cols) | `fused-features-italy-v1` | ~5-15 MB |
| `data/splits/spatial_kfold_v1/` (5 folds) | `spatial-kfold-italy-v1` | <1 MB |
| `artifacts/scaler_v1.pkl` (StandardScaler fold-0) | `scaler-v1` | <10 KB |

```bash
# Generar outputs reales (no demo) — Isaac corre con creds GEE
make features-fuse-italy
# Versionar
dvc add data/features/features_fused_v1.parquet
dvc add data/splits/spatial_kfold_v1/
dvc add artifacts/scaler_v1.pkl
git add data/features/features_fused_v1.parquet.dvc data/splits/spatial_kfold_v1.dvc artifacts/scaler_v1.pkl.dvc .gitignore
git commit -m "data(E3): track US-016 multisensor outputs via DVC"
make dvc-push
git tag fused-features-italy-v1 -m "US-016 fused features v1 (189 cols)"
git tag spatial-kfold-italy-v1 -m "US-016 spatial K-fold v1 (5 folds, buffer 1km)"
git tag scaler-v1 -m "US-016 StandardScaler v1 (fold_0 train)"
git push origin --tags
```

Actualizar `docs/us-handoff/us-016.md`: marcar AC-14 ✅, cambiar
`ready-to-close` → `closed`. Crear `docs/us-resolved/us-016.md`.

### P4.2 — Datasets grandes (EuroCrops FR + parcel subset)

| Artefacto | Tag git | Tamano | Uso |
|-----------|---------|--------|-----|
| `data/reference/eurocrops/FR_2018/` (5 archivos shapefile) | `eurocrops-fr-2018-v1` | ~11 GB | ground truth vectorial Francia, validacion cruzada PASTIS-R |
| `data/test_fixtures/feature_selection_parcels_subset.parquet` | `feature-selection-parcels-v1` | ~76 MB | subset US-018 consumido por notebook 03b + 05 |

```bash
dvc add data/reference/eurocrops/FR_2018/
dvc add data/test_fixtures/feature_selection_parcels_subset.parquet
git add data/reference/eurocrops/FR_2018.dvc data/test_fixtures/feature_selection_parcels_subset.parquet.dvc
git commit -m "data(E3): track EuroCrops FR + parcel subset via DVC"
make dvc-push
git tag eurocrops-fr-2018-v1 -m "EuroCrops FR_2018 shapefile (9.5M parcelas, HCAT)"
git tag feature-selection-parcels-v1 -m "Subset PASTIS-R parcel-level feature engineering"
git push origin --tags
```

Limpiar entradas manuales temporales de `data/.gitignore` (DVC las maneja).

### P4.3 — Import `agrosat-tfstate` al state de Terraform

Hoy el bucket `agrosat-tfstate` (backend del propio state) **NO esta managed**
por Terraform — paradoja de bootstrap clasica. Si alguien lo borra o cambia
region/lifecycle, Terraform no detecta drift.

```bash
cd infrastructure/terraform/environments/dev
# Verificar que el recurso existe en el modulo GCP
grep -n "tfstate" ../../modules/gcp/main.tf || echo "FALTA declarar resource"
# Si falta, agregarlo a `local.buckets` con `versioning_enabled = true`.
terraform import 'module.gcp.google_storage_bucket.tfstate' agrosat-tfstate
terraform plan   # debe reportar 0 cambios
```

Documentar en `docs/operations/terraform-bootstrap.md` el procedimiento de
re-import para futuros developers.

**Por que P4 (no P1)**: el FULL CUDA y el FarSLIP no dependen de DVC para
producir resultados; bloqueariamos infra que no es critica del modelo. DVC
viene despues como housekeeping de reproducibilidad cross-machine.

**Criterios de aceptacion P4**:

| AC | Criterio | Como verificar |
|----|----------|----------------|
| AC-1 | 6 artefactos DVC versionados + tags creados | `git tag -l` muestra los 6 tags + `dvc list . data/` los lista |
| AC-2 | `dvc pull` en checkout limpio reconstruye con MD5 identico | clone en `/tmp` + `make dvc-pull` + `md5sum` comparado |
| AC-3 | `terraform state list` incluye `module.gcp.google_storage_bucket.tfstate` | `terraform state list \| grep tfstate` |
| AC-4 | `terraform plan` reporta 0 cambios despues del import | output limpio |
| AC-5 | `docs/us-resolved/us-016.md` creado + US-016 marcada `closed` | grep `Status: closed` |

---

## P5 — Rama semantica Gemini 3.5 Flash real

**Que**: ejecutar el pipeline de la rama semantica fenologica sobre un
subset estratificado (~216 parcelas, 12 por clase x 18 clases efectivas)
para validar D-4 del plan US-022b (ablation con vs sin bloque `pheno_text_*`).

**Pre-requisitos** (verificados 2026-05-23):
- [x] `google-genai 2.6.0` instalado
- [x] API publica habilitada (`generativelanguage.googleapis.com`)
- [x] `GEMINI_API_KEY` en `.env.local`
- [x] Cliente `_default_google_genai_client` smoke verde (5s, respuesta agronomica correcta)

**Etapas** (0.5 dia, ~$0.20 USD):

```python
# Subset estratificado
from ml.train.baseline import _load_baseline_dataset, _prepare_dataframe
import polars as pl

df = _prepare_dataframe(_load_baseline_dataset(
    "data/test_fixtures/feature_selection_parcels_subset.parquet"
))
subset = df.group_by("class_id").agg(
    pl.all().sample(n=12, seed=42, with_replacement=False)
).explode([c for c in df.columns if c != "class_id"])

# Generar descripciones + cache por parcel_id
from ml.features.phenology_description import build_phenology_text_block
text_block = build_phenology_text_block(
    parcel_ndvi_frame=subset,
    model="gemini-3.5-flash",
    cache_dir=Path("data/cache/phenology_descriptions"),
    skip_llm=False,
)
# Output: 216 x 385 (parcel_id, year, pheno_text_000..pheno_text_383)
```

```bash
# Persistir para fusion
mkdir -p data/features
mv subset_pheno_text.parquet data/features/phenology_text_italy.parquet

# Re-correr notebook 05 con RUN_SEMANTIC_BRANCH=True
poetry run papermill notebooks/feature_engineering/05_reencuadre_fenologico.ipynb \
  notebooks/feature_engineering/05_reencuadre_fenologico.ipynb \
  -p MAX_SAMPLES 216 -p RUN_SEMANTIC_BRANCH True
```

**Criterios de aceptacion D-1..D-5** (del plan US-022b §3.4):

| AC | Criterio |
|----|----------|
| D-1 | Generador determinista (`temperature=0` + cache por hash NDVI) |
| D-2 | Text-encoder → vector denso 384-dim |
| D-3 | `pheno_text_*` en fusion.py (`EXPECTED_COL_COUNT_WITH_PHENO_TEXT = 573`) |
| D-4 | Ablation con vs sin pheno_text reportada en `ablation_table.parquet` |
| D-5 | Costo Gemini < $10 USD documentado en `docs/l4_log.md` |

**Por que P5 (no P3)**: el resultado D-4 es **publicable en ambos sentidos**
y no bloquea el cierre de US-022b. Es el primer candidato a diferir al Paper
Track (R1/R3 del plan canonico).

---

## Cierre US-022b → PR a `develop`

Cuando P1-P5 esten cerrados:

- [ ] Mergear `us-022-b` → `develop` con PR (Conventional Commit `feat(E5): cierra US-022b — deuda FarSLIP + infra L4 + reencuadre fenologico + rama semantica`)
- [ ] Crear `docs/us-resolved/us-022b.md` con resumen ejecutivo + tabla final de resultados (XGB + TempCNN + InceptionTime + opcional FarSLIP + opcional pheno_text)
- [ ] Marcar `docs/us-handoff/us-022b.md` como `ready-to-close` → `closed`
- [ ] Actualizar `context/RefinamientoPlaneacionAgroSatCopilot_v6.md` referenciando US-022b + ADR-006 en EPIC 3/4/5
- [ ] Deuda US-017 marcada como **saldada** (Fase 4 cerrada via P1)
- [ ] Reutilizar los 3 checkpoints `.pt` (XGB no aplica, TempCNN + InceptionTime + opcional FarSLIP student) como base learners del Stacking ensemble en EPIC 6 (Avance 5, 31-may)

---

## Referencias

- Handoff US-022b: [`docs/us-handoff/us-022b.md`](../us-handoff/us-022b.md)
- Plan canonico US-022b: [`docs/us-planning/us-022b.md`](../us-planning/us-022b.md)
- ADR-006 (Aceptada): [`docs/decisions/ADR-006-reencuadre-baseline-fenologico.md`](../decisions/ADR-006-reencuadre-baseline-fenologico.md)
- Paper-faro: Wen et al. (2025), DOI 10.1016/j.isprsjprs.2025.07.002
- Manual-test US-016 (comandos exactos para versionar): [`docs/manual-test/us-016.md`](../manual-test/us-016.md)
- Azure (fuera de scope, vive aparte): [`us-019-2-azure-pendiente.md`](us-019-2-azure-pendiente.md)
