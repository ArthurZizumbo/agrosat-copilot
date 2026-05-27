# L4 usage log

Registro de uso de GPU L4 spot en GCP Vertex AI custom-jobs. Equivalente al
`h100_log.md` planificado para Azure H100 (US-024). Sirve para:

- Auditoria FinOps (gate A-6 de US-022b-A): nunca dejar L4 colgada.
- Trazar SP consumidos vs presupuesto (10-12 h L4 para US-022b completo, ADR-006).
- Reconstruir reproducibilidad: cada entrada referencia el run MLflow y el SHA
  de la imagen `ml-train` usada.

## Convenciones

- `Job ID`: ultimo segmento de `gcloud ai custom-jobs describe NAME --region=R --format='value(name)'`.
- `Display name`: `train-<EPIC>-<US>[-smoke]`.
- `Image SHA`: `${SHORT_SHA}` con el que Cloud Build construyo `ml-train`.
- `Duracion`: minutos efectivos (no incluye queue ni cold start MLflow).
- `Costo estimado`: g2-standard-8 spot L4 ~$0.30 USD/h on-demand,
  ~$0.10 USD/h spot (price aproximado may-2026). Verificar en
  `gcloud billing` mensualmente.
- `MLflow run`: link relativo o run_id si el server esta accesible.

---

## 2026-05-24 09:36 UTC — Quota L4 spot APROBADA

`gcloud alpha services quota list --service=aiplatform.googleapis.com --consumer=projects/agrosat-copilot --filter="metric:custom_model_training_preemptible_nvidia_l4_gpus"` confirma:

| Region | defaultLimit | effectiveLimit |
|--------|--------------|----------------|
| us-central1 | (no default) | **1** (approved) |
| us-east1/east4/west1/west2/west4 | None | None |

Quota on-demand `custom_model_training_nvidia_l4_gpus` us-central1 = 1 idem. Reanudacion P1 etapas 3b -> 7 desencadenada inmediatamente.

## 2026-05-24 09:52 UTC — US-022-c P1 etapa 3b smoke L4 REAL (post-quota)

### Configuracion

- YAML: `ml/configs/l4_smoke.yaml` revertido a `machineType: g2-standard-8 + NVIDIA_L4 x1` (parche T4 retirado).
- Fix shape TempCNN aplicado: `X=(B,T,C)=(64,24,10)` directo + `model(X)` sin `.transpose(1,2)` manual (TempCNN.forward hace el transpose internamente).
- `bootDiskSizeGb=100`, `bootDiskType=pd-ssd`, `scheduling.timeout: 1200s`, `strategy: SPOT`.
- MLflow URI: `https://agrosat-mlflow-dev-gox5zzl7wa-uc.a.run.app` con ID token desde metadata server.

### Jobs lanzados

| Job ID | Display name | Machine | Estrategia | Estado | Comentario |
|--------|-------------|---------|------------|--------|------------|
| `7359114595530702848` | `train-E5-US-022b-smoke` | g2-standard-8 + L4 | SPOT | CANCELLED | 22 min en PENDING sin asignacion (sin capacidad L4 spot us-central1 en la ventana 09:52-10:14 UTC). Cancelado manualmente. |
| `57653699656286208` | `train-E5-US-022b-smoke` | g2-standard-8 + L4 | STANDARD (on-demand) | (en curso al cierre del agente) | Lanzado tras editar `l4_smoke.yaml: strategy: STANDARD` para validar la cadena rapido. Costo on-demand L4 g2-standard-8 ~$0.71/h x 0.2h max = ~$0.14. |

Console URL smoke v2: https://console.cloud.google.com/vertex-ai/training/custom-jobs/locations/us-central1/training-pipelines/57653699656286208?project=agrosat-copilot

Comandos manejo (smoke v2):

```bash
# Monitor estado
gcloud ai custom-jobs describe 57653699656286208 --region=us-central1 --project=agrosat-copilot --format='value(state,error.message,startTime,endTime)'

# Logs
gcloud ai custom-jobs stream-logs 57653699656286208 --region=us-central1 --project=agrosat-copilot

# Cancelar
gcloud ai custom-jobs cancel 57653699656286208 --region=us-central1 --project=agrosat-copilot
```

### Dataset farslip_pairs subido a GCS

- Local: `data/farslip_pairs/` 15 GB, 30176 files (pianura_padana 10112 + puglia ? + toscana ?).
- Remoto: `gs://agrosat-artifacts-dev/datasets/farslip_pairs_v1/` 21.4 GB.
  - **Nota**: hay un subdirectorio espurio `gs://.../datasets/farslip_pairs_v1/farslip_pairs/` (13.4 GB) producto de un rsync con doble prefix. **No afecta** al training job (descarga solo prefix `datasets/farslip_pairs_v1/{pianura_padana,puglia,toscana}/`). Limpieza opcional:
    ```bash
    gcloud storage rm -r gs://agrosat-artifacts-dev/datasets/farslip_pairs_v1/farslip_pairs/
    ```
- Sumario de tipos en GCS: `pianura_padana/` (10112 tif + 1 manifest.parquet), `puglia/`, `toscana/` y `farslip_pairs/` (basura).

### YAML l4_spot.yaml actualizado para Etapa 5

- `scheduling.timeout: 25200s` (7h, antes 21600s = 6h) — incluye ~30 min descarga dataset.
- Container args hidrata dataset al disco `/app/data/farslip_pairs` via `google.cloud.storage` SDK Python con ThreadPoolExecutor(max_workers=32).
- CLI invocado: `python -m ml.farslip.train --rois italy --epochs 4 --batch-size 64 --lr 1e-5 --seed 42 --output-dir /app/artifacts/farslip --gcs-output-uri gs://agrosat-artifacts-dev/vertex-jobs/farslip/farslip-clip-italy-v1/ --dataset-root /app/data/farslip_pairs --teacher-model-id openai/clip-vit-base-patch16 --time-cap-hours 6.5`.
- MLflow tags `epic/us/data_version/code_version` los registra el trainer dentro de `ml/farslip/distill.py:558-572` (no pasan por CLI).

## 2026-05-24 — Backlog historico — US-022-c P1 etapa 3b smoke (cadena validada, GPU bloqueada por quota)

### Tabla resumen de jobs lanzados

| Job ID | Display name | Machine | Estado | Causa | Duracion |
|--------|-------------|---------|--------|-------|----------|
| `566771563781685248` | `train-E5-US-022b-smoke` | n1-standard-8 + T4 SPOT | FAILED | driver CUDA 12020 host vs imagen CUDA 13.0 (`torch.cuda.is_available()=False`) | ~15 min queue + 30s run |
| `8937485527435706368` | `train-E5-US-022b-smoke-cpu` | n1-standard-4 (sin GPU) | FAILED | MLflow `403 Forbidden` (faltaba `roles/run.invoker` para `ml-train-runner-sa`) | ~90s |
| `1595492234914955264` | `train-E5-US-022b-smoke-cpu-retry` | n1-standard-4 (sin GPU) | FAILED | `RuntimeError` shape TempCNN (input incorrecto, bug en el script smoke) | ~90s |
| `1759873621313978368` | `train-E5-US-022b-smoke-cpu-v3` | n1-standard-4 (sin GPU) | FAILED | mismo bug shape TempCNN | ~30s |
| `8053654100564246528` | `train-E5-US-022b-smoke-cpu-v4` | n1-standard-4 (sin GPU) | FAILED | mismo bug shape TempCNN (TempCNN.forward hace transpose interno y se confunden las dos versiones) | ~70s |

**Costo total real estimado**: < $0.10 USD (5 jobs CPU + 1 T4 cortos).

### Logros parciales validados end-to-end

- Cloud Build `ml-train:latest` (5.65 GB CUDA 13.0 + torch cu130 + breizhcrops) en Artifact Registry: **OK**.
- Cloud Build `mlflow:latest` con wrapper `mlflow-start.sh` + Cloud SQL socket: **OK**.
- Cloud Run `agrosat-mlflow-dev` revision `00005-*` READY=True: **OK**.
- Cloud SQL `agrosat-pg-dev` user `agrosat` con password sync con Secret Manager: **OK**.
- Vertex AI custom-jobs → metadata server → ID token con audience=Cloud Run URL: **OK** (`ID token len=821`).
- IAM `roles/run.invoker` en Cloud Run `agrosat-mlflow-dev` para `ml-train-runner-sa@agrosat-copilot.iam.gserviceaccount.com`: **OK** (aplicado en esta sesion 2026-05-24 08:34 UTC — el prompt afirmaba que ya estaba pero `gcloud run services get-iam-policy` mostro etag vacio).
- MLflow run real creado desde Vertex job CPU: **OK**, run `fd1f4db41ebf4466b9ccfc23bde3e22d` en experiment `1` (`us-022b-a-smoke`), URL `https://agrosat-mlflow-dev-gox5zzl7wa-uc.a.run.app/#/experiments/1/runs/fd1f4db41ebf4466b9ccfc23bde3e22d`.

### Bloqueante critico — Quota Vertex AI L4 = 0 (todas regiones)

`gcloud alpha services quota list --service=aiplatform.googleapis.com --consumer=projects/agrosat-copilot --filter="metric:custom_model_training_*nvidia_l4_gpus"` confirma:

- `custom_model_training_preemptible_nvidia_l4_gpus`: **0 / 0** en todas las regiones US y EU (default=0, effective=0).
- `custom_model_training_nvidia_l4_gpus` (on-demand): **0 / 0** idem.
- Vertex AI T4 SPOT `custom_model_training_preemptible_nvidia_t4_gpus`: 1 / 1 en `us-central1, us-east1, us-west1, europe-west2, europe-west4` — pero T4 (16 GB VRAM) NO sirve para FarSLIP student (necesita 18-20 GB) ni para entrenamiento real, solo smoke.
- Driver T4 del host Vertex (CUDA 12020 / 12.2.x) es incompatible con la imagen `ml-train:latest` (compilada para CUDA 13.0 + torch cu130). Para usar T4 habria que rebuildear toda la imagen contra CUDA 12 o downgrade torch, lo cual rompe alineacion con el Dockerfile y la rama `us-022-b`.

**Accion requerida (Arthur, fuera de scope auto-mode)**:

1. Abrir [GCP Console → IAM & Admin → Quotas](https://console.cloud.google.com/iam-admin/quotas?project=agrosat-copilot).
2. Filtrar por `Service: Vertex AI API` + `Quota: Custom model training Nvidia L4 GPUs per region` y por `Preemptible Custom model training Nvidia L4 GPUs per region`.
3. Solicitar `+1` en region `us-central1` para ambas (SPOT preferida, on-demand fallback). Tiempo aprobacion tipico: 1-2 dias habiles.
4. Una vez aprobada, ejecutar:

   ```bash
   export MLFLOW_TRACKING_URI="https://agrosat-mlflow-dev-gox5zzl7wa-uc.a.run.app"
   # Restaurar smoke a L4 (revertir downgrade a T4 en ml/configs/l4_smoke.yaml):
   #   machineType: g2-standard-8
   #   acceleratorType: NVIDIA_L4
   make train-l4-smoke   # AC-5 real
   # Si OK, lanzar FarSLIP:
   make train-l4 epic=E3 us=US-017 script=farslip/train_student.py
   ```

### Smoke bug residual (no bloqueante para reanudacion L4)

El script `ml/configs/cpu_smoke.yaml` (y por extension `l4_smoke.yaml`) tiene mismatch
de shape con `breizhcrops.models.TempCNN.forward`: la version instalada
hace `x.transpose(1,2)` interno antes de `conv_bn_relu1`. El input debe ser
`(B, T=sequencelength=24, C=input_dim=10)` SIN `.transpose(1,2)` antes de pasarlo al
`model(...)`. Mi yaml actual usa `(B, C=10, T=24)` que despues del transpose interno
queda `(B, T, C) = (B, 24, 10)`, lo que conv1d interpreta como `(B, C=24, L=10)` y
falla porque espera `C=input_dim=10`. Fix trivial al cierre del bloqueante L4:

```yaml
# ml/configs/cpu_smoke.yaml + l4_smoke.yaml — fix shape
X = torch.randn(64, 24, 10, device=device)  # (B, T, C) — TempCNN.forward hace transpose
...
logits = model(X)  # sin transpose manual
```

---

## Backlog historico previo

### 2026-05-22 — US-022b-A AC-5 smoke (planificado, no ejecutado)

Bloqueado por billing/credenciales hasta 2026-05-24. Ver entrada de arriba.

### Siguientes entradas (placeholders post-L4-quota)

- ETA + 2 dias post-aprobacion quota L4 — US-022-c P1 FarSLIP student italia v1 (~6h L4 spot, ~$1.7 USD).
- ETA + 2 dias post-aprobacion quota L4 — US-022b-C TempCNN + InceptionTime spatial CV (~3 h, ~$0.30 USD).

---

## us-022-c P5 — Rama semantica Gemini 3.5 Flash real (2026-05-23)

**Job ID**: n/a (CPU local, sin Vertex AI job)
**Modelo**: `gemini-3.5-flash` via Gemini API publica (NO Vertex AI; allowlist
preview).
**Provider**: `google-genai` 2.6 (cliente nativo, `thinking_level="minimal"`).
**Auth**: `GEMINI_API_KEY` desde `.env.local` (no commiteada).

**Parametros**:
- 216 parcelas estratificadas (12 por clase × 18 clases efectivas PASTIS-R)
- `temperature=0.0` (R7 enforced)
- `max_output_tokens=512`
- Cache: `data/cache/phenology_descriptions/` por hash sha256(parcel_id+curve+model+prompt_v1)

**Wall clock**: 559.4 s (~9.3 min) — primer run sin cache.
**Token usage estimado** (sin metricas oficiales del cliente):
- ~216 requests
- ~800 tokens input/request, ~150 tokens output/request
- Total: ~173 K input + ~32 K output tokens

**Costo real estimado** (Gemini 3.5 Flash pricing 2026):
- Input: 173 K × $0.075/M = $0.013
- Output: 32 K × $0.30/M = $0.010
- **Total: ~$0.023 USD** (muy por debajo del cap $0.20 / hard cap $10)

**Artefactos**:
- `data/features/phenology_text_italy.parquet` shape (216, 386): 2 metadata
  (parcel_id, year) + 384 dims `pheno_text_000..pheno_text_383` (encoder
  `sentence-transformers/all-MiniLM-L6-v2`).
- 216 archivos JSON cache en `data/cache/phenology_descriptions/{hash16}.json`.
- `reports/baseline/reencuadre_fenologico/ablation_table_pheno_text.parquet`:
  ablation `full` (F1=0.21) vs `with_pheno_text` (F1=0.09) sobre subset 216
  con XGB + spatial 3-fold (k=3 por low-data, b=0.5km).

**Conclusion P5**: bloque `pheno_text_*` funcional end-to-end. En este subset
acotado (216 parcelas, k=3 folds) la rama semantica NO supera al baseline `full`
(delta -0.12), pero el resultado es honesto y reportable. Escalar a 85951
parcelas requiere presupuesto Gemini ~$10 USD + queda como item EPIC 6.

---

## 2026-05-25 — US-023-preview P4 — Ampliacion `pheno_text` SKIP HONESTO

**Plan**: ampliar el bloque `pheno_text_*` de 216 (US-022-c P5) a >=1000 parcelas
balanceadas con Gemini Flash 3.5 para validar/refutar la rama semantica
definitivamente. Presupuesto previsto: <= $5 USD (holgura 50x sobre el smoke US-022-c).

**Estado**: SKIP HONESTO documentado en `notebooks/baseline/05_reencuadre_fenologico.ipynb`
§3.3. Razones:

1. `GEMINI_API_KEY` / `GOOGLE_API_KEY` no estan configuradas en el entorno
   local de Fase 3 de US-023-preview (solo-dev environment); la llamada real
   queda bloqueada por la pre-condicion documentada en la celda de bootstrap.
2. El bloque actual `data/features/phenology_text_italy.parquet` (216 parcelas,
   shape `(216, 386)`) permanece valido y la ablation lo reusa via cache JSON
   por parcela (`data/cache/phenology_descriptions/{hash16}.json`).
3. La logica de ampliacion esta lista en `ml/features/phenology_description.py:
   build_phenology_text_block(max_parcels=1000, skip_llm=False)` — un solo
   comando con la API key configurada genera los 784 textos faltantes.

**Costo cloud incurrido por P4 (esta corrida)**: **$0.00 USD** (skip honesto).

**Deuda P4** (item backlog US-024 o cierre US-023-preview-followup):

- Configurar `GEMINI_API_KEY` en `.env.local` + ejecutar
  `poetry run python -c "from ml.features.phenology_description import build_phenology_text_block; ..."`
  sobre subset estratificado >= 1000 parcelas balanceadas.
- Esperado: ~5x mas requests que US-022-c P5 (216 -> 1000+) ~ $0.10 USD; con
  holgura por reintentos quedaria en $0.20-$0.50 USD (cap $5 USD se respeta).
- Decision tras ampliar: promover al baseline si delta >= +0.01, o mantener
  como base learner del stacking EPIC 6 si no aporta senal.

**Conclusion P4**: el skip es **reportable y honesto** — la rama semantica
sigue en el subset US-022-c sin re-evaluarse en esta US. La decision tecnica
queda diferida sin bloquear US-023-preview.

---

## 2026-05-26 11:48 UTC — US-023-preview P4 — Ampliacion `pheno_text` EJECUCION REAL

**Plan**: revertir el SKIP HONESTO previo. `GEMINI_API_KEY` SI estaba configurada
en `.env.local` (39 chars, prefijo `AIzaSy`); la sesion anterior no habia cargado
`python-dotenv`. Esta corrida lo carga y ejecuta el bloque end-to-end.

**Estado**: COMPLETADO real (no skip). Resultado replicable desde
`scripts/us023_p4_pheno_text_ablation.py`.

### Configuracion

- Dataset full: `data/test_fixtures/feature_selection_parcels_subset.parquet` (85951 parcelas x 192 cols, 18 clases efectivas).
- Subset balanceado: 60 parcelas por clase x 18 clases = **1080 parcelas** (cumple AC-P4-2: `>= 1000`).
- Modelo Gemini: `gemini-3.5-flash` via cliente `google-genai` (env `AGROSAT_LLM_PROVIDER=google-genai`).
- Text-encoder: `sentence-transformers/all-MiniLM-L6-v2` -> 384 dim.
- Ablation: XGBoost CUDA, spatial CV 5-fold, buffer 1 km, seed 42 (mismo splitter cacheado de US-022-b).
- Subsets evaluados: `full` (185 features sin `geom_*`), `with_pheno_text` (185 + 384 = 569), `pheno_text_only` (384).

### Resultados

| Subset             | n_features | f1_macro  | f1_weighted | mIoU     | delta_vs_full |
|--------------------|-----------:|----------:|------------:|---------:|--------------:|
| full               |        185 | 0.328598  | 0.328598    | 0.212026 | -             |
| with_pheno_text    |        569 | 0.293236  | 0.293236    | 0.187010 | **-0.035362** |
| pheno_text_only    |        384 | 0.074728  | 0.074728    | 0.039618 | -0.253870     |

### Costos Gemini Flash 3.5

- `n_requests` reales (no cache): **918**
- `n_cache_hits` (reutilizados US-022-c P5 + corrida): **162**
- `tokens_in` estimados: **275,400** (chars_in / 4, prompt ~1200 chars x 918 calls)
- `tokens_out` estimados: **164,641** (chars_out reales / 4)
- Tarifas Gemini 2.5/3.5 Flash: input $0.30/1M tok, output $2.50/1M tok
- `cost_in` = $0.0826 USD · `cost_out` = $0.4116 USD
- **`cost_usd_total` = $0.4942 USD** (cumple AC-P4-4: `<= $5 USD`)
- Wall clock fase pheno descriptions: 2681 s (~45 min)
- Wall clock total (incluye encoding + ablation 5-fold x 3 sets en RTX 4070): **3449 s (~57 min)**

### Decision (AC-P4-5)

`delta_pheno_text_vs_full = -0.0354` < -0.01 -> **DEUDA US-024** (escalar a full
85951 parcelas con presupuesto adicional). El bloque `pheno_text_*` con encoder
sentence-transformers MiniLM y subset 1080 NO supera al baseline tabular en F1-macro.

Posibles causas estructurales (no se investigan aqui, quedan como hipotesis para US-024):

1. Encoder MiniLM no se beneficia de la senal agronomica especifica; un text-encoder
   contrastivo entrenado en remote sensing (FarSLIP CLIP, US-017) podria capturar mejor
   la firma fenologica.
2. La cardinalidad 1080 puede ser insuficiente para que XGBoost aprenda 384 dimensiones
   nuevas sobre 18 clases (overfitting en folds pequenos: fold 4 cayo a 0.094 en `with_pheno_text`).
3. Las descripciones Gemini son **redundantes** con las features fenologicas ya presentes
   (sog_doy, peak_doy, ndvi_auc, etc.) que el prompt explicitamente lista — el LLM no
   anade informacion nueva, solo la verbaliza.

### Decision recomendada para EPIC 6 (stacking)

Mantener `pheno_text_*` como **base learner del stacking** (no descartar): un meta-learner
puede aprovechar la calibracion diferencial del bloque incluso si su F1 marginal es bajo.
Recomendado entrar al stacking con peso bajo (e.g., voting weight 0.1 vs XGB peso 0.5).

### Artefactos persistidos

- `data/features/phenology_text_italy.parquet` -- shape `(1080, 386)` (parcel_id + year + 384 cols `pheno_text_*`). DVC tracking pendiente (gate DVC).
- `reports/baseline/feature_ablation/ablation_table_pheno_text_v2.parquet` -- 3 filas (`full`, `with_pheno_text`, `pheno_text_only`).
- `reports/baseline/feature_ablation/us023_p4_summary.json` -- summary completo.
- `data/cache/phenology_descriptions/*.json` -- 1080+ archivos JSON cache (reutilizables sin nuevo costo Gemini).
- MLflow run: `02d979a6b48042ac82a7b15c6ec304ac` (experimento `baseline-pheno-text-ablation`, tracking `file:./mlruns`).
- Script reproducible: `scripts/us023_p4_pheno_text_ablation.py`.
- Log corrida: `scripts/us023_p4.log`.

### Costo cloud incurrido P4 (acumulado historico)

| Corrida           | Fecha       | n_parcels | n_requests | cost_usd  |
|-------------------|-------------|----------:|-----------:|----------:|
| US-022-c P5 smoke | 22-may-2026 |       216 |        216 | 0.023     |
| US-023-preview P4 | 26-may-2026 |      1080 |        918 | **0.494** |
| **Total**         |             |           |       1134 | **0.517** |

Holgura sobre cap $5 USD: ~9.7x. US-024 escalando a 85951 estimaria ~$0.494 * (85951/1080) = ~$39 USD,
fuera del cap actual; quedaria como propuesta de presupuesto adicional.
