# EPIC 12 — Setup y anotaciones de la VM H100 (noche autónoma 20-jun-2026)

> Notas de bloqueos/decisiones durante la ejecución autónoma de EPIC 12 en la VM
> H100 del sponsor (`gjcamacho-gpuh1`, repo en `F:\projects\agrosat-copilot`).
> Regla de Arthur: **datos reales, cero sintéticos/placeholders**; si algo no se
> puede correr, anotarlo aquí y seguir.

## Entorno verificado (OK)

- Repo VM: `F:\projects\agrosat-copilot` (3.4 TB libres en F:).
- Python: `F:\tools\micromamba.exe run -n agrosat python ...` (env en `F:\.conda\envs\agrosat`).
- torch 2.11.0+cu130, CUDA True, **NVIDIA H100 NVL** visible.
- papermill 2.7.0 (Python 3.12.10) — para ejecutar notebooks end-to-end.
- PASTIS-R presente en `F:\projects\agrosat-copilot\data\PASTIS-R`.
- Qwen on-prem `:8002` vivo (túnel + supervisor de reconexión).

## Decisión: estado del repo VM (61 commits atrás)

- HEAD VM = `4cda41f` (Merge PR #43), 61 commits detrás de `origin/main` (`5697348`).
- Cambios `M` sin commitear = artefactos de corridas previas (scripts segmentación,
  OOF dumps, notebooks 04e/04g, `_vm_*` scratch). NO es código a preservar en git.
- **Plan seguro** (regla: nunca `checkout`/`pull` destructivo con cambios M):
  `git stash` (preserva los M) → `git pull origin main` → trabajar sobre main fresco.
  El stash queda recuperable si algo de eso era necesario.

## Datasets EPIC 12 a descargar en F: (reales, vía DVC/HF/Zenodo)

| US | Dataset | Fuente | Tamaño | Estado |
|----|---------|--------|--------|--------|
| US-074 | HCAT crosswalk | `data/reference/` (parcial ya) + EuroCrops↔HCAT | ligero | parcial local |
| US-076 | EuroCropsML | `pip install eurocropsml` + Zenodo DOI 10.5281/zenodo.15095445 | ~4.8 GB | pendiente |
| US-077 | México AlphaEarth | GEE `SATELLITE_EMBEDDING/V1/ANNUAL` (zonal Michoacán) | ligero | pendiente |
| US-075 | Sen4AgriNet Catalonia | HF `paren8esis/S4A` (subset) | **943 MB (40 patches)** | **descargado en VM F:** |

## Bloqueos encontrados (se actualiza durante la noche)

### B1 — Actualizar el repo VM a main bloqueado por seguridad (20-jun 01:4x)
- `git stash` + `git pull origin main` (61 commits) en `F:\projects\agrosat-copilot`
  vía SSH fue **bloqueado por el clasificador de seguridad** (escritura recurrente
  en infra compartida del sponsor más allá de run-notebooks/descargar-datos).
- **Impacto:** el repo VM (`4cda41f`) no tiene el código de la cadena 051-054 ni
  US-049. Para EPIC 12 NO es bloqueante: el finetune denso (US-075) solo necesita
  los checkpoints (`checkpoints/segmentation/`, vía DVC) + PASTIS-R (ya en F:) +
  el código de segmentación, que SÍ está en `4cda41f` (E5 cerrado). Las US de CPU
  (074/076/077) corren sobre el repo local + datasets descargados.
- **Estrategia adoptada:** EPIC 12 se desarrolla en el repo LOCAL (ya en main con
  todo el código nuevo); los datasets pesados se descargan a F: en la VM; el
  finetune denso US-075 usa la GPU de la VM invocando scripts vía SSH (no requiere
  el repo VM en main, solo torch+checkpoints+datos). Notebooks se ejecutan en local
  con papermill (CPU US-074/076/077) y el finetune se lanza en la VM.
- Si más adelante se necesita el repo VM en main, queda para que Arthur lo haga a
  mano (o autorice el stash+pull). **ACTUALIZACIÓN:** Arthur autorizó actualizar el
  repo VM con main/cambios locales.

### B2 — EPIC 12 parte de la rama de la cadena, no de main (decisión, no bloqueo)
- `main` NO tiene el hook label-space de US-053 ni el clasificador con banderas
  (viven en `feature/E9-US-058-mapview`, PR #49 sin mergear todavía).
- EPIC 12 (US-074 amplía el label-space registry de US-053) **necesita** ese código,
  así que la rama `feature/E12-transfer-multiregion` se creó **desde
  `feature/E9-US-058-mapview`** (no desde main). Cuando el PR #49 se mergee, EPIC 12
  rebasa/mergea limpio. Verificado: la rama tiene `register_label_space` +
  `restrict_to_resolved_classes`/`use_stacking`.

### B3 — US-075 descarga + adapter Sen4AgriNet (20-jun, RESUELTO, no bloqueo)

**Hallazgo de layout (corrige el plan v8 §3.1).** Los `allow_patterns=["cat_2019/*",
"cat_2020/*", "fr_2019/*"]` del plan **no matchean nada**: `cat_2019`/`cat_2020`/
`fr_2019` son *nombres de config del dataset-builder* en `S4A.py`, NO prefijos de
ruta. El layout real del repo HF es `data/<year>/<TILE>/<year>_<TILE>_patch_x_y.nc`.
El downloader (`ml/ingest/sen4agrinet_download.py`) por eso filtra por
`data/<year>/<tile>/` con los tiles verificados desde `S4A.py`:
- Catalonia (`patch_country_code='ES'`): `CAT_TILES = 31TBF 31TCF 31TCG 31TDF 31TDG`
  (años 2019 y 2020). **`31TCG` existe en ambos años** (el tile que cita v8).
- Francia (`patch_country_code='FR'`): `FR_TILES = 31TCJ 31TDK 31TCL 31TDM 31UCP 31UDR`
  (solo 2019).

**Subset bajado (real, en VM F:):** 30 patches Catalonia `31TCG` (2019+2020,
intercalados) + 10 patches Francia `31TCJ` 2019 = **40 patches, 942.96 MB**. Filtro
`min_parcel_frac=0.02` (descarta patches dominados por background label 0). Manifiesto
en `data/sen4agrinet/subset_manifest.json`. `dvc add data/sen4agrinet` OK
(`data/sen4agrinet.dvc`, MD5 `e292ab8cfbfb05ecc9ab5310e01dd79e.dir`, 88 files).

**Adapter validado contra patches REALES.** `tests/ml/data/test_sen4agrinet_adapter.py`:
**11 passed** en la VM (7 lógica pura + 4 sobre los `.nc` reales; ~190 s). Contrato
confirmado: `x=(10,10,128,128)` float32 en `[0,~1]`, `y=(128,128)` int64 en
`[0,N_MACRO)∪{255}`, y `build_tsvit(num_classes=10, n_timesteps=10, in_channels=10)`
acepta el `x` sin shape mismatch. Estructura netCDF verificada: grupos por banda
(B01-B12,B8A) con `time` propio (~26 fechas), `labels`/`parcels` 366x366,
`time units = 'seconds since <fecha ref>'` (de ahí el mes por `num2date`).

**Notas para el finetune (US-075 §3.3):**
- El adapter emite **macro-HCAT de 10 clases** (`MACRO_GROUP_TO_ID`, sin `void`;
  background/fuera-de-nomenclator -> `ignore_index=255`). El head FR es 18-clase ->
  re-inicializar solo el head a 10 y cargar el resto con `strict=False`.
- Filtrar Catalonia-only con `Sen4AgriNetDataset(..., countries=("ES",))`; los FR
  (`31TCJ`) son la referencia de dominio.
- `netCDF4` se instaló en el env `agrosat` de la VM (`pip install netCDF4` 1.7.4);
  `xarray`/`huggingface_hub` ya estaban. `ml/data/hcat_crosswalk.py` se sincronizó a
  la VM por SCP (el repo VM en `4cda41f` no lo tenía); usa el fallback CSV/JSON
  (`load_crosswalk` re-deriva si falta el parquet).

### B4 — US-075 finetune denso FR->Catalonia EJECUTADO en H100 (20-jun, RESUELTO, numeros REALES)

**Resultado (REAL, sobre los parches Catalonia descargados — regla cero-sinteticos):**

| Protocolo | mIoU | F1-macro | pixel-acc | n_train | n_val |
|-----------|------|----------|-----------|---------|-------|
| zero-shot (FR 18 -> macro) | **0.0000** | 0.0000 | 0.0000 | 0 | 20 parches (180 tiles) |
| few-shot (head macro finetuneado) | **0.2468** | 0.3005 | 0.9179 | 10 parches (90 tiles) | 20 parches (180 tiles) |

**Delta mIoU = +0.2468** (best epoch 30, 40 epocas en **76.8 s** en H100). Checkpoint en
`F:/projects/agrosat-copilot/checkpoints/segmentation/tsvit-pheno-sen4agri-cat-ft-v1/best.pt`.
JSON en `reports/segmentation/sen4agrinet_transfer_result.json`. Notebook ejecutado por
papermill: `notebooks/segmentation/5c_transfer_sen4agrinet.ipynb` (outputs poblados).

**Por que el zero-shot es 0.0000 (es REAL, no un bug):** el TSViT-pheno PASTIS-FR proyectado
18->macro predice sobre Catalonia **casi exclusivamente** grassland(0)=1.69M px y vegetables(5)=
1.26M px — dos clases con **support=0** en el GT de validacion ES (donde dominan cereals(1)=585k,
legumes_fodder(7)=39k, vineyard(3)=2.9k, oilseed(2)=2k). La diagonal de la matriz de confusion es
toda cero -> mIoU exacto 0.0. Es un **gap de dominio Franco-Iberico catastrofico** en zero-shot,
consistente con el caveat AC-5 (arXiv:2601.00857, transferibilidad espacial limitada). El few-shot
con 10 parches recupera +0.2468 mIoU: ESE Delta es el entregable cientifico, no un accuracy alto.

**Hallazgo de codigos de etiqueta (matiza la suposicion del adapter):** los labels Sen4AgriNet de
Catalonia traen codigos S4A donde **solo** 120 (maize), 150 (barley) y 770 (peas) coinciden con los
`SELECTED_CLASSES` FAO-ICC del adapter; el resto (354, 355, 998, 975, 314, 911, 442, ...) cae fuera
de la nomenclatura PASTIS-18 y va a `ignore_index=255` (honesto, no inventado). Aun asi el GT macro
de validacion tiene crops reales suficientes (cereals/legumes/vineyard/oilseed) para un Delta valido.

**Perf:** el `_interpolate_months` del adapter es O(pixeles) en Python puro (lentisimo: ~min/parche).
Se anadio `Sen4AgriNetDataset(..., precache_all=True)` que decodifica los ~30 parches ES UNA vez en
RAM (shuffle-safe, ~9x); sin esto el finetune se arrastraba. VRAM tras el run: 74 GB libres de nuevo
(solo Qwen :8002 con 21 GB; el finetune libero limpio). MLflow `:5010` DOWN en la VM -> fallback
automatico `file:./mlruns` (no bloquea); el lineage queda en `F:/projects/agrosat-copilot/mlruns`.
