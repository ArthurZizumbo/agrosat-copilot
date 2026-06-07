# Dagster Sub-Agente — AgroSatCopilot

> Sobreescribe al orquestador root para `dagster_project/`. Las reglas globales (Polars no pandas, secrets, DRY, MLflow `data_version`+`code_version`, DVC, sin emojis, idioma) viven en [`../AGENTS.md`](../AGENTS.md) — NO se repiten aquí.

**Rol**: orquestación asset-oriented (Dagster 1.9+) con lineage declarativo entre datasets, features y modelos.

## Estado

PARCIAL. Solo aterrizó la cadena FarSLIP + features + modelos temporales. NO existen schedules ni sensors (`schedules/__init__.py` vacío); el entrenamiento es on-demand para no gastar GPU por accidente.

Assets reales (registrados en `definitions.py` vía `load_assets_from_modules`):

- `hello_world` — smoke-test de arranque (`assets/health.py`, group `bootstrap`).
- `sentinel2_crops_256` — crops Sentinel-2 256x256 px, particionado estático por ROI italiana (`pianura_padana`, `toscana`, `puglia`). US-017.
- `farslip_embeddings_italy` — extracción bulk de embeddings 512-dim por ROI. US-017.
- `farslip_embeddings_consolidated` — consolida a `data/farslip/embeddings_pastis.parquet` (consumido por `ml/features/fusion.py`). US-022b-B.
- `parcel_features_fused` + `parcel_splits_spatial_kfold` + `parcel_features_scaler` — bundle de 3 assets de fusión multisensor parcel-level (group `feature_engineering`). US-016.
- `phenology_model_tempcnn` + `phenology_model_inceptiontime` + `temporal_models_comparison` — modelos temporales con spatial CV 5-fold (group `phenology_models`). US-022b-C.

External `AssetSpec` (solo lineage UI, no materializables): `farslip_pairs_italy`, `farslip_clip_italy_v1` (modelo en MLflow Registry, se entrena con `make train-l4`).

Job: `farslip_full_pipeline_job` (`AssetSelection.groups("farslip")`). Resource: `mlflow`.

## Comandos

```bash
make dagster-ui                      # dagster dev -m dagster_project.definitions (UI :3011)
make dagster-materialize-features    # --select parcel_features_fused+ (features -> splits -> scaler)
dagster asset list -m dagster_project.definitions
dagster job execute -m dagster_project.definitions -j farslip_full_pipeline_job
```

## Stack local

- `dagster dev` arranca desde `workspace.yaml` (`python_package: dagster_project`) o con `-m dagster_project.definitions`.
- MLflow: env `MLFLOW_TRACKING_URI`, fallback `file:./mlruns` (`resources/mlflow.py`).
- Fixtures locales: `data/test_fixtures/parcels_demo_3regions.parquet`, `feature_selection_parcels_subset.parquet` — corren los assets sin GEE/GCS.

## Convenciones (✅/❌)

- ✅ `@asset(deps=[AssetKey("...")])` con lineage explícito; nunca dependencias implícitas.
- ✅ `import polars as pl` dentro del cuerpo del asset (lazy import para no cargar torch/geopandas en `dagster definitions validate`).
- ✅ Retornar `MaterializeResult(metadata={...})` con `MetadataValue` (incluyendo `data_version` + `code_version` vía `git_sha(short=True)`).
- ✅ Graceful skip: si `is_gcs_auth_error(exc)` o no hay upstream, retornar `MaterializeResult` con `status="skipped_no_gcs"` / `"skipped_no_upstream"` y `rows=0` — NO lanzar excepción (CI sin secrets debe pasar). Errores reales (`AttributeError`, `KeyError`) sí burbujean.
- ✅ MLflow vía `context.resources.mlflow` con `required_resource_keys={"mlflow"}`; el logging falla silencioso si está offline.
- ✅ Lógica de negocio delegada a `ml/` (`ml.features.*`, `ml.train.*`, `ml.farslip.*`); el asset solo orquesta.
- ❌ Afirmar particionado temporal: solo hay `StaticPartitionsDefinition` por ROI (`ITALY_REGIONS`). No hay particiones por año/fecha.
- ❌ Reintroducir assets aspiracionales (`alphaearth_annual`, `spectral_indices`, `baseline_model`, `final_vlm`, `ensemble`, `pgstac_catalog`, drift Evidently): no existen como assets.

## No tocar

- `.dagster/` (DAGSTER_HOME, run storage) ni `mlruns/` — estado local, fuera de git.
- Parquets versionados por DVC y pesos en GCS (`gs://agrosat-models/farslip/...`) — nunca al repo.
- `FARSLIP_EXPERIMENT = "farslip-clip-italy"` y `FARSLIP_RUN_NAME` en `resources/mlflow.py` — congelados (contrato MLflow Registry B-5).

## Tests

```bash
poetry run pytest tests/dagster/ -v
```

Cubren `test_farslip_assets.py`, `test_farslip_pipeline.py`, `test_features_assets.py` (materialización con fixtures + skip sin GCS).

## Skills

- [`agrosat-dagster-mlops`](../.claude/skills/agrosat-dagster-mlops/SKILL.md) — definir assets, jobs, resources.
- [`agrosat-dvc-mlflow`](../.claude/skills/agrosat-dvc-mlflow/SKILL.md) — versionado de data y tracking de experimentos.
