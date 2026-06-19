# Handoff — Cablear el modelo XGBoost real + parcelas PASTIS reales en el agente

**Objetivo.** Hacer que la tool `classify_parcel` del agente clasifique con el **modelo entrenado real** sobre **parcelas PASTIS reales con geometría**, en vez del fallback `stored:crop_class`. Decisión tomada: usar el modelo existente (camino "opción 1"), **NO reentrenar** por ahora.

> Construye sobre lo ya hecho (no rehacer): la geometría ya fluye end-to-end (`Finding.geometry` / `ParcelRecord.geometry` / `ST_AsGeoJSON` / el mapa la pinta) y el `aoi_id` ya se cablea de la UI al agente. Ver [ADR-011](../decisions/ADR-011-arquitectura-sistema-conversacional-be-my-eyes.md) y `docs/orchestration/mvp-conversacional-runbook.md`.

## Dónde está cada cosa (investigado el 2026-06-15)

| Pieza | Ubicación | Detalle |
|-------|-----------|---------|
| **Modelo entrenado** | `reports/baseline/04_baseline/best_model_xgb.joblib` | Un `BaselineResult` (joblib) con `.model` (`XGBClassifier`, `num_class=18`), `.label_classes` (18), `.label_encoder`, `.feature_cols` (**185**). |
| **Matriz de features** | `data/test_fixtures/feature_selection_parcels_subset.parquet` | 85.951 parcelas × 192 cols = 185 features + `parcel_id`/`year`/`class_id`. Sin geometría. (También `data/features/features_fused_winning_pastis.parquet`.) |
| **Geometría de parcela** | `data/PASTIS-R/metadata.geojson` (patches, EPSG:2154) + rasters de parcel-id | Los **polígonos de parcela NO vienen listos**: hay que vectorizarlos de los rasters de PASTIS-R. `parcels_demo_3regions.parquet` (9 parcelas) tiene geometría pero `parcel_id` 1-9 sintéticos que **no** cruzan con la matriz de features. |
| **Notebook del modelo final** | `notebooks/final_model/06c_ensambles.ipynb` | El "modelo final" del proyecto es un **ensamble Stacking-5** (multi-modelo, demasiado pesado para inline). El agente usa el **baseline XGBoost**, por diseño v8. |

## El detalle crítico (lo que hoy hace que NO enganche)

El `best_model_xgb.joblib` fue entrenado sobre **185 features de índices espectrales Sentinel-2** (NDVI/NDWI/EVI/NDMI/NBR/MSAVI2/NDRE/MCARI/CCCI/GCVI/PSRI/NDCI/FAPAR/LAI/RENDVI — cada uno con mean/std/min/max/percentiles + armónicos FFT temporales). **NO usa AlphaEarth** (verificado: 0 columnas AlphaEarth en `feature_cols`). El nombre "XGBoost+AlphaEarth" es engañoso.

Pero hoy:
- `features_parcels` solo guarda `alphaearth_embedding VECTOR(64)` (64-dim), no las 185.
- `classify_parcel._predict_one` pasa el embedding de 64 → `64 != 185` → devuelve `None` → fallback. Por eso aunque apuntes la tool al `.joblib`, cae al `stored:crop_class`.

## Plan (5 pasos, todo con datos ya locales)

1. **Vectorizar polígonos de parcela** desde los rasters de parcel-id de `data/PASTIS-R/` (`rasterio.features.shapes` por patch), reproyectar EPSG:2154 → 4326. Para un **subset** (unos patches → cientos de parcelas) para el demo. Llave: `parcel_id`.
2. **Join** por `parcel_id` con `feature_selection_parcels_subset.parquet` (185 features + `class_id`).
3. **Migración dbmate** (`make db-new name=add_parcel_feature_vector`): añadir a `features_parcels` una columna para el vector de 185 features alineado a `feature_cols` — recomendado `model_features JSONB` (`{nombre: valor}`, robusto a orden) o `float8[]`. NO editar migraciones aplicadas.
4. **Script de carga** (`scripts/load_pastis_parcels.py`, idempotente, estilo `scripts/seed_demo_parcels.py`): insertar el subset en una sesión: `parcels` (geom 4326 + `crop_class` vía `PASTIS_R_CLASSES[class_id]` + `year`) + `features_parcels` (las 185 features).
5. **Cablear la tool** (`ml/agent/tools/classify_parcel.py`):
   - `_load_classifier`: si la URI es una ruta local `.joblib`, cargar con `joblib.load` (hoy solo intenta `mlflow.sklearn.load_model`). Default URI configurable → `reports/baseline/04_baseline/best_model_xgb.joblib` (env `CROP_CLASSIFIER_MODEL_URI`, ver `ml/agent/settings.py`).
   - El predict debe construir el vector de **185** desde `FeatureRecord` (extender `ParcelReader.get_features` / `FeatureRecord` para exponer `model_features` en orden `feature_cols`), no el embedding de 64.
   - `feature_cols` order: `list(bundle.feature_cols)`; armar `[features[c] for c in feature_cols]`.

## Verificación

- `poetry run pytest tests/ml/agent backend/tests/unit -q` verde (mockear el modelo y el reader).
- End-to-end: `make db-seed` + cargar el subset + preguntar en la UI → `tool_result` con `used_model=true`, `source="XGBoost-indices"`, y parcelas reales pintadas en el mapa.
- Runbook de arranque del stack: `docs/orchestration/mvp-conversacional-runbook.md`. DB local en puerto **55432** (`.env.local`).

## Implicación para #3 (ingesta GEE on-demand)

Como el modelo es de **índices Sentinel-2** (no AlphaEarth), clasificar una zona nueva por GEE exige calcular las 185 features = series temporales Sentinel-2 → 15 índices → stats + FFT (pipeline pesado), **no** un simple muestreo de AlphaEarth. GEE ya está desbloqueado (ADC de la cuenta del owner sobre proyecto `agrosat-copilot`; `GEE_PROJECT_ID` en `.env.local`).

## Skills relevantes
`agrosat-ml-baseline`, `agrosat-ml-features` (índices/temporal Polars), `agrosat-db-migrations`, `agrosat-db-models`, `agrosat-google-adk-agent`, `agrosat-gee-alphaearth`.
