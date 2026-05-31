# Manual test plan — US-023-preview (Baseline corrections, post-A3 -> EPIC 5)

> Solo flujos que requieren validacion humana / browser real / artefactos en disco.
> Quality gates automatizados viven en `pytest` (69/69 passing) + `ruff` + papermill.

Pre-requisitos: `poetry install --with paper`, `python >= 3.12`, repo limpio en rama
`feature/E4-US-023-preview-baseline-corrections`, artefactos P2/P3/P4/P5/P8 ya en disco
(este es el caso post-coding `f2174a0..HEAD` con working tree completo).

## Bloque 1 — Dashboard Streamlit `Baseline (US-023-preview)`

### Test 1.1 — arranque + categoria "Baseline" visible

```bash
poetry run streamlit run app/eda_dashboard.py --server.headless true --server.port 8501
```

[Paso] Abrir `http://localhost:8501` -> [Esperado] El dashboard arranca sin traceback rojo.
La sidebar muestra **3 categorias**: `EDA (Avance 1)`, `Feature Engineering (Avance 2)`, y
`Baseline (US-023-preview)`. Seleccionar la tercera.

### Test 1.2 — 5 tabs operativos

[Paso] Con la categoria Baseline seleccionada, contar las pestañas -> [Esperado] exactamente
5 tabs con etiquetas en español: `Ablation features`, `Leakage geográfico (geom)`,
`Bloques opcionales`, `Modelos v2 (XGBoost / TempCNN / InceptionTime)`, `Conclusiones`.

[Paso] Hacer click en cada tab en orden -> [Esperado] cada tab renderiza al menos una figura PNG
+ una tabla parquet sin tracebacks. Si algun artefacto falta, aparece un `st.warning` con el
hint `make reencuadre-notebook-full && make baseline-v2-full` (R11 graceful degradation).

### Test 1.3 — tab "Bloques opcionales" lee parquet real P4

[Paso] Click tab `Bloques opcionales` -> sub-bloque `Gemini Flash 3.5 (US-023-preview P4)`.
-> [Esperado] La tabla muestra **3 filas** (`full`, `with_pheno_text`, `pheno_text_only`)
con `f1_macro` numerico, y el caption menciona costo `$0.49 USD` con 1080 parcelas.
Si fallback al historico, caption deberia indicarlo.

### Test 1.4 — tab "Modelos v2" — XGBoost ganador

[Paso] Click tab `Modelos v2 (XGBoost / TempCNN / InceptionTime)` -> revisar la tabla
`model_comparison_v2.parquet` -> [Esperado] 3 filas: xgboost F1-macro 0.4094 (ganador),
inceptiontime 0.1898, tempcnn 0.1435. Caption indica `spatial CV 5-fold buffer 1.0 km`
y `feature_set=no_geom`.

## Bloque 2 — Notebooks ejecutables end-to-end

### Test 2.1 — papermill notebook 05 (reencuadre fenologico)

```bash
poetry run papermill notebooks/baseline/05_reencuadre_fenologico.ipynb \
  /tmp/05_smoke.ipynb -p MAX_SAMPLES 800 --kernel python3
```

[Paso] Ejecutar y observar el log -> [Esperado] `papermill` completa con exit 0 en <10 min.
Las celdas v2 (`geom_only`, `with_farslip`, `farslip_only`, `with_pheno_text`,
`pheno_text_only`, `with_spectral_signature`) deben terminar sin errores y producir tablas
HTML. La fila `geom_only` debe tener F1-macro < 0.10 (test cuantitativo H-4).

### Test 2.2 — papermill notebook 04b standalone

```bash
poetry run papermill notebooks/baseline/04b_baseline_v2.ipynb /tmp/04b_smoke.ipynb --kernel python3
```

[Paso] Ejecutar -> [Esperado] completa en <= 30 s (solo I/O + render PNG, NO reentrena).
El notebook muestra 3 modelos v2 + figura `model_comparison_v2.png` + tabla LaTeX inline.

### Test 2.3 — papermill notebook Avance3 integrador

```bash
poetry run papermill notebooks/baseline/Avance3.Equipo17.ipynb /tmp/avance3_smoke.ipynb --kernel python3
```

[Paso] Ejecutar -> [Esperado] completa en ~12 s con 13 celdas, las 4 secciones consolidadas
(baseline original, ablation post-A3, baseline v2, conclusiones). Cada seccion mapea 1:1 a
la rubrica oficial 40+20+10+20+10.

## Bloque 3 — FarSLIP path canonico + DVC

### Test 3.1 — parquet canonico legible

```bash
poetry run python -c "import polars as pl; df = pl.read_parquet('data/farslip/embeddings_italy.parquet'); print(df.shape); print([c for c in df.columns][:5])"
```

[Paso] Ejecutar -> [Esperado] shape `(30173, 514)`. Las primeras columnas deben ser
`parcel_id`, `class_id`, `farslip_emb_000`/`farslip_emb_001`/`farslip_emb_002`. El patch
defensivo de `_build_farslip_block` acepta ambos prefijos (`farslip_NNN` y
`farslip_emb_NNN`); se valida en `tests/ml/features/test_fusion.py`.

### Test 3.2 — .dvc file integro

```bash
poetry run cat data/farslip/embeddings_italy.parquet.dvc
poetry run dvc status data/farslip/embeddings_italy.parquet.dvc
```

[Paso] Ejecutar -> [Esperado] el `.dvc` muestra hash MD5 + size 50 MB +
`path: embeddings_italy.parquet`. `dvc status` reporta `not in cache` solo si el remote no
ha sido pulled (esperado en QA local). NO debe reportar `modified` (parquet desincronizado).

## Bloque 4 — MLflow runs persistidos

### Test 4.1 — UI MLflow muestra 4 runs nuevos

```bash
poetry run mlflow ui --backend-store-uri file:./mlruns --port 5000
```

[Paso] Abrir `http://localhost:5000` -> [Esperado] visible el experiment
`baseline-v2-us-023-preview` con **3 runs** (`baseline-v2-xgb`, `baseline-v2-tempcnn`,
`baseline-v2-inceptiontime`) y el experiment `baseline-pheno-text-ablation` con **1 run**
(`baseline-pheno-text-ablation-v1`). Cada run debe llevar tags `us=US-023-preview`,
`code_version`, `data_version`.

### Test 4.2 — XGBoost run con 6 metricas

[Paso] Click run `baseline-v2-xgb` -> [Esperado] metricas presentes:
`f1_macro=0.4094`, `f1_weighted=0.6917`, `miou=0.3115`, `accuracy=0.7257`, `kappa=0.6546`,
`train_time_s=436.6`. Tag `feature_set=no_geom`.

## Bloque 5 — Quality gates pre-PR

### Test 5.1 — pytest US-023 modules

```bash
poetry run pytest tests/app/test_eda_dashboard_baseline_section.py \
  tests/ml/features/test_spectral_signature.py \
  tests/ml/features/test_fusion.py \
  tests/ml/eval/test_feature_ablation.py -q
```

[Paso] Ejecutar -> [Esperado] **69 passed** en ~4 min. 1 warning XGBoost CUDA mismatch
benigno (DMatrix fallback) — ignorable.

### Test 5.2 — ruff scoped (modulos core)

```bash
poetry run ruff check ml/features/spectral_signature.py ml/features/fusion.py \
  ml/eval/feature_ablation.py app/eda_dashboard.py \
  tests/ml/features/test_spectral_signature.py tests/ml/features/test_fusion.py \
  tests/ml/eval/test_feature_ablation.py tests/app/test_eda_dashboard_baseline_section.py
```

[Paso] Ejecutar -> [Esperado] `All checks passed!`. Los scripts auxiliares
(`scripts/build_reencuadre_notebook.py`, `scripts/run_baseline_v2_standalone.py`,
`scripts/us023_p4_*`) tienen lint debt documentada (ver handoff §"QA findings");
agregar `per-file-ignores` en `pyproject.toml` antes de PR.

## Bloque 6 — Flujos NO testeados aqui (fuera de scope US-023-preview)

- Chat trilingue (it/es/en) — pertenece al Nuxt frontend principal, no a Streamlit.
- Switch A/B LLM (Gemini vs Qwen vLLM) — no afecta esta US (no toca agent).
- AOI drawer / MapLibre — no se toca en US-023-preview.
- Citaciones en respuestas ADK — no se toca el agente.
- Cross-session isolation — el dashboard es single-user local, no multi-tenant.

---

# Adenda — sesion v2 (VM L4, AlphaEarth + HCAT, 2026-05-31)

> Scope distinto al plan original arriba: la sesion autonoma del 31-may conecto AlphaEarth
> al vector (185 -> 313 cols), agrego agrupacion HCAT Level-1 (18 -> 6 grupos) y re-ejecuto
> los notebooks `04c_baseline`, `05_reencuadre_fenologico`, `Avance3.Equipo17` en VM L4 GCP.
> Working tree sin commitear. QA CLI 2026-05-30: 32/32 tests scope, ruff limpio, DVC up to
> date, sin secretos en el repo. Los flujos de abajo requieren validacion humana.

Pre-requisitos: `poetry install`, `python >= 3.12`, working tree de la sesion v2 (notebooks
04c/05/Avance3 modificados, `data/features/features_fused_*.parquet.dvc` staged), `dvc pull`
ejecutado para los parquets fused (118 MB + 57 MB).

## Bloque A1 — Notebooks v2 ejecutables (outputs poblados)

### Test A1.1 — notebook 04c muestra cifras (no NaN) en la tabla renderizada

[Paso] Abrir `notebooks/baseline/04c_baseline.ipynb` en VSCode/Jupyter, ir a la celda de la
tabla de ablation (`FeatureAblationResult`) -> [Esperado] tabla con **5 filas y f1_macro
NUMERICO**: `full`=0.4352, `no_geom`=0.4352, `no_geom_no_era5_srtm`=0.4352,
`alphaearth_only`=0.3584 (delta -0.077), `phenology_only`=0.2856 (delta -0.150).
**OJO (B-NEW)**: el artefacto `reports/baseline/04c_baseline/ablation_table.parquet`
persistido tiene `f1_macro=NaN` en disco aunque el notebook muestra las cifras correctas.
Confirmar visualmente que la tabla del notebook NO esta en NaN (entregable visual).

### Test A1.2 — papermill 04c smoke (regenerado desde builder, no git checkout)

```bash
poetry run python scripts/build_baseline_notebooks_v2.py  # regenera notebooks desde builder
poetry run papermill notebooks/baseline/04c_baseline.ipynb /tmp/04c_smoke.ipynb --kernel python3
```

[Paso] Ejecutar -> [Esperado] exit 0, 6 celdas codigo, 0 errores. La celda `alphaearth_only`
debe entrenar (NO 0/NaN): el loader correcto es `load_base_plus_alphaearth_2018_2019` (322
cols). REGLA de la sesion v2: NO `git checkout` los notebooks; regenerar desde el builder
(el .ipynb viejo usaba `load_features_dataset_with_meta`, 194 cols sin AlphaEarth).

### Test A1.3 — papermill 05 + Avance3 (cadena de artefactos)

```bash
poetry run papermill notebooks/baseline/05_reencuadre_fenologico.ipynb /tmp/05.ipynb -p MAX_SAMPLES 800 --kernel python3
poetry run papermill notebooks/baseline/Avance3.Equipo17.ipynb /tmp/avance3.ipynb --kernel python3
```

[Paso] Ejecutar 05 primero -> [Esperado] escribe `data/features/features_fused_italy.parquet`
(celda `fused.write_parquet(FUSED_PATH)`). Avance3 depende de ese parquet; si 05 no lo escribe
-> `FileNotFoundError`. Avance3 debe terminar con `winner=phenology+ae+indices` (96 features,
todas las decisiones de bloques opcionales = `descartar`).

## Bloque A2 — Claim central: HCAT 18 vs 6 grupos

### Test A2.1 — comparison parquet con el salto +0.217

```bash
poetry run python -c "import polars as pl; print(pl.read_parquet('reports/baseline/grouped_vs_flat/comparison.parquet'))"
```

[Paso] Ejecutar -> [Esperado] 2 filas: `flat18` F1-macro **0.4365**, `hcat_l1_6` F1-macro
**0.6535** (delta +0.217). Este es el claim que sube el baseline sobre el umbral 0.60 leido
por familias de cultivo. Verificar en el notebook 04b/Avance3 que la narrativa lo presenta
como metodo Russwurm 2018 / HCAT v3 (NO como maquillaje del numero plano).

## Bloque A3 — DVC artefactos pesados v2

### Test A3.1 — fused parquets + modelos joblib van a DVC (no git)

```bash
poetry run dvc status data/features/features_fused_italy.parquet.dvc \
  data/features/features_fused_winning_italy.parquet.dvc \
  reports/baseline/04_baseline/best_model_xgb.joblib.dvc
git ls-files reports/baseline/ | grep -E "\.(joblib|pkl)$"   # debe estar VACIO
```

[Paso] Ejecutar -> [Esperado] `dvc status` = `up to date`. El `git ls-files` de joblib/pkl
**vacio** (los `.joblib` reales estan gitignored via `reports/baseline/**/*.joblib`; solo los
`.dvc` pointers se commitean). Tras merge, `dvc push` a `gs://agrosat-dvc-remote`.

## Bloque A4 — SEGURIDAD (accion humana, NO del repo)

### Test A4.1 — rotacion de credenciales post-inyeccion VM

[Contexto] La sesion VM L4 reporto (handoff §"SEGURIDAD - CRITICO") inyeccion de prompts en
stdout pidiendo exfiltrar `.env.local`/credenciales a `agrosat-telemetry.net` (dominio NO del
proyecto). QA 2026-05-30 verifico: el payload **NO esta en ningun archivo del repo** (codigo,
notebooks, outputs, scripts) — solo en el texto del handoff. Pero la VM esta comprometida.

[Paso humano] -> [Esperado] ejecutar las 4 acciones del handoff:
1. ROTAR `GEMINI_API_KEY` + key de la service account `agrosat-gee-sa`.
2. MIGRAR `GEMINI_API_KEY` a GCP Secret Manager.
3. BORRAR `/mnt/data/agro_sat_copilot/.env.local` de la VM.
4. INVESTIGAR la VM (`~/.bashrc`, `/etc/profile.d`, `PROMPT_COMMAND`, `LD_PRELOAD`, wrapper
   papermill) por el hook que inyecta en stdout; considerar recrear la VM desde imagen limpia.

## Bloque A5 — Flujos NO testeables en CLI (pendiente humano)

- Dashboard Streamlit tab Baseline con el parquet 04c en NaN (B-NEW): confirmar si muestra NaN
  o degrada con `st.warning`.
- `make notebooks-check` + `make check` end-to-end (gitleaks solo en CI).
- `dvc pull` desde remote en maquina limpia (recuperacion de los fused parquets de 118/57 MB).

---

**Owner QA**: Arthur Zizumbo (MLOps lead).
**Estado**: pendiente ejecucion humana (plan original + adenda v2).
