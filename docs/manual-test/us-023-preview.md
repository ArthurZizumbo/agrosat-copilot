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

**Owner QA**: Arthur Zizumbo (MLOps lead).
**Estado**: pendiente ejecucion humana.
