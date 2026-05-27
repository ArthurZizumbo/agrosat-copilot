# US-023-preview — Plan canonico — Correcciones al baseline previo a EPIC 5

**Status**: Planning (canonico) · creado 2026-05-25 · listo para Fase 3 (coding)
**Epic**: E4 (Baseline) · transversal con E5 (preview de la transicion al modelado denso)
**Avance**: post-A3 (24-may, ya calificado 100/100) — alimenta A4 (24-may) y A5 (31-may)
**Sprint**: S6 (25-may → 31-may)
**Rama prevista**: `feature/E4-US-023-preview-baseline-corrections` (alias corto `us-023-preview`)
**Owner**: solo-dev (todos los sub-bloques P1..P9)
**SP estimados**: **14 SP** (P1=1, P2=2, P3=1, P4=2, P5=1, P6=1, P7=1, P8=3, P9=2)
**ADR de referencia**: [`ADR-006-reencuadre-baseline-fenologico.md`](../decisions/ADR-006-reencuadre-baseline-fenologico.md)
**US predecesoras**: [US-022](../us-resolved/us-022.md), [US-022-b](../us-resolved/us-022b.md), [US-022-c](../us-resolved/us-022-c.md)
**Paper-faro**: Wen et al. (2025), "Phenology description is all you need!", DOI 10.1016/j.isprsjprs.2025.07.002.

> **Alcance**: cerrar las 6 observaciones detectadas el 25-may sobre la libreta entregable
> `notebooks/baseline/05_reencuadre_fenologico.ipynb` (movida desde `notebooks/feature_engineering/`)
> y `notebooks/baseline/04_baseline.ipynb`. La US **no entrega Avance nuevo**: refina el baseline
> post-A3 para que el bloque de modelos densos (EPIC 5 US-023 U-Net en adelante) arranque sobre
> conjuntos de features y resultados ya validados con FarSLIP materializado, sin ruido geografico y
> con la ablation `with_farslip`/`farslip_only` cuantificada.

---

## 1. Observaciones origen (25-may-2026)

| # | Observacion del usuario | Sub-bloque |
|---|--------------------------|------------|
| O-1 | Las 2 libretas `notebooks/baseline/04_baseline.ipynb` y `05_reencuadre_fenologico.ipynb` se movieron desde `notebooks/` y `notebooks/feature_engineering/` — los builders, Makefile y los paths internos deben recalcularse y respetar [`notebooks/CLAUDE.md`](../../notebooks/CLAUDE.md). | P1 |
| O-2 | FarSLIP no materializado en el path esperado por `fusion.py`: `data/farslip/embeddings_italy.parquet` no existe, solo `embeddings_italy_v1.parquet` (30173 filas) y `embeddings_italy_v2.parquet` (extraccion real epoch_2, deuda de US-022-c P1). La ablation omite `with_farslip` y `farslip_only`. | P2 |
| O-3 | Falta comparativa explicita del Feature Engineering **con y sin** las 3 columnas `geom_*` (area, perimetro, elongacion). En la corrida 22-may delta = 0.0, pero el plot esta agrupado: el lector no ve la barra `no_geom` aislada y los riesgos de leakage no quedan visibles. | P3 |
| O-4 | Falta ablation cuantitativa del bloque `pheno_text_*` generado con **Gemini 3.5 Flash** (US-022-c P5 corrio sobre 216 parcelas en subset acotado con delta = -0.12). Se necesita validar/refutar sobre el dataset full o sobre un subset suficiente para descartarlo definitivamente del baseline. | P4 |
| O-5 | Investigar si una variable derivada de **firma espectral** (Sentinel-2: indice de reflectancia tipo `Red-edge / NIR` o angulo espectral) puede agregar señal complementaria al embedding AlphaEarth, dado que las features actuales ya tienen FFT pero no un descriptor compacto de la forma espectral por epoca. | P5 |
| O-6 | Confirmar que las dos libretas cumplen el estandar `notebooks/CLAUDE.md` (idioma de strings vs identificadores, `display()` sobre `print()`, sin emojis decorativos, conclusiones sin "US-XXX/EPIC/AC-X", paths via `pathlib`, etc.). | P6 |
| O-8 | Una vez aplicadas P2/P3/P4/P5, el notebook `notebooks/baseline/04_baseline.ipynb` queda desactualizado: la comparativa entrenada en US-022 corrio con FarSLIP omitido, `geom_*` incluidas, sin `pheno_text` real y sin descriptor espectral. Se necesita una **v2 del baseline** que entrene los **3 modelos canonicos del A3 (XGBoost + TempCNN + InceptionTime)** sobre el conjunto de features ganador post-ablation. | P8 |
| O-9 | El dashboard Streamlit (`app/eda_dashboard.py`) solo tiene secciones EDA y FE — no expone los resultados del baseline. Una vez generados los artefactos de P2/P3/P4/P5/P8 (tablas ablation, plots, MLflow runs, decisiones), se debe agregar la categoria **"Baseline"** al selector de secciones, replicando el patron de cards/figuras/tablas ya usado en EDA/FE. | P9 |

Adicional cosmetico:
| O-7 | Plan v6 (`context/RefinamientoPlaneacionAgroSatCopilot_v6.md`) no referencia US-023-preview. | P7 |

---

## 2. Criterios de aceptacion (metricas verificables)

### 2.1 P1 — Rutas y builders post-movimiento (1 SP)

| AC | Criterio | Como verificar |
|----|----------|----------------|
| AC-P1-1 | `scripts/build_baseline_notebook.py` y `scripts/build_reencuadre_notebook.py` emiten los notebooks en `notebooks/baseline/` por defecto | `grep -n "notebooks/baseline" scripts/build_baseline_notebook.py scripts/build_reencuadre_notebook.py` muestra los nuevos paths; no quedan referencias a `notebooks/feature_engineering/05_reencuadre*` ni a `notebooks/04_baseline.ipynb` raiz |
| AC-P1-2 | `Makefile` targets `baseline-notebook`, `reencuadre-notebook`, `reencuadre-notebook-check`, `reencuadre-notebook-full` apuntan a `notebooks/baseline/*.ipynb` | `make -n baseline-notebook` y `make -n reencuadre-notebook` muestran los paths nuevos |
| AC-P1-3 | `make notebooks-check` (papermill end-to-end) pasa para `notebooks/baseline/04_baseline.ipynb` y `notebooks/baseline/05_reencuadre_fenologico.ipynb` | exit 0; outputs poblados; HTML en `notebooks/baseline/html/` regenerado |
| AC-P1-4 | Documentacion de la tabla canonica en [`notebooks/CLAUDE.md`](../../notebooks/CLAUDE.md) §"Estructura Canonica" actualizada: `04_baseline.ipynb` y `05_reencuadre_fenologico.ipynb` viven en `notebooks/baseline/` | grep en `notebooks/CLAUDE.md` muestra la nueva ruta |

### 2.2 P2 — FarSLIP materializado en el path canonico + fix discrepancia naming (2 SP)

**Estado verificado en disco 2026-05-25**:

- `data/farslip/embeddings_italy_v1.parquet` existe — shape `(30173, 514)`, md5 `e3b74eec...`, **placeholder determinista seeded** (`_project_parcels_to_embeddings`, implementacion real crops live diferida a US-025 segun [`docs/us-resolved/us-022-c.md:35`](../us-resolved/us-022-c.md#L35)).
- `data/farslip/embeddings_italy_v2.parquet` existe — shape `(30173, 514)`, md5 `dc5c3805...`, **extraccion real** epoch_2 (commit `0f01255`).
- `data/farslip/embeddings_italy.parquet` (path canonico) **NO existe**.
- Tag git `farslip-embeddings-italy-v1` **NO existe** (gate B-4 de US-022-c quedo pendiente, ver [`docs/us-resolved/us-022-c.md:104`](../us-resolved/us-022-c.md#L104) marcado PENDIENTE).
- **Bug de naming detectado**: [`ml/farslip/extract_embeddings.py:52`](../../ml/farslip/extract_embeddings.py#L52) escribe columnas `farslip_emb_000`...`farslip_emb_511` (prefix `farslip_emb_`), pero [`ml/features/fusion.py:582`](../../ml/features/fusion.py#L582) en `_build_farslip_block` espera `farslip_000`...`farslip_511` (prefix `farslip_`). Aun con el path canonico resuelto, `_build_farslip_block` levantaria `ValueError("FarSLIP parquet no trae las 512 columnas esperadas")`.
- Efecto colateral: en [`ml/eval/feature_ablation.py:154`](../../ml/eval/feature_ablation.py#L154) el filtro `c.startswith("farslip_")` SI matchea `farslip_emb_*` por coincidencia lexica (todo `farslip_emb_XXX` empieza por `farslip_`), por lo que el conjunto `with_farslip`/`farslip_only` se construiria con 512 cols correctas si el bloque llegara al DataFrame fusionado — pero `_build_farslip_block` falla antes y nunca llegamos ahi.

| AC | Criterio | Como verificar |
|----|----------|----------------|
| AC-P2-1 | Existe `data/farslip/embeddings_italy.parquet` con shape `(30173, 514)`, columnas `parcel_id`, `year`, `farslip_000`...`farslip_511` (alineado con el contrato `fusion.py`). Decision D-1: promover **v2** (extraccion real epoch_2). | `python -c "import polars as pl; df=pl.read_parquet('data/farslip/embeddings_italy.parquet'); print(df.shape, df.columns[:6], df.columns[-3:])"` muestra `(30173, 514)` + `parcel_id, year, farslip_000, farslip_001, farslip_002, farslip_003` + `farslip_509, farslip_510, farslip_511` |
| AC-P2-2 | El parquet canonico se genera renombrando columnas de v2 (`farslip_emb_XXX -> farslip_XXX`) y se versiona con DVC sin duplicar las versiones historicas (v1, v2 quedan tracked) | `data/farslip/embeddings_italy.parquet.dvc` existe; `git status` no muestra `embeddings_italy_v{1,2}.parquet` modificados |
| AC-P2-3 | Patch defensivo en [`ml/features/fusion.py:582-588`](../../ml/features/fusion.py#L582) acepta AMBOS prefijos (`farslip_` y `farslip_emb_`) para evitar romper si alguien usa el v1/v2 directo via `farslip_path=...` explicito | spotcheck del diff + 1 test nuevo en `tests/ml/features/test_fusion.py` que carga un parquet con prefix `farslip_emb_` |
| AC-P2-4 | Tag git `farslip-embeddings-italy-v1` creado apuntando al commit que materializa el path canonico + `dvc push` ejecutado | `git tag -l 'farslip-embeddings-italy-v1'` no vacio; `dvc list . data/farslip/` muestra blob remoto |
| AC-P2-5 | La ablation del notebook 05 ejecuta los 5+2 conjuntos: `full`, `no_geom`, `no_geom_no_era5_srtm`, `alphaearth_only`, `phenology_only`, **`with_farslip`**, **`farslip_only`** | `ablation_table.parquet` en `reports/baseline/feature_ablation/` con 7 filas (`n_features > 0` en `with_farslip` y `farslip_only`) |
| AC-P2-6 | Delta `f1_macro(with_farslip) - f1_macro(full)` reportado en tabla + interpretacion + nota explicita "FarSLIP v2 = embeddings reales epoch_2, 30173 parcelas; el target 85951 queda en US-025" | celda Markdown del notebook 05 |
| AC-P2-7 | MLflow run `baseline-farslip-ablation-v1` registrado con tags `data_version` (DVC short hash del parquet canonico) + `code_version` (git sha) + metricas de los 7 conjuntos | UI MLflow muestra el run con 7 metricas + tags |
| AC-P2-8 | Decision documentada: si `with_farslip - full >= +0.02` se promueve FarSLIP al baseline; si esta entre [-0.02, +0.02] queda como base learner del stacking EPIC 6; si `< -0.02` se descarta del baseline con justificacion. Nota R1: el delta puede estar penalizado por la cardinalidad 30173 vs 85951 del dataset full — gate honesto, no bloqueante. | seccion §"Decision FarSLIP" del notebook 05 |
| AC-P2-9 | Gate B-4 de US-022-c marcado VERDE retroactivamente en [`docs/us-resolved/us-022-c.md`](../us-resolved/us-022-c.md) con anotacion "cerrado por US-023-preview P2 — shape resultante (30173, 514) en lugar de (85951, 514) por R1" | grep `B-4.*VERDE` en `us-022-c.md` |

### 2.3 P3 — Comparativa explicita FE con/sin `geom_*` (1 SP)

| AC | Criterio | Como verificar |
|----|----------|----------------|
| AC-P3-1 | Plot `ablation_geom_comparison.png` en `paper/figures/us-023-preview/` con 2 barras: `full` vs `no_geom`, mismo eje Y + anotacion del delta + nota interpretativa | archivo PNG existente + embebido en celda del notebook 05 |
| AC-P3-2 | Test de leakage cuantitativo: F1-macro de un modelo que **solo** usa `geom_*` se reporta y queda < 0.10 (no aporta señal de clase) | celda nueva en notebook 05; resultado en `ablation_table.parquet` fila `geom_only` |
| AC-P3-3 | Conclusion explicita en el notebook + en [`docs/us-resolved/us-022b.md`](../us-resolved/us-022b.md) §"Hipotesis C-2" enlazando aqui | grep `US-023-preview` en `us-022b.md` |
| AC-P3-4 | Por que las 3 columnas `geom_*` meten "ruido importante": narrativa en lenguaje accesible que explica que se interpretan como un proxy de region (leakage espacial) y como rompen la hipotesis C-2 del Dr. Camacho | seccion §"Por que descartar `geom_*`" en el notebook 05 |

### 2.4 P4 — Ablation cuantitativa del bloque `pheno_text_*` Gemini Flash 3.5 (2 SP)

| AC | Criterio | Como verificar |
|----|----------|----------------|
| AC-P4-1 | El parquet `data/features/phenology_text_italy.parquet` existe con shape `(N, 1 + 384)` (parcel_id + 384 dim sentence-transformers) | `python -c "import polars as pl; print(pl.read_parquet('data/features/phenology_text_italy.parquet').shape)"` |
| AC-P4-2 | Subset estratificado de >=1000 parcelas balanceadas por clase (vs 216 en US-022-c P5) y/o sobre el dataset full si Gemini API quota lo permite | celda en notebook 05 documenta `n_parcels_per_class` |
| AC-P4-3 | Ablation reporta filas `full`, `with_pheno_text`, `pheno_text_only` en `ablation_table.parquet` con n_features correcto (185 / 569 / 384) y delta vs `full` | tabla en notebook + parquet persistido |
| AC-P4-4 | Costo Gemini Flash 3.5 documentado en `docs/l4_log.md` (entrada US-023-preview-P4) con `n_requests`, `tokens_in/out`, `cost_usd <= $5 USD` | entrada nueva en `docs/l4_log.md` |
| AC-P4-5 | Decision documentada: promover `pheno_text` al baseline si delta >= +0.01, o mantenerlo como base learner del stacking EPIC 6 si no aporta señal en este volumen | seccion §"Decision pheno_text" en notebook 05 |
| AC-P4-6 | MLflow run `baseline-pheno-text-ablation-v1` con tags `data_version` + `code_version` + costos Gemini en `params` | UI MLflow |

### 2.5 P5 — Descriptor de firma espectral como feature adicional (1 SP)

| AC | Criterio | Como verificar |
|----|----------|----------------|
| AC-P5-1 | Modulo `ml/features/spectral_signature.py` con clase `SpectralSignatureFeatures(BaseEstimator, TransformerMixin)` que produce K features compactas por parcela. Candidatas seleccionadas (elegir 1-2): (a) Red-edge Position estacional (`REP`), (b) Spectral Angle Mapper vs centroide de la clase mayoritaria, (c) momento estadistico de la reflectancia red-edge (mean/var/skew) en los 3 picos fenologicos (SOG/peak/senescence). Documentar justificacion agronomica + paper de referencia. | archivo existe + clase sklearn-compatible + docstring Google en espanol + paper citado |
| AC-P5-2 | Tests pytest >=6: `tests/ml/features/test_spectral_signature.py` con fixtures sinteticas (cobertura >= 80%) | `poetry run pytest tests/ml/features/test_spectral_signature.py -q` exit 0 |
| AC-P5-3 | Bloque `spectral_signature_*` integrado en `ml/features/fusion.py` como bloque OPCIONAL (`include_spectral_signature: bool = False`) con LEFT JOIN sobre parcel_id (mismo patron que FarSLIP / pheno_text); constante `EXPECTED_COL_COUNT_WITH_SPECTRAL_SIGNATURE` declarada | grep `spectral_signature` en `ml/features/fusion.py` |
| AC-P5-4 | Ablation reporta fila `with_spectral_signature` y delta vs `full` en `ablation_table.parquet` | tabla en notebook 05 |
| AC-P5-5 | Decision documentada (mismo patron P2/P4): promover al baseline si delta >= +0.01; quedarse como deuda de investigacion si no aporta | seccion §"Decision firma espectral" del notebook |
| AC-P5-6 | Implementacion no rompe la cuota Sentinel-2 / GEE: las features se derivan de las columnas espectrales ya muestreadas y persistidas en `data/features/*` (no nueva ingesta) | grep `gee_sampler` o `init_ee` en `spectral_signature.py` debe estar vacio (consumimos parquet ya construido) |

### 2.6 P6 — Cumplimiento del estandar `notebooks/CLAUDE.md` (1 SP)

| AC | Criterio | Como verificar |
|----|----------|----------------|
| AC-P6-1 | Imports + configs al inicio (`%load_ext autoreload`, `%autoreload 2`, `pl.Config.set_tbl_*`, `plt.rcParams`) en celda 3 estandar | celda 3 del .ipynb |
| AC-P6-2 | Polars en todas las celdas (sin `pandas` salvo conversion final a `.to_pandas()` para SHAP/matplotlib) | grep `pandas` en notebook acotado a `to_pandas()` |
| AC-P6-3 | Strings al lector (markdown / display / print / titulos) con acentos UTF-8; identificadores / cache keys / comentarios tecnicos en ASCII puro | spotcheck manual: 4-5 celdas elegidas al azar |
| AC-P6-4 | `display()` para DataFrames + `plt.close(fig)` despues de `display(fig)` para evitar doble render | grep `display(` y `plt.close(fig)` |
| AC-P6-5 | Seccion "Conclusiones" en lenguaje accesible sin US-XXX / EPIC / AC-X / "rubrica" / "papermill" / "CI", con numeros reales + "Lo que sigue" | inspeccion ultima celda Markdown |
| AC-P6-6 | Sin emojis decorativos, sin separadores ASCII `===`/`---` decorativos, sin `print()` con "Step 1" / "Step 2" | grep negativo: `===` ` `, `Step 1`, etc. |
| AC-P6-7 | Paths con `pathlib` y `find_repo_root()` — sin paths absolutos hardcodeados | grep `Path(\"C:` o `Path('C:` debe ser 0 |
| AC-P6-8 | Las dos notebooks pasan el QA Checklist completo de `notebooks/CLAUDE.md` §"QA Checklist Notebooks" (16 items) | checklist marcado en `docs/manual-test/us-023-preview.md` |

### 2.7 P7 — Plan v6 actualizado (1 SP)

| AC | Criterio | Como verificar |
|----|----------|----------------|
| AC-P7-1 | `context/RefinamientoPlaneacionAgroSatCopilot_v6.md` contiene la entrada de US-023-preview en EPIC 4 (post US-022-b) con el mismo formato (Como/quiero/para que + Criterios + Tareas + Estimacion) | grep `US-023-preview` muestra la entrada |
| AC-P7-2 | La entrada referencia [ADR-006](../decisions/ADR-006-reencuadre-baseline-fenologico.md), este plan y `docs/us-handoff/us-023-preview.md` | grep en la entrada nueva |
| AC-P7-3 | Calendario `§Secuenciacion semanal` actualizado: US-023-preview encaja en S6 (25-31 may) sin desplazar A4/A5 | seccion §"Secuenciacion semanal" del v6 |

### 2.8 P8 — Baseline v2 con los 3 modelos canonicos del A3 (3 SP)

Una vez P2/P3/P4/P5 cierran (path FarSLIP canonico + ablation `geom_only` + `pheno_text` real + firma espectral), el conjunto de features ganador queda decidido. P8 reentrena los **3 modelos del A3** sobre ese conjunto y produce la **v2 del notebook `04_baseline.ipynb`** que reemplaza la corrida de US-022.

| AC | Criterio | Como verificar |
|----|----------|----------------|
| AC-P8-1 | `notebooks/baseline/04_baseline.ipynb` v2 ejecutado end-to-end con papermill: entrena los 3 modelos (**XGBoost** tabular + **TempCNN** temporal + **InceptionTime** temporal) sobre el conjunto de features ganador post-P2/P3/P4/P5 (decidido en la celda §"Decisiones de los bloques opcionales"). | papermill exit 0; outputs poblados; tabla `model_comparison_v2.parquet` con 3 filas (XGBoost, TempCNN, InceptionTime) + F1-macro / F1-weighted / mIoU / accuracy / kappa / train_time_s |
| AC-P8-2 | Plot `model_comparison_v2.png` con 3 barras + comparacion contra v1 (US-022 baseline original, F1-macro 0.4094 XGB, 0.1430-0.1456 TempCNN, 0.1865 InceptionTime) — delta documentado por modelo. | PNG embebido en notebook + tabla de deltas en celda Markdown |
| AC-P8-3 | Hiperparametros baseline v2 documentados en `notebooks/baseline/04_baseline.ipynb` v2 — XGB: `n_estimators=500, max_depth=8, learning_rate=0.05` (GridSearchCV ligero); TempCNN/InceptionTime: 200 epochs + 4 fixes ML US-022-b (`class_weights`, `weighted_sampler`, `lr_scheduler` warmup+cosine, `early_stopping` patience=20). | celda Markdown §"Hiperparametros v2" |
| AC-P8-4 | Spatial CV 5-fold (mismo splitter de US-022b para comparabilidad) — buffer 1 km entre folds, stratificado por clase | grep `build_spatial_kfold` o `SpatialBlockCV` en celda 3 modelos + log de cada fold |
| AC-P8-5 | 3 MLflow runs nuevos registrados con tags `data_version` (DVC short hash del fused-features-v2) + `code_version` (git sha): `baseline-v2-xgb`, `baseline-v2-tempcnn`, `baseline-v2-inceptiontime` | UI MLflow muestra los 3 runs |
| AC-P8-6 | Decision documentada en seccion §"Modelo baseline v2 ganador" del notebook: el modelo con F1-macro mas alto se promueve como referencia para EPIC 5 (US-023 U-Net) + 2 modelos restantes quedan como base learners del stacking EPIC 6. | celda Markdown final |
| AC-P8-7 | Wall clock total <= 90 min (3 modelos x ~30 min cada uno en RTX 4070 + spatial CV 5-fold) | tiempo en cada celda + total al cierre |
| AC-P8-8 | Tabla LaTeX `baseline_v2_comparison.tex` exportada a `paper/tables/us-023-preview/` para reuso futuro en Paper Track | archivo existe |
| AC-P8-9 | DVC tag nuevo `fused-features-italy-v2` con el parquet de features post-decisiones P2/P3/P4/P5 | `git tag -l 'fused-features-italy-v2'` no vacio |

### 2.9 P9 — Categoria "Baseline" en dashboard Streamlit (2 SP)

| AC | Criterio | Como verificar |
|----|----------|----------------|
| AC-P9-1 | Nueva constante `_SECTION_BASELINE = "Baseline (US-023-preview)"` agregada al selector de secciones en [`app/eda_dashboard.py`](../../app/eda_dashboard.py) — `_SECTION_OPTIONS` pasa de `(_SECTION_EDA, _SECTION_FE)` a `(_SECTION_EDA, _SECTION_FE, _SECTION_BASELINE)` | grep `_SECTION_BASELINE` en `app/eda_dashboard.py` |
| AC-P9-2 | Funcion `_render_baseline_section()` que reusa el patron `_render_section_divider` + cards + figuras + tablas ya usado en EDA/FE | grep `_render_baseline_section` |
| AC-P9-3 | La seccion Baseline muestra 5 tabs (replica el patron de tabs de EDA): **(1) Ablation de features** (plot `ablation_optional_blocks.png` + tabla `ablation_table.parquet` con 7-10 conjuntos + interpretacion en lenguaje accesible) **(2) Leakage geografico** (plot `ablation_geom_comparison.png` + tabla `geom_only` vs `full` + narrativa "Por que descartar `geom_*`") **(3) Bloques opcionales** (3 plots y 3 tablas: FarSLIP P2 + Gemini P4 + firma espectral P5 + decisiones promover/descartar) **(4) Modelos baseline v2** (plot `model_comparison_v2.png` + tabla con 3 modelos x 6 metricas + comparativa v1 vs v2 + decision modelo ganador) **(5) Conclusiones** (resumen de los 4 hallazgos H-1..H-4 + lo que sigue en EPIC 5) | inspeccion visual + spotcheck de los 5 tabs |
| AC-P9-4 | Las figuras se leen desde `paper/figures/us-023-preview/` (artefactos P3+P8) — sin duplicar PNGs | `os.path.exists(...)` de cada PNG referenciado |
| AC-P9-5 | Las tablas se leen desde `reports/baseline/feature_ablation/` y `reports/baseline/model_comparison_v2/` (artefactos P2+P4+P5+P8) — sin duplicar parquets | `pl.read_parquet(...)` exitoso en `app/eda_dashboard.py` |
| AC-P9-6 | Idioma: todos los textos visibles al usuario en espanol UTF-8; identificadores ASCII puro — alineado con `notebooks/CLAUDE.md` §"Idioma" | spotcheck manual |
| AC-P9-7 | Test smoke: `streamlit run app/eda_dashboard.py --server.headless true` arranca sin errores en consola, navegacion a "Baseline" muestra los 5 tabs sin tracebacks | manual `streamlit run` + screenshot |
| AC-P9-8 | Docstring de la nueva funcion `_render_baseline_section` con Google style en espanol, type hints | spotcheck |
| AC-P9-9 | Sin nuevas dependencias en `pyproject.toml` (reusa `streamlit`, `polars`, `matplotlib` ya instalados) | `git diff pyproject.toml` debe ser vacio para este AC |

---

## 3. Arquitectura de la solucion

```
US-023-preview — Flujo de capas
================================

[P1 rutas + builders + Makefile]
        |
        v
[P6 estandar notebooks/CLAUDE.md] <----+
        |                              |
        v                              |
[notebooks/baseline/05_reencuadre_fenologico.ipynb]
        |       |              |       |
        v       v              v       v
   [P3 geom] [P2 FarSLIP] [P4 Gemini] [P5 firma espectral]
        |       |              |       |
        v       v              v       v
   ablation_table.parquet (7-10 conjuntos) + MLflow runs + paper/figures/
        |
        v
[P8 baseline v2 — 3 modelos sobre conjunto ganador]
        |
        +--> notebooks/baseline/04_baseline.ipynb v2
        +--> model_comparison_v2.parquet (XGBoost + TempCNN + InceptionTime)
        +--> 3 MLflow runs baseline-v2-*
        +--> DVC tag fused-features-italy-v2
        |
        v
[P9 dashboard Streamlit — categoria Baseline]
        |
        +--> _SECTION_BASELINE + _render_baseline_section()
        +--> 5 tabs: Ablation / Leakage / Bloques opcionales / Modelos v2 / Conclusiones
        |
        v
[P7 plan v6 referencia US-023-preview]
        |
        v
EPIC 5 arranca con baseline ya saneado, conjuntos de features decididos,
3 modelos baseline v2 reentrenados y dashboard visual operativo
```

**Estado origen** (commit base, rama `us-022-cierre`): notebooks movidos a `notebooks/baseline/`, FarSLIP en `data/farslip/embeddings_italy_v{1,2}.parquet`, ablation entregada sin `with_farslip`/`farslip_only`, sin barra aislada `no_geom`, sin firma espectral, sin ablation real `pheno_text`.

**Estado destino**: notebooks ejecutables end-to-end con papermill desde `notebooks/baseline/`, FarSLIP visible en el path canonico, ablation con 7-10 conjuntos (incluyendo `with_farslip`, `farslip_only`, `with_pheno_text`, `pheno_text_only`, `with_spectral_signature`, `geom_only`), conclusiones que cuantifican cada bloque opcional y descisiones documentadas para EPIC 5 / EPIC 6.

---

## 4. Plan de implementacion

### 4.1 Archivos a crear

| Ruta | Sub-bloque | Proposito |
|------|------------|-----------|
| `ml/features/spectral_signature.py` | P5 | Transformer sklearn-compatible `SpectralSignatureFeatures` con K features compactas (REP / SAM / momentos red-edge en SOG/peak/senescence) |
| `tests/ml/features/test_spectral_signature.py` | P5 | 6+ tests con fixtures sinteticas (cobertura >= 80%) |
| `docs/manual-test/us-023-preview.md` | cierre QA | Comandos exactos para reproducir P1-P9 |
| `docs/us-handoff/us-023-preview.md` | tracking | Template de handoff (creado en este mismo planning) |
| `docs/us-resolved/us-023-preview.md` | cierre | Resumen final + tabla de resultados de los 9 sub-bloques (al cerrar la US) |
| `paper/figures/us-023-preview/ablation_geom_comparison.png` | P3 | Plot con 2 barras `full` vs `no_geom` |
| `paper/figures/us-023-preview/ablation_optional_blocks.png` | P2+P4+P5 | Plot agregado: `full` vs `with_farslip` vs `with_pheno_text` vs `with_spectral_signature` |
| `paper/figures/us-023-preview/model_comparison_v2.png` | P8 | Plot con 3 barras (XGBoost + TempCNN + InceptionTime) v2 + overlay deltas vs v1 |
| `paper/tables/us-023-preview/baseline_v2_comparison.tex` | P8 | Tabla LaTeX comparativa para reuso futuro en Paper Track |
| `reports/baseline/model_comparison_v2/model_comparison_v2.parquet` | P8 | Tabla con 3 modelos x 6 metricas (F1-macro, F1-weighted, mIoU, accuracy, kappa, train_time_s) |
| `tests/app/test_eda_dashboard_baseline_section.py` | P9 | Smoke test de la nueva seccion Baseline (import + render no-error) |

### 4.2 Archivos a modificar

| Ruta | Sub-bloque | Cambio |
|------|------------|--------|
| `scripts/build_baseline_notebook.py` | P1 + P8 | Default `--out notebooks/baseline/04_baseline.ipynb`; agregar celdas v2 para reentrenar XGBoost + TempCNN + InceptionTime sobre conjunto ganador |
| `scripts/build_reencuadre_notebook.py` | P1 + P2 + P3 + P4 + P5 + P6 | Default `--out notebooks/baseline/05_reencuadre_fenologico.ipynb`; agregar celdas para `geom_only`, `with_farslip`, `farslip_only`, `with_pheno_text`, `pheno_text_only`, `with_spectral_signature`; aplicar QA `notebooks/CLAUDE.md` (display, autoreload, conclusiones sin US-XXX) |
| `Makefile` | P1 + P8 | Targets `baseline-notebook`, `reencuadre-notebook`, `reencuadre-notebook-check`, `reencuadre-notebook-full` apuntan a `notebooks/baseline/`; nuevo target `baseline-v2-full` (papermill notebook 04 v2 con CUDA) |
| `notebooks/CLAUDE.md` | P1 | Actualizar §"Estructura Canonica" con paths `notebooks/baseline/04_baseline.ipynb` y `notebooks/baseline/05_reencuadre_fenologico.ipynb` |
| `ml/eval/feature_ablation.py` | P3 + P5 | Agregar `geom_only` a `build_default_feature_sets` (tupla de `geom_*` cols, solo si existen) y `with_spectral_signature`/`spectral_signature_only` cuando el bloque este materializado |
| `ml/features/fusion.py` | P2 + P5 | P2: patch defensivo en `_build_farslip_block` (aceptar prefix `farslip_` y `farslip_emb_`). P5: parametro `include_spectral_signature: bool = False`, constante `EXPECTED_COL_COUNT_WITH_SPECTRAL_SIGNATURE`, LEFT JOIN sobre parcel_id |
| `tests/ml/features/test_fusion.py` | P2 + P5 | P2: 1 test para parquet con prefix `farslip_emb_`. P5: 2 tests nuevos para el bloque opcional spectral_signature (presencia/ausencia) |
| `tests/ml/eval/test_feature_ablation.py` | P3 + P5 | 2 tests nuevos: `geom_only` aparece cuando hay cols `geom_*`; `with_spectral_signature` aparece cuando hay cols `spectral_signature_*` |
| `data/farslip/embeddings_italy.parquet` (+ `.dvc`) | P2 | Promover v2 (real epoch_2) al path canonico estable (sin sufijo) + renombrar cols `farslip_emb_XXX -> farslip_XXX`. DVC tracked. |
| `notebooks/baseline/05_reencuadre_fenologico.ipynb` (regen) | P2-P6 | Salida de papermill con outputs poblados, 0 errores |
| `notebooks/baseline/04_baseline.ipynb` (regen v2) | P1 + P6 + P8 | Path nuevo + QA estandar + 3 modelos baseline v2 (XGB + TempCNN + InceptionTime) sobre conjunto ganador post-ablation |
| `notebooks/baseline/html/*.html` | P1 + P8 | Regenerar (incluye `04_baseline.html` v2) |
| `app/eda_dashboard.py` | P9 | Agregar `_SECTION_BASELINE` al selector + funcion `_render_baseline_section()` con 5 tabs (Ablation / Leakage / Bloques opcionales / Modelos v2 / Conclusiones) |
| `docs/us-resolved/us-022b.md` | P3 | Anexar nota §"Resultados US-023-preview" enlazando al notebook nuevo |
| `docs/us-resolved/us-022-c.md` | P2 | Marcar gate B-4 VERDE retroactivamente |
| `docs/l4_log.md` | P4 | Entrada nueva con costos Gemini |
| `docs/licenses/DATA_LICENSE.md` | P5 (si aplica) | Si el descriptor de firma espectral usa un dataset/paper nuevo (e.g., Frampton et al. 2013 para REP), agregar atribucion |
| `context/RefinamientoPlaneacionAgroSatCopilot_v6.md` | P7 | Entrada nueva US-023-preview en EPIC 4 + secuenciacion semanal |

### 4.3 Archivos existentes reutilizados (NO duplicar)

- `ml/eval/feature_ablation.py:build_default_feature_sets` — ya construye `with_farslip` / `farslip_only` / `with_pheno_text` graceful. Solo se agregan dos llaves (`geom_only`, `with_spectral_signature`).
- `ml/features/fusion.py:build_fused_features` — ya soporta `include_farslip` y `include_phenology_text`. Replicar patron para `include_spectral_signature`.
- `ml/features/phenology_description.py` — cliente Gemini 3.5 Flash + text-encoder ya implementado en US-022-b. P4 lo ejecuta sobre subset mas grande con `skip_llm=False`.
- `ml/eval/reencuadre_plots.py:plot_ablation_bars`, `plot_model_comparison_bars` — funciones ya existen. P3/P8 las invocan con orden nuevo.
- `ml/train/baseline.py:train_one_model` (XGB), `ml/train/phenology_models.py:train_temporal_model` (TempCNN, InceptionTime) — reusados directamente por P8 (no reescribir wrappers).
- `scripts/build_reencuadre_notebook.py`, `scripts/build_baseline_notebook.py` — patrones del builder. Se agregan celdas (no se reescribe).
- `ml/utils/notebook_setup.py:find_repo_root`, `configure_ee_from_env` — bootstrap estandar reusado.
- `ml/utils/spatial_cv.py:build_spatial_kfold` — splitter spatial CV reusado en P8 (mismo de US-022b para comparabilidad v1 vs v2).
- `app/eda_dashboard.py:_render_section_divider`, `_render_card_section`, `_render_figures_section`, `_render_tables_section` — helpers de Streamlit reusados por P9 sin modificar.
- `data/farslip/embeddings_italy_v{1,2}.parquet` — v2 se promueve al path canonico (no se regenera FarSLIP).

### 4.4 Interfaces publicas nuevas

```python
# ml/features/spectral_signature.py
from sklearn.base import BaseEstimator, TransformerMixin
import polars as pl


class SpectralSignatureFeatures(BaseEstimator, TransformerMixin):
    """Genera features compactas derivadas de la firma espectral por parcela.

    Args:
        bands: Tupla de bandas Sentinel-2 disponibles en el frame de entrada
            (default ``("B04", "B05", "B06", "B07", "B08")`` — red + 3 red-edge + NIR).
        phenology_anchors: Anclajes temporales sobre los que se calculan
            momentos espectrales: ``("sog", "peak", "senescence")``.
        descriptor: Tipo de descriptor. Uno de ``"rep"`` (Red Edge Position,
            Frampton et al. 2013), ``"sam"`` (Spectral Angle Mapper vs centroide
            de clase), o ``"redge_moments"`` (mean/var/skew de red-edge en
            cada anclaje).

    Returns:
        ``pl.DataFrame`` con columnas ``parcel_id, year, spectral_signature_*``.
    """

    def __init__(
        self,
        bands: tuple[str, ...] = ("B04", "B05", "B06", "B07", "B08"),
        phenology_anchors: tuple[str, ...] = ("sog", "peak", "senescence"),
        descriptor: str = "rep",
    ) -> None: ...

    def fit(self, X: pl.DataFrame, y: object | None = None) -> "SpectralSignatureFeatures": ...

    def transform(self, X: pl.DataFrame) -> pl.DataFrame: ...


# ml/features/fusion.py — parametro nuevo
def build_fused_features(
    parcels: gpd.GeoDataFrame,
    year: int,
    *,
    include_spectral_signature: bool = False,
    spectral_signature_path: str | Path | None = None,
    ...
) -> pl.DataFrame: ...


# ml/eval/feature_ablation.py — nuevas llaves del mapping
def build_default_feature_sets(
    available_cols: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """
    ...
    - ``geom_only``: solo ``geom_*`` (3 cols) — test cuantitativo de leakage espacial.
    - ``with_spectral_signature``: ``phenology_only`` + ``spectral_signature_*``.
    - ``spectral_signature_only``: solo ``spectral_signature_*``.
    """
```

---

## 5. Dominios tocados y subagentes Fase 3

- [ ] backend
- [x] **frontend** — P9 (dashboard Streamlit `app/eda_dashboard.py`, no Nuxt — es la app de EDA del equipo)
- [x] **ml** — P2 (path FarSLIP + patch fusion), P3 (ablation geom), P4 (ablation pheno_text Gemini real), P5 (modulo `spectral_signature.py`), P6 (estandar notebooks), P8 (baseline v2 con 3 modelos)
- [ ] agent (ml/agent/)
- [ ] infra
- [ ] db (sin migraciones)
- [ ] dagster (no se tocan assets)

Skills a activar:

| Sub-bloque | Skill |
|------------|-------|
| P1 + P6 | `agrosat-ml-baseline` (notebooks baseline + estandar) |
| P2 | `agrosat-ml-baseline` + `agrosat-dvc-mlflow` (promover parquet FarSLIP + DVC tag + MLflow) |
| P3 | `agrosat-ml-baseline` + `agrosat-ml-evaluation` (plots ablation + interpretacion) |
| P4 | `agrosat-llm-finetuning` (Gemini Flash 3.5 client) + `agrosat-ml-baseline` |
| P5 | `agrosat-ml-features` (transformer + indices) + `agrosat-testing` (pytest) |
| P7 | `agrosat-git-workflow` (Conventional Commit + PR a `develop`) |
| P8 | `agrosat-ml-baseline` (XGBoost + spatial CV) + `agrosat-ml-evaluation` (3 modelos comparativa) + `agrosat-dvc-mlflow` (3 MLflow runs + DVC tag fused-features-v2) |
| P9 | `agrosat-frontend-components` (patron de cards/tabs en Streamlit es analogo a Vue components) + `agrosat-testing` (smoke test) |

**Subagentes profundos** (lanzar en Fase 3 paralelo si el coding excede 6h):

- `ml-engineer` — P2, P3, P4, P5, P8 (ML, modelo, ablation, baseline v2)
- `mlops-engineer` — P1 paths + P7 plan v6
- `frontend-engineer` — P9 dashboard Streamlit (notar: aunque Streamlit no es Nuxt, el patron de secciones/tabs/cards es similar al UX que ya domina este subagente)

---

## 6. Plan de tests

| Modulo | Tests | Cobertura objetivo |
|--------|-------|---------------------|
| `ml/features/spectral_signature.py` | `test_spectral_signature.py` — 6 tests: shape, determinismo (seed), 3 descriptores (rep/sam/redge_moments), error si bandas faltan, edge case (parcela sin anclaje fenologico) | >= 80% |
| `ml/features/fusion.py` | 3 tests nuevos en `test_fusion.py`: bloque spectral_signature presente/ausente + parquet FarSLIP con prefix `farslip_emb_` (patch defensivo P2) | (sumar al 79% actual; mantener >= 75%) |
| `ml/eval/feature_ablation.py` | 2 tests nuevos en `test_feature_ablation.py`: `geom_only` cuando hay `geom_*`; `with_spectral_signature` cuando hay `spectral_signature_*` | (mantener >= 90%) |
| `app/eda_dashboard.py` (P9) | `test_eda_dashboard_baseline_section.py` — 4 tests: `_SECTION_BASELINE` esta en `_SECTION_OPTIONS`; `_render_baseline_section` callable sin error; falta de un parquet artefacto no rompe el render (graceful degradation con `st.warning`); idioma de strings en espanol | (cobertura no critica por ser UI; smoke test es suficiente) |
| Notebook smoke | `make notebooks-check` ejecuta papermill end-to-end sobre las 2 libretas (incluida la v2 de `04_baseline.ipynb`) | exit 0 |
| Manual test | `docs/manual-test/us-023-preview.md` con comandos paso-a-paso para reproducir P1-P9 | exit 0 esperado |

**Suite global esperada**: 117 tests US-022-b + 6 spectral_signature + 3 fusion + 2 ablation + 4 dashboard = **132 tests passing**.

---

## 7. Computo y presupuesto

- **GPU**: 0 h H100 (no toca V1-V6). 0 h L4 Vertex AI. P4 (Gemini Flash 3.5) corre cloud-side via API key. P2 (promocion FarSLIP) es operacion de filesystem + DVC, sin computo.
- **CPU/GPU local**: P3 + P4 + P5 + papermill notebooks en RTX 4070 / CPU equipo. Wall clock estimado ~3-4 h total.
- **P8 baseline v2** (3 modelos): RTX 4070 batch=128. XGBoost ~5 min CPU + TempCNN ~30 min CUDA + InceptionTime ~30 min CUDA + spatial CV 5-fold = wall clock ~90 min total (`AC-P8-7`).
- **P9 dashboard Streamlit**: 0 GPU. Pre-renderiza tablas/figuras desde parquet/PNG ya generados — sin re-computo.
- **Gemini Flash 3.5**: <= $5 USD (P4 sobre subset >= 1000 parcelas). Smoke previo: $0.023 USD por 216 parcelas = ~$0.10 USD por 1000 parcelas linealmente, holgura 50x.
- **DVC push** (P2 + P8): <= 200 MB del parquet FarSLIP promovido + ~10 MB de `fused-features-italy-v2` (ya optimizado).

**Presupuesto Avance 5 / H100 intacto**: 80 h H100 reservadas para V2-V6 sin tocar.

---

## 8. Versionado de datos y modelos

| Artefacto | Tag git previsto | Tamano | Donde |
|-----------|------------------|--------|-------|
| `data/farslip/embeddings_italy.parquet` (promocion) | crea `farslip-embeddings-italy-v1` (gate B-4 US-022-c pendiente) | ~50 MB (v2) | `gs://agrosat-dvc-remote` |
| `data/features/phenology_text_italy.parquet` (ampliado vs subset US-022-c) | `phenology-text-italy-v1` | ~1-3 MB | `gs://agrosat-dvc-remote` |
| `data/features/spectral_signature_italy.parquet` (nuevo P5) | `spectral-signature-italy-v1` | <1 MB | `gs://agrosat-dvc-remote` |
| `data/features/features_fused_v2_italy.parquet` (nuevo P8) | `fused-features-italy-v2` | ~10 MB | `gs://agrosat-dvc-remote` |
| MLflow runs P2/P4/P5 ablations | `baseline-farslip-ablation-v1`, `baseline-pheno-text-ablation-v1`, `baseline-spectral-signature-ablation-v1` | n/a | UI MLflow Cloud Run |
| MLflow runs P8 baseline v2 | `baseline-v2-xgb`, `baseline-v2-tempcnn`, `baseline-v2-inceptiontime` | n/a | UI MLflow Cloud Run |

Cada run con tags `data_version` (DVC short hash) + `code_version` (git sha) — regla R10 de [`CLAUDE.md`](../../CLAUDE.md).

---

## 9. Riesgos y mitigaciones

| ID | Riesgo | Probabilidad | Impacto | Mitigacion |
|----|--------|---------------|---------|------------|
| R1 | FarSLIP v1/v2 entrega 30173 parcelas vs dataset full 85951 — al promoverlo al path canonico, `_build_farslip_block` hace LEFT JOIN dejando 55778 parcelas con cols FarSLIP en NaN. XGBoost tolera NaN; otros modelos pueden necesitar imputacion. La extraccion real para las 85951 parcelas queda en US-025 (ver `_project_parcels_to_embeddings` placeholder en US-022-c). | alta | medio | Documentar `n_matched_parcels=30173` y `n_with_nan=55778` en notebook; usar XGBoost (NaN-aware) como modelo base; reportar delta sobre las 30173 matched (subset) Y sobre el full (con NaN imputado a media) para honestidad |
| R2 | Gemini Flash 3.5 quota / rate limit en subset >=1000 parcelas | baja | medio | Batching + retry con backoff exponencial (ya implementado en `phenology_description.py`); fallback cap a 500 parcelas con justificacion documentada |
| R3 | Firma espectral no aporta señal (delta <= 0) | media | bajo | AC-P5-5 prevee documentar como deuda de investigacion sin bloquear la US (resultado honesto, mismo patron P3 US-022-c con TempCNN) |
| R4 | Tests cobertura cae < 75% por nuevos modulos | baja | medio | 6 tests por modulo (skill `agrosat-testing` patron) con fixtures sinteticas; spot-check coverage antes del PR |
| R5 | Conflictos de merge con PR #24/#25 (refactor masivo, 79+ archivos modificados) | alta | alto | Mergear PR #24 o PR #25 a `develop` ANTES de iniciar Fase 3; rebasing `us-023-preview` sobre `develop` actualizado; spot-check `ml/eval/feature_ablation.py`, `ml/features/fusion.py`, `scripts/build_reencuadre_notebook.py` (tocados por ambos) |
| R6 | El path canonico `data/farslip/embeddings_italy.parquet` puede colisionar con cache DVC si hay artefactos previos | baja | bajo | Antes del `dvc add` correr `dvc remove data/farslip/embeddings_italy.parquet.dvc` si existe y verificar `.gitignore` |
| R7 | El movimiento de notebooks (P1) puede dejar enlaces rotos en docs viejos (us-022-b, us-022-c, ADR-006) | media | bajo | grep cross-repo de `notebooks/feature_engineering/05_reencuadre` y `notebooks/04_baseline.ipynb`; actualizar enlaces en `us-022-b.md`, `us-022-c.md`, `ADR-006.md` |
| R8 | Si PR #23 (script `train_phenology_models.py`) introduce emoji `✓` en print stderr, podria romper R4 sin emojis | baja | bajo | Refactor `print(f"✓ {model_kind} complete")` -> `print(f"OK {model_kind} complete")` o aceptarlo como marcador semantico discreto en stderr (no en codigo de aplicacion) |
| R9 | P8 baseline v2: TempCNN/InceptionTime sobre conjunto ganador post-ablation pueden tardar > 90 min si el conjunto incluye `with_farslip` + `with_pheno_text` + `with_spectral_signature` (mas features = mas memoria + tiempo) | media | medio | Cap batch=64 si VRAM excede; early stopping patience=15 (vs 20 default); documentar wall clock real en notebook v2 |
| R10 | P8 modelos v2 pueden NO superar v1 (ya un riesgo conocido en `us-022-d`: TempCNN/InceptionTime ~0.18 vs XGB 0.41) — la "v2" puede ser solo XGB el ganador otra vez | alta | bajo | Documentar honestamente; el valor de P8 es la **trazabilidad reproducible** del baseline post-ablation, no necesariamente superar v1. Si XGB v2 supera XGB v1 (>0.4094), ya hay valor incremental |
| R11 | P9 dashboard: si los artefactos de P2/P4/P5/P8 no estan en disco al cargar Streamlit, los renders fallan con tracebacks | media | bajo | Patron graceful: `if not parquet_path.exists(): st.warning("Artefacto pendiente — ejecuta `make reencuadre-notebook-full && make baseline-v2-full`")` |
| R12 | P8 + P9 dependen de P2/P3/P4/P5 cerrados — si alguno se atrasa, P8/P9 se bloquean | media | medio | Orden estricto en Fase 3: P1->P2->(P3//P4//P5)->P6->P8->P9->P7. Si un sub-bloque opcional (P4 o P5) falla, P8 corre con el conjunto ganador disponible y P9 muestra solo los tabs poblados |

---

## 10. Mapeo a rubrica del Avance correspondiente

US-023-preview **no entrega Avance nuevo** (A3 ya calificado 100/100). Refuerza retrospectivamente los criterios del A3:

| Criterio A3 | Pts | Como lo refuerza esta US |
|-------------|-----|---------------------------|
| Algoritmo (40 pts) | 40 | P2 + P4 + P5 amplian la ablation a 7-10 conjuntos cuantificando aporte de cada bloque opcional. P8 reentrena los 3 modelos canonicos del A3 sobre el conjunto ganador. Justifica el algoritmo seleccionado con evidencia comparativa explicita (v1 vs v2) |
| Caracteristicas importantes (20 pts) | 20 | P3 grafica `geom_only` aislada + interpretacion; P5 introduce descriptor compacto de firma espectral con justificacion agronomica |
| Sub/sobreajuste (10 pts) | 10 | Spatial CV 5-fold preservado en todas las ablations (P2/P4/P5) y en P8 — sin cambios sobre US-021 |
| Metrica (20 pts) | 20 | F1-macro + mIoU reportados en todas las nuevas filas y para los 3 modelos v2; delta vs `full` y delta vs v1 documentados |
| Desempeno (10 pts) | 10 | Decisiones promover/descartar/diferir documentadas por bloque (P2/P4/P5) + decision modelo ganador v2 (P8) — alimenta criterio "Desempeno" del A4 |

**Aporte prospectivo al A4 (24-may)** y al ensemble del **EPIC 6** (A5 31-may): el conjunto de features, los 3 base learners reentrenados y el dashboard visual (P9) ya quedan definidos antes de arrancar U-Net (US-023 v6), reduciendo iteracion en EPIC 5.

---

## 11. Decisiones tecnicas clave

- **D-1**: P2 promueve **v2** al path canonico `data/farslip/embeddings_italy.parquet`. v2 es la extraccion real (epoch_2 student `safetensors`, commit `0f01255`), v1 es placeholder determinista seeded (ver [`docs/us-resolved/us-022-c.md:35`](../us-resolved/us-022-c.md#L35)). Ademas se renombran columnas `farslip_emb_XXX -> farslip_XXX` para alinear con el contrato de [`ml/features/fusion.py:582`](../../ml/features/fusion.py#L582) `_build_farslip_block` (sin tocar v2 original — el rename ocurre en el script de promocion y produce el parquet canonico nuevo). Patch defensivo opcional en `fusion.py` para aceptar ambos prefijos (futuro-proof).
- **D-2**: P4 corre Gemini Flash 3.5 sobre **subset >= 1000 parcelas balanceadas**, NO sobre el dataset full (85951) — costo + tiempo wall clock. Si delta es positivo y significativo, US-024 (backlog) escalaria a full.
- **D-3**: P5 elige UN descriptor de firma espectral (no los 3 candidatos) para no introducir overhead de exploracion. Recomendado: **Red Edge Position (REP)** por Frampton et al. 2013 — bien establecido en literatura agronomica, computable desde S2 ya muestreado.
- **D-4**: P3 NO modifica la decision ya documentada en US-022-b (descartar `geom_*`). Solo agrega evidencia visual aislada + test `geom_only` por trazabilidad.
- **D-5**: La US **no toca H100** (V1-V6 intactos). Tampoco lanza training pesado en L4 Vertex AI. Trabajo local + 1 llamada cloud (Gemini).
- **D-6**: P1 mueve notebooks a `notebooks/baseline/` definitivamente. `notebooks/feature_engineering/` queda solo para `03a/03b/03c/Avance2.Equipo17.ipynb` (EPIC 3). `notebooks/features/04_farslip_eval_pastis.ipynb` (US-022-c) queda donde esta.
- **D-7**: P7 actualiza el plan v6 pero NO renombra US-023 U-Net del EPIC 5 — esta US se llama `US-023-preview` precisamente para no colisionar.
- **D-8**: PRs #24/#25 (refactor masivo en ramas `User/abocanegra/improve_ci` y `fix/ci-and-xgb`) se mergean ANTES de arrancar US-023-preview Fase 3 — evita rebases dolorosos. PR #23 (script `train_phenology_models.py`) se puede mergear independientemente pero R8 aplica.
- **D-9 (P8)**: el conjunto de features ganador post-P2/P3/P4/P5 puede tener entre 185 y ~1100 features (185 base + 512 FarSLIP + 384 pheno_text + K spectral_signature). Si supera 1000 features, P8 aplica un paso de feature selection ligero con XGBoost importance top-200 antes de TempCNN/InceptionTime para evitar OOM en VRAM y reducir wall clock. XGBoost v2 si corre con el conjunto completo (tolera dimensionalidad).
- **D-10 (P8)**: el "modelo ganador v2" se decide por F1-macro sobre spatial CV 5-fold (mismo splitter que US-022b). Empate por F1-macro se rompe por F1-weighted; empate por F1-weighted se rompe por mIoU.
- **D-11 (P9)**: el dashboard Streamlit es la herramienta interna de EDA/FE/baseline del equipo — distinto de la app Nuxt principal (chat conversacional con ADK). P9 no toca `frontend/` Nuxt.
- **D-12 (P9)**: tabs del dashboard se cargan lazy con `st.cache_data` sobre `pl.read_parquet(...)` para evitar leer todos los artefactos cada vez que el usuario cambia de tab.

---

## 12. Aprovechamiento de PRs abiertas (#23, #24, #25)

Auditoria 25-may-2026:

| PR | Titulo | Estado | Aporte a US-023-preview |
|----|--------|--------|--------------------------|
| #23 | `feat(E4): fixing train phenology inline issues` | abierta, CI rojo (1 archivo) | **Conservar**. Refactor de `Makefile:phenology-train` a `scripts/train_phenology_models.py` (68 lineas, argparse limpio). Util como helper P4/P8 si se reentrena temporal con `pheno_text` ampliado o se llama desde el notebook v2. Riesgo: emoji `✓` en stderr (R8 arriba) — refactor en US-023-preview. |
| #24 | `fix: ci and xgb` | abierta, CI verde | **Mergear ANTES de US-023-preview Fase 3**. Refactor masivo (79 archivos, +1184/-1396): mejoras a `ml/eval/feature_ablation.py`, `ml/features/fusion.py`, `ml/features/phenology_description.py`, `scripts/build_reencuadre_notebook.py` — todos tocados por esta US. Conservar las simplificaciones. |
| #25 | `User/abocanegra/improve ci` | abierta, CI verde | **Mergear DESPUES de #24 si #25 es superset, o en lugar de #24 si son alternativas**. Aparenta ser el mismo refactor con `.github/workflows/ci.yml` mas extenso (+93/-13 vs +38/-5). Decidir cual mergear (uno solo). |

**Plan de merge**:

1. Mergear #24 o #25 (no ambos — son redundantes) a `develop`.
2. Mergear #23 a `develop` (puede ir en paralelo).
3. Crear rama `feature/E4-US-023-preview-baseline-corrections` desde `develop` actualizado.
4. Arrancar Fase 3 (coding) de US-023-preview sobre base limpia.

---

## 13. Checklist de cierre

- [ ] Rama `feature/E4-US-023-preview-baseline-corrections` mergeada via PR a `develop`
- [ ] Conventional Commit: `feat(E4): US-023-preview correcciones baseline post-A3 — FarSLIP path canonico + ablation geom/Gemini/firma espectral + estandar notebooks`
- [ ] Codigo en ingles; docstrings Google en espanol con type hints
- [ ] Tests cobertura >= 70% backend (n/a aqui), >= 75% ML diff
- [ ] `make check` limpio (ruff + secrets-scan + i18n-check)
- [ ] `make notebooks-check` exit 0 sobre `notebooks/baseline/04_baseline.ipynb` y `notebooks/baseline/05_reencuadre_fenologico.ipynb`
- [ ] Notebooks commiteados con outputs poblados (HTML tables + PNG inline)
- [ ] Migraciones: n/a (sin schema change)
- [ ] Secrets via `.env.local` (Gemini API key) — sin literales en code
- [ ] MLflow tags `data_version` + `code_version` set en los 3 runs nuevos
- [ ] DVC: `data/farslip/embeddings_italy.parquet.dvc` + `data/features/phenology_text_italy.parquet.dvc` + `data/features/spectral_signature_italy.parquet.dvc` + `data/features/features_fused_v2_italy.parquet.dvc` commiteados; `dvc push` ejecutado
- [ ] Tag git `farslip-embeddings-italy-v1` creado y publicado (gate B-4 US-022-c cerrado retroactivamente)
- [ ] Tag git `fused-features-italy-v2` creado y publicado (P8)
- [ ] Streamlit P9: `streamlit run app/eda_dashboard.py` arranca sin errores; navegacion a "Baseline" muestra los 5 tabs poblados
- [ ] i18n: n/a (Streamlit del equipo, solo espanol UTF-8)
- [ ] Atribucion licencia: si P5 cita Frampton et al. 2013, agregar a `docs/licenses/DATA_LICENSE.md`
- [ ] `docs/us-resolved/us-023-preview.md` creado con resumen ejecutivo
- [ ] `docs/us-handoff/us-023-preview.md` actualizado a `Status: closed`
- [ ] `context/RefinamientoPlaneacionAgroSatCopilot_v6.md` referencia US-023-preview en EPIC 4 + secuenciacion S6
- [ ] engram-memory: `mem_save` con cierre (resultados ablation 7-10 conjuntos + decisiones promover/descartar)

---

## 14. Referencias

- ADR-006: [`docs/decisions/ADR-006-reencuadre-baseline-fenologico.md`](../decisions/ADR-006-reencuadre-baseline-fenologico.md)
- ADR-007: [`docs/decisions/ADR-007-farslip-fidelity-paper.md`](../decisions/ADR-007-farslip-fidelity-paper.md)
- US predecesoras: [US-022](../us-resolved/us-022.md), [US-022-b](../us-resolved/us-022b.md), [US-022-c](../us-resolved/us-022-c.md)
- Estandar notebooks: [`notebooks/CLAUDE.md`](../../notebooks/CLAUDE.md)
- Paper-faro fenologico: Wen et al. (2025), *Phenology description is all you need!*, ISPRS J. 228, 141-165. DOI [10.1016/j.isprsjprs.2025.07.002](https://doi.org/10.1016/j.isprsjprs.2025.07.002)
- Paper Red Edge Position (P5 D-3 candidato): Frampton, W.J. et al. (2013), *Evaluating the capabilities of Sentinel-2 for quantitative estimation of biophysical variables in vegetation*, ISPRS J. 82, 83-92. DOI [10.1016/j.isprsjprs.2013.04.007](https://doi.org/10.1016/j.isprsjprs.2013.04.007)
- Backlog relacionado: [`us-022-d-temporal-daily-series.md`](../product-backlog/us-022-d-temporal-daily-series.md), [`us-022-e-farslip-france-augment.md`](../product-backlog/us-022-e-farslip-france-augment.md)
- PRs auditadas: [#23](https://github.com/ArthurZizumbo/agro_sat_copilot/pull/23), [#24](https://github.com/ArthurZizumbo/agro_sat_copilot/pull/24), [#25](https://github.com/ArthurZizumbo/agro_sat_copilot/pull/25)
- Comandos make relevantes: `make baseline-notebook`, `make reencuadre-notebook`, `make reencuadre-notebook-check`, `make reencuadre-notebook-full`, `make feature-ablation`, `make phenology-train`, `make notebooks-check`
