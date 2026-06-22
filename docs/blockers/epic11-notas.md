# EPIC 11 (Paper Track) -- Notas de bloqueos

Epica OPCIONAL post-presentacion. Aqui se registra todo lo que NO se pudo
completar de forma autonoma en local (necesita H100/GPU, auth GEE, revision
humana, eval largo, o un artefacto que no esta materializado). Regla Arthur:
cifras REALES o nada; si un dato no existe, va aqui, no se inventa.

---

## US-072 -- Seccion FarSLIP contrastivo-fenologico

Estado: seccion redactada y tablas reales generadas. Bloqueos / pendientes:

### B-072-1 -- Tabla de ablacion de bandas SIN metrica de evaluacion (log truncado)
- **Que falta**: F1-macro / mIoU por variante (rgb / nir\_rgb / 4band).
- **Causa**: los 3 logs `reports/farslip/logs/{baseline-rgb,baseline-nir,4band-pheno}.log`
  solo registran las PERDIDAS de destilacion finales (`training done`, epoch 3 de 4:
  loss_total, loss_patch, loss_cls, loss_aux). No hay un bloque de evaluacion
  consolidada (F1/mIoU) en ninguno de los 3 logs (verificado con grep de
  `macro_f1|miou|f1_macro|eval`: 0 coincidencias).
- **Lo que SI se reporto** (real, en `paper/tables/farslip-method/band_ablation.tex`):
  loss_total rgb=2.2893 / nir_rgb=2.2858 / 4band=2.3553; loss_patch
  0.0179 / 0.0181 / 0.0786. La columna F1-macro queda marcada "pendiente".
- **Como completar**: re-correr la evaluacion (no el entrenamiento) de los 3
  checkpoints `F:\checkpoints\farslip\{baseline-rgb,baseline-nir,4band-pheno}\student_epoch_3.safetensors`
  con el harness de eval por parcela, y volcar macro_f1/mIoU a un CSV
  (p.ej. `reports/farslip/metrics/band_ablation.csv`). Luego actualizar la tabla.
  Requiere la VM/GPU donde viven los checkpoints (ruta `F:\`).

### B-072-2 -- Coherencia del claim de mejora del ensamble (+0.123 vs +0.0118 vs +0.0016)
- **Que falta**: decision editorial de QUE cifra de mejora atribuir a FarSLIP en
  la seccion del manuscrito.
- **Las cifras reales** (distintas, todas correctas en su contexto):
  - `docs/final_doc/Avance7_equipo17.tex:211`: "Stacking + FarSLIP (FINAL) 0.749 / 0.850 / +0.123"
    (delta del stacking final vs el mejor modelo individual 0.625).
  - `reports/ensemble/us043_farslip_summary.json`: stacking_f1_delta_5_vs_3 = +0.01183
    (5 miembros con FarSLIP vs 3 sin FarSLIP, OOF CV).
  - El texto previo del .tex:270 dice "+0.002 en F1-macro sobre el stacking base"
    (otro corte). El plan v8 mencionaba +0.0016 del grid interno.
- **Lo que hizo la seccion US-072**: reporto el delta OOF honesto (+0.0118 del
  5-vs-3) y aclaro que difiere del +0.123 del manuscrito (que es delta vs mejor
  individual, no la contribucion aislada de FarSLIP). NO se sobre-afirmo.
- **Decision para Arthur**: confirmar si el manuscrito mantiene el "+0.123" como
  mejora del SISTEMA (correcto) y reserva el "+0.0118" como contribucion aislada
  de FarSLIP (correcto), o si se unifica el lenguaje. Es editorial, no factual.

### B-072-3 -- Ubicacion de la seccion (paper/sections/ vs docs/final_doc/)
- **Que falta**: OK del equipo sobre donde vive la seccion definitiva.
- **Lo hecho**: se creo `paper/sections/method_farslip.tex` modular (lo que la
  US-072 pide literalmente como entregable) + las 3 tablas en
  `paper/tables/farslip-method/` + el bib en `paper/bib/farslip_refs.bib`. NO se
  inyecto la seccion completa dentro de `docs/final_doc/Avance7_equipo17.tex`
  para no alterar la estructura del manuscrito ratificado sin acuerdo
  (`paper/AGENTS.md` veta "inventar main.tex/pipeline LaTeX sin acordarlo").
- **Lo que SI se toco del manuscrito** (correccion factual minima, AC-5): el
  `\bibitem{li2025farslip}` tenia arXiv id placeholder `arXiv:2502.xxxxx` y un
  titulo incorrecto; se corrigio a `arXiv:2511.14901` + titulo real, y se
  anadieron `wen2025phenology` y `huang2025bemyeyes` en ES y EN.
- **Decision para Arthur**: (a) mantener la seccion modular como anexo del Paper
  Track arXiv (US-071 ensambla el `main.tex`), o (b) incrustar
  `\input{../../paper/sections/method_farslip.tex}` dentro del manuscrito. Si (b),
  ajustar el `\graphicspath` y mover las tablas o sus rutas relativas.

### B-072-4 -- Figura de fusion dual-head (AC-6) no generada
- **Que falta**: PNG/PDF del diagrama esquematico FarSLIP 512-dim/parcela ->
  18 prototipos -> cosine per-pixel por broadcast -> alpha aprendible.
- **Causa**: es un diagrama esquematico (no un resultado entrenado). Se puede
  hacer con matplotlib/TikZ en local SIN GPU, pero requiere decision de diseno
  (TikZ inline vs PNG) que se difiere para no introducir un asset sin acuerdo.
- **Como completar**: dibujar con TikZ inline en la seccion o con un script
  matplotlib en `paper/figures/farslip-method/dual_head_fusion.png`. 100%
  autonomo cuando se decida el formato. OJO dimension: el banco a nivel parcela
  es 512-dim (extractor), la reproyeccion de prototipos del loss es 768-dim;
  la figura debe distinguirlos (no mezclar 512 y 768).

### B-072-5 -- Compilacion PDF (make docs-pdf)
- **Que falta**: PDF compilado para validar que la seccion + tablas + bib
  renderizan sin error.
- **Causa**: requiere TeX Live o Docker con LaTeX; no verificado en este entorno.
- **Como completar**: `make docs-pdf` (o `docs-pdf-docker`) en un entorno con
  LaTeX. Si la seccion modular se compila aislada, anadir su propio preambulo
  (la cabecera de `method_farslip.tex` documenta el `\graphicspath` y los
  `\bibitem` requeridos).

### B-072-6 -- Inconsistencia menor "Gemini 2.5-Flash" en el abstract del manuscrito
- **Observacion (no corregida en US-072, fuera de scope)**: el abstract y el
  parrafo "Soberania de datos" de `docs/final_doc/Avance7_equipo17.tex` nombran
  "Gemini 2.5-Flash" como reasoner del copiloto. La correccion factual del
  proyecto indica que el reasoner FROZEN es **Gemini 2.5-pro GA** (1M ctx);
  "Gemini Flash" se usa solo para GENERAR las descripciones fenologicas de los
  prototipos (rol distinto). La seccion nueva `method_farslip.tex` ya usa la
  atribucion correcta (reasoner = Gemini 2.5-pro GA / Qwen 3.5-35B-A3B;
  generador de descripciones = Gemini Flash).
- **Decision para Arthur**: corregir "2.5-Flash" -> "2.5-pro" en el abstract y en
  el parrafo de soberania del manuscrito (afecta tambien al frontend/agente que
  gestiona otra US). Se deja como nota para no pisar el scope del agente.

### Datos REALES verificados (citados en la seccion, EXACTOS de los artefactos)
- `reports/farslip/metrics/us037_farslip_fiel_vs_alphaearth.csv`: FarSLIP fiel v2
  F1-macro 0.555056 +/- 0.020472, silhouette 0.012034, 768-dim, n=567;
  AlphaEarth F1-macro 0.644642 +/- 0.041515, silhouette 0.014459, 64-dim.
  Delta FarSLIP vs AlphaEarth = -0.0024 (no supera).
- `reports/farslip/metrics/faithful_v2_summary.csv`: faithful_v2 0.164/0.111;
  faithful_v2_cw 0.102/0.060 (class weights EMPEORAN); US-036-a 0.576 (4 clases).
- `reports/farslip/metrics/parcel_sweep.csv`: 4cls 0.7025 ... 12cls 0.3328.
- `reports/ensemble/us043_farslip_summary.json`: stacking-5 0.6477, stacking-3
  0.6359, delta 5-vs-3 = +0.01183.
- `reports/segmentation/sen4agrinet_transfer_result.json`: transfer FR->Catalonia
  zero-shot mIoU 0.0, few-shot 0.2468, F1 0.3005, pixel-acc 0.9179 (contexto US-073).

---

## US-068 -- Benchmark AgroMind-IT/ES (500 pares bilingues)

Entregado autonomo en esta sesion (codigo + esquema + tests, sobre fixture):

- `ml/eval/agromind_it_es/schema.py` -- `QAItem`, enum `QuestionFamily` (10
  familias), `to_agromind_item` (compat AgroMind verificable via import del
  `AgroMindItem` real de `agent_bench.py`), `dump_jsonl`/`load_jsonl` con guard
  eval-only.
- `ml/eval/agromind_it_es/generate_seed.py` -- generador seed con prompts por
  familia x idioma, `SeedGenerator` que lee el modelo de `get_settings()`
  (`gemini_model`, spec `gemini-2.5-pro`), modo dry-run sin API.
- `ml/eval/agromind_it_es/review_app.py` -- app Streamlit de revision humana
  (aceptar / editar / rechazar, log reviewer + idioma, export del split aceptado).
- `ml/eval/agromind_it_es/zenodo_metadata.py` -- builder de metadata Zenodo
  (CC-BY-4.0, descripcion eval-only); SIN llamada de upload.
- `data/benchmark/agromind_it_es/README.md` + `seed.fixture.jsonl` (3 pares de
  ejemplo de estructura, `source=fixture`; NO es el benchmark).
- `tests/ml/eval/test_agromind_it_es_schema.py` -- esquema, 10 familias,
  round-trip, compat `AgroMindItem`, guard eval-only, dry-run, metadata Zenodo.
- Settings nueva `zenodo_token` declarada en `backend/app/core/config.py`
  (`extra=forbid` no rompe).

### B-068-1 -- Generacion real del seed (500 pares): Gemini 2.5-pro + imagenes S2 Italia
- **Que falta**: (a) auth GEE / ADC para descargar imagenes **Sentinel-2 reales
  de Italia** (mismo blocker que el resto de jobs GEE; `gcsfs` no usa la auth del
  gcloud CLI, MEMORY `vm-h100-dvc-pull-401-no-adc`); (b) key **Gemini 2.5-pro**
  activa -- verificar leyendo `.env.local` Y probando la API real, NUNCA asumir
  por el shell (MEMORY `env-local-keys-verify-not-shell`, fue el error de US-033).
- **Estado (datos reales)**: codigo listo y probado en dry-run con cliente Gemini
  mockeado. Ningun par "de relleno" entra al dataset: el dry-run marca
  `source=dry-run`; el seed real marcaria `source=gemini-seed` (borrador
  pre-revision). Sin credenciales el generador corre en dry-run (emite el plan,
  no llama a la API) -- modo autonomo entregado.
- **Como completar**:
  ```bash
  # service account GEE (ADC) + GEMINI_API_KEY en .env.local, luego:
  python -m ml.eval.agromind_it_es.generate_seed \
      --image-root data/s2_italia --n-per-family 25 --languages it es \
      --out data/benchmark/agromind_it_es/seed.jsonl
  ```

### B-068-2 -- Revision humana nativa (italiano + espanol)
- **Que falta**: revisor **italiano** de Scuola Superiore Sant'Anna (via sponsor)
  para los 250 pares `it`, y **miembro del equipo** hablante de espanol para los
  250 `es`. Solo los pares aceptados/editados por humano nativo
  (`source=human-edited`) entran al benchmark publicado.
- **Estado**: la app `review_app.py` esta lista (importable, smoke test verde).
  Correrla con los reviewers reales es lo bloqueado.
- **Como completar**:
  ```bash
  streamlit run ml/eval/agromind_it_es/review_app.py
  # cada reviewer acepta/edita/rechaza; export del split aceptado a
  # data/benchmark/agromind_it_es/agromind_it_es_500.jsonl
  ```

### B-068-3 -- Upload a Zenodo + DOI
- **Que falta**: cuenta / token **Zenodo** del sponsor (`ZENODO_TOKEN` en
  `.env.local`, ya declarado en `Settings`). El builder de metadata
  (`zenodo_metadata.py`) NO incluye la llamada HTTP de upload a proposito.
- **Estado**: la metadata (`build_zenodo_metadata`, CC-BY-4.0, eval-only) esta
  lista y probada (estructura valida). Falta subir el `.jsonl` + `.zenodo.json` y
  obtener el DOI.
- **Como completar**: generar `.zenodo.json` con `write_zenodo_metadata`, subir
  `agromind_it_es_500.jsonl` + `.zenodo.json` a Zenodo con el token, tomar el DOI
  y anclarlo en el README del dataset y en el paper (US-070/071).

### Nota leakage (garantia de diseno, no es blocker)
El benchmark es **eval-only**: el esquema no tiene campo `split` y
`validate_record` rechaza cualquier registro con `split=train`/`training` o
`is_train=true`. Fine-tunear sobre AgroMind-IT/ES seria fuga de datos, igual que
con el AgroMind original (~28,482 Q&A sin train split).

---

## US-069 -- Harness `ml/eval/paper_bench.py` (eval comparativa multi-benchmark)

Entregado autonomo en esta sesion (runner + tests, backends mockeados, cero red).
El modulo es **hermano** de `ml/eval/agent_bench.py` (NO lo muta): reusa
`ReasonerVariant`, `_aggregate`, `_resolve_backend`, `_run_backend_text`,
`_save_checkpoint`, `eval_agromind`, `load_agromind_subset`, y las metricas puras de
`ml/eval/agent_metrics.py` (`exact_match`, `f1_squad`, `bertscore_f1`,
`hallucination_rate`). Anade: `load_geobench2` + `load_agromind_itses`,
`eval_geobench2` + `eval_agromind_itses`, `wilcoxon_paired` (scipy signed-rank
pareado), `macro_f1`, helpers de latencia p50/p95 y costo/query, `run_paper_benchmark`
(3 benchmarks x variantes x 3 seeds, mean+-std), export LaTeX a
`paper/tables/us-069/benchmark_comparison.tex`, logging MLflow
(`code_version`+`data_version`, experimento `us069_paper_bench`, server Docker :5010)
y CLI `python -m ml.eval.paper_bench`. Tests en `tests/ml/eval/test_paper_bench.py`
con fixtures de FORMA (`data/test_fixtures/geobench2_mini/manifest.json` +
`data/test_fixtures/agromind_itses_mini.jsonl`, cero dato cientifico real).

**Variantes (AC literal):** SOLO dos reasoners frozen (patron Be My Eyes,
arXiv:2511.19417): **Gemini 2.5-pro** (nube, GA, 1M ctx, NO 2M, NO "3.1") y
**Qwen3.5-35B-A3B vLLM** (on-prem, GPTQ-Int4, single-GPU). Gemma 4 26B base-only es
OUT del par titular (su LoRA esta OUT, ADR-009); se anade por CLI desde
`agent_bench.DEFAULT_VARIANTS` si se desea una columna extra.

**Targets de referencia (NO sobre-afirmar):** AgroMind >= 0.75 Gemini / >= 0.70 Qwen.
NO se afirma "VLM fine-tuned supera a Gemini" (la capa LLM = comunicacion/explicacion
+ soberania de datos, no clasificador de pixeles).

### B-069-1 -- Ejecucion Qwen3.5-35B-A3B (US-048 + ventana H100)
- **Que falta**: serving vLLM de US-048 vivo en H100 (`:8002/v1`, GPTQ-Int4
  single-GPU, sin `--tensor-parallel-size`) + ventana H100.
- **Estado**: el runner corre con `gemini` solo y deja `qwen` para la ventana; los
  backends son inyectables -> tests verdes sin endpoint. Qwen texto SKIP de items
  que exigen imagen (reporta `n_skipped`, nunca scorea como si viera el tile).
- **Como completar**: levantar vLLM (skill `agrosat-llm-finetuning`), setear
  `settings.vllm_qwen35_url` (o `VLLM_QWEN35_URL`), correr
  `python -m ml.eval.paper_bench --variants gemini qwen`.

### B-069-2 -- AgroMind-IT/ES (depende de US-068)
- **Que falta**: el JSONL bilingue de US-068, tras revision humana nativa (it por
  Sant'Anna via sponsor, es por el equipo) + Zenodo. US-068 lo exporta hoy a
  `data/benchmark/agromind_it_es/agromind_it_es_500.jsonl` (ver B-068-2); el default
  de `paper_bench` es `data/agromind_itses/agromind_itses_500.jsonl`.
- **Decision menor de ruta**: al cerrar US-068 unificar la ruta -- o copiar/symlink el
  JSONL al default de `paper_bench`, o pasar `--itses-path data/benchmark/agromind_it_es/agromind_it_es_500.jsonl`.
  El schema que `load_agromind_itses` espera (`item_id, lang(it/es), question,
  options?, answer, image_path?, family`) es compatible con el `QAItem` de US-068
  (mismo shape AgroMind); si difiere algun nombre de campo, ajustar el loader (el
  test `test_load_agromind_itses_reads_both_languages` fija el contrato).
- **Estado**: sin el archivo, el benchmark se marca `pending` (no se inventa).

### B-069-3 -- GEO-Bench-2 subset agricola (descarga + DVC)
- **Que falta**: NO esta en `data/`. GEO-Bench-2 = benchmark de vision EO de
  ServiceNow (sucesor de GEO-Bench 2023, pip `geo-bench`), **distinto** de
  GeoAnalystBench. Descarga + inferencia de vision exige GPU/cuota.
- **Como completar**: descargar via el loader oficial `geobench`, filtrar >=3 tasks
  agricolas (crop-type / land-cover con clases de cultivo), materializar
  `data/geobench2/manifest.json` con shape
  `{tasks:[{id,name,modality,label_space,split,items:[{item_id,image_path,gold_label,question?}]}]}`
  + los tiles, `dvc add data/geobench2`. `load_geobench2` ya lee ese manifest; el
  fixture mini documenta la forma exacta. La variante multimodal (Gemini) clasifica
  el tile; la text-only (Qwen) hace SKIP de los que exigen imagen.
- **Nota dep**: el AC menciona `poetry add geobench`. NO se agrego en esta sesion
  para no introducir una dep pesada (tree-sitter/torch-vision transitivos) sin
  correrla; el loader trabaja sobre el manifest JSON materializado, no sobre el
  paquete, asi que los tests no lo necesitan. Agregar `geobench` al hacer la descarga
  real (`poetry add geobench`), no antes.

### B-069-4 -- Subset AgroMind 1000 (hoy 500)
- **Que falta**: el AC pide 1000; hoy hay 500 en
  `data/agromind/agromind_subset_500.json`.
- **Como completar**: re-correr el sampler estratificado a n=1000 sobre el corpus
  AgroMind (~28k). `load_agromind_subset` es agnostico al tamano. Si no se amplia, se
  corre con 500 y se documenta el delta vs AC. AgroMind es **eval-only** (sin train
  split): fine-tunear sobre el = LEAKAGE.

### B-069-5 -- Costo Gemini real (3 corridas x 3 benchmarks)
- **Que falta**: gasto cloud no trivial; correrlo completo se hace en la ventana
  buffer con presupuesto acordado y API Gemini configurada (`settings.gemini_api_key`).
- **Mitigacion en el runner**: `--checkpoint` + `--resume` evitan re-pagar variantes
  ya hechas; Gemini se evalua PRIMERO (orden del runner) para no recomputarlo si una
  variante on-prem falla.

### Reproduccion (cuando los bloqueos esten resueltos)
```bash
poetry run python -m ml.eval.paper_bench \
  --variants gemini qwen \
  --seeds 0 1 2 \
  --geobench-root data/geobench2 \
  --geobench-tasks m-crop-type m-land-cover m-eurocrops \
  --agromind-path data/agromind/agromind_subset_500.json \
  --itses-path data/agromind_itses/agromind_itses_500.jsonl \
  --checkpoint reports/paper_bench/checkpoint.json --resume \
  --out-latex paper/tables/us-069/benchmark_comparison.tex
```
Sale: tabla LaTeX booktabs poblada (mean+-std + p-value Wilcoxon en el caption),
metricas en MLflow `us069_paper_bench` (tags `code_version`+`data_version`), resumen
JSON a stdout. Gotcha MLflow (`ml/CLAUDE.md`): el lineage vive en el server Docker
`:5010`, no en `./mlruns`; cerrar runs de subprocess que queden `RUNNING`.

### Atribuciones obligatorias (en el caption LaTeX + comentarios del modulo)
- AlphaEarth = `SATELLITE_EMBEDDING/V1/ANNUAL`, data **v1.1**, 64-dim, CC-BY-4.0 (NUNCA "v2.1").
- Gemini 2.5-pro: GA, **1M ctx** (NO 2M, NO "3.1").
- AgroMind y AgroMind-IT/ES: **eval-only** (sin re-entrenamiento).
- Qwen3.5-35B-A3B: vLLM **GPTQ-Int4 single-GPU** (variante on-prem, soberania de datos).

---

## US-070 -- Figuras y tablas reproducibles del paper

Entregado autonomo en esta sesion (datos reales, sin hardcode):
- `ml/report/paper_figures.py` -- estilo cientifico CVPR/ISPRS (rcParams serif,
  300 DPI, seed `PAPER_SEED=17`) + exportador `save_fig_svg_png` (SVG vector +
  PNG 300 DPI) + figuras recompuestas desde CSV/JSON o promovidas desde PNG
  existentes de `reports/**`. Ninguna figura se fabrica: fuente ausente ->
  retorna `None` y se loguea.
- `ml/report/paper_tables.py` -- 6 tablas `.tex` (`booktabs`, espejo de
  `paper/tables/us-023-preview/baseline_v2_comparison.tex`) leidas de CSV/JSON
  reales; cero literales numericos (contrato anti-hardcode: el numero del `.tex`
  == numero del artefacto, verificado por test).
- Salidas reales generadas: `paper/figures/us-070/*.{svg,png}` (10 figuras +
  `conversational_examples.json`) y `paper/tables/us-070/*.tex` (6 tablas).
- Tests: `tests/ml/report/test_paper_figures.py` + `test_paper_tables.py`
  (22 verdes; fixtures con cifras-centinela inexistentes en el repo prueban que
  se lee del artefacto y no se hardcodea).
- Targets `make paper-figures` (papermill end-to-end de los 4 notebooks) y
  `make paper-tables` (regenera los `.tex`).

Figuras/tablas AUTONOMAS ya en disco: `benchmark_barplot_fold5`,
`farslip_sweep_curve`, `transfer_fr_catalonia`, `llm_benchmark_barplot`
(Gemini+Qwen reales), `umap_alphaearth`, `curves_tsvit`, `confusion_tsvit`,
`confusion_stacking`, `spatial_residuals`, `per_class_iou_tsvit`; tablas T1
`fm_comparison`, T2 `segmentation_individual_fold5`, T3 `ensembles_e6`, T4
`llm_benchmark`, T5 `tool_ablation`, Tx `farslip_band_ablation`.

### B-070-1 -- F2 Mapas AOI Italia (requiere auth GEE)
- **Que falta**: render geoespacial de AOIs reales de Italia sobre AlphaEarth
  (`SATELLITE_EMBEDDING/V1/ANNUAL` v1.1) + basemap/tiles. Sin auth GEE ni
  service-account aqui, y sin AOI GeoJSON de Italia materializado.
- **Como completar**: autenticar GEE (skill `agrosat-gee-alphaearth`), exportar
  la AOI a GeoJSON, anadir celda en `paper/notebooks/03_figures_embeddings_fm.ipynb`
  que la lea y la pinte con `contextily`/`folium`. No se fabrica un mapa sin AOI real.

### B-070-2 -- F4 curvas full-config H100 (requiere job GPU)
- **Que falta**: curvas TSViT/TSViT-pheno "full-config H100" (target Full-M mIoU
  0.68-0.72): es un re-run de entrenamiento en H100.
- **Estado (datos reales)**: `curves_tsvit` se promueve desde la curva real
  existente `reports/segmentation/figures/curves_tsvit.png` (fold-4/fold-5). El
  delta full-M esta en `reports/segmentation/metrics/tsvit_pheno_vs_base_fold5.csv`
  (tsvit-base-fullm 0.6789, tsvit-pheno-fullm 0.6756; rama fenologica ~0 en
  supervisado, valido). No se inventa la curva full-config.
- **Como completar**: correr el full-M en H100 y exportar la curva real con
  `ml/eval/avance4_figures.py:curves_from_mlflow`, re-promover.

### B-070-3 -- T4/F7-LLM benchmark completo (depende de US-068/US-069 + H100)
- **Que falta**: columnas AgroMind-IT/ES (US-068, revision nativa) + 3 corridas
  con error bars y Wilcoxon (US-069) + corrida on-prem Qwen definitiva con H100.
  Detalle de reproduccion en B-069-1..5 mas arriba.
- **Estado (datos reales)**: `reports/agent_bench/us049_system_eval.json` ya tiene
  bloques reales de Gemini Y Qwen; T4 y F7-LLM se generan con AMBOS modelos. La
  columna AgroMind-IT/ES se marca `\textit{pendiente}` (nunca fabricada). std=0 en
  la corrida unica -> error bars pendientes de las 3 repeticiones (US-069).
- **Como completar**: ejecutar US-068 + US-069 (ver bloqueos arriba), regenerar el
  JSON y re-correr `make paper-tables paper-figures`.

### B-070-4 -- F5 ejemplos conversacionales IT (depende de US-068)
- **Que falta**: trace conversacional en italiano. Los traces reales
  (`reports/agent_bench/traces/trace_gemini_*.jsonl`) cubren ES/EN.
- **Estado (datos reales)**: `export_conversational_examples` extrae ejemplos
  ES/EN reales a `paper/figures/us-070/conversational_examples.json` con
  `note_it: "pendiente AgroMind-IT US-068"`. No se fabrica trace IT sintetico.
- **Como completar**: generar traces IT con AgroMind-IT (US-068) y re-extraer.

### B-070-5 -- Fx/Tx ablacion bandas FarSLIP 3 variantes (requiere re-extraccion H100)
- **Que falta**: ablacion de las 3 variantes de banda (rgb vs nir-rgb falso-color
  vs 4band-pheno): re-extraer embeddings FarSLIP por variante
  (`farslip-extract-embeddings`), job de GPU. Coincide con B-072-1 (log truncado).
- **Estado (datos reales)**: Fx/Tx se generan con la evidencia real disponible:
  `reports/farslip/metrics/parcel_sweep.csv` (barrido de cardinalidad) y
  `us037_farslip_fiel_vs_alphaearth.csv` (FarSLIP fiel 0.5551 vs AlphaEarth 0.6446).
  Las 3 variantes de banda completas no estan materializadas.
- **Como completar**: `make farslip-extract-embeddings` por variante en H100,
  escribir metricas a `reports/farslip/metrics/`, extender
  `build_farslip_band_ablation_table` con esas filas.

### Nota F1 arquitectura
- F1 (diagrama de arquitectura) no requiere computo ni dato; se deriva de
  `paper/figures/paper-methods/` + docs. Pendiente de dibujo final (draw.io ->
  SVG); no es blocker de datos.

---

## US-073 -- Seccion de transferencia multi-region (Sen4AgriNet denso + EuroCropsML few-shot)

Entregado autonomo en esta sesion (datos REALES de E12, sin hardcode, sin sinteticos):

- `paper/sections/experiments_multiregion.tex` -- seccion `\input`-eable (prosa EN,
  recipe train-FR -> extend-elsewhere): label-space HCAT v3 (11 macro-clases),
  camino denso Sen4AgriNet Catalonia (tabla domain-gap), camino few-shot EuroCropsML
  (tabla + figura curva k-shot), demo Mexico cualitativa (sin F1), parrafo FUTURE
  WorldCereal/HGC, caveat arXiv:2601.00857. SIN `\documentclass` (lo ensambla US-071).
- `scripts/build_us073_transfer_figures.py` -- genera DESDE artefactos reales (cero
  numeros a mano): 2 tablas `.tex` (`booktabs`) + 2 figuras (`.png`+`.svg`). Reusa
  `ml.transfer.eurocropsml_fewshot.summarize_curve` (DRY). Determinista, idempotente,
  `structlog`, type hints, sin emojis. Target `make us073-transfer-figures`.
- Salidas reales en disco: `paper/tables/us-073-transfer/{sen4agrinet_domain_gap,
  eurocropsml_kshot}.tex` + `paper/figures/us-073-transfer/{kshot_curve,
  mexico_phenology}.{png,svg}`.
- Tests: `tests/scripts/test_build_us073_transfer_figures.py` (6 verdes; la tabla
  densa == JSON real 0.0000/0.2468/0.3005/0.9179, la tabla k-shot == mean/std real del
  parquet, regresion "no v2.1", protocolo LV[+PT]->EE no "France->Estonia", idempotencia,
  figuras emitidas). `ruff` + `mypy` limpios.
- `DATA_LICENSE.md`: las 3 atribuciones (Sen4AgriNet CC-BY-SA-4.0, EuroCropsML
  CC-BY-SA-4.0, AlphaEarth V1 CC-BY-4.0) + la tabla por-region YA estaban (US-066);
  NO se duplico nada (verificado antes de editar).

### Datos REALES verificados (citados/renderizados, EXACTOS de los artefactos)
- Denso FR->Catalonia (`reports/segmentation/sen4agrinet_transfer_result.json`):
  zero-shot mIoU 0.0000 / F1 0.0000 / pixel-acc 0.0000; few-shot mIoU 0.2468 /
  F1 0.3005 / pixel-acc 0.9179; Delta mIoU +0.2468; 40 epocas, 10 train / 20 val patches.
- k-shot EuroCropsML (`data/transfer/eurocropsml_fewshot_results.parquet`, 63 filas,
  8 clases, 3 seeds): LV->EE (pre-train) sube 0.321 (k=1) -> 0.482 (k=500); LV+PT->EE
  0.315 -> 0.420; sin-pretrain 0.015 (k=1) -> 0.472 (k=500). std de poblacion (ddof=0,
  el de `summarize_curve`).
- Mexico (`data/transfer/mexico_demo_ndvi.parquet`): 104 fechas aguacate (Uruapan) +
  60 guayaba (Calvillo), NDVI real GEE; AlphaEarth 64-dim real (`mexico_demo_alphaearth.parquet`).
  Cualitativo, SIN F1/accuracy (regla US-077).

### B-073-1 -- Figura UMAP FR/ES domain-gap (AC tarea 2): panel ES no materializado
- **Que falta**: el panel UMAP de Catalonia (ES) requiere embeddings AlphaEarth (o
  features S2) de los parches Sen4AgriNet en un array materializado para proyectar; el
  panel FR ya existe (`paper/figures/us-011/sec2_umap_francia_pastis.png`).
- **Decision tomada (honesta)**: en vez de un UMAP ES a medias, la figura del domain-gap
  denso se sustituyo por la **tabla** `sen4agrinet_domain_gap.tex` (zero 0.0000 -> few
  0.2468), que es la evidencia cuantitativa exacta del gap. El UMAP FR/ES queda
  pendiente como figura ilustrativa adicional, no como evidencia (la tabla ya la da).
- **Como completar**: materializar el embedding AlphaEarth de los ~30 parches ES del
  subset Sen4AgriNet (DVC `data/sen4agrinet.dvc`) -- requiere GEE auth/ADC (mismo blocker
  GEE recurrente, MEMORY `vm-h100-dvc-pull-401-no-adc`) -- proyectar FR+ES juntos con
  `umap-learn` y anadir el panel. Reusar `ml/report/paper_figures.py:fig_umap_*` (US-070).

### B-073-2 -- Curvas NDVI desfasadas FR vs ES (AC tarea 2, parte del domain-gap)
- **Que falta**: comparativa fenologica NDVI FR<->ES (siembra/cosecha desfasada por
  latitud) como segundo panel del domain-gap.
- **Causa**: necesita las series S2 temporales de AMBAS regiones alineadas por clase;
  PASTIS FR esta en DVC, las series ES del subset Sen4AgriNet hay que extraerlas.
- **Como completar**: cargar series S2 del subset Sen4AgriNet (DVC) + PASTIS FR, calcular
  NDVI medio por macro-clase y fecha con el codigo de fenologia existente
  (`ml/features/`), plotear el desfase. CPU, sin GPU, pero depende de tener las series ES
  materializadas (ligado a B-073-1).

### B-073-3 -- Re-ejecucion del finetune denso (NO requerido para la seccion)
- **Que falta**: nada para la seccion. El run vive en H100
  (`F:/projects/.../tsvit-pheno-sen4agri-cat-ft-v1/best.pt`); MLflow `:5010` estaba DOWN
  en la VM (fallback `file:./mlruns`).
- **Estado**: el JSON de resultados es la fuente de verdad y el notebook
  `notebooks/segmentation/5c_transfer_sen4agrinet.ipynb` ya tiene outputs poblados. NO se
  re-corre. Solo si se necesitan metricas nuevas: H100 + `dvc pull`.

### B-073-4 -- Revision humana del texto EN (calidad de manuscrito)
- **Que falta**: revision de ingles del borrador (US-071 preve Grammarly + revision
  Dr. Camacho antes del submission).
- **Estado**: borrador EN listo y autocontenido (`\input`-eable). Pasa por la revision de
  US-071.

### B-073-5 -- WorldCereal / HGC tropical: explicitamente FUTURE
- **Que falta**: nada en esta ventana (declarado FUTURE en el AC). La seccion tiene el
  parrafo "Future / scale" sin metricas; atribuciones pre-registradas en DATA_LICENSE.md.
- **Como completar**: fuera de scope de la ventana buffer; ingesta WorldCereal RDM /
  Harmonized Global Crops post-paper.

### B-073-6 -- Integracion en el manuscrito: RESUELTA (US-071 ya la ensamblo)
- **Estado**: HECHA. US-071 (sibling) ya hace `\input{sections/experiments_multiregion}`
  en `paper/main.tex` (linea 97, con alias `\label{sec:farslip_transfer}` que US-072
  forward-referencia a `sec:multiregion`). El preambulo ya trae `booktabs`, `graphicx`,
  `amsmath` y un `\graphicspath{{figures/}...}` que resuelve mis rutas
  `figures/us-073-transfer/*` y `tables/us-073-transfer/*`.
- **Verificado en este entorno**: `make paper-cite-check` verde (20/20 keys resuelven,
  incluidos mis 9: brown2025alphaearth, garnot2021pastis, google2025gemini25,
  harvesting2026alphaearth, huang2025bemyeyes, reuss2025eurocropsml, sykas2022sen4agrinet,
  tarasiou2023tsvit, yang2025qwen3). `make paper-pdf` compila `paper/main.pdf` (15 pag, 0
  refs/citas indefinidas); mi seccion es la Section 7 (subsec 7.1-7.6), Tablas 8 y 9,
  Figura 1, todas renderizadas (verificado en `main.aux`/`main.log`).
- **Convertido a `\cite{}`**: las citas en prosa de mi seccion se ataron a las entradas
  reales de `paper/bib/refs.bib` (que US-071 ya poblo). NO queda nada pendiente de
  integracion. (La seccion sigue siendo `\input`-eable y autocontenida; NO se toco
  `docs/final_doc/Avance7_*.tex`, que es el doc del curso.)

---

## US-071 -- Manuscrito LaTeX modular (estructura `paper/` + `bib/refs.bib` + `main.tex`)

Estado: manuscrito estructurado y COMPILADO en local (15 paginas, 0 referencias
indefinidas, todas las figuras resueltas). Lo autonomo se entrego; los pasos de
submission y los numeros que dependen de H100 quedan aqui.

### Lo que SI se completo (autonomo, en este entorno)
- `paper/main.tex` (preambulo PRIMEarxiv + `\input{}` de 9 secciones nucleares +
  `\subimport{sections/}{method_farslip}` de US-072 + `\input` de
  `experiments_multiregion` de US-073 + `\bibliography{bib/refs}`).
- `paper/sections/00_abstract.tex` .. `08_appendix.tex` (prosa en ingles derivada del
  Avance7 EN + Related Work nuevo). Abstract <=250 palabras, corregido.
- `paper/bib/refs.bib` (20 entradas BibTeX con ids arXiv reales). `make paper-cite-check`
  verde: cada `\cite{}` resuelve.
- `paper/PRIMEarxiv.sty` (copia local para build autonomo), `make paper-pdf` /
  `make paper-pdf-docker` / `make paper-pdf-clean` / `make paper-cite-check` y
  `infrastructure/docker/paper-latex.Dockerfile` (texlive + bibtex).
- Correcciones factuales aplicadas: Gemini 2.5-pro (no Flash), FarSLIP
  `arXiv:2511.14901`, AlphaEarth `SATELLITE_EMBEDDING/V1/ANNUAL` v1.1 CC-BY-4.0
  (no "v2.1"), sin Swin-UNETR entrenado (AnySat lo sustituye), SegFormer B0 RGB,
  sin Gemma 4 LoRA, Be My Eyes (reasoner frozen). AgroMind eval-only declarado.
- Cada cifra del Results lleva `% src: <artefacto real>`. Cifras EXACTAS:
  TSViT-pheno fold-4 mIoU 0.6253 / F1 0.7500 (fold-5 0.6139 / 0.7401);
  Stacking-5 OOF f1_macro 0.6477 (delta 5-vs-3 +0.0118); FarSLIP fiel 0.5551 vs
  AlphaEarth 0.6446; AnySat fold-4 0.4459; transfer FR->Catalonia zero-shot 0.0000,
  few-shot 0.2468 (F1 0.3005, pixel-acc 0.9179, delta +0.2468).

### Cierra blockers de US-072/US-073
- **B-073-6 RESUELTO**: `main.tex` hace `\input{sections/experiments_multiregion}` con
  `\graphicspath` a `paper/figures/` + reports, y paquetes booktabs/graphicx/amsmath en
  el preambulo. Las 4 referencias del texto (Harvesting AlphaEarth, Be My Eyes,
  EuroCropsML, Sen4AgriNet) estan en `refs.bib`.
- **B-072-3 RESUELTO (opcion a)**: la seccion FarSLIP vive modular en
  `paper/sections/method_farslip.tex` e integrada al `main.tex` del Paper Track (NO en
  `docs/final_doc`, que es el doc del curso). `main.tex` la trae via `\subimport`
  (sus rutas `../tables/...` son relativas a `sections/`).
- **B-072-6 RESUELTO en el manuscrito del paper**: el abstract de `paper/main.tex` usa
  Gemini 2.5-pro / Qwen on-prem (no Flash). El `docs/final_doc/Avance7_*.tex` (curso)
  conserva su texto; corregirlo alli sigue siendo su propio pendiente.

### B-LATEX-LOCAL -- Toolchain LaTeX (RESUELTO en este entorno)
- **Estado**: MiKTeX presente (`pdflatex`, `bibtex`, `latexmk` en PATH). `make paper-pdf`
  genera `paper/main.pdf` (15 pp). El PDF es artefacto regenerable (gitignored), NO se
  commitea. En entornos sin LaTeX usar `make paper-pdf-docker` (solo Docker).

### B-MDPI-TEMPLATE -- Template Remote Sensing MDPI oficial
- **Que falta**: `mdpi.cls` del kit MDPI. No se versiona una `.cls` de terceros sin
  acuerdo. Hasta entonces el build usa `PRIMEarxiv.sty`. Migrar a MDPI es paso de
  submission (mover preambulo + secciones al `mdpi.cls`, sin tocar el cuerpo).

### B-OVERLEAF -- Proyecto Overleaf
- **Que falta**: subir `paper/` a Overleaf (cuenta del equipo). El build local +
  `make paper-pdf-docker` cubren la reproducibilidad; Overleaf es para coedicion humana.

### B-GRAMMARLY -- Revision ortografica/gramatical EN
- **Que falta**: pasada Grammarly sobre la prosa inglesa. Paso humano post-redaccion.

### B-CAMACHO -- Revision por Dr. Camacho antes de submission
- **Que falta**: revision academica del sponsor. Humano.

### B-ARXIV -- Submission a arXiv cs.CV
- **Que falta**: cuenta arXiv + endorsement cs.CV. Humano. (Garantiza prioridad temporal.)

### B-VENUE -- Submission a venue
- **Que falta**: cuenta + deadlines. Orden de prioridad: Remote Sensing MDPI (rolling),
  CVPR EarthVision Workshop 2026 (si el deadline lo permite), ISPRS Journal. Humano.

### B-LLM-BENCH-NUMS -- Cifras de benchmark LLM (Tabla US-069)
- **Que falta**: las celdas de `Table~\ref{tab:llm-bench}` (accuracy / grounding /
  LLM-judge para Gemini 2.5-pro vs Qwen on-prem). Dependen de la eval en H100 (US-069,
  ver B-069-1..5). La tabla queda con estructura de columnas y `% src: pendiente US-069`.
  NO se inventan numeros (regla de datos reales). `paper/tables/us-069/benchmark_comparison.tex`
  (US-070) es el destino data-driven cuando existan.
