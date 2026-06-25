# Papers adicionales — propuesta estrategica de publicacion

**Proyecto**: AgroSatCopilot (Equipo 17, MNA Tec de Monterrey)
**Autores propuestos**: Avila, Bocanegra, Zizumbo
**Fecha**: 2026-06-24
**Estado**: borrador estrategico (NO commit, NO sometido)

## Contexto y premisa

El **paper principal** (`paper/main.tex` + `paper/sections/`) ya cubre el sistema
end-to-end: stack de percepcion (segmentacion SITS + tabular AlphaEarth + rama FarSLIP),
ensamble de stacking heterogeneo, capa conversacional Be My Eyes, transferencia
multi-region y FinOps. Es un paper de *sistema*. Por su amplitud, varios hallazgos
quedan comprimidos a una sub-seccion cada uno y no se profundizan.

Esta nota propone **3 papers adicionales**, cada uno con un angulo que el principal no
desarrolla a fondo, **reutilizando artefactos reales del repo** (rutas verificadas). Para
cada propuesta: titulo, contribucion, evidencia disponible, lo que **falta**, venue y
esfuerzo. La regla de oro del proyecto se mantiene: **cero cifras inventadas**; donde un
experimento no existe, se dice explicitamente.

### Cifras canonicas que TODOS los papers deben respetar

| Concepto | Valor | Fuente verificada |
|---|---|---|
| Modelo final (Stacking-5 +FarSLIP heterogeneo, held-out fold-5, 18 clases) | F1-macro **0.7470**, acc **0.8490** | `reports/ensemble/metrics/ec_neighborhood_result.json` (champion); `comparison_us040.csv` |
| Mejor individual (TSViT-pheno, held-out fold-4) | mIoU 0.6253 / F1 0.7500 | `paper/sections/05_results.tex` |
| Ganancia ensamble vs mejor individual | **+12.3 pp** F1-macro | `paper/sections/05_results.tex` |
| Stacking-5 OOF-CV (libre de fuga) | F1-macro 0.6477 / acc 0.7935 | `reports/ensemble/us043_farslip_summary.json` |
| Delta 5 vs 3 miembros (aporte FarSLIP, stacking OOF) | +0.0118 F1-macro | `reports/ensemble/us043_farslip_summary.json` |
| FarSLIP fiel v2 vs AlphaEarth (probe 567 parcelas) | 0.5551 < 0.6446 (-0.0024) | `reports/farslip/metrics/us037_farslip_fiel_vs_alphaearth.csv` |
| Re-cableo perceiver al campeon (13 481 parcelas, france-9) | acc 0.8308 -> **0.9413** (+11.0 pp); F1 0.6868 -> 0.9007 (+21.4 pp) | `reports/agent_bench/perceiver_champion_eval.json` |
| Transfer FR->Catalonia (Sen4AgriNet) | zero-shot mIoU 0.0 -> few-shot 0.2468 (+0.2468) | `reports/segmentation/sen4agrinet_transfer_result.json` |
| E-c vecindad estructural (france-9) | delta +0.0002 = ruido (NO material) | `reports/ensemble/metrics/ec_neighborhood_result.json` |
| LLM reasoner | Gemini 2.5 Pro (NO Flash) + Qwen 3.5-35B-A3B on-prem; Be My Eyes (reasoner frozen) | `paper/CLAUDE.md`, `paper/sections/00_abstract.tex` |

**Atribuciones obligatorias**: AlphaEarth (Khanna et al., GEE `SATELLITE_EMBEDDING/V1/ANNUAL`
v1.1, CC-BY-4.0), FarSLIP (arXiv:2511.14901), Be My Eyes (arXiv:2511.19417), Sen4AgriNet /
EuroCropsML (CC-BY-SA-4.0), PASTIS-R (Garnot, ICCV 2021), TSViT / U-TAE / SegFormer / AnySat.

---

## Paper A — Metodologico: cardinalidad, calidad-cobertura y heterogeneidad

### (1) Titulo tentativo
*"How many crops should you promise? A quality-coverage trade-off and the case for
heterogeneous over homogeneous ensembles in parcel-level crop mapping"*

(alt: *"Cardinality-aware crop classification: choosing the deployed label set, and why
spatial context is already in the embedding"*)

### (2) Angulo / contribucion principal
Un paper de **decisiones de diseno ML**, no de sistema. Tres hallazgos publicables que en
el principal son una linea cada uno:

1. **Trade-off calidad-cobertura por cardinalidad**: cuantas clases comprometer en
   produccion es una decision medible, no arbitraria. La curva de *dropout honesto*
   muestra que ir de 18 a 9 clases mas cardinales sube F1-macro de **0.7486 a 0.9121**
   reteniendo el ~82% de parcelas; 12 clases dan **0.8573** (90% cobertura, acc 0.8677);
   8 clases superan **0.92** (80% cobertura). Es una frontera de Pareto operativa.
2. **Heterogeneidad gana sobre homogeneidad**: el meta-modelo de stacking actua como
   *arbitro por clase*, no como promediador. El ensamble heterogeneo de 5 miembros
   decorrelacionados produce **+12.3 pp** de F1-macro sobre el mejor individual — una
   ganancia macro desproporcionada a la de accuracy, justo lo que distingue arbitraje de
   averaging. (El paper principal afirma esto; aqui se *demuestra* con el ablation
   homogeneo-vs-heterogeneo y el barrido de miembros.)
3. **Resultado NEGATIVO de E-c (vecindad espacial)**: anadir un eje estructural de
   vecindad k-NN al campeon **NO mejora de forma material** (france-9 delta=+0.0002, bajo
   el umbral de ruido 0.01 sobre 16 640 parcelas). Interpretacion: el contexto espacial
   **ya esta absorbido por el embedding AlphaEarth** (consistente con ADR-010). Un
   resultado negativo limpio y bien enmarcado es publicable y honesto.

Hilo conductor: **el embedding de fundacion ya codifica fenologia y contexto; lo que falta
es arbitraje entre perceivers decorrelacionados y una eleccion racional del conjunto de
clases**.

### (3) Resultados / figuras del proyecto que usa (rutas reales)
- `reports/ensemble/metrics/us043_honest_dropout_curve.csv` — curva cardinalidad
  18->8 (F1, acc, n_parcelas por k). **Nucleo del hallazgo 1.**
- `reports/farslip/metrics/parcel_sweep.csv` + `paper/tables/farslip-method/cardinality_sweep.tex`
  — barrido complementario FarSLIP-pheno 4->12 (otro angulo de la misma tesis).
- `reports/ensemble/us043_farslip_summary.json` — stacking 5 vs 3 miembros, blending 5 vs 3
  (evidencia de heterogeneidad; delta +0.0118).
- `paper/tables/us-070/ensembles_e6.tex`, `paper/tables/us-070/segmentation_individual_fold5.tex`
  — comparativa individual vs ensamble.
- `reports/ensemble/metrics/ec_neighborhood_result.json` — **resultado negativo E-c**
  completo (sweep k={5,10} x alpha + veredicto honesto). **Nucleo del hallazgo 3.**
- Figuras de `reports/ensemble/figures/us043_farslip/`.

### (4) Que FALTA para completarlo (honesto)
- **Ablation explicito homogeneo vs heterogeneo**: el repo tiene el ensamble heterogeneo y
  el barrido de miembros, pero un Voting *homogeneo top-3* contrastado lado a lado en la
  **misma cosecha de folds** no esta materializado como artefacto unico. Existe la receta
  (skill `agrosat-ml-ensemble`) pero hay que correr y guardar el CSV comparativo. **Esfuerzo
  bajo: re-uso de predicciones OOF ya guardadas.**
- **Intervalos de confianza / test estadistico** sobre la curva de cardinalidad (bootstrap
  por parcela) — falta; deseable para venue ML.
- **Frontera de Pareto graficada** (F1 vs cobertura) como figura unica — falta generarla
  (los datos existen en el CSV de dropout).
- El E-c y la curva de cardinalidad estan sobre **France/PASTIS-R unicamente**; el paper
  debe acotar que la tesis "contexto ya en el embedding" se valida en una sola region.
- **NO falta** ningun entrenamiento nuevo grande: es mayormente analisis + figuras sobre
  artefactos existentes.

### (5) Venue / target sugerido
- **Primario**: *Remote Sensing* (MDPI, IF ~5) o *ISPRS Journal of Photogrammetry and RS*
  (track metodologico). El angulo "cuantas clases prometer" es muy del gusto aplicado de RS.
- **Workshop alternativo**: NeurIPS/ICLR *Tackling Climate Change with ML* o CVPR
  *EarthVision* (el resultado negativo E-c encaja bien en venues que valoran honestidad).

### (6) Esfuerzo estimado
**Bajo-medio: ~2-3 semanas.** Los 3 hallazgos ya tienen artefactos. El trabajo es: (a)
correr el ablation homogeneo-vs-heterogeneo faltante (~2-3 dias), (b) bootstrap CIs (~2
dias), (c) escritura + 4-5 figuras nuevas. Es el **paper de menor riesgo** porque casi todo
ya esta medido y es el angulo mas autocontenido.

---

## Paper B — Transferencia multi-region de embeddings de fundacion EO

### (1) Titulo tentativo
*"Train in France, extend elsewhere: measuring the spatial transferability of frozen
Earth-observation foundation embeddings for crop mapping"*

(alt: *"The few-shot budget of foundation-model crop classifiers: FR->Catalonia,
transnational k-shot, and a zero-shot domain gap you can see"*)

### (2) Angulo / contribucion principal
El principal dedica a esto la seccion `experiments_multiregion.tex`, pero la comprime. Aqui
**la transferibilidad ES el paper**. Contribucion:

1. **El delta zero-shot -> few-shot como entregable cientifico** (no la accuracy absoluta).
   FR->Catalonia (Sen4AgriNet): zero-shot mIoU **exactamente 0.0** (gap Franco-Iberico
   catastrofico, real, no bug) -> few-shot **0.2468** (+0.2468) con solo 10 parches.
2. **El costo few-shot transnacional cuantificado**: curva EuroCropsML k-shot
   (LV[+PT]->EE) real, 3 semillas, k in {1..500}. A k=1 el pre-entrenamiento fuente ya da
   ~0.32 F1 vs ~0.015 sin pre-train; las curvas convergen — *un punado de etiquetas
   locales, no magia zero-shot, cierra el gap*.
3. **El domain-gap hecho visible**: UMAP de embeddings AlphaEarth FR vs Catalonia
   (regiones parcialmente disjuntas) + desfase fenologico NDVI (~calendario corrido) que
   explica geometricamente por que el zero-shot falla. Esta es la **contribucion
   interpretativa** que el principal apenas menciona.
4. **Mexico zero-shot CUALITATIVO por diseno** (aguacate Uruapan, guayaba Calvillo): sin
   F1, con un meta-test que *prohibe* importar metricas de clasificacion en esa ruta.
   Honestidad metodologica como feature.

Tesis: corrobora empiricamente el limite reportado en "Harvesting AlphaEarth"
(`harvesting2026alphaearth`) — los embeddings globales frozen transfieren mal sin
adaptacion local — con la evidencia mas fuerte posible (mIoU=0 zero-shot) y la curva del
presupuesto few-shot.

### (3) Resultados / figuras del proyecto que usa (rutas reales)
- `reports/segmentation/sen4agrinet_transfer_result.json` — zero->few-shot FR->Catalonia.
- `paper/tables/us-073-transfer/sen4agrinet_domain_gap.tex` — tabla domain gap.
- `paper/tables/us-073-transfer/eurocropsml_kshot.tex` +
  `data/transfer/eurocropsml_fewshot_results.parquet` — curva k-shot real, 3 semillas.
- `paper/figures/us-073-transfer/kshot_curve.{png,svg}` — figura k-shot.
- `paper/figures/us-073/domain_gap_umap.png` — UMAP FR vs Catalonia.
- `paper/figures/us-073/ndvi_phenology_offset.png` — desfase fenologico NDVI.
- `paper/figures/us-073-transfer/mexico_phenology.png` — demo Mexico cualitativa.
- `docs/data/hcat_crosswalk.md` + `data/reference/hcat_crosswalk.parquet` — armonizacion
  HCAT v3 (18 PASTIS -> 11 macro), pieza metodologica clave para transfer cross-dataset.
- Notebook ejecutado: `notebooks/segmentation/5c_transfer_sen4agrinet.ipynb`.

### (4) Que FALTA para completarlo (honesto)
- **Mas de un par dense-transfer**: hoy solo existe FR->Catalonia denso. Para un paper de
  transferibilidad robusto conviene **>=1 segundo par denso** (p.ej. FR->otra region de
  Sen4AgriNet, o PASTIS->un tile EuroCrops con etiquetas densas). **Requiere correr un
  finetune nuevo** (no existe). Esfuerzo medio (datos en DVC, receta lista).
- **AlphaEarth-via-GEE para EuroCropsML**: hoy la ruta tabular usa la serie Sentinel-2
  cruda del parcel, NO el embedding AlphaEarth (EuroCropsML no lo trae). El paper seria mas
  fuerte si la curva k-shot tambien existiera *sobre embeddings AlphaEarth* extraidos via
  GEE para EE/LV/PT. **Esto NO existe; es trabajo nuevo de ingestion GEE.** Declararlo como
  experimento adicional o como future work.
- **Tamano de muestra dense**: el few-shot Catalonia es deliberadamente diminuto (10
  parches). Para venue exigente conviene **una curva few-shot densa** (variar #parches),
  no un solo punto. Requiere varios finetunes cortos. Esfuerzo bajo-medio.
- **Tropical/WorldCereal**: explicitamente fuera de alcance hoy; mantener como future work
  (licencias ya pre-registradas).
- Mexico: por diseno NO se le pondra metrica; mantenerlo cualitativo.

### (5) Venue / target sugerido
- **Primario**: *ISPRS Journal of Photogrammetry and RS* o *IEEE TGRS* (transferibilidad de
  FM EO es tema central en TGRS) — o *Remote Sensing of Environment* si se anade el segundo
  par denso + curva.
- **Workshop**: CVPR *EarthVision* / ICCV *LUAR* (transfer + domain gap visual encaja).

### (6) Esfuerzo estimado
**Medio: ~4-6 semanas.** Lo ya medido (zero/few FR->Cat, curva k-shot, UMAP, NDVI, Mexico)
sostiene el ~70% del paper. El 30% restante (segundo par denso + idealmente AlphaEarth-GEE
para EuroCropsML + curva few-shot densa) **requiere experimentos nuevos** — el principal
factor de riesgo y de tiempo. Sin esos extras, es publicable en workshop pero quizas no en
journal Q1.

---

## Paper C — Agente fundamentado: Be My Eyes aplicado a lo geoespacial

### (1) Titulo tentativo
*"Be My Eyes for crop mapping: grounding a frozen LLM reasoner on a specialist perceiver
to answer geospatial questions without hallucinating the pixels"*

(alt: *"A perceiver/reasoner copilot for satellite crop mapping: re-cabling to the champion,
Spatial-RAG, and a bilingual agronomic QA benchmark"*)

### (2) Angulo / contribucion principal
El principal describe la arquitectura; aqui **el agente y su evaluacion SON el paper**.
Contribucion:

1. **Patron perceiver/reasoner para geoespacial**: el LLM frontera (Gemini 2.5 Pro o
   Qwen3.5-35B-A3B on-prem) esta *frozen* y **nunca clasifica pixeles**; nuestros modelos
   densos/tabulares *perciben* y emiten texto; el reasoner razona, invoca herramientas y
   responde. Cota de alucinacion por construccion: toda cifra citada nace de un
   `tool_result` auditable.
2. **El re-cableo al campeon como resultado de ingenieria medible**: cambiar el perceiver
   del agente del baseline XGBoost-AlphaEarth al campeon Stacking-5 sube la accuracy del
   agente de **0.8308 a 0.9413 (+11.0 pp)** y F1-macro **+21.4 pp** sobre 13 481 parcelas
   reales (france-9). El reasoner *hereda* la accuracy del campeon sin tocar un pixel:
   evidencia cuantitativa de por que la separacion paga.
3. **Spatial-RAG hibrido** (pre-filtro PostGIS `ST_DWithin` + similitud pgvector sobre
   embeddings AlphaEarth 64-dim) como mecanismo de grounding sobre parcelas vecinas reales.
4. **Doble backend con soberania de datos**: el mismo agente corre cloud (Gemini) u on-prem
   (Qwen vLLM GPTQ-Int4, single-GPU) — relevante para clientes agro con datos sensibles.
5. **Benchmark bilingue AgroMind-IT/ES** (italiano/espanol) como contribucion de evaluacion.

### (3) Resultados / figuras del proyecto que usa (rutas reales)
- `reports/agent_bench/perceiver_champion_eval.json` — **re-cableo al campeon** (nucleo).
- `reports/agent_bench/us049_system_eval.json` + `us049_system_report.html` — metricas de
  uso de tools (Gemini: tool-sel 0.55, arg-match 0.95, routing 1.0, crop-match 0.92,
  halluc-RAG 0.10; Qwen on-prem: tool-sel 0.75, arg-match 0.98) — corrida REAL de 20
  escenarios.
- `paper/tables/us-070/llm_benchmark.tex` y `paper/tables/us-070/tool_ablation.tex` —
  tablas Gemini vs Qwen (con columnas AgroMind-IT/ES marcadas *pendiente*).
- `reports/agent_bench/us049_report.html` / `us049_report_v2.html` — reportes LLM-as-judge.
- `reports/agent_bench/traces/` — trazas ADK auditables (Be My Eyes en accion).
- Skills/arquitectura: `agrosat-google-adk-agent`, `agrosat-spatial-rag`,
  `agrosat-ml-evaluation`.

### (4) Que FALTA para completarlo (honesto) — ESTE ES EL PAPER MAS INCOMPLETO
- **AgroMind / AgroMind-IT/ES NO esta evaluado de verdad**: la tabla US-069
  (`paper/tables/us-069/benchmark_comparison.tex`) es un **placeholder explicito sin
  poblar**. El benchmark bilingue existe SOLO como **fixtures semilla minusculas**
  (`data/benchmark/agromind_it_es/seed.fixture.jsonl` = **3 lineas**;
  `data/test_fixtures/agromind_itses_mini.jsonl` = 6 lineas). **NO hay benchmark de tamano
  ni una corrida real.** Afirmar resultados AgroMind-IT/ES hoy seria inventar datos.
  - Falta: (a) **construir** el set AgroMind-IT/ES completo (target ~500 items
    bilingues), (b) **correr** `ml/eval/paper_bench.py::run_paper_benchmark` en H100 con
    Qwen3.5 vLLM (US-048) y Gemini, 3 corridas + Wilcoxon. Bloqueado por H100/presupuesto.
  - **Esfuerzo alto, dependencia de hardware.** Es el cuello de botella real del paper.
- **Targets a NO sobre-afirmar**: AgroMind >=0.75 Gemini / >=0.70 Qwen son *objetivos*, no
  resultados. **NUNCA** afirmar "VLM fine-tuned supera a Gemini": la capa LLM es
  comunicacion/explicacion, no clasificacion.
- **Reduccion de alucinacion por Spatial-RAG**: hoy `hallucination_reduction_delta` es
  **NaN** en el JSON (la rama ungrounded no se midio). Falta un A/B grounded-vs-ungrounded
  completo para poder *cuantificar* la reduccion (~30% es el patron de GeoAnalystBench, no
  un numero medido aqui todavia).
- **n pequeno**: la corrida de tools es 20 escenarios. Para journal conviene mayor n y
  varianza entre semillas/runs (hoy std=0.0 porque es 1 corrida).
- **Qwen routing bajo (0.31) / crop-match bajo (0.23)** en la corrida actual: hay que
  investigar/mejorar antes de publicar, o reportarlo honestamente como limitacion on-prem.

### (5) Venue / target sugerido
- **Primario**: *Computers and Electronics in Agriculture* (Elsevier) — agente conversacional
  agro aplicado encaja perfecto — o un workshop NeurIPS/ACL de *LLM agents* / *grounded
  generation* (el angulo Be My Eyes + anti-alucinacion es muy actual).
- **Benchmark venue**: si AgroMind-IT/ES madura como dataset, *NeurIPS Datasets & Benchmarks*.

### (6) Esfuerzo estimado
**Alto: ~6-10 semanas, con riesgo de hardware.** La arquitectura, el re-cableo (+11 pp) y la
corrida de tools de 20 escenarios ya existen y son solidos. Pero el corazon evaluativo
(AgroMind-IT/ES, A/B de alucinacion, mas semillas) **requiere construir dataset + correr en
H100**, hoy bloqueado por presupuesto/ventana GPU (US-048/068/069). **Sin esa corrida real,
el paper queda cojo en la seccion de resultados.** Es el de mayor upside narrativo y mayor
riesgo de ejecucion.

---

## Recomendacion de priorizacion

| # | Paper | Madurez evidencia | Experimentos nuevos | Riesgo | Esfuerzo | Prioridad |
|---|---|---|---|---|---|---|
| A | Cardinalidad / heterogeneidad / E-c negativo | **Alta** (~90% medido) | 1 ablation menor | Bajo | 2-3 sem | **1 (primero)** |
| B | Transferencia multi-region | Media-alta (~70%) | 2do par denso + GEE EuroCrops | Medio | 4-6 sem | **2** |
| C | Agente Be My Eyes | **Baja en evaluacion** | dataset IT/ES + run H100 + A/B halluc | Alto (hardware) | 6-10 sem | **3 (ultimo)** |

**Logica**: Paper A es *spin-off de bajo riesgo* — casi todo esta medido, el resultado
negativo E-c y el trade-off de cardinalidad son hallazgos limpios y autocontenidos; debe
salir primero. Paper B tiene una espina dorsal solida (zero->few + curva k-shot + domain gap
visual) pero necesita 1-2 experimentos nuevos para journal Q1. Paper C tiene la narrativa
mas atractiva (anti-alucinacion + soberania de datos) pero su evaluacion central **no
existe todavia** (AgroMind-IT/ES = 3 lineas de fixture, tabla US-069 placeholder, delta de
alucinacion = NaN) y depende de la ventana H100; arrancar su *escritura de metodo* en
paralelo, pero su sometimiento va al final.

## Anti-patrones a evitar en los tres papers (honestidad)

- NO reportar la columna AgroMind-IT/ES como si estuviera medida (hoy es placeholder).
- NO afirmar zero-shot fuera de Francia (FR->Cat zero-shot = 0.0; Mexico = cualitativo).
- NO afirmar que un VLM fine-tuned supera a Gemini (la capa LLM = comunicacion, no
  clasificacion de pixeles).
- NO afirmar TSViT mIoU >0.75 (PASTIS-R satura ~0.70 para este set de clases).
- NO presentar el delta E-c (+0.0002) como mejora: es ruido; el valor es el resultado
  NEGATIVO bien enmarcado.
- NO inventar reduccion de alucinacion de Spatial-RAG: hoy el delta es NaN, falta el A/B.
- Mantener atribuciones de licencia (CC-BY-4.0 AlphaEarth, CC-BY-SA-4.0 Sen4AgriNet/
  EuroCropsML) y share-alike en derivados combinados.
