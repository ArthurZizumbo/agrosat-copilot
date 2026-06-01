# Datasets y modelos: atribuciones de licencia

Documenta TODOS los datasets y modelos usados durante el proyecto. Sin esto, el cumplimiento legal del MVP falla.

## Datasets

### AlphaEarth Foundations v2.1 — Google DeepMind
- Source: Google Earth Engine Data Catalog `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`
- License: [GEE Terms of Service](https://earthengine.google.com/terms/)
- Use: research + commercial with attribution
- Citation: Brown et al. (2024). AlphaEarth Foundations. Google DeepMind.
- Attribution required: "Google AlphaEarth Foundations" en figuras y reportes
  derivados. Cada query a `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` debe cumplir
  con los limites de cuota EE para uso no-comercial; uso comercial requiere
  contrato con Google Earth Engine for Business.
- Use scope US-011: muestreo on-the-fly de 64 dims via `sample()` y
  `reduceRegions()` sobre las 3 ROIs italianas (Pianura Padana, Toscana,
  Apulia) y sobre los 2,433 patches PASTIS-R en Francia. Cache parquet local
  en `data/cache/gee/` (gitignored).

### Sentinel-2 L2A & Sentinel-1 GRD — Copernicus
- Source: Copernicus Data Space Ecosystem · Google Earth Engine
  `COPERNICUS/S2_SR_HARMONIZED` (S2 L2A surface reflectance) y
  `COPERNICUS/S1_GRD` (S1 IW GRDH ascending+descending, sigma0 dB).
- License: Copernicus Open Access (free, full, open) — CC-BY-SA equivalente.
- Attribution required: "Contains modified Copernicus Sentinel data 2017-2025"
- Use scope US-010/011/012: muestreo on-the-fly desde GEE para EDA univariado de las 3 ROIs italianas (Pianura Padana, Toscana centrale, Apulia).
- Use scope US-016 (Sentinel-1 GRD): bloque backscatter VV+VH del vector
  multisensor fusionado por parcela. Preset operativo: IW GRDH ascending +
  descending mosaicados, despeckle Lee 7×7, sigma0 calibrado en dB.
  Stats anuales `{mean, std, p25, p50, p95}` por polarización (10 cols).
  Helper `sample_s1_roi_for_parcels` en `ml/ingest/gee_sampler.py`.

### SRTM v3 — NASA / USGS
- Source: GEE `USGS/SRTMGL1_003` (SRTM v3, ~30 m resolution).
- License: U.S. public domain (NASA / USGS distribuyen sin restricciones).
- Citation: Farr, T.G. et al. (2007). *The Shuttle Radar Topography Mission*.
  Reviews of Geophysics 45, RG2004. DOI
  [10.1029/2005RG000183](https://doi.org/10.1029/2005RG000183).
- Use scope US-016: bloque terreno del vector multisensor (3 cols:
  `srtm_elev_mean`, `srtm_slope_mean`, `srtm_aspect_dominant`). `slope` y
  `aspect` derivados server-side con `ee.Terrain.slope` / `ee.Terrain.aspect`.
  Helper `sample_srtm_terrain` en `ml/ingest/gee_sampler.py`.

### PASTIS-R — INRAE / Sainte-Fare-Garnot et al. 2021
- Source: Zenodo · HuggingFace `INRAE/PASTIS-R`
- License: CC-BY-SA 4.0
- Citation: Sainte-Fare-Garnot, V., Landrieu, L. (2021). _Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks_. ICCV 2021.
- Companion paper (Radar): Sainte-Fare-Garnot, V., Landrieu, L., Chehata, N. (2022). _Multi-modal temporal attention models for crop mapping from satellite time series_. ISPRS Journal.
- Contents: 2,433 patches Sentinel-2 multitemporales (T,10,128,128) + S1 ascending/descending + anotaciones panopticas + metadata.geojson EPSG:2154 (Lambert-93 Francia) + NORM_*.json por fold. 20 clases canónicas (0 background + 1-18 cultivos + 19 void).
- Use scope US-010: PASTIS-R sirve como dataset de control con labels semánticos verificados, dado que los GSAA italianos aún no están en disco (US-006/007 diferidos).

### BreizhCrops — Rußwurm et al. (ENSTA / TUM)
- Source: bucket S3 público `breizhcrops.s3.eu-central-1.amazonaws.com` (sin autenticación) · paquete PyPI `breizhcrops`
- License: dataset distribuido bajo CC-BY-SA 4.0; código del paquete `breizhcrops` bajo MIT
- Citation: Rußwurm, M., Pelletier, C., Zollner, M., Lefèvre, S., Körner, M. (2020). _BreizhCrops: A Time Series Dataset for Crop Type Mapping_. ISPRS Archives — International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, XLIII-B2-2020, 1545-1551. DOI [10.5194/isprs-archives-XLIII-B2-2020-1545-2020](https://doi.org/10.5194/isprs-archives-XLIII-B2-2020-1545-2020).
- Predecessor paper (referenciado por feedback del sponsor): Rußwurm, M. & Körner, M. (2018). _Multi-Temporal Land Cover Classification with Sequential Recurrent Encoders_. ISPRS International Journal of Geo-Information 7(4), 129. DOI [10.3390/ijgi7040129](https://doi.org/10.3390/ijgi7040129).
- Contents: series temporales Sentinel-2 por parcela (Bretaña, Francia, 2017), nivel L2A con 10 bandas ópticas + máscaras CLD/EDG/SAT, índice tabular por región (`frh01`..`frh04`), `classmapping.csv` con 9 clases agronómicas (barley, wheat, rapeseed, corn, sunflower, orchards, nuts, permanent meadows, temporary meadows). Bases HDF5 ~640 MB (frh04) / ~987 MB (frh01).
- Use scope EPIC 2: BreizhCrops es el sucesor moderno y mantenido del dataset de Rußwurm & Körner 2018 (pedido en feedback del profesor). Se usa como conjunto de control cross-region para validar que las features temporales (FFT + fenología) calibradas sobre PASTIS-R generalizan a otra región francesa. Descarga manual única vía `scripts/download_breizhcrops.sh`; versionado en DVC (`data/breizhcrops.dvc`).

### BavarianCrops / MTLCC — Rußwurm & Körner (TUM)
- Source: `github.com/MarcCoru/MTLCC`
- License: distribuido para uso académico con atribución
- Citation: Rußwurm, M. & Körner, M. (2018). _Multi-Temporal Land Cover Classification with Sequential Recurrent Encoders_. ISPRS International Journal of Geo-Information 7(4), 129. DOI [10.3390/ijgi7040129](https://doi.org/10.3390/ijgi7040129). arXiv:1802.02080.
- Contents: series temporales Sentinel-2 (Baviera, Alemania, 2016-2017), 17 clases de cultivo agregadas, labels del Bavarian StMELF.
- Use scope EPIC 2/3: **referencia** del paper provisto por el sponsor. No se descarga por defecto (BreizhCrops cumple el rol cross-regional operativo); citado en el plan v6 §3.5 y en `docs/research/datasets-investigacion-adicional.md`. Los métodos de EDA/FE extraídos de su lectura están en `ml/analysis/paper_methods.py`.

### Context-Self Contrastive Pre-training (T31TFM-1618) — Tarasiou et al. (Imperial College London)
- Source: `github.com/michaeltrs/DeepSatModels`
- License: código MIT; dataset T31TFM-1618 distribuido para uso académico con atribución
- Citation: Tarasiou, M., Güler, R. A. & Zafeiriou, S. (2022). _Context-self contrastive pretraining for crop type semantic segmentation_. IEEE Transactions on Geoscience and Remote Sensing. arXiv:2104.04310.
- Use scope EPIC 3/5: **referencia** del segundo paper provisto por el sponsor. El hallazgo interior-vs-frontera (Fig. 2) se implementa como método de FE en `ml/analysis/paper_methods.py` (`boundary_interior_stats`, `compute_boundary_ratio`). Citado en el plan v6 §3.6.

### ERA5-Land Daily Aggregates — Copernicus Climate Change Service (C3S)
- Source: GEE `ECMWF/ERA5_LAND/DAILY_AGGR`
- License: Copernicus C3S Climate Data Store ToS (free, full, open)
- Attribution required: "Contains modified Copernicus Climate Change Service information 2024"
- Citation: Munoz Sabater, J., (2019). ERA5-Land hourly data from 1950 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). DOI: 10.24381/cds.e2161bac
- Use scope US-012: variable `total_precipitation_sum` agregada anualmente
  por ROI (bbox PASTIS-R) para detectar anomalias climaticas (anos secos /
  cantidos) y cruzarlas con NDVI maximo anual derivado de Sentinel-2 (AC-8).
  Cache parquet local en `data/cache/gee/` (gitignored).
- Use scope US-016: agregaciones mensuales server-side via GEE para el
  bloque ERA5 del vector multisensor fusionado por parcela (24 cols:
  `era5_tmean_m01..m12` en °C y `era5_prec_m01..m12` acumulado mensual).
  Helper `sample_era5_monthly_climate` en `ml/ingest/gee_sampler.py`.

### Dynamic World — Google + WRI
- Source: GEE `GOOGLE/DYNAMICWORLD/V1`
- License: CC-BY-4.0
- Attribution required: "Dynamic World near real-time LULC, Google + World
  Resources Institute, 2022" en figuras derivadas.
- 9 clases LULC (water, trees, grass, flooded_vegetation, crops,
  shrub_and_scrub, built, bare, snow_and_ice).
- Use scope US-011: labels proxy para AlphaEarth × LULC sobre Italia (Seccion 1
  del notebook 02b). Sustituye temporalmente al GSAA italiano hasta US-008.

### EuroCrops / HCAT3 — TUM (Schneider et al.)
- Source: [EuroCrops project](https://www.eurocrops.tum.de/) · HuggingFace `Lobster/EuroCrops`
- License (taxonomia HCAT3 + harmonizacion EuroCrops): CC-BY-4.0
- License (capa subyacente `FR_2018/`, Registre Parcellaire Graphique francias):
  **ODbL 1.0** (Open Database License) — datos publicados por el Institut
  Geographique National (IGN) via la plataforma etalab.gouv.fr. EuroCrops
  redistribuye los shapefiles RPG sin modificar la licencia.
- Citation: Schneider, M., Schelte, T., Schmitz, F., Korner, M. (2023). _EuroCrops: The largest harmonized open crop dataset across the European Union_. Scientific Data.
- Contents: Hierarchical Crop and Agriculture Taxonomy v3 (HCAT3) con ~270 clases canónicas armonizadas + parcelas vectoriales por país EU. Italia y Francia disponibles.
- Attribution required: "EuroCrops / HCAT3 (Schneider et al. 2023, CC-BY-4.0, TUM)" + "Registre Parcellaire Graphique 2018 (IGN, ODbL 1.0)" en figuras y reportes derivados de Francia.
- Use scope US-013/EPIC 8: taxonomía HCAT3 como referencia para alinear PASTIS-R (Francia) ↔ futuros labels GSAA (Italia, US-006/007 diferidos) bajo un sistema canónico común.
- Use scope US-022-c P4.2: `data/reference/eurocrops/FR_2018/` (~11 GB)
  versionado en DVC (tag `eurocrops-fr-2018-v1`, MD5
  `52802ffe4cee88ac99a9ed42c658d8d7.dir`) como dataset de control
  cross-region complementario a PASTIS-R.

### GSAA Italia — AGEA Open Data
- Source: portales regionales Open Data AGEA (Agenzia per le Erogazioni in
  Agricoltura), descarga manual por region (Pianura Padana, Toscana, Puglia).
- License: **ODbL 1.0** (Open Database License) — publicado bajo Codice
  dell'Amministrazione Digitale + IODL 2.0 (compatible ODbL upstream).
- Attribution required: "Geospatial Aid Application Italia, AGEA (ODbL 1.0)"
  en figuras y reportes derivados.
- Use scope US-007/US-016 futuro: parcelas administrativas oficiales por
  region italiana (Pianura Padana, Toscana centrale, Puglia) para
  alimentar el bloque `parcels` de Postgres + features tabulares por
  parcela. **Estado al cierre US-022-c**: dataset no descargado al
  momento (US-006/US-007 diferidos en el plan v6). Atribucion registrada
  proactivamente como deuda heredada de US-022-b B-6.

### AgroMind Benchmark
- Source: HuggingFace `AgroMind/AgroMind`
- License: CC-BY
- 28482 QA pairs; subset 1000 usado en eval

### AgroMind-IT/ES (own contribution)
- Source: build by team, validated by Scuola Sant'Anna native reviewer
- License (target): CC-BY-4.0
- DOI Zenodo: TBD (publicación semana 10-11)

## Modelos

### Gemma 4 26B-MoE — Google DeepMind
- HF: `google/gemma-4-26b-it`
- License: Apache 2.0
- Multimodal img+video+audio, 256K ctx, 140 idiomas

### Qwen3.5-35B-A3B & Qwen3-VL-30B-A3B — Alibaba Qwen Team
- HF: `Qwen/Qwen3.5-35B-A3B` (sin `-Instruct`), `Qwen/Qwen3-VL-30B-A3B-Instruct`
- License: Apache 2.0

### DINOv3-satellite — Meta
- HF: `facebook/dinov3-vitl16-pretrain-sat493m`
- License: DINOv3 License (research + commercial con restricciones específicas)
- Aceptar términos antes de descargar

### AnySat — IGN / Gabriel Astruc et al.
- Source: `torch.hub` repo `gastruc/anysat` (entrypoint `anysat`, pesos preentrenados)
- License: MIT (codigo y pesos del repositorio oficial)
- Citation: Astruc, G., Gonthier, N., Mallet, C., Landrieu, L. (2024). _AnySat: An
  Earth Observation Model for Any Resolutions, Scales, and Modalities_. arXiv:2412.14123.
- Attribution required: "AnySat (Astruc et al., 2024, IGN)" en figuras y reportes
  derivados del modelo #6 del Avance 4.
- Use scope Avance 4 (EPIC 5): foundation model EO multimodal/multitemporal usado
  con **encoder congelado** como extractor de features densas + cabeza lineal
  entrenable para segmentacion semantica sobre PASTIS-R. Carga via `torch.hub` en
  Colab/L4 (wrapper `ml/models/anysat_wrapper.py`). NO se reentrena el encoder.

### e5-mistral-7b-instruct (embeddings RAG)
- HF: `intfloat/e5-mistral-7b-instruct`
- License: MIT

### Gemini 3.1 Pro — Google
- Access: Vertex AI API
- License: Google Cloud ToS

## Librerías de feature engineering

### spyndex — David Montero Loaiza et al.
- Repo: [awesome-spectral-indices/spyndex](https://github.com/awesome-spectral-indices/spyndex) `^0.10.0`
- License: MIT
- Citation: Montero, D., Aybar, C., Mahecha, M.D. et al. (2023). *A standardized catalogue of spectral indices to advance the use of remote sensing in Earth system research*. Scientific Data 10, 197. DOI [10.1038/s41597-023-02096-0](https://doi.org/10.1038/s41597-023-02096-0).
- Use scope US-014: backend principal de `ml/features/spectral_indices.py` para 14 de los 17 índices canónicos. Mapeo y alias documentados en [`docs/spectral_indices.md`](../spectral_indices.md).

### eemont — David Montero Loaiza
- Repo: [davemlz/eemont](https://github.com/davemlz/eemont) `^2025.7.1`
- License: MIT
- Use scope US-014: wrapper opcional `compute_index_ee` para pipelines server-side de Earth Engine (US-006/US-009).

### h3-py — Uber Technologies
- Repo: [uber/h3-py](https://github.com/uber/h3-py) `^4.1.2`
- License: Apache 2.0
- Citation: Brodsky, I. (2018). *H3: Uber's Hexagonal Hierarchical Spatial
  Index*. Uber Engineering Blog. https://eng.uber.com/h3/
- Use scope US-016: tessellation hexagonal H3 res 5 (~252 km²) sobre el bbox
  de las parcelas italianas. Centroides de las celdas se clusterizan con
  KMeans (K=5) y se aplica un buffer de exclusion de 1 km entre folds
  vecinos para evitar leakage espacial. Implementado en
  `ml/features/spatial_split.py::build_spatial_kfold`.

### xgboost — DMLC / XGBoost contributors
- Repo: [dmlc/xgboost](https://github.com/dmlc/xgboost) `^3.2.0`
- License: Apache 2.0
- Citation: Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree
  Boosting System*. Proceedings of the 22nd ACM SIGKDD. DOI
  [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785).
- Use scope US-019: uno de los dos clasificadores del baseline tabular
  (`ml/train/baseline.py`). `tree_method="hist"` con `device="cuda"` si hay
  GPU NVIDIA local disponible, degrada a CPU en CI.

### mlflow — Databricks / MLflow contributors
- Repo: [mlflow/mlflow](https://github.com/mlflow/mlflow) `^3.1`
- License: Apache 2.0
- Use scope US-019: tracking de experimentos del baseline (params, metrics,
  artefactos, tags `data_version` + `code_version`) y Model Registry. El
  servidor local corre en Docker (`infrastructure/docker/mlflow.Dockerfile`)
  con backend store en PostgreSQL.

### shap — Scott Lundberg et al. / SHAP contributors
- Repo: [shap/shap](https://github.com/shap/shap) `^0.50.0`
- License: MIT
- Citation: Lundberg, S. M. & Lee, S.-I. (2017). *A Unified Approach to
  Interpreting Model Predictions*. Advances in Neural Information Processing
  Systems 30 (NeurIPS).
- Use scope US-020: interpretabilidad del baseline tabular
  (`ml/eval/interpretability.py`). `TreeExplainer` (algoritmo TreeSHAP exacto,
  CPU) sobre los modelos RF/XGB production — summary, dependence y waterfall
  plots multiclase. Pinneado a `0.50.x`: `shap >=0.51` fija `numba <0.63`
  (marker Darwin que poetry universaliza) y choca con `numba 0.65` de vllm.

## Bibliografía agronómica de los índices custom (US-014)

Las 3 fórmulas custom del catálogo (`LAI`, `FAPAR`, `CCCI`) implementan
versiones canónicas del proyecto con DOI propio:

- **LAI**: Boegh et al. (2002). *Remote Sensing of Environment* 81(2-3), 179-193. DOI [10.1016/S0034-4257(01)00342-X](https://doi.org/10.1016/S0034-4257(01)00342-X).
- **FAPAR**: Myneni & Williams (1994). *Remote Sensing of Environment* 49(3), 200-211. DOI [10.1016/0034-4257(94)90016-7](https://doi.org/10.1016/0034-4257(94)90016-7).
- **CCCI**: Barnes et al. (2000). Proc. 5th International Conference on Precision Agriculture, Bloomington MN.

Tabla académica completa con DOIs por índice en [`docs/spectral_indices.md`](../spectral_indices.md).

## US-022-b — Reencuadre fenologico (descripciones LLM + text-encoder)

### Gemini 3.5 Flash — Google / Vertex AI
- Source: Vertex AI Generative AI en GCP (region `europe-west4` y `us-central1`).
- License: [Vertex AI Service Specific Terms](https://cloud.google.com/terms/service-terms).
  Modelo cerrado, uso pay-per-token. Para Paper Track con datos no
  confidenciales del proyecto (curvas NDVI publicas PASTIS-R), los terminos
  estandar cubren el caso de uso (sin datos sensibles ingresan al prompt).
- Citation: Google DeepMind (2025). Gemini 3.5 Flash technical report.
  Vertex AI Generative AI catalog.
- Use scope US-022b-D: generador de descripciones fenologicas estructuradas
  (`ml/features/phenology_description.py`) a partir de la curva NDVI por
  parcela. Invocacion via LiteLLM (no SDK directo `google-genai`).
  `temperature=0` obligatorio + cache por hash de curva + subset
  estratificado (presupuesto < $10 USD, R7).

### sentence-transformers (all-MiniLM-L6-v2) — UKPLab
- Repo: [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers) `^5.0.0`
- Model: `sentence-transformers/all-MiniLM-L6-v2` (HF hub).
- License: Apache 2.0 (libreria) + Apache 2.0 (pesos del modelo MiniLM
  pre-entrenado por Microsoft).
- Citation: Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence
  Embeddings using Siamese BERT-Networks*. EMNLP 2019.
  arXiv [1908.10084](https://arxiv.org/abs/1908.10084).
- Use scope US-022b-D: text-encoder default de la rama semantica
  (`encode_descriptions` en `ml/features/phenology_description.py`).
  Vector denso 384-dim normalizado (cosine similarity) que se concatena
  al vector tabular como bloque `pheno_text_*` via LEFT JOIN en
  `ml/features/fusion.py`.

### breizhcrops — Russwurm, Pelletier et al.
- Repo: [dl4sits/BreizhCrops](https://github.com/dl4sits/BreizhCrops) `^0.0.4.1`
- License: MIT
- Citation: Russwurm, M., Pelletier, C., Zollner, M., Lefevre, S., Korner, M.
  (2020). *BreizhCrops: A Time Series Dataset for Crop Type Mapping*.
  ISPRS Archives Volume XLIII-B2-2020.
  DOI [10.5194/isprs-archives-XLIII-B2-2020-1545-2020](https://doi.org/10.5194/isprs-archives-XLIII-B2-2020-1545-2020).
- Use scope US-022b-C (actualizado 2026-05-22): arquitecturas TempCNN
  (Pelletier et al. 2019) e InceptionTime (Fawaz et al. 2020) portadas
  nativas al repo en `ml/models/temporal.py`. La implementacion se basa
  en los papers originales y en el codigo de referencia `breizhcrops`
  (MIT) y se reimplementa con tres bloques Conv1D + dense head para
  TempCNN y 6 modulos Inception con shortcut residual para
  InceptionTime. Los pesos He uniform, BatchNorm intermedio y dropout
  configurable provienen de los papers. **El paquete `breizhcrops` ya no
  es dependencia de runtime de `phenology_models.py`** (D-ARQ-2
  actualizado), solo sigue presente para descargar el dataset
  BreizhCrops cross-region.

### TempCNN — Pelletier, Webb & Petitjean (2019)
- Citation: Pelletier, C., Webb, G. I., & Petitjean, F. (2019).
  *Temporal Convolutional Neural Network for the Classification of
  Satellite Image Time Series*. Remote Sensing 11(5):523.
  DOI [10.3390/rs11050523](https://doi.org/10.3390/rs11050523).
- License de la arquitectura: codigo de referencia del paper bajo MIT
  (publicado en GitHub por los autores).
- Use scope US-022b-C: `ml/models/temporal.py::TempCNN` reimplementa
  la arquitectura desde el paper + el codigo de referencia.

### InceptionTime — Fawaz et al. (2020)
- Citation: Fawaz, H. I., Lucas, B., Forestier, G., Pelletier, C.,
  Schmidt, D. F., Weber, J., Webb, G. I., Idoumghar, L., Muller, P. A.,
  & Petitjean, F. (2020). *InceptionTime: Finding AlexNet for Time
  Series Classification*. Data Mining and Knowledge Discovery 34,
  1936-1962.
  DOI [10.1007/s10618-020-00710-y](https://doi.org/10.1007/s10618-020-00710-y).
- License de la arquitectura: codigo de referencia del paper bajo MIT
  (`hfawaz/InceptionTime` en GitHub).
- Use scope US-022b-C: `ml/models/temporal.py::InceptionTime`
  reimplementa la arquitectura (6 modulos Inception con shortcut
  residual cada 3, global average pooling + dense head).

## US-022-b — Paper-faro (referencia academica, no codigo)

### Wen et al. (2025) — "Phenology description is all you need"
- Source: ISPRS Journal of Photogrammetry and Remote Sensing 228, 141-165.
- DOI: [10.1016/j.isprsjprs.2025.07.002](https://doi.org/10.1016/j.isprsjprs.2025.07.002).
- License de la metodologia: open access (Elsevier hybrid). El metodo
  (prompt 3-bloques, text-encoder contrastivo) se implementa de cero
  en `ml/features/phenology_description.py` siguiendo Fig. 2 y Fig. 3a.

## US-022-c — Paper-faro FarSLIP (referencia academica, no codigo)

### Li et al. (2025) — "FarSLIP: Few-shot Adaptation of CLIP for Remote Sensing"
- Source: arXiv:2511.14901 (nov-2025, preprint).
- License de la metodologia: open access (arXiv). El metodo (PatchDistillationLoss
  parche-a-parche §3.2 + RegionCategoryAlignmentLoss InfoNCE CLS §3.3 + student init
  desde teacher con NIR=mean(RGB) anti-dead-neuron §3.1) se reimplementa en
  `ml/farslip/distill.py` con fidelidad 1:1 documentada en `docs/decisions/ADR-007-farslip-fidelity-paper.md`.
- Attribution required: "FarSLIP method adapted from Li et al. (2025), arXiv:2511.14901"
  en figuras y reportes derivados.
- Use scope US-022-c P1: ejecucion del training student en L4 GCP spot sobre 3 ROIs
  italianas (Pianura Padana, Toscana, Puglia) + eval cross-region en PASTIS-R Francia.

## US-023-preview — Paper-faro Frampton 2013 (Red Edge Position)

### Frampton et al. (2013) — "Evaluating the capabilities of Sentinel-2 for quantitative estimation of biophysical variables in vegetation"
- Source: ISPRS Journal of Photogrammetry and Remote Sensing 82, 83-92.
- DOI: [10.1016/j.isprsjprs.2013.04.007](https://doi.org/10.1016/j.isprsjprs.2013.04.007).
- License de la metodologia: open access. La formula linear-4-bands Red
  Edge Position (REP, eq. 1) se implementa de cero en
  `ml/features/spectral_signature.py::compute_rep` siguiendo la
  parametrizacion del paper sobre las bandas Sentinel-2 B04/B05/B06/B07.
- Attribution required: "REP descriptor adapted from Frampton et al. (2013),
  DOI 10.1016/j.isprsjprs.2013.04.007" en figuras y reportes derivados.
- Use scope US-023-preview P5: descriptor compacto de firma espectral
  por parcela para ablation `with_spectral_signature` / `spectral_signature_only`
  + integracion como bloque opcional en `ml/features/fusion.py`.
