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
- License: CC-BY-4.0
- Citation: Schneider, M., Schelte, T., Schmitz, F., Korner, M. (2023). _EuroCrops: The largest harmonized open crop dataset across the European Union_. Scientific Data.
- Contents: Hierarchical Crop and Agriculture Taxonomy v3 (HCAT3) con ~270 clases canónicas armonizadas + parcelas vectoriales por país EU. Italia y Francia disponibles.
- Attribution required: "EuroCrops / HCAT3 (Schneider et al. 2023, CC-BY-4.0, TUM)" en figuras y reportes derivados.
- Use scope US-013/EPIC 8: taxonomía HCAT3 como referencia para alinear PASTIS-R (Francia) ↔ futuros labels GSAA (Italia, US-006/007 diferidos) bajo un sistema canónico común.

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

## Bibliografía agronómica de los índices custom (US-014)

Las 3 fórmulas custom del catálogo (`LAI`, `FAPAR`, `CCCI`) implementan
versiones canónicas del proyecto con DOI propio:

- **LAI**: Boegh et al. (2002). *Remote Sensing of Environment* 81(2-3), 179-193. DOI [10.1016/S0034-4257(01)00342-X](https://doi.org/10.1016/S0034-4257(01)00342-X).
- **FAPAR**: Myneni & Williams (1994). *Remote Sensing of Environment* 49(3), 200-211. DOI [10.1016/0034-4257(94)90016-7](https://doi.org/10.1016/0034-4257(94)90016-7).
- **CCCI**: Barnes et al. (2000). Proc. 5th International Conference on Precision Agriculture, Bloomington MN.

Tabla académica completa con DOIs por índice en [`docs/spectral_indices.md`](../spectral_indices.md).
