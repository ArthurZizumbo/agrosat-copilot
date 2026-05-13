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
- Source: Copernicus Data Space Ecosystem · Google Earth Engine `COPERNICUS/S2_SR_HARMONIZED`
- License: Copernicus Open Access (free, full, open)
- Attribution required: "Contains modified Copernicus Sentinel data 2017-2025"
- Use scope US-010/011/012: muestreo on-the-fly desde GEE para EDA univariado de las 3 ROIs italianas (Pianura Padana, Toscana centrale, Apulia).

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

### Dynamic World — Google + WRI
- Source: GEE `GOOGLE/DYNAMICWORLD/V1`
- License: CC-BY-4.0
- Attribution required: "Dynamic World near real-time LULC, Google + World
  Resources Institute, 2022" en figuras derivadas.
- 9 clases LULC (water, trees, grass, flooded_vegetation, crops,
  shrub_and_scrub, built, bare, snow_and_ice).
- Use scope US-011: labels proxy para AlphaEarth × LULC sobre Italia (Seccion 1
  del notebook 02b). Sustituye temporalmente al GSAA italiano hasta US-008.

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
