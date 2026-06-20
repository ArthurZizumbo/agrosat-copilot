# Glosario tecnico — AgroSatCopilot (IT / ES / EN)

Referencia unica de terminos del proyecto, estandarizada en italiano, espanol e
ingles. Consistente con la regla de CLAUDE.md: **codigo (identificadores,
comentarios, docstrings) en ingles; prosa visible (docs, notebooks, prints) en
espanol neutro**. La columna "Termino canonico (EN)" es la forma a usar en
codigo; el italiano y el espanol son para la prosa de notebooks/docs.

Sin emojis. Agrupado por dominio.

---

## 1. Observacion de la Tierra (EO) y datasets

| Termino canonico (EN) | Italiano | Espanol | Definicion (ES) |
|---|---|---|---|
| AlphaEarth V1/ANNUAL v1.1 | AlphaEarth V1/ANNUAL v1.1 | AlphaEarth V1/ANNUAL v1.1 | Foundation Model de observacion de la Tierra de Google DeepMind. Embedding anual de 64 dimensiones (`SATELLITE_EMBEDDING/V1/ANNUAL`, data v1.1), global incluido Mexico, gratis en Google Earth Engine bajo CC-BY-4.0. **No usar "v2.1": es un termino muerto.** |
| Earth Observation (EO) | Osservazione della Terra | Observacion de la Tierra | Adquisicion de datos de la superficie terrestre por sensores satelitales. |
| Sentinel-2 | Sentinel-2 | Sentinel-2 | Satelite optico multiespectral (ESA Copernicus); base de las series temporales del proyecto. |
| Sentinel-1 | Sentinel-1 | Sentinel-1 | Satelite radar SAR (ESA Copernicus); fuente complementaria all-weather. |
| DINOv3-satellite | DINOv3-satellite | DINOv3-satellite | Modelo self-supervised de features visuales (`facebook/dinov3-vitl16-pretrain-sat493m`), usado congelado (frozen). |
| Embedding | Embedding | Embedding (vector latente) | Representacion vectorial densa de baja dimension de una imagen/parcela. |
| COG | COG | COG (GeoTIFF optimizado para nube) | Cloud-Optimized GeoTIFF; raster con tiling interno para lectura parcial por HTTP. |
| AOI | AOI | AOI (area de interes) | Area of Interest; poligono que delimita la zona a analizar. |

## 2. Datasets de etiquetas

| Termino canonico (EN) | Italiano | Espanol | Definicion (ES) |
|---|---|---|---|
| PASTIS-R | PASTIS-R | PASTIS-R | Dataset de series Sentinel-2 con etiquetas densas de cultivo (Francia); base de segmentacion y baselines. |
| Sen4AgriNet | Sen4AgriNet | Sen4AgriNet | Dataset multi-region de cultivos (CC-BY-SA-4.0); subset Catalonia 31TCG usado para transfer FR -> Catalonia. |
| EuroCropsML | EuroCropsML | EuroCropsML | Dataset europeo de cultivos para few-shot (CC-BY-SA-4.0, Zenodo). |
| HCAT v3 | HCAT v3 | HCAT v3 | Hierarchical Crop and Agriculture Taxonomy v3; taxonomia jerarquica de cultivos para crosswalk de etiquetas. |
| GSAA | GSAA | GSAA | Geo-Spatial Aid Application; declaraciones parcelarias de la PAC usadas como etiqueta. |

## 3. Arquitecturas de modelo

| Termino canonico (EN) | Italiano | Espanol | Definicion (ES) |
|---|---|---|---|
| U-TAE | U-TAE | U-TAE | U-Net con Temporal Attention Encoder para series satelitales. |
| TSViT | TSViT | TSViT | Temporal-Spatial Vision Transformer (Tarasiou et al., 2023); factoriza atencion temporal y espacial. |
| TSViT-pheno | TSViT-pheno | TSViT-pheno | Variante propia de TSViT con rama fenologico-contrastiva (ADR-006). |
| SegFormer-B2 | SegFormer-B2 | SegFormer-B2 | Transformer de segmentacion semantica jerarquico. |
| Swin-UNETR | Swin-UNETR | Swin-UNETR | U-Net con encoder Swin Transformer para segmentacion. |
| FarSLIP | FarSLIP | FarSLIP | Metodo vision-language para cultivos (Li et al., 2025); alinea imagen con prototipos de texto. |
| Ensemble | Insieme (ensemble) | Ensamble (ensemble) | Combinacion de varios modelos base en un predictor final. |
| Voting | Voting | Votacion (Voting) | Ensemble por voto de los top-3 modelos homogeneos. |
| Bagging | Bagging | Bagging | Ensemble por bootstrap aggregating (XGB sobre AlphaEarth). |
| Stacking | Stacking | Stacking (apilamiento) | Ensemble con meta-learner sobre predicciones out-of-fold de los miembros. |
| Blending | Blending | Blending (mezcla) | Ensemble con pesos optimizados (Optuna) sobre un holdout. |

## 4. MLOps e infraestructura

| Termino canonico (EN) | Italiano | Espanol | Definicion (ES) |
|---|---|---|---|
| DVC | DVC | DVC | Data Version Control; versiona rasters/COG/pesos fuera de Git (remote en GCS). |
| MLflow | MLflow | MLflow | Tracking de experimentos y Model Registry. El lineage real vive en el server `:5010`, no en `./mlruns`. |
| Dagster | Dagster | Dagster | Orquestador orientado a assets con lineage declarativo. |
| asset | asset | asset (activo de datos) | Unidad de datos materializable en Dagster con lineage explicito. |
| lineage | lineage | linaje (lineage) | Trazabilidad de dependencias dataset -> feature -> modelo. |
| dbmate | dbmate | dbmate | Herramienta de migraciones SQL rollforward (PostgreSQL). |
| `data_version` | `data_version` | `data_version` | Tag MLflow con la version del dato usada en un run. |
| `code_version` | `code_version` | `code_version` | Tag MLflow con el SHA corto de git del codigo de un run. |

## 5. Agente y LLM

| Termino canonico (EN) | Italiano | Espanol | Definicion (ES) |
|---|---|---|---|
| ADK | ADK | ADK (Google Agent Dev Kit) | Framework del agente conversacional Plan-and-React con tracing integrado. |
| reasoner | reasoner | reasoner (razonador) | LLM que planifica y razona en el copiloto (Gemini cloud variante A / Qwen on-prem variante B). |
| perceiver | perceiver | perceiver (perceptor) | Componente que observa el contexto multimodal y emite observaciones al reasoner. |
| tool-call | tool-call | tool-call (llamada a herramienta) | Invocacion de una FunctionTool geoespacial por el agente. |
| RLS | RLS | RLS (seguridad a nivel de fila) | Row-Level Security de PostgreSQL; aislamiento por tenant/sesion (pendiente US-051). |
| multi-tenant | multi-tenant | multi-inquilino (multi-tenant) | Aislamiento de datos por `session_id` en toda query. |
| p95 | p95 | p95 (percentil 95) | Percentil 95 de latencia; SLO de observabilidad del chat. |
| SSRF | SSRF | SSRF (falsificacion de peticion del servidor) | Server-Side Request Forgery; mitigado en el tiler con `validate_cog_url` (allowlist). |

## 6. Agronomia y fenologia

| Termino canonico (EN) | Italiano | Espanol | Definicion (ES) |
|---|---|---|---|
| NDVI | NDVI | NDVI | Normalized Difference Vegetation Index; indice de vigor de la vegetacion. |
| phenology | fenologia | fenologia | Estudio de los estadios temporales del cultivo (siembra, crecimiento, senescencia). |
| zero-shot | zero-shot | zero-shot (sin ejemplos) | Inferencia en una region/clase sin ejemplos de entrenamiento locales. |
| few-shot | few-shot | few-shot (pocos ejemplos) | Adaptacion con muy pocas etiquetas locales (p.ej. 10 patches). |
| k-shot | k-shot | k-shot | Regimen de aprendizaje con k ejemplos por clase. |
| mIoU | mIoU | mIoU (IoU media) | mean Intersection over Union; metrica de segmentacion. |
| Delta mIoU | Delta mIoU | Delta mIoU | Variacion de mIoU entre dos regimenes (p.ej. few-shot vs zero-shot: +0.2468 en transfer FR -> Catalonia). |
| F1 macro | F1 macro | F1 macro | Media no ponderada del F1 por clase. |

---

> Convencion de uso: en codigo y nombres de variables se usa siempre el termino
> canonico EN (snake_case o el id oficial). En prosa de notebooks y docs IT/ES/EN
> se usa la forma del idioma correspondiente, manteniendo el termino EN entre
> parentesis la primera vez que aparece para evitar ambiguedad.
