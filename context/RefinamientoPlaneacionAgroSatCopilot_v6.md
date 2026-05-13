# Plan de Proyecto: AgroSatCopilot

**Cuantificación de Superficies de Cultivo mediante Segmentación Semántica de Imágenes Satelitales, Foundation Models (AlphaEarth Foundations) y procesamiento conversacional por LLMs (Gemma 4, Qwen3-VL, Qwen3.5-35B-A3B y Gemini 3.1 Pro)**

---

**Documento:** Planeación SCRUM detallada del Proyecto Integrador — MNA (Maestría en Inteligencia Artificial Aplicada)
**Trimestre:** 20 de abril a 3 de julio de 2026 (11 semanas calendario, 10 semanas efectivas + Paper Track opcional semanas 10-11)
**Documento acompañante:** `docs/general/01.Planteamiento del Proyecto_v6.md`



### Equipo

| Integrante | Matrícula | Rol |
|------------|-----------|-----|
| Arthur Jafed Zizumbo Velasco | **MLOps / Platform Engineer (lead)** — Terraform mono-cloud GCP + Azure H100, CI/CD, DVC, MLflow, Dagster, dbmate, observabilidad, FinOps, comunicación con sponsor |
| Carlos Aaron Bocanegra Buitrón | **Full-Stack / Backend Lead** — FastAPI, TiTiler, integración con Nuxt 4 SSR, endpoints del agente con Google ADK, seguridad |
| Carlos Isaac Ávila Gutiérrez | **ML Engineer / Data Scientist** — modelos baseline y avanzados, fine-tuning Gemma 4 y Qwen3-VL, ingesta AlphaEarth, EDA con Polars, feature engineering |

### Sponsor Académico

**Dr. Gerardo Jesús Camacho González** — Profesor Investigador del Departamento de Computación, Escuela de Ingeniería y Ciencias, Tecnológico de Monterrey, Campus Santa Fe. Colaborador activo del Instituto de Inteligencia Mecánica, Scuola Superiore Sant'Anna (Pisa, Italia). Correo: gjcamacho@tec.mx.

### Capacidad y Recursos de Cómputo

- **Capacidad del equipo:** 3 desarrolladores × 12 horas semanales × 10 semanas efectivas = **360 horas-persona** equivalentes a aproximadamente **150 story points** con un factor de conversión de 2.4 horas por story point (considera pair programming, curva de aprendizaje de Gemma 4 y Google ADK, y overhead de coordinación).
- **Azure VM NVIDIA H100 NVL 96GB (prestada por el sponsor, 1×GPU single-node):** ventanas totales de **80 horas** distribuidas en 6 sesiones nocturnas para fine-tuning de Gemma 4 26B-MoE, Qwen3-VL-30B-A3B comparativo, Qwen3.5-35B-A3B serving y LoRA, modelos temporales de segmentación y evaluación final.
- **Google Cloud L4 24GB:** spot instances para iteraciones continuas de baselines, fallback de Gemma 4 E4B, DINOv3 extraction, desarrollo. Presupuesto ~50 horas efectivas.
- **Máquinas locales:** RTX 4060 y 4080 del equipo para desarrollo, debugging y sanity checks previos a jobs en cloud.

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Propuesta de Valor y Diferenciadores](#2-propuesta-de-valor)
3. [Antecedentes Académicos (Papers del Profesor)](#3-antecedentes-academicos)
4. [Estado del Arte 2025-2026](#4-estado-del-arte)
5. [Google Earth AI: Referencia Industrial a Diferenciar](#5-earth-ai)
6. [Stack Tecnológico](#6-stack-tecnologico)
7. [Arquitectura de la Solución](#7-arquitectura)
8. [FinOps: Presupuesto de Cómputo y Operación](#8-finops)
9. [Datasets, Modelos y Licenciamiento Legal](#9-datasets)
10. [Mapa de Épicas y Distribución de Story Points](#10-mapa-de-epicas)
11. [EPIC 0: Infraestructura, Cookiecutter y MLOps Base](#epic-0)
12. [EPIC 1: Ingesta de Datos](#epic-1)
13. [EPIC 2: Análisis Exploratorio de Datos](#epic-2)
14. [EPIC 3: Feature Engineering](#epic-3)
15. [EPIC 4: Baseline](#epic-4)
16. [EPIC 5: Modelos Alternativos](#epic-5)
17. [EPIC 6: Modelo Final con Gemma 4 26B-MoE + Ensambles](#epic-6)
18. [EPIC 7: Agente Conversacional con Google ADK](#epic-7)
19. [EPIC 8: Backend API + Worker Pub/Sub + Tiling](#epic-8)
20. [EPIC 9: Frontend Web Bilingüe + Switch A/B](#epic-9)
21. [EPIC 10: Observabilidad, Drift, FinOps, Seguridad](#epic-10)
22. [EPIC 11: Paper Track opcional](#epic-11)
23. [Roadmap de Sprints Semanales](#11-roadmap)
24. [Gates de Sprint 1](#12-gates)
25. [Gestión de Riesgos](#13-riesgos)
26. [Criterios de Éxito del MVP](#14-criterios)
27. [Alineación con Rúbricas del Curso](#15-rubricas)
28. [Apéndice: Decisiones Técnicas Clave](#16-apendice)

---

## 1. Resumen Ejecutivo {#1-resumen-ejecutivo}

**AgroSatCopilot** es una plataforma SaaS conversacional open-source que permite a agrónomos, gestores de política agrícola y productores interactuar en lenguaje natural (italiano, español e inglés) con imágenes satelitales multimodales para obtener análisis avanzados de cultivos en segundos. La plataforma actúa como un copiloto digital que combina cuatro tecnologías de última generación:

1. **AlphaEarth Foundations v2.1** (Google DeepMind, actualización 2025). Embeddings pre-computados de 64 dimensiones por píxel de 10 metros por año, disponibles gratuitamente en Google Earth Engine para el periodo 2017-2025 con cobertura global. Fusionan internamente Sentinel-2 óptico, Landsat, datos SAR de Sentinel-1, modelos digitales de elevación, variables climáticas ERA5 y descripciones textuales.

2. **Gemma 4 26B-MoE** (Google DeepMind, liberado 2-abr-2026). Vision-Language Model open-weight bajo licencia Apache 2.0 con 26 mil millones de parámetros totales y 4 mil millones activos por token, soporte multimodal nativo para imagen, video y audio, contexto de 256K tokens y cobertura multilingüe en más de 140 idiomas. Se fine-tunea con LoRA sobre AgroMind y AgroMind-IT/ES. Cabe con margen en una sola H100 NVL 96GB.

3. **Qwen3.5-35B-A3B** (Alibaba Qwen Team, abril 2026). LLM open-source MoE con 35B totales y 3B activos, 128K contexto, Apache 2.0. Se despliega como orquestador on-premise con vLLM en H100. Se compara contra Gemini 3.1 Pro (variante cloud Vertex AI) mediante switch A/B visible en la UI.

4. **Gemini 3.1 Pro** (Google DeepMind, Vertex AI). LLM cloud con ventana de contexto de 2 millones de tokens. Variante cloud del orquestador y baseline comparativo.

### Flujo de Uso Real

```
Usuario (italiano): "Analizza questo appezzamento. Di che coltura si tratta?"
Copilot: "È mais in fase di maturazione fisiologica (picco NDVI 0.87 in ago-2025,
          discesa a 0.54 in ott-2025). Superficie: 42.3 ha. AlphaEarth embedding
          cluster: 'row crop, temperate, irrigated'."

Usuario: "E quali zone mostrano stress idrico?"
Copilot: "Zona nord-est (8.1 ha). NDWI è sceso da 0.31 a 0.09 tra il 15-set e
          il 10-ott. Sentinel-1 SAR conferma riduzione umidità superficiale.
          NVIDIA Earth-2 prevede siccità persistente nei prossimi 7 giorni."
```

### Capacidades del Copiloto

El sistema responde con trazabilidad completa (citación de fuentes, tool calls transparentes gracias al tracing built-in de Google ADK, caching de resultados intermedios) sobre:

- **Clasificación de cultivos** con segmentación semántica densa sobre AlphaEarth + VLM Gemma 4 fine-tuneado.
- **Cuantificación de superficies** en hectáreas mediante intersección con shapefiles administrativos GSAA/LPIS italianos.
- **Vigor vegetativo** (NDVI, EVI, GCVI) y salud de la planta.
- **Estrés hídrico** (NDWI, NDMI) con fusión Sentinel-1 SAR y pronóstico NVIDIA Earth-2.
- **Fenología** con detección automática de fases (emergencia, desarrollo, madurez, senescencia).
- **Detección de anomalías** y cambios temporales mediante ChangeDINO.
- **Razonamiento cross-parcela** con Spatial-RAG híbrido sobre PostGIS + pgvector.

### Alineación con el Curso

El plan respeta las diez semanas efectivas del curso con entregables que cumplen las rúbricas CRISP-ML(Q) del Proyecto Integrador (Avance 0 el 26 de abril hasta Presentación Final el 21 de junio de 2026). La redacción del Paper Track se ejecuta como actividad opcional en las semanas 10-11 post-presentación (22 de junio al 3 de julio), sin comprometer los entregables evaluados.

### Método Central: Segmentación Semántica Densa

El proyecto se articula alrededor de la **segmentación semántica densa píxel-por-píxel** como método nuclear que habilita la precisión en la cuantificación de superficies de cultivo. A diferencia de clasificadores que operan a nivel de parcela completa o que producen bounding boxes aproximados, la segmentación semántica etiqueta cada píxel individual de 10m × 10m con la clase de cultivo correspondiente. Esto permite cuantificación exacta en hectáreas mediante conteo de píxeles positivos multiplicado por el área unitaria (100 m² por píxel Sentinel-2), con error sistemático menor al 1% cuando la segmentación alcanza mIoU ≥ 0.70.

El pipeline jerárquico se despliega en tres niveles complementarios:

- **Nivel 1 (EPIC 4):** segmentación píxel a nivel tabular con Random Forest + XGBoost sobre features combinados (AlphaEarth 64-dim + 17 índices + DINOv3). Baseline interpretable, meta F1-macro ≥ 0.60.
- **Nivel 2 (EPIC 5):** seis arquitecturas profundas (U-Net, DeepLabv3+, SegFormer, U-TAE, TSViT, Swin-UNETR) sobre patches Sentinel-2 con labels PASTIS-R. Métricas mIoU, F1-macro, pixel accuracy.
- **Nivel 3 (EPIC 6):** Gemma 4 26B-MoE fine-tuneado con LoRA para generar coordenadas y polígonos dentro de respuestas en lenguaje natural. Un adaptador rasteriza la respuesta textual a máscara binaria.

Cuatro ensambles (EPIC 6) combinan las salidas: voting homogéneo sobre top-3 temporales, bagging sobre XGB+AlphaEarth, stacking heterogéneo con Gemma 4 como base learner, y blending con pesos optimizados por Optuna. El ensamble final reduce el error espacial respecto al mejor modelo individual y alimenta las respuestas de cuantificación del agente. La precisión resultante habilita la veracidad de todas las respuestas del copiloto: el LLM no inventa los números, los lee del mapa segmentado.

---

## 2. Propuesta de Valor y Diferenciadores {#2-propuesta-de-valor}

### Posicionamiento frente a alternativas actuales

| Categoría | Google Earth AI (2025) | QGIS / ArcGIS clásico | **AgroSatCopilot** |
|-----------|------------------------|-----------------------|------------------------|
| Backbone de Earth Observation | AlphaEarth (cerrado, Google Cloud) | No aplica (manual) | **AlphaEarth pre-computado GEE + DINOv3-satellite frozen** |
| Vision-Language Model | Remote Sensing FM Google | No | **Gemma 4 26B-MoE fine-tune agrícola (Apache 2.0)** |
| Orquestador LLM | Gemini 3.1 Pro closed | No | **Gemini 3.1 Pro + Qwen3.5-35B-A3B con switch A/B** |
| Framework de agente | Interno Google | No | **Google ADK + Vertex AI Agent Engine** |
| Idioma nativo | Inglés predominante | Plug-ins de terceros | **Italiano + Español + Inglés** |
| Despliegue | Google Cloud only | Desktop / servidor propio | **GCP primario + Azure H100 puntual + on-premise** |
| Foco de dominio | Planeta genérico | Agnóstico | **Mediterráneo / Italia-centric** |
| Licencia | Propietaria Google | Open Source con costo de analistas expertos | **Apache 2.0 / MIT / DINOv3 License** |
| Weather integration | No nativa | Plug-ins | **NVIDIA Earth-2 nowcasting + ERA5** |
| RAG | Gemini nativa interna | No | **Spatial-RAG híbrido PostGIS + pgvector** |
| Benchmark agrícola IT/ES | No existe | No | **AgroMind-IT/ES, contribución original del equipo** |
| Drift monitoring | No expuesto | No | **Evidently AI con reportes HTML semanales** |
| Costo operativo | Alto (Gemini API + Cloud) | Licencias + personal | **~$115 USD/mes con scale-to-zero** |

### Diferenciadores técnicos concretos

1. **Uso inteligente del trabajo de Google DeepMind.** Aprovechamos los petabytes ya procesados para AlphaEarth (gratis vía GEE) en lugar de entrenar un FM propio, liberando las ventanas H100 para fine-tuning de VLM y LLM donde el aporte de dominio agrícola mediterráneo tiene impacto real.

2. **Gemma 4 26B-MoE como VLM principal.** Liberado el 2 de abril de 2026 bajo Apache 2.0 con backing de Google DeepMind, multimodal nativo imagen+video+audio, 140 idiomas incluyendo italiano y español, ranking #3 en Arena open-model leaderboard. Con 4B activos cabe con margen en una sola H100 NVL 96GB para fine-tuning LoRA en BF16.

3. **Doble variante operativa de LLM orquestador con switch A/B visible.** Gemini 3.1 Pro como variante cloud de alta disponibilidad y Qwen3.5-35B-A3B como variante on-premise open-source. La UI permite cambiar la variante en vivo durante la demo.

4. **Google ADK como framework del agente.** Tracing built-in elimina trabajo custom de observabilidad del agente, deploy nativo a Vertex AI Agent Engine simplifica producción, soporte de backends OpenAI-compatible permite usar Qwen3.5-35B-A3B vía vLLM sin adaptadores.

5. **DINOv3-satellite frozen** (`facebook/dinov3-vitl16-pretrain-sat493m`, Meta 2025) como extractor self-supervised de features de vigor, LAI y canopy height. Corre en L4 sin reentrenamiento, reduce error de canopy de 4.1 m a 1.2 m según los estudios publicados.

6. **Benchmark bilingüe AgroMind-IT/ES.** 500 pares Q&A en italiano y español construidos por el equipo y validados con reviewer nativo italiano (Scuola Sant'Anna vía sponsor). Publicable con DOI en Zenodo durante el Paper Track opcional.

7. **Stack 100% open-source y reproducible** que garantiza soberanía de datos para cooperativas agrícolas italianas sujetas a las normativas europeas de protección de datos agroalimentarios.

8. **Dagster asset-oriented** encaja perfectamente con el flujo DVC + MLflow del proyecto: cada dataset, feature y modelo es un asset con lineage declarativo visible en la UI, permitiendo auditoría de "este modelo depende de este feature que depende de este raster".

---

## 3. Antecedentes Académicos {#3-antecedentes-academicos}

Cuatro artículos científicos delimitan el espacio conceptual del proyecto. Dos provistos por el patrocinador (TSViT y Phenology Description) fundamentan el pipeline de modelado temporal y de generación fenológica; dos adicionales publicados en noviembre de 2025 (Be My Eyes y FarSLIP) fundamentan la arquitectura multi-agente del copiloto y la técnica de adaptación CLIP para teledetección fine-grained.

### 3.1 Paper 1 — TSViT: Vision Transformers for Satellite Image Time Series

**Cita:** M. Tarasiou, E. Chavez y S. Zafeiriou, "ViTs for SITS: Vision Transformers for Satellite Image Time Series", arXiv:2301.04944v3, Imperial College London, 2023.

**Aporte nuclear.** Propone un Transformer factorizado en dos encoders secuenciales, uno temporal y otro espacial, invirtiendo el orden convencional de los Transformers para video. Introduce cuatro elementos clave:

1. Tokenización por parches tridimensionales (t=1, h, w) con convolución 2D paralela aplicada a cada imagen.
2. Tabla de positional encoding indexada por el tiempo real de adquisición de cada escena (crítica porque Sentinel-2 tiene revisitas irregulares afectadas por nubosidad).
3. Múltiples cls tokens, uno por clase de cultivo, separables explícitamente entre encoders.
4. Dos heads de salida, uno para clasificación global y otro para segmentación densa píxel-por-píxel.

Alcanza el estado del arte sobre el benchmark PASTIS, superando U-TAE, ConvLSTM y arquitecturas 3D-CNN.

**Aplicación en AgroSatCopilot.** TSViT forma parte del EPIC 5 como uno de los seis modelos comparativos requeridos por la rúbrica del Avance 4. Su encoder temporal-espacial factorizado sirve además como punto de referencia conceptual en el Apéndice para la decisión de arquitectura del pipeline temporal.

### 3.2 Paper 2 — Phenology Description is All You Need!

**Cita:** S. Wen et al., "Phenology Description is All You Need! Mapping Unknown Crop Types with Remote Sensing Time-Series and LLM Generated Text Alignment", ISPRS Journal of Photogrammetry and Remote Sensing, vol. 228, 2025.

**Aporte nuclear.** Combina cuatro componentes en una arquitectura zero-shot para clasificación de cultivos desconocidos:

1. CLIP como encoder visual de curvas NDVI más un ViT espacial entrenado desde cero.
2. Generación automática de descripciones fenológicas mediante GPT-4 con prompt engineering estructurado en tres capas (instrucciones generales, instrucciones sobre series temporales, instrucciones restrictivas con contexto geográfico).
3. Graph Convolutional Network sobre keywords fenológicos extraídos (pico, declive, senescencia) para capturar dependencias topológicas entre fases.
4. Aprendizaje contrastivo con LoRA para fine-tuning eficiente del encoder textual.

Demuestra transferencia de California a Kansas y Rumania sin reentrenamiento, una propiedad rara en clasificación supervisada de cultivos.

**Aplicación en AgroSatCopilot.** El patrón de prompt engineering de tres capas se replica en el EPIC 7 dentro del tool `phenology_descriptor`, sustituyendo GPT-4 por Gemma 4 26B-MoE fine-tuneado y por Gemini 3.1 Pro. El concepto de alineación visual-textual contrastiva fundamenta la decisión de fine-tunear Gemma 4 sobre AgroMind en lugar de usarlo fuera de dominio.

### 3.3 Paper 3 — Be My Eyes: Extending Large Language Models to New Modalities Through Multi-Agent Collaboration

**Cita:** J. Y. Huang, S. Zhang, Q. Liu, G. Qin, T. Zhu, T. Naumann, M. Chen y H. Poon, "Be My Eyes: Extending Large Language Models to New Modalities Through Multi-Agent Collaboration", arXiv:2511.19417, 2025.

**Aporte nuclear.** Plantea un marco multi-agente totalmente open-source donde dos agentes especializados colaboran vía conversación estructurada:

1. **Agente perceiver** — un VLM ligero (Qwen2.5-VL-7B en el paper original) responsable únicamente de extraer evidencia visual relevante de la imagen y emitirla como descripciones textuales estructuradas.
2. **Agente reasoner** — un LLM text-only potente y especializado en razonamiento (DeepSeek-R1 en el paper original) que consume las descripciones del perceiver, planifica, invoca herramientas si las hay, y sintetiza la respuesta final.
3. **Pipeline de síntesis de datos** que genera trazas de diálogo perceiver↔reasoner y entrena al perceiver con supervised fine-tuning para optimizar la calidad de la información visual que pasa al reasoner.

Demuestra que el sistema (Qwen2.5-VL-7B + DeepSeek-R1) supera a VLMs propietarios unificados como GPT-4o en tareas knowledge-intensive de razonamiento visual, manteniendo el costo de entrenamiento confinado a un perceiver compacto.

**Aplicación en AgroSatCopilot.** El patrón perceiver–reasoner se traslada literalmente al EPIC 7 del proyecto. **Gemma 4 26B-MoE** fine-tuneado con LoRA sobre AgroMind + AgroMind-IT/ES actúa como perceiver agrícola (Qwen3-VL-30B-A3B en la rama comparativa); **Qwen3.5-35B-A3B** on-premise vía vLLM o **Gemini 3.1 Pro** vía Vertex AI actúan como reasoner orquestador plan-and-react que invoca las nueve tools geoespaciales. El desacoplamiento percepción/razonamiento es lo que permite que el switch A/B de la UI cambie de variante sin reentrenar el perceiver y que el costo de fine-tuning se mantenga acotado a las ventanas H100 disponibles. El pipeline de síntesis de trazas inspira la generación de ejemplos de tool calls del agente para los benchmarks AgroMind-IT/ES.

### 3.4 Paper 4 — FarSLIP: Discovering Effective CLIP Adaptation for Fine-Grained Remote Sensing Understanding

**Cita:** Z. Li, W. Yu, D. Muhtar, X. Zhang, P. Xiao, P. Ghamisi y X. X. Zhu, "FarSLIP: Discovering Effective CLIP Adaptation for Fine-Grained Remote Sensing Understanding", arXiv:2511.14901, 2025.

**Aporte nuclear.** Introduce una técnica de adaptación de CLIP específicamente diseñada para teledetección de alta granularidad espacial, articulada en tres elementos:

1. **Destilación parche-a-parche (patch-to-patch distillation)** que alinea señales visuales locales y globales, sustituyendo el esquema convencional de destilación patch-to-CLS y preservando la coherencia semántica del modelo.
2. **Alineación región-categoría basada en el token CLS** que utiliza supervisión a nivel de objeto sin necesidad de matching explícito a nivel de parche, manteniendo la robustez de CLIP.
3. **Dataset MGRS-200k** — primera colección multi-granularidad imagen-texto para remote sensing con supervisión textual a nivel de objeto.

Reporta estado del arte en open-vocabulary semantic segmentation, zero-shot classification y retrieval imagen-texto para teledetección.

**Aplicación en AgroSatCopilot.** FarSLIP se implementa en el proyecto como técnica activa de feature extraction y refinamiento espacial:

- **EPIC 3 — Feature Engineering (US-016).** Destilación parche-a-parche de un CLIP fine-tuneado sobre crops Sentinel-2 (256×256 px) en las tres regiones italianas. Las features destiladas se anexan a la Familia 1 como banco adicional de embeddings espaciales fine-grained de 512-dim que complementan AlphaEarth (semántica global) y DINOv3 (estructura vegetal).
- **EPIC 5 — Modelos Alternativos (US-024 SegFormer-B2).** Cabezal open-vocabulary con alineación región-categoría sobre token CLS, que permite a SegFormer producir máscaras semánticas sub-parcela alineadas a categorías de cultivo descritas en lenguaje natural (italiano/español/inglés).
- **EPIC 6 — Modelo Final.** La salida open-vocabulary FarSLIP entra como uno de los base learners del stacking heterogéneo, complementando a Gemma 4, U-TAE, TSViT, Swin-UNETR y XGB-AlphaEarth.

Esta técnica reduce el margen de error sistemático en la cuantificación de superficies en hectáreas y mejora la consistencia espacial de las fronteras de cultivo, dos atributos críticos para que el copiloto reporte cifras trazables.

---

## 4. Estado del Arte 2025-2026 {#4-estado-del-arte}

La revisión bibliográfica abarca treinta fuentes publicadas en los años 2025 y 2026 con dos excepciones de 2023 y 2024 que siguen siendo canónicas. Se organizan en seis familias temáticas.

### 4.1 Foundation Models y Embeddings Pre-computados para Earth Observation

| # | Referencia | Año | Uso en AgroSatCopilot |
|---|-----------|-----|----------------------|
| 1 | **AlphaEarth Foundations v2.1** — Google DeepMind, Satellite Embedding V1 Annual | 2024-2025 | **Backbone principal** del pipeline de features |
| 2 | Google Earth AI, arXiv:2510.18318 | 2025 | **Referencia industrial** a diferenciar |
| 3 | TerraFM, arXiv:2506.06281 | 2025 | Referencia comparativa de FM multisensor |
| 4 | Foundation Models for Remote Sensing Survey, arXiv:2410.16602 | 2024-25 | Referencia metodológica |
| 5 | DOFA-CLIP, arXiv:2503.06312 | 2025 | Referencia de VLM geoespacial |
| 6 | HighFM, arXiv:2604.04306 | 2025 | Trabajo futuro: integración geoestacionaria |

### 4.2 Vision-Language Models para Imágenes Satelitales y Agricultura

| # | Referencia | Año | Uso en AgroSatCopilot |
|---|-----------|-----|----------------------|
| 7 | **Gemma 4** — Google DeepMind, HuggingFace `google/gemma-4-*` | abr-2026 | **VLM principal** del EPIC 6. Fine-tune LoRA rank 16 sobre variante 26B-MoE |
| 8 | **Qwen3-VL-30B-A3B-Instruct** — HuggingFace | nov-2025 | **VLM comparativo** del EPIC 6 |
| 9 | GeoChat, arXiv:2311.15826 | 2023-25 | Baseline comparativo en tabla Avance 4 |
| 10 | TEOChat, arXiv:2410.06234 | 2024-25 | Baseline temporal comparativo |
| 11 | VLM meets RS Survey, arXiv:2505.14361 | 2025 | Referencia de métricas de evaluación |
| 12 | GeoGround, arXiv:2411.11904 | 2024-25 | Patrón de grounding espacial usado en `alphaearth_query` |
| 13 | SkyMoE, arXiv:2512.02517 | 2025 | Referencia eficiencia MoE |
| 13b | **FarSLIP, arXiv:2511.14901** (Li et al.) | nov-2025 | **Implementado** en EPIC 3 US-016 (destilación parche-a-parche) y EPIC 5 US-024 (cabezal open-vocabulary sobre SegFormer-B2) |
| 13c | **Be My Eyes, arXiv:2511.19417** (Huang et al.) | nov-2025 | **Patrón arquitectónico** del EPIC 7: desacopla perceiver (Gemma 4 / Qwen3-VL) y reasoner (Qwen3.5 / Gemini 3.1 Pro) |

### 4.3 Agentes LLM, Razonamiento Geoespacial y RAG Espacial

| # | Referencia | Año | Uso en AgroSatCopilot |
|---|-----------|-----|----------------------|
| 14 | **Qwen3.5-35B-A3B** — Alibaba Qwen Team (MoE 35B/3B, 128K, Apache 2.0) | abr-2026 | **Orquestador open-source principal** del EPIC 7 |
| 15 | **Gemini 3.1 Pro** — Google Vertex AI | 2026 | **Orquestador cloud de referencia** del EPIC 7 |
| 16 | **Google ADK (Agent Development Kit)** | 2026 | **Framework del agente**|
| 17 | MiniMax-M2.7, HuggingFace | 2026 | Referencia de LLM MoE de gran escala (~230B totales) |
| 18 | Kimi K2.6, arXiv:2507.20534 | 2025 | Referencia de LLM MoE de gran escala (~1T totales) |
| 19 | Spatial-Agent, arXiv:2601.16965 | 2026 | Patrón plan-and-react con GeoFlow Graphs aplicado en ADK |
| 20 | GeoAgentic-RAG Multi-Agent PostGIS+raster | 2025 | Patrón multi-agente con 85.3% pass rate |
| 21 | Spatial-RAG, arXiv:2502.18470 | 2026 | **Implementación directa** en el RAG del agente |
| 22 | GeoAnalystBench, arXiv:2509.05881 | 2025 | **Benchmark** de evaluación del agente |
| 23 | GeoBenchX, arXiv:2503.18129 | 2025 | Segundo benchmark para triangulación |
| 24 | GeoAgentBench, arXiv:2604.13888 | 2026 | Tercer benchmark (opcional Paper Track) |
| 25 | LLM Agent for Geospatial Analysis, arXiv:2410.18792 | 2024-25 | Inspiración del `rasterio_tool` |

### 4.4 Segmentación Semántica Temporal de Cultivos

| # | Referencia | Año | Uso en AgroSatCopilot |
|---|-----------|-----|----------------------|
| 26 | AgriFM, arXiv:2505.21357 | 2025-26 | Referencia de FM agrícola comparativo |
| 27 | Hierarchical Crop EnMAP + S2, arXiv:2506.06155 | 2025 | Referencia para features hiperespectrales (futuro) |
| 28 | **Swin-UNETR para Crop Seg SITS, arXiv:2412.01944** | 2024-25 | **Implementado** en EPIC 5 como modelo 6 |
| 29 | ViTs in Precision Agriculture Survey, arXiv:2504.21706 | 2025 | Referencia metodológica para selección de arquitecturas |

### 4.5 Feature Extraction Self-Supervised y Detección de Cambios

| # | Referencia | Año | Uso en AgroSatCopilot |
|---|-----------|-----|----------------------|
| 30 | **Meta DINOv3 + Satellite Backbone** (`facebook/dinov3-vitl16-pretrain-sat493m`) | ago-2025 | **Feature extractor frozen** del EPIC 3 |
| 31 | ChangeDINO, arXiv:2511.16322 | 2025 | Arquitectura transferida a detección de anomalías en EPIC 7 |
| 32 | RS2-SAM2, arXiv:2503.07266 | 2025 | Referencia de segmentación promptable (trabajo futuro) |

### 4.6 Benchmarks, Clima y Orquestación

| # | Referencia | Año | Uso en AgroSatCopilot |
|---|-----------|-----|----------------------|
| 33 | **GEO-Bench-2** — IBM/ServiceNow/NASA/ESA, arXiv:2511.15658 | 2025 | **Benchmark formal** del Paper Track |
| 34 | **AgroMind Benchmark, arXiv:2505.12207** | 2025 | 28,482 QA pairs; subset 1000 como eval EPIC 7 |
| 35 | **NVIDIA Earth-2 Open Models** | 2026 | **Tool del agente** para pronóstico climático |
| 36 | **Dagster 1.9+** — Asset-oriented orchestration | 2026 | Orquestador de flujos |
| 37 | **Polars 1.x** — Fast DataFrame library Rust | 2025-26 | Motor principal de DataFrames analíticos |
| 38 | **dbmate** — Database migration tool | 2026 | Migraciones SQL puras framework-agnósticas |

### 4.7 Insights que impactan el diseño

1. Los Foundation Models pre-computados públicos (AlphaEarth v2.1) hacen innecesario entrenar un FM propio para un proyecto académico. El aporte se concentra en el fine-tuning de la capa VLM y la capa de orquestación.

2. Las arquitecturas plan-and-react superan a ReAct monolítico en razonamiento multi-paso geoespacial según GeoAgentBench 2026.

3. El fine-tuning con LoRA o QLoRA es el estándar industrial 2026 para adaptar modelos de decenas a cientos de miles de millones de parámetros en GPUs acotadas (L4 24GB y H100 96GB).

4. Los benchmarks agrícolas específicos (AgroMind con 28,482 QA pairs) existen en inglés. El vacío en italiano y español es una oportunidad publicable.

4b. El **patrón perceiver–reasoner** propuesto por Be My Eyes (Huang et al., 2025) permite combinar un VLM compacto fine-tuneado en dominio (Gemma 4 26B-MoE) con un LLM razonador potente (Qwen3.5-35B-A3B o Gemini 3.1 Pro) y superar a VLMs propietarios unificados como GPT-4o, manteniendo el costo de fine-tuning acotado y habilitando el switch A/B del orquestador sin reentrenar la percepción.

4c. La **adaptación CLIP con destilación parche-a-parche** propuesta por FarSLIP (Li et al., 2025) supera el dilema clásico entre coherencia semántica global (CLIP estándar) y discriminabilidad espacial local (fine-tunes naive), produciendo features fine-grained críticos para cuantificación de superficies sub-parcela en agricultura mediterránea.

5. Spatial-RAG híbrido (filtrado espacial PostGIS + similitud semántica pgvector) reduce la tasa de alucinación del agente aproximadamente 30 por ciento frente a RAG textual puro.

6. Los VLM nativos multilingües (Gemma 4 con 140+ idiomas, Qwen3-VL con 201) sustituyen a los VLM especializados mono-idioma (TEOChat, GeoChat) en escenarios donde el usuario final no habla inglés.

7. Google ADK simplifica el deployment productivo de agentes a Vertex AI Agent Engine con observabilidad built-in, reduciendo el esfuerzo de orquestación manual para equipos ya en Google Cloud.

8. Los pipelines orquestados con Dagster exhiben lineage declarativo entre datasets, features y modelos, facilitando la auditoría y la reproducibilidad exigida por el curso.

---

## 5. Google Earth AI: Referencia Industrial a Diferenciar {#5-earth-ai}

Google DeepMind lanzó en 2025 el producto Google Earth AI, que combina Foundation Models de Remote Sensing, AlphaEarth Foundations y un agente de razonamiento geoespacial impulsado por Gemini. La arquitectura conceptual coincide con AgroSatCopilot pero el posicionamiento de producto es divergente.

### 5.1 Componentes de Google Earth AI

1. **Remote Sensing Foundation Models:** VLM, detección open-vocabulary y backbones adaptables.
2. **Geospatial Reasoning Agent:** impulsado por Gemini 3.1 Pro, descompone consultas complejas en planes multi-paso, invoca foundation models y herramientas geoespaciales, fusiona resultados.
3. **AlphaEarth Foundations:** embeddings anuales 64-dim por píxel (también base de AgroSatCopilot).
4. **Cobertura:** modelos para imaginería planetaria, población y ambiente.
5. **Partners industriales:** Planet Labs, Airbus, FAO, Harvard Forest, Stanford, MapBiomas (más de 50 organizaciones trusted tester).
6. **Integración en producto:** Gemini en Google Earth, Google Maps Platform y Google Cloud.
7. **Resultado reportado:** +64% sobre Gemini 2.5 Pro baseline (+37% descriptivas, +124% analítico-relacionales).

### 5.2 Limitaciones operativas del producto comercial

| Limitación | Implicación |
|------------|-------------|
| Licencia propietaria Google | No desplegable on-premise por organizaciones que no pueden exportar datos a Google Cloud |
| Dependencia de Gemini closed | El usuario no puede auditar ni modificar el modelo |
| Inglés predominante | No sirve como interfaz nativa para agrónomos italianos o hispanohablantes |
| Foco planetario genérico | Sin especialización en cultivos mediterráneos ni benchmark agrícola propio |
| Sin benchmark público | La comparación cuantitativa con alternativas requiere evaluación externa |

### 5.3 Nicho que aborda AgroSatCopilot

AgroSatCopilot replica el patrón arquitectónico validado por Google Earth AI (FM de EO + agente + tools) pero cubre tres vacíos: (1) stack 100% open-source con alternativa Qwen3.5-35B-A3B al Gemini propietario y switch A/B visible; (2) especialización en agricultura mediterránea italiana con soporte nativo italiano/español en VLM (Gemma 4) y frontend (Nuxt 4 SSR con `@nuxtjs/i18n`); (3) benchmark público bilingüe AgroMind-IT/ES construido y publicado en Zenodo por el equipo.

---

## 6. Stack Tecnológico {#6-stack-tecnologico}

### 6.1 Backend y Persistencia

| Componente | Tecnología | Justificación |
|------------|-----------|---------------|
| API REST | FastAPI (Python 3.12) | Asíncrono, tipado con Pydantic v2, dominado por el equipo |
| Tiling GIS | TiTiler (FastAPI + rio-tiler) | Servidor COG dinámico para overlays NDVI en el mapa |
| ORM | SQLModel (SQLAlchemy 2.0 + Pydantic) | Tipado end-to-end |
| Base de datos transaccional | PostgreSQL 15 + PostGIS + pgvector | Geometrías, embeddings y RLS multi-tenant |
| Motor de DataFrames analíticos | **Polars 1.x** | 5-10× más rápido que pandas, ergonomía mejor, ideal para fusión multisensor |
| Motor SQL analítico (opcional) | DuckDB en notebooks de exploración | Queries ad-hoc sobre Parquet |
| Migraciones de esquema | **dbmate** | SQL puro, framework-agnóstico, preferencia del equipo |
| Colas asíncronas | Cloud Pub/Sub + Cloud Tasks | Jobs de inferencia desacoplados (US-040) |
| Caché | Redis Memorystore | Cache de tiles, sesiones, embeddings AlphaEarth |
| Object storage | GCS (primario) + Azure Blob (checkpoints H100) | Tiles COG, datasets, pesos LoRA |
| Dependency manager | Poetry | Lockfile determinístico |
| Containerización | Docker + Docker Compose | Paridad desarrollo-producción |

### 6.2 Pipeline de Machine Learning

| Componente | Tecnología | Justificación |
|------------|-----------|---------------|
| Framework DL | PyTorch 2.4 con `torch.compile` | Compatibilidad con todos los modelos seleccionados |
| Fine-tuning eficiente | HuggingFace `transformers` + `peft` (LoRA) | Adaptación de Gemma 4 26B-MoE y Qwen3-VL dentro de ventanas H100 |
| Entrenamiento distribuido single-GPU | FSDP + FlashAttention-2 + gradient checkpointing | Optimizar VRAM en 1×H100 96GB |
| Baselines tabulares | XGBoost 2.1 + LightGBM + scikit-learn | Sobre AlphaEarth embeddings 64-dim |
| Segmentación CNN | `segmentation_models.pytorch` | U-Net, DeepLabv3+ |
| Transformers temporales | U-TAE (VSainteuf/utae-paps), TSViT (paper del profesor) | Modelos comparativos del EPIC 5 |
| Foundation Model de EO | AlphaEarth Foundations v2.1 vía GEE | Backbone principal (no se entrena) |
| Feature extractor frozen | **DINOv3-satellite `facebook/dinov3-vitl16-pretrain-sat493m`** | Vigor, LAI, canopy height sin reentrenamiento |
| VLM principal | **Gemma 4 26B-MoE** (`google/gemma-4-26b-it`) | Apache 2.0, multimodal imagen+video+audio, 140 idiomas |
| VLM comparativo | **Qwen3-VL-30B-A3B-Instruct** (`Qwen/Qwen3-VL-30B-A3B-Instruct`) | Contraste con Gemma 4 |
| VLM fallback local | **Gemma 4 E4B** (`google/gemma-4-e4b-it`) | Corre en L4 24GB o laptop 4060-4080 |
| LLM orquestador cloud | Gemini 3.1 Pro vía Vertex AI | Alta disponibilidad, calidad superior |
| LLM orquestador on-premise | **Qwen3.5-35B-A3B** (`Qwen/Qwen3.5-35B-A3B`) | Soberanía; cabe en H100 96GB BF16 con margen |
| Framework del agente | **Google ADK** (Agent Development Kit) | Deploy nativo Vertex AI Agent Engine, tracing built-in, soporta Gemini + vLLM |
| Serving LLM | vLLM con continuous batching | Máximo throughput en H100 96GB |
| Evaluación | LM-Eval-Harness, DeepEval, GEO-Bench-2, AgroMind | Benchmarks estándar y propio |

### 6.3 Capa Geoespacial

| Componente | Tecnología | Justificación |
|------------|-----------|---------------|
| Acceso AlphaEarth | Google Earth Engine Python API (`ee`) | Extracción de embeddings 64-dim |
| Procesamiento raster | rasterio + xarray + rioxarray | Estándar de facto |
| Procesamiento vector | geopandas + Shapely | Intersecciones GSAA |
| Índices espectrales | `eemont` (GEE) + `spyndex` | 200+ índices documentados |
| Descarga Sentinel-1/2 | Copernicus Data Space Ecosystem (CDSE) + `sentinelhub-py` + GEE | Redundancia |
| Pre-procesamiento SAR | SNAP-py + pyroSAR | Sentinel-1 GRD → Gamma-0 |
| Máscara de nubes | s2cloudless | Modelo probado |
| Catálogo STAC | `pystac-client` + `pgstac` | Metadata queryable |
| Weather / nowcasting | NVIDIA Earth-2 API + ERA5 (CDS) | Pronóstico climático |

### 6.4 Frontend

| Componente | Tecnología | Justificación |
|------------|-----------|---------------|
| Framework | **Nuxt 4 SSR puro** (Vue 3 Composition API) | Web app, sin PWA ni desktop, reduce complejidad |
| Mapa interactivo | MapLibre GL JS + deck.gl | OSS, sin Mapbox |
| GeoJSON editor | `maplibre-gl-draw` wrapper Vue | Dibujo de polígonos reactivos |
| Chat UI | `@ai-sdk/vue` + Nuxt UI Pro | Streaming con composable `useChat()` |
| Gráficas | Apache ECharts (via `vue-echarts`) | Series temporales NDVI/NDWI |
| Estado global | Pinia + pinia-plugin-persistedstate | Reactivo y persistente |
| Styling | TailwindCSS v4 + Nuxt UI Pro | Design system |
| i18n | `@nuxtjs/i18n` | Italiano, español, inglés con rutas localizadas |
| Auth | Clerk Nuxt module (free tier) | OAuth Google/Microsoft |

### 6.5 MLOps y DevOps

| Componente | Tecnología | Justificación |
|------------|-----------|---------------|
| Versionado de datos | DVC 3.48+ con remote GCS | Ya dominado |
| Experiment tracking | MLflow 2.16 | Ya dominado, integra con Dagster vía `dagster-mlflow` |
| Orquestación | **Dagster 1.9+** | Asset-oriented, lineage declarativo, mejor para ML que Prefect |
| CI/CD | GitHub Actions + Cloud Build | Tests, build, deploy |
| Registro de modelos | MLflow Model Registry + Artifact Registry | Versionado |
| Infraestructura como código | Terraform 1.9+ | Reproducibilidad GCP + Azure H100 |
| Monitoreo drift | **Evidently AI** (US-047) | Drift en bandas Sentinel-2 y AlphaEarth embeddings |
| Observabilidad agente | **Google ADK tracing built-in** + Cloud Monitoring | Tool calls, latencia, errores |
| Observabilidad API | Prometheus + Grafana | Métricas técnicas |
| Secretos | Secret Manager (GCP) + Key Vault (Azure) | API keys seguras |
| Testing | pytest + pytest-asyncio + Playwright | Cobertura ≥70% backend, ≥50% frontend |

---

## 7. Arquitectura de la Solución {#7-arquitectura}

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CLIENTES WEB (bilingüe IT/ES/EN)                         │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐    │
│  │ Agrónomo   │  │ Admin AGEA   │  │Investigador │  │ Demo Presentación│    │
│  └─────┬──────┘  └──────┬───────┘  └──────┬──────┘  └─────────┬────────┘    │
│        │                │                  │                   │             │
│  ┌─────┴────────────────┴──────────────────┴───────────────────┴─────┐       │
│  │ Nuxt 4 SSR (Vue 3 + MapLibre + deck.gl + i18n + switch A/B LLM)   │       │
│  └──────────────────────────┬────────────────────────────────────────┘       │
└─────────────────────────────┼────────────────────────────────────────────────┘
                              │ HTTPS / SSE
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Cloud Load Balancer + Clerk OAuth (OIDC)                     │
└─────────────┬───────────────────────────────────────────────────────────────┘
              ▼
   ┌──────────┴────────┐
   ▼                    ▼
┌──────────────────┐  ┌──────────────────┐
│ FastAPI API      │  │ TiTiler COG      │
│ /chat (SSE)      │  │ /tiles/{z/x/y}   │
│ /aois/{id}/seg   │  └───────┬──────────┘
│ /timeseries      │          │
│ /stac/search     │          ▼
│ /llm/switch      │   GCS COG bucket
└──────┬───────────┘
       │
  ┌────┼─────────────────────────────────┐
  ▼    ▼               ▼                 ▼
┌──────────┐ ┌─────────────────┐ ┌────────────────────────┐
│ PostGIS  │ │ Pub/Sub +       │ │ Cloud Run GPU L4       │
│ + pgvec  │ │ Cloud Tasks     │ │ Inference Worker       │
│ STAC     │ │ ┌─────────────┐ │ │ - DINOv3-satellite     │
│ catalog  │ │ │alphaearth   │ │ │ - XGBoost AlphaEarth   │
│ Spatial  │ │ │ingest-s2    │ │ │ - Gemma 4 26B-MoE LoRA │
│ RAG      │ │ │drift-check  │ │ │ - ChangeDINO           │
│ pgstac   │ │ │(Evidently)  │ │ └──────────────┬─────────┘
└──────────┘ │ └─────────────┘ │                │
             └─────────────────┘                ▼
                               ┌──────────────────────────────────┐
                               │ Google ADK Agent                 │
                               │ Plan-and-React + Spatial-RAG     │
                               │ + Tracing built-in               │
                               │ + Tools:                         │
                               │  • alphaearth_query              │
                               │  • sentinel_search               │
                               │  • rasterio_tool                 │
                               │  • geopandas_intersect           │
                               │  • ndvi_calculator               │
                               │  • timeseries_extractor          │
                               │  • phenology_descriptor          │
                               │  • dinov3_extract                │
                               │  • crop_classifier_tool          │
                               └───────────┬──────────────────────┘
                                           │
                              ┌────────────┴──────────────┐
                              ▼                           ▼
                   ┌─────────────────────┐    ┌─────────────────────────┐
                   │ Variante A (cloud)  │    │ Variante B (on-premise) │
                   │ Gemini 3.1 Pro      │    │ Qwen3.5-35B-A3B         │
                   │ Vertex AI API       │    │ (35B total / 3B activos)│
                   │                     │    │ vLLM en H100 NVL 96GB   │
                   └──────────┬──────────┘    └────────────┬────────────┘
                              │                            │
                              └────────────┬───────────────┘
                                           ▼
                   ┌────────────────────────────────────────────────┐
                   │ Fuentes de datos externas                      │
                   │ ┌──────────────────────┐ ┌──────────────────┐  │
                   │ │ Google Earth Engine  │ │ NVIDIA Earth-2   │  │
                   │ │ - AlphaEarth 64-dim  │ │ - Nowcasting     │  │
                   │ │ - Sentinel-1/2       │ │ - Data assim.    │  │
                   │ │ - Dynamic World      │ └──────────────────┘  │
                   │ └──────────────────────┘                        │
                   └────────────────────────────────────────────────┘
```

### 7.1 Flujos principales

**Flujo de consulta conversacional (end-to-end):**

1. El usuario dibuja un AOI en el mapa MapLibre y envía una consulta en italiano, español o inglés.
2. El frontend Nuxt 4 envía la consulta como SSE a `/chat` en FastAPI, junto con el GeoJSON del AOI y la variante de LLM seleccionada (A Gemini o B Qwen3.5).
3. FastAPI delega al agente Google ADK, que invoca Spatial-RAG para recuperar parcelas similares y docs agronómicos relevantes.
4. El planner (LLM) genera un plan con los tools a invocar.
5. Cada tool ejecuta (descarga AlphaEarth, calcula NDVI, extrae DINOv3, llama clasificador XGBoost, etc.) y devuelve resultados trazables.
6. Para tareas pesadas (clasificación de región completa), el API publica un mensaje Pub/Sub y el Cloud Run GPU L4 worker procesa asíncronamente.
7. El LLM sintetiza la respuesta y la envía al frontend vía SSE con citaciones, tool calls colapsables y gráficas ECharts inline. El tracing de ADK queda persistido para auditoría.

**Flujo de ingesta de datos (una vez por ROI):**

1. Dagster agenda la descarga mensual de AlphaEarth + Sentinel para los 3 ROI italianos como assets declarativos con dependencias.
2. Exports de GEE suben COGs a `gs://agrosat-data/alphaearth/{roi}/{year}.tif`.
3. PostGIS actualiza el catálogo STAC con BBOX, fecha y storage URI.
4. Evidently AI monitorea drift en distribución de bandas y embeddings y publica reporte HTML semanal.

---

## 8. FinOps: Presupuesto de Cómputo y Operación {#8-finops}

### 8.1 Costos únicos de entrenamiento (durante el proyecto)

| Recurso | Uso | Duración | Costo estimado |
|---------|-----|----------|----------------|
| Azure H100 NVL 96GB spot | Gemma 4 26B-MoE LoRA + Qwen3-VL LoRA + Qwen3.5 serving/LoRA + TSViT/U-TAE/Swin-UNETR + eval | 80 h en 6 ventanas nocturnas | $220 (spot a $2.74/h) a $560 (on-demand a $6.98/h) |
| GCP L4 Spot | Baselines, LoRA dev-scale, DINOv3 extraction, Gemma 4 E4B | ~50 h | $14 |
| GCP L4 on-demand | CI/CD smoke tests, jobs ligeros | ~10 h | $7 |
| GCS Standard | AlphaEarth subset + COG Sentinel | 200 GB × 3 meses | $12 |
| Azure Blob Hot | Checkpoints LoRA + modelos base | 150 GB × 3 meses | $9 |
| **Total entrenamiento** | | | **$262 – $602** |

### 8.2 Costos operativos mensuales (tras despliegue)

| Servicio | Configuración | Costo/mes |
|----------|---------------|-----------|
| Cloud Run FastAPI | min=0, 512 MB, 1 vCPU | ~$8 |
| Cloud Run TiTiler | min=0, 512 MB | ~$5 |
| Cloud Run Nuxt 4 SSR | min=0, 256 MB | ~$5 |
| Cloud Run GPU L4 inferencia | min=0, on-demand | ~$15 |
| Cloud SQL PostGIS + pgvector | db-f1-micro, 20 GB | ~$14 |
| Cloud Storage | 250 GB Standard | ~$6 |
| Redis Memorystore Basic 1 GB | | ~$15 |
| Pub/Sub + Cloud Tasks | <10 GB | ~$3 |
| Vertex AI (Gemini 3.1 Pro) | ~500k tokens/mes producción | ~$12 |
| Qwen3.5-35B-A3B self-hosted | Azure spot H100 ~3 h/día cuando activo | ~$30 |
| Secret Manager, CDN | | ~$3 |
| Copernicus CDSE, Google Earth Engine | Gratuito | $0 |
| NVIDIA Earth-2 API | Research tier | $0 – $5 |
| **Total mensual** | | **~$115** |

### 8.3 Estrategia de optimización

Se aplican cuatro estrategias para mantener el presupuesto acotado sin sacrificar funcionalidad:

1. **Scale-to-zero** en todos los Cloud Run (cold start tolerable en contexto académico).
2. **Azure H100 spot** únicamente durante ventanas de entrenamiento nocturnas y durante la sesión de demo, con shutdown automático vía script `make azure-h100-stop` al terminar cada jornada.
3. **LoRA rank 16 en BF16** en todo el fine-tuning para mantener calidad cerca del full fine-tune con fracción del costo de VRAM.
4. **Spot instances L4** para todo desarrollo iterativo, reservando on-demand solo para CI/CD crítico.

---

## 9. Datasets, Modelos y Licenciamiento Legal {#9-datasets}

El proyecto utiliza exclusivamente fuentes abiertas con licencia verificable. Cada fuente se lista con su licencia, portal oficial de descarga y restricciones relevantes.

### 9.1 Datasets

| Recurso | Fuente oficial | Licencia | Restricción | Método descarga |
|---------|---------------|----------|-------------|-----------------|
| **AlphaEarth Foundations Satellite Embedding V1** | Google Earth Engine Data Catalog (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`) + `gs://alphaearth_foundations` (Requester Pays) | GEE Terms of Service + Research Use | Permitida investigación y uso comercial con atribución a Google | `ee.batch.Export.image.toCloudStorage()` desde Python (service account JSON en Secret Manager) |
| **Sentinel-2 L2A** | Copernicus Data Space Ecosystem | Copernicus Open Access | Atribución obligatoria: "Contains modified Copernicus Sentinel data" | API OData y STAC vía `sentinelhub-py` o `pystac-client` |
| **Sentinel-1 GRD IW** | CDSE | Copernicus Open Access | Misma que S-2 | Misma que S-2 |
| **PASTIS-R** | HuggingFace Datasets (INRAE Francia) | CC-BY-SA | Atribución a INRAE; share-alike | `datasets.load_dataset("INRAE/PASTIS-R")` + DVC tracking |
| **Dynamic World** | GEE (`GOOGLE/DYNAMICWORLD/V1`) y WRI | CC-BY-4.0 | Atribución Google + WRI | GEE export to GCS |
| **Sen4AgriNet** | Harokopio University of Athens | CC-BY | Atribución | Portal universitario |
| **GSAA / LPIS Italia** | AGEA — portales regionales | Open Data italiana (CC-BY derivada) | Atribución AGEA y región | GeoPackage desde data.agea.gov.it |
| **ERA5** | Copernicus Climate Data Store | ECMWF Open Data License | Atribución a ECMWF/Copernicus | Python `cdsapi` |
| **SRTM DEM** | NASA/USGS EarthExplorer | Dominio Público NASA | Sin restricciones | `elevation` (Python) |
| **USGS Spectral Library** | USGS Digital Spectral Library | Dominio Público | Sin restricciones | Descarga directa |
| **AgroMind Benchmark** (28,482 QA pairs) | HuggingFace `AgroMind/AgroMind` | CC-BY | Atribución autores | `datasets.load_dataset` |
| **GEO-Bench-2** (Paper Track) | HuggingFace + IBM repo | Mixed open (CC-BY / MIT) | Atribución por subdataset | `huggingface_hub.snapshot_download` |
| **AgroMind-IT/ES** (propio) | Construido por el equipo | CC-BY-4.0 (a publicar Zenodo) | Atribución al equipo AgroSatCopilot | Semilla sintética con Gemini 3.1 Pro, validación humana nativa |

### 9.2 Modelos (IDs verificados en HuggingFace)

| Modelo | ID HuggingFace | Licencia | Uso | Notas |
|--------|---------------|----------|-----|-------|
| **Gemma 4 26B-MoE** | `google/gemma-4-26b-it` | Apache 2.0 | **VLM principal** EPIC 6 | Multimodal img+video+audio, 256K ctx, 140 idiomas, 4B activos |
| **Gemma 4 E4B** | `google/gemma-4-e4b-it` | Apache 2.0 | **Fallback local** | 4.5B efectivos, corre en L4 o laptop |
| **Qwen3-VL-30B-A3B-Instruct** | `Qwen/Qwen3-VL-30B-A3B-Instruct` | Apache 2.0 | **VLM comparativo** | MoE 30B/3B, 256K ctx, publicado nov-2025 |
| **Qwen3.5-35B-A3B** | `Qwen/Qwen3.5-35B-A3B` | Apache 2.0 | **LLM on-premise** | MoE 35B/3B, 128K ctx. **OJO: sin sufijo `-Instruct`** |
| **DINOv3-satellite ViT-L/16** | `facebook/dinov3-vitl16-pretrain-sat493m` | DINOv3 License | **Feature extractor frozen** | Entrenado en 493M imágenes Maxar 0.6m |
| **Gemini 3.1 Pro** | Vertex AI API | Google Cloud ToS | **LLM cloud** | $2/$12 por 1M tokens in/out, 2M ctx |
| **e5-mistral-7b (embeddings)** | `intfloat/e5-mistral-7b-instruct` | MIT | Spatial-RAG pgvector | |
| **U-TAE** | GitHub `VSainteuf/utae-paps` | MIT | Modelo 4 EPIC 5 | |
| **TSViT** | GitHub `michaeltrs/DeepSatModels` | MIT | Modelo 5 EPIC 5 (Paper 1 profesor) | |
| **s2cloudless** | `sentinel-hub/sentinel2-cloud-detector` | MIT | Máscara de nubes | |

### 9.3 Cumplimiento legal y ético

Todas las fuentes utilizadas son verificablemente abiertas. El equipo documentará las atribuciones obligatorias en un archivo `DATA_LICENSE.md` del repositorio y en la sección Acknowledgments del paper opcional. Los datos derivados que se publiquen (AgroMind-IT/ES 500 pares) se licenciarán como CC-BY-4.0 en Zenodo con DOI. No se utilizan datos de productores individuales ni datos personales sujetos a GDPR en el MVP.

**Nota sobre Gemma 4 y DINOv3.** Ambos requieren aceptar términos en HuggingFace antes de descargar. Gemma 4 es Apache 2.0 real sin restricciones comerciales. DINOv3 tiene licencia custom de Meta que permite uso comercial con restricciones específicas documentadas en `docs/licenses/dinov3_license_notes.md`.

---

## 10. Mapa de Épicas y Distribución de Story Points {#10-mapa-de-epicas}

### 10.1 Épicas y presupuesto

La capacidad del equipo es de 360 horas-persona efectivas (3 devs × 12 h/semana × 10 semanas). Con factor de conversión de 2.4 h por story point, la capacidad es de aproximadamente **150 story points** para el MVP.

| # | Épica | Fase CRISP-ML(Q) | Avance | Sprint | SP |
|---|-------|-----------------|--------|--------|-----|
| E0 | Infraestructura, Cookiecutter y MLOps Base (Dagster + dbmate) | 0 — Infra | Avance 0 | S1 | 10 |
| E1 | Ingesta de Datos (AlphaEarth, Sentinel, DINOv3) | 1 — Data Understanding | Avance 0-1 | S1-S2 | 12 |
| E2 | Análisis Exploratorio de Datos (Polars) | 1 — Data Understanding | Avance 1 | S2-S3 | 14 |
| E3 | Feature Engineering + Índices Espectrales + FarSLIP | 2 — Data Preparation | Avance 2 | S3-S4 | 18 |
| E4 | Baseline (AlphaEarth + XGBoost/RF) | 3 — Modeling | Avance 3 | S4-S5 | 10 |
| E5 | Modelos Alternativos (seis arquitecturas, SegFormer con cabezal FarSLIP) | 3 — Modeling | Avance 4 | S5-S6 | 21 |
| E6 | Modelo Final (Gemma 4 26B-MoE LoRA + 4 Ensambles) | 3 — Modeling | Avance 5 | S6 | 20 |
| E7 | Agente Conversacional Google ADK perceiver–reasoner (Gemini + Qwen3.5) | 4 — Evaluation | Avance 5-6 | S7 | 14 |
| E8 | Backend API + Worker Pub/Sub + Tiling | 4 — Deployment | Avance 6 | S7 | 9 |
| E9 | Frontend Web Nuxt 4 SSR + Switch A/B | 4 — Deployment | Avance 6 | S8 | 10 |
| E10 | Observabilidad, Evidently Drift, FinOps, Seguridad | 5 — Monitoring | Avance 6-7 | S8 | 8 |
| **Subtotal MVP (Avances 0 a 7 + Presentación)** | | **Avances 0-7 + Pres** | **9 semanas** | **146** |
| E11 | Paper Track (semanas 10-11 post-presentación, opcional) | Reporting externo | — | S10-S11 | 28 |
| **TOTAL** | **12 Épicas** | | | | **174** |

### 10.2 Distinción MUST-HAVE vs STRETCH

Las épicas E0 a E6 cubren entregables obligatorios del curso (Avances 0-5) y deben completarse íntegramente. Las épicas E7 a E10 construyen la plataforma conversacional end-to-end requerida por los Avances 6 y 7.

**Stretch candidates** (sacrificables en orden de prioridad si hay atrasos):

1. US-045 Switch A/B de LLM en UI (1 SP) — sustituible por dos pestañas del frontend en la demo.
2. US-047 Evidently drift pipeline automatizado (2 SP) — sustituible por análisis textual en el Avance 6.
3. US-040 Worker Pub/Sub inferencia (2 SP) — sustituible por inferencia síncrona con timeout para AOIs pequeños.

El equipo tomó la decisión explícita de **mantener los tres en el MVP** por su valor para la demo de la presentación final, aceptando el riesgo asociado.

### 10.3 Secuenciación semanal

```
S1  (20-26 abr): E0 Setup + pulir Avance 0 PDF → Avance 0 dom 26-abr
S2  (27-abr a 3-may): E1 Ingesta + E2 EDA univariado → Avance 1 dom 3-may
S3  (4-10 may):  E2 completo + arrancar E3 FE
S4  (11-17 may): E3 FE + arrancar E4 Baseline → Avance 2 dom 17-may
S5  (18-24 may): E4 Baseline + E5 modelos 1-3 → Avance 3 mié 20-may, Avance 4 dom 24-may
S6  (25-31 may): E5 modelos 4-6 + ensambles + Gemma 4 LoRA → Avance 5 dom 31-may
S7  (1-7 jun):   E6 VLM fine-tune + E7 agente ADK + E8 backend → Avance 6 dom 7-jun
S8  (8-14 jun):  E9 frontend + E10 observabilidad + Avance 7 → Avance 7 dom 14-jun
S9  (15-21 jun): Pulido final + dry-runs + grabar demo → Presentación dom 21-jun
S10-S11 (22-jun a 3-jul): Buffer + Paper Track opcional
```

---

## EPIC 0: Infraestructura, Cookiecutter y MLOps Base {#epic-0}

**Objetivo.** Establecer la estructura base del proyecto, el entorno reproducible local, la infraestructura mono-cloud GCP más la VM H100 en Azure, y el pipeline MLOps antes de iniciar el trabajo con datos. Permite que los tres desarrolladores tengan paridad absoluta desde el primer commit y que cada experimento quede versionado y trazable.

**Alineado con.** Avance 0 (26 abril 2026) — Propuesta del proyecto.

**Estrategia.** Maximizar la reutilización del stack MLOps del proyecto previo del equipo (DVC, MLflow, GitHub Actions, Terraform) e incorporar Dagster asset-oriented y dbmate para reducir la curva de aprendizaje y el presupuesto de story points.

**Puntos totales de la épica: 10.**

---

### US-001 — Cookiecutter template del monorepo

**Como** equipo de 3 desarrolladores,
- **quiero** un template cookiecutter que genere la estructura completa del monorepo AgroSatCopilot con un único comando,
- **para que** cualquier módulo nuevo se cree de forma consistente y el onboarding de cualquier colaborador externo sea inmediato.

**Criterios de Aceptación:**

- El comando `cookiecutter gh:agrosatcopilot/cookiecutter-agrosat` genera el proyecto completo en menos de dos minutos en macOS, Linux y WSL2.
- El template solicita interactivamente: `project_name`, `gcp_project_id`, `azure_subscription_id`, `region` (por defecto `europe-west1` por proximidad a Italia), `db_name`, `team_lead_email`.
- La estructura de directorios generada es: `backend/`, `frontend/`, `ml/`, `infrastructure/`, `notebooks/`, `data/`, `docs/`, `paper/`, `scripts/`, `.github/workflows/`, `db/migrations/`.
- Incluye `pyproject.toml` con Poetry y grupos `dev`, `test`, `ml`, `geo`, `paper`; `package.json` con pnpm; `Makefile` con comandos estandarizados (`make dev`, `make db-migrate`, `make train-l4`, `make train-h100`); `.env.example` con todas las variables requeridas documentadas.
- Incluye Dockerfiles multi-stage para backend y frontend, `docker-compose.yml` para desarrollo local, `cloudbuild.yaml`, módulos Terraform base para GCP y Azure, `dagster.yaml`, y configuración inicial `dbmate` en `db/migrations/`.

**Tareas técnicas:**

- [ ] Crear repositorio `cookiecutter-agrosat` en GitHub con licencia MIT
- [ ] Implementar templates Jinja2 para todos los archivos de configuración
- [ ] Escribir hook `post_gen_project.py` que ejecuta `poetry install`, `pnpm install` y `git init`
- [ ] Pipeline de validación en GitHub Actions con matrix de sistemas operativos (Ubuntu, macOS)
- [ ] Documentar el uso del template en el README del repositorio

**Estimación:** 2 puntos (~1 día).

---

### US-002 — Entorno Docker Compose multiservicio

**Como** desarrollador del equipo,
- **quiero** un entorno local reproducible levantado con `make dev`,
- **para que** los tres miembros del equipo trabajemos sobre exactamente los mismos componentes y versiones, y para que CI/CD tenga la misma especificación.

**Criterios de Aceptación:**

- El comando `make dev` levanta simultáneamente ocho servicios: FastAPI (puerto 8000), Nuxt 4 (3000), PostgreSQL con PostGIS y pgvector (5432), Redis (6379), TiTiler (8001), MLflow UI (5000), **Dagster UI (3001)** y Ollama local (11434) para pruebas de LLM pequeños (Gemma 4 E4B).
- `poetry install --with dev,test,ml,geo` completa sin conflictos de dependencias (validado con `poetry check`).
- Hot-reload funciona en FastAPI (vía uvicorn `--reload`) y Nuxt 4 (vía Vite HMR) dentro de los contenedores Docker.
- Las variables de entorno se cargan desde `.env.local` con validación Pydantic Settings en startup.
- PostgreSQL ejecuta seed automático la primera vez que se levanta usando **dbmate** (`dbmate up`): tablas base, datos demo de 1 parcela en Toscana.
- Healthchecks configurados en todos los servicios con retries exponenciales.

**Tareas técnicas:**

- [ ] Escribir `docker-compose.yml` con los ocho servicios y red bridge compartida
- [ ] Configurar Dockerfile multi-stage backend con builder (compila wheels) y runtime (slim)
- [ ] Configurar Dockerfile frontend Nuxt 4 con cache de pnpm
- [ ] Migración inicial `db/migrations/001_initial_schema.sql` con tablas base y parcela demo
- [ ] Documentar troubleshooting común (puerto ocupado, rate limit Docker Hub)

**Estimación:** 2 puntos (~1 día).

---

### US-003 — Infraestructura GCP + Azure H100 con Terraform

**Como** MLOps Engineer,
- **quiero** la infraestructura declarada en Terraform para GCP primario y la VM H100 en Azure,
- **para que** el entorno de staging y producción sea reproducible y para que encender o apagar la VM H100 sea trivial.

**Criterios de Aceptación:**

- Módulo `terraform/gcp/` provisiona: Cloud Run services (api, frontend, tiling, inference-worker), Cloud SQL PostgreSQL 15 con extensiones PostGIS y pgvector, GCS buckets (data, artifacts, dvc-remote), Cloud Pub/Sub topics (`inference-jobs`, `inference-results`), Secret Manager con 6 secretos base, Artifact Registry para imágenes Docker, Cloud CDN, IAM roles mínimos necesarios.
- Módulo `terraform/azure/` provisiona: VM `Standard_NC40ads_H100_v5` con H100 NVL 96GB on-demand + variante spot, Azure Blob Storage Hot, VNet privada, NSG que sólo permite SSH desde IPs de los 3 devs.
- Workspaces de Terraform separados: `dev`, `staging`, `prod`.
- Scripts `make azure-h100-start` y `make azure-h100-stop` automatizan el encendido/apagado de la VM H100 con timer de auto-shutdown configurable (por defecto 12 h).
- `terraform plan` y `terraform apply` ejecutan desde el pipeline Cloud Build con back-end de estado GCS versionado.

**Tareas técnicas:**

- [ ] Escribir módulos Terraform con variables parametrizadas y outputs
- [ ] Backend de estado en bucket `gs://agrosat-tfstate` con versionado activado
- [ ] Scripts Bash `scripts/azure_h100_start.sh` y `scripts/azure_h100_stop.sh`
- [ ] Tests con `terraform validate` en GitHub Actions
- [ ] Documentar en `docs/infrastructure.md` el flujo para aprovisionar y destruir

**Estimación:** 2 puntos (~1 día).

---

### US-004 — DVC + MLflow + Dagster + dbmate MLOps base

**Como** equipo,
- **quiero** versionado de datos, tracking de experimentos, orquestación asset-oriented y migraciones de base de datos desde el primer commit,
- **para que** cualquier experimento reportado en los avances del curso sea ejecutable por un tercero a partir del repositorio.

**Criterios de Aceptación:**

- **DVC 3.48** inicializado con remote `gcs://agrosat-dvc-remote` y autenticación vía service account.
- **Dagster 1.9+** desplegado en Cloud Run con assets declarativos: `alphaearth_annual`, `sentinel2_scenes`, `dinov3_features`, `spectral_indices`, `parcel_features`, `baseline_model`, `alt_models`, `final_vlm`, `ensemble`, `drift_check`. Cada asset con dependencias explícitas y lineage visible en Dagster UI.
- **MLflow 2.16** server en Cloud Run con tracking store PostgreSQL y artifact store GCS; URL accesible para el equipo. Integración Dagster→MLflow vía package `dagster-mlflow`.
- **dbmate** configurado en `db/migrations/`, con scripts `make db-migrate` (`dbmate up`) y `make db-rollback` (`dbmate down`). Migración inicial crea tablas base.
- Todos los scripts de entrenamiento del EPIC 4, 5, 6 registrarán automáticamente en MLflow: parámetros, métricas cada epoch, artefactos (checkpoints, matrices de confusión, curvas ROC), tags (`data_version` con el hash DVC y `code_version` con el sha git).

**Tareas técnicas:**

- [ ] Inicializar DVC y configurar remote con service account
- [ ] Escribir `dagster_project/assets.py` con definiciones de los 10 assets principales
- [ ] Desplegar MLflow server con `mlflow server --backend-store-uri postgresql://...`
- [ ] Configurar dbmate con `.dbmate/config.yml` y migración inicial
- [ ] Template `ml/utils/mlflow_utils.py` con decorador `@track_experiment`

**Estimación:** 2 puntos (~1 día).

---

### US-005 — Pipeline CI/CD con GitHub Actions y Cloud Build

**Como** equipo,
- **quiero** un pipeline automatizado que valide y despliegue cada cambio,
- **para que** cualquier merge a `main` llegue a staging con smoke tests en menos de 10 minutos y sin intervención manual.

**Criterios de Aceptación:**

- Cada push a `develop` dispara: instalación de dependencias Poetry, linting con `ruff check`, formateo con `ruff format --check`, tipado con `mypy`, tests unitarios con `pytest`, verificación de cobertura ≥70% backend con `pytest-cov`, `dvc status` para detectar archivos sin versionar.
- Cada push a `main` dispara además: build de las imágenes Docker multi-stage, push a Artifact Registry con tag `sha-{git-sha}` y `latest`, aplicación de migraciones de base de datos con `dbmate up`, deploy a Cloud Run de los cuatro servicios (api, frontend, tiling, inference-worker), smoke tests contra `/healthz` de cada servicio, Playwright end-to-end test básico en staging que valida el flujo de chat con un query fijo.
- El pipeline falla si la cobertura de tests cae por debajo del umbral o si los smoke tests no pasan.
- Los secretos utilizados (API keys de Gemini, Copernicus CDSE, HuggingFace tokens) se leen desde GitHub Secrets y se inyectan a Cloud Run desde Secret Manager.

**Tareas técnicas:**

- [ ] Workflow `.github/workflows/test.yml` para `develop`
- [ ] Workflow `.github/workflows/deploy.yml` para `main` con steps encadenados
- [ ] `cloudbuild.yaml` con substituciones parametrizadas
- [ ] Test E2E Playwright `tests/e2e/chat_smoke.spec.ts`
- [ ] Badge de estado del pipeline en el README del proyecto

**Estimación:** 2 puntos (~1 día).

---

**Subtotal EPIC 0: 10 story points.**

---

## EPIC 1: Ingesta de Datos — AlphaEarth, Sentinel, DINOv3 {#epic-1}

**Objetivo.** Automatizar la descarga, preprocesamiento, conversión a Cloud-Optimized GeoTIFF y catalogación STAC de las fuentes de datos públicas necesarias para el proyecto, cubriendo las tres regiones piloto italianas y el benchmark de control francés PASTIS.

**Alineado con.** Avance 0 (entendimiento de los datos) y Avance 1 (disponibilidad para EDA).

**Regiones de interés:** Pianura Padana (Lombardía y Emilia-Romaña, ~1,500 km² subset), Toscana central (~800 km²), Apulia (Tavoliere delle Puglie, ~1,200 km²), más el control PASTIS en Francia metropolitana.

**Puntos totales de la épica: 12.**

---

### US-006 — Pipeline de ingesta de AlphaEarth Foundations desde GEE

**Como** ML Engineer,
- **quiero** descargar los embeddings AlphaEarth Foundations 64-dim para las tres regiones piloto italianas,
- **para que** sean la fuente principal de features del pipeline de modelado sin necesidad de entrenar un foundation model propio.

**Criterios de Aceptación:**

- Se define un archivo `config/rois.yaml` con las cuatro geometrías (tres regiones italianas + PASTIS control) en formato GeoJSON, con metadatos `name`, `bbox`, `crs` (EPSG:4326) y `preferred_crs_projection` (EPSG:32633 para Italia).
- El script `scripts/download_alphaearth.py` con CLI Typer ejecuta query a la colección `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` para los años 2017 a 2025 para cada ROI.
- Los exports se lanzan vía `ee.batch.Export.image.toCloudStorage()` con destino `gs://agrosat-data/alphaearth/{roi_name}/{year}.tif`, formato COG, compresión DEFLATE, nodata declarado.
- La tabla PostGIS `alphaearth_tiles` registra automáticamente cada archivo descargado con columnas `id`, `roi_name`, `year`, `bbox` (geometría), `storage_uri`, `size_mb`, `download_date`.
- El **asset Dagster `alphaearth_annual`** maneja reintentos con backoff exponencial en caso de rate limit y registra eventos en MLflow. El lineage es visible en Dagster UI.
- Documentación en `docs/data/alphaearth.md` incluye la atribución obligatoria a Google y referencia a los GEE Terms of Service.

**Tareas técnicas:**

- [ ] Service account con rol Earth Engine Resource Writer
- [ ] Credenciales JSON en Secret Manager
- [ ] Script `download_alphaearth.py` con CLI Typer
- [ ] Migración `dbmate new create_alphaearth_tiles` + RLS por tenant
- [ ] Definir asset Dagster en `dagster_project/assets/alphaearth.py`
- [ ] Documentar atribución y licencia

**Licencia / legal:** Google Earth Engine Terms of Service (https://earthengine.google.com/terms/). Uso académico y comercial permitido con atribución.

**Estimación:** 3 puntos (~1.5 días).

---

### US-007 — Pipeline de descarga Sentinel-2 L2A vía CDSE

**Como** ML Engineer,
- **quiero** descargar escenas Sentinel-2 L2A completas para consultas que requieran resolución fina,
- **para que** el agente pueda invocar visualizaciones reales de las 13 bandas espectrales cuando el usuario pida detalle visual o el VLM necesite procesar la imagen cruda.

**Criterios de Aceptación:**

- Query STAC vía `pystac-client` contra el endpoint CDSE (`https://catalogue.dataspace.copernicus.eu/stac`).
- Filtros configurables: `eo:cloud_cover<30`, rango temporal, bandas B02/B03/B04/B05/B06/B07/B08/B8A/B11/B12 y SCL.
- Descarga concurrente controlada (máximo 4 conexiones simultáneas para respetar rate limits CDSE) con backoff exponencial en caso de 429.
- Conversión automática a COG con `rio cogeo create --profile deflate` y tiling interno 512×512.
- Almacenamiento en `gs://agrosat-data/raw/s2/{roi}/{date}/B{nn}.tif`.
- Tabla PostGIS `sentinel2_scenes` con `scene_id`, `bbox`, `datetime`, `cloud_cover`, `bands_available`, `storage_uri`.
- **Asset Dagster `sentinel2_scenes`** con reintentos y dependencia declarada de `alphaearth_annual` (misma ROI).

**Tareas técnicas:**

- [ ] Credenciales CDSE (registro gratuito en dataspace.copernicus.eu) en Secret Manager
- [ ] Script `scripts/download_s2.py` con CLI Typer
- [ ] Asset Dagster `sentinel2_scenes` con reintentos
- [ ] Dockerfile del worker con `gdal`, `rasterio`, `sentinelhub-py`, `pystac-client`
- [ ] Migración `dbmate new create_sentinel2_scenes` con índices GIST espaciales

**Licencia / legal:** Copernicus Open Access License. Atribución: "Contains modified Copernicus Sentinel data [year]".

**Estimación:** 3 puntos (~1.5 días).

---

### US-008 — Setup DINOv3-satellite y descarga de PASTIS-R y Dynamic World

**Como** ML Engineer,
- **quiero** DINOv3-satellite disponible como feature extractor frozen y los datasets benchmark PASTIS-R y Dynamic World descargados a DVC,
- **para que** los pipelines de EDA, baseline y modelos alternativos puedan consumir estas fuentes sin esperas en tiempo de experimento.

**Criterios de Aceptación:**

- Checkpoint **`facebook/dinov3-vitl16-pretrain-sat493m`** descargado desde HuggingFace tras aceptar licencia DINOv3 y cacheado en `gs://agrosat-models/dinov3/`.
- Módulo `ml/extractors/dinov3_extractor.py` con clase `DINOv3Extractor` que expone método `.extract(image: np.ndarray) -> torch.Tensor` y devuelve features 1024-dim (ViT-L/16).
- PASTIS-R descargado desde HuggingFace Datasets a `data/raw/pastis/` y versionado con DVC.
- Dynamic World subset Italia (tres regiones piloto, 2022-2025) descargado vía GEE export a `gs://agrosat-data/dynamic_world/`.
- USGS Spectral Library descargada a `data/reference/spectral_library.parquet`.
- GSAA Italia (parcelas administrativas de las regiones piloto) descargado desde los portales AGEA regionales a `data/reference/gsaa_italia/{region}.gpkg`.

**Tareas técnicas:**

- [ ] Wrapper DINOv3 con caching en Redis (hash de imagen → features)
- [ ] Script batch `extract_dinov3_all.py` para pre-computar sobre todos los tiles
- [ ] Scripts separados para cada dataset auxiliar (PASTIS, Dynamic World, USGS, GSAA)
- [ ] Tests unitarios del extractor con fixtures pequeños

**Licencia / legal:** DINOv3 Meta AI Research License (verificar exactos en HuggingFace antes de extender a producto comercial); PASTIS-R CC-BY-SA; Dynamic World CC-BY-4.0; USGS Spectral Library Public Domain.

**Estimación:** 3 puntos (~1.5 días).

---

### US-009 — Catálogo STAC interno con pgstac

**Como** equipo,
- **quiero** un catálogo STAC interno queryable sobre todas las fuentes ingresadas,
- **para que** tanto los scripts de EDA como el agente LLM puedan localizar la escena o el embedding correctos mediante una sola consulta HTTP.

**Criterios de Aceptación:**

- Extensión PostGIS `pgstac` instalada en la base de datos.
- Ingest automático desde Dagster al finalizar cada asset de descarga.
- Endpoint FastAPI `GET /stac/search` con filtros `bbox`, `datetime`, `collection`, `query` siguiendo la especificación STAC API.
- Índice GIST sobre `geometry` y BTREE sobre `datetime` para latencia <100 ms en queries típicas.
- Documentación OpenAPI 3.1 auto-generada.

**Tareas técnicas:**

- [ ] Instalar pgstac y crear collections (`alphaearth`, `sentinel-2-l2a`, `sentinel-1-grd`, `dynamic-world`)
- [ ] Endpoint FastAPI con validación Pydantic
- [ ] Tests de integración con fixtures de escenas mock

**Estimación:** 3 puntos (~1.5 días).

---

**Subtotal EPIC 1: 12 story points.**

---

## EPIC 2: Análisis Exploratorio de Datos {#epic-2}

**Objetivo.** Producir un análisis exploratorio riguroso que responda las diez preguntas guía de la rúbrica del Avance 1 y que sirva como evidencia empírica para justificar las decisiones de Feature Engineering del EPIC 3.

**Alineado con.** Avance 1 (3 de mayo de 2026) — notebook secuencial en GitHub.

**Entregable.** Repositorio GitHub con tres notebooks Jupyter ejecutables secuencialmente más un dashboard Streamlit resumen y un reporte PDF para el anexo del Avance 1.

**Puntos totales de la épica: 14.** Motor principal de DataFrames: **Polars 1.x**.

---

### US-010 — Notebook EDA univariado sobre Sentinel-2 crudo

**Como** Data Scientist,
- **quiero** un notebook que caracterice estadísticamente cada banda Sentinel-2 y cada índice espectral derivado para las tres regiones italianas,
- **para que** el Avance 1 cubra exhaustivamente las diez preguntas de la rúbrica del curso.

**Criterios de Aceptación (mapeados 1:1 con rúbrica Avance 1):**

- Análisis de valores faltantes usando la capa SCL (Scene Classification Layer) y detección de patrones de ausencia por región y temporada.
- Estadísticas resumidas por banda (**computadas con Polars**): media, desviación estándar, mínimo, máximo, percentiles 5/25/50/75/95.
- Detección de outliers por banda con IQR y con Isolation Forest sobre muestras de 100k píxeles estratificados por clase.
- Cardinalidad de variables categóricas (clases de cultivo en PASTIS y Dynamic World).
- Análisis de distribuciones por banda (histogramas, pruebas Shapiro-Wilk, tests de normalidad) y evaluación de necesidad de transformaciones Box-Cox o Yeo-Johnson.
- Identificación de tendencias temporales: curva NDVI mensual promedio 2022-2025 por clase de cultivo y por región.
- Evaluación de si las imágenes requieren normalización para visualización (stretch 2-98 percentil, ejemplos visuales).
- Conclusiones concretas que justifiquen las decisiones del EPIC 3 de Feature Engineering.

**Tareas técnicas:**

- [ ] Notebook `notebooks/02a_eda_sentinel2.ipynb` con ejecución secuencial
- [ ] Muestreo estratificado con Polars (evitar OOM sobre 180 GB de Sentinel-2)
- [ ] Visualizaciones con matplotlib, folium (mapas interactivos) y plotly
- [ ] Sección final "Conclusiones y decisiones para FE"

**Estimación:** 5 puntos (~2.5 días).

---

### US-011 — Notebook EDA sobre embeddings AlphaEarth

**Como** Data Scientist,
- **quiero** caracterizar las 64 dimensiones de los embeddings AlphaEarth para las tres regiones italianas,
- **para que** entienda qué información semántica llevan y cuáles dimensiones son más discriminativas para el tipo de cultivo.

**Criterios de Aceptación:**

- Visualización 2D de los embeddings con t-SNE y UMAP, coloreada por clase de cultivo según GSAA italiano.
- Matriz de correlación entre las 64 dimensiones (heatmap) para detectar redundancia.
- Distribución por dimensión (histogramas, QQ plots) para verificar si vienen pre-normalizadas por DeepMind.
- Análisis de estabilidad temporal del embedding de una misma parcela entre 2022 y 2025.
- Identificación preliminar de las diez dimensiones más discriminativas usando feature importance de Random Forest contra labels GSAA.
- Comparativa visual entre AlphaEarth embedding y NDVI clásico para la misma parcela.

**Tareas técnicas:**

- [ ] Notebook `notebooks/02b_eda_alphaearth.ipynb` secuencial
- [ ] Muestreo estratificado de 100k píxeles por región y clase
- [ ] Parches de visualización reutilizables

**Estimación:** 4 puntos (~2 días).

---

### US-012 — Análisis bivariado, multivariado y temporal

**Como** Data Scientist,
- **quiero** cuantificar las correlaciones entre bandas, índices espectrales y labels, más un análisis de fenología,
- **para que** las variables redundantes se identifiquen antes del EPIC 3 y la separabilidad temporal de los cultivos quede documentada.

**Criterios de Aceptación:**

- Matrices de correlación Pearson y Spearman entre las 10 bandas Sentinel-2 y los 17 índices espectrales (computadas con Polars).
- Análisis VIF (Variance Inflation Factor) para detectar multicolinealidad.
- Gráficos de pares (pairplot seaborn) por clase de cultivo.
- Análisis bivariado categórico: tipo de cultivo vs pico de NDVI, distribución de timing de pico por clase.
- Análisis temporal: ACF/PACF del NDVI por parcela, clusterización temporal con DTW (`tslearn`), identificación de mono-cultivo vs doble ciclo.
- Detección de anomalías temporales (años secos vs normales) cruzando con ERA5.

**Tareas técnicas:**

- [ ] Notebook `notebooks/02c_eda_bivariado_temporal.ipynb`
- [ ] Funciones utilitarias en `ml/analysis/correlations.py`
- [ ] Gráficos exportados como PNG de alta resolución para el anexo

**Estimación:** 3 puntos (~1.5 días).

---

### US-013 — Dashboard Streamlit de EDA y reporte PDF

**Como** equipo,
- **quiero** un dashboard ejecutable y un reporte PDF que resuman el EDA,
- **para que** la rúbrica del Avance 1 valore las conclusiones claramente y el sponsor pueda revisar el trabajo sin abrir notebooks.

**Criterios de Aceptación:**

- Dashboard Streamlit `app/eda_dashboard.py` con seis tabs: univariado Sentinel-2, AlphaEarth, bivariado, temporal, espacial (mapa folium), conclusiones.
- Exportación PDF de las conclusiones vía `weasyprint` o `reportlab` para anexar al Avance 1.
- Conclusiones explícitas y mapeadas al contexto CRISP-ML(Q) Data Understanding.

**Tareas técnicas:**

- [ ] Dashboard Streamlit con navegación
- [ ] Función `export_report_pdf()`
- [ ] Integración con notebooks vía papermill

**Estimación:** 2 puntos (~1 día).

---

**Subtotal EPIC 2: 14 story points.**

---

## EPIC 3: Feature Engineering e Índices Espectrales {#epic-3}

**Objetivo.** Convertir los datos crudos en features listos para modelado, cubriendo los cuatro criterios de la rúbrica del Avance 2 (Construcción 30 pts, Normalización 30 pts, Selección/Extracción 30 pts, Conclusiones 10 pts).

**Alineado con.** Avance 2 (17 de mayo de 2026).

**Puntos totales de la épica: 18** (14 baseline + 4 SP de US-016b FarSLIP).

---

### US-014 — Biblioteca de 17 índices espectrales con justificación agronómica

**Como** equipo,
- **quiero** calcular al menos 17 índices espectrales estándar sobre Sentinel-2 con justificación documentada,
- **para que** el criterio "Construcción" de la rúbrica del Avance 2 (30 pts) quede cubierto con profundidad.

**Criterios de Aceptación:**

- Implementación vectorizada con `eemont` (sobre Google Earth Engine) y `spyndex` de los siguientes índices con sus justificaciones agronómicas documentadas en docstring: **NDVI** (vigor), **NDWI** (contenido de agua en hoja), **NDMI** (humedad de canopy), **EVI** (vigor mejorado en canopy denso), **SAVI** (vigor ajustado por suelo), **MSAVI2** (versión mejorada SAVI), **NBR** (detección de estrés por fuego/sequía), **MCARI** (clorofila en canopy), **CCCI** (clorofila corregida por canopy), **LAI** (Leaf Area Index), **FAPAR** (fracción de radiación absorbida), **PSRI** (senescencia), **NDCI** (clorofila en ambientes acuáticos agrícolas), **GCVI** (green chlorophyll), **RENDVI** (Red-Edge NDVI), **NDRE** (Red-Edge NDVI para cultivos densos), **TSAVI** (SAVI transformado).
- Módulo `ml/features/spectral_indices.py` con API consistente: cada índice es una función que acepta un `xarray.DataArray` con bandas como dimensión y devuelve un `xarray.DataArray` con el índice computado.
- Soporte para cálculo sobre series temporales (axis=time) con reduce.
- Cache en Redis con clave `{scene_id}:{index_name}` para evitar recómputo.
- Tests unitarios con valores de referencia conocidos (e.g., NDVI de píxel de bosque caducifolio en junio debe estar entre 0.7 y 0.9).

**Tareas técnicas:**

- [ ] Función `compute_index(da: xr.DataArray, index: str) -> xr.DataArray`
- [ ] Tabla de referencias académicas por índice en `docs/spectral_indices.md`
- [ ] Tests con fixtures sintéticos y fixtures reales de una parcela demo

**Estimación:** 4 puntos (~2 días).

---

### US-015 — Features temporales agregados por parcela

**Como** ML Engineer,
- **quiero** features temporales agregados a nivel parcela (armónicos FFT, percentiles, fenología),
- **para que** los modelos baseline del EPIC 4 puedan capturar dinámica temporal sin necesidad de arquitecturas temporales explícitas.

**Criterios de Aceptación:**

- Estadísticos temporales por índice espectral a lo largo del ciclo vegetativo (**computados con Polars LazyFrame**): media, std, min, max, percentiles 5/25/50/75/95.
- Descomposición harmónica (FFT) con las primeras tres componentes de frecuencia (amplitud y fase).
- Features fenológicos derivados: fecha de inicio del verdor (día en que NDVI cruza 0.3 ascendente), fecha de pico, valor del pico, fecha de senescencia, integral AUC del NDVI sobre el ciclo completo.
- Features derivativos: pendiente NDVI pre-pico, pendiente post-pico, duración del período de madurez.
- Todos los features disponibles en tabla `features_parcels` en PostgreSQL con UNIQUE `(parcel_id, year)`.

**Tareas técnicas:**

- [ ] Función `extract_temporal_features(parcel_timeseries: xr.DataArray) -> pl.DataFrame`
- [ ] Migración `dbmate new create_features_parcels`
- [ ] Tests contra parcela demo con curva NDVI conocida

**Estimación:** 3 puntos (~1.5 días).

---

### US-016 — Fusión multisensor a nivel parcela

**Como** ML Engineer,
- **quiero** un vector de features combinado por parcela,
- **para que** los modelos de EPIC 4-6 consuman una tabla única con features heterogéneos ya alineados.

**Criterios de Aceptación:**

- Vector combinado con las siguientes componentes por parcela: 64 dimensiones AlphaEarth (media sobre la parcela), 17 índices espectrales con sus estadísticos temporales (5 stats × 17 = 85 features), backscatter Sentinel-1 VV y VH con sus stats temporales (5 × 2 = 10 features), elevación media, pendiente media y orientación dominante desde SRTM DEM, temperatura media mensual y precipitación acumulada mensual desde ERA5 (24 features), geometría: superficie en ha, perímetro en m, elongación (3 features).
- **Banco FarSLIP fine-grained:** 512 dimensiones de embeddings producidos por la rama de destilación parche-a-parche descrita en US-016b (CLIP adaptado con técnica FarSLIP sobre crops Sentinel-2 256×256 px). Mejoran la discriminabilidad espacial sub-parcela sin degradar la coherencia semántica.
- Shape final aproximado: 64 + 85 + 10 + 3 + 24 + 3 + 512 (FarSLIP opcional) = **189 features tabulares clásicos por parcela + 512-dim FarSLIP**. Los modelos pueden consumir el vector completo o sólo el subset tabular según ablation declarada en el notebook.
- Normalización z-score global con estadísticos guardados en `artifacts/scaler_v1.pkl`.
- Split train/val/test estratificado espacialmente (K=5 folds por regiones no contiguas) guardado en `data/splits/`.
- **Fusión implementada con Polars `LazyFrame`** para eficiencia en memoria.

**Tareas técnicas:**

- [ ] Script `scripts/build_parcel_features.py` con asset Dagster
- [ ] Spatial train-test split con `geopandas` y tessellation
- [ ] Guardar scaler y splits con versionado DVC

**Estimación:** 3 puntos (~1.5 días).

---

### US-016b — Destilación FarSLIP parche-a-parche sobre crops Sentinel-2

**Como** ML Engineer,
- **quiero** entrenar una rama de adaptación CLIP siguiendo la técnica FarSLIP (Li et al., 2025) sobre crops Sentinel-2 de las tres regiones italianas,
- **para que** el pipeline disponga de embeddings fine-grained de 512 dimensiones que mejoren la cuantificación sub-parcela y alimenten tanto el banco de features de US-016 como el cabezal open-vocabulary de SegFormer-B2 en EPIC 5 US-024.

**Criterios de Aceptación:**

- Implementación de la pérdida de destilación parche-a-parche y de la alineación región-categoría basada en token CLS, siguiendo el procedimiento del paper (arXiv:2511.14901).
- Backbone teacher: CLIP ViT-B/16 pretrained; student: ViT-B/16 con las mismas dimensiones, inicializado desde teacher y fine-tuneado sobre crops Sentinel-2 256×256 px etiquetados con texto agronómico generado a partir de las clases CAP italianas.
- Dataset interno `data/farslip_pairs/` con al menos 30,000 pares imagen-texto cubriendo Pianura Padana, Toscana y Puglia.
- Entrenamiento en GCP L4 spot (4 epochs aproximadamente, ~6 horas) con MLflow run `farslip-clip-italy-v1`.
- Outputs: pesos student en `gs://agrosat-models/farslip/`, módulo `ml/extractors/farslip_extractor.py` con clase `FarSLIPExtractor`.
- Métrica de calidad de la adaptación: mejora de mIoU open-vocabulary ≥ 5 pp sobre CLIP-RemoteCLIP estándar al evaluar sobre subset PASTIS-R con clases verbalizadas en italiano y español.

**Tareas técnicas:**

- [ ] Reproducir la lógica de destilación parche-a-parche del paper en `ml/farslip/distill.py`
- [ ] Asset Dagster `farslip_embeddings_italy` con dependencia de `sentinel2_crops_256`
- [ ] Suite de tests unitarios para la pérdida y para la alineación CLS

**Estimación:** 4 puntos (~2 días).

---

### US-017 — Selección, extracción y normalización

**Como** ML Engineer,
- **quiero** aplicar métodos de filtrado y extracción con justificación empírica,
- **para que** el criterio "Selección/extracción" y "Normalización" de la rúbrica del Avance 2 (30+30 pts) quede cubierto.

**Criterios de Aceptación:**

- Métodos de filtrado ejecutados y documentados: VarianceThreshold (elimina features con varianza <0.01), correlación (remueve un feature de cada par con |r|>0.95), chi-cuadrado para categóricas, ANOVA F-score para numéricas.
- Métodos de extracción ejecutados: PCA con análisis de varianza explicada (objetivo 95%), Factor Analysis para firmas espectrales, UMAP 2D para visualización.
- Feature importance de Random Forest y XGBoost entrenados sobre todos los features como complemento.
- Tabla comparativa antes/después con métricas F1-macro y mIoU cross-validadas con split espacial.
- Transformaciones numéricas justificadas: StandardScaler para modelos lineales/SVM, MinMax para redes neuronales, Yeo-Johnson para variables sesgadas (como NDVI que puede ser negativo), log-transform para LAI y biomasa.
- Sección "Conclusiones CRISP-ML(Q) Data Preparation" al final del notebook.

**Tareas técnicas:**

- [ ] Notebook `notebooks/03_feature_engineering.ipynb` secuencial
- [ ] Funciones reutilizables en `ml/features/selection.py`
- [ ] Reporte tabular antes/después

**Estimación:** 4 puntos (~2 días).

---

**Subtotal EPIC 3: 18 story points** (14 originales + 4 de US-016b FarSLIP).

---

## EPIC 4: Baseline — AlphaEarth + XGBoost/RF {#epic-4}

**Objetivo.** Construir un baseline sólido sobre features tabulares AlphaEarth + índices espectrales con Random Forest y XGBoost, cubriendo los cinco criterios de la rúbrica del Avance 3 (Algoritmo 40 pts, Características 20 pts, Sub/sobreajuste 10 pts, Métrica 20 pts, Desempeño 10 pts).

**Alineado con.** Avance 3 (20 de mayo de 2026).

**Hipótesis clave.** Dado que AlphaEarth ya encapsula información multisensor en 64 dimensiones compactas, un baseline tabular sobre estas dimensiones debe superar en F1-macro a baselines clásicos que usan únicamente bandas Sentinel-2 crudas, alcanzando la meta de F1-macro ≥ 0.60 sobre PASTIS-R.

**Puntos totales de la épica: 10.**

---

### US-018 — Random Forest y XGBoost sobre features combinados

**Como** ML Engineer,
- **quiero** entrenar Random Forest y XGBoost sobre el vector de features del EPIC 3,
- **para que** el criterio "Algoritmo" (40 pts) de la rúbrica quede justificado con elección interpretable, robusta a outliers, bajo costo computacional y con feature importance nativa.

**Criterios de Aceptación:**

- Entrenamiento en GCP L4 spot con scikit-learn (RandomForestClassifier) y XGBoost 2.1 (XGBClassifier).
- Justificación documentada en el notebook: AlphaEarth ya codifica información multisensor; RF/XGB sobre estos 64-dim es un baseline fuerte, interpretable y computacionalmente barato; sirve como lower bound para evaluar viabilidad.
- Métricas reportadas: F1-macro (principal), F1-weighted, mIoU (para segmentación a nivel píxel), accuracy, Cohen's kappa.
- Hyperparameter tuning ligero con GridSearchCV (5-fold spatial CV).
- Desempeño mínimo declarado: F1-macro ≥ 0.60 sobre PASTIS-R. Si no se alcanza, el notebook documenta las causas probables y las decisiones para el EPIC 5.
- Modelos finales registrados en MLflow con runs `baseline-rf-alphaearth-v1` y `baseline-xgb-alphaearth-v1`.

**Tareas técnicas:**

- [ ] Script `ml/train/train_baseline.py` con CLI
- [ ] MLflow autologging para RF y XGB
- [ ] Serialización de modelos con joblib en MLflow artifacts

**Estimación:** 3 puntos (~1.5 días).

---

### US-019 — Feature importance y análisis SHAP

**Como** ML Engineer,
- **quiero** identificar y visualizar los features más relevantes,
- **para que** el criterio "Características importantes" (20 pts) de la rúbrica quede justificado con interpretación y representación visual.

**Criterios de Aceptación:**

- Feature importance nativa de Random Forest (Gini) y XGBoost (Gain).
- Análisis SHAP (explainable AI) sobre top 20 features globalmente con `shap.TreeExplainer`.
- SHAP dependency plots para los cinco features más importantes.
- Identificación explícita de cuáles dimensiones AlphaEarth dominan (dato relevante para el Paper Track).
- Conclusiones que validen (o refuten) las decisiones de Feature Engineering del EPIC 3.

**Tareas técnicas:**

- [ ] Notebook con SHAP waterfall y summary plots
- [ ] Guardar gráficos como PNG de alta resolución
- [ ] Sección de conclusiones con recomendación de ajustes a FE si aplica

**Estimación:** 2 puntos (~1 día).

---

### US-020 — Curvas de aprendizaje, validación y análisis de sub/sobreajuste

**Como** ML Engineer,
- **quiero** diagnosticar sub y sobreajuste con visualizaciones,
- **para que** el criterio "Sub/sobreajuste" (10 pts) de la rúbrica quede cubierto con evidencia gráfica.

**Criterios de Aceptación:**

- Curva de aprendizaje (accuracy train/val vs número de muestras de entrenamiento).
- Curva de validación (accuracy vs hiperparámetros críticos: `max_depth` para RF, `n_estimators` y `learning_rate` para XGB).
- Cross-validation 5-fold estratificado espacial (splits por regiones no contiguas para evitar data leakage geográfico).
- Diagnóstico explícito del gap train-val: si >10% se documenta como sobreajuste; si accuracy train y val ambos bajos se documenta como subajuste.

**Tareas técnicas:**

- [ ] Funciones `plot_learning_curve` y `plot_validation_curve`
- [ ] Documentación del criterio de spatial CV

**Estimación:** 2 puntos (~1 día).

---

### US-021 — Notebook secuencial y comparativa vs Sentinel-2 crudo

**Como** equipo,
- **quiero** un notebook `notebooks/04_baseline.ipynb` ejecutable de principio a fin más una comparativa AlphaEarth vs Sentinel-2 crudo,
- **para que** el criterio de libreta secuencial de la rúbrica se cumpla y el valor incremental de AlphaEarth quede documentado empíricamente.

**Criterios de Aceptación:**

- Notebook secuencial que ejecuta todas las celdas sin intervención manual.
- Tabla comparativa: RF+XGB sobre (a) AlphaEarth 64-dim puro, (b) Sentinel-2 crudo (10 bandas medias), (c) vector combinado completo del EPIC 3.
- Métrica principal F1-macro + otras dos métricas relevantes + tiempo de entrenamiento.
- Discusión de resultados y conclusiones para el EPIC 5.

**Tareas técnicas:**

- [ ] Ejecución secuencial validada con papermill en CI
- [ ] Exportación de resultados a tabla LaTeX para uso futuro en Paper Track

**Estimación:** 3 puntos (~1.5 días).

---

**Subtotal EPIC 4: 10 story points.**

---

## EPIC 5: Modelos Alternativos — Seis Arquitecturas {#epic-5}

**Objetivo.** Construir seis modelos individuales diversos (mínimo requerido por la rúbrica del Avance 4), compararlos y ajustar los dos mejores.

**Alineado con.** Avance 4 (24 de mayo de 2026) — notebook secuencial en GitHub. Rúbrica: Comparativa 60 pts + Ajuste fino 30 pts + Modelo individual final 10 pts.

**Arquitecturas seleccionadas:**

1. U-Net con backbone ResNet-50 pretrained (CNN clásica, spatial only).
2. DeepLabv3+ con backbone MobileNetV3 (CNN eficiente con ASPP).
3. SegFormer-B2 (Transformer de segmentación, spatial only).
4. U-TAE — Temporal Attention Encoder (baseline temporal de referencia).
5. TSViT — Vision Transformer factorizado temporal-espacial (implementación del Paper 1 del profesor).
6. Swin-UNETR para SITS (Transformer moderno para series temporales, arXiv:2412.01944).

**Puntos totales de la épica: 21** (20 baseline + 1 SP adicional en US-024 SegFormer-B2 por integración del cabezal FarSLIP open-vocabulary).

---

### US-022 — Modelo 1: U-Net con ResNet-50

**Como** ML Engineer,
- **quiero** entrenar U-Net sobre patches 256×256 de una imagen Sentinel-2 sin dimensión temporal,
- **para que** dispongamos de un baseline denso CNN spatial-only contra el cual comparar arquitecturas temporales.

**Criterios de Aceptación:**

- Backbone ResNet-50 pretrained en ImageNet; head U-Net con skip connections.
- Loss combined CrossEntropy + Dice con pesos {0.5, 0.5}.
- Entrenamiento en GCP L4 con batch 8 y Automatic Mixed Precision BF16.
- Métricas reportadas: mIoU, pixel accuracy, F1 por clase.
- Run MLflow `alt-unet-resnet50-v1`.

**Tareas técnicas:**

- [ ] Script `ml/train/train_unet.py` usando `segmentation_models.pytorch`
- [ ] Pipeline de datos con `WebDataset` para streaming de patches
- [ ] Early stopping con patience 5

**Estimación:** 3 puntos (~1.5 días).

---

### US-023 — Modelo 2: DeepLabv3+ con MobileNetV3

**Como** ML Engineer,
- **quiero** entrenar DeepLabv3+ como alternativa eficiente,
- **para que** tengamos una CNN con ASPP (Atrous Spatial Pyramid Pooling) en la comparativa.

**Criterios de Aceptación:**

- Backbone MobileNetV3-Large pretrained; head DeepLabv3+ con ASPP rates {6, 12, 18}.
- Mismo pipeline de datos y loss que US-022.
- Run MLflow `alt-deeplabv3plus-mobilenet-v1`.

**Tareas técnicas:**

- [ ] Reusar pipeline de datos de US-022
- [ ] Configurar backbone desde `segmentation_models.pytorch`

**Estimación:** 2 puntos (~1 día).

---

### US-024 — Modelo 3: SegFormer-B2 con cabezal open-vocabulary FarSLIP

**Como** ML Engineer,
- **quiero** SegFormer-B2 como representante Transformer de segmentación spatial-only, acoplado a un cabezal open-vocabulary basado en la técnica FarSLIP (Li et al., 2025),
- **para que** la comparativa incluya arquitecturas CNN y Transformer y para que el sistema produzca máscaras semánticas sub-parcela alineadas a categorías de cultivo descritas en lenguaje natural (italiano, español e inglés).

**Criterios de Aceptación:**

- Variante SegFormer-B2 pretrained en ADE20K, head adaptado a 18 clases PASTIS.
- Cabezal complementario open-vocabulary cuyo encoder visual proviene del student FarSLIP entrenado en US-016b, con alineación región-categoría basada en token CLS. Inferencia con clases verbalizadas como prompts (por ejemplo "campo di mais in maturazione", "viñedo en senescencia", "olive grove with pruning").
- Fusión final por máscara que combina la salida supervisada (18 clases PASTIS) con la salida open-vocabulary, ponderada por la confianza de cada rama.
- Fine-tuning con LoRA opcional para reducir memoria.
- Runs MLflow: `alt-segformer-b2-v1` (rama supervisada) y `alt-segformer-b2-farslip-ov-v1` (rama open-vocabulary).
- Métrica de ablation: mIoU del SegFormer con vs sin cabezal FarSLIP, esperando una mejora ≥ 3 pp en zonas de borde inter-parcela.

**Tareas técnicas:**

- [ ] Cargar SegFormer desde `transformers.SegformerForSemanticSegmentation`
- [ ] Adaptar head a 18 clases PASTIS
- [ ] Integrar el `FarSLIPExtractor` de US-016b como rama paralela y la pérdida CLS de alineación
- [ ] Diseñar la regla de fusión por máscara con ablation documentada

**Estimación:** 4 puntos (~2 días, +1 SP vs base por integración FarSLIP).

---

### US-025 — Modelo 4: U-TAE

**Como** ML Engineer,
- **quiero** entrenar U-TAE sobre las series temporales Sentinel-2,
- **para que** el baseline temporal de referencia de PASTIS esté en la comparativa.

**Criterios de Aceptación:**

- Implementación oficial de VSainteuf (`utae-paps`) integrada en el repo.
- Input: T=20 observaciones × 10 bandas × H × W.
- Positional encoding temporal absoluto.
- Entrenamiento en **H100 ventana V2 (~12 h compartidas con TSViT y Swin-UNETR)**.
- Run MLflow `alt-utae-v1`.

**Tareas técnicas:**

- [ ] Clonar repo oficial y adaptarlo al pipeline del proyecto
- [ ] Configurar dataloader para secuencias temporales PASTIS-R

**Estimación:** 3 puntos (~1.5 días).

---

### US-026 — Modelo 5: TSViT (Paper 1 del profesor)

**Como** ML Engineer,
- **quiero** replicar TSViT con el encoder temporal-espacial factorizado y múltiples cls tokens,
- **para que** implementemos directamente la propuesta del Paper 1 del profesor como contribución al benchmark y como componente del ensemble final.

**Criterios de Aceptación:**

- Reproducción fiel de Tarasiou et al. 2023: temporal encoder → spatial encoder factorizado.
- Múltiples cls tokens (K=18 clases PASTIS) separables entre encoders.
- Positional encoding temporal por fecha real de adquisición (tabla aprendida).
- Entrenamiento en **H100 ventana V2 (~12 h compartidas)**.
- Métricas esperadas alineadas con el paper (≥ estado del arte en PASTIS).
- Run MLflow `alt-tsvit-v1`.

**Tareas técnicas:**

- [ ] Clonar repo oficial del paper y adaptar a pipeline del proyecto
- [ ] Verificar reproducción contra el número reportado en el paper
- [ ] Integrar en la comparativa

**Estimación:** 5 puntos (~2.5 días).

---

### US-027 — Modelo 6: Swin-UNETR para SITS

**Como** ML Engineer,
- **quiero** Swin-UNETR adaptado a SITS (arXiv:2412.01944) como representante Transformer moderno,
- **para que** la comparativa incluya un modelo 2024-2025 del estado del arte reciente.

**Criterios de Aceptación:**

- Implementación basada en `monai` Swin-UNETR con adaptación a dimensión temporal.
- Entrenamiento en **H100 ventana V2**.
- Run MLflow `alt-swin-unetr-v1`.

**Tareas técnicas:**

- [ ] Instalar `monai` y cargar Swin-UNETR
- [ ] Adaptar input para series temporales

**Estimación:** 2 puntos (~1 día).

---

### US-028 — Comparativa, ajuste fino de top-2 y selección del modelo individual final

**Como** equipo,
- **quiero** una tabla comparativa ordenada por métrica principal y un ajuste fino bayesiano de los dos mejores modelos,
- **para que** los criterios "Comparativa" (60 pts), "Ajuste fino" (30 pts) y "Modelo individual final" (10 pts) de la rúbrica del Avance 4 queden cubiertos.

**Criterios de Aceptación:**

- Tabla comparativa con columnas: modelo, F1-macro, F1-weighted, mIoU, accuracy, tiempo de entrenamiento en minutos, tiempo de inferencia en ms/imagen, número de parámetros.
- Ajuste fino con Optuna (búsqueda bayesiana, ≥30 trials por modelo) sobre los dos mejores según F1-macro. Espacio de búsqueda: learning rate, weight decay, batch size, dropout, pesos del loss.
- Selección justificada del modelo individual final con trade-offs documentados (accuracy vs latencia vs tamaño vs complejidad de despliegue).
- Notebook `notebooks/05_alt_models.ipynb` secuencial.

**Tareas técnicas:**

- [ ] Script `ml/tune/optuna_tune.py` con storage persistente en PostgreSQL
- [ ] Tabla comparativa auto-generada desde MLflow API
- [ ] Sección de conclusiones con recomendación para EPIC 6

**Estimación:** 2 puntos (~1 día).

---

**Subtotal EPIC 5: 21 story points** (20 baseline + 1 SP US-024 con cabezal FarSLIP).

---

## EPIC 6: Modelo Final — Gemma 4 26B-MoE LoRA + Ensambles {#epic-6}

**Objetivo.** Construir el modelo final de máximo rendimiento mediante fine-tuning de **Gemma 4 26B-MoE** con LoRA y cuatro ensambles que combinan el modelo visual con los mejores individuales del EPIC 5 y el baseline AlphaEarth+XGB del EPIC 4.

**Alineado con.** Avance 5 (31 de mayo de 2026) — notebook secuencial en GitHub. Rúbrica: Ensambles 60 pts + Selección 20 pts + Gráficos 20 pts.

**Puntos totales de la épica: 20.**

---

### US-029 — Fine-tuning Gemma 4 26B-MoE con LoRA en H100

**Como** ML Engineer,
- **quiero** fine-tunear **Gemma 4 26B-MoE** con LoRA rank 16 sobre AgroMind + dataset agrícola italiano/español,
- **para que** el modelo visual final esté adaptado al dominio agrícola mediterráneo con soporte nativo multilingüe.

**Criterios de Aceptación:**

- Base model `google/gemma-4-26b-it` (MoE 26B totales / 4B activos por token, contexto 256K, 140 idiomas, multimodal imagen+video+audio, Apache 2.0).
- Configuración LoRA: rank 16, target modules attention (q_proj, k_proj, v_proj, o_proj) y MLP (gate_proj, up_proj, down_proj), excluye expertos MoE.
- **Entrenamiento en Azure H100 NVL 96GB ventana V3 (~24 h distribuidas en 3 noches):**
  - FSDP + BF16 + FlashAttention-2 + gradient checkpointing
  - Batch effective 16 (batch 2 × grad accum 8)
  - 3 epochs sobre AgroMind (28,482 QA pairs) + AgroMind-IT/ES seed 500 pares + synthetic augmentation
  - Checkpoint cada 30 minutos a Azure Blob
- **Plan B L4:** Gemma 4 E4B (4.5B efectivos) con QLoRA 4-bit, batch 1, duración ~24 h.
- **Comparación opcional ventana V4:** Qwen3-VL-30B-A3B LoRA en las mismas condiciones para reportar tabla comparativa Gemma 4 vs Qwen3-VL.
- Inferencia servida con vLLM continuous batching.
- Run MLflow `final-gemma4-26b-lora-v1` con todos los artefactos.

**Presupuesto de VRAM validado (1×H100 96GB):**

- Pesos Gemma 4 26B-MoE BF16: ~52 GB
- LoRA adapters (rank 16): <0.5 GB
- Optimizer states Adam (solo LoRA): ~1 GB
- KV cache batch 2 × contexto 32K: ~8 GB
- Activations con gradient checkpointing: ~15 GB
- Overhead CUDA + FlashAttn + vLLM: ~5 GB
- **Total ocupado: ~82 GB sobre 96 GB** (margen 14 GB)

**Tareas técnicas:**

- [ ] Script `ml/train/train_gemma4_lora.py` usando Accelerate + PEFT
- [ ] Config YAML `configs/gemma4_h100.yaml` y `configs/gemma4_l4_fallback.yaml`
- [ ] Callback de resume from checkpoint
- [ ] Serving script vLLM con FastAPI

**Licencia / legal:** Gemma 4 Apache 2.0; AgroMind CC-BY.

**Estimación:** 7 puntos (~3.5 días).

---

### US-030 — Adaptación de Gemma 4 a segmentación densa

**Como** ML Engineer,
- **quiero** una capa de adaptación que convierta las respuestas de Gemma 4 con coordenadas/polígonos en máscaras de segmentación densa,
- **para que** el modelo visual pueda participar en la comparativa de segmentación del Avance 5.

**Criterios de Aceptación:**

- Wrapper `ml/models/gemma4_segmenter.py` que: (a) construye el prompt de segmentación multilingüe, (b) parsea la respuesta textual con coordenadas/polígonos, (c) rasteriza a máscara numpy.
- Inferencia sliding window con overlap 32 px y weighted fusion por promedio ponderado.
- Validación de la máscara contra esquema (cerrada, no self-intersecting).

**Tareas técnicas:**

- [ ] Clase `Gemma4Segmenter` con métodos `build_prompt`, `parse_response`, `rasterize`
- [ ] Tests con polígonos sintéticos y polígonos reales de parcela demo

**Estimación:** 3 puntos (~1.5 días).

---

### US-031 — Cuatro ensambles homogéneos y heterogéneos

**Como** equipo,
- **quiero** construir cuatro ensambles que combinen los mejores modelos del EPIC 5, el baseline del EPIC 4 y Gemma 4,
- **para que** el criterio "Ensambles" (60 pts) de la rúbrica del Avance 5 quede cubierto con estrategias homogéneas y heterogéneas.

**Criterios de Aceptación:**

- **Ensamble 1 — Voting homogéneo:** majority vote sobre top-3 modelos temporales del EPIC 5 (U-TAE + TSViT + Swin-UNETR).
- **Ensamble 2 — Bagging sobre XGB AlphaEarth:** 10 XGB entrenados sobre bootstraps distintos del training set; promedio de probabilidades.
- **Ensamble 3 — Stacking heterogéneo:** U-TAE + TSViT + Swin-UNETR + XGB AlphaEarth + **Gemma 4** como base learners; meta-learner XGBoost sobre out-of-fold predictions.
- **Ensamble 4 — Blending con pesos optimizados:** los mismos base learners del ensamble 3 combinados por promedio ponderado; pesos optimizados con Optuna minimizando el gap F1 train-val.
- Tabla comparativa final: modelo individual SOTA vs 4 ensambles + tiempo y modelo elegido con justificación.

**Tareas técnicas:**

- [ ] Módulo `ml/ensemble/` con clase `EnsembleModel`
- [ ] Out-of-fold predictions generadas con spatial CV
- [ ] Optuna study para blending
- [ ] Runs MLflow individuales por ensamble

**Estimación:** 7 puntos (~3.5 días).

---

### US-032 — Gráficas interpretadas del modelo final

**Como** equipo,
- **quiero** al menos cuatro gráficas interpretadas del modelo final,
- **para que** el criterio "Gráficos" (20 pts) de la rúbrica quede cubierto.

**Criterios de Aceptación:**

- Matriz de confusión (normalizada + absoluta) con interpretación por clase.
- Curva ROC multi-clase one-vs-rest con AUC por clase y macro-average.
- Curva Precision-Recall por clase.
- Análisis de residuos espacial: mapa de errores superpuesto sobre la geometría real de las parcelas.
- Gráfica adicional: UMAP 2D de embeddings Gemma 4 coloreado por clase (para interpretabilidad y Paper Track).
- Cada gráfica con interpretación escrita de al menos un párrafo.

**Tareas técnicas:**

- [ ] Funciones reutilizables en `ml/eval/plots.py`
- [ ] Exportación PNG y SVG de alta resolución

**Estimación:** 3 puntos (~1.5 días).

---

**Subtotal EPIC 6: 20 story points.**

---

## EPIC 7: Agente Conversacional con Google ADK {#epic-7}

**Objetivo.** Construir el agente conversacional que es la esencia del producto: combina el VLM fine-tuned (Gemma 4 26B-MoE del EPIC 6), un orquestador LLM con dos variantes (Gemini 3.1 Pro cloud y Qwen3.5-35B-A3B self-hosted), nueve tools geoespaciales y un Spatial-RAG híbrido para razonamiento trazable, todo sobre **Google ADK** (Agent Development Kit).

**Alineado con.** Avances 5 y 6. Evaluación formal con benchmarks AgroMind, GeoAnalystBench y GeoBenchX.

**Puntos totales de la épica: 14** (13 baseline + 1 SP adicional en US-034 por la separación perceiver–reasoner siguiendo el patrón Be My Eyes). El tracing built-in de ADK y el deploy nativo a Vertex AI Agent Engine permiten mantener un presupuesto contenido sin observabilidad custom del agente.

---

### US-033 — Construcción de nueve tools geoespaciales con schemas Pydantic

**Como** equipo,
- **quiero** nueve tools ejecutables desde Google ADK con schemas Pydantic validados,
- **para que** el agente tenga ejecución real verificable (no hallucinations) y el pipeline sea testeable unitariamente.

**Criterios de Aceptación:**

Tools implementados como ADK `FunctionTool` con input/output schema Pydantic y logging estructurado:

- `alphaearth_query(roi_geojson: GeoJSON, year: int) -> AlphaEarthResult` — recupera embedding AlphaEarth promedio y clasificación vía XGBoost del EPIC 4.
- `sentinel_search(bbox: BBox, datetime_range: str, cloud_cover_max: float) -> List[Scene]` — query STAC.
- `rasterio_tool(scene_id: str, operation: Literal["stats","histogram","read_window"]) -> dict` — estadísticas raster.
- `geopandas_intersect(aoi: GeoJSON, layer: str) -> GeoDataFrame` — intersección con GSAA, zonas protegidas.
- `ndvi_calculator(scene_id: str, aoi: GeoJSON) -> NDVIResult` — NDVI + estadísticos espaciales.
- `timeseries_extractor(aoi: GeoJSON, start: date, end: date, index: str) -> TimeSeries` — serie temporal por parcela.
- `phenology_descriptor(timeseries: TimeSeries) -> str` — genera descripción fenológica (llama al LLM con prompt estructurado estilo Paper 2 del profesor).
- `dinov3_extract(aoi: GeoJSON) -> dict[str, float]` — vigor, LAI estimado, canopy height.
- `crop_classifier_tool(aoi: GeoJSON) -> ClassificationResult` — invoca modelo final del EPIC 6.

**Tareas técnicas:**

- [ ] Módulo `ml/agent/tools/` con un archivo por tool
- [ ] Tests unitarios por tool con fixtures determinísticos
- [ ] Documentación OpenAPI auto-generada desde schemas Pydantic

**Estimación:** 4 puntos (~2 días).

---

### US-034 — Agente Google ADK Plan-and-React con Spatial-RAG híbrido y arquitectura perceiver–reasoner

**Como** equipo,
- **quiero** un agente Google ADK que implemente plan-and-react sobre los tools, consulte un Spatial-RAG híbrido antes de actuar, y separe explícitamente percepción visual y razonamiento simbólico siguiendo el patrón Be My Eyes (Huang et al., 2025),
- **para que** el razonamiento sea trazable, auditable, reduzca la tasa de alucinación aproximadamente un 30% según GeoAnalystBench 2025, y permita intercambiar la variante de orquestador (Gemini 3.1 Pro ↔ Qwen3.5-35B-A3B) sin reentrenar el perceiver.

**Criterios de Aceptación:**

- **`agrosat_agent = Agent(model=..., tools=[...], instruction=...)`** usando Google ADK con session state nativo.
- **Arquitectura perceiver–reasoner:** el agente ADK internamente delega la lectura de imaginería (Sentinel-2, AlphaEarth, mapas segmentados) al **agente perceiver** (Gemma 4 26B-MoE fine-tuneado, o Qwen3-VL-30B-A3B en variante comparativa) y la planificación + tool calling al **agente reasoner** (Gemini 3.1 Pro o Qwen3.5-35B-A3B). El perceiver emite descripciones estructuradas textuales que el reasoner consume.
- **Spatial-RAG híbrido** con dos componentes en serie: (a) filtrado espacial vía PostGIS `ST_DWithin` para recuperar parcelas geográficamente similares, (b) similitud semántica vía pgvector sobre embeddings `intfloat/e5-mistral-7b-instruct` de docs agronómicos y metadata de escenas. Fusión con weighted score.
- **Planner:** el reasoner genera un JSON con pasos y tool calls; validación con schema Pydantic antes de ejecutar.
- **Executor:** ejecuta tools en orden, maneja errores (retry, fallback, user feedback si el error es irrecuperable), re-planifica si un tool devuelve output inesperado.
- **Memoria persistente** en PostgreSQL por `session_id` (feature nativa ADK).
- **Streaming SSE al frontend** con eventos nativos ADK: `plan_created`, `perceiver_observation`, `tool_call`, `tool_result`, `final_answer`.
- **Tracing built-in ADK** visible en Vertex AI console, elimina la necesidad de observabilidad custom del agente.

**Tareas técnicas:**

- [ ] Esqueleto ADK en `ml/agent/agent.py` con sub-agentes perceiver y reasoner
- [ ] Módulo `ml/agent/perceiver.py` que envuelve Gemma 4 / Qwen3-VL y emite descripciones estructuradas
- [ ] Módulo `ml/agent/rag.py` con Spatial-RAG híbrido
- [ ] Endpoint FastAPI `/chat` con SSE y evento `perceiver_observation`
- [ ] Pipeline de síntesis de trazas perceiver↔reasoner para fine-tuning supervisado del perceiver (inspirado en Be My Eyes)
- [ ] Tests de integración con queries canónicas

**Estimación:** 5 puntos (~2.5 días, +1 SP vs base por la separación perceiver–reasoner).

---

### US-035 — Variante A: Gemini 3.1 Pro como orquestador cloud

**Como** equipo,
- **quiero** integrar Gemini 3.1 Pro vía Vertex AI como orquestador de alta disponibilidad,
- **para que** la demo sea accesible 24/7 sin dependencia de la ventana H100.

**Criterios de Aceptación:**

- Cliente Vertex AI Gemini configurado con service account y quota (`gemini-3.1-pro`).
- Abstracción `LLMBackend` en `ml/agent/backends.py` que permite intercambiar variante con una flag.
- Latencia p50 < 2 s y p95 < 5 s para queries simples.
- Manejo de errores 429 y 5xx con retry exponencial.
- Configurable desde `config/llm.yaml` y desde la UI del frontend.

**Tareas técnicas:**

- [ ] Cliente Vertex AI con service account en Secret Manager
- [ ] Abstracción `LLMBackend` en `ml/agent/backends.py`

**Estimación:** 1 punto (~0.5 días).

---

### US-036 — Variante B: Qwen3.5-35B-A3B self-hosted en H100 NVL 96GB con vLLM

**Como** equipo,
- **quiero** desplegar **Qwen3.5-35B-A3B** (MoE 35B totales / 3B activos, contexto nativo 128K, licencia Apache 2.0) en Azure H100 NVL 96GB con vLLM como orquestador open-source on-premise,
- **para que** el copiloto sea 100% desplegable en infraestructura propia del usuario (cooperativas agrícolas italianas que no pueden exportar datos a Google Cloud) y diferenciemos el producto frente a Google Earth AI.

**Justificación de la elección (análisis de memoria).** Qwen3.5-35B-A3B es el candidato que cabe con margen en una sola H100 NVL 96GB manteniendo calidad de producción. Con pesos BF16 ocupa aproximadamente 70 GB, deja ~26 GB para KV cache (suficiente para contexto 64K con los nueve tools del agente) y activations. Alternativas más grandes (MiniMax-M2.7 con 230B y NVFP4 ~115 GB, Kimi K2.6 con ~1T) fueron evaluadas y descartadas porque no caben en H100 single-GPU sin degradación severa de calidad o sin multi-GPU.

**Criterios de Aceptación:**

- Modelo descargado desde HuggingFace `Qwen/Qwen3.5-35B-A3B` (variante BF16 oficial, sin sufijo `-Instruct`).
- Serving con vLLM configurado con `--max-model-len 65536` (contexto 64K, realista para el loop agentic multi-turn con 9 tools), `--gpu-memory-utilization 0.92`, `--enable-prefix-caching` activo para acelerar tool calls repetidos, continuous batching.
- **Despliegue durante ventana H100 V5 (~16 h):** 2 h para setup inicial, descarga y benchmark de latencia; 4-5 h opcionales para LoRA fine-tune rank 16 en BF16 sobre un dataset de 500-1000 trazas de tool calls extraídas de logs del agente con Gemini 3.1 Pro; 1 h para evaluación post fine-tune.
- Endpoint FastAPI `/v1/chat/completions` compatible con la API OpenAI que expone el modelo servido por vLLM, permitiendo intercambio transparente con Gemini 3.1 Pro desde el mismo código cliente (ADK soporta backends OpenAI-compatible).
- Latencia objetivo: p50 < 2 s y p95 < 5 s en queries simples de un solo turno; p95 < 15 s en queries multi-turno con 3-5 tool calls.
- Script `scripts/serve_qwen35.sh` que inicia el servicio vLLM, verifica health y publica el endpoint en el service discovery interno.

**Presupuesto de VRAM validado:**

- Pesos Qwen3.5-35B-A3B BF16: ~70 GB
- KV cache para context 64K en arquitectura MoE con GQA estimada: ~13 GB
- Activaciones + overhead vLLM y CUDA: ~8 GB
- **Total ocupado:** ~91 GB sobre 96 GB disponibles (margen de 5 GB para picos y prefix cache)

**Tareas técnicas:**

- [ ] Script de descarga vía `huggingface_hub.snapshot_download` con caché en Azure Blob para reusar entre ventanas
- [ ] vLLM launcher con los parámetros óptimos calibrados
- [ ] Benchmark de latencia vs batch size (1, 2, 4) y context length (16K, 32K, 64K) reportado en MLflow
- [ ] Smoke test post-launch que verifica el endpoint contra una query canónica
- [ ] Opcional: LoRA fine-tune script `ml/train/train_qwen35_tool_traces.py`
- [ ] Documentación del procedimiento de arranque y apagado en `docs/serving/qwen35.md`

**Licencia / legal:** Apache 2.0 vía HuggingFace. Uso académico y comercial permitido sin restricciones más allá de los términos estándar Apache 2.0. Atribución recomendada a Alibaba Qwen Team en publicaciones.

**Estimación:** 2 puntos (~1 día).

---

### US-037 — Evaluación del copiloto en AgroMind y GeoAnalystBench

**Como** equipo,
- **quiero** evaluar las dos variantes del agente en benchmarks estándar,
- **para que** los Avances 6 y 7 (y eventualmente el Paper Track) tengan métricas cuantitativas comparables.

**Criterios de Aceptación:**

- AgroMind subset de 500 pares evaluado con cada variante (A y B); métricas: exact match, F1-SQuAD, BERTScore, tool-call accuracy, hallucination rate (LLM-as-judge con Gemini 3.1 Pro).
- GeoAnalystBench evaluado en modo plan-and-react.
- Tabla comparativa A vs B con error bars sobre 3 corridas.
- Análisis de latencia y costo por query.

**Tareas técnicas:**

- [ ] Harness de evaluación `ml/eval/agent_bench.py`
- [ ] Ejecución en ventana H100 V5 compartida con serving de Qwen3.5
- [ ] Reporte HTML con comparativa y error bars

**Estimación:** 2 puntos (~1 día).

---

**Subtotal EPIC 7: 14 story points** (13 baseline + 1 SP por arquitectura perceiver–reasoner).

---

## EPIC 8: Backend API + Worker Pub/Sub + Tiling {#epic-8}

**Objetivo.** Exponer la plataforma como API REST, servir tiles dinámicos para el frontend y procesar inferencias pesadas asíncronamente.

**Alineado con.** Avance 6 — despliegue.

**Puntos totales de la épica: 9.**

---

### US-038 — API REST FastAPI con endpoints de plataforma

**Como** equipo,
- **quiero** una API REST documentada con OpenAPI 3.1 que exponga los endpoints de la plataforma,
- **para que** tanto el frontend Nuxt 4 como clientes terceros (evaluadores, sponsor) consuman la plataforma de forma consistente.

**Criterios de Aceptación:**

Endpoints expuestos con validación Pydantic y autenticación JWT:

- `POST /auth/session` — intercambio OAuth Clerk por JWT interno.
- `POST /aois` — crea AOI desde GeoJSON.
- `GET /aois/{id}/segment` — ejecuta segmentación (EPIC 6) con la variante seleccionada.
- `GET /aois/{id}/timeseries?index={NDVI|NDWI|NDMI}` — serie temporal.
- `POST /chat` — SSE streaming que invoca el agente Google ADK (EPIC 7).
- `GET /stac/search` — catálogo STAC (EPIC 1).
- `GET /tiles/{z}/{x}/{y}.png` — proxy a TiTiler.
- `POST /llm/switch` — cambia variante A/B del orquestador (requerido para demo).

OpenAPI 3.1 auto-generado; rate limiting Redis (60 req/min por usuario); RBAC multi-tenant con RLS PostgreSQL.

**Tareas técnicas:**

- [ ] Routers FastAPI organizados por dominio (`auth`, `aois`, `chat`, `stac`, `tiles`, `llm`)
- [ ] Middleware JWT + rate limiting
- [ ] Tests de integración con `httpx.AsyncClient`

**Estimación:** 4 puntos (~2 días).

---

### US-039 — TiTiler para tiling COG dinámico

**Como** frontend,
- **quiero** tiles PNG/WebP generados on-the-fly desde COGs en GCS,
- **para que** el mapa MapLibre muestre overlays NDVI/NDWI sin pre-renderizar todas las combinaciones.

**Criterios de Aceptación:**

- TiTiler deployado en Cloud Run con GDAL configurado para GCS.
- Endpoint `/cog/tiles/{z}/{x}/{y}.png?url={cog_url}&expression=(B8-B4)/(B8+B4)&rescale=-1,1&colormap=RdYlGn` funcional.
- Cache Redis 15 min por tile (clave hash del endpoint).
- CORS configurado para dominio frontend.

**Tareas técnicas:**

- [ ] Dockerfile TiTiler con GDAL + rio-tiler
- [ ] Deploy Cloud Run con min=0
- [ ] Configurar CORS y cache headers

**Estimación:** 3 puntos (~1.5 días).

---

### US-040 — Worker de inferencia con cola Pub/Sub

**Como** equipo,
- **quiero** un worker Cloud Run GPU L4 que consume mensajes Pub/Sub para inferencias pesadas,
- **para que** el API FastAPI no bloquee al usuario y la escalabilidad sea horizontal.

**Criterios de Aceptación:**

- Worker escucha topic `inference-jobs` con schema `{aoi_geojson, model_id, params}`.
- Resultados persistidos en GCS y notificación publicada en topic `inference-results`.
- Reintentos automáticos con DLQ (dead letter queue) tras 3 fallos.
- Logging estructurado con `job_id` trazable.
- El frontend recibe notificación vía SSE cuando job completa.

**Tareas técnicas:**

- [ ] Worker `ml/workers/inference_worker.py` con subscripción Pub/Sub
- [ ] Dockerfile con GPU L4 runtime
- [ ] DLQ topic + alerta Cloud Monitoring

**Estimación:** 2 puntos (~1 día).

---

**Subtotal EPIC 8: 9 story points.**

---

## EPIC 9: Frontend Web + Mapa + Chat Bilingüe + Switch A/B {#epic-9}

**Objetivo.** Construir la interfaz web impactante para la presentación final, con i18n italiano/español/inglés nativo y switch A/B de variante LLM en vivo.

**Alineado con.** Avance 6 + Presentación Final.

**Puntos totales de la épica: 10.**

---

### US-041 — Layout split-screen con mapa y chat

**Como** usuario,
- **quiero** un layout con mapa a la izquierda y chat a la derecha,
- **para que** pueda dibujar AOIs y conversar en paralelo.

**Criterios de Aceptación:**

- **Nuxt 4 con SSR puro** (sin PWA, sin Tauri); routing file-based; página principal `pages/index.vue`.
- MapLibre GL con basemap Esri World Imagery satelital + OSM alternative.
- `maplibre-gl-draw` wrapper Vue para dibujo de polígonos reactivos.
- Panel derecho con `@ai-sdk/vue` composable `useChat()` conectado al endpoint SSE `/chat`.
- Dark/light mode toggle con Nuxt UI Pro.

**Tareas técnicas:**

- [ ] Layout `layouts/default.vue` con split-screen
- [ ] Componente `MapView.vue` con MapLibre GL
- [ ] Componente `ChatPanel.vue` con `useChat`

**Estimación:** 3 puntos (~1.5 días).

---

### US-042 — Internacionalización italiano/español/inglés

**Como** usuario italiano o hispanohablante,
- **quiero** la interfaz en mi idioma nativo,
- **para que** el copiloto sea usable sin barrera idiomática.

**Criterios de Aceptación:**

- `@nuxtjs/i18n` con tres locales (`it`, `es`, `en`).
- Archivos `locales/{it,es,en}.json` con al menos 150 strings cubriendo navegación, mensajes de chat y errores.
- Detección automática del idioma del navegador con toggle manual persistente.
- Rutas localizadas `/it/...`, `/es/...`, `/en/...`.
- Queries al agente en el idioma del usuario (Gemma 4 responde nativamente en los tres).

**Tareas técnicas:**

- [ ] Configurar `@nuxtjs/i18n` en `nuxt.config.ts`
- [ ] Generar archivos `locales/*.json` con traducciones iniciales
- [ ] Componente `LocaleSwitcher.vue`

**Estimación:** 2 puntos (~1 día).

---

### US-043 — Overlays NDVI/NDWI/AlphaEarth y timeline

**Como** usuario,
- **quiero** activar overlays interactivos de índices espectrales sobre el mapa,
- **para que** vea visualmente el vigor y la humedad.

**Criterios de Aceptación:**

- Layer switcher con NDVI, NDWI, NDMI, EVI, True Color, False Color Infrared.
- Visualización opcional de AlphaEarth clusters (resultado de k-means sobre los 64-dim).
- Slider temporal con calendario (meses disponibles según catálogo STAC).
- Leyenda con colormap y tooltip al hover con el valor puntual.

**Tareas técnicas:**

- [ ] Componente `LayerSwitcher.vue`
- [ ] Integración con TiTiler vía URLs dinámicas
- [ ] Componente `TimeSlider.vue`

**Estimación:** 2 puntos (~1 día).

---

### US-044 — Chat UI con streaming, tool calls y citaciones

**Como** usuario,
- **quiero** ver qué tools llama el agente y las citaciones de las respuestas,
- **para que** pueda confiar en el copiloto y depurar respuestas si algo parece incorrecto.

**Criterios de Aceptación:**

- Mensajes con renderizado markdown (incluyendo tablas y bloques de código).
- Tool calls colapsables con input/output JSON formateados.
- Citaciones como links a metadata de escenas o documentos.
- Thumbnails de imágenes satelitales referenciadas cuando el tool las incluye.
- Gráficas ECharts inline generadas por el agente (serie temporal NDVI como ejemplo).

**Tareas técnicas:**

- [ ] Componente `ChatMessage.vue` con markdown + tool call collapse
- [ ] Componente `ToolCallBox.vue` con JSON formatter
- [ ] Integración `vue-echarts` para gráficas inline

**Estimación:** 2 puntos (~1 día).

---

### US-045 — Switch A/B variante LLM en UI

**Como** evaluador (demo presentación),
- **quiero** cambiar en vivo entre Gemini 3.1 Pro y Qwen3.5-35B-A3B,
- **para que** la demo muestre explícitamente el diferenciador open-source del proyecto.

**Criterios de Aceptación:**

- Selector visible en la UI con las dos variantes (y un aviso si Qwen3.5 está inactivo por ventana H100 cerrada).
- El cambio aplica inmediatamente a la siguiente query.
- Endpoint backend `POST /llm/switch` valida disponibilidad antes de aceptar.

**Tareas técnicas:**

- [ ] Componente `LLMSwitcher.vue` con badge de disponibilidad
- [ ] Health check del endpoint Qwen3.5 cada 30 s

**Estimación:** 1 punto (~0.5 días).

---

**Subtotal EPIC 9: 10 story points.**

---

## EPIC 10: Observabilidad, Evidently Drift, FinOps, Seguridad y Documentación {#epic-10}

**Objetivo.** Cumplir con los cuestionamientos de viabilidad de producción, análisis costo-beneficio y riesgos requeridos por los Avances 6 y 7 del curso, más los aspectos de seguridad y documentación del proyecto.

**Alineado con.** Avance 6 (7 junio 2026) y Avance 7 (14 junio 2026).

**Puntos totales de la épica: 8.** El tracing built-in de Google ADK absorbe la observabilidad del agente, por lo que la épica se concentra en métricas técnicas del sistema, drift de datos, FinOps y seguridad.

---

### US-046 — Dashboard de observabilidad con Prometheus y Grafana

**Como** operador,
- **quiero** métricas técnicas en tiempo real del sistema,
- **para que** cualquier anomalía (latencia, error rate, GPU util) sea visible para el equipo.

**Criterios de Aceptación:**

- Métricas exportadas por FastAPI con `prometheus-client`: latencia p50/p95/p99, RPS, error rate por endpoint, GPU utilization del worker L4, tool-call success rate del agente (integrado con **ADK tracing**), hallucination rate estimada (LLM-as-judge muestra un 5% de queries).
- Dashboards Grafana con tres paneles: API, worker ML, data pipeline.
- Alertas configuradas (vía PagerDuty Free o email): p99 latencia > 3 s, GPU OOM, error rate > 5%.

**Tareas técnicas:**

- [ ] Instrumentación FastAPI con `prometheus-client`
- [ ] Dashboards Grafana en `infrastructure/grafana/`
- [ ] Alertas Cloud Monitoring

**Estimación:** 2 puntos (~1 día).

---

### US-047 — Drift detection con Evidently AI

**Como** ML Engineer,
- **quiero** detectar drift en bandas Sentinel-2 y en predicciones del modelo,
- **para que** el equipo sepa cuándo reentrenar en el futuro.

**Criterios de Aceptación:**

- Drift de distribución de bandas Sentinel-2 (KS test) y AlphaEarth embeddings (MMD).
- Drift de distribución de clases predichas (Chi-cuadrado).
- Reporte HTML semanal automático publicado en `gs://agrosat-reports/drift/`.
- Alerta si drift score > 0.3.
- **Integrado como asset Dagster `drift_check`** que corre semanalmente con dependencia de los assets de ingesta.

**Tareas técnicas:**

- [ ] Pipeline Evidently en `ml/monitoring/drift.py`
- [ ] Asset Dagster `drift_check` con schedule semanal
- [ ] Notificación por email si drift score > umbral

**Estimación:** 2 puntos (~1 día).

---

### US-048 — Análisis costo-beneficio para Avances 6 y 7

**Como** equipo,
- **quiero** tablas de costos y beneficios cuantificables,
- **para que** el criterio "Costos" (20 pts), "Beneficios" (20 pts) e "Implementación" (30 pts) de las rúbricas de Avance 6 y 7 queden cubiertos.

**Criterios de Aceptación:**

- Tabla de costos por fase CRISP-ML(Q) reales del proyecto + proyección 12 meses: adquisición de datos ($0 fuentes públicas), training ($262-602 H100 + L4), serving (~$115/mes), Gemini API (~$12/mes), Qwen3.5 self-hosted infra (~$30/mes en ventanas), etc.
- Tabla de beneficios cuantificables para cliente tipo 500 ha: horas ahorradas de agrónomo/mes, % ahorro de agua con detección de estrés hídrico, ahorro de insumos por fertilización focalizada, reducción de tiempo de detección de plagas.
- Beneficios intangibles: trazabilidad para cumplimiento CAP europeo, reducción de riesgo regulatorio, imagen sostenibilidad.
- ROI break-even estimado en mes 3 para cliente tipo.

**Tareas técnicas:**

- [ ] Documento `docs/business/costo_beneficio.md`
- [ ] Tablas en Excel + export a LaTeX para paper

**Estimación:** 1 punto (~0.5 días).

---

### US-049 — Análisis de riesgos categorizados para Avance 7

**Como** equipo,
- **quiero** análisis exhaustivo de riesgos por categoría,
- **para que** el criterio "Riesgos" (20 pts) de la rúbrica del Avance 7 quede cubierto.

**Criterios de Aceptación:**

- Cuatro categorías de riesgos según rúbrica del curso: datos (disponibilidad CDSE, calidad labels, cobertura de nubes), ataques (adversarial attacks en modelos, DDoS API), confianza (hallucinations, sesgos regionales, falsas alarmas), cumplimiento (GDPR, licencias, políticas Copernicus).
- Cada riesgo con probabilidad (Alta/Media/Baja), impacto (Alto/Medio/Bajo) y mitigación concreta y accionable.

**Tareas técnicas:**

- [ ] Documento `docs/risks/riesgos.md`
- [ ] Matriz probabilidad × impacto visual

**Estimación:** 1 punto (~0.5 días).

---

### US-050 — Análisis comparativo de proveedores cloud

**Como** equipo,
- **quiero** justificar la elección multi-cloud con análisis comparativo,
- **para que** el criterio "Implementación" (30 pts) del Avance 6 quede cubierto (rúbrica exige mínimo 2 proveedores).

**Criterios de Aceptación:**

- Comparativa GCP vs Azure (mínimo rúbrica) + opcionalmente AWS e IBM Cloud con al menos cinco factores: precio GPU H100 on-demand y spot, ecosistema de Earth Observation (GCP Earth Engine vs Azure Planetary Computer vs AWS Open Data), latencia hacia Europa (target Italia), soporte de pipelines MLOps (Vertex AI Pipelines, Azure ML, SageMaker), disponibilidad de partnerships académicos.
- Decisión justificada: GCP primario + Azure H100 on-demand.

**Tareas técnicas:**

- [ ] Documento `docs/cloud/comparativa_proveedores.md`

**Estimación:** 1 punto (~0.5 días).

---

### US-051 — Seguridad y documentación final

**Como** equipo,
- **quiero** mejores prácticas de seguridad implementadas y documentación consolidada,
- **para que** el sistema sea production-ready y el tercero pueda reproducirlo.

**Criterios de Aceptación:**

- HTTPS obligatorio con Cloud Load Balancer y certificados managed.
- JWT con rotación cada 15 minutos y refresh tokens.
- RLS PostgreSQL por tenant (multitenant).
- Secretos nunca en git (pre-commit hook `detect-secrets`).
- Revisión OWASP Top 10 documentada en `docs/security.md`.
- Penetration test manual básico (nikto, nmap) antes de presentación.
- Model Cards publicadas para **Gemma 4 fine-tuned** y el modelo final ensemble en `docs/model_cards/`.
- Data Sheets por dataset en `docs/data_sheets/`.
- ADRs (Architecture Decision Records) en `docs/decisions/`.
- **Glosario técnico** en `docs/glosario.md` con estandarización de términos IT/ES/EN.
- README reproducible con instrucciones de setup, running y testing end-to-end.

**Tareas técnicas:**

- [ ] Configurar Cloud Load Balancer + cert managed
- [ ] Implementar JWT refresh en FastAPI
- [ ] Escribir Model Card Gemma 4 fine-tuned siguiendo template HuggingFace
- [ ] Documentar los cinco ADRs iniciales (Gemma 4 como VLM, Google ADK, Dagster, dbmate, Nuxt 4 SSR)

**Estimación:** 1 punto (~0.5 días).

---

**Subtotal EPIC 10: 8 story points.**

---

## EPIC 11: Paper Track — Semanas 10-11 Post-Presentación (Opcional) {#epic-11}

**Objetivo.** Redactar y submittear el paper a venue académico, ejecutado del 22 de junio al 3 de julio de 2026 post-presentación. **NO afecta entregables del curso.**

**Alineado con.** Esta épica es completamente externa al Proyecto Integrador. Se ejecuta después de la Presentación Final del 21 de junio y dentro de las dos semanas calendario que restan del trimestre hasta el 3 de julio, o asincrónicamente post-clase si se requiere más tiempo.

**Capacidad estimada:** 3 devs × 8 h/semana (dedicación reducida post-curso) × 2 semanas = 48 horas ≈ 20 SP realistas; con dedicación extra de miembros individuales part-time la capacidad sube a ~28 SP.

**Puntos totales de la épica: 28.**

---

### US-052 — Construcción de benchmark AgroMind-IT/ES (500 pares)

**Como** equipo,
- **quiero** construir y publicar un benchmark bilingüe italiano/español con 500 pares Q&A agrícolas,
- **para que** sea contribución académica original publicable con DOI.

**Criterios de Aceptación:**

- 250 pares en italiano + 250 en español cubriendo las diez familias de preguntas del catálogo del copiloto (clasificación, cuantificación, vigor, estrés hídrico, fenología, comparación, anomalías, metadata, intersecciones, explicabilidad).
- Seed inicial generado sintéticamente con Gemini 3.1 Pro sobre imágenes reales Sentinel-2 de Italia.
- Revisión manual por hablantes nativos: italiano por reviewer de Scuola Sant'Anna (vía sponsor), español por miembro del equipo.
- Publicación en Zenodo con DOI y licencia CC-BY-4.0.
- Esquema JSONL compatible con AgroMind original para facilitar re-uso.

**Tareas técnicas:**

- [ ] Script de generación sintética con Gemini
- [ ] Interfaz Streamlit para revisión humana
- [ ] Upload a Zenodo con metadata completa

**Estimación:** 6 puntos.

---

### US-053 — Evaluación comparativa en GEO-Bench-2, AgroMind y AgroMind-IT/ES

**Como** equipo,
- **quiero** evaluar rigurosamente las dos variantes (Gemini 3.1 Pro y Qwen3.5-35B-A3B) en tres benchmarks,
- **para que** la tabla de resultados del paper tenga error bars estadísticamente significativos.

**Criterios de Aceptación:**

- GEO-Bench-2 sobre las tasks agrícolas relevantes (≥3 de las 19 disponibles).
- AgroMind subset 1000 pares.
- AgroMind-IT/ES 500 pares.
- Métricas por variante: accuracy, F1, BERTScore, tool-call accuracy, hallucination rate, latencia p50/p95, costo por query.
- Tres corridas independientes con error bars y test Wilcoxon signed-rank para comparación pareada.

**Tareas técnicas:**

- [ ] Harness extendido `ml/eval/paper_bench.py`
- [ ] Ejecución en ventana H100 post-presentación (reutilizando serving Qwen3.5)
- [ ] Exportación tabla LaTeX

**Estimación:** 6 puntos.

---

### US-054 — Figuras y tablas reproducibles del paper

**Como** equipo,
- **quiero** las figuras y tablas del paper generadas desde notebooks Python reproducibles,
- **para que** reviewers y lectores puedan regenerar cada resultado.

**Criterios de Aceptación:**

- Ocho figuras clave: arquitectura, mapas AOI Italia, UMAP AlphaEarth, curvas de entrenamiento Gemma 4, ejemplos conversacionales (IT/ES/EN), matriz de confusión, barplot de benchmarks, mapa de error espacial.
- Cinco tablas clave: comparativa de FMs, modelos individuales EPIC 5, ensambles EPIC 6, benchmark LLMs, ablación de tools.
- Cada figura/tabla generada desde `paper/notebooks/*.ipynb` con seed fijo y datos versionados en DVC.

**Tareas técnicas:**

- [ ] Plantillas matplotlib con estilo científico (CVPR/ISPRS)
- [ ] Notebooks reproducibles con `papermill`
- [ ] Exportación SVG + PNG de alta resolución

**Estimación:** 6 puntos.

---

### US-055 — Redacción, revisión y submission

**Como** equipo,
- **quiero** redactar el paper en LaTeX, revisarlo con el sponsor y enviarlo a venue,
- **para que** el trabajo trascienda el curso.

**Criterios de Aceptación:**

- Paper 10-15 páginas en Overleaf, template Remote Sensing MDPI (prioridad) o ISPRS Journal (ambicioso).
- Estructura: Abstract (250 palabras), Introduction, Related Work, Method, Experiments, Results, Discussion, Conclusion, References, Appendix.
- Revisión por Dr. Camacho antes de submission.
- Submission a arXiv cs.CV como pre-print (garantiza prioridad temporal).
- Submission a uno de los venues priorizados en orden: Remote Sensing MDPI (rolling), CVPR EarthVision Workshop 2026 si el deadline lo permite, ISPRS Journal.
- Repositorio GitHub público con README reproducible y licencia Apache 2.0.

**Tareas técnicas:**

- [ ] Overleaf project con template MDPI
- [ ] Revisión ortográfica + gramática en inglés con Grammarly
- [ ] Respuesta a revisores (iterativa post-submission)

**Estimación:** 10 puntos.

---

**Subtotal EPIC 11: 28 story points.**

---

## 11. Roadmap de Sprints Semanales {#11-roadmap}

Trimestre 20-abr-2026 a 3-jul-2026. **Sprints semanales** para alinearse con entregas casi semanales del curso.

### Sprint 1 — Semana del 20 al 26 de abril

**Objetivo.** Cerrar Avance 0 (PDF del Planteamiento) + setup de infraestructura base.
**Story points planeados:** 10 (E0) + 5 (Avance 0 docx) = 15.

| Día | Actividad principal |
|-----|---------------------|
| Lun 20-abr | Kickoff, asignación roles, gates del Sprint 1 (ver sección 12) |
| Mar 21-abr | E0 US-001 cookiecutter + E0 US-002 docker-compose |
| Mié 22-abr | E0 US-003 Terraform GCP + Azure |
| Jue 23-abr | E0 US-004 DVC + MLflow + Dagster + dbmate + E0 US-005 CI/CD |
| Vie 24-abr | Pulir Planteamiento PDF: integrar el glosario técnico y consolidar el stack final (Gemma 4, Google ADK, Dagster) |
| Sáb 25-abr | Revisión equipo + Dr. Camacho |
| **Dom 26-abr** | **Avance 0 entregado** |

### Sprint 2 — Semana del 27 abril al 3 de mayo

**Objetivo.** Ingesta + arranque EDA.
**Story points:** 12 (E1) + 5 (E2 US-010 inicio) = 17.

| Actividades |
|---|
| E1 US-006 AlphaEarth, US-007 Sentinel-2, US-008 DINOv3+PASTIS+DW, US-009 catálogo STAC |
| E2 US-010 empezar EDA univariado |
| **Dom 3-may: Avance 1 entregado (EDA inicial con las 10 preguntas guía)** |

### Sprint 3 — Semana del 4 al 10 de mayo

**Objetivo.** Completar EDA + arrancar FE.
**Story points:** 9 (resto E2) + 4 (E3 US-014) = 13.

| Actividades |
|---|
| E2 US-011 EDA AlphaEarth, US-012 bivariado/temporal, US-013 dashboard/PDF |
| E3 US-014 biblioteca índices espectrales |

### Sprint 4 — Semana del 11 al 17 de mayo

**Objetivo.** FE completo + arranque Baseline.
**Story points:** 10 (resto E3) + 5 (E4 inicio) = 15.

| Actividades |
|---|
| E3 US-015 features temporales, US-016 fusión multisensor, US-017 selección/extracción |
| E4 US-018 empezar RF+XGB |
| **Dom 17-may: Avance 2 entregado (Feature Engineering)** |

### Sprint 5 — Semana del 18 al 24 de mayo

**Objetivo.** Baseline completo + primeros modelos alternativos.
**Story points:** 5 (resto E4) + 10 (E5 US-022 a US-024) = 15.

**GPU:** Ventana V1 (noches 18-20 may, 8-12 h L4) para baselines. Si se tiene acceso anticipado H100, ventana V2 preliminar (12 h TSViT/U-TAE).

| Actividades |
|---|
| E4 US-019 SHAP, US-020 curvas, US-021 notebook comparativo |
| E5 US-022 U-Net, US-023 DeepLabv3+, US-024 SegFormer |
| **Mié 20-may: Avance 3 entregado (Baseline)** |
| **Dom 24-may: Avance 4 entregado (6 Modelos + comparativa)** — se acepta que los últimos 3 modelos se reporten en forma inicial y se afinen en S6 |

### Sprint 6 — Semana del 25 al 31 de mayo

**Objetivo.** Completar 6 modelos + ensambles + Gemma 4 LoRA.
**Story points:** 10 (resto E5) + 20 (E6) = 30. **Sprint más pesado.**

**GPU:** Ventana V2 (noches 25-27 may, ~12 h H100) U-TAE + TSViT + Swin-UNETR. Ventana V3 (noches 28-30 may, ~24 h H100) **Gemma 4 26B-MoE LoRA**.

| Actividades |
|---|
| E5 US-025 U-TAE, US-026 TSViT, US-027 Swin-UNETR, US-028 ajuste fino |
| E6 US-029 Gemma 4 LoRA, US-030 adaptador seg, US-031 4 ensambles, US-032 gráficas |
| **Dom 31-may: Avance 5 entregado (Modelo final + ensambles)** |

### Sprint 7 — Semana del 1 al 7 de junio

**Objetivo.** Agente ADK + backend.
**Story points:** 13 (E7) + 9 (E8) = 22.

**GPU:** Ventana V4 (noches 1-3 jun, ~12 h H100) Qwen3-VL-30B-A3B LoRA para comparación opcional + ensambles re-run. Ventana V5 (noches 5-7 jun, ~16 h H100) Qwen3.5-35B-A3B setup vLLM + LoRA tool traces + eval benchmarks.

| Actividades |
|---|
| E7 US-033 tools, US-034 agente ADK + Spatial-RAG, US-035 Gemini, US-036 Qwen3.5 serving, US-037 eval |
| E8 US-038 API FastAPI, US-039 TiTiler, US-040 worker Pub/Sub |
| **Dom 7-jun: Avance 6 entregado (Conclusiones)** |

### Sprint 8 — Semana del 8 al 14 de junio

**Objetivo.** Frontend + observabilidad + resumen ejecutivo.
**Story points:** 10 (E9) + 8 (E10) = 18.

| Actividades |
|---|
| E9 US-041 layout, US-042 i18n, US-043 overlays, US-044 chat UI, US-045 switch A/B |
| E10 US-046 Prometheus/Grafana, US-047 Evidently drift, US-048 costo-beneficio, US-049 riesgos, US-050 cloud comparativa, US-051 seguridad/docs |
| **Dom 14-jun: Avance 7 entregado (Resumen ejecutivo)** |

### Sprint 9 — Semana del 15 al 21 de junio

**Objetivo.** Pulido final + dry-runs + presentación.
**Story points:** ~10 (cierre).

**GPU:** Ventana V6 (noches 18-20 jun, ~8 h H100) warm vLLM + re-runs finales para demo.

| Actividades |
|---|
| Bug fixing crítico, pulido UI, dry-runs de presentación |
| Grabación video demo de 3 minutos |
| Warmup de infra y H100 para demo en vivo |
| **Dom 21-jun: Presentación Final** |

### Sprints 10-11 — Semanas del 22 de junio al 3 de julio

**Ejecución opcional del Paper Track.** E11 US-052 a US-055. Cero impacto en calificación.

### Balance de capacidad

| Sprint | SP planeados | Capacidad (3 devs × 12 h / 2.4 h/SP) | Buffer |
|--------|--------------|--------------------------------------|--------|
| S1 | 15 | 15 | 0 (tight, Avance 0 ya casi escrito como base) |
| S2 | 17 | 15 | -2 (manejable, EDA compartido con ingesta) |
| S3 | 13 | 15 | 2 |
| S4 | 15 | 15 | 0 |
| S5 | 15 | 15 | 0 |
| **S6** | **30** | **15** | **-15 (CRÍTICO, mitigado por overnight H100)** |
| S7 | 22 | 15 | -7 |
| S8 | 18 | 15 | -3 |
| S9 | 10 | 15 | 5 |
| **Total** | **155** | **135** | **-20 global, compensado por buffer de 2 semanas post-presentación** |

**Interpretación.** Los sprints de modelado (S6) y agente (S7) están sobrecomprometidos respecto a capacidad humana por 15 y 7 SP respectivamente. Las tres mitigaciones son:

1. **S6 aprovecha entrenamiento overnight** — el código de Gemma 4 LoRA y ensambles se deja corriendo en H100 mientras el equipo trabaja otros temas de día. El tiempo humano real es ~25 SP efectivos de código + 5 SP de overnight GPU.
2. **S7 usa ADK para reducir el esfuerzo de agente** — la US-034 con Spatial-RAG se mantiene en 4 SP gracias al planner, executor y tracing built-in de ADK.
3. **Sprints 3, 5 y 9 tienen buffer** que absorbe overflow de S6/S7. Los 3 stretch candidates (switch A/B, Evidently pipeline, worker Pub/Sub, 5 SP combinados) son lo primero que se sacrifica si algún sprint crítico se atrasa.

---

## 12. Gates de Sprint 1 {#12-gates}

Validaciones obligatorias en la primera semana del trimestre para evitar sorpresas en sprints posteriores.

| Día | Gate | Criterio de éxito |
|---|---|---|
| Lun 20-abr | Verificar HuggingFace: `google/gemma-4-26b-it`, `google/gemma-4-e4b-it`, `Qwen/Qwen3.5-35B-A3B`, `Qwen/Qwen3-VL-30B-A3B-Instruct`, `facebook/dinov3-vitl16-pretrain-sat493m` | Todos los repos accesibles, licencias aceptadas, `snapshot_download` exitoso en muestra |
| Mar 21-abr | Verificar Vertex AI: modelo `gemini-3.1-pro` accesible desde service account del equipo | Response a prompt curl exitoso |
| Mié 22-abr | AlphaEarth: ejecutar export dummy de 100 km² Toscana 2024 | COG de ~50 MB descargado en GCS |
| Jue 23-abr | **Azure: pedir región al sponsor como gate día 4 del Sprint 1 y documentarla**; booking VM `Standard_NC40ads_H100_v5` spot; hello-world PyTorch + vLLM | Región confirmada (probable West Europe o East US); `nvidia-smi` + `vllm serve` funcional |
| Vie 24-abr | Google ADK: tutorial con FunctionTool dummy corre apuntando a Gemini y a endpoint OpenAI-compat local (mock) | Tool call exitoso en ambos backends |
| Sáb 25-abr | Dagster asset hello-world + MLflow tracking wired | Run aparece en UI de ambos |
| **Dom 26-abr** | **Entrega Avance 0 PDF** | Subido a Canvas como `Avance0.#Equipo.pdf` |

---

## 13. Gestión de Riesgos {#13-riesgos}

Se identifican 14 riesgos con sus mitigaciones. Categorizados según rúbrica del Avance 7 (datos, ataques, confianza, cumplimiento).

### 13.1 Los cinco más severos (resumen ejecutivo)

1. **R01 — Ventanas H100 (80 h) insuficientes para los fine-tunes.** Probabilidad Media, Impacto Alto. Mitigación: plan B con Gemma 4 E4B QLoRA 4-bit en L4 (factible en ~40 h L4 continua), Qwen3.5-9B dense como fallback si el MoE de 35B da problemas, fallback Qwen3-VL-30B-A3B si Gemma 4 26B-MoE presenta dificultades.

2. **R02 — Alucinaciones del agente LLM en respuestas factuales.** Probabilidad Alta, Impacto Alto. Mitigación: tool-calling obligatorio para cualquier dato factual, citaciones siempre, Spatial-RAG híbrido (~30% reducción según GeoAnalystBench), evaluación con LLM-as-judge, **tracing built-in de Google ADK** para auditar cada decisión.

3. **R03 — Sprint 6 sobrecomprometido (30 SP planeados vs 15 SP capacidad humana).** Probabilidad Alta, Impacto Alto. Mitigación: entrenamiento Gemma 4 corre overnight en H100, el equipo trabaja ensambles y gráficas en paralelo durante el día; los últimos 3 modelos del EPIC 5 se reportan en forma preliminar el 24-may y se afinan en S6.

4. **R04 — Construcción de AgroMind-IT/ES requiere validación nativa.** Probabilidad Alta, Impacto Medio. Mitigación: semilla sintética con Gemini 3.1 Pro, reviewer italiano vía Scuola Sant'Anna, reviewer español del equipo; publicación Zenodo puede posponerse post-submission.

5. **R05 — Región Azure H100 con cuota spot agotada.** Probabilidad Media, Impacto Medio. Mitigación: Gate día 4 Sprint 1 con sponsor. Fallback Azure on-demand ($560 USD × 80 h) o GCP A100 spot ($290 USD × 100 h) si H100 imposible.

### 13.2 Los 14 riesgos completos con categorización

| ID | Categoría | Riesgo | Prob. | Impacto | Mitigación |
|----|-----------|--------|-------|---------|------------|
| R01 | Datos | Ventanas H100 insuficientes | Media | Alto | Plan B Gemma 4 E4B QLoRA 4-bit L4 + Qwen3.5-9B dense fallback |
| R02 | Confianza | Alucinaciones agente en respuestas factuales | Alta | Alto | Tool-calling + Spatial-RAG + citaciones + ADK tracing + LLM-as-judge |
| R03 | Datos | Sprint 6 sobrecomprometido | Alta | Alto | Entrenamiento overnight + paralelización ensambles + fallback 3 stretch candidates |
| R04 | Confianza | Validación nativa AgroMind-IT/ES | Alta | Medio | Semilla sintética Gemini + reviewer Scuola Sant'Anna + reviewer ES equipo |
| R05 | Datos | Cuota Azure H100 spot agotada | Media | Medio | Gate día 4 S1 + fallback on-demand o GCP A100 |
| R06 | Datos | Rate limits CDSE en descarga masiva S-2 | Media | Medio | Descarga incremental con backoff + cache GCS + ventanas nocturnas |
| R07 | Datos | Labels ruidosos Dynamic World | Media | Medio | Validación cruzada con GSAA Italia + priorizar PASTIS-R para entrenamiento |
| R08 | Datos | Nubosidad persistente Pianura Padana afecta S-2 | Alta | Medio | Fusión con Sentinel-1 SAR + ventana temporal amplia + cloud mask s2cloudless |
| R09 | Cumplimiento | Disponibilidad GSAA Italia por región | Media | Bajo | Descarga temprana S2 + contacto vía sponsor con AGEA si falla |
| R10 | Cumplimiento | GDPR con datos futuros de productores | Baja | Alto | Anonimización a nivel parcela + convenio formal antes de ingesta |
| R11 | Cumplimiento | Licencias Copernicus / GEE ToS cambian | Baja | Medio | Atribuciones documentadas desde D1, monitoreo trimestral de ToS |
| R12 | Confianza | Sesgos regionales en Gemma 4 fine-tuned | Media | Medio | Cross-validation espacial + Model Card con limitaciones documentadas |
| R13 | Datos | Ausencia de un miembro del equipo por enfermedad | Media | Alto | Pair programming, documentación diaria, ramas `feature/*` con commits frecuentes |
| R14 | Ataques | DDoS o abuso del endpoint `/chat` público | Baja | Medio | Rate limiting Redis 60 req/min por usuario + Cloud Armor |

**Nota sobre disponibilidad de modelos en HuggingFace:** verificada el 24-abr-2026. Todos los modelos seleccionados están confirmados con sus IDs reales y licencias accesibles.

---

## 14. Criterios de Éxito del MVP {#14-criterios}

### 14.1 Métricas técnicas

- Baseline F1-macro ≥ 0.60 sobre AlphaEarth + XGBoost.
- Modelo final F1-macro ≥ 0.80 (**Gemma 4 26B-MoE LoRA** + ensambles).
- mIoU ≥ 0.70 en segmentación densa del modelo final.
- Latencia inferencia p95 < 5 s por parcela de 5 ha.
- Latencia chat p95 < 3 s queries simples; < 15 s multi-step.
- Cobertura tests ≥ 70% backend, ≥ 50% frontend.
- Score AgroMind ≥ 0.70 variante Qwen3.5; ≥ 0.75 variante Gemini 3.1 Pro.
- Cero alertas de drift Evidently en la semana previa a presentación.

### 14.2 Métricas de producto

- Usuario dibuja polígono y obtiene respuesta en < 10 s.
- Overlays NDVI/NDWI/NDMI + AlphaEarth clusters interactivos.
- Chat soporta ≥ 10 tipos de preguntas en italiano, español, inglés.
- Switch A/B variante LLM funcional en UI.
- Video demo end-to-end de 3 minutos grabado.

### 14.3 Métricas de MLOps

- 100% datasets versionados con DVC.
- 100% experimentos trackeados en MLflow.
- 100% pipelines orquestados con Dagster assets.
- 100% commits a main desplegables automáticamente.
- 100% infraestructura declarada en Terraform.
- Reproducibilidad completa en GCP y Azure.

### 14.4 Métricas del Paper Track (opcional, semanas 10-11)

- AgroMind-IT/ES 500 pares publicado Zenodo con DOI.
- Tabla comparativa 2 LLMs × 3 benchmarks con error bars significativos.
- Paper enviado arXiv y al menos un venue priorizado.
- Repo público con README reproducible.

---

## 15. Alineación con Rúbricas del Curso {#15-rubricas}

### 15.1 Checklist por Avance

**Avance 0 — Propuesta y convenios (26 abr 2026, S1, 100 pts):** portada con datos del equipo; título AgroSatCopilot; empresa Tec + Scuola Sant'Anna; sector 111 Agricultura; ubicación Italia con 3 regiones; dominio CV + NLP + Predictivo + Recomendación; antecedentes con los 2 papers del profesor y 30+ fuentes del estado del arte; entendimiento del negocio; entendimiento de los datos con AlphaEarth prominente; convenios; bibliografía IEEE.

**Avance 1 — EDA (3 may 2026, 100 pts):** 10 preguntas guía de la rúbrica respondidas en notebooks secuenciales; análisis univariado, bivariado y temporal con Polars; análisis específico de embeddings AlphaEarth; conclusiones CRISP-ML(Q) Data Understanding; repo GitHub compartido.

**Avance 2 — Feature Engineering (17 may 2026, 100 pts):** construcción 30 pts con 17 índices espectrales justificados; normalización 30 pts con transformaciones justificadas por histograma; selección/extracción 30 pts con VIF, PCA, FA, UMAP, feature importance; conclusiones CRISP-ML(Q) Data Preparation 10 pts.

**Avance 3 — Baseline (20 may 2026, 100 pts):** algoritmo 40 pts con RF y XGB sobre AlphaEarth + features combinados y justificación; feature importance con SHAP 20 pts; sub/sobreajuste 10 pts con curvas de aprendizaje y validación; métrica F1-macro + mIoU justificadas 20 pts; desempeño mínimo F1-macro ≥ 0.60 establecido 10 pts.

**Avance 4 — Modelos alternativos (24 may 2026, 100 pts):** 6 modelos individuales (U-Net, DeepLabv3+, SegFormer, U-TAE, TSViT, Swin-UNETR) con tabla comparativa 60 pts; ajuste fino Optuna 30+ trials sobre los 2 mejores 30 pts; modelo individual final justificado 10 pts.

**Avance 5 — Modelo final (31 may 2026, 100 pts):** 4 ensambles (homogéneos y heterogéneos) 60 pts; selección con tabla comparativa 20 pts; ≥4 gráficos interpretados 20 pts.

**Avance 6 — Conclusiones (7 jun 2026, 100 pts):** análisis del modelo vs criterios de éxito 50 pts; accionables para stakeholders 20 pts; análisis comparativo de al menos 2 proveedores cloud 30 pts.

**Avance 7 — Resumen ejecutivo (14 jun 2026, 100 pts):** síntesis entrelazada de avances 40 pts; costos por fase CRISP-ML(Q) 20 pts; beneficios cuantificables e intangibles 20 pts; riesgos categorizados en datos, ataques, confianza y cumplimiento 20 pts.

**Presentación final (21 jun 2026, 100 pts):** calidad de diapositivas 10 pts; profundidad de análisis con referencia a CRISP-ML(Q) 30 pts; comprensión de la solución con trade-offs de ensambles 30 pts; contexto de negocio con métrica vinculada a objetivo comercial 30 pts.

### 15.2 Mapeo Épicas → Rúbrica

| Épica | Avance(s) cubiertos | Criterios rúbrica satisfechos |
|-------|---------------------|-------------------------------|
| E0 | A0 | Convenios, infraestructura reproducible |
| E1 | A0, A1 | Entendimiento de datos (volumen, fuentes, licencias) |
| E2 | A1 | 10 preguntas guía EDA |
| E3 | A2 | Construcción 30 + Normalización 30 + Selección 30 + Conclusiones 10 |
| E4 | A3 | Algoritmo 40 + Features 20 + Sub/sobreajuste 10 + Métrica 20 + Desempeño 10 |
| E5 | A4 | Comparativa 60 + Ajuste fino 30 + Modelo individual 10 |
| E6 | A5 | Ensambles 60 + Selección 20 + Gráficos 20 |
| E7 | A5, A6 | Tool-calling + eval benchmarks + tracing ADK |
| E8 | A6 | Implementación cloud + APIs |
| E9 | A6, Pres | Demo impactante + switch A/B visible |
| E10 | A6, A7 | Costos + Beneficios + Riesgos + 2 proveedores cloud |
| E11 | Post-curso | Contribución científica original (opcional) |

---

## 16. Apéndice: Decisiones Técnicas Clave {#16-apendice}

### A.1 Por qué AlphaEarth Foundations v2.1 como backbone

La decisión de usar AlphaEarth Foundations como fuente principal de features, en lugar de entrenar un foundation model propio, se sustenta en tres factores:

1. Google DeepMind ya procesó petabytes de datos multisensor globales y publicó los embeddings 64-dim gratuitamente en Google Earth Engine con licencia permisiva para uso académico y comercial con atribución.
2. Los 64 dimensiones compactos son más eficientes computacionalmente que cualquier alternativa (XGBoost sobre 64 features corre en minutos en L4).
3. El contexto académico de un proyecto de 10 semanas con 3 devs a 12 h/semana no justifica dedicar ventanas H100 a replicar trabajo de DeepMind cuando el aporte original puede concentrarse en la capa VLM y LLM.

### A.2 Por qué Gemma 4 26B-MoE como VLM principal

Gemma 4 fue liberado el 2 de abril de 2026 por Google DeepMind bajo **Apache 2.0** con cuatro variantes (E2B, E4B, 26B-MoE, 31B dense). La variante 26B-MoE con 4B parámetros activos por token reúne cinco propiedades alineadas con los objetivos del proyecto:

1. **Apache 2.0 limpio** con backing de Google.
2. **Multimodal nativo con audio** además de imagen y video.
3. **Contexto 256K**, **140 idiomas** incluyendo italiano y español nativos.
4. **Fit holgado en 1×H100 96GB:** ~82 GB con LoRA, dejando margen para activaciones y KV cache.
5. **Ranking #3 en Arena open-model text leaderboard** al momento de la liberación.

Qwen3-VL-30B-A3B se incluye como VLM comparativo del EPIC 6 — se puede correr zero-shot o con LoRA secundario en la ventana V4 si el tiempo H100 lo permite, para reportar una tabla comparativa Gemma 4 vs Qwen3-VL en el paper opcional.

### A.3 Por qué Google ADK como framework del agente

Google ADK (Agent Development Kit), disponible desde 2026 bajo Apache 2.0, simplifica el desarrollo y despliegue de agentes en cuatro dimensiones críticas:

1. **Deploy nativo a Vertex AI Agent Engine** con un solo comando, sin construir Docker + Cloud Run custom.
2. **Tracing built-in** visible en Vertex AI console, evitando trabajo de observabilidad manual sobre el agente.
3. **Soporte para backends OpenAI-compatible**: Qwen3.5-35B-A3B vía vLLM funciona como backend ADK sin adaptadores custom.
4. **Protocolo A2A nativo** que permite interoperar con otros agentes a futuro sin migración.

El trade-off es un control de flujo menos granular (árbol jerárquico en lugar de grafo explícito). Para los nueve tools en plan-and-react del proyecto, ADK es suficiente.

### A.4 Por qué Dagster como orquestador

Dagster 1.9+ adopta un modelo **asset-oriented** que encaja naturalmente con pipelines ML:

- Dataset AlphaEarth → asset con lineage a Sentinel-2 y regiones.
- Features → asset con dependencia de AlphaEarth y DINOv3.
- Modelo → asset con dependencia de features y código.
- MLflow tracking → integrado vía `dagster-mlflow`.

Para un proyecto ML donde se necesita auditar "este feature table depende de este raster, que depende de este download", el modelo de assets con lineage declarativo es la opción más natural y su UI muestra el grafo directamente sin instrumentación adicional.

### A.5 Polars como motor principal de DataFrames

Polars 1.x ofrece 5-10× más velocidad que pandas en operaciones típicas de ML (group-by temporal, joins espaciales por parcela, agregaciones). Para features tabulares sobre 189 columnas y millones de píxeles estratificados, la diferencia es práctica. DuckDB queda como herramienta opcional de exploración SQL en notebooks, no en código de producción.

### A.6 dbmate como herramienta de migraciones

Decisión explícita del equipo tras experiencia previa con dbmate en proyectos anteriores. dbmate es framework-agnóstico, usa SQL puro, genera un binario Go sin dependencias y funciona igual en CI local, Cloud Build y Vertex AI Pipelines, con baja fricción operativa y skill ya consolidado en el equipo.

### A.7 Por qué dos variantes de LLM orquestador

Ofrecer Gemini 3.1 Pro (cloud, propietario) y Qwen3.5-35B-A3B (open-source, on-premise) cumple tres funciones: (1) diferenciación frente a Google Earth AI que obliga dependencia de Gemini y Google Cloud; (2) validación empírica del trade-off calidad-latencia-costo-soberanía de datos; (3) permitir despliegue privado para cooperativas agrícolas italianas que no pueden exportar datos a Google Cloud por normativa o contrato. La demo de presentación muestra el switch en vivo como diferenciador visible y cuantificable.

Qwen3.5-35B-A3B se eligió tras un análisis de memoria sobre el catálogo open-source 2026 frente a una H100 96GB single-GPU:

| Modelo | Totales / Activos | Memoria pesos | KV cache 64K | ¿Cabe H100 96GB? |
|--------|-------------------|---------------|--------------|-------------------|
| **Qwen3.5-35B-A3B** (elegido) | 35B / 3B | 70 GB BF16 | ~13 GB | **Sí, margen 5 GB** |
| Qwen3.5-27B dense | 27B | 54 GB BF16 | ~9 GB | Sí, margen 25 GB |
| Llama 3.3-70B QLoRA 4-bit | 70B | 40 GB INT4 | ~13 GB | Sí, margen 35 GB |
| MiniMax-M2.7 MoE | 230B / 10B | 115 GB NVFP4 | ~15 GB | **No**, requiere 2×H100 o H200 |
| Kimi K2.6 MoE | ~1T / 32B | >500 GB | — | **No**, requiere cluster multi-nodo |

Qwen3.5-35B-A3B ofrece el mejor balance entre capacidad (35B totales con razonamiento competitivo frente a Gemini 3.1 Pro en benchmarks Alibaba), eficiencia de inferencia (MoE con solo 3B activos por token, latencia similar a un modelo 3B dense), soporte multilingüe italiano/español/inglés nativo, licencia Apache 2.0 sin restricciones, y compatibilidad con el resto del stack (Gemma 4 como VLM en EPIC 6, Qwen3-VL como comparativo).

### A.8 Ventanas H100 redistribuidas (80 h en 6 sesiones)

| Ventana | Noches | Horas | Uso |
|---------|--------|-------|-----|
| V1 | 18-20 may | 8h | Baselines + preliminar TSViT (compartido con L4 si hace falta) |
| V2 | 25-27 may | 12h | U-TAE + TSViT + Swin-UNETR training |
| V3 | 28-30 may | 24h | **Gemma 4 26B-MoE LoRA rank 16 BF16** (3 epochs AgroMind + IT/ES) |
| V4 | 1-3 jun | 12h | Qwen3-VL-30B-A3B LoRA comparación + re-runs ensambles |
| V5 | 5-7 jun | 16h | Qwen3.5-35B-A3B serving vLLM + LoRA tool traces + eval benchmarks |
| V6 | 18-20 jun | 8h | Warm vLLM + re-runs finales para demo presentación |
| **Total** | | **80h** | |

### A.9 Por qué Nuxt 4 SSR puro

Nuxt 4 Server-Side Rendering en Cloud Run con `nuxt build` estándar cubre el 100% de los requerimientos del MVP: routing file-based, streaming SSE, i18n, MapLibre, chat UI. Mantener el frontend como web app pura (sin PWA, sin empaquetado desktop) reduce complejidad y deja abierta la opción de añadir service workers o un wrapper desktop post-MVP sin cambios estructurales. El equipo ya domina Nuxt 4 por proyectos previos.

### A.10 Política de branching y commits

Convenciones de Conventional Commits (`feat(E5): add TSViT training script`). Branches: `main` protegido con PR review obligatorio, `develop` para integración, `feature/E{epic}-{us}-{slug}` por user story. Pre-commit hooks: ruff, black, mypy, detect-secrets, nbstripout.

### A.11 Documentación del repositorio

Estructura de documentación:

- `README.md` overview y quick start.
- `docs/architecture.md` diagramas Mermaid.
- `docs/model_cards/` un Model Card por modelo versionado, incluyendo `gemma4-agrosat.md` para el fine-tuned.
- `docs/data_sheets/` un Data Sheet por dataset.
- `docs/decisions/` ADRs (Architecture Decision Records). Los cinco ADRs iniciales documentan: Gemma 4 26B-MoE como VLM principal, Google ADK como framework del agente, Dagster como orquestador asset-oriented, dbmate como herramienta de migraciones, y Nuxt 4 SSR como arquitectura del frontend.
- `docs/glosario.md` glosario técnico estandarizado en italiano, español e inglés.
- `docs/security.md` revisión OWASP Top 10.
- Jupyter notebooks con markdown explicativo para los Avances del curso.
- `paper/` con drafts, figuras reproducibles y LaTeX espejo de Overleaf (Paper Track opcional).

### A.12 Estrategia de publicación (Paper Track opcional)

Orden de envíos: pre-print arXiv cs.CV inmediato al cierre de la actividad Paper Track (garantiza prioridad temporal); Remote Sensing MDPI (rolling submission, 2-3 meses, open-access, alta probabilidad de aceptación) como prioridad; CVPR EarthVision Workshop 2026 si el deadline lo permite dada la fecha de cierre en julio; ISPRS Journal of Photogrammetry and Remote Sensing (IF 12.7, ambicioso pero alineado temáticamente con el Paper 2 del profesor) como opción stretch. Cada envío con repo GitHub actualizado y DOI Zenodo del dataset AgroMind-IT/ES.

---

## Anexo A: Catálogo de Preguntas del Copiloto

Familias de preguntas soportadas en italiano, español e inglés (cinco ejemplos de cada familia se publican como seed en el benchmark AgroMind-IT/ES):

- **Clasificación.** "¿Qué cultivo hay en esta parcela?" / "Di che coltura si tratta?" / "What crop is in this parcel?"
- **Cuantificación.** "¿Cuántas hectáreas de maíz hay en esta región?" / "Quanti ettari di mais?" / "How many hectares of maize?"
- **Vigor.** "¿Cómo está el vigor de esta parcela en los últimos 3 meses?"
- **Estrés hídrico.** "¿Detectas estrés hídrico en esta parcela?"
- **Temporalidad / fenología.** "¿En qué fase fenológica está este cultivo?"
- **Comparación.** "Compara esta parcela con la vecina del norte."
- **Anomalías.** "¿Hay anomalías detectadas en los últimos 30 días?"
- **Metadata y explicabilidad.** "¿Qué escena usaste para esta predicción? ¿Por qué clasificas esto como trigo?"
- **Intersecciones geoespaciales.** "¿Esta parcela está dentro de una zona protegida?"
- **Recomendaciones.** "¿Qué acciones de manejo recomiendas para esta parcela esta semana?"

---

## Anexo B: Glosario Técnico (IT/ES/EN)

Para mantener consistencia terminológica entre italiano, español e inglés a lo largo del documento, se estandariza el uso de los siguientes términos:

| Término inglés | Traducción / uso en el documento |
|---|---|
| pipeline | "pipeline de datos" en primera mención; "pipeline" después |
| embedding | "embedding vectorial" en primera mención; "embedding" después |
| fine-tune | "ajuste fino con LoRA" o "fine-tune" indistinto |
| foundation model | "modelo fundacional" o "FM" |
| backbone | "arquitectura base" o "backbone" |
| inference | siempre "inferencia" |
| tool call | primera mención "invocación de herramienta"; posterior "tool call" |
| agent | siempre "agente" |
| plan-and-react | primera mención "planear-y-reaccionar"; posterior "plan-and-react" |
| Spatial-RAG | "RAG espacial" o "Spatial-RAG" |
| streaming | "streaming" (aceptado) |
| checkpoint | "checkpoint" o "punto de control" |
| dataset | "dataset" o "conjunto de datos" |
| segmentation | siempre "segmentación" |
| patch | "parche" |
| benchmark | "benchmark" o "conjunto de referencia" |
| lineage | "trazabilidad" o "lineage" |
| drift | "drift" con nota "(deriva de distribución)" |
| overfitting | siempre "sobreajuste" |
| underfitting | siempre "subajuste" |
| cloud | "la nube" en texto prosa, "cloud" en stack técnico |
| deployment | "despliegue" |
| framework | "framework" (aceptado) |
| open-source | "open-source" |
| prompt engineering | "ingeniería de prompts" |
| ensemble | siempre "ensamble" |

---

**FIN DEL DOCUMENTO**

**Última actualización:** 24 de abril de 2026
**Mantenedor:** Arthur Zizumbo (MLOps lead)
**Próxima revisión:** lunes 27 de abril de 2026 (tras entrega Avance 0)
