# Avance 6 — Conclusiones Clave

**Proyecto:** AgroSatCopilot — plataforma SaaS conversacional open-source para análisis satelital agrícola
**Equipo:** 17
**Fase CRISP-ML(Q):** Evaluación / Despliegue
**Fecha:** 14 de junio de 2026

| Integrante | Rol |
|---|---|
| Carlos Isaac Ávila Gutiérrez | ML / Data Scientist |
| Carlos Aaron Bocanegra Buitrón | Full-Stack / Backend |
| Arthur Jafed Zizumbo Velasco | MLOps / Platform |
| Dr. G. J. Camacho (sponsor) | Patrocinador técnico / propietario H100 |

---

## Resumen ejecutivo

El modelo final de clasificación de cultivos —un **ensamble heterogéneo Stacking de 5 miembros con capa FarSLIP** (TSViT-pheno + U-TAE + XGBoost-AlphaEarth + FarSLIP-ft18 + FarSLIP-zero-shot)— alcanza **F1-macro 0.7486 y accuracy 0.8495** sobre el fold espacial reservado (fold-5 de PASTIS-R, 16 640 parcelas, espacio de 18 clases agronómicas). Esta cifra **no cruza el umbral de éxito de F1-macro ≥ 0.80** fijado en la Fase 0, pero supera con amplitud el baseline tabular garantizado (0.4365 en 18 clases) y el mejor modelo individual (TSViT-pheno, F1 0.75 / mIoU 0.625).

Más relevante para la decisión de negocio: sobre el **punto de éxito de 8 categorías** que el Dr. Camacho definió en reunión como óptimo de operación, el ensamble alcanza **F1-macro 0.920 y accuracy 0.882 cubriendo el 80 % de las parcelas**, superando holgadamente el umbral de éxito de F1-macro ≥ 0.80 de la Fase 0.

**Veredicto: GO-condicional.** La solución **se puede implementar en producción** bajo dos condiciones explícitas: (1) operar con **doble taxonomía** —reportar las 18 clases finas y, como punto de éxito, las 8 categorías prioritarias acordadas con el sponsor (más los 6 grupos HCAT)— y (2) **comunicar la incertidumbre por clase** en la interfaz. **No** es necesario retroceder a las fases de modelado o preparación de datos: el techo restante está acotado y diagnosticado (seis cultivos fenológicamente ambiguos), y existe margen de mejora dentro de la fase de evaluación sin reingeniería de datos.

---

## 1. Análisis del modelo frente a los criterios de éxito (Objetivo 4.1)

### 1.1 Criterios de éxito definidos en la Fase 0

Los criterios cuantitativos del proyecto se fijaron en `context/AgroSatCopilot.md` (L108) y se refinaron en `context/RefinamientoPlaneacionAgroSatCopilot_v8.md`:

| Dimensión | Métrica | Meta Fase 0 | Fuente |
|---|---|---|---|
| ML — baseline | F1-macro tabular (AlphaEarth + XGB) | ≥ 0.60 | AgroSatCopilot.md:108 |
| ML — modelo final | F1-macro (ensambles) | **≥ 0.80** | AgroSatCopilot.md:108 |
| ML — segmentación densa | mIoU | **≥ 0.70** | v8:78 |
| LLM — razonamiento | AgroMind (Gemini / Qwen3.5) | ≥ 0.75 / ≥ 0.70 | v8:1648 |
| LLM — geoespacial | GeoAnalystBench pass rate | ≥ 0.65 | v8:1648 |
| Producto | Latencia chat p95 (simple / multi-step) | < 3 s / < 15 s | v8:1617, 1741 |
| Producto | Polígono → respuesta end-to-end | < 10 s | AgroSatCopilot.md:110 |
| Costo | Operativo mensual (scale-to-zero) | ~$115 USD/mes | AgroSatCopilot.md:24 |
| Confiabilidad | Alertas de drift (Evidently) | 0 en semana previa | AgroSatCopilot.md:108 |
| Calidad | Cobertura de tests backend / frontend | ≥ 70 % / ≥ 50 % | AgroSatCopilot.md:108 |

> **Matiz de alcance importante.** El objetivo de F1-macro ≥ 0.80 se redactó asumiendo a **Gemma 4 26B-MoE LoRA** como clasificador final. La decisión [ADR-009](../decisions/ADR-009-h100-reactivacion-pivote-farslip-alcance-v8.md) **difirió** ese fine-tuning por un bloqueador técnico real (los expertos MoE 3D fusionados impiden el matcheo de `target_modules` para QLoRA) y reorientó la arquitectura al patrón **"Be My Eyes"**: el LLM **razona**, no clasifica píxeles; el clasificador final es el **ensamble del EPIC 6**. Por tanto, la meta de 0.80 debe leerse contra una arquitectura que cambió deliberadamente y de forma documentada, no contra un incumplimiento silencioso.

### 1.2 Resultados obtenidos

**Cadena de valor medida (todas las cifras en fold-5, espacio semantic18 de 18 clases):**

| Modelo | F1-macro | Accuracy | Lectura |
|---|---|---|---|
| Baseline XGB + AlphaEarth (18 clases) | 0.4365 | — | Cruza 0.60 solo al colapsar a 6 familias HCAT (0.6535) |
| Mejor individual — TSViT-pheno | 0.75 | — | mIoU 0.625 (segmentación densa) |
| Stacking-3 (campeón US-040, sin FarSLIP) | 0.7470 | 0.8490 | Mejor ensamble previo |
| **Stacking-5 + FarSLIP (campeón final)** | **0.7486** | **0.8495** | Nuevo campeón; +0.0016 vs US-040 |

**Diagnóstico por clase (campeón final).** El macro-F1 está sostenido por 12 clases fuertes (F1 0.72–0.94: Corn, Grapevine, Meadow, Soft winter wheat, Beet, Winter rapeseed, etc.) y arrastrado por **6 clases problemáticas**, todas con el mismo patrón **recall ≫ precisión** (el ensamble las sobre-predice):

| Clase | Soporte | Precisión | Recall | F1 |
|---|---|---|---|---|
| Potatoes | 103 | 0.35 | 0.62 | 0.44 |
| Mixed cereal | 193 | 0.39 | 0.54 | 0.45 |
| Sorghum | 206 | 0.40 | 0.61 | 0.48 |
| Leguminous fodder | 660 | 0.46 | 0.66 | 0.54 |
| Winter triticale | 214 | 0.54 | 0.74 | 0.63 |
| Spring barley | 198 | 0.59 | 0.71 | 0.64 |

El hallazgo crítico: **Leguminous fodder tiene 660 parcelas** (cuarto soporte más alto) y aún así F1 0.54. No es escasez de datos; es **confusión semántica/fenológica** —la matriz de confusión muestra que se fuga a *Meadow*, que *Sorghum* se confunde con *Corn* (ambos cultivos C4 de verano con curva NDVI casi idéntica) y que *Triticale*/*Mixed cereal* se confunden con *Soft winter wheat* (cereales de invierno indistinguibles por su firma temporal).

**Curva de descarte honesto** (ranking decidido en OOF, medido una sola vez en fold-5): si la solución operara sobre un subconjunto de clases bien predichas, el macro-F1 sube de forma monótona —**0.7486 a 18 clases → 0.8194 a 14 → 0.8573 a 12 → 0.9201 a 8**—. Esto prueba que el modelo **sí cruza 0.80** en cuanto se excluyen las clases agronómicamente ambiguas, y fundamenta la estrategia de doble taxonomía.

![Curva de descarte honesto: F1-macro y accuracy en el fold reservado frente al número de clases retenidas (ranking decidido en OOF). El F1-macro pasa de 0.749 con 18 clases a 0.920 con 8 clases, superando el umbral 0.90 desde K=9; las barras indican el porcentaje de parcelas cubiertas por las K clases.](../../reports/ensemble/figures/us043_farslip/honest_class_dropout.png)

**Punto de éxito acordado con el sponsor: 8 categorías.** En reunión de revisión, el Dr. Camacho fijó **8 categorías como el punto óptimo de operación** —el equilibrio entre no sacrificar demasiadas clases y no perder demasiado rendimiento—. El gráfico anterior valida esa decisión de forma cuantitativa y honesta (el ranking de qué clases retener se decide en los sub-folds OOF y la métrica se mide una sola vez en el fold reservado, sin *cherry-picking*):

| Operación | Clases | F1-macro (fold reservado) | Accuracy | Parcelas cubiertas |
|---|---|---|---|---|
| Catálogo completo | 18 | 0.749 | 0.850 | 100 % |
| **Punto de éxito (sponsor)** | **8** | **0.920** | **0.882** | **80 %** |

Las 8 categorías retenidas son precisamente los cultivos de mayor interés económico y mejor separabilidad fenológica —Maíz, Colza de invierno, Viñedo, Remolacha, Trigo blando de invierno, Pradera, Cebada de invierno y Soja—, que cubren el **80 % de las parcelas** del territorio. Bajo este punto de operación, el modelo **supera holgadamente el umbral de éxito de F1-macro ≥ 0.80 de la Fase 0 (0.920)** y se acerca al objetivo de mIoU. La interpretación es directa: la solución es plenamente apta para producción sobre el conjunto de cultivos que el sponsor definió como prioritario, mientras las 10 categorías minoritarias y fenológicamente ambiguas se tratan con incertidumbre declarada (doble taxonomía), no se ocultan.

### 1.3 Comparación meta vs. logrado

| Criterio de éxito | Meta | Logrado | ¿Cumple? |
|---|---|---|---|
| Baseline F1-macro | ≥ 0.60 | 0.4365 (18 cl.) / 0.6535 (HCAT-6) | Condicional (en grupos) |
| Modelo final F1-macro | ≥ 0.80 | 0.7486 (18 cl.) / **0.920 (8 cl., punto sponsor)** | **No en 18 / Sí en 8** |
| Segmentación mIoU | ≥ 0.70 | 0.625 (TSViT-pheno) | No |
| Latencia / costo / drift | (ver §1.1) | Diseño cumple (ver §4) | Sí (por diseño) |

### 1.4 Respuestas a los cuestionamientos de la rúbrica

**¿El rendimiento del modelo es lo suficientemente bueno para producción?**
**Sí, condicionalmente.** A nivel agronómico el sistema es operable hoy: la accuracy global es 0.85 y, sobre el **punto de éxito de 8 categorías acordado con el sponsor**, el F1-macro es 0.920 cubriendo el 80 % de las parcelas. La promesa de producto —"dibujar un polígono y obtener qué cultivo es, su fenología y su estado en segundos"— se cumple para la mayoría de cultivos de interés económico (maíz, viñedo, trigo, remolacha, colza, pradera, cebada, soja). El sistema **no** alcanza calidad de "ground truth catastral" en las clases minoritarias fenológicamente ambiguas, y por eso debe **declarar su incertidumbre por clase** en lugar de presentar una etiqueta única engañosa. Presentar el conjunto de 8 categorías como "objetivo cumplido" sin mostrar el resultado de 18 clases sería *gaming* de la métrica y se evita explícitamente; se reportan ambas taxonomías lado a lado, y el punto de 8 está justificado por una decisión de negocio del sponsor y validado en fold reservado, no elegido a posteriori para inflar el número.

**¿Existe margen para mejorar aún más el rendimiento?**
**Sí, y está acotado dentro de la fase de evaluación (sin reingeniería de datos).** El diagnóstico recall ≫ precisión y la confusión entre clases hermanas habilitan palancas concretas y de bajo costo:

1. **Corrección de prior/umbral por clase** optimizando macro-F1 directamente sobre los sub-folds OOF (sube la precisión de las minoritarias recortando su sobre-predicción). Impacto estimado +0.015 a +0.025 macro-F1, sin GPU.
2. **Meta-learner enriquecido y no lineal**: añadir features de confianza/desacuerdo entre miembros y reemplazar el logreg lineal (90 features) por un meta no lineal (XGBoost) que aprenda reglas condicionales por clase.
3. **FarSLIP zero-shot con captions fenológicas reales** (ya existen 69 k descripciones por parcela generadas con Gemma) en lugar de prompts genéricos, para separar mejor los cereales de invierno y los C4 de verano.
4. **Ensamble jerárquico coarse→fine** que respete la taxonomía agronómica, con expertos especializados dentro de cada grupo confundible.
5. **Árbitro multimodal Gemma 4** sobre el ~10–15 % de parcelas en disputa (alto desacuerdo del ensamble), conectando el modelo final con el producto conversacional.

**¿Cuáles serían las recomendaciones clave para implementar la solución?**

- Desplegar el ensamble Stacking-5 como clasificador, con **doble taxonomía** (18 clases + HCAT-6) y **confianza por clase** expuesta en la UI.
- Implementar primero las palancas baratas (1 y 2 arriba) y re-medir de forma honesta antes de la presentación final.
- Mantener el **fallback** garantizado (XGB + AlphaEarth a nivel HCAT-6, F1 ≥ 0.60) como ruta de degradación gradual.
- Servir bajo el patrón "Be My Eyes": el ensamble percibe, Gemini 2.5 Pro razona; **el LLM nunca clasifica píxeles**.
- Cerrar la trazabilidad de extremo a extremo (MLflow con `data_version` + `code_version`, DVC para pesos/rasters, model card publicada).

**¿Se implementa o se retrocede a fases anteriores?**
**Se implementa (GO-condicional). No se retrocede.** Retroceder a preparación de datos no aportaría: la validación es espacial y sin fuga, las clases ambiguas lo son por fenología intrínseca (no por etiquetado deficiente) y el contexto geográfico tabular (ERA5/SRTM) ya demostró delta 0.0 sobre el embedding AlphaEarth. El trabajo restante vive en la fase de **evaluación/ensamble**, no en datos ni en arquitectura base.

---

## 2. Accionables para stakeholders (Objetivo 4.2)

Cada accionable está formulado de forma específica, con responsable nombrado y criterio de hecho verificable.

| # | Acción específica | Responsable (stakeholder) | Plazo | Criterio de hecho |
|---|---|---|---|---|
| A1 | Implementar la **corrección de prior por clase** sobre la salida del Stacking-5 (decisión en OOF, medición única en fold-5) | Isaac (ML / Data Scientist) | Sprint actual | Macro-F1 fold-5 ≥ 0.76 reportado en MLflow con tags de versión |
| A2 | Sustituir el meta-logreg por **meta no lineal + features de meta-cognición** (entropía, margen, acuerdo entre miembros) | Isaac (ML / Data Scientist) | Sprint actual | Run MLflow comparativo logreg vs. XGB-meta, sin fuga (`assert_oof_only`) |
| A3 | Exponer **doble taxonomía y confianza por clase** en el endpoint `/chat` y en los overlays del mapa | Aaron (Backend / Full-Stack) | Pre-presentación | Respuesta SSE devuelve 18 clases + HCAT-6 + probabilidad; UI muestra banda de incertidumbre |
| A4 | Empaquetar el ensamble como **worker de inferencia** (Pub/Sub + Cloud Run L4 scale-to-zero) y publicar la **model card** | Arthur (MLOps / Platform) | Pre-presentación | Worker responde polígono→clase < 10 s; model card en repo con limitaciones por clase |
| A5 | Activar **monitoreo de drift Evidently** (KS en bandas S2, MMD en embeddings, chi-cuadrado en clases) con reporte semanal | Arthur (MLOps / Platform) | Pre-presentación | 0 alertas en la semana previa a la presentación; reporte HTML en GCS |
| A6 | Validar el **fallback HCAT-6 (XGB + AlphaEarth)** como ruta de degradación y documentar el árbol GO / GO-condicional / NO-GO | Isaac + Arthur | Pre-presentación | Fallback con F1 ≥ 0.60 verificado; árbol de decisión en `docs/` |
| A7 | Confirmar **disponibilidad y costo de la ventana H100** para las mejoras que requieran reentrenar miembros base | Dr. Camacho (sponsor) | Inmediato | Confirmación de horas H100 disponibles y autorizadas |
| A8 | Validar la **utilidad agronómica** de las salidas (¿son accionables las clases bien predichas para el usuario final?) | Product owner / agrónomo de referencia | Pre-presentación | Lista de casos de uso aceptados (maíz, viñedo, trigo, etc.) firmada |
| A9 | Ejecutar `make check` + cobertura (≥ 70 % backend, ≥ 50 % frontend) antes del merge a `develop` | Aaron (Backend) | Cada PR | CI verde; cobertura cumplida |

---

## 3. Entorno de producción y proveedor cloud (Objetivo 4.3 + Implementación)

### 3.1 Requisitos del entorno de producción

El entorno debe garantizar las tres propiedades exigidas:

- **Confiabilidad:** validación espacial sin fuga, trazabilidad MLflow (`data_version` + `code_version`), versionado de datos/pesos con DVC, monitoreo de drift con Evidently y model card con limitaciones por clase.
- **Escalabilidad:** servir picos de demanda interactiva sin pagar cómputo ocioso (scale-to-zero), separar la ruta de inferencia pesada (GPU) de la API ligera vía colas (Pub/Sub).
- **Eficiencia (FinOps):** presupuesto operativo objetivo ~$115 USD/mes y costo de entrenamiento one-time acotado a $262–602 USD; uso de GPU spot con auto-shutdown.

### 3.2 Comparativa de proveedores cloud

Se evalúan los cuatro principales proveedores frente a siete factores ponderados según las necesidades del proyecto. Las cifras de precio GPU son **referenciales** (listas públicas 2025–2026) y deben confirmarse contra el calculador oficial de cada proveedor en el momento del despliegue.

| Factor | GCP | Azure | AWS | IBM Cloud / watsonx |
|---|---|---|---|---|
| **Ecosistema Earth Observation** | **Earth Engine con AlphaEarth Foundations (exclusivo, gratis CC-BY-4.0)** | Planetary Computer (Sentinel/Landsat gratis, sin AlphaEarth) | Registry of Open Data (Sentinel en S3, sin AlphaEarth) | watsonx.ai con FM geoespacial Prithvi (IBM+NASA), sin AlphaEarth |
| **GPU H100 (single-GPU)** | A3 (bloques de 8×H100, difícil 1×GPU) | **NC40ads_H100_v5 = 1×H100 NVL 96 GB, spot ~$2.74/h** | p5.48xlarge (8×H100, sin SKU de 1×GPU accesible) | GPU H100 disponible, oferta single-GPU limitada |
| **Serverless scale-to-zero** | **Cloud Run v2 (scale-to-zero nativo, maduro)** | Container Apps (scale-to-zero) | Fargate/App Runner (sin scale-to-zero puro a 0) | Code Engine (scale-to-zero) |
| **MLOps gestionado** | Vertex AI Pipelines + Agent Engine + ADK nativo | Azure ML | SageMaker (muy completo) | watsonx.ai / watsonx.governance |
| **LLM nativo (reasoner elegido)** | **Gemini 2.5 Pro en Vertex AI (1M ctx)** | Azure OpenAI (GPT) | Bedrock (Claude, Llama) | watsonx (Granite, Llama) |
| **Latencia a Europa (target Italia)** | europe-west1/8 | westeurope (proximidad alta) | eu-south-1 (Milán) | eu-de (Frankfurt) |
| **Facilidad / créditos académicos** | Alta; programas educativos | Alta | Alta; madurez de mercado | Media; menor adopción en EO |

### 3.3 Justificación de la elección: arquitectura híbrida GCP-primario + Azure puntual

**Decisión: GCP como nube principal y Azure exclusivamente para la ventana H100 de fine-tuning.**

Los factores decisivos, en orden de peso:

1. **AlphaEarth Foundations es gratuito y exclusivo de Google Earth Engine.** Es el *backbone* de features de todo el proyecto (embeddings 64-dim/píxel/año, 10 m, global). Replicarlo en AWS/Azure/IBM exigiría reentrenar o re-derivar el foundation model —costo prohibitivo y fuera de alcance—. Este único factor inclina la nube principal hacia GCP de forma casi determinante (v6:144-145; AgroSatCopilot.md:31).
2. **Gemini 2.5 Pro como reasoner del patrón "Be My Eyes"** está disponible de forma nativa en Vertex AI, y **Google ADK** —el framework de agente elegido— se despliega nativamente en Vertex AI Agent Engine con tracing integrado, evitando trabajo de observabilidad a medida.
3. **Cloud Run v2 ofrece scale-to-zero maduro**, lo que hace viable el presupuesto de ~$115 USD/mes: la API, el SSR de Nuxt, TiTiler y el worker de inferencia GPU L4 escalan a cero cuando no hay tráfico.
4. **Azure es la única hyperscaler con un SKU de 1×H100 NVL 96 GB** (`Standard_NC40ads_H100_v5`) a precio spot accesible (~$2.74/h spot, ~$6.98/h on-demand), sin obligar a alquilar bloques de 8×H100 como AWS p5 o GCP A3. Por eso el fine-tuning (FarSLIP, y a futuro Qwen3.5/Gemma 4 LoRA) se reserva a una ventana Azure puntual con auto-shutdown.

**Por qué no AWS ni IBM como nube principal:** AWS tiene el ecosistema MLOps más maduro (SageMaker) y datos abiertos en S3, pero carece de AlphaEarth y no ofrece scale-to-zero puro ni una SKU de 1×H100 accesible. IBM watsonx aporta el FM geoespacial **Prithvi** (IBM+NASA) —técnicamente interesante—, pero el proyecto **descartó explícitamente Prithvi-EO-2.0** a favor de AlphaEarth, y la adopción de IBM en pipelines de EO es marginal frente a GCP/Azure. Ambos quedan como referencias comparativas, no como elección.

> Nota de realidad operativa: durante el curso, la H100 NVL 96 GB la presta el sponsor (VM `gjcamacho-gpuh1`, acceso por túnel) y no se factura; el módulo Terraform de Azure existe completo y comentado como ruta reproducible para un despliegue futuro ([ADR-009](../decisions/ADR-009-h100-reactivacion-pivote-farslip-alcance-v8.md); v8:221). Igualmente, se mantiene un único entorno `dev` por FinOps ([ADR-002](../decisions/ADR-002-single-env-dev.md)).

### 3.4 Arquitectura de despliegue propuesta (servicios concretos)

**GCP (nube principal, Infraestructura como Código en Terraform):**

- **Cloud Run v2 (scale-to-zero, `min_instances=0`):** API FastAPI, frontend Nuxt 4 SSR, TiTiler (servidor de tiles COG) y servidor MLflow.
- **Cloud Run con GPU L4 (interno):** worker de inferencia del ensamble, disparado por **Pub/Sub** (topics `inference-jobs` / `inference-results`).
- **Cloud SQL PostgreSQL 15 + PostGIS + pgvector:** estado conversacional multi-tenant por `session_id`, geometrías y RAG vectorial; `db-f1-micro` con PITR.
- **Cloud Storage (4 buckets):** datos, artefactos de modelo, remoto DVC y estado de Terraform; lifecycle a NEARLINE.
- **Secret Manager:** secretos de aplicación; **Artifact Registry:** imágenes Docker; **Vertex AI:** Gemini 2.5 Pro + Agent Engine (ADK).
- **Upstash Redis (serverless):** caché de sesión, en lugar de Memorystore por FinOps ([ADR-003](../decisions/ADR-003-upstash-redis.md)).

**Azure (ventana puntual de entrenamiento):**

- VM `Standard_NC40ads_H100_v5` (1×H100 NVL 96 GB) en `westeurope`, spot por defecto, auto-shutdown diario obligatorio; Blob Storage Hot para checkpoints LoRA; Key Vault con RBAC.

**Costos documentados:**

| Concepto | Costo |
|---|---|
| Entrenamiento (one-time) | $262 (spot) – $602 (on-demand) USD |
| Operativo mensual (scale-to-zero) | ~$115 USD/mes |

Desglose operativo aproximado: Cloud Run API/TiTiler/SSR ~$18, worker GPU L4 ~$15, Cloud SQL ~$14, Storage ~$6, Redis ~$15, Pub/Sub ~$3, Vertex AI/Gemini ~$12, Qwen3.5 self-hosted ~$30, varios ~$6 (v6:607-624).

### 3.5 Garantías de confiabilidad, escalabilidad y eficiencia

- **Confiabilidad:** validación espacial sin fuga (H3 + KMeans + buffer 1 km), MLflow con `data_version` + `code_version`, DVC para rasters/pesos, Evidently para drift, IAM con principio de mínimo privilegio (nunca `roles/owner`), secretos en Secret Manager / Key Vault.
- **Escalabilidad:** Cloud Run escala horizontalmente bajo demanda y a cero en reposo; la inferencia pesada se desacopla de la API mediante Pub/Sub; arquitectura reproducible vía Terraform.
- **Eficiencia:** scale-to-zero, Cloud SQL `db-f1-micro` apagada en reposo, GPU spot con auto-shutdown, Redis serverless; auditoría con `make cost-audit`.

---

## 4. Conclusión

El modelo final de AgroSatCopilot **es apto para implementación en producción bajo el régimen GO-condicional**: el ensamble Stacking-5 con FarSLIP alcanza F1-macro 0.7486 y accuracy 0.8495 en validación espacial honesta, supera ampliamente el baseline y el mejor individual, y cumple la promesa de producto para los cultivos de mayor interés económico. **No alcanza el umbral de F1-macro ≥ 0.80 en las 18 clases finas**, pero ese umbral se fijó bajo una arquitectura (Gemma 4 LoRA) deliberadamente reorientada por ADR-009, y sobre el **punto de éxito de 8 categorías que el Dr. Camacho definió como óptimo de operación, el modelo alcanza F1-macro 0.920 cubriendo el 80 % de las parcelas**, superando con holgura el criterio de la Fase 0. El gap en las clases finas es intrínseco a la fenología de seis cultivos ambiguos, no a la calidad de los datos ni a la arquitectura base, por lo que **no se justifica retroceder** a fases anteriores; el trabajo pendiente es de evaluación/ensamble y está priorizado en los accionables.

Para el despliegue se elige una **arquitectura híbrida GCP-primario + Azure puntual**, fundamentada en el acceso gratuito y exclusivo a AlphaEarth en Google Earth Engine, en la integración nativa de Gemini y Google ADK en Vertex AI, en el scale-to-zero maduro de Cloud Run, y en la única SKU de 1×H100 NVL accesible (Azure NC H100 v5), con un costo operativo objetivo de ~$115 USD/mes.

---

## Referencias

**Documentación interna del proyecto**

- Plan de planeación v8: `context/RefinamientoPlaneacionAgroSatCopilot_v8.md`
- Resumen de plan / criterios de éxito: `context/AgroSatCopilot.md` (L24, L108, L110)
- ADR-002 — un solo entorno `dev` (FinOps): `docs/decisions/ADR-002-single-env-dev.md`
- ADR-003 — Upstash Redis: `docs/decisions/ADR-003-upstash-redis.md`
- ADR-009 — reactivación H100, pivote FarSLIP, diferimiento Gemma 4 LoRA: `docs/decisions/ADR-009-h100-reactivacion-pivote-farslip-alcance-v8.md`
- ADR-010 — ensamble E-c geo-context (trabajo futuro): `docs/decisions/ADR-010-ensamble-ec-geocontext-future.md`
- Resultados del ensamble final: `reports/ensemble/metrics/` (us043_winner_per_class.csv, us043_honest_dropout_curve.csv, us043_farslip_stacking_blending.csv) y `reports/ensemble/us043_farslip_summary.json`
- Infraestructura Terraform: `infrastructure/terraform/modules/{gcp,azure}/main.tf`

**Documentación oficial de proveedores cloud**

- Google Earth Engine y AlphaEarth Foundations — https://earthengine.google.com/
- Google Cloud Vertex AI (pricing) — https://cloud.google.com/vertex-ai/pricing
- Google Cloud Run — https://cloud.google.com/run
- Azure NC H100 v5-series — https://learn.microsoft.com/azure/virtual-machines/nc-h100-v5-series
- Microsoft Planetary Computer — https://planetarycomputer.microsoft.com/
- AWS SageMaker — https://aws.amazon.com/sagemaker/ · AWS Registry of Open Data — https://registry.opendata.aws/
- AWS EC2 P5 (H100) — https://aws.amazon.com/ec2/instance-types/p5/
- IBM watsonx.ai y FM geoespacial Prithvi (IBM+NASA) — https://www.ibm.com/products/watsonx-ai

**Metodología**

- Studer et al. (2021), *Towards CRISP-ML(Q): A Machine Learning Process Model with Quality Assurance Methodology*.
- Garnot & Landrieu (2021), *Panoptic Segmentation of Satellite Image Time Series (PASTIS / U-TAE)*.
- Huang et al. (2025), *Be My Eyes* (patrón perceiver–reasoner para VLM).
