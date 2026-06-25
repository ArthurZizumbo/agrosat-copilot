# Comparativa de proveedores cloud — AgroSatCopilot (US-063)

> **Epic** E10 (Observabilidad + Docs) · **Avance del curso** A6 · **Plan vigente**
> `context/RefinamientoPlaneacionAgroSatCopilot_v8.md` §US-063, ratificado por
> [ADR-009](../decisions/ADR-009-h100-reactivacion-pivote-farslip-alcance-v8.md).
>
> Documento de decision multi-cloud. Justifica la arquitectura **GCP primario + H100 NVL 96GB
> del sponsor (Azure / on-prem) para training intensivo y serving Qwen vLLM**, comparando como
> minimo GCP vs Azure (y de forma opcional AWS e IBM Cloud) sobre cinco factores tecnicos.

> **Nota sobre cifras.** Toda cifra presentada como gasto o recurso REAL del proyecto (costo GCP
> acumulado, costo Gemini, tamano de disco, VM del sponsor, regiones Terraform) esta verificada
> contra el plan v8, la memoria del proyecto y el codigo Terraform del repo. Los precios de lista
> de GPU H100 por proveedor son **precios publicados de referencia, con fuente y fecha de consulta
> explicitas**; NO son mediciones del proyecto. El gasto real del proyecto en GPU H100 es **$0**
> porque el sponsor presta una H100 NVL 96GB 24/7 sin costo al equipo.

---

## 1. Contexto y objetivo

AgroSatCopilot es un SaaS conversacional open-source de analisis satelital agricola. La rubrica del
Avance 6 (criterio "Implementacion", 30 puntos) exige exhibir **al menos dos proveedores cloud** y
argumentar la eleccion con datos reales del proyecto, no con benchmarks genericos.

La arquitectura no es mono-cloud por accidente sino por diseno: GCP es el plano de control y de
datos (el FM EO del proyecto vive en Google Earth Engine), mientras que el training intensivo de
VLM/segmentacion y el serving on-prem del LLM corren sobre una **H100 NVL 96GB que el sponsor presta
24/7**. Este documento mide ambos mundos en cinco factores y cierra con la decision.

Los cinco factores evaluados (plan v8 §US-063):

1. Precio GPU H100 on-demand y spot.
2. Ecosistema de Earth Observation (GEE vs Planetary Computer vs AWS Open Data).
3. Latencia hacia Europa (target Italia).
4. Soporte de pipelines MLOps (Vertex AI Pipelines vs Azure ML vs SageMaker).
5. Disponibilidad de partnerships academicos / creditos.

---

## 2. Proveedores evaluados

| Proveedor | Rol en AgroSatCopilot | Estado |
|-----------|-----------------------|--------|
| **Google Cloud (GCP)** | Plano de control y datos primario | En uso (Terraform `modules/gcp`) |
| **Microsoft Azure** | Host de la H100 NVL 96GB del sponsor (training + serving) | En uso (VM `gjcamacho-gpuh1`, administrada por el sponsor) |
| Amazon Web Services (AWS) | Comparativa de referencia (opcional) | No usado |
| IBM Cloud | Comparativa de referencia (opcional) | No usado |

GCP y Azure son los proveedores reales. AWS e IBM Cloud se incluyen como referencia para
contextualizar precios de GPU y ecosistemas EO; no forman parte del despliegue.

---

## 3. Tabla comparativa maestra

Filas = los 5 factores. Columnas = proveedores. Los precios H100 son referencias publicadas (ver
fuentes y fechas en §4 y §11); el resto son hechos de arquitectura.

| Factor | GCP (primario) | Azure (H100 sponsor) | AWS (referencia) | IBM Cloud (referencia) |
|--------|----------------|----------------------|------------------|------------------------|
| **1. H100 on-demand** (USD/GPU-h, referencia) | ~$9.80-$10.98 (A3, us-central1) | ~$6.98 (NC40ads_H100_v5, East US) | ~$6.88 (p5.48xlarge / 8 GPU, us-east-1) | Disponible (VPC GPU `gx3`, H100 NVL); sin precio oficial publicado a la fecha — ver §4.1 y §11 |
| **1b. H100 spot** (USD/GPU-h, referencia) | ~$3.69 (A3 Spot, us-central1) | ~$2.19-$2.49 (NC40ads Spot) | ~$3.83 (p5 Spot, us-east-1) | No verificado a la fecha — ver §11 |
| **2. Ecosistema EO** | Google Earth Engine (AlphaEarth `SATELLITE_EMBEDDING/V1/ANNUAL`) | Planetary Computer (STAC + Hub) | Registry of Open Data (Sentinel-2 COG) | Limitado (Sentinel via terceros) |
| **3. Latencia a Italia** | `europe-west*` (p. ej. `europe-west8` Milan) | `westeurope` (Holanda; proximidad a Italia) | `eu-south-1` (Milan) | `eu-de` (Frankfurt) |
| **4. MLOps** | Vertex AI Pipelines + Agent Engine | Azure ML | SageMaker Pipelines | watsonx.ai |
| **5. Partnerships / creditos** | GCP credits / Vertex AI Agent Builder trial (ver caveat §8) | Azure for Students / sponsor academico | AWS Educate / Activate | IBM Academic Initiative |

> El gasto REAL del proyecto en H100 es **$0** (sponsor 24/7). Las celdas de precio son lista de
> referencia para dimensionar el ahorro, no facturacion del equipo.

---

## 4. Factor 1 — Precio GPU H100 (on-demand y spot)

Es el factor mas sensible a inventar numeros, asi que se trata con cuidado: los precios son **de
lista, publicados, con fuente y fecha**; el gasto del proyecto es **cero**.

### 4.1 Precios de referencia por proveedor

| Proveedor | SKU H100 | On-demand (USD/GPU-h) | Spot (USD/GPU-h) | Fuente · fecha de consulta |
|-----------|----------|------------------------|------------------|----------------------------|
| GCP | A3 (`a3-highgpu-8g`), us-central1 | ~$9.80-$10.98 | ~$3.69 | Spheron / gpucost.org · consultado 2026-06-20 (datos a 2026-05-22) |
| Azure | `Standard_NC40ads_H100_v5`, East US | ~$6.98 (1 H100/VM) | ~$2.19-$2.49 | Vantage / CloudPrice · consultado 2026-06-20 (datos a 2025-10) |
| AWS | `p5.48xlarge` (8 H100), us-east-1 | ~$6.88/GPU (post-recorte jun-2025) | ~$3.83/GPU | DoiT / Spheron · consultado 2026-06-20 |
| IBM Cloud | VPC GPU `gx3` (H100 NVL) | Sin precio oficial publicado (ver nota IBM abajo) | No verificado a la fecha | Aggregadores · consultado 2026-06-25 (ver §11) |

Notas de lectura:

- GCP A3 se factura por nodo de 8 GPU (~$87.84/h on-demand); la columna muestra el equivalente por
  GPU. El descuento por compromiso de 1 ano baja a ~$8.78/GPU-h.
- Azure `NC40ads_H100_v5` es una VM de **una sola H100 NVL 96GB**, exactamente la familia del
  sponsor; por eso su precio por GPU es directamente comparable y resulta el mas bajo de los tres
  on-demand verificados.
- AWS `p5.48xlarge` es un nodo de 8 H100; tras el recorte de junio 2025 (~44% en P5 on-demand) el
  costo por GPU cae a ~$6.88 on-demand.
- IBM Cloud ofrece H100 en VPC (perfil acelerado `gx3`, H100 NVL), pero **IBM no publica un precio
  oficial por GPU-hora** en sus paginas de pricing a la fecha (re-verificado 2026-06-25). Los
  agregadores que el resto de esta tabla usa (ComputePrices.com, GPUPerHour, Spheron 2026,
  IntuitionLabs) **omiten IBM Cloud** o declaran explicitamente "pricing details are not widely
  published, so we omit them here" (IntuitionLabs, dato a 2026-06-20). Una busqueda devolvio un
  dato secundario sin respaldo oficial verificable de **~$0.99/GPU-h H100 NVL** (clusters de 8x);
  por la regla de datos reales se anota como **NO confirmado contra fuente oficial** y NO se usa
  como cifra de la comparativa. Queda registrado en `docs/blockers/epic10-notas.md` (B15) y NO se
  rellena con un numero fabricado.

### 4.2 Implicacion para AgroSatCopilot

El **gasto real del proyecto en H100 = $0**. El sponsor presta una **H100 NVL 96GB 24/7** (VM
`gjcamacho-gpuh1`, familia `Standard_NC40ads_H100_v5` en `westeurope`), administrada por el sponsor,
no por nuestro Terraform. El argumento de decision NO es "que proveedor de H100 es mas barato" sino
"no compramos H100 en ningun cloud porque el sponsor la presta". El training unico historico del
proyecto costo **$262 (spot) - $602 (on-demand)** cuando se uso capacidad propia; hoy ese coste
tiende a $0 (ver §10).

---

## 5. Factor 2 — Ecosistema Earth Observation

| Proveedor | Plataforma EO | Aporte clave |
|-----------|---------------|--------------|
| **GCP** | **Google Earth Engine** | Aloja **AlphaEarth Foundations** `SATELLITE_EMBEDDING/V1/ANNUAL`, **data v1.1**, 64-dim, global incluido Mexico, **CC-BY-4.0**, gratis. Es el FM EO del proyecto. |
| Azure | Microsoft Planetary Computer | Catalogo STAC abierto + Hub con Sentinel-2 / Landsat; util para ingesta, sin el FM AlphaEarth. |
| AWS | Registry of Open Data | Sentinel-2 COG en S3 (`sentinel-s2-l2a`); datos crudos sin embeddings pre-computados. |
| IBM Cloud | watsonx.ai geospatial (Prithvi) | Foundation models geoespaciales propios; Prithvi-EO esta **descartado** en el stack del proyecto. |

### Implicacion para AgroSatCopilot

Este factor **ata la decision a GCP de forma dura**: el feature backbone del proyecto es AlphaEarth,
que solo se sirve gratis y global desde Google Earth Engine. Azure Planetary Computer y AWS Open
Data ofrecen Sentinel-2 crudo, pero ninguno entrega los embeddings 64-dim de AlphaEarth listos para
consumo. Migrar el backbone EO fuera de GCP implicaria recomputar el FM, lo cual el proyecto
explicitamente NO hace (decision irrevocable: no entrenar FM propio). Por tanto GCP es primario por
necesidad del stack, no por preferencia.

---

## 6. Factor 3 — Latencia hacia Europa (target Italia)

El usuario objetivo de la primera fase es Italia; la latencia a region europea importa para el
serving del API y el chat SSE.

| Proveedor | Region elegida / mas proxima a Italia | Nota |
|-----------|----------------------------------------|------|
| **GCP** | `europe-west8` (Milan) para serving; `europe-west*` para datos | Region GCP fisicamente en Italia. |
| **Azure** | `westeurope` (Paises Bajos) | Region de la H100 del sponsor; `azure/variables.tf:1` la documenta como "Italy proximity, H100 availability". |
| AWS | `eu-south-1` (Milan) | Referencia. |
| IBM Cloud | `eu-de` (Frankfurt) | Referencia. |

### Implicacion para AgroSatCopilot

GCP tiene region **en Milan** (`europe-west8`), optima para el serving del API y el frontend SSR
cercano al usuario italiano. Azure `westeurope` (Holanda) es la region donde el sponsor mantiene la
H100; la latencia de training no es critica (es batch/offline), por lo que la proximidad "Italy
proximity" de `westeurope` es suficiente para el rol de Azure (training + serving Qwen), mientras
que el path latency-sensible (chat, tiles) se sirve desde GCP Milan.

---

## 7. Factor 4 — Soporte de pipelines MLOps

| Proveedor | Servicio MLOps gestionado | Rol en AgroSatCopilot |
|-----------|---------------------------|------------------------|
| **GCP** | Vertex AI Pipelines + **Vertex AI Agent Engine** | Agent Engine hospeda el agente ADK (reasoner Gemini 2.5 Pro). Vertex Pipelines NO es el orquestador de datos del proyecto. |
| **Azure** | Azure Machine Learning | No usado; la H100 corre MLflow nativo + scripts, no Azure ML. |
| AWS | SageMaker Pipelines | Referencia. |
| IBM Cloud | watsonx.ai | Referencia. |

### Implicacion para AgroSatCopilot

La realidad MLOps del proyecto es **stack-agnostico, no un servicio gestionado de un vendor**:

- Orquestacion: **Dagster** (`dagster_project/`), assets con lineage declarativo.
- Tracking: **MLflow server en Docker `:5010`** (no `./mlruns`; gotcha documentado de los dos
  almacenes), con tags `data_version` + `code_version` por run.
- Versionado de datos/pesos: **DVC** con remote en GCS.

De los servicios gestionados, el unico realmente en uso es **Vertex AI Agent Engine** para hospedar
el agente conversacional ADK. NO se usan Vertex AI Pipelines, Azure ML ni SageMaker como
orquestadores: el lineage vive en Dagster + DVC + MLflow. Esto hace al pipeline portable y evita
lock-in del orquestador, alineado con la naturaleza open-source del proyecto.

---

## 8. Factor 5 — Partnerships academicos / creditos

| Proveedor | Programa academico / creditos | Aplicabilidad al proyecto |
|-----------|--------------------------------|---------------------------|
| **GCP** | GCP credits / Vertex AI Agent Builder trial | Ver **caveat** abajo: el credito grande no cubre la SKU que el proyecto usa. |
| **Azure** | Azure for Students + sponsor academico | La H100 del sponsor es el "partnership" real y mas valioso (24/7, $0 al equipo). |
| AWS | AWS Educate / AWS Activate | Referencia. |
| IBM Cloud | IBM Academic Initiative | Referencia. |

### Caveat obligatorio sobre creditos GCP

El "Trial credit for GenAI App Builder" (**$17,178**) es un credito de **Vertex AI Search / Agent
Builder**, y **NO cubre la SKU de generacion de texto de Gemini API**. Es decir, el credito grande
visible en la consola no subvenciona las llamadas del reasoner Gemini que el copiloto factura por
token. Por eso el gasto de Gemini API se paga aparte (centavos, ~$0.0001/descripcion FarSLIP; ver
§10) y no debe sobre-venderse como "cubierto por creditos academicos".

### Implicacion para AgroSatCopilot

El partnership academico de mayor impacto NO es un pool de creditos cloud sino el **sponsor que
presta la H100 NVL 96GB 24/7 sin costo al equipo**. Eso elimina el rubro mas caro (GPU de training)
del presupuesto y es la razon por la cual el operativo objetivo del proyecto es ~$115 USD/mes.

---

## 9. Decision arquitectonica

**GCP primario + H100 NVL 96GB del sponsor (Azure / on-prem) para training intensivo y serving
Qwen vLLM.** Justificacion por factor:

1. **EO (Factor 2) ata GCP de forma dura**: AlphaEarth `SATELLITE_EMBEDDING/V1/ANNUAL` v1.1
   CC-BY-4.0 solo se sirve gratis y global desde Google Earth Engine. Sin alternativa real fuera de
   GCP sin recomputar el FM (prohibido por decision de stack).
2. **Plano de control en GCP**: Cloud Run (scale-to-zero), Cloud SQL PostGIS+pgvector, GCS, Pub/Sub,
   Vertex AI Agent Engine, Artifact Registry, Secret Manager — todo en
   `infrastructure/terraform/modules/gcp/`.
3. **Latencia a Italia (Factor 3)**: GCP tiene region en Milan (`europe-west8`) para el path
   sensible (chat SSE, tiles); el training en Azure `westeurope` no es latency-critico.
4. **Precio H100 (Factor 1) irrelevante para el gasto**: no se compra H100 en ningun cloud; el
   sponsor la presta 24/7 ($0 al equipo). El precio de lista solo dimensiona el ahorro.
5. **MLOps (Factor 4) portable**: Dagster + MLflow `:5010` + DVC, no lock-in a Vertex Pipelines /
   Azure ML / SageMaker. Vertex AI Agent Engine solo para el agente ADK.

### Narrativa de soberania de datos (on-prem / sponsor)

El serving del LLM on-prem corre en la H100 del sponsor: **Qwen3.5-35B-A3B con vLLM GPTQ-Int4
single-GPU**. Esto da una narrativa de **soberania de datos**: las consultas del copiloto que el
operador desee mantener fuera de un LLM cloud (Gemini en Vertex) se pueden resolver on-prem sin que
los datos del cliente salgan del perimetro controlado por el equipo/sponsor. Gemma 4 LoRA queda
**OUT** del scope actual (ADR-009, future).

### Realidad v8 (no parked)

La H100 esta **activa 24/7, NO parked/apagada**. Es la VM `gjcamacho-gpuh1` (familia
`Standard_NC40ads_H100_v5`, H100 NVL 96GB, `westeurope`), prestada por el sponsor y administrada por
el. El bloque `module "azure"` en `infrastructure/terraform/environments/dev/main.tf:54` esta
**comentado a proposito** (la VM esta fuera del scope de nuestro Terraform porque la gestiona el
sponsor), no porque la H100 este apagada. Cualquier narrativa previa de "H100 reactivada/parked" del
plan v6 esta superada por ADR-009.

---

## 10. Trazabilidad FinOps (cifras reales del proyecto)

| Concepto | Cifra real | Fuente |
|----------|------------|--------|
| Operativo objetivo | **~$115 USD/mes** | Cloud Run scale-to-zero, `cloudrun_min_instances = 0` (`environments/dev/main.tf:44`, var `gcp/variables.tf:57`) |
| Training unico historico | **$262 (spot) - $602 (on-demand)** | Capacidad propia historica; hoy H100 sponsor 24/7 -> coste efectivo tiende a $0 |
| GCP acumulado a la fecha | **~$0.30-$0.49 USD** | Facturacion GCP del proyecto |
| Gemini API | **centavos (~$0.0001/descripcion FarSLIP)** | Vertex AI Gemini, por token |
| Cloud SQL dev | **apagada** (`db_activation_policy = "NEVER"`) | `environments/dev/main.tf:43`, var `gcp/variables.tf:40` (evita drift) |
| Disco `farslip-data` | reducido **250 -> 125 GB** | Procedimiento snapshot -> disco nuevo -> rsync -> import TF |
| H100 NVL 96GB | **$0 al equipo** (sponsor, 24/7) | VM `gjcamacho-gpuh1`, prestada por el sponsor |

> **Caveat de creditos (repetido por importancia):** el credito "Trial credit for GenAI App Builder"
> ($17,178) es de Vertex AI Search / Agent Builder y NO cubre la SKU de generacion de texto de
> Gemini API. No se debe presentar como cobertura del gasto del reasoner.

---

## 11. Referencias

**Anclas reales del proyecto:**

- `infrastructure/terraform/modules/gcp/main.tf` — Cloud Run, Cloud SQL PostGIS+pgvector, GCS,
  Pub/Sub, Vertex AI, Artifact Registry, Secret Manager.
- `infrastructure/terraform/environments/dev/main.tf:43-44` — `db_activation_policy = "NEVER"`,
  `cloudrun_min_instances = 0`.
- `infrastructure/terraform/environments/dev/main.tf:54` — `module "azure"` comentado (VM del
  sponsor fuera del scope TF).
- `infrastructure/terraform/modules/azure/variables.tf:1` — `westeurope` = "Italy proximity, H100
  availability".
- `infrastructure/terraform/modules/gcp/variables.tf:40,57` — vars `db_activation_policy`,
  `cloudrun_min_instances`.
- [ADR-009](../decisions/ADR-009-h100-reactivacion-pivote-farslip-alcance-v8.md) — reactivacion H100
  + pivote FarSLIP + alcance v8.
- `context/RefinamientoPlaneacionAgroSatCopilot_v8.md` §US-063 (linea 2056).
- AlphaEarth Foundations `SATELLITE_EMBEDDING/V1/ANNUAL`, data v1.1, 64-dim, CC-BY-4.0 (Google Earth
  Engine).

**Precios de lista de GPU (referencia externa, consultados 2026-06-20):**

- GCP A3 / H100: Spheron Blog (`spheron.network/blog/google-cloud-a3-h100-pricing/`); gpucost.org
  (`gpucost.org/provider/gcp`). Datos a 2026-05-22.
- Azure `NC40ads_H100_v5`: Vantage Instances (`instances.vantage.sh/azure/vm/nc40adsh100-v5`);
  CloudPrice (`cloudprice.net/vm/Standard_NC40ads_H100_v5`). Datos a 2025-10.
- AWS `p5.48xlarge`: DoiT Compute (`compute.doit.com/spot/us-east-1/p5.48xlarge`); Spheron Blog
  (`spheron.network/blog/aws-h100-pricing-2026/`).
- IBM Cloud H100 (re-investigado, consultado 2026-06-25): **sin precio oficial por GPU-hora
  publicado**. Fuentes consultadas — ComputePrices.com (`computeprices.com/providers/ibm`, "We're
  actively tracking prices for IBM Cloud. Check back soon"); GPUPerHour (`gpuperhour.com`, no lista
  IBM entre sus 28 proveedores); Spheron 2026 (`spheron.network/blog/gpu-cloud-pricing-comparison-2026/`,
  dato a 2026-05-14, no incluye IBM); IntuitionLabs (`intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison`,
  dato a 2026-06-20, "IBM Cloud started offering H100 in late 2024 but pricing details are not widely
  published, so we omit them here"); docs IBM VPC accelerated profiles
  (`cloud.ibm.com/docs/vpc?topic=vpc-accelerated-profile-family`). Un dato secundario suelto cito
  ~$0.99/GPU-h (H100 NVL, 8x) **sin respaldo oficial** -> NO confirmado, no se usa. Registrado en
  `docs/blockers/epic10-notas.md` (B15).

> Los precios de lista cambian con frecuencia; verificar contra el calculador de pricing de cada
> proveedor antes de presupuestar. Aqui se citan como referencia con fecha, no como compromiso de
> gasto del proyecto (gasto H100 real = $0, sponsor).
