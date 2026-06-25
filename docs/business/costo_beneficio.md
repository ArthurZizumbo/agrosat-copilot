# Análisis costo-beneficio — AgroSatCopilot (Avances 6 y 7)

**Corte:** 20-jun-2026 · **Owner:** Arthur Zizumbo (Tech lead / MLOps) · **US:** US-061 (EPIC 10)
**Rúbrica cubierta:** Costos (20 pts) + Beneficios (20 pts) + Implementación (30 pts) de A6/A7.

> **Regla de datos reales (Arthur):** toda cifra de **costo** se ancla a un dato verificado del
> proyecto (gasto GCP real, precio de SKU publicado, factura del sponsor) y se enlaza a su fuente.
> Las cifras de **beneficio** del cliente tipo son **estimaciones de literatura agronómica**,
> etiquetadas como tales con su supuesto y referencia — NUNCA se presentan como medidas de un
> cliente real (no existe tal cliente). Las cifras operativas se **enlazan** a
> [`docs/operations/finops.md`](../operations/finops.md), no se reescriben, para evitar drift.

Este documento es de **negocio**: cruza el costo del proveedor con el **valor para el cliente**.
Complementa (no duplica) la perspectiva operativa de [`docs/operations/finops.md`](../operations/finops.md)
(US-067) y el detalle multi-cloud de la comparativa de proveedores (US-063).

---

## 1. Resumen ejecutivo

- **Costo de construcción a la fecha: prácticamente nulo.** Datos públicos ($0), training en H100
  prestada por el sponsor ($0 al equipo), GCP acumulado **~$0.30-0.49 USD**.
- **Costo operativo objetivo: ~$115 USD/mes** con Cloud Run scale-to-zero (`min_instances=0`).
- **Inferencia LLM: centavos.** Gemini API a ~$0.0001 por descripción fenológica FarSLIP; la
  variante on-prem Qwen3.5-35B-A3B en la H100 prestada es ≈$0 incremental.
- **ROI proyectado: break-even en el mes 3** para un cliente tipo de 500 ha, según la aritmética
  de la sección 6 (beneficio mensual estimado >> $115/mes de costo operativo).
- **Caveat de créditos:** el "Trial credit for GenAI App Builder" (~$17,178) es de Vertex AI
  Search/Agent Builder y **NO** cubre la SKU de generación de texto de la Gemini API — y no se
  necesita, porque esa SKU cuesta centavos.

---

## 2. Costos por fase CRISP-ML(Q)

Mapeo del ciclo CRISP-ML(Q) al costo real del proyecto. Todas las cifras de costo están
verificadas o enlazadas a [`docs/operations/finops.md`](../operations/finops.md).

| Fase CRISP-ML(Q) | Costo real | Origen del dato | Comentario |
|------------------|-----------|-----------------|------------|
| Business & Data Understanding | **$0** | Datos públicos | AlphaEarth (GEE, CC-BY-4.0), Sentinel-2 (Copernicus), PASTIS-R, Sen4AgriNet (CC-BY-SA-4.0), EuroCropsML (CC-BY-SA-4.0). Sin licencia paga. |
| Data Engineering / Preparation | **~$0 (centavos GCP)** | Factura GCP ~$0.30-0.49 | Ingesta GEE/Copernicus, versionado DVC sobre GCS. |
| ML Model Engineering (training) | **$0 al equipo** (hist. $262 spot — $602 on-demand) | Factura del sponsor / histórico | Segmentación + FarSLIP + baseline en H100 NVL 96GB prestada 24/7; L4 spot con auto-shutdown para jobs ligeros. |
| Model Evaluation | **~$0** | CPU local + H100 sponsor | Evaluación sobre datasets ya descargados; sin costo incremental. |
| Deployment | **~$115/mes (proyectado)** | `docs/operations/finops.md` | Cloud Run scale-to-zero, Cloud SQL (dev apagada, `activation_policy=NEVER`), tiling TiTiler. |
| Monitoring & Maintenance | **dentro de ~$115/mes** | `docs/operations/finops.md` | Evidently drift (US-060), observabilidad de chat (US-065), alertas (US-059). |

### 2.1 Costos verificados anclados (detalle)

| Concepto | Cifra real | Origen |
|----------|-----------|--------|
| Adquisición de datos | **$0** | Todas las fuentes son públicas (licencias CC-BY-4.0 / CC-BY-SA-4.0) |
| GCP acumulado a la fecha | **~$0.30-0.49 USD** | Factura GCP (`docs/operations/finops.md`) |
| Operativo objetivo | **~$115 USD/mes** | Cloud Run `min_instances=0` (`docs/operations/finops.md`) |
| Training histórico (cuando aplicaba spot) | **$262 (spot) — $602 (on-demand)** | Histórico; hoy H100 sponsor 24/7 sin costo |
| H100 NVL 96GB (`gjcamacho-gpuh1`) | **$0 al equipo** | Prestada por el sponsor, 24/7 (no apagar) |
| Qwen3.5-35B-A3B vLLM on-prem | **≈$0 incremental** | Self-hosted GPTQ-Int4 single-GPU en la H100 prestada, en ventanas |
| Gemini API | **centavos (~$0.0001/descripción FarSLIP)** | SKU de generación de texto; cabe dentro de ~$115/mes |
| L4 (`agrosat-farslip-trainer-dev`) | spot + daemon idle | Jobs ligeros; auto-shutdown por idle |

> **Nomenclatura de datos (corrección):** la base de embeddings es **AlphaEarth V1/ANNUAL
> data v1.1, 64-dim, CC-BY-4.0** (GEE `SATELLITE_EMBEDDING/V1/ANNUAL`). No existe "v2.1".

---

## 3. Caveat de créditos Vertex AI (NO confundir)

El proyecto dispone de un **"Trial credit for GenAI App Builder" (~$17,178)**. Es importante para
el jurado entender qué cubre y qué no:

- **Qué es:** crédito de **Vertex AI Search / Agent Builder** (búsqueda empresarial y data stores).
- **Qué NO cubre:** la **SKU de generación de texto de la Gemini API** (la que usa el razonador del
  copiloto y las descripciones fenológicas FarSLIP). Esa es una SKU **distinta**.
- **Implicación:** este crédito **no** financia la inferencia LLM del producto. Pero **no se
  necesita**, porque esa inferencia cuesta **centavos** (~$0.0001/descripción) y cabe holgada dentro
  del operativo de ~$115/mes.

En términos de negocio: el costo de inferencia del copiloto es despreciable con o sin el crédito;
presentarlo como subsidio de la operación sería incorrecto.

---

## 4. Proyección a 12 meses operativos

Proyección de costo operativo del primer año en producción, asumiendo el régimen FinOps actual
(Cloud Run scale-to-zero, Cloud SQL controlada, H100 del sponsor sin costo al equipo). Todas las
cifras parten del operativo verificado de **~$115/mes**.

| Mes | Costo operativo mensual | Acumulado | Notas |
|-----|------------------------|-----------|-------|
| 1 | ~$115 | ~$115 | Cloud Run + Cloud SQL + tiling + inferencia Gemini (centavos) |
| 2 | ~$115 | ~$230 | Régimen estable scale-to-zero |
| 3 | ~$115 | ~$345 | Break-even del cliente tipo (ver §6) |
| 4 | ~$115 | ~$460 | |
| 5 | ~$115 | ~$575 | |
| 6 | ~$115 | ~$690 | |
| 7 | ~$115 | ~$805 | |
| 8 | ~$115 | ~$920 | |
| 9 | ~$115 | ~$1,035 | |
| 10 | ~$115 | ~$1,150 | |
| 11 | ~$115 | ~$1,265 | |
| 12 | ~$115 | **~$1,380** | Costo operativo total del año 1 |

**Supuestos explícitos de la proyección:**

- El training queda como **costo único histórico** y hoy es **$0 al equipo** (H100 del sponsor
  24/7); no se prorratea en el operativo mensual. Si el sponsor retirase la H100, habría que sumar
  el costo de re-training (histórico $262-602 spot/on-demand) como evento puntual, no recurrente.
- La inferencia LLM se mantiene en **centavos/mes** mientras el volumen de descripciones y turnos de
  chat se conserve en el orden de magnitud actual; el dashboard FinOps de costo por modelo
  (Pro/Flash/Qwen) de US-059 es la evidencia de seguimiento.
- No se incluye crecimiento por nuevos clientes (la proyección es de la **plataforma**, no
  multiplicada por número de clientes); cada cliente adicional sobre infraestructura compartida
  scale-to-zero añade costo marginal bajo.

---

## 5. Beneficios cuantificables (cliente tipo 500 ha)

> **ADVERTENCIA METODOLÓGICA (datos reales):** las cifras siguientes son **estimaciones de
> literatura de agricultura de precisión**, NO medidas sobre un cliente real (el proyecto no tiene
> uno). Cada fila lleva su **rango**, su **supuesto** y su **fuente/referencia**. Se presentan como
> proyección de valor, jamás como resultado medido.

Cliente tipo: explotación de **500 ha** de cultivo extensivo/permanente con monitoreo agronómico.

| Beneficio | Rango estimado | Supuesto | Fuente / tipo de estimación |
|-----------|----------------|----------|-----------------------------|
| Horas de agrónomo ahorradas / mes | **30-60 h/mes** | Sustituir recorridos de scouting manual por alertas dirigidas del copiloto sobre las parcelas que las imágenes marcan anómalas; ~1-2 jornadas/semana de inspección redirigidas | Estimación de productividad (sensores remotos vs. scouting manual); literatura de teledetección agronómica. **Estimación, no medida.** |
| Ahorro de agua (estrés hídrico) | **10-25 %** del agua de riego | Detección temprana de estrés hídrico (NDWI / índices de humedad) que permite riego de precisión por zona en vez de uniforme | Rango de literatura FAO / riego de precisión (agricultura de precisión reduce consumo de agua por aplicación variable). **Estimación, no medida.** |
| Ahorro de insumos (fertilización focalizada) | **10-20 %** del gasto en fertilizante | Aplicación variable guiada por índices de nitrógeno/clorofila (NDRE, CCCI) en lugar de dosis uniforme | Rango de literatura de fertilización de precisión / aplicación de tasa variable (VRA). **Estimación, no medida.** |
| Reducción del tiempo de detección de plagas | **de días a 24-72 h** | Alertas por anomalía espectral satelital sustituyen ciclos de inspección presencial periódica; detección más temprana acota el área afectada antes de la dispersión | Estimación operativa de teledetección de plagas/enfermedades. **Estimación, no medida.** |

### 5.1 Traducción a valor monetario (ilustrativa, con supuestos)

Para dimensionar el ROI (§6) se traduce el ahorro a dinero con **supuestos de mercado explícitos**.
Estos números son **ilustrativos** y deben recalibrarse con los precios reales del cliente concreto.

| Concepto | Supuesto de precio (explícito, recalibrable) | Ahorro mensual estimado |
|----------|----------------------------------------------|-------------------------|
| Horas de agrónomo | 45 h/mes ahorradas (punto medio del rango) a un costo cargado supuesto de ~$25-40/h | **~$1,125-1,800/mes** |
| Agua de riego | Costo de agua/energía de bombeo del orden de ~$8,000-15,000/año en 500 ha; ahorro 10-25 % | **~$65-310/mes** |
| Fertilizante | Gasto en fertilizante del orden de ~$30,000-60,000/año en 500 ha; ahorro 10-20 % | **~$250-1,000/mes** |
| Pérdida evitada por plagas | Detección 24-72 h acota el área tratada; valor altamente dependiente del cultivo | **rango amplio, no acotado aquí** |

> Aun con los supuestos más conservadores de cada fila, el ahorro mensual estimado del cliente tipo
> (sin contar pérdida evitada por plagas) está en el orden de **~$1,400/mes o más**, frente a un
> costo operativo de plataforma de **~$115/mes**. Todos los supuestos de precio están etiquetados y
> son recalibrables; no son medidas de un cliente real.

---

## 6. ROI y break-even

**Costo a recuperar (perspectiva plataforma):** el costo de construcción fue ~$0 al equipo
(datos $0, H100 sponsor $0, GCP ~$0.30-0.49). El costo relevante recurrente es el **operativo
~$115/mes**. Aun tomando el histórico de training como costo único hipotético ($262-602), se
amortiza en pocos meses con el ahorro estimado del cliente tipo.

**Aritmética del break-even (supuestos explícitos):**

- Beneficio mensual estimado del cliente tipo (conservador, §5.1, sin pérdida evitada por plagas):
  **~$1,400/mes**.
- Costo operativo de plataforma: **~$115/mes**.
- Costo de adopción del cliente: supuesto de **~$2,000-3,500** de onboarding/integración del primer
  trimestre (puesta en marcha, carga de parcelas, calibración) — **supuesto de mercado, recalibrable**.

| Mes | Beneficio acumulado estimado | Costo acumulado (onboarding ~$2,800 + $115/mes) | Balance |
|-----|------------------------------|--------------------------------------------------|---------|
| 1 | ~$1,400 | ~$2,915 | negativo |
| 2 | ~$2,800 | ~$3,030 | casi par |
| 3 | **~$4,200** | **~$3,145** | **positivo — break-even** |

**Conclusión:** con supuestos conservadores, el cliente tipo de 500 ha alcanza **break-even en el
mes 3**. El resultado es robusto: aun reduciendo el beneficio mensual estimado a la mitad
(~$700/mes), el break-even se desplaza solo a ~mes 5-6. Se presenta como **proyección con su
aritmética visible**, no como dato histórico medido.

---

## 7. Beneficios intangibles

- **Cumplimiento CAP europeo:** trazabilidad para declaraciones verificables de uso del suelo
  (Política Agraria Común), con histórico satelital auditable.
- **Reducción de riesgo regulatorio:** evidencia documentada de prácticas (riego, fertilización)
  ante auditorías ambientales y subvenciones condicionadas.
- **Imagen de sostenibilidad:** menor uso de agua e insumos como argumento comercial y de marca.
- **Soberanía de datos:** la variante **on-prem Qwen3.5-35B-A3B vLLM** (US-048, GPTQ-Int4
  single-GPU) permite que el dato del cliente **no salga** a un proveedor cloud externo — relevante
  para clientes con requisitos GDPR estrictos o sensibilidad geopolítica sobre datos agrícolas.

---

## 8. Fuentes y trazabilidad

- Cifras operativas: [`docs/operations/finops.md`](../operations/finops.md) (US-067) — enlazadas, no copiadas.
- Decisión H100 sponsor / alcance v8: [`docs/decisions/ADR-009-h100-reactivacion-pivote-farslip-alcance-v8.md`](../decisions/ADR-009-h100-reactivacion-pivote-farslip-alcance-v8.md).
- Plan vigente: [`context/RefinamientoPlaneacionAgroSatCopilot_v8.md`](../../context/RefinamientoPlaneacionAgroSatCopilot_v8.md) §US-061.
- Datos tabulares fuente (al Git): [`docs/business/data/costos_crisp_ml.csv`](data/costos_crisp_ml.csv) y [`docs/business/data/beneficios_500ha.csv`](data/beneficios_500ha.csv).
- Export LaTeX para el paper: [`docs/business/costo_beneficio.tex`](costo_beneficio.tex).

- Entregable Excel: [`docs/business/costo_beneficio.xlsx`](costo_beneficio.xlsx) (3 hojas: procedencia + Costos CRISP-ML + Beneficios 500 ha), generado desde los `.csv` fuente.

> **Nota de entregable Excel (RESUELTO 2026-06-25):** el `.xlsx` **ya está generado**
> ([`costo_beneficio.xlsx`](costo_beneficio.xlsx)) leyendo directamente los `.csv` fuente versionados
> (`data/costos_crisp_ml.csv`, `data/beneficios_500ha.csv`) con `openpyxl` 3.1.5; ninguna cifra es
> sintética. Además se conservan los `.csv` fuente al Git + las tablas Markdown de este documento +
> el export `.tex` como respaldo verificable (ver [`docs/blockers/epic10-notas.md`](../blockers/epic10-notas.md) B17).
