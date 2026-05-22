# ADR-006 — Reencuadre del baseline hacia un enfoque fenologico-temporal

**Status**: Propuesta · pendiente visto bueno equipo + sponsor (Dr. Camacho)
**Fecha**: 2026-05-22
**Decisores**: Arthur Zizumbo (MLOps Lead), Aaron Bocanegra, Isaac Avila
**US relacionada**: US-019/020/021/022 (Baseline EPIC 4), US-017 (FarSLIP), nueva US-022-b
**Avance**: A3 (Baseline · calificacion dom 24-may-2026)
**Paper-faro**: Wen et al. (2025), "Phenology description is all you need!", ISPRS Journal of
Photogrammetry and Remote Sensing 228, pp. 141-165. DOI 10.1016/j.isprsjprs.2025.07.002.
Archivo local: [`docs/general/paper.pdf`](../general/paper.pdf).

---

## Contexto

El baseline tabular del EPIC 4 (Random Forest + XGBoost sobre features combinados)
entrego F1-macro ~0.32 sobre 17 clases PASTIS-R con desbalance de clases ~31x
(`permanent_long_cycle` 44.930 parcelas vs `root_crops` 1.422). El resultado esta
muy por debajo de la meta de la rubrica (F1-macro >= 0.60).

Tres observaciones convergen y motivan este ADR:

1. **Feedback del sponsor (Dr. Camacho).** Indico explicitamente: indagar por que
   los clusters estan revueltos, quitar las coordenadas para clusterizar las FE,
   explicar que pasa en los clusters, filtrar las clases dominantes por el
   desbalance, obtener embeddings por fenologia y reconocer fenologias en otras
   zonas, y producir descripciones textuales de parcelas/mascaras.

2. **El paper-faro.** El Dr. apunto al paper Wen et al. (2025). Su tesis: la
   clasificacion de cultivos *closed-set* tiene techo bajo en escenarios de
   generalizacion (sus baselines TempCNN/LSTM/CLIP dan F1 zero-shot 8-20%); la via
   es zero-shot guiado por **descripcion fenologica** generada por un LLM, alineando
   curvas NDVI temporales con texto fenologico. Wen et al. alcanzan OA 55.8% /
   F1 53.4% zero-shot cross-region. **Entrenaron en una RTX 3090 (24 GB)** — hardware
   comparable a una L4.

3. **El baseline RF no esta alineado con el problema.** RF/XGB son tabulares y
   atemporales; el problema (clasificacion de cultivos) es intrinsecamente temporal
   (fenologico). El baseline cumplio la rubrica del A3 pero no es el baseline
   conceptualmente correcto para lo que el proyecto resuelve.

Restriccion de hardware confirmada (2026-05-22): la cuota GCP aprobada es
`GPUS_ALL_REGIONS=1` + `NVIDIA_L4_GPUS=1` en las 7 regiones revisadas.
`NVIDIA_A100_GPUS=0` y `A2_CPUS=0` en todas. No hay A100 disponible; la H100 de
Azure sigue sin confirmarse.

---

## Decision

Se reencuadra el baseline del proyecto hacia un enfoque **fenologico-temporal**,
manteniendo el plan v6 (no se reescribe) y ajustandolo con las siguientes
decisiones:

### D1 — El baseline del A3 pasa a ser fenologico-temporal

El Avance 3 (calificacion 24-may) se entrega con un baseline coherente con el
problema, no solo con RF/XGB tabular:

- Se explota la FE fenologica **ya existente** en `ml/features/temporal_features.py`
  (24 columnas FFT — 4 amplitudes + 4 fases sobre NDVI/NDWI/EVI; 8 columnas
  fenologicas — SOG, peak, senescencia, AUC, slopes, duracion de madurez).
- Se entrenan **tres modelos**: XGBoost (tabular-fenologico, CPU) + TempCNN +
  InceptionTime (temporales, GPU local).
- Se ejecutan **ablations de features** para decidir el conjunto optimo
  (con/sin geometria, con/sin ERA5+SRTM redundante con AlphaEarth, solo-fenologicas).
- Se descarta la **zona geografica** (lat/lon, columnas `geom_*`) de la matriz de
  clustering/clasificacion: introduce leakage espacial (el modelo aprende *donde*
  esta la parcela, no *que* cultivo es). Confirmado por feedback del Dr.

### D2 — El paper Wen et al. (2025) es el marco de referencia del proyecto

La narrativa del proyecto se reencuadra: el baseline closed-set (F1 0.32) es la
evidencia del techo del enfoque simple; EPIC 5 (TSViT, U-TAE, SegFormer) y EPIC 6
implementan el pivote hacia el enfoque fenologico/zero-shot que Wen et al.
demuestran superior. RF/XGB no se elimina — queda como baseline tabular de
referencia.

### D3 — L4 es el hardware objetivo; se deja de bloquear el proyecto por A100/H100

Wen et al. lograron SOTA zero-shot en una RTX 3090 (24 GB). Las arquitecturas del
PASTIS benchmark son ligeras (TSViT 1.6M params, U-TAE 1.1M; sub-patches 24x24 y
variante TSViT-S disenadas para consumer-grade GPU). El proyecto se disena para
**L4 24 GB**. La H100 queda como dependencia unica de Gemma 4 26B-MoE LoRA
(EPIC 6), a resolver con Azure sin bloquear el resto.

### D4 — Gemini 3.5 Flash para generar descripciones fenologicas

Para la rama semantica (descripciones fenologicas tipo-GPT4 del metodo Wen) se usa
**Gemini 3.5 Flash** en lugar de Gemini 3.1 Pro: es ~10-20x mas barato y la tarea
(texto estructurado a partir de curvas NDVI) no requiere el modelo mas potente.
Esto ajusta el stack aprobado (el v6 lista "Gemini 3.1 Pro" como LLM cloud) y por
eso se documenta aqui. **Gemma 4 26B-MoE se mantiene** como el VLM de fine-tuning
LoRA del EPIC 6 — no se reemplaza; cumplen roles distintos.

### D5 — Se crea la US-022-b para la deuda tecnica y la infra

Se formaliza una nueva **US-022-b** que agrupa el trabajo que NO cabe en el A3 y
NO debe mezclarse con el (ver [`docs/us-planning/us-022b.md`](../us-planning/us-022b.md)).
Incluye, entre otras, completar FarSLIP (deuda de US-017 Fase 4) e infra GCP L4.
US-022-b se ejecuta **despues del A3**, en su propia ventana.

### D6 — Multi-dataset y cross-region van al Paper Track

La extension a otros datasets geoespaciales (TF311 Alemania, EuroCrops) y la
evaluacion cross-region (replicar el experimento de Wen et al. en datos europeos)
se planifican como Paper Track post-21-jun. No entran al MVP por calendario.

### D7 — Sin modelos nuevos fuera del stack v6

Se confirma el descarte de modelos no aprobados (Prithvi-EO-2.0 y otros listados
en CLAUDE.md). Embeddings recientes (Presto, Galileo, TerraMind) se anotan como
trabajo futuro/ablation; no se integran al MVP. El stack v6 (AlphaEarth, DINOv3,
las 6 arquitecturas de segmentacion, Gemma 4) es suficiente.

---

## Consecuencias

**Positivas**:

- El A3 entrega un baseline coherente con el problema (temporal), no un RF puesto
  por cumplir rubrica.
- El proyecto deja de bloquearse esperando A100/H100; L4 es suficiente para casi
  todo (Wen et al. lo demuestran).
- La narrativa gana un marco academico solido (paper ISPRS top-tier) y conecta
  directamente con el feedback del sponsor.
- La FE fenologica pesada **ya existe** — el A3 es analisis + uso, no construccion.

**Negativas / riesgos asumidos**:

- D3: TSViT/U-TAE en L4 son mas lentos que en H100; pueden requerir subset de
  patches o menos epocas (resultado "preliminar" documentado). Swin-UNETR puede
  no entrenar a escala completa en L4 — se documenta como limitacion de recursos.
- D4: usar Gemini 3.5 Flash ajusta el stack v6; requiere visto bueno del equipo.
- El reencuadre concentra la ambicion en A5 + Paper Track; A3 y A4 cumplen rubrica
  con lo disponible.

---

## Alternativas consideradas

- **Reescribir el plan v6**: descartada. El v6 ya incluye TSViT, FarSLIP,
  AlphaEarth, Gemma 4 — apunta al territorio correcto. Solo necesita ajuste.
- **Conseguir A100/H100 antes del A3**: descartada. Cuota A100=0 en 7 regiones;
  no es problema regional sino de historial de billing del proyecto.
- **Mantener RF/XGB tabular como unico baseline**: descartada. No esta alineado
  con el problema temporal; insuficiente como baseline conceptual.
- **Meter FarSLIP + infra + multi-dataset en el A3**: descartada. 7 tareas en 48h
  es inviable; hunde la entrega del domingo. Se mueven a US-022-b.

---

## Trazabilidad

- Engram: decision del reencuadre, creacion de US-022-b, estado real de US-022.
- Plan de refinamiento: ver nota en [`docs/us-planning/us-022b.md`](../us-planning/us-022b.md).
- Resolucion US-022: [`docs/us-resolved/us-022.md`](../us-resolved/us-022.md).
