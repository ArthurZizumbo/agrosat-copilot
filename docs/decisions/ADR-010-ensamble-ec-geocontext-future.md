# ADR-010 — Ensamble E-c geo-contextual (clima + elevacion + vecindad + refinamiento estructurado) como trabajo FUTURE

**Status**: Propuesta (FUTURE — diseno-only, NO se ejecuta antes de la presentacion del 27-jun-2026)
**Fecha**: 2026-06-12
**Decisores**: Arthur Zizumbo (MLOps Lead) · equipo Equipo 17
**US relacionada**: US-043 (EPIC 6 — Ensamble E-c, diseno-only, 2 SP documentales)
**Punto de partida conceptual**: [US-042](../us-resolved/us-040.md) (E-b: stacking E-a + AlphaEarth, member-generico, reconciliacion pixel<->parcela, anti-fuga fold-5)
**Fundamento**: [`context/RefinamientoPlaneacionAgroSatCopilot_v8.md`](../../context/RefinamientoPlaneacionAgroSatCopilot_v8.md) (EPIC 6, US-043; §FUTURE) · evidencia propia US-020 / US-022b / US-022-c
**Codigo de referencia (a extender en el futuro, NO ahora)**: [`ml/ensemble/stacking.py`](../../ml/ensemble/stacking.py), [`ml/features/fusion.py`](../../ml/features/fusion.py), [`ml/utils/parcel_reconcile.py`](../../ml/utils/parcel_reconcile.py)

---

## Contexto

El EPIC 6 construye el modelo final como una familia de **ensambles**: los 4 de
rubrica (Voting/Bagging/Stacking/Blending, US-040) mas tres incrementales — **E-a**
(TSViT-pheno + FarSLIP, US-041), **E-b** (+ AlphaEarth tabular, US-042) y **E-c**
(geo-context, este ADR). El campeon medido hasta hoy es el Stacking heterogeneo a
nivel parcela (F1-macro **0.7470** fold-5 held-out, 18 clases; el Stacking-5 con
FarSLIP lo lleva a 0.7486). E-b responde, de forma honesta, si el embedding FM
tabular AlphaEarth aporta senal complementaria sobre el camino denso + contrastivo.

La pregunta natural que cierra la serie es: **¿queda senal que el ensamble actual
NO captura por construir su decision parcela-a-parcela de forma independiente, sin
mirar el contexto geografico (clima, relieve) ni la estructura espacial (que
cultivan los vecinos)?** E-c es el sketch de como se atacaria esa pregunta. Esta US
es **DISENO ONLY**: el entregable es este documento, no codigo. El motivo es de
calendario y computo, no de falta de interes: a tres semanas de la presentacion el
cuello de botella es **tiempo** (ADR-009 D-1), y E-c implica ingesta GEE zonal nueva
(ERA5, SRTM), construccion de un grafo de adyacencia y entrenar/calibrar una capa de
refinamiento estructurado — un esfuerzo de 4-6 semanas que pertenece al Paper Track /
post-presentacion.

### Evidencia propia que obliga a un encuadre honesto

El equipo **ya midio** parte de la hipotesis de E-c, y el resultado es desfavorable
para la via mas obvia:

- **US-020** (feature importance + SHAP, plan v8 L766): las dimensiones AlphaEarth
  dominan; `geom_*` / ERA5 / SRTM son **redundantes** (delta F1 = **0.0**).
- **US-022b / US-022-c** (reencuadre fenologico, ADR-006; plan v8 L844, L859):
  quitar ERA5 + SRTM del vector tabular **no degrada** (delta = 0.0); "AlphaEarth ya
  los codifica". El modelo aprende fenologia, no geografia.

Es decir: **anadir ERA5/SRTM como columnas mas a un XGBoost-AlphaEarth NO mejora.**
Cualquier ADR que prometa ganancia por re-introducir esas features tabulares
contradiria la evidencia del propio proyecto. Por eso E-c **no se justifica por las
features tabulares**, sino por dos ejes que el equipo NO ha medido:

1. **El eje estructural** (vecindad espacial + refinamiento CRF/GNN): el ensamble
   actual decide cada parcela aislada. Un cultivo improbable rodeado de un cultivo
   dominante, o un borde de parcela ruidoso, son errores que una capa que imponga
   **consistencia espacial** podria corregir — y eso no es una feature tabular, es
   una restriccion sobre la *salida*.
2. **El eje denso / grafo**: el delta=0.0 se midio sobre el espacio **tabular** (XGB
   por parcela). ERA5/SRTM entrando como **canales de contexto de un CRF denso** o
   como **atributos de nodo de un GNN de parcelas** es un espacio distinto, no
   cubierto por aquella ablacion. Que ahi tampoco aporten es una hipotesis abierta,
   no un hecho ya establecido.

Este ADR documenta **ambos ejes** (features + estructura), con la advertencia
explicita de que la via tabular esta saturada.

---

## Decision

**Documentar E-c como sketch arquitectonico FUTURE, sin ejecutarlo**, con el
siguiente diseno. No se promete mejora cuantitativa: el entregable cientifico de una
futura US de implementacion seria responder, con la misma honestidad que E-b, si el
contexto geografico y/o el refinamiento estructural aportan sobre el campeon actual.

### D-1 — E-c parte de la salida de E-b, no la reemplaza

E-c **no es un ensamble nuevo de base learners**: es una **capa de post-proceso /
refinamiento** sobre el mapa de probabilidades del mejor ensamble vigente (E-b, o el
Stacking campeon). Reusa el contrato de E-b (probs post-softmax por parcela o densas
`(18, H, W)`, anti-fuga fold-5, reconciliacion pixel<->parcela ya implementada en
`ml/utils/parcel_reconcile.py`).

### D-2 — Diagrama del flujo

```
                       SALIDA DE E-b (US-042)
            probs post-softmax   [ por parcela (18,) ]  o  [ densa (18, H, W) ]
                                  |
        +-------------------------+-------------------------+
        |                         |                         |
   GEO-CONTEXT (nuevo)      VECINDAD (nuevo)         ESTRUCTURA (nuevo)
        |                         |                         |
  ERA5 (clima)            grafo de adyacencia        capa de refinamiento
  SRTM (elevacion)        de parcelas PASTIS         (UNA de dos opciones):
  via GEE zonal           (k-NN espacial +
        |                  borde compartido)          (A) CRF denso
        |                         |                       unary = probs E-b
        +-----------+-------------+                       pairwise = geo + espacio
                    |                                  (B) GNN de parcelas
            features de contexto                          nodo = [probs ‖ ERA5 ‖ SRTM]
            por parcela / pixel                           arista = adyacencia / k-NN
                    |                                         |
                    +--------------------+--------------------+
                                         |
                            SALIDA REFINADA (E-c)
                  probs / labels por parcela, espacialmente coherentes
                                         |
                       eval fold-5 held-out (18 clases, harness US-030)
                       comparacion HONESTA E-c vs E-b (delta, sin sobre-afirmar)
```

### D-3 — Features candidatas (con su nota de redundancia conocida)

| Grupo | Feature | Fuente / asset GEE | Resolucion | Como entra a E-c | Nota honesta |
|-------|---------|--------------------|------------|------------------|--------------|
| **Clima (ERA5)** | Temp. media/min/max mensual; precip. acumulada mensual; GDD (grados-dia de crecimiento); deficit hidrico | `ECMWF/ERA5_LAND/MONTHLY_AGGR` (GEE) | ~11 km (downscale a parcela por interpolacion zonal) | Canal de contexto del CRF (pairwise) o atributo de nodo del GNN | Como **columna tabular XGB** no aporta (delta=0.0, US-020/022b). Como contexto espacial/grafo: no medido. |
| **Elevacion (SRTM)** | Elevacion media; pendiente; orientacion (aspect); TWI (indice topografico de humedad) | `USGS/SRTMGL1_003` (GEE) | 30 m | Igual que ERA5: contexto/nodo, no columna XGB | Igual: redundante en tabular; abierto en el eje estructural. |
| **Vecindad de parcela** | Clase modal de los k vecinos; entropia de clases vecinas; fraccion del cultivo dominante local; distancia al borde del cultivo | Derivado del raster `ParcelIDs` + geometrias (`metadata.geojson`, EPSG:2154) | parcela | Atributo de nodo (GNN) o termino pairwise (CRF) | **Sin medir** — es justamente la senal que el ensamble parcela-aislada ignora; el caso de mayor valor potencial de E-c. |

> Las features de clima/relieve **no se ingieren en esta US** (DISENO ONLY). La tabla
> documenta el *como* para cuando se implemente. ERA5/SRTM ya existen parcialmente en
> `ml/features/fusion.py` (medidos redundantes en el eje tabular); E-c las usaria en
> un eje distinto, no las re-anadiria al XGBoost.

### D-4 — Capa de refinamiento estructurado: CRF vs GNN (dos sub-opciones, NO se elige aqui)

Se documentan dos caminos. La eleccion entre ellos es parte de la futura US de
implementacion (depende de si se trabaja en el espacio denso o de parcelas).

**Opcion A — CRF denso sobre el mapa de probabilidades (espacio denso `(18, H, W)`)**

- Libreria de referencia: `pydensecrf` (fully-connected CRF, Krahenbuhl & Koltun 2011).
- **Unary potentials** = `-log(probs)` de E-b por pixel (la confianza del ensamble).
- **Pairwise potentials** = kernels gaussianos sobre (a) posicion espacial (x, y),
  (b) similitud de geo-context (ERA5/SRTM) y, opcionalmente, (c) color del composite
  S2. Penaliza que dos pixeles vecinos con clima/relieve/posicion similares tengan
  etiquetas distintas -> impone **coherencia espacial**.
- Pros: sin entrenamiento de red (los pesos de los kernels se ajustan por grid/Optuna
  sobre sub-folds del fold-5, anti-fuga); barato; interpretable.
- Contras: opera en pixeles, no respeta de forma nativa los bordes de parcela PASTIS
  (habria que reconciliar a parcela despues, como en E-b); el clima ERA5 a ~11 km
  varia poco dentro de un patch de 1.28 km -> su termino pairwise puede ser casi
  constante (poco discriminante a esa escala).

**Opcion B — GNN de parcelas (espacio de parcelas, grafo de adyacencia)**

- Libreria de referencia: `torch_geometric` (GraphSAGE o GAT).
- **Nodo** = una parcela, con embedding inicial `[probs E-b (18) ‖ ERA5 ‖ SRTM ‖
  features de vecindad]`.
- **Arista** = adyacencia espacial (parcelas que comparten borde) y/o k-NN por
  centroide. El message-passing propaga contexto entre vecinos.
- **Salida** = probs refinadas por parcela; eval directo a nivel parcela (sin
  reconciliacion extra, ya esta en el espacio correcto).
- Pros: nativo al nivel parcela (el espacio donde el ensamble ya opera y donde la GT
  es limpia, pureza ~98%); el grafo modela explicitamente "que cultivan mis vecinos";
  ERA5/SRTM entran como atributos sin el problema de escala del CRF denso.
- Contras: requiere entrenar una red (mas computo y riesgo de overfit con pocas
  parcelas por fold); el grafo de adyacencia hay que construirlo desde `ParcelIDs` +
  geometrias; el anti-fuga debe cuidar que los sub-folds espaciales no partan un
  vecindario (R-LEAK sobre aristas).

> **Recomendacion tentativa para la futura US** (no vinculante): empezar por la
> **Opcion B (GNN de parcelas)**, porque opera en el nivel donde el ensamble ya es
> fuerte y la vecindad —el eje no medido y de mayor valor potencial— es de primera
> clase en el grafo. El CRF denso queda como baseline barato de comparacion.

### D-5 — Anti-fuga y reporte (heredados de E-a/E-b, NON-NEGOTIABLE)

Cualquier implementacion futura de E-c **debe** respetar el harness US-030: reporte
en **fold-5 held-out**, 18 clases (apples-to-apples con E-a/E-b y el Stacking 0.7470),
probs post-softmax (no logits), capa de refinamiento ajustada SOLO sobre sub-folds
espaciales del fold-5 (`build_spatial_kfold`, `assert_oof_only`), con la cautela
extra de **no partir vecindarios** entre train/eval del grafo (R-LEAK-GRAPH).

### D-6 — Encuadre honesto (no se promete mejora)

El entregable de una futura implementacion es **responder** si E-c supera a E-b, no
afirmar que lo hara. Resultados validos:

- E-c > E-b: el refinamiento estructural recupera errores de coherencia espacial.
- E-c ~= E-b: el ensamble ya captura el contexto via AlphaEarth (consistente con el
  delta=0.0 tabular extendido al eje estructural) — resultado cientifico, no fracaso.
- E-c < E-b: el refinamiento sobre-suaviza y borra parcelas raras correctas — una
  leccion util sobre los limites de imponer coherencia espacial en un mosaico
  agricola heterogeneo.

---

## Estimacion de esfuerzo (FUTURE, 4-6 semanas)

| Etapa | Trabajo | Estimacion |
|-------|---------|------------|
| Ingesta GEE zonal ERA5 + SRTM | Jobs GEE zonales por parcela (clima mensual + relieve), cache parquet, atribucion de licencia | 1-1.5 sem |
| Grafo de adyacencia de parcelas | Construir aristas (borde compartido + k-NN) desde `ParcelIDs` + geometrias; validar conectividad; particion espacial sin cortar vecindarios | 1 sem |
| Capa de refinamiento | Implementar CRF (`pydensecrf`) y/o GNN (`torch_geometric`); ajustar sobre sub-folds del fold-5; checkpoints | 1.5-2 sem |
| Eval + reporte honesto | Eval fold-5 18 clases; tabla E-c vs E-b + delta; figuras (confusion, residuos espaciales antes/despues del refinamiento); run MLflow | 0.5-1 sem |
| Documentacion / Paper Track | Seccion del paper con el resultado (mejore o no) | 0.5 sem |

**Dependencias**: US-042 (E-b) cerrada y su OOF disponible; cuota GEE para los jobs
zonales; (si GNN) GPU para entrenar la red. **No** consume la ventana H100 de la
presentacion (es post-27-jun).

---

## Licencia / legal (a documentar cuando se ingiera, futuro)

- **ERA5 / ERA5-Land** — Copernicus Climate Change Service (C3S), Copernicus Climate
  Data Store (CDS). Licencia de uso del CDS (Licence to Use Copernicus Products);
  requiere atribucion a Copernicus/ECMWF. Acceso via GEE
  (`ECMWF/ERA5_LAND/MONTHLY_AGGR`) o CDS API. Se anadira a
  [`docs/licenses/DATA_LICENSE.md`](../licenses/DATA_LICENSE.md) al ingerir.
- **SRTM** — USGS / NASA. Dominio publico (sin restriccion de uso); se cita la
  fuente (USGS EROS / NASA JPL). Acceso via GEE (`USGS/SRTMGL1_003`). Se anadira a
  `DATA_LICENSE.md` al ingerir.

---

## Consecuencias

- **Positiva**: la serie de ensambles (E-a -> E-b -> E-c) queda cerrada con un
  trabajo futuro **creible y honesto**, no con una promesa vacia. El sketch es
  ejecutable por quien retome el Paper Track sin re-investigar.
- **Positiva**: el ADR convierte un resultado negativo propio (delta=0.0 de
  ERA5/SRTM tabular) en un encuadre de diseno preciso — el valor de E-c, si lo hay,
  esta en lo estructural, no en re-anadir features saturadas.
- **Neutra**: no consume computo ni schedule del horizonte de 3 semanas (DISENO ONLY).
- **Riesgo asumido**: E-c puede no mejorar a E-b; el ADR lo declara de antemano. El
  entregable futuro es la respuesta honesta, no la ganancia.

---

## Relacionado

- [ADR-009](ADR-009-h100-reactivacion-pivote-farslip-alcance-v8.md) — reactivacion H100 y alcance v8 (define la regla "FUTURE/diseno-only se difiere"; este ADR ejecuta esa politica para E-c).
- [US-040 resolved](../us-resolved/us-040.md) — los 4 ensambles base + el Stacking campeon (0.7470) que E-c refinaria.
- [US-042 planning](../us-planning/us-042.md) — E-b (punto de partida conceptual de E-c).
- [ADR-006](ADR-006-reencuadre-baseline-fenologico.md) — reencuadre fenologico (origen del hallazgo delta=0.0 de ERA5/SRTM).
- Skill [`agrosat-ml-ensemble`](../../.claude/skills/agrosat-ml-ensemble/SKILL.md) — enlaza a este ADR como referencia de E-c FUTURE.
- Plan v8 §EPIC 6 / US-043 — bloque de origen de esta US.
