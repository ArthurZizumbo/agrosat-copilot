# US-078 y US-079 — Homologo PASTIS de Italia + Extension del modelo campeon

> Planeacion detallada (sin alucinaciones). Cada paso esta anclado a artefactos y
> cifras REALES verificadas el 25-jun-2026. Las rutas, formatos y numeros que
> aparecen abajo fueron comprobados ejecutando codigo, no de memoria.

## Contexto y motivacion (por que estas dos US)

El modelo campeon (ensamble de segmentacion + tabular) se entreno y valido sobre
**PASTIS-R** (Francia). Para demostrar **transfer learning real** y la hipotesis
de **taxonomia enriquecida** (clases finas que PASTIS no tiene), se necesita un
segundo dataset:

1. En el **mismo formato que PASTIS** (patch temporal denso + mascara de
   segmentacion), para que los modelos densos (TSViT-pheno, U-TAE) operen en su
   formato nativo y el transfer sea limpio.
2. De un **dominio distinto** (clima mediterraneo) para que el transfer sea
   genuino, no in-distribution.

**Italia 2018 cumple ambas** y el dataset de poligonos ya esta descargado.

### Hallazgos verificados que fundamentan el diseno

- **PASTIS formato real** (verificado): imagen `S2_*.npy` de forma
  `(T, 10, 128, 128)` (T~40-60 fechas, 10 bandas, 128px, 10 m/px, L2A) +
  anotacion `ParcelIDs_*.npy` de forma `(128, 128) int32` (mascara densa: cada
  pixel lleva el id/clase de su parcela).
- **EuroCrops Italia 2018** (descargado, `data/reference/eurocrops_v2/iti1_2018.parquet`,
  229 MB): **643,206 parcelas con poligonos reales** (MultiPolygon, EPSG:3035),
  columnas `original_code`, `area_ha`, `geometry`.
- **Crosswalk codigo->HCAT** (descargado, `data/reference/eurocrops_v2/eurocrops_mapping.csv`):
  mapea `original_code` -> `hcat4_name` por pais (`nuts`). Para Italia (`nuts`
  empieza con `it`) mapea el **100%** de las parcelas (0 sin mapear).
- **Sentinel-2 es 10 m nativo** (verificado con context7 + prueba real): pedir
  mas fino (p.ej. "5 m") solo hace UPSAMPLING (el `std` espacial es identico al de
  10 m: 0.1535 = 0.1535); NO anade detalle. El homologo se descarga a **10 m/px**,
  exactamente como PASTIS.
- **Pipeline path -> serie -> mascara** validado end-to-end (imagenes en
  `reports/eurocrops_viz/`): se bajo la serie temporal real de un poligono
  italiano (fechas <20% nubes via catalogo CDSE) y se rasterizo una zona densa
  (166 parcelas, 10 clases, 84% del path cubierto por cultivo).
- **Cliente Sentinel Hub** (`ml/ingest/sh_client.py`): descarga recortes reales,
  con cache a disco (`_sh_patch_cache`, commit 042fc60) que evita re-descargar y
  retry/backoff en 429 (commit ee0a80d).

### EDA Italia 2018 (real, ejecutado)

- 643,206 parcelas, **176 clases HCAT4** (mapeo 100%).
- **38 clases con >=500 parcelas**, 31 con >=1000 (entrenables).
- Top 10 clases = 82.7% de las parcelas; top 20 = 94.5%.
- Top: `olive` 19.5% (MEDITERRANEO, no en PASTIS), `vineyards/grapes` 11.8%
  (= Grapevine en PASTIS), `permanent_grassland` 7.6% (= Meadow), `durum_hard_wheat`
  3.1% (= Winter durum wheat), `common_soft_wheat` 1.9% (= Soft winter wheat).
- **Lectura para el transfer**: Italia comparte ~6-8 clases con PASTIS (grapevine,
  meadow, wheat, sunflower, maize, barley) Y aporta clases nuevas mediterraneas
  (olive, durum) -> ideal para medir transfer + enriquecimiento de taxonomia.

### Decision del combinador (por que Voting-3 ponderado, no Stacking)

Resultados reales sobre PASTIS fold-5. CLAVE: hay dos formas de evaluar y dan
resultados OPUESTOS (corroborado el 25-jun):

**(A) Sobre las 18 clases completas** (`headline_voting4.csv`): el Voting iguala
al Stacking pero con menos pesos.

| Combinador | F1-macro (18 cl.) | Accuracy | F1 spatial-CV | # pesos |
|------------|-------------------|----------|---------------|---------|
| Stacking (meta-LogReg) | 0.747 | 0.849 | 0.536 | 54 |
| **Voting ponderado (3m)** | 0.7444 | **0.863** | **0.558** | **3** |
| Voting simple (1/N) | 0.673 | 0.833 | 0.488 | 3 |

**(B) RESTRINGIENDO el modelo a las clases bien resueltas** (`france9_headline.csv`,
`france10_headline.csv`) -- el escenario de despliegue real: el **Voting-3 GANA
claramente** y mantiene **F1 > 0.9**.

| Label-space | Voting-3 | Stacking-5 (campeon) | Voting gana |
|-------------|----------|----------------------|-------------|
| **france-9** (9 clases) | **0.9200** | 0.9035 | **+1.65 pts** |
| **france-10** (10 clases) | **0.9069** | 0.8927 | **+1.42 pts** |

CORROBORADO (Arthur, 25-jun): el Voting-3 gana **cuando se considera el espacio
restringido de ~10 clases bien resueltas** (modo B "restrict"), no en la curva de
las 18 (modo A). La razon: con menos clases el meta-LogReg del Stacking (54 pesos)
sobreajusta mas que el Voting (3 pesos). Como el despliegue real opera sobre las
clases bien resueltas (la "doble taxonomia" del proyecto), **el Voting-3 es el
combinador elegido: F1 0.9069 sobre 10 clases, mas ligero (3 pesos), mejor
generalizacion (spatial-CV 0.558 vs 0.536)**.

OBJETIVO de calidad para US-079: **F1-macro > 0.9 sobre las ~10 mejores clases**
del homologo italiano (espejo del 0.9069 france-10 del Voting-3 en PASTIS).

---

## US-078 — Generar el dataset homologo a PASTIS de Italia 2018

**Como agente / cientifico de datos, quiero** construir un dataset de Italia 2018
con el mismo formato exacto que PASTIS-R (patches temporales densos + mascaras de
segmentacion), **para** poder aplicar y evaluar los modelos densos del campeon en
un dominio nuevo (mediterraneo) sin cambiar su formato de entrada.

### Alcance

- Generar **N patches** de `(T, 10, 128, 128)` + mascara `(128, 128)` cada uno,
  desde los poligonos de Italia 2018, replicando la estructura de PASTIS.
- Mantener **10 m/px** (nativo Sentinel-2 = PASTIS), patches de **128px = 1.28 km**.
- Patches ubicados en **zonas densas de parcelas** (no zonas vacias).
- Serie temporal **densa** (objetivo: 30-45 fechas <20% nubes en la temporada).

### Flujo detallado (paso a paso, sin ambiguedad)

**Paso 1 — Cargar y etiquetar los poligonos.**
- Leer `data/reference/eurocrops_v2/iti1_2018.parquet` con geopandas
  (`engine="pyogrio"`); reproyectar de EPSG:3035 a EPSG:4326.
- Mapear `original_code` -> `hcat4_name` con `eurocrops_mapping.csv` filtrado a
  `nuts` que empiece con `it` (cobertura 100% verificada).
- Mapear `hcat4_name` -> id de clase contiguo `[0, K)`. Reservar id 0 para
  "fondo/no-cultivo" (pixeles sin parcela). GOTCHA: agrupar clases con <MIN_SUPPORT
  parcelas en una clase "other" para no inflar el espacio de etiquetas.

**Paso 2 — Seleccionar las zonas de los patches (los "paths").**
- Calcular el centroide de cada parcela (en CRS proyectado, no geografico — el
  centroid en EPSG:4326 da warning y es impreciso).
- Agrupar centroides en una grilla de ~0.012 deg (~1.3 km, el tamano del patch).
- Quedarse con las celdas de grilla con **mas parcelas** (zonas densas). Cada celda
  densa define el bbox de un patch de 1.28 km. GOTCHA verificado: si se elige una
  celda dispersa, el patch sale casi vacio (probado: una zona dispersa dio 1
  parcela, una densa dio 166).
- Estratificar la seleccion de celdas para cubrir las clases objetivo (que cada
  clase de interes aparezca en varios patches).

**Paso 3 — Descargar la serie temporal de cada patch (path) via Sentinel Hub.**
- Para el bbox del patch: consultar el catalogo (CDSE OData,
  `cdse_client.search_s2`) las fechas con **<20% nubes** en la temporada (p.ej.
  2018-03-01 a 2018-10-31).
- Para cada fecha limpia, descargar el recorte `(10, 128, 128)` a 10 m/px con
  `sh_client.crop(bbox, date_from, date_to, size=128, max_cloud=20)`. Las 10
  bandas en el orden PASTIS (`PASTIS_BANDS`).
- Apilar las fechas -> `(T, 10, 128, 128)`. El cache (`_sh_patch_cache`) garantiza
  que **ninguna fecha-patch se re-descarga** (proteccion de cuota, regla aprendida).
- COSTO: ~1 peticion por (patch, fecha). Con T~30 fechas, un patch son ~30
  peticiones. Por eso N de patches se dimensiona contra la cuota SH disponible
  (~10,800 peticiones restantes al 25-jun). Ej.: 100 patches x 30 fechas = 3,000
  peticiones (~28% de la cuota). Empezar con un piloto de 20-30 patches.

**Paso 4 — Rasterizar las mascaras (los labels densos estilo PASTIS).**
- Para el bbox del patch, construir el transform con
  `rasterio.transform.from_bounds(*bbox, 128, 128)`.
- Rasterizar los poligonos de las parcelas que caen en el bbox:
  `rasterio.features.rasterize([(geom, class_id), ...], out_shape=(128,128),
  transform=transform, fill=0)`. fill=0 = fondo.
- GOTCHA verificado: las geometrias son `MultiPolygon` (no `Polygon`) -> al
  dibujar/iterar usar `.geoms`. La rasterizacion en si acepta MultiPolygon
  directamente.
- Resultado: mascara `(128, 128) int32` con la clase de cada pixel (= formato
  `ParcelIDs/TARGET` de PASTIS).

**Paso 5 — Persistir el dataset en el layout de PASTIS.**
- Guardar por patch: `S2_<id>.npy` `(T,10,128,128)` + `TARGET_<id>.npy`
  `(128,128)` + un `dates_<id>.npy` con las fechas (DOY) de la serie (TSViT usa
  DOY). Estructura espejo de `data/PASTIS-R/`.
- Un `metadata.parquet` con: patch_id, bbox, n_parcelas, n_fechas, clases
  presentes, % cubierto, fold espacial (para CV sin fuga).
- Versionar con **DVC** (datos pesados, nunca al git). MLflow tags
  `data_version` + `code_version`.

**Paso 6 — Validar el dataset generado.**
- Notebook de EDA del homologo: distribucion de clases por pixel/parcela,
  ejemplos de patch RGB + mascara superpuesta, comparativa de textura vs PASTIS
  (`std` NDVI deberia ser ~0.2, como PASTIS, no ~0.05 del pixel-punto).
- Verificar cobertura: % de pixeles con clase != fondo por patch (objetivo >70%).
- Verificar la serie temporal: n fechas, distribucion temporal, % nubes residual.

### Modulos a crear (en `ml/data/` o `ml/ingest/`)

- `ml/data/eurocrops_pastis_builder.py`: orquesta paso 1-5. Funciones puras y
  testeables: `load_labeled_polygons(region)`, `select_dense_patches(gdf, ...)`,
  `download_patch_series(bbox, season, ...)`, `rasterize_patch_mask(gdf, bbox)`,
  `save_pastis_format(...)`.
- `scripts/build_italia_pastis.py`: runner con `--n-patches`, `--season`,
  `--min-support`, `--out`. Corre incremental (cache) y persiste por patch
  (resume si se corta).
- Tests en `tests/ml/data/`: mockear SH (sin red) y verificar shapes, el mapeo de
  clases y la rasterizacion sobre poligonos sinteticos PEQUENOS de prueba
  (NOTA: los tests SI pueden usar geometrias de juguete; la PROHIBICION de datos
  sinteticos aplica a los RESULTADOS del experimento, no a fixtures de test).

### Criterios de aceptacion (AC)

1. Se generan >=20 patches piloto en formato PASTIS exacto
   (`(T,10,128,128)` + `(128,128)`), versionados en DVC.
2. EDA del homologo ejecutado: distribucion de clases + ejemplos visuales + `std`
   NDVI ~0.2 (textura comparable a PASTIS), commiteado con outputs.
3. Cobertura media de pixeles-con-clase por patch >70%.
4. Mapeo `original_code`->HCAT documentado, 100% de las parcelas del subset
   mapeadas (o el resto agrupado en "other" de forma explicita).
5. Cero peticiones SH desperdiciadas (cache activo); cuota consumida reportada.
6. `make check` limpio; modulo con type hints + docstrings Google + structlog.

### Riesgos y mitigaciones

- **Cuota SH**: dimensionar N patches x T fechas contra la cuota; piloto chico
  primero; cache obligatorio.
- **Nubes**: si una temporada tiene pocas fechas limpias, ampliar la ventana o
  bajar el umbral a 30%; reportar el % de nubes residual honestamente.
- **Desbalance de clases**: olive domina (19.5%); estratificar la seleccion de
  patches para que las clases compartidas con PASTIS (las que importan al transfer)
  esten representadas.

---

## US-079 — Extender el modelo campeon al homologo italiano (transfer + Voting-3)

**Como agente / cientifico de datos, quiero** aplicar y extender el modelo
campeon (modelos densos PASTIS + combinador Voting ponderado) al dataset homologo
de Italia, **para** medir el transfer real Francia->Italia y demostrar que el
ensamble ligero (Voting-3) mantiene F1-macro > 0.9 sobre las clases bien
resueltas en el dominio nuevo.

### Alcance

- Usar el dataset US-078 (homologo PASTIS de Italia).
- Dos miembros densos del campeon: **TSViT-pheno** (el mejor individual, F1 0.737
  en PASTIS) y **U-TAE**, en su formato nativo de segmentacion.
- Combinador **Voting ponderado** (3 pesos aprendibles), NO meta-LogReg.
- Eval jerarquica (la "papaya/fruits"): F1 fino (clases italianas) + colapsado al
  nivel comun con PASTIS.

### Flujo detallado

**Paso 1 — Espacio de etiquetas del transfer.**
- Definir el label-space italiano: clases compartidas con PASTIS (grapevine,
  meadow, soft/durum wheat, maize, sunflower, barley) + clases nuevas
  mediterraneas (olive, ...). Mapear ambos al HCAT comun (crosswalk).
- Marcar las clases CONSERVADAS (mapean a PASTIS-18, warm-start desde la cabeza
  PASTIS, la "bandera" ya implementada en `ml/transfer/finetune_baltico.py`) vs
  las NUEVAS (cabeza nueva, init random).

**Paso 2 — Transfer de cada miembro denso (fine-tune, no zero-shot).**
- Cargar el checkpoint PASTIS del backbone (TSViT-pheno-fullm, U-TAE) como init.
- Reemplazar la cabeza por una del tamano del label-space italiano; warm-start de
  las clases conservadas (kept-class flag).
- Fine-tune sobre los patches italianos con split ESPACIAL (sin fuga; usar los
  folds del metadata US-078). Reusar el patron de `train_segmentation` y
  `finetune_baltico` (ya existe el builder + warm-start).
- Producir, por patch de TEST, la prediccion densa `(K, 128, 128)` de cada miembro.

**Paso 3 — Combinar con el Voting ponderado.**
- Sobre las predicciones post-softmax de TSViT + U-TAE (+ opcionalmente
  xgb-alphaearth si se materializa el embedding por parcela), aprender los pesos
  del Voting que maximizan F1-macro en un split de validacion italiano
  (`ml/ensemble/voting_weighted.py`, ya construido en la otra sesion).
- Reusar la infra de Voting (validacion anti-fuga, post-softmax).

**Paso 4 — Evaluacion.**
- Metricas densas (mIoU + F1-macro por pixel) sobre el TEST italiano, al nivel
  FINO (clases italianas) y COARSE (colapsado al HCAT comun con PASTIS).
- Curva de descarte honesto (como en EPIC 6): F1-macro vs numero de clases
  retenidas, para localizar el subconjunto de ~10 clases con F1 > 0.9.
- Comparar: (a) zero-shot del campeon frances (cota inferior), (b) Voting-3
  fine-tuneado, (c) cada miembro solo. Reportar el delta del transfer.

**Paso 5 — Demo cualitativa de granularidad (papaya/fruits).**
- Mostrar parcelas donde PASTIS solo diria una clase generica y el modelo
  extendido dice la clase fina italiana (p.ej. olive, que PASTIS no tiene).

### Criterios de aceptacion (AC)

1. TSViT-pheno y U-TAE fine-tuneados al label-space italiano desde el checkpoint
   PASTIS (warm-start de clases conservadas verificado), con split espacial.
2. Voting ponderado (3 pesos) entrenado sobre las predicciones italianas;
   reportado con sus pesos aprendidos (interpretabilidad).
3. **F1-macro > 0.9 sobre las ~10 clases mejor resueltas** del homologo italiano
   (objetivo de calidad, espejo del 0.92 france-9 del campeon).
4. Eval jerarquica fino vs coarse reportada (mIoU + F1); delta del transfer
   (fine-tune vs zero-shot) cuantificado.
5. Notebook de evaluacion ejecutada con outputs (matriz de confusion, per-clase,
   ejemplos de prediccion densa, demo de granularidad).
6. MLflow con `data_version` + `code_version`; `make check` limpio.

### Riesgos y mitigaciones

- **Transfer dificil (mediterraneo != Francia)**: por eso se fine-tunea (no
  zero-shot); si una clase nueva (olive) tiene poco soporte, reportar honestamente
  su F1 con el support.
- **Olive no existe en PASTIS**: es clase NUEVA -> cabeza nueva, sin warm-start;
  su F1 mide cuanto aprende el backbone frances de una clase que nunca vio (caso
  ideal de la hipotesis de enriquecimiento).
- **Voting vs Stacking**: si en el dominio italiano el Stacking superara al
  Voting, reportarlo; la decision Voting-3 se toma por su mejor F1 spatial-CV en
  PASTIS, pero se mide en Italia.

---

## Dependencias y orden

- US-078 **antes** que US-079 (079 consume el dataset de 078).
- Ambas dependen de: poligonos Italia (descargado), crosswalk (descargado),
  cliente SH con cache (commiteado), builder de patch + warm-start (existente),
  Voting ponderado (existente en la otra sesion).
- Cuota SH compartida: US-078 es la que consume peticiones (descarga); US-079 no
  baja nada nuevo (entrena sobre lo de 078).

## Resumen de artefactos REALES ya disponibles (no inventados)

| Artefacto | Ruta | Estado |
|-----------|------|--------|
| Poligonos Italia 2018 | `data/reference/eurocrops_v2/iti1_2018.parquet` | descargado (229 MB, 643k parcelas) |
| Crosswalk codigo->HCAT | `data/reference/eurocrops_v2/eurocrops_mapping.csv` | descargado (mapea 100% IT) |
| Nombres HCAT4 | `data/reference/eurocrops_v2/hcat4.csv` | descargado |
| Cliente SH + cache | `ml/ingest/sh_client.py` | commiteado (042fc60, ee0a80d) |
| Cliente catalogo CDSE | `ml/ingest/cdse_client.py` | commiteado (ec1c526) |
| Builder patch + warm-start | `ml/transfer/finetune_baltico.py` | commiteado |
| Voting ponderado | `ml/ensemble/voting_weighted.py` | otra sesion (sin commit aun) |
| PASTIS formato referencia | `data/PASTIS-R/` | local |
| Validacion visual del flujo | `reports/eurocrops_viz/` | 7 imagenes generadas |
