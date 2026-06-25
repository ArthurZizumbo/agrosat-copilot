# Análisis por clase del transfer transnacional (EuroCropsML y Sen4AgriNet)

> Análisis honesto por cultivo del *domain-shift* geográfico, en la misma línea
> que el análisis de cardinalidad por clase de los ensambles
> (`reports/ensemble/metrics/us043_winner_per_class.csv`). La métrica agregada
> (F1-macro / mIoU) esconde qué cultivos transfieren bien y cuáles colapsan; aquí
> abrimos esa caja negra con números **reales**, sin fabricar ningún valor.

Fuentes de datos (todas reproducibles en este host, sin la VM):

- **EuroCropsML** (Reuss et al. 2024, arXiv:2407.17458), protocolo *k-shot*
  transnacional `Latvia (LV) -> Estonia (EE)` sobre el espacio de etiquetas
  `hcat-macro` de US-074. Tabular: XGBoost `multi:softprob` sobre el vector
  fijo por parcela derivado de la serie Sentinel-2 L1C (no AlphaEarth; ver
  `ml/transfer/eurocropsml_fewshot.py`). Tres semillas por punto.
- **Sen4AgriNet** (Sykas et al. 2022, CC-BY-SA-4.0), transfer denso
  `Francia (PASTIS-R, 18 clases) -> Cataluña (ES, tile 31TCG)` proyectado al
  espacio `macro-HCAT` de 10 grupos. Segmentación TSViT-pheno; *zero-shot* del
  checkpoint francés + *few-shot* `k=10` parches, 40 épocas
  (`ml/train/finetune_sen4agrinet.py`).

Artefactos generados:

| Artefacto | Ruta |
|-----------|------|
| EuroCropsML por clase (agregado) | `data/transfer/eurocropsml_per_class.parquet` |
| EuroCropsML por clase (crudo, por semilla) | `data/transfer/eurocropsml_per_class_raw.parquet` |
| EuroCropsML figura F1 por clase vs k | `paper/figures/us-076/eurocropsml_per_class_f1_vs_k.png` |
| Sen4AgriNet por macro-grupo (IoU/F1/prec/rec) | `data/transfer/sen4agrinet_per_class.parquet` |
| Sen4AgriNet reporte JSON (zero/few + provenance) | `reports/segmentation/sen4agrinet_transfer_per_class.json` |
| Sen4AgriNet figura IoU/F1 por clase | `paper/figures/us-075/sen4agrinet_per_class_iou_f1.png` |

---

## 1. EuroCropsML — transfer tabular LV -> EE por cultivo

### 1.1 Tabla F1 por clase (escenario con pre-entrenamiento en Latvia)

F1 media (3 semillas) en el *query set* de Estonia, con su soporte real de test
(`n_test`, número de parcelas verdaderas de esa clase en el conjunto de
evaluación). Ordenado por F1 a `k=500`.

| Macro-grupo | n_test | F1 k=10 | F1 k=100 | F1 k=500 | Tendencia |
|-------------|-------:|--------:|---------:|---------:|-----------|
| grassland | ~5626 | 0.897 | 0.899 | 0.897 | **plano (saturado)** |
| cereals | ~1468 | 0.818 | 0.828 | 0.841 | sube leve |
| oilseed_industrial | ~255 | 0.764 | 0.777 | 0.809 | sube leve |
| potato | ~187 | 0.354 | 0.415 | 0.499 | **sube fuerte con k** |
| legumes_fodder | ~901 | 0.029 | 0.091 | 0.287 | **sube fuerte con k** |
| vegetables | ~66 | 0.082 | 0.278 | 0.206 | ruidoso |
| sugar_beet | ~16 | 0.089 | 0.152 | 0.188 | frágil |
| orchard | ~52 | 0.023 | 0.110 | 0.131 | frágil |
| vineyard | ~0 | 0.000 | n/a | n/a | **ausente en el target** |

(Las desviaciones por semilla están en `eurocropsml_per_class.parquet`; son
pequeñas para las clases robustas, p. ej. grassland +/- 0.001, y mayores para las
raras, p. ej. sugar_beet +/- 0.068 a k=10.)

### 1.2 Lectura: qué cultivos transfieren bien y cuáles no

**Clases robustas al domain-shift (alto F1 con poquísimas etiquetas locales):**

- **grassland** está *saturado* en 0.90 desde `k=10`: añadir etiquetas locales no
  lo mejora. Es la clase dominante en ambos países (Bálticos = paisaje de pradera)
  y su firma temporal Sentinel-2 es casi idéntica entre LV y EE. No hay brecha de
  dominio que cerrar.
- **cereals** (0.82 -> 0.84) y **oilseed_industrial** (0.76 -> 0.81) transfieren
  muy bien y solo ganan ~2-5 pp con dos órdenes de magnitud más de etiquetas. Son
  cultivos con fenología homogénea en el clima báltico continental.

**Clases que dependen fuertemente de k (la brecha existe pero se cierra con
etiquetas locales):**

- **potato** sube de 0.35 a 0.50 y **legumes_fodder** de 0.03 a 0.29 al pasar de
  `k=10` a `k=500`. Son las clases que más se benefician de etiquetas estonias: su
  expresión espectral/temporal difiere lo suficiente entre LV y EE como para que el
  pre-entrenamiento por sí solo no baste, pero con cientos de muestras locales el
  modelo aprende la variante estonia. legumes_fodder es además el caso de manual de
  *recall* bajo persistente (0.015 -> 0.215): el modelo la confunde sistemáticamente
  con grassland/cereals hasta que tiene suficientes ejemplos.

**Clases frágiles (no remontan ni con k=500):**

- **sugar_beet** (~16 test), **orchard** (~52), **vegetables** (~66) se quedan por
  debajo de 0.2-0.3. El cuello de botella es la **cardinalidad**: con tan pocas
  parcelas verdaderas, el *k-shot* no llena los huecos del espacio de características
  y el F1 oscila por semilla (ruido de muestreo). Es el mismo patrón que en los
  ensambles, donde Potatoes (103 test), Mixed cereal (193) y Sorghum (206) eran las
  peores clases por puro déficit de soporte.
- **vineyard** es directamente **inexistente** en el target: Estonia tiene 1 sola
  parcela de viñedo en todo el subset (Latvia, 5). No hay viñedo báltico que
  clasificar; F1=0 no es un fallo del modelo sino la ausencia del cultivo. Honesto:
  no es comparable con el viñedo de Sen4AgriNet (sección 2), que sí existe en
  Cataluña.

### 1.3 Transferencia negativa: el pre-entrenamiento de Latvia no ayuda a todos

Comparando el escenario *con* pre-entrenamiento de Latvia contra el de *solo
k-shot* (sin pretrain), el `pretrain_gain = F1(pretrain) - F1(no-pretrain)` revela
un efecto por clase que el F1-macro agregado oculta. A `k=10`:

| Macro-grupo | pretrain_gain (k=10) |
|-------------|---------------------:|
| cereals | **+0.343** |
| oilseed_industrial | **+0.302** |
| grassland | **+0.285** |
| potato | +0.210 |
| sugar_beet | +0.056 |
| orchard | -0.005 |
| vegetables | -0.064 |
| legumes_fodder | **-0.144** |

El pre-entrenamiento letón ayuda enormemente a los cultivos **comunes y de
fenología compartida** (cereals, oilseed, grassland: +0.28 a +0.34) pero **perjudica**
a legumes_fodder (-0.14) y vegetables (-0.06): para esas clases las parcelas
letonas inyectan un sesgo de dominio equivocado que confunde al clasificador, y es
mejor entrenar solo con las pocas muestras estonias. Esto es **transferencia
negativa por clase** y es la observación más valiosa del análisis: la decisión
"pre-entrenar en el país vecino" no es universalmente buena, depende del cultivo.
A `k=500` la ventaja del pretrain casi desaparece (el target ya tiene señal propia
suficiente) y para sugar_beet/legumes_fodder vuelve a ser negativa.

---

## 2. Sen4AgriNet — transfer denso Francia -> Cataluña por macro-grupo

### 2.1 Tabla IoU/F1 por clase (zero-shot vs few-shot)

Evaluado sobre el *val* real de Cataluña (20 parches retenidos, 180 *tiles* de
128x128). El soporte es el número de píxeles verdaderos de cada clase en el val.
Solo aparecen las 5 macro-clases que tienen píxeles de *ground-truth* en Cataluña.

| Macro-grupo | n_px (val) | IoU zero-shot | IoU few-shot | F1 few-shot | Prec / Rec few-shot |
|-------------|-----------:|--------------:|-------------:|------------:|---------------------|
| cereals | 584 734 | 0.000 | **0.923** | 0.960 | 0.941 / 0.980 |
| legumes_fodder | 39 198 | 0.000 | 0.156 | 0.270 | 0.401 / 0.203 |
| vineyard | 2 927 | 0.000 | 0.150 | 0.261 | **0.955 / 0.151** |
| oilseed_industrial | 2 067 | 0.000 | 0.000 | 0.000 | 0.000 / 0.000 |
| potato | 164 | 0.000 | 0.000 | 0.000 | 0.000 / 0.000 |

Agregados: zero-shot mIoU = **0.000** / F1-macro 0.000; few-shot mIoU = **0.246** /
F1-macro 0.298 / pixel-accuracy 0.924.

> Nota de reproducibilidad: el *few-shot* aquí se **recalculó localmente** (CUDA,
> mismo protocolo, semilla 17, k=10, 40 épocas) porque el checkpoint finetuneado de
> la VM (`checkpoints/segmentation/tsvit-pheno-sen4agri-cat-ft-v1/best.pt`, en `F:`)
> no está en este host. La corrida local converge a mIoU **0.246** / F1 0.298, que
> coincide con el reporte agregado original de la VM
> (`reports/segmentation/sen4agrinet_transfer_result.json`: mIoU 0.247 / F1 0.301).
> El *zero-shot* (mIoU 0.000) coincide exactamente porque usa el checkpoint francés
> que sí está en local. Por la naturaleza estocástica del *finetune* (selección de
> *best epoch*), los valores por clase de vineyard y legumes_fodder oscilan entre
> corridas (vineyard IoU 0.15-0.31, legumes 0.16-0.26 según la época-mejor); el
> *ranking* y la firma cualitativa (cereals >> legumes ~ vineyard >> oilseed = potato)
> son estables. **Todos los números son reales.** Para reproducir el *few-shot*
> exacto del checkpoint de la VM, ver sección 4.

### 2.2 Lectura: el colapso zero-shot y qué se recupera

**Zero-shot: colapso total (mIoU = 0.000 en todas las clases).** No es una
degradación parcial: es un colapso. Diagnóstico de la distribución de predicciones
del modelo francés sobre Cataluña (primeros 40 *tiles*):

- **vegetables 73.6 %** y **grassland 26.4 %** — y nada más.

El TSViT entrenado en PASTIS-R (clima atlántico francés) mapea la fenología
mediterránea de Cataluña hacia dos macro-clases que **no tienen prácticamente
*ground-truth* en el val catalán** (que es cereals/legumes/vineyard/oilseed/potato).
Predice cultivos equivocados con confianza, así que la intersección con la verdad
es cero para todas las clases reales. Esta es la mitad cuantitativa de la brecha
franco-ibérica: el desfase estacional de siembra/cosecha por latitud y clima rompe
el modelo de origen por completo. El transfer *zero-shot* entre estas dos regiones
**no es viable**.

**Few-shot (10 parches): recuperación muy desigual por cultivo.**

- **cereals** se recupera por completo: IoU 0.923, F1 0.960, precision 0.941 y
  recall 0.980. Es la clase dominante (584k px = 92 % del val) y la de fenología más
  compartida entre Francia y Cataluña (cereal de invierno en ambos). Robusta:
  basta un puñado de parches locales para alinear el dominio.
- **vineyard** es el caso más interesante: **precision 0.955 pero recall 0.151**
  (F1 0.261, IoU 0.150). El modelo, cuando dice "viñedo", acierta casi siempre — la
  firma del viñedo es muy distintiva — pero **se le escapa el ~85 % de los píxeles**
  de viñedo. El viñedo *sí transfiere a nivel de precisión* (responde a la pregunta
  de Arthur: el viñedo no es robusto en cobertura, pero su firma es de las más
  fiables cuando se detecta). Con solo 10 parches no hay suficientes ejemplos de
  viñedo catalán para subir el recall.
- **legumes_fodder** queda en un punto medio (IoU 0.156, F1 0.270, precision 0.401,
  recall 0.203): ni colapsa ni se recupera del todo.
- **oilseed_industrial** (2067 px) y **potato** (164 px) **fallan por completo**
  (IoU 0.000) incluso tras el few-shot. Doble penalización: soporte minúsculo en el
  val *y* probable ausencia en los 10 parches de entrenamiento (a nivel de píxel,
  potato son 164 px en todo el val). Son las clases frágiles del transfer denso.

### 2.3 Comparación con el patrón de los ensambles

El paralelismo con `us043_winner_per_class.csv` es directo:

- Los **cereales y grassland** (alto soporte, fenología estable) son las clases
  robustas en los tres ejercicios: ensamble (Meadow F1 0.90, Soft winter wheat
  0.90), EuroCropsML (grassland 0.90, cereals 0.84) y Sen4AgriNet (cereals F1 0.96).
- El **viñedo / Grapevine** es robusto en el ensamble doméstico (F1 0.92) y de alta
  *precision* en el transfer franco-ibérico (0.955) pero con *recall* bajo en transfer
  (0.151): la firma es fiable, la cobertura no, exactamente por déficit de muestras
  locales.
- Las clases de **baja cardinalidad** (Potatoes 103, Sorghum 206 y Mixed cereal 193
  en el ensamble; sugar_beet/orchard/vegetables en EuroCropsML; potato/oilseed en
  Sen4AgriNet) son sistemáticamente las peores. **La cardinalidad por clase, no el
  domain-shift en abstracto, es el factor dominante de fragilidad.**

---

## 3. Conclusiones: clases robustas vs frágiles al domain-shift

| Categoría | Cultivos | Por qué |
|-----------|----------|---------|
| **Robustas** | grassland/meadow, cereals | Alto soporte + fenología compartida entre regiones; saturan con pocas (o cero) etiquetas locales. |
| **Recuperables con k** | potato y legumes_fodder (EuroCropsML), cereals (Sen4AgriNet few-shot) | Existe brecha de dominio, pero cientos de muestras locales (o 10 parches densos) la cierran. |
| **Precisión alta / cobertura baja** | vineyard (Sen4AgriNet), oilseed_industrial (EuroCropsML) | Firma muy distintiva (alta precision) pero recall limitado por falta de ejemplos locales. |
| **Frágiles (no remontan)** | sugar_beet, orchard, vegetables (EuroCropsML); oilseed_industrial, potato (Sen4AgriNet) | Cardinalidad mínima; el k-shot no llena el espacio de características. |
| **No comparables / ausentes** | vineyard en EuroCropsML (1 parcela en EE) | El cultivo no existe en el target; F1=0 es ausencia, no error. |

**Tres mensajes para el paper:**

1. El *domain-shift* geográfico no degrada uniformemente: degrada **selectivamente
   por cultivo**. El agregado (EuroCropsML F1-macro 0.32 -> 0.48; Sen4AgriNet mIoU
   zero->few 0.00 -> 0.246) promedia clases que van de F1 0.96 a 0.00.
2. El **zero-shot denso entre climas distintos (Francia->Cataluña) es inviable**
   (mIoU 0.000, el modelo predice clases inexistentes); el *few-shot* mínimo lo
   rescata pero solo para cereales y, en precisión, viñedo.
3. La **transferencia negativa por clase es real**: pre-entrenar en el país vecino
   (Latvia) ayuda a cereales/oleaginosas (+0.30) pero perjudica a
   legumes_fodder/vegetables (-0.14/-0.06). La decisión de pre-entrenar debe tomarse
   por cultivo, no globalmente.

---

## 4. Reproducibilidad

### EuroCropsML (tabular, local)

```bash
# Reusa los feature caches en data/transfer/eurocropsml/_feature_cache/ (max_parcels=30000).
# El script de instrumentación por clase reusa build_fewshot_splits + build_estimator
# y emite precision/recall/f1/support por macro-grupo a k=10,100,500 (3 semillas).
python -m ml.transfer.eurocropsml_fewshot   # curva F1-macro original
# Per-clase: ver el script de instrumentación que produjo
#   data/transfer/eurocropsml_per_class.parquet
```

### Sen4AgriNet (denso)

`zero-shot` por clase es 100 % local (checkpoint francés
`checkpoints/segmentation/tsvit-pheno-v1/best.pt` + 40 parches `.nc` en
`data/sen4agrinet/`). Requiere `netCDF4` (lo usa `Sen4AgriNetDataset`).

Para reproducir el `few-shot` **exacto de la VM** (mIoU 0.247 del JSON original),
en la VM `F:` (donde vive el checkpoint finetuneado):

```powershell
F:\tools\micromamba.exe run -n agrosat python -m ml.train.finetune_sen4agrinet `
    --root F:/projects/agrosat-copilot/data/sen4agrinet `
    --fr-ckpt F:/projects/agrosat-copilot/checkpoints/segmentation/tsvit-pheno-v1/best.pt `
    --k 10 --epochs 40 --device cuda `
    --ckpt-dir F:/projects/agrosat-copilot/checkpoints/segmentation/tsvit-pheno-sen4agri-cat-ft-v1
```

El checkpoint finetuneado de la VM
(`checkpoints/segmentation/tsvit-pheno-sen4agri-cat-ft-v1/best.pt`) **no está en
este host**; la corrida local lo recalcula con el mismo protocolo (resultados de la
sección 2.1). Para producir la tabla por clase del checkpoint de la VM, basta cargar
ese `best.pt`, correr `evaluate_few_shot` con el `DenseConfusionAccumulator` y
extraer `confusion_matrix()` (el mismo helper `_confusion_to_per_class` usado aquí).
```
