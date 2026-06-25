# Modelo multi-region de "clases rescatadas" (EPIC 12)

> Idea de Arthur. Un unico clasificador entrenado sobre el embedding AlphaEarth de
> 64 dimensiones agrupando TODAS las regiones de transfer (PASTIS Francia,
> EuroCropsML Estonia + Letonia, Sen4AgriNet Cataluna/Francia, WorldCereal
> Brasil/India). La hipotesis: una clase DEBIL en una region por falta de
> muestras puede ser FUERTE en otra que abunda en ella, y al combinar todas se
> obtiene una taxonomia mas fina que cualquier dataset por separado (por ejemplo
> "cebada de primavera" donde PASTIS solo dice "cereal").

Codigo: [`ml/transfer/multiregion_model.py`](../../ml/transfer/multiregion_model.py).
Resultados: `data/transfer/multiregion_*.parquet`, `data/transfer/multiregion_summary.json`.
Figuras: [`figures/`](figures/).

Reproducir:

```python
from pathlib import Path
from ml.transfer.multiregion_model import (
    run_multiregion_experiment, _save_outputs, make_figures,
)
r = run_multiregion_experiment()
_save_outputs(r, Path("data/transfer"))
make_figures(r, Path("docs/transfer/figures"))
```

Todos los numeros de este documento provienen de esa corrida (semilla 42, XGBoost
`multi:softprob`, la receta campeona de `ml/train/baseline.py`). No hay ninguna
cifra fabricada: si un parquet faltara, el constructor lanza un error en vez de
inventar filas.

---

## 1. Armonizacion al espacio HCAT comun

Cada parcela queda con su embedding AlphaEarth de 64 dim mas DOS niveles de
etiqueta, usando el crosswalk real de US-074
([`ml/data/hcat_crosswalk.py`](../../ml/data/hcat_crosswalk.py)) y la referencia
HCAT v3 (`data/reference/eurocrops_hcat3.csv`, 384 nodos):

- nivel **HOJA** fino (`hcat_leaf_name`), y
- nivel **MACRO** grueso (`macro_hcat_group`).

Que aporta cada region a la etiqueta FINA es asimetrico y se respeta tal cual:

| Region | Parcelas (pool) | Etiqueta nativa | Aporta hoja fina? |
|---|---|---|---|
| PASTIS Francia | 6 000 (cap) | `class_id` semantic18 -> hoja HCAT via crosswalk | Si (18 hojas) |
| EuroCropsML Estonia | 6 000 (cap) | `hcat_code` 10-digitos -> nombre HCAT | Si (hojas finas extra) |
| EuroCropsML Letonia | 6 000 (cap) | `hcat_code` 10-digitos -> nombre HCAT | Si (hojas finas extra) |
| Sen4AgriNet ES (Cataluna) | 6 000 (cap) | solo macro (10 grupos) | No, solo macro |
| Sen4AgriNet FR | 3 510 | solo macro | No, solo macro |
| WorldCereal Brasil Cerrado | 1 554 | maize/wintercereals/other/non_crop | No, solo macro |
| WorldCereal India Karnataka | 1 425 | idem tropical | No, solo macro |

Solo PASTIS y EuroCropsML traen una HOJA HCAT real; Sen4AgriNet y WorldCereal solo
resuelven un macro/grupo. **No se inventa una hoja fina para un dataset que no la
etiqueta.** Las hojas con menos de 50 parcelas (cola larga de EuroCropsML:
lavanda=2, lupulo=1, ...) se descartan del cabezal fino por ser imposibles de
aprender o evaluar con honestidad; siguen aportando su macro.

**Taxonomia resultante** (el punto central de la idea): tras unir las regiones, el
espacio de hojas finas aprendibles pasa de **18** (solo PASTIS) a **30** hojas
HCAT distintas. Las 12-14 hojas extra vienen de EuroCropsML y son justamente las
que PASTIS colapsa:

- `spring_barley` vs `winter_barley` (PASTIS solo distingue una cebada),
- `oats`, `rye`, `spring_common_soft_wheat` (PASTIS los funde en "cereal"),
- `spring_rapeseed_rape` / `summer_rapeseed_rape` (PASTIS solo "winter rapeseed"),
- `apples`, `quinces` (PASTIS solo "orchard"),
- `clover`, `alfalfa_lucerne` (PASTIS solo "leguminous fodder"),
- `carrots_daucus`, `beetroot_beets`, `finola`, `phacelia`, etc.

Provenance completa por hoja en `data/transfer/multiregion_fine_leaf_provenance.parquet`.
El embedding AlphaEarth es el MISMO espacio de 64 dim en todas las regiones, asi
que la union es representacionalmente valida aunque los espacios de etiqueta
difieran.

---

## 2. Entrenamiento (few-shot por region, no zero-shot)

El hallazgo del transfer es vinculante: el zero-shot Europa->tropico FALLA (maiz
Brasil F1 ~ 0.0095, la frontera de decision no se traslada) y solo se recupera
con few-shot (Brasil k=20 F1 ~ 0.626). Por eso el modelo multi-region se entrena
**con un slice de cada region** y se evalua sobre un test held-out estratificado
por region (sin fuga de parcelas entre train y test). No es una extrapolacion
zero-shot.

Decision metodologica clave: el **cabezal fino se entrena solo con las parcelas
que tienen hoja fina real** (PASTIS + EuroCropsML, 17 516 parcelas). Las regiones
macro-only (Sen4AgriNet, WorldCereal) NO inyectan una pseudo-clase gruesa
`cereals__macro` en el cabezal fino, porque colisionaria con las hojas finas del
mismo macro (`oats`, `spring_barley`...) y las hundiria artificialmente. Esas
regiones se reservan para la evaluacion MACRO colapsada. Comprobamos
empiricamente que sin esta separacion las hojas finas caian ~0.10-0.15 de F1 por
esa fuga de niveles.

---

## 3. Medicion A — jerarquica medible (la metrica honesta)

NO reportamos el F1-macro sobre N clases (baja mecanicamente al crecer N: PASTIS
9 clases = 0.91 vs 18 = 0.749). Contamos **cuantas clases-hoja individuales
cruzan F1 >= umbral**, y comparamos contra el baseline solo-PASTIS. El conteo es
un RESULTADO, no un objetivo. Figura: [`figures/multiregion_k_over_threshold.png`](figures/multiregion_k_over_threshold.png).

### Conteo de clases sobre umbral (nivel HOJA)

| Modelo | Clases totales | F1>=0.70 | F1>=0.80 | F1>=0.85 |
|---|---|---|---|---|
| **Multi-region** | 30 hojas | **7** | 2 | **2** |
| Solo-PASTIS (capped 6k, mismo presupuesto) | 18 | 6 | 3 | 2 |
| Solo-PASTIS (full 86k, referencia) | 18 | 10 | 4 | 2 |

Las dos clases que cruzan 0.85 en multi-region son `winter_rapeseed_rape`
(F1=0.948, region dominante EuroCropsML Estonia) y
`vineyards_wine_vine_rebland_grapes` (F1=0.905, region dominante PASTIS, reforzada
por Cataluna). Tabla por clase en `data/transfer/multiregion_per_class_leaf.parquet`,
figura [`figures/multiregion_leaf_f1.png`](figures/multiregion_leaf_f1.png).

### Clases rescatadas

Definimos "rescatada" como una hoja fina que cruza F1>=0.85 en multi-region y que
NO existe como clase aprendible en el espacio solo-PASTIS. El resultado es
**0 clases rescatadas a umbral 0.85**: las dos clases buenas del multi-region
(`winter_rapeseed_rape`, `vineyard`) ya estaban en PASTIS. A umbrales mas
permisivos tampoco aparece una hoja EXCLUSIVA de EuroCropsML por encima de 0.85;
la mejor hoja fina genuinamente nueva es `apples` (F1=0.686, solo en
EuroCropsML), seguida de `spring_rapeseed_rape` (0.719) y la separacion
`spring_barley` (0.432) vs `winter_barley` (0.393), que el modelo intenta pero no
resuelve con calidad de produccion.

### Nivel MACRO colapsado (no degradar el nivel grueso)

La prediccion fina se colapsa a su macro HCAT (papaya->fruits seria un acierto al
nivel medible). Sobre las 10 macros europeas el modelo multi-region da **F1-macro
= 0.658**, en linea con el baseline tabular del proyecto (0.6535, finding v8):
**el multi-region NO degrada el nivel grueso europeo** pese a predecir 30 hojas en
vez de 18. Cuando se incluyen las dos macros tropicales (`non_crop`,
`other_cropland`) el F1-macro cae a 0.548, porque el cabezal europeo no puede
predecir clases tropicales (F1=0.0 en ambas): no tiene una hoja que colapse a
ellas. Figura [`figures/multiregion_leaf_vs_macro_f1.png`](figures/multiregion_leaf_vs_macro_f1.png).

---

## 4. Medicion B — demo cualitativa (estilo Mexico)

Parcelas reales de Mexico (aguacate Uruapan, guayaba Calvillo) con la etiqueta
fina predicha y su confianza. SIN claim de accuracy (no hay ground truth fino
verificable). Tabla en `data/transfer/multiregion_mexico_demo.parquet`.

| AOI | Cultivo real | Pred. top-1 | Confianza | Top-2 | Top-3 |
|---|---|---|---|---|---|
| aguacate_uruapan | aguacate | pasture_meadow_grassland_grass | 0.632 | winter_rapeseed_rape (0.080) | potatoes (0.075) |
| guayaba_calvillo | guayaba | pasture_meadow_grassland_grass | 0.759 | fresh_vegetables (0.081) | potatoes (0.055) |

Lectura honesta: el modelo confunde los huertos perennes tropicales
(aguacate/guayaba) con pradera/pastizal, con confianza media-alta. **No tiene una
clase de huerto tropical y no la inventa.** Esto confirma cualitativamente el
hallazgo del transfer: las clases tropicales no mapean al espacio europeo y un
modelo europeo, por muy multi-region que sea dentro de Europa, no las resuelve sin
few-shot local de esas clases (que aqui no existe: solo hay 2 parcelas Mexico, sin
GT fino).

---

## 5. Veredicto honesto

**La idea de Arthur funciona PARCIALMENTE, en su parte taxonomica, pero NO en su
parte de rescate por umbral, con estos datos.**

Lo que SI se cumple:

1. **La taxonomia se amplia de verdad**: 18 -> 30 hojas finas reales, sin inventar
   mapeos. EuroCropsML aporta distinciones agronomicas que PASTIS no tiene
   (cebada primavera/invierno, avena, centeno, colza de primavera/verano,
   manzanas, trebol, alfalfa). El punto "mas taxonomia" es cierto.
2. **El nivel grueso no se degrada**: F1-macro europeo 0.658, igual que el
   baseline mono-region. Predecir 30 hojas no rompe la lectura de macro-grupo.
3. **La separabilidad existe para los cultivos "faciles"**: colza y vinedo cruzan
   0.85; maiz, trigo blando de invierno y pradera rondan 0.73-0.80.

Lo que NO se cumple:

4. **No hay rescate neto de clases buenas**: a umbral 0.85 el multi-region tiene
   2 clases, lo mismo que solo-PASTIS. A 0.70 el multi-region (7) supera al
   solo-PASTIS con el mismo presupuesto de datos (6), pero queda por debajo del
   solo-PASTIS full (10). Anadir 12 hojas finas nuevas NO añade 12 clases buenas:
   la mayoria de las hojas finas de EuroCropsML (avena, centeno, trebol, alfalfa,
   trigo de primavera, hortalizas finas) se quedan en F1 0.06-0.44 porque son
   **intrinsecamente poco separables en AlphaEarth anual**: distinguir cebada de
   primavera de cebada de invierno, o avena de trigo, exige fenologia intra-anual
   que el embedding anual de 64 dim promedia.
5. **Una clase debil en una region no se vuelve fuerte solo por agregar otra**: el
   mecanismo que Arthur intuye (mas muestras de la misma clase desde otra region)
   solo aplica cuando las regiones COMPARTEN esa hoja fina, y aqui el solapamiento
   fino real entre PASTIS y EuroCropsML es escaso (cereales y colza). Cataluna
   (vinedo) y los tropicales aportan al MACRO, no hojas finas nuevas verificables.
6. **El tropico sigue fuera**: la demo Mexico y el F1=0.0 de `non_crop`/
   `other_cropland` confirman que el cabezal europeo no cubre cultivos tropicales.
   Un modelo verdaderamente multi-region global necesitaria few-shot etiquetado en
   cada zona climatica, no solo agregar Europa.

### Conclusion operativa

El modelo multi-region es valioso como **vocabulario mas rico a nivel macro y como
generador de hipotesis finas** (la demo cualitativa da una etiqueta + confianza
util para un copiloto), pero **no como clasificador fino de produccion**: a umbral
de calidad 0.85 no rescata mas clases que PASTIS solo. La granularidad fina extra
es real en la TAXONOMIA pero no se traduce todavia en F1 alto para las hojas
nuevas. El cuello de botella no es la cantidad de regiones sino la
**separabilidad fenologica en el embedding anual**: el siguiente paso con mayor
expectativa de rescate seria features temporales intra-anuales (series Sentinel-2
multi-fecha) sobre las hojas finas de cereales/leguminosas, no mas regiones.
