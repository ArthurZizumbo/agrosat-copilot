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
La metrica honesta de rescate vive en `data/transfer/multiregion_paired_delta.parquet`
(funcion `evaluate_per_class_delta`). Figuras: [`figures/`](figures/), incluida
[`figures/multiregion_paired_delta.png`](figures/multiregion_paired_delta.png).

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

## 3. Medicion A — delta de F1 por clase (la metrica HONESTA)

> **Correccion metodologica (load-bearing).** La version previa de este documento
> media el exito con un **conteo binario** de "cuantas hojas cruzan F1>=0.85" y
> con una nocion de "clase rescatada" = hoja que cruza 0.85 *y* no existe en
> solo-PASTIS. Ese conteo daba **0 rescatadas** y **ocultaba el rescate real**:
> una mejora de +0.45 que aterriza en F1=0.65 nunca cruza la barrera de 0.85, asi
> que el conteo es ciego a ella. Peor aun, el baseline solo-PASTIS evaluaba las
> hojas raras en Francia (p. ej. `potatoes`, soporte de test ~9) sobre un slice
> de test minusculo y distinto del que ve el multi-region (~570), de modo que los
> dos F1 nunca se median sobre las MISMAS parcelas. La metrica correcta es el
> **delta de F1 por clase, entrenando dos modelos al MISMO presupuesto y
> evaluandolos sobre el MISMO test held-out** (no PASTIS-full 86k contra
> multi-region-reducido, que es injusto). Codigo:
> `evaluate_per_class_delta` en `ml/transfer/multiregion_model.py`; tabla en
> `data/transfer/multiregion_paired_delta.parquet`; figura
> [`figures/multiregion_paired_delta.png`](figures/multiregion_paired_delta.png).

**Protocolo justo.** Se entrenan DOS clasificadores XGBoost sobre el MISMO split
held-out por region (mismo presupuesto: el modelo solo-PASTIS simplemente carece
de las filas de Estonia/Letonia que nunca tuvo, NO es un PASTIS sub-muestreado) y
se puntuan AMBOS sobre las MISMAS parcelas de test de hoja fina:

- **multi-region** entrenado sobre PASTIS + EuroCropsML (parcelas `has_fine`);
- **solo-PASTIS** entrenado sobre el subconjunto PASTIS de ese mismo split.

Para una hoja que solo-PASTIS nunca vio (`apples`, `oats`, ...), su F1 sobre esas
parcelas es 0 por construccion: es el costo honesto de NO agrupar regiones, no un
artefacto. Esas hojas se reportan en un panel aparte (no se suman al rescate de
hojas compartidas).

### Panel 1 — Hojas COMPARTIDAS (mismo test, mismo presupuesto)

Delta = F1(multi-region) − F1(solo-PASTIS) sobre las MISMAS parcelas. Columna
`region_test` = de donde provienen la mayoria de las parcelas de test de esa hoja
(donde se mide el rescate).

| Hoja HCAT | F1 solo-PASTIS | F1 multi-region | Delta F1 | Soporte test | region_test dominante |
|---|---|---|---|---|---|
| potatoes | 0.007 | 0.650 | **+0.643** | 572 | EuroCropsML Estonia |
| spring_barley | 0.044 | 0.426 | **+0.381** | 127 | EuroCropsML Estonia |
| pasture_meadow_grassland_grass | 0.463 | 0.741 | **+0.278** | 1254 | PASTIS Francia |
| winter_common_soft_wheat | 0.492 | 0.766 | **+0.275** | 383 | PASTIS Francia |
| legumes_harvested_green | 0.082 | 0.350 | **+0.269** | 265 | EuroCropsML Estonia |
| fresh_vegetables | 0.137 | 0.283 | +0.146 | 264 | EuroCropsML Estonia |
| winter_rapeseed_rape | 0.805 | 0.945 | +0.140 | 463 | EuroCropsML Estonia |
| orchards_fruits | 0.435 | 0.521 | +0.085 | 142 | EuroCropsML Estonia |
| soy_soybeans | 0.655 | 0.689 | +0.033 | 34 | PASTIS Francia |
| grain_maize_corn_popcorn | 0.775 | 0.802 | +0.027 | 281 | PASTIS Francia |
| winter_barley | 0.355 | 0.381 | +0.026 | 77 | PASTIS Francia |
| vineyards_wine_vine_rebland_grapes | 0.886 | 0.901 | +0.015 | 239 | PASTIS Francia |
| winter_triticale | 0.208 | 0.213 | +0.004 | 35 | PASTIS Francia |
| sugar_beet | 0.718 | 0.703 | -0.015 | 20 | PASTIS Francia |
| winter_durum_hard_wheat | 0.595 | 0.578 | -0.017 | 37 | PASTIS Francia |
| sunflower | 0.600 | 0.489 | -0.111 | 30 | PASTIS Francia |

**Resumen del panel 1 (16 hojas compartidas):**

- **13 hojas MEJORAN** (delta > 0), suma de mejoras = **+2.323 F1**.
- **3 hojas EMPEORAN** (delta < 0), suma de caidas = **-0.143 F1**.
- **Rescate NETO (suma de todos los deltas compartidos) = +2.179 F1.**
- F1 medio por hoja compartida: **0.454 (solo-PASTIS) -> 0.590 (multi-region)**.

Estos numeros son **estables**: en tres semillas (42 / 7 / 123) los grandes
rescates se mantienen — `potatoes` +0.643 / +0.668 / +0.615, `spring_barley`
+0.381 / +0.374 / +0.375, `pasture` +0.278 / +0.289 / +0.265, `winter_common_soft_wheat`
+0.275 / +0.329 / +0.203, `winter_rapeseed_rape` +0.140 / +0.218 / +0.228.

**Por que mejoran unas y empeoran otras.** El mecanismo es el presupuesto de
datos por clase. Las hojas que mejoran son justamente las que Francia apenas
cultiva y que Estonia/Letonia abundan; las que empeoran son cultivos
PASTIS-nativos de soporte pequeno (girasol, trigo duro, remolacha) que se diluyen
levemente al ampliar el espacio de etiquetas. Reparto de parcelas de
ENTRENAMIENTO por region (split semilla 42):

| Hoja | train FR | train EE | train LV | % no-FR |
|---|---|---|---|---|
| potatoes | 21 | 722 | 630 | **98.5 %** |
| winter_rapeseed_rape | 91 | 511 | 493 | 91.7 % |
| spring_barley | 47 | 176 | 81 | 84.5 % |
| fresh_vegetables | 125 | 308 | 160 | 78.9 % |
| legumes_harvested_green | 157 | 461 | 0 | 74.6 % |
| winter_common_soft_wheat | 393 | 229 | 308 | 57.7 % |
| orchards_fruits | 134 | 171 | 0 | 56.1 % |
| pasture_meadow_grassland_grass | 1539 | 711 | 675 | 47.4 % |

`potatoes` es el caso de manual: Francia aporta **21** parcelas de
entrenamiento, Estonia+Letonia **1 352** (98.5 %). Solo-PASTIS no tiene con que
aprenderla (F1=0.007) y el multi-region la rescata a 0.650 con las muestras
balticas. **La idea de Arthur — "una clase debil en una region se vuelve fuerte
con muestras de otra que abunda en ella" — se cumple empiricamente** para
potatoes, spring_barley, rapeseed y leguminosas. Las caidas NO son de la idea
multi-region en si, sino de que esos cultivos siguen siendo casi puramente
franceses y el ensanchamiento de clases los diluye un poco.

### Panel 2 — Hojas de TAXONOMIA NUEVA (solo el pool las resuelve)

14 hojas finas que solo-PASTIS no puede etiquetar (no existen en su espacio). No
se suman al rescate compartido; se reportan con su F1 absoluto como ganancia de
GRANULARIDAD. **4 de 14 cruzan F1>=0.50**; 7 de 14 cruzan 0.30.

| Hoja HCAT (nueva) | F1 multi-region | Soporte test | region_test dominante |
|---|---|---|---|
| spring_rapeseed_rape | 0.717 | 50 | EuroCropsML Estonia |
| apples | 0.697 | 198 | EuroCropsML Letonia |
| summer_rapeseed_rape | 0.571 | 37 | EuroCropsML Letonia |
| finola | 0.541 | 23 | EuroCropsML Estonia |
| clover | 0.443 | 295 | EuroCropsML Letonia |
| spring_common_soft_wheat | 0.321 | 91 | EuroCropsML Letonia |
| oats | 0.307 | 108 | EuroCropsML Letonia |
| carrots_daucus | 0.182 | 17 | EuroCropsML Letonia |
| alfalfa_lucerne | 0.170 | 90 | EuroCropsML Letonia |
| quinces | 0.100 | 37 | EuroCropsML Letonia |
| beetroot_beets | 0.083 | 18 | EuroCropsML Letonia |
| rye | 0.056 | 31 | EuroCropsML Estonia |
| aromatic_medicinal_culinary_plants_spices_herbs | 0.000 | 13 | EuroCropsML Letonia |
| phacelia | 0.000 | 25 | EuroCropsML Letonia |

### Conteo binario sobre umbral (el que ENGANABA — se conserva por trazabilidad)

NO reportamos el F1-macro sobre N clases (baja mecanicamente al crecer N: PASTIS
9 clases = 0.91 vs 18 = 0.749). El conteo binario que sigue es el que se mostraba
antes como metrica principal; se mantiene solo como referencia historica, pero
**no es la metrica de exito**: es ciego a las mejoras grandes que no cruzan 0.85.
Figura: [`figures/multiregion_k_over_threshold.png`](figures/multiregion_k_over_threshold.png).

| Modelo | Clases totales | F1>=0.70 | F1>=0.80 | F1>=0.85 |
|---|---|---|---|---|
| **Multi-region** | 30 hojas | 7 | 2 | 2 |
| Solo-PASTIS (capped 6k, mismo presupuesto) | 18 | 6 | 3 | 2 |
| Solo-PASTIS (full 86k, referencia) | 18 | 10 | 4 | 2 |

Las dos clases que cruzan 0.85 en multi-region son `winter_rapeseed_rape`
(F1=0.948) y `vineyards_wine_vine_rebland_grapes` (F1=0.905). Que el conteo a 0.85
sea 2 vs 2 fue lo que llevo a la conclusion erronea de "0 rescatadas, no
funciona": el conteo NO ve que `potatoes` pasa de 0.007 a 0.650 porque 0.650 no
cruza 0.85. Tabla por clase en `data/transfer/multiregion_per_class_leaf.parquet`,
figura [`figures/multiregion_leaf_f1.png`](figures/multiregion_leaf_f1.png).

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

**La idea de Arthur SI rescata varias clases.** Medida con el delta de F1 por
clase al MISMO presupuesto y sobre el MISMO test (no con el conteo binario que la
ocultaba), el multi-region MEJORA 13 de 16 hojas compartidas, con un rescate neto
de **+2.179 F1** y una media por hoja que sube de **0.454 a 0.590**. El conteo
binario "clases sobre 0.85" daba 2 vs 2 y llevaba a la conclusion erronea de "0
rescatadas, no funciona"; esa metrica es ciega a mejoras grandes que aterrizan por
debajo de 0.85 (p. ej. `potatoes` 0.007 -> 0.650).

Lo que SI se cumple:

1. **Hay rescate REAL por clase**, encabezado por los casos exactos que la idea
   predice — clase debil en Francia, fuerte en los balticos:
   - `potatoes` **+0.643** (0.007 -> 0.650); Francia aporta 21 parcelas de train,
     Estonia+Letonia 1 352 (98.5 %).
   - `spring_barley` **+0.381**; `pasture` **+0.278**; `winter_common_soft_wheat`
     **+0.275**; `legumes_harvested_green` **+0.269**; `winter_rapeseed_rape`
     **+0.140** (84-92 % de su entrenamiento viene de fuera de Francia).
   Estos deltas son estables en 3 semillas (42/7/123). **El mecanismo que Arthur
   intuye queda verificado**: cuando una hoja se comparte y otra region la abunda,
   agregar esa region la rescata.
2. **La taxonomia se amplia de verdad**: 18 -> 30 hojas finas reales, sin inventar
   mapeos. De las 14 hojas nuevas que solo-PASTIS no puede etiquetar, 4 cruzan
   F1>=0.50 (`spring_rapeseed_rape` 0.717, `apples` 0.697, `summer_rapeseed_rape`
   0.571, `finola` 0.541) y 7 cruzan 0.30. Es granularidad agronomica nueva
   (colza de primavera/verano, manzanas, trebol) a una calidad util como
   generador de hipotesis.
3. **El nivel grueso no se degrada**: F1-macro europeo 0.658, igual que el
   baseline mono-region. Predecir 30 hojas no rompe la lectura de macro-grupo.

Lo que NO se cumple (los limites reales, sin maquillaje):

4. **Las caidas existen pero son pequenas y por PRESUPUESTO, no por la idea.**
   Solo 3 hojas empeoran: `sunflower` -0.111, `winter_durum_hard_wheat` -0.017,
   `sugar_beet` -0.015 — todas cultivos PASTIS-nativos de soporte pequeno que el
   ensanchamiento del espacio de etiquetas diluye levemente. La suma de caidas
   (-0.143) es ~16x menor que la suma de mejoras (+2.323).
5. **Algunas hojas finas nuevas siguen bajas** (avena 0.307, centeno 0.056,
   alfalfa 0.170, hortalizas finas 0.283) porque distinguir cebada de primavera
   de cebada de invierno, o avena de trigo, exige **fenologia intra-anual** que el
   embedding anual de 64 dim promedia. Esto NO contradice el rescate: es el techo
   del feature anual para esas separaciones concretas, no un fallo del agregado
   multi-region.
6. **El tropico sigue fuera**: la demo Mexico y el F1=0.0 de `non_crop`/
   `other_cropland` confirman que el cabezal europeo no cubre cultivos tropicales.
   Un modelo verdaderamente multi-region global necesitaria few-shot etiquetado en
   cada zona climatica, no solo agregar Europa.

### Conclusion operativa

El modelo multi-region **funciona como mecanismo de rescate por clase**: agregar
Estonia/Letonia rescata cultivos que Francia apenas tiene (papa, colza, cebada de
primavera, leguminosas), subiendo su F1 entre +0.14 y +0.64 al mismo presupuesto,
y ademas amplia la taxonomia 18 -> 30 hojas. La lectura correcta NO es "0
rescatadas, no funciona", sino **"rescata 13 de 16 hojas compartidas (+2.18 F1
neto) y aporta granularidad nueva; el conteo binario 0.85 lo ocultaba"**. Sus
limites son honestos y acotados: caidas pequenas en cultivos puramente franceses
de bajo soporte, hojas que exigen fenologia intra-anual, y el tropico que necesita
few-shot local. El siguiente paso de mayor expectativa de rescate adicional son
features temporales intra-anuales (series Sentinel-2 multi-fecha) sobre las hojas
de cereales/leguminosas que el embedding anual aun no separa — combinadas con,
no en lugar de, el agregado multi-region que ya demostro su valor.
