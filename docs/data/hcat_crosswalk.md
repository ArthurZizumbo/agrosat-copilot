# Crosswalk taxonomico PASTIS-18 -> HCAT v3 (US-074)

Espacio de etiquetas unificado para la transferencia multi-region (EPIC 12). Mapea
las 18 clases agronomicas PASTIS-R (espacio contiguo `semantic18`, ids 0..17) a
codigos HCAT v3 hoja (10 digitos, prefijo `33` = `crop_type`) y las colapsa a
macro-clases canonicas via el nivel de grupo de la jerarquia HCAT.

Fuente de verdad de los codigos: `data/reference/eurocrops_hcat3.csv` (385 nodos
HCAT v3, codigos reales verificados). Nombres y conteos de parcelas:
`data/reference/pastis_class_mapping.json`. **Ningun codigo HCAT se inventa**: cada
`hcat_leaf_code` de la tabla aparece textualmente en el CSV (test
`test_all_leaf_codes_exist_in_eurocrops`). Artefacto consumible:
`data/reference/hcat_crosswalk.parquet` (18 filas, Polars, < 50 KB, en Git).

## 1. Tabla PASTIS-18 -> HCAT v3 hoja

`semantic18_id = pastis_id - 1`. `match_quality = exact` salvo las tres clases sin
hoja 1:1 (ver §1.1), marcadas `approx`.

| PASTIS id | sem18 | Nombre PASTIS | HCAT leaf name | HCAT leaf code | n_parcels | match |
|----------:|------:|---------------|----------------|---------------:|----------:|:-----:|
| 1  | 0  | Meadow                      | pasture_meadow_grassland_grass     | 3302000000 | 31292 | exact |
| 2  | 1  | Soft winter wheat           | winter_common_soft_wheat           | 3301010101 | 8206  | exact |
| 3  | 2  | Corn                        | grain_maize_corn_popcorn           | 3301010600 | 13123 | exact |
| 4  | 3  | Winter barley               | winter_barley                      | 3301010401 | 2766  | exact |
| 5  | 4  | Winter rapeseed             | winter_rapeseed_rape               | 3301060401 | 1769  | exact |
| 6  | 5  | Spring barley               | spring_barley                      | 3301010402 | 908   | exact |
| 7  | 6  | Sunflower                   | sunflower                          | 3301060500 | 1355  | exact |
| 8  | 7  | Grapevine                   | vineyards_wine_vine_rebland_grapes | 3303060000 | 10640 | exact |
| 9  | 8  | Beet                        | sugar_beet                         | 3301290700 | 871   | exact |
| 10 | 9  | Winter triticale            | winter_triticale                   | 3301010801 | 1208  | exact |
| 11 | 10 | Winter durum wheat          | winter_durum_hard_wheat            | 3301010201 | 1704  | exact |
| 12 | 11 | Fruits, vegetables, flowers | fresh_vegetables                   | 3301070000 | 2619  | approx |
| 13 | 12 | Potatoes                    | potatoes                           | 3301030000 | 551   | exact |
| 14 | 13 | Leguminous fodder           | legumes_harvested_green            | 3301090300 | 3174  | approx |
| 15 | 14 | Soybeans                    | soy_soybeans                       | 3301160000 | 1212  | exact |
| 16 | 15 | Orchard                     | orchards_fruits                    | 3303010000 | 2998  | exact |
| 17 | 16 | Mixed cereal                | cereal                             | 3301010000 | 848   | approx |
| 18 | 17 | Sorghum                     | millet_sorghum                     | 3301010900 | 707   | exact |

### 1.1. Tres clases sin hoja 1:1 (`match_quality = approx`)

Decisiones explicitas, no fallos de matching; se mapean al nodo HCAT mas cercano y
se documenta la perdida de granularidad:

- **Mixed cereal (sem18=16) -> `cereal` (3301010000)**: mezcla de cereales sin especie
  unica; se asigna al **nodo de grupo L2 cereal**, NO a una hoja.
- **Fruits, vegetables, flowers (sem18=11) -> `fresh_vegetables` (3301070000)**: la clase
  PASTIS funde horticultura mixta + flores; se elige `fresh_vegetables` como hoja
  representativa dominante (se pierde la rama de flores/frutos).
- **Leguminous fodder (sem18=13) -> `legumes_harvested_green` (3301090300)**: forraje
  leguminoso cosechado en verde (no `legumes_dried_pulses`), decision agronomica.

## 2. Colapso a macro-clases (nivel de grupo HCAT)

Regla determinista: `macro = ancestro de grupo del codigo hoja`. El nodo de grupo es
el ancestro L2 (6 digitos significativos), salvo pasture/permanent que ya son L1. Da
**10 macro-grupos de cultivo** sobre las 18 clases (rango pedido 10-15) + la macro
`void` de partial-label (§3) = **11 macro-clases canonicas** (`MACRO_HCAT_GROUPS`).

| Macro (HCAT group) | HCAT group code | sem18 ids | n clases | n_parcels acum. |
|--------------------|----------------:|-----------|---------:|----------------:|
| grassland          | 3302000000 | 0 (Meadow) | 1 | 31292 |
| cereals            | 3301010000 | 1,2,3,5,9,10,16,17 | 8 | 30570 |
| oilseed_industrial | 3301060000 | 4,6 (rapeseed, sunflower) | 2 | 3124 |
| vineyard           | 3303060000 | 7 (grapevine) | 1 | 10640 |
| sugar_beet         | 3301290000 | 8 (beet) | 1 | 871 |
| vegetables         | 3301070000 | 11 (fruits/veg/flowers) | 1 | 2619 |
| potato             | 3301030000 | 12 (potatoes) | 1 | 551 |
| legumes_fodder     | 3301090000 | 13 (leguminous fodder) | 1 | 3174 |
| soybean            | 3301160000 | 14 (soybeans) | 1 | 1212 |
| orchard            | 3303010000 | 15 (orchard) | 1 | 2998 |
| **void**           | (n/a) | background / out-of-nomenclature | - | ignore |

**Mitigacion del long-tail**: Meadow (~45% de las parcelas crop) queda aislado en
`grassland`, y los 8 cereales hermanos (incl. los inseparables Soft/Durum winter
wheat) se funden en `cereals`, reduciendo la varianza inter-clase que hundia el
F1-macro a 18 clases.

### 2.1. Variante legada de 6 familias (comparativa, no canonica)

El parquet trae tambien `macro_hcat_l1_6`, la particion de 6 familias HCAT L1
(`CEREALS / OILSEEDS / ROOT_CROPS / LEGUMES / PERMANENT_WOODY / GRASSLAND_OTHER`)
del JSON `pastis_class_mapping.json:groupings.hcat_l1_6`, que es la que ya logro
**XGBoost 0.6535 F1-macro** (hallazgo de producto v8, compatibilidad E4/E6). El
notebook `02f_crosswalk_hcat.ipynb` muestra AMBAS: 6 familias legadas y las 11
macro-clases HCAT (mas finas, canonicas de E12). El label-space registrado
(`hcat-macro`) expone ambos niveles en `class_names`.

## 3. Convencion void/background unificada

| Dataset | Background | Void / out-of-scope | Convencion unificada |
|---------|-----------|---------------------|----------------------|
| PASTIS-R | id `0` (no-agricola, 'stuff') | id `19` (cultivo fuera de nomenclatura o <50% overlap) | ambos -> `ignore_index = 255` (ya en `remap_20_to_18`) |
| Sen4AgriNet | clase `0` background del `linear_encoder` | parcelas sin codigo FAO-ICC mapeable a HCAT | `0` -> `ignore_index`; sin-mapeo -> `null-class` (partial-label) |
| EuroCropsML | sin pixel background (tabular parcela) | clase HCAT ausente en el split few-shot | ausencia -> `null-class`, NO falso negativo |

**Regla unica**: `background / void -> ignore` (no entra a la matriz de confusion);
`ausencia cross-dataset -> null-class` (etiqueta desconocida, partial-label, NO
negativo duro). `ignore_index = 255` reutiliza `HARNESS_IGNORE_INDEX` de US-030
(ya aplicado en `remap_20_to_18`).

## 4. Estrategia clases disjuntas (recomendacion, sin implementar el loss)

Protocolo **partial-label / null-class estilo UniSeg** para US-075/US-076 (NO se
implementa aqui; US-074 solo deja el espacio comun y la convencion documentada):

- Una clase etiquetada en el dataset A pero ausente (background) en B NO cuenta como
  negativo duro en B -> se trata como `null-class` (etiqueta desconocida).
- Loss recomendado: **BCE class-independent** (un sigmoide por clase HCAT, no softmax
  mutuamente excluyente) + **cross-dataset relation loss** que penaliza solo las
  clases presentes en cada dataset, evitando el conflicto de gradiente entre regiones.
- US-074 entrega: la tabla de mapeo, el parquet, la convencion null-class y el
  label-space `hcat-macro` en el registry. El loss lo consumira la US siguiente.

## 5. Artefactos

- `data/reference/hcat_crosswalk.parquet` — tabla §1 + §2 (esquema en
  `ml/data/hcat_crosswalk.py:CROSSWALK_SCHEMA`; codigos `Utf8` para preservar ceros).
- `ml/data/hcat_crosswalk.py` — `build_crosswalk` (re-deriva + valida contra el CSV),
  `write_crosswalk`, `load_crosswalk`, `MACRO_HCAT_GROUPS`.
- `ml/eval/class_remap.py` — label-space `hcat-macro` registrado via
  `register_label_space` (US-053 AMPLIADO; `france-9` y `classify.py` intactos).
- `notebooks/eda/02f_crosswalk_hcat.ipynb` — figuras/tablas reales (long-tail,
  comparativa 18 vs 11 macro vs 6 familias, jerarquia HCAT, roundtrip parquet).
