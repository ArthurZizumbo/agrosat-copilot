# US-049 v2 — Analisis cualitativo de trazas (por que puntuan asi los modelos)

Fuente: `reports/agent_bench/traces/trace_<variante>_<benchmark>.jsonl`.
Variantes: `gemini` (cloud flash), `qwen` (vLLM on-prem), `gemma-base` (MoE Ollama on-prem), `qwen36-vl` (on-prem VL).
Este documento explica el origen de los numeros, no solo los reproduce.

## 1. AgroMind: exact_match desglosado por answer_type

El score global de AgroMind mezcla cinco tipos de respuesta con metricas incompatibles. El `exact_match` solo es interpretable en `multiple_choice` (comparar una letra) y, parcialmente, en `open_number` / `yes_no`. En `open_numeric_bbox` y `open_text` el `exact_match` es estructuralmente incapaz de premiar una respuesta buena.

| Variante | multiple_choice | open_number | open_numeric_bbox | open_text | yes_no | Global |
|----------|-----------------|-------------|-------------------|-----------|--------|--------|
| gemini      | 63/99 = 0.636 | 12/35 = 0.343 | 0/8 = 0.000 | 1/10 = 0.100 | 6/15 = 0.400 | 82/167 = 0.491 |
| gemma-base  | 25/99 = 0.253 |  6/35 = 0.171 | 0/8 = 0.000 | 1/10 = 0.100 | 6/15 = 0.400 | 38/167 = 0.228 |
| qwen36-vl   | 45/99 = 0.455 | 12/35 = 0.343 | 0/8 = 0.000 | 0/10 = 0.000 | 6/15 = 0.400 | 63/167 = 0.377 |
| qwen (on-prem texto) | — | 0/2 | — | — | — | 0/2 (corrida abortada) |

Hallazgos:

- **`open_numeric_bbox` = 0.000 para TODAS las variantes (0/8 cada una).** No es que los modelos fallen al localizar: es que el oro es un bounding box normalizado (p. ej. `[0.4167, 0.362, 0.9745, 0.74]`) y `exact_match` exige igualdad textual exacta. Ningun modelo puede acertar 4 decimales por coincidencia de cadena. Este bucket **no aporta senal** y arrastra el global hacia abajo por igual en todos.
- **`multiple_choice` es donde vive la senal real** (99 de 167 items). Ahi el oro es una letra y el parser extrae `Respuesta: X` de forma fiable. La separacion entre variantes proviene casi toda de aqui: gemini 0.636 > qwen36-vl 0.455 > gemma-base 0.253.
- `yes_no` esta empatado a 0.400 (6/15) en las tres variantes con corrida completa: el binario tiene poca varianza y no discrimina.
- `open_text` ronda 0.0-0.1 por la misma razon que el bbox: el match exacto de texto libre es casi imposible (haria falta LLM-judge, no exact_match).
- La variante `qwen` (endpoint de texto on-prem) solo tiene 2 registros: la corrida AgroMind aborto porque el benchmark es visual y el endpoint de solo-texto no procesa imagenes. No es comparable; usar `qwen36-vl` como representante on-prem en AgroMind.

Conclusion del punto 1: el global de AgroMind esta diluido por buckets no puntuables (bbox + open_text = 18/167 condenados a ~0). La comparacion honesta entre modelos se hace **sobre `multiple_choice`**.

## 2. Ejemplos concretos de fallo (oro vs prediccion)

### (a) Item bbox que nadie puede puntuar (gemini)

- **gold**: `[0.4167, 0.362, 0.9745, 0.74]`
- **prediction**: razonamiento correcto en prosa ("se observa una unica parcela agricola... en la parte central derecha, forma poligonal de tonos marrones...") y termina describiendo la region adecuada.
- **exact_match = 0.0**. La localizacion cualitativa es razonable, pero `exact_match` solo compara cadenas: jamas igualara cuatro flotantes. Este 0 es un artefacto de la metrica, no una incapacidad del modelo.

### (b) multiple_choice donde el modelo razono a la letra equivocada (gemini)

El parser SI extrae la letra final (`Respuesta: X`); son errores reales de eleccion, no de formato:

- **gold = A**, pred termina `...abarca todo el ancho y alto del sector derecho cultivado. Respuesta: C`. El modelo eligio C tras un razonamiento espacial plausible pero invirtio el cuadrante.
- **gold = E**, pred termina `...los arboles estan localizados de manera principal y mas densa en la region superior central. Respuesta: B`. Confunde "superior central" (B) con el centro real (E) en la cuadricula 3x3.
- **gold = G** (anomalia, 7 opciones), pred clasifica como `double_plant ... Respuesta: B` cuando el oro era otra categoria de anomalia. Error semantico de teledeteccion, no de parsing.

Esto demuestra que el gap de `multiple_choice` es senal genuina de razonamiento espacial/agronomico, no ruido de extraccion.

## 3. GeoAnalystBench: pass_rate y dispersion

| Variante | pass_rate | workflow_sim (media [min,max]) | codebleu (media [min,max]) | preds con codigo |
|----------|-----------|-------------------------------|----------------------------|------------------|
| gemini      | 28/50 = 0.560 | 0.396 [0.12, 0.75] | 0.287 [0.06, 0.44] | 49/50 |
| qwen        | 34/50 = 0.680 | 0.488 [0.19, 0.76] | 0.302 [0.18, 0.45] | 50/50 |
| gemma-base  | 37/50 = 0.740 | 0.439 [0.10, 0.72] | 0.005 [0.00, 0.11] | 7/50 |
| qwen36-vl   | 34/50 = 0.680 | 0.412 [0.12, 0.71] | 0.196 [0.00, 0.42] | 23/50 |

Hallazgos:

- **pass_rate on-prem >= cloud-flash**: gemma 0.74, qwen 0.68, qwen36-vl 0.68, todas por encima de gemini 0.56. El `passed` se decide por similitud de flujo de trabajo (`workflow_sim` sobre umbral), no por la fidelidad del codigo.
- **codebleu cuenta otra historia y delata el "como"**: gemma-base tiene codebleu medio 0.005 (47/50 items en ~0) porque **emite el plan en prosa/LaTeX, no en Python** (solo 7/50 predicciones contienen codigo). CodeBLEU puntua n-gramas/AST/dataflow de codigo; sin codigo, colapsa a 0. Aun asi gemma lidera pass_rate porque su secuencia de pasos (workflow_sim 0.439) es correcta.
- qwen es el mas equilibrado: alto en ambos (ws 0.488, codebleu 0.302, codigo en 50/50). qwen36-vl emite codigo solo en 23/50, de ahi su codebleu intermedio (0.196).
- Lectura: **pass_rate mide "entendio el procedimiento", codebleu mide "lo escribio como codigo ejecutable".** gemma gana lo primero pero falla lo segundo; para un copiloto que debe generar codigo real, qwen es la opcion on-prem mas solida.

### Ejemplo de buen workflow generado (qwen, ws=0.758, passed)

```
WORKFLOW:
1. Load the rural area boundaries from dataset/ShikokuMetropolitan.geojson and filter for only rural areas (AREATYPE field).
2. Load the population data from dataset/ShikokuPopulation.geojson and intersect it with the rural area boundaries.
3. Load the all-season roads data from dataset/AllSeasonRoads.geojson.
4. Create a 2-kilometer buffer around the all-season roads (convert 2 km to CRS units).
5. Clip the buffered roads layer to the extent of the rural areas.
6. Calculate ... (poblacion rural cubierta por el buffer)
```

Secuencia correcta load -> filter -> intersect -> buffer -> clip -> aggregate: por eso pasa con workflow_sim alto.

## 4. Titular honesto

- **Generacion de codigo (GeoAnalystBench, por pass_rate): on-prem >= cloud-flash.** gemma 0.74 / qwen 0.68 / qwen36-vl 0.68 superan a gemini-flash 0.56. Matiz obligado: gemma gana por plan de pasos, no por codigo (codebleu ~0); qwen es el mejor "de verdad ejecutable" on-prem.
- **Opcion multiple visual (AgroMind, por multiple_choice): gemini lidera** (0.636 vs qwen36-vl 0.455 vs gemma-base 0.253). El razonamiento espacial/agronomico fino sobre imagen sigue favoreciendo a la nube.
- **El global de AgroMind engaña**: los buckets `open_numeric_bbox` y `open_text` (18/167) estan condenados a ~0 por la metrica, no por los modelos. Comparar siempre por `multiple_choice`.
