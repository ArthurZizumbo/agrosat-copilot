# AgroMind-IT/ES — benchmark bilingue italiano/espanol (eval-only)

> Contribucion academica original de AgroSatCopilot (EPIC 11, US-068). Benchmark
> de 500 pares de preguntas y respuestas (Q&A) agricolas sobre imagenes
> **Sentinel-2 reales de Italia**, en italiano y espanol, cubriendo las diez
> familias de preguntas del copiloto. Publicacion destino: **Zenodo con DOI**,
> licencia **CC-BY-4.0**.

## Estado

- **Codigo LISTO** (este repo): esquema JSONL, generador seed con Gemini 2.5-pro,
  app Streamlit de revision humana, builder de metadata Zenodo.
- **Datos pendientes** (BLOCKER, ver `docs/blockers/epic11-notas.md`): la
  generacion de los 500 pares reales (necesita auth GEE + key Gemini + imagenes
  S2 de Italia), la revision humana nativa (IT por Scuola Sant'Anna, ES por el
  equipo) y el upload a Zenodo con DOI.
- En el repo solo va `seed.fixture.jsonl` (3 pares de **ejemplo de estructura**,
  marcados `source=fixture`) para los tests. **NO** es el benchmark.

## EVAL-ONLY (regla irrevocable)

Este benchmark es **estrictamente de evaluacion**: NO existe particion de
entrenamiento. El AgroMind original (~28,482 Q&A) tampoco tiene train split;
ajustar (fine-tune) cualquier modelo sobre estos pares seria **fuga de datos
(leakage)**. Garantias de diseno:

- El esquema **no tiene** campo `split`.
- El validador (`ml/eval/agromind_it_es/schema.py::validate_record`) **rechaza**
  cualquier registro con `split=train`/`training` o `is_train=true`.
- La unica provenance que entra al set publicado es `human-edited` (post-revision
  nativa); `gemini-seed` y `dry-run` son borradores intermedios.

## Esquema JSONL (compatible AgroMind)

Un objeto JSON por linea. Campos:

| Campo | Tipo | Descripcion |
|-------|------|-------------|
| `item_id` | `str` | Id estable, p.ej. `it-vigor-0007` (`<lang>-<family>-<NNNN>`). |
| `category` | `str` | Una de las 10 familias (ver abajo). |
| `lang` | `str` | `it` o `es` (ISO-639-1). |
| `question` | `str` | Pregunta en `lang`. |
| `options` | `obj` | `{A, B, C, D, ...}` (vacio en items abiertos numericos/texto). |
| `answer` | `str` | Letra de opcion, o numero / texto corto en items abiertos. |
| `image` | `str\|null` | Ruta relativa a la imagen Sentinel-2 de Italia (o `null`). |
| `is_multimodal` | `bool` | Derivado: `true` si hay `image`. |
| `reviewed` | `bool` | `true` tras aceptacion de un revisor nativo. |
| `reviewer` | `str\|null` | Id del revisor humano. |
| `source` | `str` | `gemini-seed` \| `dry-run` \| `human-edited` \| `fixture`. |
| `type_id` | `int` | Id numerico estilo AgroMind (default: ordinal de la familia). |

La compatibilidad con AgroMind se verifica (no se declara) via
`schema.to_agromind_item(item)`, que construye el `AgroMindItem` real de
`ml/eval/agent_bench.py`; si construye sin error, el record es consumible por el
harness de evaluacion sin cambios (US-069). El test `test_agromind_compat` lo
ejerce.

## Las 10 familias de preguntas del catalogo del copiloto

| # | `category` | Intent ancla del copiloto |
|---|-----------|---------------------------|
| 1 | `classification` | `classify_new_parcel` (tipo de cultivo) |
| 2 | `quantification` | area / conteo por clase |
| 3 | `vigor` | indices NDVI / EVI |
| 4 | `water_stress` | NDWI / humedad |
| 5 | `phenology` | `phenology_descriptor` (Wen et al. 2025) |
| 6 | `comparison` | `compare_models` / parcela-vs-parcela |
| 7 | `anomaly` | deteccion de outliers temporales |
| 8 | `metadata` | atributos de parcela / fecha de adquisicion |
| 9 | `intersection` | spatial join / vecindad (PostGIS) |
| 10 | `explainability` | `explain_prediction` (perceiver -> texto) |

Objetivo: 250 `it` + 250 `es` => ~25 pares por familia por idioma.

## Reproducir (cuando los insumos esten disponibles)

```bash
# 1. Generar el seed (BLOCKER B1: requiere key Gemini + imagenes S2 de Italia).
#    Sin credenciales corre en dry-run (emite el plan, no llama a la API).
python -m ml.eval.agromind_it_es.generate_seed \
    --image-root data/s2_italia --n-per-family 25 --languages it es \
    --out data/benchmark/agromind_it_es/seed.jsonl

# 2. Revision humana nativa (BLOCKER B2: reviewer Sant'Anna IT + miembro ES).
streamlit run ml/eval/agromind_it_es/review_app.py

# 3. Metadata Zenodo + upload con DOI (BLOCKER B3: token Zenodo del sponsor).
python -c "from pathlib import Path; from ml.eval.agromind_it_es.zenodo_metadata import write_zenodo_metadata; write_zenodo_metadata(Path('data/benchmark/agromind_it_es/.zenodo.json'))"
```

## Atribucion

- Imagenes: **Sentinel-2** (Copernicus / ESA), descargadas via Google Earth
  Engine; embeddings de contexto: **AlphaEarth Foundations** (Satellite Embedding
  V1 Annual, data version 1.1, CC-BY-4.0).
- Seed: **Gemini 2.5-pro** (GA, 1M ctx).
- Revision nativa: Scuola Superiore Sant'Anna (italiano) + equipo AgroSatCopilot
  (espanol).
- Formato compatible con **AgroMind** (eval-only, ~28,482 Q&A).
- Licencia del dataset: **CC-BY-4.0**.
