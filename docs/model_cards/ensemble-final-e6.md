# Model Card — Ensemble final (EPIC 6)

Modelo final del MVP: ensemble de los miembros base del proyecto (segmentacion
temporal + tabular AlphaEarth + FarSLIP) combinados a nivel de parcela. En
espanol neutro. **Todas las cifras provienen de los artefactos reales citados
en cada seccion.**

---

## 1. Model Details

- **Nombre**: Ensemble final AgroSatCopilot (EPIC 6).
- **Estrategias**: Stacking (meta-learner `logreg` sobre las probabilidades
  out-of-fold de los miembros) y Blending (Optuna sobre pesos). El catalogo de
  arquitecturas de EPIC 6 lista cuatro familias — Voting top-3, Bagging
  XGB+AlphaEarth, Stacking, Blending; las dos campeonas reportadas aqui son
  Stacking y Blending.
- **Meta-learner**: regresion logistica (`meta: "logreg"`).
- **Miembros (configuracion de 5)**: `tsvit-pheno-fullm`, `utae`,
  `xgb-alphaearth`, `farslip-ft18`, `farslip-zeroshot`. La configuracion de 3 es
  `tsvit-pheno-fullm`, `utae`, `xgb-alphaearth`.
- **Espacio de etiquetas**: `semantic18` (18 clases). La cifra HCAT-6 (0.6535)
  es **otro eje y no es comparable** con estas (nota del propio artefacto).

## 2. Intended Use

Clasificacion de cultivos a nivel de parcela sobre PASTIS-R como componente
final del copiloto agricola. **Fuera de alcance**: uso operativo en regiones sin
validacion local (ver multi-region en la card de FarSLIP) y decisiones
agronomicas automatizadas sin revision humana.

## 3. Training Data

PASTIS-R en el universo de evaluacion de EPIC 6 (`n_universe: 16640` parcelas;
`n_universe: 16640` en `us043_farslip_summary.json`). Espacio de 18 clases. Las
probabilidades out-of-fold de los miembros base alimentan el meta-learner.

## 4. Evaluation

Validacion cruzada out-of-fold para Stacking; fold-5 para Blending. Metricas:
`f1_macro` y `accuracy`. Nota honesta del artefacto: el "Stacking +Gemma 4" del
catalogo de arquitecturas se documenta como **diseno**; el meta-learner
realmente entrenado es el `logreg` que reporta el CSV. No se reclama Gemma 4
porque no hay run de Gemma (ADR-011, future).

## 5. Metrics

Fuente: `reports/ensemble/us043_farslip_summary.json` y
`reports/ensemble/metrics/us043_farslip_stacking_blending.csv` (espacio
semantic18, mejor miembro individual `tsvit-pheno-fullm`).

| Estrategia | f1_macro | accuracy | Delta f1 vs 3 miembros |
|---|---|---|---|
| Stacking-3 (referencia) | 0.6359 | 0.7877 | 0.0 |
| Stacking-5 (+farslip-ft18, +farslip-zeroshot) | **0.6477** | 0.7935 | +0.0118 |
| Blending-3 (referencia) | 0.5651 | 0.7697 | 0.0 |
| Blending-5 (+farslip) | 0.5866 | 0.7864 | +0.0215 |

**Lectura honesta**: anadir los dos miembros FarSLIP aporta una mejora pequena
pero real (+0.0118 f1_macro en Stacking, +0.0215 en Blending). La mejor cifra de
ensemble en este eje es **Stacking-5 = 0.6477 f1_macro**.

Una segunda rejilla (`reports/ensemble/metrics/us043_farslip_grid.csv`) reporta
el campeon US-040 `tsvit-pheno` con Stacking-5 = **0.7486 f1_macro** /
0.8495 accuracy (delta FarSLIP +0.0016). Las dos rejillas usan distinto miembro
de referencia (mejor individual `tsvit-pheno-fullm` vs campeon US-040
`tsvit-pheno`); ambas se reportan tal cual, sin promediarlas.

Comparativa de las cuatro familias (`reports/ensemble/metrics/comparison_us040.csv`):

| Modelo | f1_macro | accuracy | chosen |
|---|---|---|---|
| E3 Stacking (parcela) | 0.7470 | 0.8490 | si |
| E4 Blending (parcela) | 0.7414 | 0.8618 | no |
| TSViT-pheno (individual) | 0.6253 | (NaN) | no |
| E1 Voting (pixel) | 0.6225 | 0.8090 | no |
| E2 Bagging (parcela) | 0.5864 | 0.7816 | no |

El miembro **elegido** (`chosen=true`) es E3 Stacking a nivel de parcela
(0.7470 f1_macro). Las cifras de inferencia (s) por estrategia tambien viven en
ese CSV.

> Cualquier metrica del cierre de EPIC 6 que no este en estos artefactos queda
> **pendiente de cierre E6** y NO se reporta aqui (regla de datos reales).

## 6. Limitations & Ethical Considerations

- Evaluado solo sobre PASTIS-R (Francia); la generalizacion a otras regiones se
  documenta en la card de FarSLIP (transfer multi-region).
- El delta por anadir FarSLIP es pequeno y puede ser negativo en una de las
  rejillas de Blending (`us043_farslip_grid.csv`: Blending-5 campeon = -0.0117);
  se reporta tal cual.
- "Stacking +Gemma 4" es diseno, no run entrenado. No se reclama esa cifra.

## 7. Licenses & Attribution

- Datos de entrenamiento: PASTIS-R (ver `docs/licenses/DATA_LICENSE.md`).
- Embeddings de un miembro: AlphaEarth V1/ANNUAL v1.1, CC-BY-4.0 (atribucion
  "Google DeepMind AlphaEarth Foundations (Satellite Embedding V1 Annual,
  CC-BY-4.0)").

## 8. Reproducibility / MLflow

- Tags MLflow obligatorios: `data_version` + `code_version` (via
  `ml/utils/mlflow_utils.py::track_experiment`).
- Artefactos: `reports/ensemble/us043_farslip_summary.json`,
  `reports/ensemble/metrics/us043_farslip_stacking_blending.csv`,
  `reports/ensemble/metrics/us043_farslip_grid.csv`,
  `reports/ensemble/metrics/comparison_us040.csv`.
- **Gotcha**: el lineage MLflow vive en el server Docker `:5010`, no en
  `./mlruns`; un run por subprocess contra el server equivocado queda `RUNNING`.
  Ver [README de model cards](README.md).
