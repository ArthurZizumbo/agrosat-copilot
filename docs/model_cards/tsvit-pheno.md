# Model Card — TSViT-pheno

Modelo de **segmentacion semantica temporal** de cultivos sobre series
Sentinel-2: reimplementacion limpia de TSViT (Tarasiou et al., 2023) con una
rama fenologico-contrastiva ([ADR-006](../decisions/ADR-006-reencuadre-baseline-fenologico.md)).
En espanol neutro. Todas las cifras provienen de los CSV reales citados.

---

## 1. Model Details

- **Nombre**: TSViT-pheno.
- **Arquitectura**: Temporal-Spatial Vision Transformer (TSViT) factorizado
  temporal-luego-espacial, con cabeza de segmentacion densa y rama
  fenologico-contrastiva adicional. Reimplementacion propia.
- **Codigo**: `ml/models/tsvit_wrapper.py`.
- **Entrada**: series temporales Sentinel-2, 10 bandas (`in_channels=10`), sin
  resize (`needs_resize=false`).
- **Variantes**: `tsvit-base` (sin rama fenologica, US-038) y `tsvit-pheno`
  (con rama fenologica, US-039). La variante `-fullm` usa la mascara completa.

## 2. Intended Use

Segmentacion densa de cultivos por pixel sobre tiles Sentinel-2 con etiquetas
PASTIS-R. Es el mejor miembro de segmentacion del proyecto y entra como miembro
del ensemble final (EPIC 6). **Fuera de alcance**: inferencia en regiones sin
fine-tune local sin medir el transfer (ver card FarSLIP / multi-region).

## 3. Training Data

PASTIS-R (Sentinel-2 multitemporal sobre Francia, etiquetas densas de cultivo).
Evaluacion sobre fold-5. Espacio de etiquetas de segmentacion del proyecto.

## 4. Evaluation

Protocolo fold-5 (`n_patches: 496`). Metricas: mIoU, F1 macro, pixel accuracy,
Cohen kappa, balanced accuracy. La rama fenologica se evalua por su delta contra
la base.

## 5. Metrics

**Comparativa fold-5 de arquitecturas** (`reports/segmentation/metrics/model_comparison_fold5.csv`):

| Modelo | mIoU | F1 macro | Pixel acc | Cohen kappa |
|---|---|---|---|---|
| **tsvit-pheno** | **0.6139** | **0.7401** | 0.8637 | 0.8310 |
| segformer | 0.2715 | 0.3777 | 0.6395 | 0.5538 |
| deeplabv3plus | 0.2540 | 0.3614 | 0.6539 | 0.5676 |
| unet | 0.2056 | 0.2847 | 0.6394 | 0.5494 |
| anysat | 0.1684 | 0.2240 | 0.6994 | 0.5926 |
| utae | 0.1605 | 0.2240 | 0.6010 | 0.5080 |

TSViT-pheno es claramente el mejor modelo de segmentacion del banco (mIoU 0.6139
/ F1 0.7401 en fold-5).

**Delta de la rama fenologica** (`reports/segmentation/metrics/tsvit_pheno_vs_base_fold5.csv`):

| Variante | mIoU fold5 | F1 macro fold5 | Pixel acc | Delta mIoU |
|---|---|---|---|---|
| tsvit-base-fullm (US-038) | 0.6789 | 0.7942 | 0.8878 | 0.0 |
| tsvit-pheno-fullm (US-039) | 0.6756 | 0.7918 | 0.8885 | -0.0033 |

**Lectura honesta del artefacto**: en regimen supervisado el delta fenologico
esperado es ~0 (saturacion supervisada, plan v8): con etiquetas densas el modelo
ya aprende las firmas temporales, y la rama contrastiva aporta poco. Un
delta >= +0.03 seria SOSPECHA; un delta < 0 es valido. El valor real de la
fenologia esta en el regimen self-supervised / zero-shot de FarSLIP, no aqui.

En la comparativa de ensemble (`reports/ensemble/metrics/comparison_us040.csv`),
TSViT-pheno individual figura con **f1_macro 0.6253** (accuracy NaN: la fila se
reporta solo por f1_macro a nivel de parcela en ese eje).

## 6. Limitations & Ethical Considerations

- Evaluado sobre PASTIS-R (Francia); transfer a otras regiones medido aparte.
- La rama fenologica no mejora el regimen supervisado denso (delta ~0 / leve
  negativo); se incluye por su aporte en el regimen self-supervised de FarSLIP.
- mIoU/F1 dependen del fold; se reportan los de fold-5 tal cual.

## 7. Licenses & Attribution

- Datos: PASTIS-R (ver `docs/licenses/DATA_LICENSE.md`).
- Arquitectura derivada de Tarasiou et al. (2023), TSViT; reimplementacion
  propia.

## 8. Reproducibility / MLflow

- Tags MLflow: `data_version` + `code_version` (`ml/utils/mlflow_utils.py`).
- Codigo: `ml/models/tsvit_wrapper.py`.
- Artefactos: `reports/segmentation/metrics/model_comparison_fold5.csv`,
  `reports/segmentation/metrics/tsvit_pheno_vs_base_fold5.csv`.
- **Gotcha** MLflow dos almacenes (`:5010` vs `./mlruns`): ver [README](README.md).
