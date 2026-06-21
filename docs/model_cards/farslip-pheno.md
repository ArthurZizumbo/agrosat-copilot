# Model Card — FarSLIP-pheno

Modelo **vision-language** para clasificacion de cultivos con prototipos de texto
fenologicos, fidelidad 1:1 al paper de Li et al. (2025)
([ADR-007](../decisions/ADR-007-farslip-fidelity-paper.md)) y reencuadre
fenologico ([ADR-006](../decisions/ADR-006-reencuadre-baseline-fenologico.md)).
En espanol neutro. Cifras desde los CSV reales citados.

---

## 1. Model Details

- **Nombre**: FarSLIP-pheno.
- **Metodo**: FarSLIP (Li et al., 2025), fidelidad 1:1 al paper documentada en
  ADR-007. Alinea embeddings de imagen con prototipos de texto generados a partir
  de descripciones fenologicas (texto generado con Gemini Flash; coste ~centavos
  por descripcion).
- **Codigo**: `ml/farslip/` (`train.py`, `caption_encoder.py`,
  `mpcl_loss.py`, `parcel_crop_dataset.py`, ...).
- **Espacio de salida**: variantes a nivel de parcela (`semantic18`) y barridos
  de cardinalidad de clases.

## 2. Intended Use

Clasificacion de cultivos a nivel de parcela, especialmente en regimen
self-supervised / zero-shot y few-shot, donde la senal fenologica aporta valor
(complementa a TSViT-pheno, que satura en supervisado denso). Entra como dos
miembros (`farslip-ft18`, `farslip-zeroshot`) del ensemble final.

## 3. Training Data

PASTIS-R con prototipos de texto fenologicos reales. Para el transfer
multi-region se usa **Sen4AgriNet** (subset Catalonia 31TCG, CC-BY-SA-4.0).

## 4. Evaluation

- Separabilidad y F1 macro del espacio FarSLIP fiel vs AlphaEarth.
- Barrido de cardinalidad de clases a nivel de parcela.
- Transfer multi-region FR -> Catalonia (zero-shot y few-shot).

## 5. Metrics

**FarSLIP fiel vs AlphaEarth** (`reports/farslip/metrics/us037_farslip_fiel_vs_alphaearth.csv`):

| Espacio | silhouette | f1_macro (mean) | n_dims | n_samples |
|---|---|---|---|---|
| farslip_pheno (fiel v2) | 0.0120 | 0.5551 | 768 | 567 |
| alphaearth_2019 | 0.0145 | 0.6446 | 64 | 567 |

**Lectura honesta**: el espacio FarSLIP fiel (768-dim) no supera a AlphaEarth
(64-dim) en separabilidad ni F1 macro sobre esta evaluacion; se reporta tal cual.
El valor de FarSLIP aparece como miembro complementario del ensemble (delta
positivo pequeno, ver card del ensemble) y en el transfer multi-region.

**Barrido de cardinalidad por parcela** (`reports/farslip/metrics/parcel_sweep.csv`):

| n_classes | macro_f1 | macro_iou | n parcelas eval |
|---|---|---|---|
| 4 | 0.7025 | 0.5547 | 1301 |
| 6 | 0.4579 | 0.3120 | 1843 |
| 8 | 0.4075 | 0.2657 | 2301 |
| 10 | 0.3589 | 0.2288 | 2706 |
| 12 | 0.3328 | 0.2071 | 3200 |

El F1 cae con el numero de clases (4 clases faciles 0.7025 -> 12 clases 0.3328),
como es esperable; la dificultad crece con la cardinalidad.

**Baseline fiel sin class weights** (`reports/farslip/metrics/faithful_v2_summary.csv`):
variante ganadora `faithful_v2` macro_f1 0.164 / macro_iou 0.111 en el espacio de
18 clases (4 clases con senal clara: Meadow / Grapevine / Corn / Orchard). Las
class weights inverse-freq empeoraron el balance neto (macro_f1 0.102).

## 5.1 Generalizacion — transfer multi-region (FR -> Catalonia)

Fine-tune denso del checkpoint frances sobre Sen4AgriNet Catalonia 31TCG
(`reports/segmentation/sen4agrinet_transfer_result.json`):

| Regimen | mIoU | F1 macro | Pixel acc |
|---|---|---|---|
| Zero-shot | 0.0000 | 0.0000 | 0.0000 |
| Few-shot (40 epochs, 10 train patches) | **0.2468** | 0.3005 | 0.9179 |
| **Delta mIoU** | **+0.2468** | | |

El zero-shot es 0.0000 (cambio de region y espacio de etiquetas); con few-shot
(10 patches de entrenamiento, 20 de validacion) el modelo recupera mIoU 0.2468
sobre 10 macro-grupos (grassland, cereals, vineyard, ...). El delta es **real y
positivo** (+0.2468) y demuestra que pocas etiquetas locales bastan para
adaptar el modelo a una region nueva.

## 6. Limitations & Ethical Considerations

- El espacio FarSLIP fiel no supera a AlphaEarth en separabilidad sobre la
  evaluacion fiel; el aporte es complementario, no sustituto.
- Zero-shot cross-region = 0.0000: el modelo NO transfiere sin few-shot local.
  Se reporta sin maquillar.
- Caida fuerte de F1 al subir la cardinalidad de clases.

## 7. Licenses & Attribution

- PASTIS-R (entrenamiento base); ver `docs/licenses/DATA_LICENSE.md`.
- **Sen4AgriNet** (transfer Catalonia): CC-BY-SA-4.0 — atribucion obligatoria
  (HF `paren8esis/S4A`).
- Embeddings de comparacion: AlphaEarth V1/ANNUAL v1.1, CC-BY-4.0.
- Metodo: Li et al. (2025), FarSLIP; fidelidad 1:1 en ADR-007.

## 8. Reproducibility / MLflow

- Tags MLflow: `data_version` + `code_version` (`ml/utils/mlflow_utils.py`).
- Codigo: `ml/farslip/`.
- Artefactos: `reports/farslip/metrics/us037_farslip_fiel_vs_alphaearth.csv`,
  `reports/farslip/metrics/parcel_sweep.csv`,
  `reports/farslip/metrics/faithful_v2_summary.csv`,
  `reports/segmentation/sen4agrinet_transfer_result.json`,
  `reports/baseline/04_farslip_eval_pastis/`.
- **Gotcha** MLflow dos almacenes (`:5010` vs `./mlruns`): ver [README](README.md).
