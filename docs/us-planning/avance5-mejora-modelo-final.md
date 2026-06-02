# Avance 5 — Plan de mejora del modelo individual final (TSViT) y compuerta a producción

**Fecha**: 2026-06-02 · **Responsable ML**: Isaac Ávila · **Soporte**: Arthur (MLOps), Aaron (backend/serving)
**Entregables**: Avance 5 (modelo final + 4 ensambles) dom **7-jun**; Avance 6 (decisión producción) dom **14-jun** ([ADR-008](../decisions/ADR-008-rediseno-calendario-presentacion-27jun.md)).
**Insumo**: resultados Avance 4 (`reports/segmentation/metrics/`, `reports/avance4/artifacts/`).

> Este plan se construyó con análisis multi-agente + verificación adversarial. Los *lifts* son los **ajustados honestos** (post-escrutinio), no los optimistas. Cada palanca cita evidencia `file:line`.

---

## 0. TL;DR — el veredicto honesto

- **Mejor modelo del Avance 4 = TSViT-pheno** (temporal): mIoU **0.6253**, F1-macro **0.7500**, pixel-acc 0.8759 (PASTIS-R, fold-4, 482 parches). Dobla a los spatial-only (U-Net/DeepLab/SegFormer ~0.24-0.27 mIoU). **La señal está en el tiempo.**
- **La brecha a producción es desbalance de clases**: pixel-acc 0.88 vs F1-macro 0.75 ⇒ las clases minoritarias hunden el macro. El target (F1 ≥ 0.80, mIoU ≥ 0.70 — [CLAUDE.md](../../CLAUDE.md)) está a **+0.05 F1 y +0.075 mIoU**.
- **El "ajuste fino Optuna" del Avance 4 fue un no-op**: 30 trials pero 3 épocas/trial y anclado al checkpoint, solo lr/wd/batch ⇒ mejora ≈ 0 (`ml/tune/optuna_segmentation.py:181-236,461`).
- **Honestidad dura**: con las palancas legítimas, el modelo **individual flat-18** realistamente llega a **F1 ~0.78-0.80 / mIoU ~0.64-0.68**. **El mIoU ≥ 0.70 en flat-18 NO se alcanza** en Avance 5: es el techo de los modelos temporales (~0.63 mIoU) y el desbalance que el class-weighting mejora a costa del micro.
- **Por eso la aceptación de producción NO se juega a "cruzar el número mágico"** sino a un **GO-condicional documentado**: taxonomía agronómica grouped-6 + ensambles + capa de respaldo (baseline XGB+AlphaEarth, F1 ≥ 0.60 garantizado) + model card. Es defendible ante el stakeholder sin "mover la portería".

---

## 1. Diagnóstico Avance 4

### 1.1 Ranking de los 6 (7) modelos — PASTIS-R fold-4 (n_val=482)

| # | Modelo | mIoU | F1-macro | Pixel-acc | mIoU grouped-6 | Tipo |
|---|--------|------|----------|-----------|----------------|------|
| 1 | **TSViT-pheno** | **0.625** | **0.750** | 0.876 | (no entrenado) | temporal |
| 2 | TSViT-base | 0.622 | 0.747 | 0.872 | — | temporal |
| 3 | U-TAE | 0.474 | 0.609 | — | 0.500 | temporal |
| 4 | AnySat-fast | 0.446 | 0.572 | 0.750 | 0.591 | proxy (sustituye Swin-UNETR del plan) |
| 5 | DeepLabv3+ | 0.271 | 0.386 | 0.674 | 0.468 | spatial-only |
| 6 | U-Net | 0.242 | 0.346 | 0.692 | 0.452 | spatial-only |
| 7 | SegFormer | 0.232 | 0.342 | — | 0.237 | open-vocab |

Fuente: `reports/segmentation/metrics/model_comparison_avance4_*.parquet`, `notebooks/segmentation/Avance4.Equipo17.ipynb`. Métricas pixel-level con `ignore_index=void(19)` (`ml/eval/dense_metrics.py:10,92-96`).

### 1.2 Por qué TSViT gana
El encoder temporal separa cultivos que en una sola fecha se confunden por su **calendario fenológico** (8 cereales que sólo se distinguen por curva de crecimiento). Brecha TSViT/DeepLabv3+ ≈ 2.3× cuantifica el aporte de la serie temporal.

### 1.3 Causa raíz de la brecha = desbalance
`pixel-acc 0.876` (domina cereales, mayoritarios) vs `F1-macro 0.750` (penaliza LEGUMES/ROOT_CROPS/OILSEEDS, ~2 clases y poca área). **No se reportaron métricas per-clase** en Avance 4 — primer hueco a cerrar para calibrar pesos con criterio.

### 1.4 Palancas de alto ROI que se dejaron sin tocar (hallazgos del código)
1. **`class_weights` cableado a `None`**: el loss `build_dice_ce_loss` YA acepta pesos (`ml/models/deeplabv3plus.py:171,209-210,241`) pero el caller nunca los pasa (`ml/train/train_segmentation.py:1121-1126`). **Free win.**
2. **Sin augmentation alguna**: `PASTISSegmentationDataset.__getitem__` devuelve S2 crudo normalizado (`ml/data/pastis_seg_dataset.py:373-411`). El patrón flip/rot90 consistente x-máscara YA existe para U-TAE (`ml/tune/optuna_segmentation.py:326-337`), portable.
3. **DOY real nunca alimentado**: TSViT corre `model(x)` sin fechas (`ml/train/train_segmentation.py:692-698`) ⇒ cae al PE **ordinal** de respaldo (`ml/models/tsvit_wrapper.py:367-370`); el dataset no emite fechas. El wrapper SÍ soporta DOY. Bug latente: el 0.6253 se midió sin fechas absolutas.
4. **grouped-6 HCAT nunca entrenado nativo**: el flag `--target hcat6` ya está cableado (`train_segmentation.py:1425,1510`, LUT en `pastis_seg_dataset.py:264-289`); hoy grouped sólo se evalúa post-hoc.
5. **`dropout=0.0` nunca testeado** (`ml/models/tsvit_wrapper.py:260`); **early-stopping disponible pero apagado** (`train_segmentation.py:1303`, patience=0); best epoch = 28/30 ⇒ el modelo aún mejoraba al cortar.
6. **Optuna mal configurado** (ver 0.).

### 1.5 Limitaciones metodológicas
- **Validación single-fold** (sólo fold-4): número no defendible sin CV espacial; riesgo de overfit a la partición.
- **`ml/ensemble/` está VACÍO**: los 4 ensambles obligatorios de EPIC6 (entregable del Avance 5) están 100 % por construir.

---

## 2. Palancas de mejora, ordenadas por ROI (lifts ajustados honestos)

> Convención: los lifts **se solapan, no se suman** (casi todas atacan el mismo desbalance). El total realista combinado está en §3, no es la suma de filas.

| # | Palanca | Lift F1 / mIoU (honesto) | Esfuerzo | Cómputo | Riesgo | Evidencia |
|---|---------|--------------------------|----------|---------|--------|-----------|
| **P1** | **Class-weighted Dice+CE** (effective-number/inverse-freq, clip a [0.5,4]) — cablear lo que ya existe | **+0.02–0.04 / +0.01–0.03** | S (3–5 h) | 1 run ~3 GPU-h L4 | bajo (revierte con `w=None`) | `deeplabv3plus.py:171,241`; caller `train_segmentation.py:1121` |
| **P2** | **Augmentation geométrica** flips H/V + rot90 (SIN jitter temporal) | +0.02–0.03 / +0.02–0.04 (solapa P1) | M (5–7 h) | mismo run de P1 | bajo (label-invariante) | `pastis_seg_dataset.py:373-411`; patrón en `optuna_segmentation.py:326-337` |
| **P3** | **+40 épocas con `patience=8`** (best=28/30 ⇒ no convergió) | +0.01 mIoU | S (2 h) | +50 % tiempo del run | bajo | `train_segmentation.py:1146-1153,1303` |
| **P4** | **Voting soft top-3 temporales** (TSViT-pheno + TSViT-base + U-TAE) — **obligatorio rúbrica** | +0.01–0.02 / +0.01 (modelos correlacionados) | S (5–7 h) | 0 GPU (logits cacheados) | medio (U-TAE puede degradar) | `ml/ensemble/` vacío; `dense_metrics.py:36` |
| **P5** | **Blending Optuna** (pesos + bias logit per-class) — **obligatorio rúbrica** | +0.01–0.02 generalizable (no el +0.04 si se valida en hold-out) | M (8–10 h) | 0 GPU | **overfit a fold-4** si no hay hold-out | `ml/ensemble/blending.py` (crear) |
| **P6** | **Entrenar grouped-6 HCAT nativo** (criterio agronómico complementario, NO reemplazo del target) | mIoU **0.58–0.68**, F1 0.74–0.82 (NO 0.70-0.74 "con holgura") | M (6–9 h)* | 1 run ~3 GPU-h L4 | medio (leerse como "bajar liston") | `train_segmentation.py:1425,1510` |
| **P7** | **HPO Optuna rediseñado** (ASHA/Hyperband + espacio de desbalance: class_weight, label_smoothing, dropout, dice/ce) — reemplaza el no-op | +0.025–0.045 / +0.02–0.045 | M-alto (12–16 h) | ~10–13 GPU-h L4 (con ASHA) | medio | `optuna_segmentation.py:76,461,523` |
| **P8** | **Post-proceso por parcela (CC-derived) + TTA D4** | +0.02 / +0.015 (CC-derived, prod-honesto) | M (8–10 h) | 0 GPU (CPU) | **oracle-instance = leakage**: sólo cota superior etiquetada | `pastis_loader.py:136`; `segmentation_inference.py:111,214`; pureza parcela=1.0 |
| **P9** | **DOY real** en TSViT (parsear fechas, reentrenar) | +0.015–0.04 mIoU (**signo incierto**) | M (6–8 h) | 1 reentreno (16 h L4 / ventana H100) | alto: reentreno from-scratch, puede empeorar | `train_segmentation.py:692-698`; `tsvit_wrapper.py:367-370` |
| **P10** | **3-fold CV espacial** (val=1,3,5) del config ganador → número defendible con CI | 0 (robustez, no lift) | M (6–8 h) | ~48 GPU-h L4 | coste GPU | `train_segmentation.py:283,1368-1369` |

\* P6 para **tsvit-pheno** exige **colapsar los prototipos fenológicos a K=6** (`pheno_semantic_branch.py`): no es reuso directo. Atajo barato: correr hcat6 sobre **tsvit-base** (sin rama pheno), que sí reusa la infra end-to-end.

### Descartados / diferidos
- **Focal + "Optuna-de-loss" full-epoch**: 6–10 h H100 que colisionan con la ventana V3 (Gemma 4 LoRA, irrevocable). Diferir.
- **Fusion-input AlphaEarth/DINOv3 como canales extra**: 12–16 GPU-h + alineación de grilla riesgosa (16 px vs 128 px) + roba ventana a Gemma 4. **Único camino real a mIoU 0.70**, pero **no cabe** antes del 7-jun → Avance 6 condicionado a reactivar H100.
- **EMA/SWA, jitter temporal, CRF/morfología, Optuna multi-objetivo NSGA-II**: nice-to-have post-7-jun.

---

## 3. Lift combinado realista (lo que de verdad se logra)

- **Modelo individual TSViT-pheno flat-18** (P1+P2+P3, una corrida combinada): **F1 0.77–0.80 · mIoU 0.65–0.68**.
- **+ ensambles post-hoc** (P4+P5, modelos correlacionados): **F1 ~0.78–0.80 · mIoU ~0.66–0.68**.
- **flat-18: F1 roza 0.80 (borde superior), mIoU 0.70 NO se alcanza.** El cierre de mIoU exige reentrenos (DOY, fusion-input) fuera del calendario del Avance 5.
- **grouped-6 HCAT (P6)**: F1 0.74–0.82 · mIoU 0.58–0.68 — alcanza el bar en F1 pero **sin holgura**; es métrica agronómica complementaria, **no** sustituto del target.

**Conclusión**: la mejora real es honesta y significativa (sube el macro atacando minoritarias, añade los ensambles obligatorios), pero **no cruza el 0.80/0.70 binario en flat-18**. Eso obliga a una compuerta de producción por criterio (§5), no por umbral.

---

## 4. Plan de ejecución (2-jun → 7-jun, y → 14-jun)

Restricciones: **H100 PARKED** (0 h), V2 (1-3 jun) casi cerrada, **V3 (4-6 jun) reservada a Gemma 4 LoRA** (irrevocable). ⇒ **toda la segmentación va a L4 GCP spot vía Pub/Sub + Cloud Run worker** (regla global 9, training nunca síncrono).

| Día | Acción | Owner | Salida |
|-----|--------|-------|--------|
| **Mar 2** | Tabla per-clase recall/precision/F1 de TSViT desde la confusion ya guardada (`dense_metrics.py:144`, portar `hcat_grouping.py:215`). Identifica qué minoritarias castigan el macro. | Isaac | figura + parquet |
| **Mié 3** | Cablear **P1 class-weights** + **P2 flips/rot90** (3 funciones: `build_and_train`→`train_segmentation`→`build_dice_ce_loss` + helper de frecuencias polars sobre folds 1-3, sin leakage). Test unitario aug. | Isaac | rama `feature/E5-US029-tsvit-tuning` |
| **Mié 3 → Jue 4** | Lanzar **1 run combinado** P1+P2+P3 (40 ep, patience=8) en L4 spot. Comparar vs baseline en MLflow (`data_version`+`code_version`). | Isaac/Arthur | best.pt + métricas |
| **Jue 4** | **P6**: lanzar tsvit-base `--target hcat6` (reusa infra). En paralelo (CPU): empezar `ml/ensemble/export_probs.py` (logits OOF a `.npy`). | Isaac | parquet hcat6 + probas |
| **Vie 5** | **P4 Voting** + **P5 Blending** sobre logits cacheados (0 GPU). Hold-out honesto para el bias (mitad de fold-4 o fold-5 re-evaluado). Stubs documentados de **Stacking**(+Gemma4) y **Bagging XGB+AlphaEarth** para cumplir forma de rúbrica EPIC6. | Isaac | `ml/ensemble/{voting,blending,stacking,bagging}.py` |
| **Sáb 6** | **≥4 gráficas interpretadas** (rúbrica Avance 5): curvas entrenamiento, per-class IoU, matriz de confusión, comparativa modelo-vs-ensamble, convergencia Optuna. Tabla de selección del modelo final. | Isaac | notebook `Avance5.Equipo17.ipynb` ejecutado con outputs |
| **Dom 7** | Consolidar notebook + `make check` + `make notebooks-check` (papermill). **Avance 5 entregado.** | Equipo | entregable |
| **8-14 jun** | (Si H100 se reactiva) **P9 DOY** con ablación corta de 5 ep antes de comprometer; **P10 3-fold CV** del config ganador para el número defendible con CI. Model card + árbol de decisión (§5). **Avance 6.** | Isaac/Arthur | conclusiones producción |

Cómputo total Avance 5: **~9–12 GPU-h L4 spot** (cabe en presupuesto operativo ~$115/mes). No requiere H100 ni rompe la reserva de Gemma 4.

---

## 5. Compuerta de aceptación para producción (Avance 6)

**No se decide sobre un único número.** Árbol de decisión documentado:

```
GO            si grouped-6 HCAT  F1>=0.80 y mIoU>=0.70  (criterio agronómico que el stakeholder consume)
GO-condicional si flat-18 F1>=0.78 y el ensamble grouped-6 >=0.68 mIoU
                 → despliegue con alcance recortado a 6 grupos + disclaimer + revisión humana en clases de baja confianza
NO-GO         si grouped-6 < 0.68 mIoU
                 → fallback a baseline XGB+AlphaEarth (F1>=0.60 garantizado) como capa de respaldo + escalado humano
```

Componentes obligatorios del paquete de decisión:
1. **Doble taxonomía reportada lado a lado**: flat-18 (techo científico/rúbrica EPIC5) **y** grouped-6 HCAT (criterio de negocio). Nunca presentar grouped-6 como "target cumplido" sin el flat-18 al lado — evita la lectura de *gaming* ante Dr. Camacho.
2. **Error analysis per-clase** (recall/precision/F1 + soporte) y matriz de confusión (figuras ya existen en `ml/eval/avance4_figures.py`).
3. **Model card**: dataset (PASTIS-R, **gap de k-fold CV reconocido**), limitaciones (**sin robustez testeada a nubes/gaps** — el submuestreo equiespaciado `pastis_seg_dataset.py:179-197` no maneja huecos por nubes), uso previsto, métricas por taxonomía, número con CI (P10).
4. **Despliegue**: inferencia temporal > 2 s ⇒ **Pub/Sub + Cloud Run GPU L4 worker** (regla global 9), nunca síncrono. Monitoreo **Evidently** sobre drift de distribución de clases predichas.
5. **Capa de respaldo**: baseline XGB+AlphaEarth como red de seguridad (F1 ≥ 0.60) si el segmentador denso no alcanza confianza en una AOI.

**Mensaje honesto al stakeholder**: el modelo temporal mejora de forma real y es la mejor arquitectura; alcanza calidad de producción **en la taxonomía agronómica de 6 grupos** (la que aporta valor de decisión: trigo-vs-trigo no lo aporta), y se despliega con una arquitectura de respaldo + gobernanza. La discriminación fina de 18 clases queda como techo científico en mejora continua (DOY, fusion AlphaEarth) post-presentación.

---

## 6. Decisiones que requieren al equipo / stakeholder

1. **Estrategia de aceptación** (§5): ¿se adopta el GO-condicional con doble taxonomía (recomendado), o se persigue 0.80/0.70 en flat-18 a toda costa (alto riesgo de calendario y de robar ventana a Gemma 4)?
2. **Reactivación de H100**: si se reactiva, habilita P9/P10 y el experimento de fusion-input para Avance 6. Si no, todo va a L4 spot y la fusion se difiere.
3. **Stacking con Gemma 4**: depende del LoRA de V3 (4-6 jun). Si Gemma 4 se retrasa, el stacking obligatorio queda como stub documentado — confirmar que es aceptable para la rúbrica.

## 7. Riesgos

- **Velocidad L4 spot no benchmarkeada** (AMP overhead sin medir): el "~3 h/run" es estimación; si va CPU-bound o hay preempción, se duplica. Mitigar: 1 run combinado, no 4 separados.
- **Overfit a fold-4**: todo lift se mide en un fold. Validar en 3-fold (P10) antes de declarar producción; reportar grouped-6 como sanity-check.
- **Carga sobre 1 ML scientist** en 5 días: el alcance completo (3 runs + 4 ensambles + model card) es ajustado. Priorizar P1+P4+P5 (lo seguro, sin H100); P6/P7 si hay holgura.

---

## 8. Estado de implementación (2-jun) — P1 + P2 ya codificadas

Las palancas de mayor ROI (**P1 class-weights** + **P2 augmentation**) ya están **cableadas en el working tree** (listas para correr en L4; el commit lo hace el equipo):

- **`ml/data/pastis_seg_dataset.py`**: `apply_synchronized_augment()` (D4 flips/rot90 imagen-máscara) + flag `augment` (solo train) + método `class_pixel_counts()` (cuenta píxeles por clase leyendo solo el TARGET, sin S2).
- **`ml/train/train_segmentation.py`**: `train_segmentation(..., class_weights=...)` cableado a `build_dice_ce_loss` (que ya los soportaba); helpers `_class_weights_from_counts` (effective-number Cui 2019 / inverse-freq, clip [0.5,4], ausentes neutras) y `_resolve_class_weights` (cómputo cacheado en `reports/segmentation/metrics/class_counts_<target>_<folds>.json`, **solo folds de train, sin leakage**); `build_and_train(..., augment, class_balance, class_balance_beta)`; flags CLI `--augment --class-balance --class-balance-beta`.
- **`ml/eval/dense_metrics.py`**: `DenseConfusionAccumulator.per_class_metrics()` → tabla recall/precision/F1/IoU/soporte por clase (calibra pesos + alimenta el model card del Avance 6).
- **`tests/ml/test_pastis_seg_augment.py`**: 7 tests (consistencia x-máscara, derivación de pesos, métricas per-clase). Verde + 41 tests previos sin regresión + ruff limpio.
- **Notebook del entregable** en carpeta propia `notebooks/best_model/` (separada de `segmentation/`): `Avance5.Equipo17.ipynb`, generado reproduciblemente por `scripts/build_avance5_notebook.py`. Esqueleto Colab-ready (runtime GPU, `RUN_TRAINING=True`, `PASTIS_ROOT`); orquesta el run mejorado, comparativa baseline-vs-mejorado, tabla per-clase y compuerta de producción.

**Comando del run combinado (P1+P2+P3) en L4** (lanzar vía Pub/Sub + Cloud Run worker, no síncrono):

```bash
poetry run python -m ml.train.train_segmentation \
  --model tsvit-pheno --target semantic18 \
  --epochs 40 --patience 8 --batch-size 4 --device cuda \
  --augment --class-balance effective --class-balance-beta 0.9999 \
  --run-name alt-tsvit-pheno-cw-aug-v1
```

Comparar en MLflow contra el baseline `alt-tsvit-pheno-v1` (tags `data_version`+`code_version`); si supera, ese checkpoint entra al Voting/Blending (P4/P5) y, si hay holgura, al 3-fold CV (P10).

> Pendiente de decisión del equipo antes de gastar más GPU: confirmar reactivación H100 (habilita P9/P10) y aceptabilidad del stub de Stacking-con-Gemma4 si V3 se retrasa.

---

*Plan derivado de análisis multi-agente con verificación adversarial sobre el código y métricas reales del repo. Lifts = ajustados honestos post-escrutinio. Última actualización: 2026-06-02.*
