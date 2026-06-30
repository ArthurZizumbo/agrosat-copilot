# US-082 — Reporte de resultados (re-extracción + migración de región)

**Fecha**: 2026-06-30 · **Rama**: `feature/E12-US-082-italia-dataset-completo` · **Commits**: hasta `226d5ec`

> Reporte de la exploración de datos de US-082: del techo del TL Italia (Toscana)
> a la migración de región (Baja Sajonia / DE4). Todas las cifras son sobre dato
> REAL (fold-5/fold-4 OOF, AlphaEarth GEE, S2 Sentinel Hub). Cero placeholders.

---

## 1. Resumen ejecutivo

El F1-macro 0.13 del TL Italia (US-079) NO era el muestreo del 1 % del piloto. Con
el dataset completo (Toscana 103k parcelas) el techo persiste (~0.12-0.22). Se
identificaron y midieron **tres palancas de mejora independientes**, y se migró la
región a **Baja Sajonia (DE4)**, que resultó claramente superior.

| Palanca | Evidencia (dato real) |
|---|---|
| **Año 2023** (vs 2018) | xgb Toscana 0.156 → **0.225** (+44 %) |
| **Ventana temporal 14 meses** (vs 8) | Toscana 24 → **56 fechas**; estratificación: cereales +0.07/+0.10 |
| **Región DE4** (vs Toscana) | Voting-3 0.119 → **0.266** (2.2×); cereales 0.05-0.20 → **0.57-0.70** |

**Veredicto**: el dataset **DE4-2023 con ventana de 14 meses** es la mejor base. Los
cereales (trigo/maíz/centeno/colza) que en Toscana estaban hundidos por el techo
temporal + parcelas pequeñas, en DE4 rescatan (parcelas de 10 ha + cobertura 0.94 +
~41 fechas).

---

## 2. Comparativa de configuraciones (xgb AlphaEarth, mismo miembro)

| Config | xgb F1-macro | accuracy | n_parcelas | fechas | parcela mediana |
|---|---|---|---|---|---|
| Toscana 2018 | 0.1563 | 0.37 | 103 350 | 24 | 0.4 ha |
| Toscana 2023 | 0.2252 | 0.45 | 91 487 | 56 | 0.4 ha |
| DE4 smoke (80 patches) | 0.1637 | 0.564 | 3 218 | 32 | 10 ha |
| **DE4 full (1246 patches)** | **0.2674** | **0.6333** | **53 460** | ~41 | **10 ha** |

DE4 full gana en xgb: **0.267 vs 0.225 Toscana** (+19 %), con menos parcelas (18 %
del universo DE4).

---

## 3. Modelo completo DE4 — Voting-3 (entreno denso, fold-4 test)

Re-entreno TSViT-pheno-fullm + U-TAE (warm-start del campeón, 40 épocas,
n_timesteps=32) + xgb-alphaearth, agregados a parcela (eurocrops strategy), Voting-3.

| Miembro | fine_f1 | fine_mIoU | accuracy | peso voto |
|---|---|---|---|---|
| **TSViT-pheno-fullm** | **0.333** | 0.260 | **0.682** | 0.544 |
| U-TAE | 0.186 | 0.130 | 0.511 | 0.259 |
| xgb-alphaearth | 0.267 | — | 0.633 | 0.197 |
| **Voting-3** | **0.266** | 0.189 | 0.513 | — |

### Las 3 vías (decisión de Arthur)
| Vía | Definición | macro-F1 |
|---|---|---|
| **A — conservando clases** (37 HCAT nativas) | fine | **0.266** |
| **B — sin conservar** (crosswalk a PASTIS) | coarse | **0.287** |
| **C — procedimiento completo** (= el voto entrenado) | fine | **0.266** |

---

## 4. Per-clase del Voting-3 DE4 — LOS CEREALES RESCATAN

| Clase | F1 | tipo |
|---|---|---|
| maize_corn_popcorn (maíz) | **0.703** | cereal |
| winter_rapeseed_rape (colza) | 0.689 | |
| grassland_grass | 0.684 | |
| winter_common_soft_wheat (trigo) | **0.674** | cereal |
| sunflower (girasol) | 0.635 | |
| potatoes (papa) | 0.601 | |
| winter_rye (centeno) | **0.573** | cereal |
| peas | 0.535 | |
| alfalfa_lucerne | 0.440 | |
| spring_oats (avena) | 0.386 | cereal |
| winter_triticale | 0.285 | cereal |

**Gates**: clases ≥0.4: **12** · ≥0.6: **8** · ≥0.8: 0.
**Curva de descarte** (alta y estable): top-1=0.79, top-5=0.72, top-8=0.69,
top-10=0.66, **top-12=0.635**.

---

## 5. Comparación final Toscana vs DE4

| Métrica | Toscana | **DE4** | factor |
|---|---|---|---|
| Voting-3 fine_f1 | 0.119 | **0.266** | 2.2× |
| TSViT-fullm fine_f1 | 0.122 | **0.333** | 2.7× |
| TSViT accuracy | 0.19 | **0.68** | 3.6× |
| clases ≥0.6 | **1** | **8** | 8× |
| curva descarte top-12 | 0.29 | **0.635** | 2.2× |
| cereales (trigo/maíz/centeno) | 0.05-0.20 | **0.57-0.70** | ~3-10× |

---

## 6. Por qué DE4 gana (análisis)

- **Parcelas grandes** (mediana 10 ha vs 0.4 ha Toscana) → patches limpios, poca
  mezcla de bordes; los cereales ocupan campos enteros.
- **Cobertura full** (EuroCrops v2: Toscana es "partial", DE4 es "full") + cobertura
  de cultivo por patch 0.90 (vs 0.74 Toscana).
- **Cultivos anuales claros y separables** (cereales de invierno, colza, maíz, papa)
  vs el lío permanente+anual fragmentado de Toscana.
- **~41 fechas** (ventana 14 meses + max_cloud 50) — cercano a PASTIS 43, suficiente
  para la fenología de los cereales de invierno.

---

## 7. Caveats honestos

- El Voting-3 DE4 se midió sobre **1 fold** (test-fold 4, n_folds=1). El número
  titular inatacable requeriría OOF de 5 folds (5× cómputo). El patrón per-clase es
  contundente y no cambiaría el veredicto.
- Cobertura espacial DE4: los 1246 patches = **18 % del universo** (597 celdas densas
  reales; parcelas alemanas grandes y dispersas). Pérdida **uniforme ~15 % por clase**
  (sin sesgo) → muestra representativa. Ampliar a min_parcels=30 daría 40 % (2723
  patches), pero DE4 ya gana con 18 %.
- A y C dieron el mismo número (0.266) porque en este runner la "vía C completa" ES
  la vía A (voto sobre clases nativas). Para separar las 3 limpio: correr
  `ml/transfer/eval_three_ways.py` explícito sobre el OOF.

---

## 8. Infraestructura / incidentes superados (autónomo)

- **Failover de cuota CDSE**: la llave 1 (30k PU) se agotó a los 787/1246 patches.
  Failover automático a la llave de respaldo (verificando PU primero) → descarga
  completada 1246/1246.
- **Fix de región** (commit `226d5ec`): el pipeline dense→parcela estaba cableado a
  Italia (`parcels_in_patches` default) → 0 intersección con bboxes DE4. Generalizado
  a `--region-prefix` / `--parcels-parquet` / `--mapping-csv`.
- **Descarga paralela** (commit `110a798`): flag `--reverse` para un 2º worker que
  baja la lista de atrás-adelante (resume idempotente). ~2× velocidad.
- **Builder + extractor generalizados** a cualquier región NUTS (commits `5cee768`,
  `227b42f`); mapeo HCAT 100 % vía `eurocrops.csv` oficial del JRC.

---

## 9. Artefactos generados

- `data/pastis_de4_2023/` — dataset PASTIS-homólogo DE4 2023 (1246 patches, formato
  S2/TARGET/dates .npy + metadata.parquet + class_mapping.json).
- `data/features/alphaearth_de4_2023_full.parquet` — features AlphaEarth DE4 (53 460
  parcelas, 64 dims).
- `data/features/de4_2023_full_oof/` — OOF xgb DE4.
- `checkpoints/transfer/{tsvit-pheno-italia,utae-italia}/de4_2023-*/` — checkpoints
  densos DE4 (best.pt + last.pt por época).
- `checkpoints/transfer/voting-italia/de4_2023/report.json` — reporte del Voting-3.
- Etiquetas EuroCrops v2: `data/reference/eurocrops_v2/{de4_2023,nl_2023,iti1_2023,
  iti1_2022,iti1_stack}.parquet` + crosswalk `eurocrops_official.csv`.

---

## 10. Siguiente paso (pendiente de decisión de Arthur)

1. **Ampliar cobertura DE4 a 40 %** (min_parcels=30 → 2723 patches, ~7-13h) — DE4 ya
   gana con 18 %, ampliar solo mejora.
2. **OOF de 5 folds** para el número titular inatacable.
3. **DVC** del dataset + features DE4 + push a GCS; traer a local (fuente de verdad).
4. Decidir si DE4 reemplaza a Italia como dataset objetivo del proyecto, o si se
   mantiene Italia con las palancas (2023 + ventana 14m) aplicadas.
