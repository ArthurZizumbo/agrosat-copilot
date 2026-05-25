# US-022-cierre — Plan canonico — FarSLIP downstream: extract real + integracion FE + lift baselines

**Status**: Planning (canonico) · creado 2026-05-24 · listo para Fase 3 (coding)
**Epic**: E3/E5/E6 (Feature Engineering + Modelos + Ensembles)
**Avance**: post-A3 — alimenta A4 (24-may) + A5 (31-may) + A6 (7-jun)
**Sprint**: S7 (1-jun → 7-jun)
**Rama prevista**: `feature/E5-US-022-cierre-farslip-fe` (alias corto `us-022-cierre`)
**Owner**: Arthur (P1/P4 extract + MLflow Registry) + Isaac (P2/P3 FE + baselines)
**SP estimados**: **8 SP** (P1=3, P2=2, P3=2, P4=1)
**ADR de referencia**: [`ADR-007-farslip-fidelity-paper.md`](../decisions/ADR-007-farslip-fidelity-paper.md)
**Plan padre**: [`us-022-c.md`](us-022-c.md) (US-022-c cerrada 2026-05-24 con P1/P2/P3/P4/P5 OK pero gates B-3/B-4/B-5 parciales por placeholder)
**Backlog origen**: pendientes formales US-022-c §"Limitacion honesta" (placeholder seeded bloquea gate mIoU)

> **Alcance**: convierte los embeddings sinteticos seeded (`embeddings_italy_v1.parquet`) en embeddings
> reales del student FarSLIP epoch_2 + integra el bloque `farslip_emb_*` (512 dims) al pipeline
> de feature engineering existente + mide lift baseline XGBoost/LightGBM con vs sin FarSLIP. Cierra
> los gates B-3, B-4, B-5 que quedaron parciales en US-022-c.

---

## 1. Justificacion y reencuadre

US-022-c entrego el training FarSLIP completo en L4 us-west4-a (47 min, 4 epochs, best `epoch_2 loss=2.357`)
y los 4 checkpoints en GCS (`gs://agrosat-artifacts-dev/farslip/v1/`). El extract local genero un
parquet `data/farslip/embeddings_italy_v1.parquet` (30173 × 514) versionado en DVC, **pero**
`ml/farslip/extract_embeddings.py:_project_parcels_to_embeddings` es un placeholder determinista
(`torch.randn` seeded) — NO usa el student real para inferir embeddings de las imagenes Sentinel-2.

Esto deja 3 gates de US-022-c en estado parcial:

- **B-3** (mIoU PASTIS-R gate `mIoU_farslip - mIoU_remoteclip >= +0.05`): imposible medir con
  embeddings sinteticos. Documentado como "negativo aceptable" via R6 pero el equipo decidio cerrar
  el ciclo formalmente.
- **B-4** (shape `(85951, 514)` + tag git `farslip-embeddings-italy-v1`): el parquet actual tiene
  shape `(30173, 514)` (manifests italianos) en vez de `(85951, 514)` (parcelas features fusionadas
  de US-016). Tag git no creado.
- **B-5** (MLflow Model Registry `farslip-clip-italy-v1@Production`): checkpoint epoch_2 subido a
  GCS Artifacts pero NO registrado formalmente.

US-022-cierre implementa el extract real (P1), integra al feature engineering (P2), mide el lift
downstream (P3) y cierra los gates formales (P4).

---

## 2. Criterios de aceptacion (metricas verificables)

### 2.1 P1 — Extract real FarSLIP embeddings (3 SP)

Reemplaza el placeholder `_project_parcels_to_embeddings` por inferencia real student + crops Sentinel-2.

| AC | Criterio | Como verificar |
|----|----------|----------------|
| AC-P1-1 | `_project_parcels_to_embeddings` carga crops de `manifest_path.parent / "crops" / f"{crop_id}.tif"`, los resizea a 224x224, normaliza uint16/10000 y forward por `model.vision_model(pixel_values)` → pooler_output 512-dim | inspeccion codigo + grep `pooler_output` en `ml/farslip/extract_embeddings.py` |
| AC-P1-2 | Test unitario `test_extract_embeddings.py::test_extract_real_not_placeholder` valida que dos seeds distintos sobre el mismo input producen el **mismo** embedding (ya no es randn) | `pytest tests/ml/test_extract_embeddings.py -k test_extract_real_not_placeholder` exit 0 |
| AC-P1-3 | Output `data/farslip/embeddings_italy_v2.parquet` shape `(30173, 514)` con columnas `parcel_id_str` + `parcel_id` + `year` + 512 dims, generado con `student_epoch_2.safetensors` | `polars.read_parquet(...).shape == (30173, 514)` |
| AC-P1-4 | Embedding norms `||v||_2` distribuidos no triviales: `0.5 < mean(norms) < 50` (sanity check vs randn que daria mean ~22.6) | script de validacion + log en MLflow `extract_real_run` |
| AC-P1-5 | DVC track + push: `data/farslip/embeddings_italy_v2.parquet.dvc` commiteado + blob en `gs://agrosat-dvc-remote/files/md5/<hash>/...` | `dvc push` reporta `1 file pushed` + `gcloud storage ls` |

### 2.2 P2 — Integracion FE (`farslip_emb_*` en fusion.py) (2 SP)

Agrega el bloque `farslip_emb_*` (512 dims) al fusion.py existente para que el feature parquet
fusionado incluya FarSLIP junto a AlphaEarth + S1 + S2 + pheno_text.

| AC | Criterio | Como verificar |
|----|----------|----------------|
| AC-P2-1 | `ml/features/fusion.py` agrega bloque `_attach_farslip_block(df, farslip_parquet, on=["parcel_id_str","year"])` que joinea las 512 cols | grep en fusion.py + test unitario `test_fusion.py::test_attach_farslip_block` |
| AC-P2-2 | `EXPECTED_COL_COUNT_WITH_FARSLIP = 573 + 512 = 1085` constante exportada | grep en fusion.py |
| AC-P2-3 | Notebook 02 `02_features_fusion.ipynb` ejecutado con bloque FarSLIP activo (flag `--with-farslip`) genera parquet `features_fused_v2.parquet` con 1085 cols | `polars.read_parquet(...).width == 1085` |
| AC-P2-4 | Tests `test_fusion.py` pasan con cobertura del nuevo bloque >= 80% | `pytest tests/ml/test_fusion.py --cov=ml.features.fusion --cov-fail-under=80` |

### 2.3 P3 — Lift baselines downstream (2 SP)

Mide el lift del bloque FarSLIP sobre los baselines tabulares existentes (XGBoost + LightGBM) via
ablation: `full_sin_farslip` vs `full_con_farslip` en spatial K-fold.

| AC | Criterio | Como verificar |
|----|----------|----------------|
| AC-P3-1 | Notebook nuevo `notebooks/features/05b_farslip_lift_baseline.ipynb` ejecuta XGBoost + LightGBM con spatial K-fold sobre `features_fused_v2.parquet`, dos feature sets (`base`, `base + farslip_emb_*`) | inspeccion notebook + ejecucion `make notebooks-check` |
| AC-P3-2 | `reports/baseline/farslip_lift/comparison.parquet` con 4 filas (XGB-base, XGB-farslip, LGB-base, LGB-farslip) + columnas `f1_macro`, `f1_std`, `delta_vs_base` | `polars.read_parquet(...).shape == (4, 5+)` |
| AC-P3-3 | Resultado documentado en `docs/us-resolved/us-022-cierre.md` §"Resultados lift FarSLIP": tabla + interpretacion (positivo, negativo o neutro — TODOS aceptables siempre que esten reportados) | grep en md |
| AC-P3-4 | MLflow runs registrados con tags `data_version` (DVC short hash del v2 parquet) + `code_version` (git sha) + `feature_set` ("base"\|"base+farslip") | UI MLflow muestra 4 runs |

### 2.4 P4 — Cierre formal gates B-3/B-4/B-5 US-022-c (1 SP)

Cierra los gates formales que quedaron en estado parcial.

| AC | Criterio | Como verificar |
|----|----------|----------------|
| AC-P4-1 | Tag git `farslip-embeddings-italy-v1` creado sobre el commit del .dvc del v2 parquet (no v1, que era placeholder) | `git tag -l 'farslip-embeddings-italy-v1'` no vacio |
| AC-P4-2 | MLflow Model Registry: `farslip-clip-italy-v1@Production` apunta al checkpoint `student_epoch_2.safetensors` (path GCS + artifact_uri) | UI MLflow Registry stage Production + comando `mlflow models get-model-version-by-alias --name farslip-clip-italy-v1 --alias Production` |
| AC-P4-3 | `docs/l4_log.md` con entrada: Job ID `vm:agrosat-farslip-trainer-dev:us-west4-a:2026-05-24T21:18`, Image SHA `e9860b7`, Duracion `46.7 min`, Costo `~$1.20 USD`, MLflow run `train-full-farslip-l4-2026-05-24` | grep en l4_log.md |
| AC-P4-4 | Gate B-3 mIoU PASTIS-R: notebook `04_farslip_eval_pastis.ipynb` ejecutado end-to-end con embeddings v2 reales — resultado positivo/neutro/negativo aceptable, MUST estar documentado | `nbformat` valida outputs poblados + cell §"mIoU comparison" tiene tabla |

---

## 3. Fuera de scope (explicito)

- **NO** reentrenar FarSLIP (los 4 checkpoints L4 estan en GCS, usamos `student_epoch_2.safetensors`)
- **NO** re-implementar dataset.py o train.py (cerrados en US-022-c)
- **NO** tocar H100 / Gemma 4 / Qwen3-VL (vive en US-019-2 + US-026)
- **NO** subir a 85951 parcelas (B-4 original): nos quedamos con 30173 italianos para no bloquear
  por dependencia con US-024 (features_fused_v1 con 85951 parcelas). Documentamos el desvio.

---

## 4. Dependencias

- ✅ US-022-c cerrada (4 checkpoints en GCS, embeddings v1 placeholder en DVC, infra VM operativa pero apagada)
- ✅ DLVM imagen `pytorch-2-9-cu129-ubuntu-2204-nvidia-580` validada
- ✅ Dataset `data/farslip_pairs/` local + en GCS
- ⏳ Acceso MLflow Cloud Run (deploy del wrapper Secret Manager — ya operativo desde US-022-c P1)

---

## 5. Presupuesto computo

- P1 extract real: CPU + GPU local (RTX 4070), 30173 forward passes batch=64 ≈ ~5 min. **$0 USD.**
- P2 FE integration: local, $0 USD.
- P3 baselines: local CPU (XGBoost+LightGBM spatial KFold), ~10 min. **$0 USD.**
- P4 cierre: MLflow Cloud Run query + tag git + edit md, $0 USD.

**Total US-022-cierre: $0 USD** (todo local, no L4/H100).

---

## 6. Riesgos y mitigaciones

| ID | Riesgo | Prob | Impacto | Mitigacion |
|----|--------|------|---------|------------|
| R1 | Extract real produce embeddings degenerados (collapse a constante) | baja | alto | Sanity check AC-P1-4 + visual UMAP de los 512 dim por clase CAP; si collapsa, fallback a `epoch_1.safetensors` o `epoch_0` |
| R2 | Lift FarSLIP downstream < 0 (FarSLIP empeora baselines) | media | bajo | **Aceptable y reportable** (mismo R6 que US-022-c). Documentar como hallazgo + hipotesis (CLIP distillation no transfiere a clasificacion tabular CAP); FarSLIP queda como base learner opcional ensemble |
| R3 | mIoU PASTIS gate B-3 sigue negativo con embeddings reales | media | bajo | Resultado negativo aceptable per R6 US-022-c. Documentar honestamente; FarSLIP no se usa en EPIC 6 stacking si no aporta |
| R4 | MLflow Cloud Run no resuelve `mlflow://` URI desde local (auth) | baja | medio | Fallback: registrar manualmente via API REST `POST /api/2.0/mlflow/registered-models/create` con artifact_uri = path GCS |
| R5 | Tag git ya existe (debug previo) | muy baja | bajo | `git tag -d <name>` y recrear; tags son moviles |

---

## 7. Checklist de cierre US

- [ ] Rama `feature/E5-US-022-cierre-farslip-fe` creada
- [ ] P1: `_project_parcels_to_embeddings` real implementado + AC-P1-1..5 verdes
- [ ] P2: `fusion.py` con bloque FarSLIP + AC-P2-1..4 verdes
- [ ] P3: notebook 05b + comparison.parquet + AC-P3-1..4 verdes
- [ ] P4: tag git + MLflow Registry + l4_log.md + notebook 04 + AC-P4-1..4 verdes
- [ ] Conventional Commit `feat(E5): US-022-cierre — FarSLIP downstream + cierre gates B-3/B-4/B-5`
- [ ] Tests cobertura ≥70% nuevos modulos
- [ ] `make check` limpio
- [ ] `docs/us-resolved/us-022-cierre.md` creado
- [ ] `docs/us-handoff/us-022-cierre.md` actualizado durante ejecucion
- [ ] US-022-c gates B-3/B-4/B-5 marcados como cerrados en `docs/us-resolved/us-022-c.md`
- [ ] PR `us-022-cierre -> main` abierto + verde

---

## 8. engram-memory keywords

`us-022-cierre`, `farslip-downstream`, `embeddings-italy-v2`, `extract-real`, `fusion-farslip-block`,
`xgboost-lift-farslip`, `lightgbm-lift-farslip`, `mlflow-registry-production`, `gate-b3-miou-pastis`,
`gate-b4-shape-tag`, `gate-b5-registry`, `us-022-cierre-resolved`.

---

## 9. Referencias

- US-022-c (closed): [`docs/us-resolved/us-022-c.md`](../us-resolved/us-022-c.md)
- ADR-007 fidelidad FarSLIP: [`docs/decisions/ADR-007-farslip-fidelity-paper.md`](../decisions/ADR-007-farslip-fidelity-paper.md)
- Paper FarSLIP: Li et al. (2025), arXiv:2511.14901
- Plan v6 resumen §3 EPIC 5: [`context/RefinamientoPlaneacionAgroSatCopilot_v6_RESUMEN.md`](../../context/RefinamientoPlaneacionAgroSatCopilot_v6_RESUMEN.md)
- Backlog d/e remoto: [`docs/product-backlog/`](../product-backlog/)
