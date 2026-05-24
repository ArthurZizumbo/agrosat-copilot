# US-022-e — FarSLIP-FR augmentation (FarSLIP-IT + FarSLIP-FR como base learners separados)

**Estado**: backlog
**Origen**: discusion 2026-05-24 con usuario sobre alcance geografico FarSLIP
**Epic**: E6 (ensemble final)
**Sprint**: post-Avance 5 (estimado 1-14 jun)
**SP estimado**: 3
**Owner**: Arthur (MLOps) + Isaac (ML)
**Trigger**: cuando se diseñe el ensemble stacking de EPIC 6 y se quiera contribuir diversidad geografica

## Contexto

US-022-c P1 entrego FarSLIP-IT (FarSLIP entrenado solo en 3 ROIs italianas: Pianura Padana,
Toscana, Puglia) por restriccion de datos abiertos + presupuesto L4 spot ~6h. ADR-007
documenta la decision + limitacion.

Para EPIC 6 stacking ensemble, agregar **FarSLIP-FR** (entrenado sobre EuroCrops FR 2018 — 11
GB ya ingestados y versionados con DVC tag `eurocrops-fr-2018-v1` por geo-data agent
2026-05-24) provee:

1. **Diversidad geografica**: FarSLIP-IT y FarSLIP-FR como base learners independientes en
   stacking → reduce varianza del meta-learner.
2. **Test cross-region simetrico**: FarSLIP-IT eval Francia + FarSLIP-FR eval Italia
   demuestra ambas direcciones de transferencia.
3. **Aprovecha EuroCrops FR ya versionado**: $0 ingesta adicional (data ya en DVC).

## Alcance propuesto

| Bloque | SP | Descripcion |
|--------|----|-------------:|
| E1 | 0.5 | Vocabulario CAP frances en `ml/farslip/cap_vocabulary.yaml` (~25 plantillas it/es/en/fr) |
| E2 | 0.5 | Adaptar `ml/farslip/dataset.py` para soportar ROIs francesas (Beauce, Loire, Bretagne, etc.) con bounding boxes |
| E3 | 1 | Build dataset FarSLIP-FR (~30k pares Sentinel-2 + GSAA FR-equivalent) — re-usa pipeline US-017 |
| E4 | 1 | Training FarSLIP-FR L4 spot ~6h ~$1.70 USD (Vertex AI custom-job) |

Total: 3 SP, ~$2 USD presupuesto.

## Criterios de aceptacion

| AC | Criterio | Verificacion |
|----|----------|--------------|
| AC-1 | Vocabulario CAP FR con >=20 plantillas | `cap_vocabulary.yaml` con `regions.fr` |
| AC-2 | Dataset FarSLIP-FR >=25k pares + balance >=0.20 | `make farslip-dataset-check rois=france` |
| AC-3 | MLflow run `farslip-clip-france-v1` con `val_clip_acc` por epoch + stage `Production` | UI MLflow |
| AC-4 | DVC tag `farslip-embeddings-france-v1` con parquet ~85k parcelas FR x 514 | `git tag -l` + parquet shape |
| AC-5 | Eval cross-region: FarSLIP-IT mIoU Francia + FarSLIP-FR mIoU Italia documentados | tabla en `docs/us-resolved/us-022-e.md` |
| AC-6 | Costo L4 FR + extract <= $2.50 USD | `make cost-audit` |

## Decisiones tecnicas

- **D-1**: Vocabulario FR usa terminos CAP oficiales UE (Reglamento 1308/2013). Plantillas
  paralelas a italiano (bare, with_phenology, with_region).
- **D-2**: ROIs francesas: Beauce (cereales norte), Loire (vinedos), Bretagne (verde).
- **D-3**: Misma arquitectura ml/farslip/distill.py (zero refactor — paper Li et al. 2025
  aplica idem). ADR-007 ya documenta fidelidad.
- **D-4**: Ensemble strategy en EPIC 6: stacking con FarSLIP-IT(512) + FarSLIP-FR(512) +
  AlphaEarth(64) + DINOv3(384) = 1472-dim input al meta-learner XGB.

## Riesgos

- R1: GSAA Italia y FR son ODbL — re-distribuir embeddings precomputados es legal pero
  documentar atribucion en `docs/licenses/DATA_LICENSE.md`.
- R2: FR mIoU Italia < +0.05 → resultado honesto, no bloquea ensemble (R2 del plan original).
- R3: Cuota L4 ya aprobada (US-022-c P1 quota request) — esta US no requiere nueva quota.

## Referencias

- ADR-007 fidelidad FarSLIP: [`docs/decisions/ADR-007-farslip-fidelity-paper.md`](../decisions/ADR-007-farslip-fidelity-paper.md)
- US-022-c P1 (FarSLIP-IT base): [`docs/us-handoff/us-022-c.md`](../us-handoff/us-022-c.md)
- Tag DVC eurocrops-fr-2018-v1: creado por geo-data agent 2026-05-24, blob 11 GB en `gs://agrosat-dvc-remote`
- Paper-faro: Li et al. (2025), arXiv:2511.14901, FarSLIP
- Skill agrosat-llm-finetuning: [`.claude/skills/agrosat-llm-finetuning/SKILL.md`](../../.claude/skills/agrosat-llm-finetuning/SKILL.md)
