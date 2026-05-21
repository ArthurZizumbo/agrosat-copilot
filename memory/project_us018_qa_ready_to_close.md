---
name: project-us018-qa-ready-to-close
description: QA final de US-018 (feature selection/extraction/normalization) cerrado 2026-05-21, ready-to-close
metadata:
  type: project
---

QA final de la US-018 (Feature Selection/Extraction/Normalization, Avance 2
CRISP-ML(Q)) completado el 2026-05-21. Estado: ready-to-close.

La US-018 fase 3-4 ya estaba mergeada en PR #17. Las extensiones fase 5
(encoding categorico) y fase 6 (parcel-level: vectorizacion PASTIS-R + samplers
GEE) vivian sin commitear en el working tree (~50 archivos) — el QA las cubrio.

**Bugs del handoff verificados (ambos resueltos):**
- `test_anova_f_top3_contains_ndvi_or_fft` ampliado a 17 prefijos vegetativos.
- Workaround numpy 2.3.5 / scipy 1.17.1 vs `pytest --cov` activo.

**Correcciones del QA:** lint (ruff F401/F541/F841/BLE001/S112), mypy en
`gee_sampler.py` (8 errores -> 0), `sys.exit(0)` muerto eliminado en 5 scripts
fase 6, +5 per-file-ignores `B008` en `pyproject.toml`.

**Gates:** ruff + mypy limpios, 228 tests scope US-018 verdes, security-audit
APROBADO, code-review APROBADO CON OBSERVACIONES.

**Fuera de scope (no bloquean):** 2 fallos en `tests/ml/report/` son trabajo
de Aaron (breizhcrops/paper-methods), no US-018. Deuda DRY fase 6: helpers
triplicados entre scripts `generate_*` -> follow-up refactor a `ml/ingest/`.

Detalle en `docs/us-handoff/us-018.md` seccion "QA final fase 7-cierre".
Relacionado con [[feedback-alphaearth-aggregation]].
