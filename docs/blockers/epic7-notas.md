# Blockers EPIC 7 (Agente conversacional) — validacion 2026-06-25

> Notas de la pasada de validacion autonoma US-045..050. El hallazgo central
> (perceiver percibia con baseline, no con champion) ya se ARREGLO en esta misma
> rama (commits a6942f4, 485d4e7). Lo que sigue son blockers residuales.

## B-E7-1 — RESUELTO: perceiver percibia con baseline, no con el champion

**Que**: el perceiver del agente (ml/agent/perceiver.py) componia el posterior del
clasificador BASELINE xgb-alphaearth en vez del ensamble CHAMPION Stacking-5
(EPIC 6 / US-043). El eval US-049 y la app percibian con el modelo equivocado.

**Causa raiz**: el flag `use_stacking` de `classify.run` esta default OFF, y el
perceiver nunca lo activaba; ademas `observe()` llamaba `_load_classifier()`
directo (solo xgb). El champion existia en el codigo pero estaba apagado.

**Fix (commits a6942f4 + 485d4e7)**: re-cableo a Stacking-5 restringido a las 9
clases france-9 (directiva Arthur), con degradacion limpia a xgb cuando no hay
OOF. Cubre Agente (via chat_service) y App con un solo cambio.

**Impacto medido (real, H100, 13.481 parcelas fold-5, ml/eval/perceiver_champion_eval.py)**:
accuracy 0.831 -> 0.941 (+11.0 pp), macro-F1 0.687 -> 0.901 (+21.4 pp), neto
+1.490 parcelas corregidas. Estado: RESUELTO.

**Caveat documentado**: el system-eval US-049 (eval_grounded_crop) usa
`_StubClassifier` por diseno (mide fidelidad del reporte del agente, NO el
clasificador), por eso su numero (0.923) no cambia con el re-cableo. La mejora se
mide con el modulo nuevo perceiver_champion_eval, no con el system-eval.

## B-E7-2 — Deuda de regresion en tests/ml/eval/: 23 fallos PRE-EXISTENTES, severidad MEDIA

**Que**: `pytest tests/ml/eval/` da 495 passed, 23 failed, 11 skipped. Los 23
fallos NO son del re-cableo (probado: restaurando perceiver/classify a main los
mismos 23 fallan). Son deuda previa a esta validacion.

**Dos causas**:
1. **Regex i18n desincronizada** (la mayoria): tests con `pytest.raises(match="...")`
   en espanol contra mensajes de produccion en INGLES (correcto por la regla de
   idioma del repo: codigo en ingles). Ej. `test_compute_metrics_length_mismatch_raises`
   espera `match="misma forma"` pero `ml/eval/metrics.py:87` emite
   `"must have the same shape"`. Igual en test_interpretability, test_learning_curves,
   test_reencuadre_plots, test_metrics, test_shap_normalize_rejects_unknown_shape.
2. **Fixtures/datasets ausentes** (entorno): `test_agent_system_eval.py` (3) ->
   `data/agent_eval/toolcalling_cases.jsonl` ausente; `test_paper_bench.py` ->
   mismo origen; `test_load_geobench2_*` -> `data/test_fixtures/geobench2_mini/manifest.json`
   (este ultimo ya tiene blocker en epic11/12; fixture versionado en PR #54/#55).

**Impacto**: NO bloquea el re-cableo ni la presentacion del agente. Es deuda de
mantenimiento de la suite de eval.

**Accion recomendada**: (1) alinear los `match=` al mensaje ingles del codigo (el
codigo esta bien, el test quedo en espanol) — trivial, ~10 tests. (2) versionar o
generar los fixtures JSONL de agent_eval. No se hizo en esta pasada para no
mezclar fix de tests con la validacion del re-cableo.

## B-E7-3 — Subset AgroMind casi todo multimodal (heredado de US-049), severidad BAJA

**Que**: el subset AgroMind quedo 494/500 multimodal -> Qwen (text-only) solo
evalua 6 items. Ya documentado en el handoff US-049.

**Accion recomendada**: para una eval real de Qwen, ampliar `make_subset` con
filtro `is_multimodal=False`. No critico para la presentacion.
