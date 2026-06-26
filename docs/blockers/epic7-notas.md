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

## B-E7-4 — Gap fenologico de la rama degradada del perceiver, severidad MEDIA

**Que**: el re-cableo al campeon (B-E7-1) sirve el Stacking-5 COMPLETO (con los
miembros temporales TSViT-pheno y U-TAE, que aportan el descriptor fenologico
denso) SOLO para parcelas presentes en el OOF de fold-5 de PASTIS-R. Para una
parcela NUEVA -- un AOI dibujado por el usuario, o una escena Sentinel recien
descargada -- no existe OOF, y el perceiver degrada limpio a `xgb-alphaearth`,
que clasifica con el embedding AlphaEarth ANUAL sin ninguna fenologia intra-anual
(ver `ml/agent/perceiver.py:382`). Es decir: el campeon con fenologia cubre el
universo evaluable, pero la rama que ve el usuario en produccion para parcelas
frescas es la SIN fenologia.

**Costo medido de la degradacion (real, mismas 13.481 parcelas fold-5,
`ml/eval/perceiver_champion_eval.py` / `reports/agent_bench/perceiver_champion_eval.json`)**:
es el espejo del +11pp del re-cableo. La rama degradada `xgb-anual` rinde
accuracy 0.831 / macro-F1 0.687 frente a 0.941 / 0.901 del campeon-OOF, es decir
una perdida de **-11.05 pp de accuracy y -21.4 pp de macro-F1** cuando una parcela
cae a la rama sin fenologia. Sobre estas parcelas son 1.490 que el campeon
corrige y la rama anual no.

**Causa raiz (de diseno, no bug)**: TSViT-pheno y U-TAE necesitan la SERIE
Sentinel-2 multi-fecha como insumo. PASTIS-R la traia (de ahi el OOF cacheado);
EuroCropsML tambien la trae (.npz por parcela). Pero los datasets/transfers que
SOLO usaron el embedding anual (multi-region original, WorldCereal tropical,
demo Mexico) y cualquier AOI nuevo NO tienen esa serie, asi que los modelos
temporales no pueden correr y el campeon pierde su fenologia.

**Camino para cerrarlo (insumo: la cuenta Sentinel/CDSE, ya funcional)**:
1. La serie S2 de una parcela nueva se descarga con `ml/ingest/cdse_client.py`
   (`CDSEClient.search_s2`, OData, probado real Toscana <10% nubes) ahora que
   `search_stac` ya esta conectado a CDSE (commit 60c73e7).
2. Con esa serie, los features temporales (`ml/transfer/temporal_features.py`,
   99-dim: stats por banda/indice + fenologia NDVI + FFT) se pueden computar
   inline y alimentar el clasificador, o correr TSViT/U-TAE offline via worker
   Pub/Sub (la regla de `ml/agent/AGENTS.md` prohibe inferencia pesada inline en
   la tool).
3. Evidencia de que el insumo temporal SI rescata fenologia: la fusion
   anual+temporal sobre EuroCropsML (mismas parcelas, mismo split espacial)
   sube las clases fenologicas (winter-vs-spring) -- ver
   `data/transfer/temporal_vs_annual/` y `docs/transfer/modelo-multiregion.md`.

**Capa barata complementaria (descriptor fenologico TEXTUAL con Gemini Flash)**:
independiente del numero del clasificador, el reasoner puede recibir un descriptor
fenologico en lenguaje natural por parcela (p. ej. "emergencia tardia, pico de
verdor en julio, senescencia rapida") generado con Gemini 2.5 Flash (barato, sin
GPU) a partir de las metricas de la curva NDVI ya calculadas en
`temporal_features._ndvi_phenology` (peak_doy, sog_doy, early/late green-up, AUC).
Esto NO sustituye a TSViT/U-TAE (que dan la PREDICCION); enriquece el bloque de
texto que el perceiver pasa al reasoner. Es la ruta de menor costo para que el
agente "hable" de fenologia incluso cuando la prediccion viene de la rama anual.

**Accion recomendada**: (1) exponer el descriptor fenologico textual NDVI en el
perceiver para parcelas con serie S2 (barato, alto valor narrativo); (2) wirear el
worker temporal CDSE->features->clasificador para que parcelas nuevas no caigan a
la rama -21pp; (3) medir end-to-end sobre un AOI real con serie descargada de
CDSE. Severidad MEDIA: no bloquea la demo (parcelas del catalogo usan campeon),
pero es el limite honesto a documentar en paper/presentacion.
