# Blockers EPIC 6 (Ensambles) — validacion 2026-06-24

> Notas de la pasada de validacion autonoma US-040..043. Cada blocker indica
> severidad, evidencia y accion recomendada. No bloquean la presentacion del
> 27-jun salvo donde se indique.

## B-E6-1 — Test desactualizado en US-043 (champion), severidad BAJA

**Que**: `tests/ml/ensemble/test_us043_orchestrator.py::test_five_member_set_is_three_plus_two_farslip`
falla de forma reproducible (44 passed, 1 failed; suite corrida 2 veces).

**Causa raiz**: el test espera el miembro `tsvit-pheno` en `_BASE_MEMBERS_3`, pero
el orquestador usa deliberadamente `tsvit-pheno-fullm` (cambio comentado en
`scripts/run_us043_farslip_ensembles.py:81-87`). Es un **test viejo que no se
actualizo** tras cambiar el miembro base a la variante full-multi, NO un bug del
run ni del champion.

**Impacto**: ninguno en el resultado. El champion Stacking-5 +FarSLIP (F1-macro
0.7486, france-9 ~0.912) es correcto y reproducible. Solo el assert del set de
miembros quedo obsoleto.

**Accion recomendada**: actualizar el test para esperar `tsvit-pheno-fullm` (1 linea).
No se toco en esta pasada para no mezclar fix de codigo con cierre documental.
Documentado con `[~]` en docs/manual-test/us-043.md (MT-1, MT-6).

## B-E6-2 — E-a/E-b: resultado negativo conceptual (NO es un bug, es hallazgo), severidad INFORMATIVA

**Que**: US-041 (E-a fusion dual-head) F1-macro 0.2694 y US-042 (E-b) 0.3395,
ambos MUY por debajo del Stacking US-040 (0.747).

**Causa raiz**: la fusion dual-head ingenua mezcla las 4 clases de FarSLIP con las
18 de PASTIS (error conceptual "DANA" documentado en el notebook Avance5). E-b
hereda esa base rota.

**Impacto**: ninguno negativo — es la **narrativa honesta** del EPIC 6: se probo la
fusion dual-head, fallo, y por eso el champion es el stacking aprendido
(US-043, FarSLIP entra via stacking ft-18 + zero-shot, no via fusion ingenua).
Documentado en us-resolved/us-041.md, us-042.md como "experimento con resultado
negativo/suboptimo, honestamente documentado".

**Accion recomendada**: ninguna. Es evidencia valida para la rubrica (muestra
que la complementariedad pesa mas que la fuerza individual). NO re-ejecutar.

## B-E6-3 — Cambio de alcance US-043 (DISENO -> EJECUTADO), severidad INFORMATIVA

**Que**: el planning/handoff original encuadraba US-043 como DISENO ONLY
(ADR-010 geo-context E-c), pero el cierre del Avance 5 la materializo como el
ensamble FarSLIP campeon ejecutado.

**Impacto**: el ADR-010 (E-c geo-context) sigue vigente como trabajo FUTURE
separado; lo ejecutado y campeon es Stacking-5 +FarSLIP. Documentado
explicitamente en los 3 docs de US-043.

**Accion recomendada**: ninguna — solo claridad para el revisor.
