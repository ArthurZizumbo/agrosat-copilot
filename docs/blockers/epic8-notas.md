# Blockers EPIC 8 (Backend FastAPI) — validacion 2026-06-25

> Pasada de validacion autonoma US-051..056. El backend esta SUSTANCIALMENTE
> COMPLETO y verificado con tests reales (146 passed, incluidos 29 testcontainers
> con Postgres real). Los blockers son residuales, ninguno bloquea la presentacion.

## B-E8-1 — RESUELTO: mypy error en metrics.py (era de US-059)

**Que**: `mypy app/` reportaba 1 error en `backend/app/api/metrics.py:27`
(`no-any-return`): el router delegaba `return render_latest()` y mypy infería
`Any` pese a que `render_latest` declara `-> Response` (contaminacion por el
`prometheus_client` sin tipos).

**Fix**: asignacion a variable tipada explicita
(`response: Response = render_latest(); return response`). Sin `type: ignore`.
mypy ahora limpio (36 files, 0 errors); ruff sigue limpio. Estado: RESUELTO en
esta rama (adelantado aunque metrics.py sea de US-059/EPIC 10, por estar en el
working tree del backend).

## B-E8-2 — Discrepancias de conteo/nombres en handoffs originales, severidad INFORMATIVA

**Que**: los handoffs originales citaban conteos de tests y archivos que no
coinciden con la realidad verificada:
- US-053 handoff decia "57 passed" para tests ML; el conteo REAL es **25**
  (test_aois_endpoint 7 + timeseries 4 + tiles_stac 3 + geo_models 12 +
  classify_flags/label_space). Corregido en us-resolved/us-053.md.
- US-053 handoff cita `tests/ml/agent/test_class_remap.py` que **NO existe**: se
  consolido en `test_label_space.py`. Citado en el cierre.

**Impacto**: ninguno funcional — son inexactitudes de documentacion, ya
corregidas en los us-resolved con los numeros reales.

**Accion recomendada**: ninguna; los us-resolved ya llevan los conteos honestos.

## B-E8-3 — Router /jobs NO montado (diferido por diseno), severidad INFORMATIVA

**Que**: el worker Pub/Sub (US-056) es scaffolding honesto: `jobs_service.py`
lanza `NotImplementedError` para el modo pubsub, el modo `sync` funciona, y el
router `/jobs` NO esta montado en main.py.

**Causa**: diferido por diseno (ADR-009/012). El MVP corre inferencia sincrona;
la cola Pub/Sub es trabajo FUTURE.

**Impacto**: ninguno — es el comportamiento esperado y documentado. Los 12 tests
del modo sync + validacion Pydantic + NotImplementedError pasan.

**Accion recomendada**: ninguna. Montar `/jobs` cuando se implemente el worker
async real (post-presentacion).

## Nota de entorno — testcontainers

Los tests de integracion con testcontainers (RLS isolation 9, aois 7, timeseries
4, llm_switch 9 = 29 tests) requieren Docker. En esta pasada Docker estaba UP y
corrieron contra Postgres real (29 min). Donde no haya Docker se auto-skipean por
entorno (skip-by-environment, no fallo).
