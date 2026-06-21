# ADR-012 — Inferencia síncrona en el MVP; worker Pub/Sub diferido a Full (US-056)

**Status**: Aceptada · 2026-06-20
**Fecha**: 2026-06-20
**Decisores**: Arthur Zizumbo (MLOps Lead) · Equipo 17
**Reemplaza/extiende**: concreta el "OUT" de US-056 que [ADR-009](ADR-009-h100-reactivacion-pivote-farslip-alcance-v8.md) dejó como diferido (§1 OUT, §9.2 del plan v8).
**Fundamento**: [`context/RefinamientoPlaneacionAgroSatCopilot_v8.md`](../../context/RefinamientoPlaneacionAgroSatCopilot_v8.md) bloque "### US-056" · regla NON-NEGOTIABLE "inferencia pesada (>2 s) vía Pub/Sub" de [`CLAUDE.md`](../../CLAUDE.md) / [`backend/AGENTS.md`](../../backend/AGENTS.md).

---

## Contexto

La regla de arquitectura del proyecto es clara: **inferencia que tarde más de ~2 s no debe correr síncrona en el request del API — se delega a una cola Pub/Sub con un worker dedicado** (`ml/agent/AGENTS.md`: "Inferencia ML pesada inline en un tool — delegar a Pub/Sub worker"). US-056 materializa esa regla con un worker Cloud Run GPU L4 que consume el topic `inference-jobs`, persiste resultados en GCS, notifica vía el topic `inference-results` y reintenta con DLQ tras 3 fallos.

Sin embargo, US-056 quedó **OUT del MVP** (diferida post-presentación) en el plan v8, por tres razones concretas:

1. **El cuello de botella es tiempo (~3 semanas al deadline 27-jun), no cómputo** ([ADR-009](ADR-009-h100-reactivacion-pivote-farslip-alcance-v8.md) D-1). Construir la cola async productiva (topics + DLQ + Cloud Run GPU + alerta Cloud Monitoring + notificador SSE) consume días que el camino crítico necesita para la demo visual.
2. **El MVP no requiere inferencia pesada.** El copiloto demoable sirve clasificación de cultivo con el miembro **`xgb-alphaearth`** (tabular, CPU-light, `functools.lru_cache`) sobre parcelas **PASTIS pre-cargadas** y embeddings AlphaEarth ya materializados. Esa inferencia es de milisegundos: corre síncrona sin bloquear al usuario (es exactamente lo que ya hace el tool `classify_new_parcel` que el reasoner de `/chat` invoca).
3. **El plan v8 §9.3 lo explicita: "NO demostrar cola Pub/Sub real"** en la presentación. Fingir una cola que no escala nada (un solo worker local, sin GPU) sería deshonesto.

La regla "inferencia >2 s vía Pub/Sub" **se respeta en producción Full**; el MVP simplemente no ejecuta ningún modelo pesado, por lo que no la viola.

## Decisión

### D-1 — El MVP corre inferencia SÍNCRONA

El backend sirve la clasificación de cultivo **inline** en el request, reutilizando el tool `classify_new_parcel` (`ml/agent/tools/classify.py`, miembro `xgb-alphaearth` / opcional `stacking-5`, ambos CPU). No se construye cola Pub/Sub, GCS de resultados, DLQ ni notificador SSE de jobs en las 3 semanas al deadline.

### D-2 — Se deja la INTERFAZ pública lista y documentada (scaffolding honesto)

Se entrega el contrato que el worker Full implementará, sin stubs que finjan funcionar:

- **`backend/app/services/jobs_service.py`** — `JobsService` con la interfaz pública:
  - `submit_job(request: JobRequest) -> job_id` y `get_job_status(job_id) -> JobStatus | None`.
  - `enqueue_mode="sync"` (default, **MVP, REAL**): ejecuta la inferencia inline (delegando al runner `classify_new_parcel`) y devuelve el job ya en estado `DONE` (o `FAILED` con error estructurado). Sin cola, sin GCS, sin worker.
  - `enqueue_mode="pubsub"` (**FUTURE, Full-only**): lanza `NotImplementedError` con mensaje claro — *"Pub/Sub mode is Full-only (US-056 deferred, ADR-009)"* — y NUNCA hace no-op silencioso ni finge éxito.
  - Modelos pesados (no en `{xgb-alphaearth, stacking-5}`) en modo `sync` devuelven `FAILED` con error honesto ("heavy models run on the Full Pub/Sub GPU worker"), no se sirven inline fingiendo.
- **Modelos Pydantic**: `JobRequest{session_id, aoi_geojson: GeoJSONGeometry, model_id, year, params}` (AOI validado por Pydantic antes del service); `JobStatus{job_id, state, result_url, result, error}`; `JobState` enum (`queued/running/done/failed`).

### D-3 — El worker Pub/Sub se entrega como esqueleto documentado, NO corriendo

- **`ml/workers/inference_worker.py`** — la firma del callback (`handle_message(message)`), el parser/validador real de mensajes (`parse_job_request` — sin dependencia GCP), el schema del topic de resultados (`InferenceResult`) y el flujo completo documentado en el docstring. Los pasos de inferencia GPU + persistencia GCS + publish son `NotImplementedError` explícitos (fallan claro, nunca fabrican resultado). El loop de subscripción (`run_subscriber`) y el `if __name__ == "__main__"` **no arrancan un subscriber real**: lanzan `NotImplementedError` explicando que requieren GCP Pub/Sub + Cloud Run GPU L4.

### D-4 — Plan de activación en Full (qué falta para encender US-056)

Cuando US-056 se active post-presentación, el trabajo es **wiring, no rediseño**:

1. **Infra (Terraform)**: topics `inference-jobs` + `inference-results` + `inference-jobs-dlq`; subscripción con `dead_letter_policy.max_delivery_attempts = 3`; alerta Cloud Monitoring sobre profundidad de la DLQ; servicio Cloud Run con runtime GPU L4 + Dockerfile.
2. **`JobsService.submit_job`** rama `pubsub`: reemplazar el `NotImplementedError` por un `PublisherClient.publish(topic, JobRequest.json())`; persistir el job en PostgreSQL para que el worker (otro proceso) actualice su estado.
3. **`inference_worker.py`**: quitar los `NotImplementedError` de `_run_inference` (cargar `model_id` en GPU, inferir, escribir a `gs://<artifacts>/inference/<job_id>.json`) y `_publish_result` (publicar `InferenceResult` a `inference-results`); arrancar `run_subscriber`.
4. **Backend SSE notifier**: subscribirse a `inference-results` y empujar un frame `job_done` a la sesión del frontend (el `ChatPanel` US-057 actualiza la tarjeta del job).
5. **Idempotencia**: resultados keyed por `job_id` en GCS (un mensaje reentregado sobrescribe el mismo objeto, no duplica trabajo).

## Consecuencias

- **Positiva (honestidad):** el MVP no finge una cola que no escala; sirve inferencia ligera síncrona de verdad, y la interfaz para escalar a Pub/Sub queda fijada y testeada en su parte real (modo `sync`).
- **Positiva (camino crítico):** cero días gastados en infra Pub/Sub/GPU productiva antes del 27-jun; el esfuerzo va a la demo visual (mapa + chat + overlays).
- **Positiva (regla respetada):** la regla "inferencia >2 s vía Pub/Sub" se cumple en Full; el MVP no ejecuta modelos pesados, así que no la viola.
- **Riesgo / deuda explícita:** el `JobsService` MVP mantiene el registro de jobs en memoria (`dict`), no en PostgreSQL — válido porque el modo `sync` completa el job antes de retornar, pero el Full debe persistirlo para coordinación inter-proceso. Documentado en el docstring del service.
- **Riesgo:** activar US-056 en Full requiere la infra GCP (topics + DLQ + Cloud Run GPU) que no se prueba hasta entonces; mitigado por el parser de mensajes (`parse_job_request`) y los modelos Pydantic, que SÍ se testean sin GCP.

## Relacionado

- [ADR-009](ADR-009-h100-reactivacion-pivote-farslip-alcance-v8.md) — alcance v8; difiere US-056 (worker Pub/Sub) a Full (§1 OUT). Este ADR concreta esa decisión a nivel de interfaz y scaffolding.
- US-055 ([`docs/us-handoff/us-055.md`](../us-handoff/us-055.md)) — TiTiler montado dentro de la imagen backend; precede a US-056 en el camino crítico C9.
- `backend/app/services/jobs_service.py` — la interfaz `JobsService` (modo `sync` real, modo `pubsub` `NotImplementedError`).
- `ml/workers/inference_worker.py` — el esqueleto del subscriber Pub/Sub (no corre en el MVP).
