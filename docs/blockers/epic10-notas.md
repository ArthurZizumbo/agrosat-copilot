# EPIC 10 - Notas y blockers (Observabilidad, Docs, Multi-region)

Registro de lo que NO se pudo verificar/correr en el entorno de desarrollo de
esta sesion. Politica (regla de Arthur): si algo no se puede verificar/correr,
se anota aqui y se SIGUE. Ninguno de estos puntos bloquea la entrega del
scaffolding.

## US-061 - Analisis costo-beneficio (A6/A7)

### B17. Export .xlsx — RESUELTO (generado desde los CSV fuente, 2026-06-25)
- **Que faltaba:** el entregable "tablas en Excel" del AC pide un `.xlsx`. En la
  sesion previa no se genero porque `openpyxl`/`xlsxwriter` no estaban confirmados.
- **Fix aplicado (2026-06-25):** confirmado `openpyxl` 3.1.5 disponible en el
  entorno, se genero **`docs/business/costo_beneficio.xlsx`** (3 hojas: "Lme" con
  procedencia de datos, "Costos CRISP-ML" 7x4, "Beneficios 500ha" 5x5) leyendo
  DIRECTAMENTE los CSV fuente versionados (`docs/business/data/costos_crisp_ml.csv`
  y `beneficios_500ha.csv`). Ninguna cifra es sintetica: el xlsx es la misma data
  real de los CSV/Markdown, costos anclados a `docs/operations/finops.md` y a la
  factura GCP real; beneficios etiquetados como estimaciones de literatura con su
  supuesto, rango y fuente.
- **Respaldo adicional:** ademas del `.xlsx` siguen versionados los `.csv` fuente,
  las tablas Markdown en `docs/business/costo_beneficio.md` y el export LaTeX
  `docs/business/costo_beneficio.tex` para el paper.
- **Pendiente (no bloqueante):** si se quiere reproducibilidad CI del xlsx, declarar
  `openpyxl` como dep de dev en `pyproject.toml` (hoy se instalo ad-hoc); el `.xlsx`
  ya esta en disco como entregable.

## US-059 - Observabilidad Prometheus (scaffolding)

### B1. GPU utilization / OOM sin scrape activo
- **Que falta:** las metricas de GPU (`DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_FB_USED`,
  `DCGM_FI_DEV_FB_FREE`) requieren `dcgm-exporter` corriendo en la H100 NVL del
  sponsor (VM `gjcamacho-gpuh1`, prestada 24/7) o en la L4 spot
  (`agrosat-farslip-trainer-dev`), y un Prometheus scrapeandolo.
- **Estado:** no hay `dcgm-exporter` ni Prometheus desplegado en este entorno.
  El panel `worker_ml.json` y la alerta `GpuOutOfMemory` quedan definidos con la
  metrica esperada documentada, sin datos inventados.
- **Decision:** entregar el panel/alerta como plantilla; poblar cuando exista el
  scrape (dependencia con el serving Qwen, US-048).

### B2. Alertas Cloud Monitoring sin proyecto desplegado
- **Que falta:** el wiring de las alertas a un canal real (email del equipo via
  Alertmanager o una alert policy de Cloud Monitoring con notification channel)
  requiere un proyecto cloud desplegado con el scrape activo.
- **Estado:** se entregan las reglas declaradas (`alerts/alert_rules.yml`,
  formato Prometheus) y el equivalente Cloud Monitoring documentado en el mismo
  archivo. No hay Alertmanager ni alert policy creada.
- **Decision:** el notification channel se creara con Terraform cuando exista el
  proyecto desplegado; pendiente, no bloquea la US.

### B3. Cloud Run scale-to-zero no scrapeable por pull
- **Que falta:** Cloud Run con `min_instances=0` (objetivo FinOps ~$115/mes) no
  es scrapeable de forma estable por un Prometheus pull; la instancia muere entre
  requests.
- **Estado:** el endpoint `/metrics` (pull) queda util en dev/staging always-on y
  para los tests. Para prod la via realista es push gateway
  (`Settings.prometheus_pushgateway`, ya declarado) o Cloud Monitoring.
- **Decision:** nota tecnica documentada en `infrastructure/grafana/README.md`;
  no es blocker.

### B4. Paneles de chat sin trafico real
- **Que falta:** el slot "latencia chat p95" y los contadores derivados de chat
  (tokens, modelo activo, tool-call success rate) se pueblan con trafico `/chat`
  real, que aterriza con US-052 y se reporta en US-065.
- **Estado:** el panel queda como plantilla (reusa el histograma HTTP filtrado a
  la ruta `/chat`). NO se fabrica trafico sintetico para "demostrar" el panel
  (regla de datos reales).
- **Decision:** la observabilidad de chat (metricas dedicadas promovidas desde
  los logs structlog `chat_stream_started` / `chat_model_resolved` /
  `chat_stream_finished` del ChatService) se entrega en US-065.

### B5. CONTENT_TYPE_LATEST = version 1.0.0 (no 0.0.4)
- **Observacion:** con `prometheus-client` 0.25.0 instalado en esta sesion,
  `CONTENT_TYPE_LATEST` es `text/plain; version=1.0.0; charset=utf-8`, no el
  historico `version=0.0.4`. El test asserta contra el simbolo real
  `CONTENT_TYPE_LATEST` de la libreria, no contra un string hardcodeado, para no
  romper si la libreria cambia el formato de exposicion.

## US-065 - Observabilidad de chat (metricas por turno)

### B6. Tokens de Qwen/vLLM no disponibles en streaming (R1) - RESUELTO (codigo)
- **Que faltaba:** el backend OpenAI-compatible de vLLM/Qwen
  (`ml/agent/backends.py::VLLMOpenAIBackend`) NO emitia `usage` en modo streaming
  salvo que se enviara `stream_options={"include_usage": True}` en
  `chat.completions.create`, lo que no se seteaba. En esa ruta el
  `chat_turn_metrics` reportaba `tokens_prompt/completion/total = None`.
- **Fix aplicado (2026-06-24, rama `fix/blockers-validacion-us040-077`):**
  - `VLLMOpenAIBackend.generate_stream` ahora pasa
    `stream_options={"include_usage": True}` en `chat.completions.create`.
  - Se anadio `_usage_from_stream_event` que lee el chunk final de usage que vLLM
    emite con `choices=[]` (`prompt_tokens` / `completion_tokens` /
    `total_tokens`, ya en las keys neutrales del proyecto) y lo normaliza
    defensivamente (servidores que no lo reportan -> `None`, sin sintesis).
  - El usage se propaga sobre el ULTIMO chunk del turno: si hay tool-calls va en
    la ultima call; si no, en un chunk usage-only. Asi llega a
    `DoneEvent.usage` -> `chat_turn_metrics.tokens_*` igual que Gemini.
- **Gemini intacto:** `GeminiBackend` sigue leyendo `response.usage_metadata` de
  la respuesta no-streaming (ruta independiente, no toca `chat.completions`).
  Verificado: los 52 tests `chat`/`metric` del backend siguen verdes.
- **Tests:** 4 casos nuevos en `tests/ml/agent/test_backends.py`
  (`requests_include_usage`, `surfaces_streamed_usage`, `usage_on_tool_call_chunk`,
  `no_usage_when_server_omits_it`). Suite `tests/ml/agent/` = 134 passed.
- **Pendiente (no bloqueante):** verificacion end-to-end contra el endpoint vLLM
  REAL del H100 (los tests usan dobles que replican el chunk de usage de OpenAI/
  vLLM; falta el serving Qwen vivo para confirmar el wire real). El codigo y los
  tests no lo requieren.

### B7. Verificacion de tags MLflow contra :5010 (server caido en la sesion)
- **Que falta:** la verificacion en vivo de que los runs llevan tags
  `code_version` + `data_version` contra el server Docker `:5010`.
- **Estado:** el server MLflow Docker NO estaba corriendo en esta sesion
  (`docker ps` no lista `agro_sat_copilot-mlflow-1`). La escritura de tags YA
  esta garantizada por `ml/utils/mlflow_utils.py::track_experiment` (US-019), que
  escribe ambos tags en todo run y resuelve el URI con fallback
  `:5010 -> file:./mlruns`. El procedimiento de verificacion queda documentado en
  `docs/observability/chat-metrics.md` (seccion 5) y no requiere `:5010` vivo
  para que el codigo y los tests pasen.
- **Decision:** levantar `make mlflow-up` y correr el snippet de verificacion de
  la doc cuando se quiera auditar los tags; gotcha de los dos almacenes (server
  real vs `./mlruns` viejo, runs por subprocess en `RUNNING`) documentado.

### B8. Panel Grafana de chat sin scrape activo
- **Que falta:** el panel `infrastructure/grafana/chat_observability.json`
  necesita un Prometheus scrapeando el `/metrics` del backend con
  `chat_metrics_prometheus_enabled=true`.
- **Estado:** se entrega el dashboard como plantilla (PromQL de p95, tool-calls y
  tokens). El export Prometheus del helper es no-op si el flag esta off o si
  `prometheus-client` falla (degradacion honesta). No se fabrica trafico
  sintetico para "demostrar" el panel; la instrumentacion se valida con el flujo
  de prueba controlado de `test_chat_metrics.py` / `test_chat_sse.py`.
- **Dependencia de B6 (RESUELTA en codigo):** la fila de tokens del panel ahora
  tiene fuente real en AMBAS rutas (Gemini + Qwen/vLLM con `include_usage`); antes
  Qwen reportaba `None`. El panel scrapeara y se poblara solo una vez que exista
  trafico `/chat` real contra un backend con Prometheus activo
  (`chat_metrics_prometheus_enabled=true`). No se sintetiza trafico para
  "demostrarlo": el dashboard queda listo y la metrica de tokens ya no esta vacia
  por diseno en la ruta Qwen. Pendiente unicamente del despliegue del scrape
  (infra externa), no del codigo.

### B9. Tests de integracion de /chat rotos PRE-EXISTENTE (ajeno a US-065)
- **Observacion:** `backend/tests/integration/test_chat_endpoint.py` y
  `test_chat_uses_session_model.py` fallan con
  `ValueError: Duplicated timeseries in CollectorRegistry:
  {'http_request_duration_seconds...'}` al construir la app por segunda vez.
- **Estado (VERIFICADO):** el fallo existe igual en el arbol limpio (`git stash`
  + pytest -> mismos fallos), proviene del middleware de metricas de US-059
  (`backend/app/middleware/metrics.py`) que re-registra el histograma HTTP en
  cada `create_app()`, NO de US-065. Los tests unitarios de chat
  (`test_chat_sse.py`, 13 + 2 nuevos) y los de backends/agent siguen verdes.
- **Decision:** fuera de scope de US-065 (instrumentacion de chat). El fix
  correcto es registrar el instrumentator HTTP en un `CollectorRegistry`
  dedicado o hacerlo idempotente; pertenece a US-059. No bloquea esta US.

## US-067 - Documentacion FinOps

### B15. Desglose fino de factura no verificable en esta sesion
- **Que falta:** el desglose exacto por SKU de la factura GCP y el costo por request
  real de la Gemini API requieren `gcloud billing` con permiso/proyecto facturable
  no disponible en este entorno de dev.
- **Estado (datos reales):** se usan los rangos oficiales del proyecto, no estimaciones
  inventadas: GCP acumulado ~$0.30-0.49 USD, Gemini API en centavos
  (~$0.0001/descripcion FarSLIP), operativo objetivo ~$115/mes con scale-to-zero,
  training unico historico $262 (spot) - $602 (on-demand) y H100 del sponsor 24/7 sin
  costo para el equipo. El precio de referencia de Gemini 2.5 Pro ($1.25/$10 por M
  tokens entrada/salida) es cifra del plan v8.
- **Decision:** la auditoria fina (billing por SKU) se corre con `make cost-audit`
  cuando haya permisos de billing; documentado en `docs/operations/finops.md`. No
  bloquea el cierre de la US.

### B16. Tabla de costo por modelo sin trafico real para poblar
- **Que falta:** las columnas de latencia/tokens observados por variante del switch A/B
  necesitan trafico `/chat` real (aterriza con US-052, se reporta en US-065) y, para
  Qwen on-prem, el fix de `stream_options={"include_usage": True}` (B6).
- **Estado:** `docs/operations/finops.md` entrega la ESTRUCTURA de la tabla (variante,
  modelo real, precio, costo marginal) y el METODO de calculo, apuntando a la fuente
  real (`chat_service.py` logs `chat_model_resolved` / `chat_stream_finished`,
  `DoneEvent.usage`). No se rellenan numeros sinteticos (regla de Arthur).
- **Decision:** se puebla con trafico real de US-065/A7; no bloquea el cierre.

## US-060 - Drift detection con Evidently

### B10. No hay current set particionado por fecha (Plan B con datos reales)
- **Que falta:** el pipeline de ingesta del repo NO produce lotes particionados
  por fecha/trimestre, por lo que no existe un "current set" temporal real con el
  cual contrastar el "reference" del ano base.
- **Estado (HONESTO, datos reales):** el asset `drift_check` usa el parquet REAL
  `data/farslip/embeddings_pastis.parquet` (81 663 filas, embeddings FarSLIP +
  `class_id` en el espacio 18-clase de US-030) y construye el contraste Plan B
  con filas reales: `reference` = subpoblacion sin la clase mayoritaria,
  `current` = solo la clase mayoritaria. El drift detectado (KS en bandas,
  Chi-cuadrado en clases, MMD en embeddings) es un cambio de distribucion REAL
  entre subpoblaciones PASTIS, nunca sintetico. El asset queda parametrizado
  (`DEFAULT_CURRENT_PARQUET`) para apuntar a un trimestre fechado en cuanto exista.
- **Decision:** entregar el pipeline + asset + schedule + resource + tests
  ejecutados sobre datos reales; el current set temporal se conecta cuando la
  ingesta lo genere. No bloquea la US.

### B11. Subida a gs://agrosat-reports/drift/ — RESUELTO (bucket creado, subida verificada)
- **Estado (RESUELTO 2026-06-24):** el bucket `gs://agrosat-reports/` YA EXISTE
  (creado por el orquestador; contiene `drift/.keep`) y el ADC funciona en este
  entorno (`google.auth.default()` -> project `ine-ubica-tu-casilla`,
  cuenta `artzizumbo@gmail.com`). Ya no es blocker de "bucket no existe".
- **Evidencia de la subida real:** se corrio el asset `drift_check` sobre el
  corpus REAL (`data/farslip/embeddings_pastis.parquet`, 10 000 filas tras el
  slice Plan B) y subio el HTML a GCS:
  - `report_uploaded_gcs = True`
  - `report_url = gs://agrosat-reports/drift/2026-W26/report.html`
  - blob verificado vivo: `drift/2026-W26/report.html` size=36 491 283 bytes,
    `updated=2026-06-25T04:39:06Z` (`storage.Client().list_blobs`).
  - `drift_score = 0.9889`, `n_columns_drifted = 444/449`, `embedding_drift=True`,
    `alert_triggered=True`, `status=ok`, `data_version=...@531f58b3...`,
    `code_version=2c8dc2b`.
- **Degradacion limpia preservada:** sin ADC/bucket el asset sigue degradando a
  `data/monitoring/drift/report_{week}.html` local con `report_uploaded_gcs=false`
  (clasificado por `is_gcs_auth_error`); no se simula el acceso a GCS. Es decir,
  el camino feliz (sube) y el degradado (local) estan ambos verificados.
- **Divergencia de bucket resuelta:** el plan v8 cita `gs://agrosat-reports/`; la
  skill `agrosat-evidently-drift` cita `gs://agrosat-artifacts/`. Se usa el del
  plan v8 (`agrosat-reports`) por ser fuente de verdad.
- **Decision:** B11 cerrado. Subida a GCS funcional y demostrada; el degradado
  local sigue cubriendo dev/CI sin secrets.

### B12. Envio SMTP del drift_notifier no probado en dev
- **Que falta:** el envio real de email del resource `drift_notifier`
  (`dagster_project/resources/notification.py`) requiere un servidor SMTP
  configurado (`DRIFT_SMTP_*`).
- **Estado:** en dev/CI el notifier esta `enabled=False` por defecto y solo
  loggea el evento `drift_alert` via structlog (sin conexion SMTP). El trigger de
  la alerta cuando `drift_score > 0.3` se valida con un mock en
  `tests/dagster/test_drift_asset.py::test_drift_check_alert_triggers_notifier`.
  El envio SMTP real solo ocurre en prod con `DRIFT_SMTP_ENABLED=true`.
- **Decision:** envio SMTP real no verificado en esta sesion; el fallback
  structlog y el trigger por umbral si estan verificados. No bloquea la US.

## US-062 - Analisis de riesgos categorizados

### B15. `docs/STATUS.md` no existe (fuente de verdad alternativa)
- **Que falta:** el plan de US-062 (y de US-051) referencia `docs/STATUS.md` como
  fuente del estado de RLS y de las metricas del MVP. Ese archivo NO existe en el
  repositorio a la fecha (verificado: `ls docs/STATUS.md` -> no such file).
- **Estado:** `docs/risks/riesgos.md` usa como fuentes de verdad reales las
  migraciones aplicadas en `db/migrations/` (verificables con `dbmate status`),
  el §4/§5 del plan v8, ADR-009 y los resultados de EPIC 12 cerrada. No se
  inventa el contenido de un STATUS.md inexistente.
- **Decision:** documentar el faltante y seguir (regla de datos reales). Si se
  crea `docs/STATUS.md` mas adelante, enlazarlo desde el indice de docs.

### B16. Estado RLS del plan US-062 desactualizado (ya aplicado)
- **Observacion:** el plan de US-062 describia el riesgo R-EJE-02 como "riesgo
  vivo" asumiendo RLS pendiente (basandose en `db/CLAUDE.md`, que dice "cero RLS").
  La realidad VERIFICADA del repo es que la migracion
  `20260620000418_rls_multi_tenant.sql` (US-051) YA esta aplicada: rol de
  aplicacion `agrosat_app` NOSUPERUSER/NOBYPASSRLS, `FORCE ROW LEVEL SECURITY` y
  politica fail-closed sobre `chat_sessions`/`aois`/`parcels`/`features_parcels`.
- **Estado:** `riesgos.md` (R-ATK-04 y R-EJE-02) refleja el estado REAL: RLS
  aplicado; riesgo residual = la app debe conectar como `agrosat_app` (no como
  superusuario, que bypassa RLS) y queda deuda de backfill `parcels.session_id`
  NOT NULL + columna `session_id` denormalizada en `features_parcels`.
- **Decision:** el doc usa la migracion aplicada como fuente de verdad, no el
  `db/CLAUDE.md` (que esta desfasado). Conviene actualizar `db/CLAUDE.md` en una
  US futura para reflejar la RLS aplicada.

## US-063 - Comparativa de proveedores cloud (docs)

### B15. Precio IBM Cloud H100 (GPU-hora) no confirmado a la fecha
- **Que falta:** el precio por GPU-hora de H100 en IBM Cloud (VPC GPU, perfil
  acelerado `gx3` H100 NVL) no se pudo confirmar contra una pagina de pricing
  oficial de IBM. IBM Cloud es proveedor OPCIONAL en US-063 (la rubrica exige GCP
  vs Azure como minimo; AWS/IBM son referencia).
- **Re-investigacion 2026-06-25 (datos reales):** se repitio la busqueda web/docs.
  Resultado: **IBM no publica un precio oficial por GPU-hora de H100**. Fuentes
  consultadas y su veredicto:
  - ComputePrices.com (`/providers/ibm`): "We're actively tracking prices for IBM
    Cloud. Check back soon" -> sin cifra H100.
  - GPUPerHour (`gpuperhour.com`): IBM Cloud NO esta entre sus 28 proveedores.
  - Spheron 2026 (`/blog/gpu-cloud-pricing-comparison-2026/`, dato a 2026-05-14):
    no incluye IBM.
  - IntuitionLabs (`/articles/h100-rental-prices-cloud-comparison`, dato a
    2026-06-20): "IBM Cloud started offering H100 in late 2024 but pricing details
    are not widely published, so we omit them here".
  - Docs IBM VPC accelerated profiles (`cloud.ibm.com/docs/vpc?topic=vpc-accelerated-profile-family`):
    confirma familia `gx3` con H100 NVL, pero sin tarifa por hora.
  - Dato secundario suelto: una busqueda devolvio ~$0.99/GPU-h (H100 NVL, 8x) SIN
    respaldo de fuente oficial verificable -> se anota como NO confirmado y NO se
    usa como cifra de la comparativa (regla de datos reales).
- **Estado:** en `docs/cloud/comparativa_proveedores.md` la celda de precio H100 de
  IBM Cloud queda como "Sin precio oficial publicado / No confirmado", con la nota
  de re-investigacion + fuentes + fecha en §4.1 y §11. SIN numero fabricado. Los
  precios GCP/Azure/AWS si llevan fuente + fecha de consulta.
- **Decision:** completar la celda cuando IBM publique tarifa oficial; no bloquea
  la US (IBM es opcional y solo de referencia).

### B13. MMD: metodo real de Evidently 0.7.21 (no proxy) + version final confirmada
- **Observacion:** el AC pide MMD para embeddings AlphaEarth. Evidently 0.7.21 SI
  expone MMD nativo via `EmbeddingsDrift` (detector por defecto sobre el grupo de
  columnas de embedding); NO se usa un proxy PSI/Wasserstein. El valor que
  devuelve es un score de separabilidad en `[0, 1]` (~0.5 = nubes
  indistinguibles = sin drift). Verificado en el recon: identico ~0.48, shift de
  una clase ~0.79. El veredicto de drift se decide contra el umbral documentado
  `_EMBEDDING_MMD_DRIFT_THRESHOLD = 0.55` en `ml/monitoring/drift.py`.
- **Version final de Evidently (verificado 2026-06-24):** `pyproject.toml` pinea
  `evidently = "^0.7.21"` y `0.7.21` ES la ultima version estable publicada en
  PyPI (`pip index versions evidently` -> `INSTALLED: 0.7.21` == `LATEST: 0.7.21`).
  No hay version mas reciente que adoptar: el pin ya esta al dia, no se recomienda
  upgrade (no existe target superior). `plotly` queda pinneado a 5.x por el
  conflicto transitivo de evidently 0.7.21 (`>=5.10,<6`); subir plotly a 6.x solo
  cuando evidently lance una 0.8+ con soporte. No se toca el lock.

### B14. `dagster definitions validate` requiere postgres — camino concreto documentado
- **Observacion:** `dagster definitions validate -m dagster_project.definitions`
  intenta crear una instancia temporal que, por el `dagster.yaml` del repo
  (`run_storage`/`event_log_storage`/`schedule_storage` = `dagster_postgres.*` con
  `DAGSTER_PG_URL`), exige conexion a Postgres y falla sin esa env var. Es
  PRE-EXISTENTE (el `dagster.yaml` es de mayo, ajeno a US-060).
- **Estado:** las `Definitions` cargan y resuelven correctamente en proceso
  (`import dagster_project.definitions` -> OK; `defs.resolve_asset_graph()` lista
  los 16 assets incluido `drift_check` con sus deps `farslip_embeddings_consolidated`
  + `parcel_features_fused`, el schedule `drift_check_weekly_schedule` cron
  `0 6 * * 1` y el resource `drift_notifier`). Verificado ademas por
  `tests/dagster/test_drift_asset.py::test_drift_check_registered_in_definitions`.
- **Camino concreto para el current-set particionado por fecha (cierra el lazo
  B10+B14):** el asset `drift_check` NO requiere Postgres para correr (es lo que
  exige Postgres es la *instancia* Dagster — UI/daemon/`validate` CLI — por el
  `dagster.yaml`, no el asset). Dos caminos verificados:
  1. **Local, sin Postgres (lo usado en esta validacion):** invocar el asset con
     `build_asset_context` y mocks de `mlflow`/`drift_notifier` desde el cwd del
     repo (donde vive `data/farslip/embeddings_pastis.parquet`). Corre el Plan B
     real, escribe el HTML local y, con ADC presente, sube a GCS. Esto es lo que
     hacen los 4 tests de `tests/dagster/test_drift_asset.py` y la corrida manual
     de esta sesion (`report_uploaded_gcs=True`). NO re-entrena nada: solo lee el
     parquet ya versionado.
  2. **UI/schedule Dagster (requiere Postgres):** `export DAGSTER_PG_URL=...`
     (driver psycopg2 sincrono, distinto del `DATABASE_URL` asyncpg de la app) y
     `poetry run dagster dev -m dagster_project.definitions`. En la VM, la misma
     instancia Cloud SQL/Postgres del proyecto sirve; en local basta un Postgres
     contenedor (`postgres:15`) con la url de storage. El schedule semanal
     (`0 6 * * 1`) materializa entonces el asset igual que el camino 1.
- **Conexion del current-set fechado (B10):** cuando la ingesta produzca lotes por
  fecha/trimestre, apuntar `DEFAULT_CURRENT_PARQUET` en
  `dagster_project/assets/drift.py` al parquet del trimestre vigente. Mientras no
  exista, el contraste Plan B (clase mayoritaria vs resto, filas reales) es el
  current-set. Ningun camino re-entrena modelos.
- **Decision:** B14 cerrado a nivel de "como se corre". El asset corre en local
  sin Postgres (camino 1, demostrado); la UI/schedule necesitan Postgres
  (`DAGSTER_PG_URL`), gotcha de entorno, no error de definiciones.

## US-064 - Seguridad, Model Cards y glosario (docs)

### B17. Penetration test manual no ejecutado (sin endpoint staging live)
- **Que falta:** ejecutar `nmap` (escaneo de puertos) y `nikto` (escaneo HTTP)
  contra el deploy de staging, mas la verificacion manual de aislamiento por
  `session_id` (A01). El AC pide el pen-test manual basico pre-presentacion.
- **Estado:** no hay un deploy de staging live en esta sesion. El procedimiento
  queda documentado y listo en `docs/security.md` §5 (comandos `nmap`/`nikto` +
  check de aislamiento por sesion), con `STAGING_HOST` como placeholder a
  sustituir. NO se inventa una salida de escaneo (regla de datos reales).
- **Decision:** la ejecucion es un paso manual humano pre-deploy, alineado con la
  postura documentada de la skill `agrosat-security-audit`. No bloquea la US.

### B18. Estado RLS / `docs/STATUS.md` referenciado por el AC de seguridad
- **Que falta:** el AC de US-064 (heredado del plan v6/v8) cita `docs/STATUS.md`
  como fuente del estado RLS. Ese archivo NO existe en `docs/` (ya anotado en
  B15). El plan v8 describe `aois`/`parcels`/`features_parcels`/`chat_sessions`
  "sin RLS".
- **Estado:** la realidad VERIFICADA del repo es que la migracion
  `20260620000418_rls_multi_tenant.sql` (US-051) YA esta aplicada con politica
  fail-closed (ver B16). En la tabla OWASP de `docs/security.md`, A01 se marca
  como Deuda anotada por prudencia documental (el wiring de la app como rol
  `agrosat_app` NOSUPERUSER y el backfill de `parcels.session_id` NOT NULL siguen
  pendientes); el estado real RLS-aplicado se cita desde la migracion, no desde el
  STATUS.md inexistente.
- **Decision:** usar la migracion aplicada y B16 como fuente de verdad; no
  bloquea la US.

### B19. Metricas de cierre EPIC 6 parciales en las Model Cards
- **Observacion:** la Model Card del ensemble final (E6) cita solo las cifras que
  existen como artefacto real (`us043_farslip_summary.json`,
  `us043_farslip_stacking_blending.csv`, `us043_farslip_grid.csv`,
  `comparison_us040.csv`). El "Stacking +Gemma 4" del catalogo de arquitecturas
  se documenta como DISENO, no como run entrenado (Gemma LoRA OUT, ADR-011).
- **Estado:** toda metrica de las cards esta trazada a su CSV/JSON. Cualquier
  cifra del cierre formal de E6 que no este en disco queda marcada "pendiente de
  cierre E6" en la card; no se fabrica.
- **Decision:** sin claim de Gemma; cifras solo desde artefactos. No bloquea.

---

## Validacion 2026-06-25 (pasada de cierre US-059..067)

Cierre documental: generados los 9 us-resolved + 9 manual-test (antes inexistentes)
y actualizados los 9 handoffs a ready-to-close (US-065 ready-to-close con WARN).
Evidencia verificada contra entregables reales en disco; cifras cruzadas entre
docs (finops/costo-beneficio/model cards <-> CSVs reales).

### B-E10-V1. US-065 quedo WARN (deuda funcional + dependencias externas)
- **tokens Qwen None en streaming** - RESUELTO en codigo (2026-06-24, ver B6): se
  anadio `stream_options={"include_usage": True}` + `_usage_from_stream_event` en
  `VLLMOpenAIBackend.generate_stream`, propagando el usage real al `DoneEvent`
  igual que Gemini. 4 tests nuevos verdes. Queda pendiente solo la verificacion
  end-to-end contra el serving Qwen vivo en el H100 (infra externa).
- **MLflow :5010 no verificado en vivo** (server Docker caido en la VM en la
  sesion previa) y panel sin scrape: dependencias de infra externa no ejecutadas.
- Entrega completa y 13 tests unitarios verdes; el WARN refleja lo no verificable
  en sesion, no un fallo del entregable.

### B-E10-V2. Fallo intermitente de aislamiento de tests Prometheus (no reproducible aislado)
- Un subagente observo `Duplicated timeseries` en tests de integracion de `/chat`
  al correr ciertas combinaciones de suites juntas (registro Prometheus global de
  US-059 re-registrado). VERIFICADO esta sesion: `test_chat_endpoint.py` solo =
  12 passed; + `test_metrics_middleware.py` = 17 passed. NO reproducible de forma
  simple. Es flakiness de aislamiento de tests (registro global), NO un bug de
  produccion (el middleware funciona; los 5 tests de metrics pasan).
- **Accion recomendada**: usar un `CollectorRegistry` dedicado por test o
  `prometheus_client` fixture con reset entre suites. No bloqueante; deuda de
  higiene de tests.

### B-E10-V3. mypy metrics.py (de US-059) YA RESUELTO en EPIC 8
- El error `no-any-return` de `app/api/metrics.py` se corrigio en la rama de
  validacion (variable tipada explicita). Ver docs/blockers/epic8-notas.md B-E8-1.

---

## Validacion 2026-06-24 (cierre drift Evidently US-060 — bucket creado)

Pasada de cierre de los blockers de drift Evidently de EPIC 10, ahora que el
orquestador creo el bucket `gs://agrosat-reports/` (con `drift/.keep`) y el ADC
esta funcional en el entorno. Rama `fix/blockers-validacion-us040-077`.

### B11 (subida a GCS) — RESUELTO con evidencia real
- Se corrio el asset `drift_check` de verdad sobre el corpus REAL
  `data/farslip/embeddings_pastis.parquet` (invocacion directa con
  `build_asset_context` + mocks de `mlflow`/`drift_notifier`, sin Postgres).
- Resultado: `status=ok`, `week=2026-W26`, `rows=10000`, `drift_score=0.9889`,
  `n_columns_drifted=444/449`, `n_embedding_dims=64`, `embedding_drift=True`,
  `alert_triggered=True`, `data_version=...@531f58b3...`, `code_version=2c8dc2b`.
- **Subida a GCS confirmada:** `report_uploaded_gcs=True`,
  `report_url=gs://agrosat-reports/drift/2026-W26/report.html`. Blob verificado
  vivo via `storage.Client().list_blobs("agrosat-reports", prefix="drift/")`:
  `drift/2026-W26/report.html` size=36 491 283 bytes, updated=2026-06-25T04:39Z.
- El doc B11 (arriba) se actualizo: ya no es blocker de "bucket no existe".

### B13 (version Evidently) — sin upgrade pendiente
- `pip index versions evidently` -> `INSTALLED: 0.7.21` == `LATEST: 0.7.21`. El
  pin `^0.7.21` del `pyproject.toml` ya esta en la ultima estable. No se toca el
  lock; no hay upgrade recomendado (no existe target superior).

### B14 (current-set / Postgres) — camino concreto documentado
- El asset corre en LOCAL sin Postgres (camino 1, demostrado en B11 de esta
  corrida). Postgres (`DAGSTER_PG_URL`) solo lo necesita la instancia Dagster
  (UI/daemon/schedule/`validate` CLI), no el asset. NO se re-entreno nada (solo
  lectura del parquet ya versionado). Detalle en B14 (arriba).

### Tests de drift — 11 passed
- `pytest tests/ml/monitoring/test_drift.py tests/dagster/test_drift_asset.py -q`
  -> **11 passed** en 85.19s (7 del pipeline puro + 4 del asset Dagster). 0 fallos.
  Coincide con el "drift 11 tests" del tablero maestro `docs/VALIDACION-US040-077.md`.
