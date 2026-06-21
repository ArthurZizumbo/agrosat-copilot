# EPIC 10 - Notas y blockers (Observabilidad, Docs, Multi-region)

Registro de lo que NO se pudo verificar/correr en el entorno de desarrollo de
esta sesion. Politica (regla de Arthur): si algo no se puede verificar/correr,
se anota aqui y se SIGUE. Ninguno de estos puntos bloquea la entrega del
scaffolding.

## US-061 - Analisis costo-beneficio (A6/A7)

### B17. Export .xlsx no generado en esta sesion (entregable de respaldo)
- **Que falta:** el entregable "tablas en Excel" del AC pide un `.xlsx`. No se genero
  programaticamente en esta sesion (la US es solo-documentacion, sin codigo de
  aplicacion, y no se confirmo `openpyxl`/`xlsxwriter` disponible en el entorno).
- **Estado (datos reales):** se entregan como respaldo verificable los `.csv` fuente
  al Git (`docs/business/data/costos_crisp_ml.csv` y `beneficios_500ha.csv`), las
  tablas Markdown en `docs/business/costo_beneficio.md` y el export LaTeX
  `docs/business/costo_beneficio.tex` para el paper. Ninguna cifra es sintetica:
  costos anclados a `docs/operations/finops.md` y a la factura GCP real; beneficios
  marcados como estimaciones de literatura con su supuesto y rango.
- **Decision:** generar el `.xlsx` desde los `.csv` cuando se confirme la dependencia
  (`polars` + `openpyxl`) en el entorno; no bloquea la entrega del documento de negocio.

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

### B6. Tokens de Qwen/vLLM no disponibles en streaming (R1)
- **Que falta:** el backend OpenAI-compatible de vLLM/Qwen
  (`ml/agent/backends.py::VLLMOpenAIBackend`) NO emite `usage` en modo streaming
  salvo que se envie `stream_options={"include_usage": True}` en
  `chat.completions.create`, lo que hoy no se setea. En esa ruta el
  `chat_turn_metrics` reporta `tokens_prompt/completion/total = None`.
- **Estado (HONESTO):** Gemini SI entrega tokens reales: `GeminiBackend` lee
  `response.usage_metadata` (`prompt_token_count` / `candidates_token_count` /
  `total_token_count`) de la respuesta completa no-streaming que ya usa, y los
  propaga via `BackendChunk.usage` -> `DoneEvent.usage` ->
  `chat_turn_metrics.tokens_*`. Para vLLM/Qwen quedan `None`: **no se inventan**
  (regla de datos reales). El resto de la US (latencia + SLO + tool-calls +
  modelo activo) se entrega completo en ambas rutas.
- **Decision / fix pendiente:** anadir `stream_options={"include_usage": True}` a
  la llamada de `VLLMOpenAIBackend.generate_stream` y leer el `usage` del ultimo
  chunk del stream (vLLM lo emite en un chunk final con `choices=[]`). Pendiente
  de verificar contra el endpoint vLLM real del H100; no bloquea la entrega.

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

### B11. Subida a gs://agrosat-reports/drift/ requiere ADC (degrada a local)
- **Que falta:** publicar el HTML semanal en `gs://agrosat-reports/drift/{week}/`
  necesita credenciales de aplicacion (ADC) y que el bucket exista.
- **Estado:** sin ADC/bucket, `drift_check` escribe el reporte en
  `data/monitoring/drift/report_{week}.html` (local) y degrada la subida sin
  fallar (`is_gcs_auth_error` clasifica el fallo). La metadata expone
  `report_uploaded_gcs=false` y `report_url` apunta al path local. No se simula
  el acceso a GCS.
- **Divergencia de bucket resuelta:** el plan v8 cita `gs://agrosat-reports/`; la
  skill `agrosat-evidently-drift` cita `gs://agrosat-artifacts/`. Se usa el del
  plan v8 (`agrosat-reports`) por ser fuente de verdad.
- **Decision:** subida a GCS pendiente de ADC + bucket; local funciona en dev/CI.

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
- **Que falta:** el precio por GPU-hora de H100 en IBM Cloud (VPC GPU) no se pudo
  confirmar contra una pagina de pricing oficial de IBM a la fecha de redaccion
  (consultado 2026-06-20). IBM Cloud es proveedor OPCIONAL en US-063 (la rubrica
  exige GCP vs Azure como minimo; AWS/IBM son referencia).
- **Fuentes intentadas:** busqueda web de pricing IBM Cloud H100 / VPC GPU. Los
  agregadores devolvieron datos para GCP/Azure/AWS con cifra y fecha, pero no una
  fuente oficial IBM con precio H100 por GPU-hora verificable.
- **Estado (regla de datos reales):** en
  `docs/cloud/comparativa_proveedores.md` la celda de precio H100 de IBM Cloud
  queda como "No confirmado a la fecha — ver §11", SIN numero fabricado. Los
  precios GCP/Azure/AWS si llevan fuente + fecha de consulta.
- **Decision:** completar la celda cuando exista fuente oficial IBM; no bloquea la
  US (IBM es opcional y solo de referencia).

### B13. MMD: metodo real de Evidently 0.7.21 (no proxy)
- **Observacion:** el AC pide MMD para embeddings AlphaEarth. Evidently 0.7.21 SI
  expone MMD nativo via `EmbeddingsDrift` (detector por defecto sobre el grupo de
  columnas de embedding); NO se usa un proxy PSI/Wasserstein. El valor que
  devuelve es un score de separabilidad en `[0, 1]` (~0.5 = nubes
  indistinguibles = sin drift). Verificado en el recon: identico ~0.48, shift de
  una clase ~0.79. El veredicto de drift se decide contra el umbral documentado
  `_EMBEDDING_MMD_DRIFT_THRESHOLD = 0.55` en `ml/monitoring/drift.py`.

### B14. `dagster definitions validate` requiere postgres (pre-existente)
- **Observacion:** `dagster definitions validate -m dagster_project.definitions`
  intenta crear una instancia temporal que, por el `dagster.yaml` del repo
  (storage postgres + `DAGSTER_PG_URL`), exige conexion a Postgres y falla sin
  esa env var. Es PRE-EXISTENTE (el `dagster.yaml` es de mayo, ajeno a US-060).
- **Estado:** las `Definitions` cargan y resuelven correctamente en proceso
  (`import dagster_project.definitions` -> OK; `defs.resolve_asset_graph()` lista
  los 16 assets incluido `drift_check` con sus deps `farslip_embeddings_consolidated`
  + `parcel_features_fused`, el schedule `drift_check_weekly_schedule` cron
  `0 6 * * 1` y el resource `drift_notifier`). Verificado ademas por
  `tests/dagster/test_drift_asset.py::test_drift_check_registered_in_definitions`.
- **Decision:** la validacion en proceso es suficiente; el `validate` CLI requiere
  levantar Postgres (gotcha de entorno), no es un error de definiciones.

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
