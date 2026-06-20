# Observabilidad Grafana / Prometheus (US-059)

Scaffolding de observabilidad tecnica de AgroSatCopilot: un endpoint `/metrics`
Prometheus expuesto por la API FastAPI mas plantillas Grafana versionadas y
reglas de alerta. **Alcance de esta US: el scaffolding, no paneles poblados con
trafico real.** Los paneles de chat se pueblan en US-065 cuando exista trafico
`/chat` real (US-052); aqui no se fabrica trafico sintetico ni numeros demo
(regla de datos reales).

## Que exporta la API

El middleware `backend/app/middleware/metrics.py` instrumenta cada request real
del proceso y expone tres metricas Prometheus:

| Metrica | Tipo | Labels | Uso |
|---|---|---|---|
| `http_request_duration_seconds` | Histogram | `method`, `path`, `status` | p50/p95/p99 via `histogram_quantile` |
| `http_requests_total` | Counter | `method`, `path`, `status` | RPS via `rate()`, error rate con `status=~"5.."` |
| `http_request_exceptions_total` | Counter | `method`, `path` | excepciones no manejadas |

El label `path` es la **plantilla de ruta** (p. ej. `/aois/{aoi_id}`), nunca la
URL cruda, para evitar una explosion de cardinalidad por `session_id` /
`parcel_id`. El propio endpoint `/metrics` se excluye del histograma para no
contaminar los percentiles con el trafico del scraper.

## Dashboards

Tres plantillas JSON en `dashboards/`, todas con datasource por variable
`${DS_PROMETHEUS}` (no se hardcodea ningun UID):

- **`api.json`** — latencia p50/p95/p99, RPS por endpoint, error rate 5xx,
  excepciones, y el slot "latencia chat p95" (lo puebla US-065 con trafico
  `/chat` real). Panel poblado por las metricas reales del middleware.
- **`worker_ml.json`** — GPU utilization, VRAM usada/% y latencia de inferencia.
  **Plantilla:** requiere `dcgm-exporter` scrapeando la H100 NVL (sponsor) o la
  L4 spot; sin scrape activo en este entorno (ver
  `docs/blockers/epic10-notas.md`). Metricas esperadas: `DCGM_FI_DEV_GPU_UTIL`,
  `DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_FB_FREE`.
- **`data_pipeline.json`** — materializaciones de assets Dagster, skips
  `skipped_no_gcs`/`skipped_no_upstream` y drift score de embeddings.
  **Plantilla:** las metricas Dagster (`dagster_asset_materializations_total`,
  `dagster_asset_skips_total`) y el `drift_share` de US-060 se exportan via push
  gateway cuando aterricen.

### Importar en Grafana

1. Grafana -> Dashboards -> Import -> Upload JSON file (o pegar el contenido).
2. Al importar, Grafana pide el datasource para `DS_PROMETHEUS`: seleccionar el
   datasource Prometheus que scrapea la API.
3. Repetir para los tres JSON. Los UID (`agrosat-api`, `agrosat-worker-ml`,
   `agrosat-data-pipeline`) son estables para enlazar entre paneles.

## Alertas

`alerts/alert_rules.yml` (reglas Prometheus, validas para Alertmanager o para la
importacion de reglas de Grafana) cubre los tres umbrales del AC:

- **`HighP99Latency`** — p99 de `http_request_duration_seconds` > 3 s (5m).
- **`HighErrorRate`** — fraccion 5xx > 5% (5m).
- **`GpuOutOfMemory`** — VRAM usada > 95% (2m), via `dcgm-exporter`.

El equivalente Cloud Monitoring (alert policies + notification channel email)
queda documentado al final del YAML. El wiring a un canal real (email /
Alertmanager / Cloud Monitoring) requiere un proyecto desplegado -> registrado
en `docs/blockers/epic10-notas.md`.

## Estrategia de scrape (Cloud Run scale-to-zero)

Cloud Run con `min_instances=0` (objetivo FinOps ~$115/mes) **no es scrapeable**
de forma estable por un Prometheus pull: la instancia muere entre requests. Las
vias realistas para produccion:

- **Push gateway** — la variable `PROMETHEUS_PUSHGATEWAY` ya esta declarada en
  `backend/app/core/config.py` (`Settings.prometheus_pushgateway`). El proceso
  empuja las metricas antes de escalar a cero.
- **Cloud Monitoring** — exportar via Managed Service for Prometheus.

El endpoint `/metrics` (pull) queda util en dev/staging always-on y para los
tests. En produccion `/metrics` no se expone publico: se restringe por
ingress de red / Cloud Run, no por auth a nivel de app (no lleva datos de
sesion, por eso tampoco se filtra por `session_id` ni se aplica rate-limit).

## Nota multi-worker

`prometheus_client` usa un registro global por proceso. Con un solo worker
uvicorn (el objetivo en Cloud Run) los contadores son correctos. Con
`workers > 1` hay que activar el modo multiprocess
(`PROMETHEUS_MULTIPROC_DIR`) o usar el push gateway, o los contadores quedarian
partidos entre workers.
