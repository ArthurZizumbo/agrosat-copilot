# Observabilidad de chat (US-065)

> EPIC 10 - Observabilidad y documentacion. Cierra las metricas estructuradas del
> endpoint `/chat` SSE (latencia por turno con SLO, conteo de tool-calls, tokens
> y modelo activo para FinOps) y documenta el lineage MLflow con sus dos
> almacenes. Regla de Arthur: datos reales, nada sintetico ni placeholders.

## 1. Que se mide por turno

Cada turno de `/chat` emite ahora una sola linea structlog canonica
`chat_turn_metrics` (ademas de las ya existentes `chat_stream_started`,
`chat_model_resolved` y `chat_stream_finished`). La emite
`backend/app/services/chat_service.py` (`_stream_reasoner`) a traves del helper
puro `backend/app/utils/chat_metrics.py`, observando los eventos del agente sin
alterar el frame SSE reenviado al cliente.

| Clave structlog | Origen | Significado |
|-----------------|--------|-------------|
| `chat_stream_started` | servicio | inicio del turno (`n_messages`, `has_parcel`, `has_aoi`) |
| `chat_model_resolved` | servicio | resolucion del reasoner (`variant`, `model`, `latency_ms` de lectura DB + build del backend) |
| `chat_turn_metrics` | helper US-065 | **metrica por turno** (ver campos abajo) |
| `chat_stream_finished` | servicio | fin end-to-end (`duration_ms`, `perceiver_observation_emitted`) |

Campos de `chat_turn_metrics`:

- `session_id`: sesion tenant (string).
- `turn_type`: `simple` (0 tool-calls) o `multi_step` (>=1 tool-call).
- `duration_ms`: latencia end-to-end del turno (mismo `time.perf_counter()` que
  `chat_stream_finished`; una sola fuente de verdad, no se re-mide).
- `slo_target_ms`: objetivo de SLO segun el tipo de turno (`3000` simple,
  `15000` multi-step).
- `slo_met`: `True` cuando `duration_ms <= slo_target_ms`.
- `tool_calls`: numero de eventos `tool_call` observados en el turno.
- `tokens_prompt` / `tokens_completion` / `tokens_total`: tokens reportados por
  el proveedor, o `None` cuando el proveedor no los entrega (nunca inventados,
  ver seccion 3).
- `model`: id concreto del reasoner (p.ej. `gemini-3.5-flash`).
- `variant`: variante LLM activa (`gemini` / `qwen-api` / `qwen-onprem` /
  `gemma`), clave de agrupacion FinOps (insumo del switch A/B de US-054).

### SLO de latencia

| Tipo de turno | Definicion | Objetivo p95 |
|---------------|------------|--------------|
| `simple` | 0 tool-calls | < 3 s (`3000 ms`) |
| `multi_step` | >= 1 tool-call | < 15 s (`15000 ms`) |

El campo `slo_met` evalua el turno individual. El SLO formal es p95: se calcula
agregando `chat_turn_metrics.duration_ms` por `turn_type` sobre una ventana (o el
panel Grafana, seccion 4). Validacion rapida sobre los logs estructurados:

```bash
# p95 de latencia de turnos simples (jq sobre el JSON renderer de staging/prod)
jq -rs '
  [ .[] | select(.event=="chat_turn_metrics" and .turn_type=="simple") | .duration_ms ]
  | sort | .[(length*0.95|floor)]
' logs/chat.jsonl
```

## 2. Modelo activo y FinOps

`variant` + `model` cruzan el costo por modelo (Gemini Pro/Flash vs Qwen on-prem
vs Gemma). Para el reporte FinOps (insumo de US-067) se agregan `tokens_total` y
`tool_calls` por `model`/`variant`. Cifras de referencia del proyecto: operativo
objetivo ~115 USD/mes (Cloud Run scale-to-zero), Gemini API en centavos
(~0.0001 USD/descripcion FarSLIP), H100 del sponsor 24/7 sin costo al equipo.

## 3. Tokens: reales o `None`, nunca sinteticos

Para reportar tokens reales se propaga el `usage` del proveedor por el contrato
de eventos, sin inventar nada:

- `ml/agent/backends.py`: `BackendChunk` gana un campo opcional `usage`
  (`prompt_tokens` / `completion_tokens` / `total_tokens`). `GeminiBackend` lo
  puebla desde `response.usage_metadata` (Gemini lo entrega en la respuesta
  completa no-streaming que ya usa el backend: `prompt_token_count`,
  `candidates_token_count`, `total_token_count`). Se adjunta solo al ultimo chunk
  del turno.
- `ml/agent/events.py`: `DoneEvent` gana `usage: dict[str, int] | None = None`
  (sigue `frozen` + `extra="forbid"`; al ser opcional no rompe la union ni el
  round-trip).
- `ml/agent/agent.py`: el loop acumula el `usage` del ultimo chunk y lo adjunta
  al `DoneEvent` terminal.

Limitacion conocida (honesta): el backend OpenAI-compatible de vLLM/Qwen
(`VLLMOpenAIBackend`) NO emite `usage` en modo streaming a menos que se envie
`stream_options={"include_usage": True}`, que hoy no se setea. En esa ruta
`tokens_*` queda `None`. No se inventan tokens. Detalle y plan en
`docs/blockers/epic10-notas.md`.

## 4. Prometheus + panel Grafana "latencia chat p95"

El helper expone, de forma perezosa y gateada por
`settings.chat_metrics_prometheus_enabled` (default `False`):

- `chat_turn_duration_seconds` (Histogram, label `turn_type`) - alimenta el panel
  p95.
- `chat_tool_calls_total` (Counter, labels `model`, `variant`).
- `chat_tokens_total` (Counter, labels `model`, `variant`).

El import de `prometheus_client` es perezoso: si el paquete no esta o el registry
choca, el export es un no-op con `logger.debug` y el chat NUNCA se cae por la
observabilidad (degradacion honesta, US-065 R3). `prometheus-client` ya esta en
`pyproject.toml` (`^0.25.0`) e instalado en el entorno.

Dashboard: `infrastructure/grafana/chat_observability.json`. PromQL del panel
principal:

```promql
# Latencia p95 de turnos de chat por tipo (simple/multi_step)
histogram_quantile(
  0.95,
  sum(rate(chat_turn_duration_seconds_bucket[5m])) by (le, turn_type)
)
```

```promql
# Tool-calls por minuto por modelo/variante
sum(rate(chat_tool_calls_total[5m])) by (model, variant)

# Tokens por minuto por modelo/variante (FinOps)
sum(rate(chat_tokens_total[5m])) by (model, variant)
```

El scaffolding del exporter HTTP `/metrics` y el wiring del datasource Prometheus
los aporta US-059 (`backend/app/middleware/metrics.py`); este panel se conecta a
esa fuente.

## 5. Lineage MLflow y el gotcha de los dos almacenes

El lineage de experimentos vive en el **server MLflow Docker en `:5010`** (no en
`./mlruns`). El helper `ml/utils/mlflow_utils.py::track_experiment` (US-019) ya es
la unica via para abrir runs y **escribe los tags obligatorios `code_version`
(git SHA via `ml/utils/git_meta.py::git_sha`) y `data_version` (hash del `.dvc`
via `dvc_data_version`)**. No se reimplementa: esta US verifica y documenta.

Resolucion del tracking URI (`resolve_tracking_uri`, prioridad):

1. `override` explicito.
2. `MLFLOW_TRACKING_URI`.
3. `http://localhost:5010` si responde a `/health`.
4. `file:./mlruns` como fallback (dev sin Docker / CI), con `logger.warning`.

### Gotcha de los dos almacenes

- **Server Docker Postgres** `http://localhost:5010` (contenedor
  `agro_sat_copilot-mlflow-1`, `make mlflow-up`): almacen REAL, de donde leen los
  notebooks y donde `resolve_tracking_uri` apunta si `/health` responde.
- **Filestore local `./mlruns/`**: almacen VIEJO y fragmentado (llega a tener dos
  directorios `agrosat-segmentation` con IDs distintos). Es solo el fallback
  cuando el server esta caido. NO confundirlo con el lineage real.
- **Runs por subprocess quedan `RUNNING`**: cuando un notebook lanza el
  entrenamiento por subprocess, el hijo no ejecuta el `__exit__` de
  `track_experiment` y el run queda en `status=RUNNING` (a veces contra el server
  equivocado). Las metricas y los tags SI estan; hay que cerrarlos a mano con
  `client.set_terminated(run_id, status="FINISHED")`.

Procedimiento de verificacion de tags contra `:5010` (no requiere el server vivo
para que el codigo/los tests pasen; si esta caido, degrada a `file:./mlruns`):

```bash
make mlflow-up   # levanta el server Docker en :5010
poetry run python - <<'PY'
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5010")
client = MlflowClient()
for exp in client.search_experiments():
    runs = client.search_runs([exp.experiment_id], max_results=1,
                              order_by=["attributes.start_time DESC"])
    for r in runs:
        tags = r.data.tags
        print(exp.name, r.info.run_id,
              "code_version=", tags.get("code_version"),
              "data_version=", tags.get("data_version"),
              "status=", r.info.status)
PY
```

Un run sano debe traer `code_version` (SHA git) y `data_version` (`<path>@<md5>`
o `untracked`). Un `status=RUNNING` viejo es candidato a cerrar (ver gotcha).

## 6. Validacion con flujo sintetico

Mientras no haya trafico real masivo, la instrumentacion se valida con un agente
stub que emite una secuencia conocida de eventos
(`tool_call` x N -> `tool_result` -> `text_delta` -> `done` con/sin `usage`) y se
asserta el `chat_turn_metrics` resultante. Esto es flujo de prueba controlado, NO
datos sinteticos de produccion: valida la instrumentacion, no falsea metricas.

- `backend/tests/unit/test_chat_metrics.py`: helper puro (clasificacion
  simple/multi_step, SLO, agregacion de tool-calls, tokens ausentes -> `None`,
  emisor sin `print`, export Prometheus gateado).
- `backend/tests/unit/test_chat_sse.py`: flujo end-to-end del servicio
  (`chat_turn_metrics` con `n_tool_calls`, `turn_type`, `slo_met`, tokens reales
  o `None`) sin red ni DB.
