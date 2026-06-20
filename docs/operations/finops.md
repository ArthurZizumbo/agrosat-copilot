# FinOps — Presupuesto y control de costos AgroSatCopilot

**Corte:** 20-jun-2026 · **Mantenedor:** Arthur Zizumbo (MLOps lead)

> Contenido movido desde `CLAUDE.md` raíz (que ahora es guía operacional, sin plan/presupuesto). El plan completo vive en [`context/RefinamientoPlaneacionAgroSatCopilot_v8.md`](../../context/RefinamientoPlaneacionAgroSatCopilot_v8.md).

## Presupuesto objetivo

| Concepto | Monto | Nota |
|----------|-------|------|
| Operativo mensual | **~$115 USD/mes** | Con scale-to-zero (Cloud Run `min_instances=0`) |
| Training único (cuando aplicaba spot) | $262 (spot) — $602 (on-demand) | Histórico; la H100 del sponsor ahora es 24/7 sin costo para el equipo |
| GCP acumulado a la fecha | ~$0.30-0.49 USD | Holgado |
| Gemini API (descripciones FarSLIP + chat) | centavos | ~$0.0001/descripción; cabe en el operativo |

### Caveat de créditos (importante)

El "Trial credit for GenAI App Builder" ($17,178) que aparece en la consola es un crédito de
**Vertex AI Search / Agent Builder** (la línea de discovery/search). **NO** cubre la SKU de
generación de texto de la **Gemini API** (las llamadas del reasoner del copiloto y las
descripciones FarSLIP facturan contra una SKU distinta). Es decir: el saldo de $17,178 no se
descuenta cuando el reasoner genera texto, y a la inversa, el gasto de generación no consume ese
crédito. No se asume cobertura de ese crédito en ninguna estimación de este documento. En la
práctica no se necesita: la generación de texto cuesta centavos (ver tabla de presupuesto).

## Cómputo

- **H100 NVL 96GB**: prestada por el sponsor, 24/7 (no apagar). VM `gjcamacho-gpuh1`. Acceso: [`docs/infra/acceso-vm-h100-tunnel.md`](../infra/acceso-vm-h100-tunnel.md).
- **GCP L4 24GB** (`agrosat-farslip-trainer-dev`): spot con daemon de auto-shutdown por idle. Pararlo antes de runs manuales largos.

## Palancas de ahorro reproducibles

Cada palanca ya está aplicada en el proyecto. Se enlaza al artefacto vivo (variable Terraform,
script o target Make) en lugar de transcribir instrucciones, para evitar drift documental.

### 1. Cloud SQL dev apagada (`activation_policy=NEVER`)

- **Qué:** la instancia Cloud SQL de dev (`agrosat-pg-dev`) está detenida preservando los datos,
  para no facturar cómputo cuando nadie la usa.
- **Cómo se reproduce:** la palanca vive en código, no en un click de consola. La variable
  Terraform [`db_activation_policy`](../../infrastructure/terraform/modules/gcp/variables.tf)
  (default `ALWAYS`, validada a `ALWAYS`/`NEVER`) se pone en `NEVER`; la consume el módulo en
  [`infrastructure/terraform/modules/gcp/main.tf`](../../infrastructure/terraform/modules/gcp/main.tf)
  y se fija para el entorno en
  [`infrastructure/terraform/environments/dev/main.tf`](../../infrastructure/terraform/environments/dev/main.tf).
- **Por qué importa:** apagar la instancia desde la consola provoca *drift* (el siguiente
  `terraform apply` la vuelve a encender). Declararlo en TF mantiene `NEVER` idempotente.

### 2. Shrink del disco `farslip-data` (250 → 125 GB)

- **Qué:** el disco persistente de datos FarSLIP se redujo de 250 GB a 125 GB (ahorro directo en
  la SKU de almacenamiento persistente).
- **Cómo se reproduce:** GCP **no** encoge discos in-place, así que el procedimiento real fue
  snapshot → crear disco nuevo de 125 GB → `rsync` de los datos → reimportar el disco nuevo a
  Terraform (`terraform import` + actualizar `size`). Es una receta manual documentada en la
  memoria del proyecto (`disk-shrink-finops-procedure`), no un target automatizado: se ejecuta
  una sola vez y se versiona el tamaño final en TF.

### 3. Daemon de auto-shutdown de la VM L4 por idle (Pub/Sub, no GPU)

- **Qué:** la VM L4 spot (`agrosat-farslip-trainer-dev`) corre un daemon (`farslip-vm-daemon`)
  que la apaga sola cuando lleva un rato ociosa, evitando facturar GPU spot olvidada encendida.
- **Cómo se reproduce:** el daemon decide *idle* por **ausencia de mensajes Pub/Sub de trabajo**,
  **no** por utilización de GPU (un entrenamiento puede tener la GPU baja sin estar ocioso). Por
  eso, antes de un run manual largo hay que **pararlo** para que no apague la VM a media corrida
  (gotcha documentado en la memoria del proyecto, `l4-vm-idle-shutdown-daemon`).

### 4. Cloud Run scale-to-zero (`min_instances=0`)

- **Qué:** los servicios Cloud Run (api, frontend, tiling, inference-worker) escalan a cero
  instancias sin tráfico; es la base del objetivo operativo ~$115/mes.
- **Cómo se reproduce / verifica:** `make scale-to-zero-check` lista los servicios con su
  `autoscaling.knative.dev/minScale` para confirmar que está en 0. Nota: con `min_instances=0`
  el `/metrics` por *pull* de Prometheus no es estable en prod (la instancia muere entre
  requests); la vía realista es push gateway o Cloud Monitoring (ver
  [`docs/blockers/epic10-notas.md`](../blockers/epic10-notas.md) B3).

## Comandos y scripts permanentes

| Comando | Script / acción | Para qué |
|---------|-----------------|----------|
| `make cost-audit` | [`scripts/cost_audit.sh`](../../scripts/cost_audit.sh) | Reporte de costos GCP + Azure |
| `make scale-to-zero-check` | `gcloud run services list` (minScale) | Verifica `min_instances=0` en Cloud Run |
| `make azure-h100-start` | [`scripts/azure_h100_start.sh`](../../scripts/azure_h100_start.sh) | Arranca la VM H100 (módulo de referencia) |
| `make azure-h100-stop` | [`scripts/azure_h100_stop.sh`](../../scripts/azure_h100_stop.sh) | Detiene la VM H100 |
| `make azure-h100-status` | [`scripts/azure_h100_status.sh`](../../scripts/azure_h100_status.sh) | Estado de la VM H100 |

```bash
make cost-audit            # reporte de costos GCP + Azure
make scale-to-zero-check   # verifica min_instances=0 en Cloud Run
make azure-h100-status     # estado de la VM H100 (start/stop son de referencia: hoy es 24/7)
```

> Los targets Make viven en el `Makefile` (líneas 317-324 para `azure-h100-*`, 392-396 para
> `cost-audit` y `scale-to-zero-check`). Los scripts `azure_h100_{start,stop,status}.sh` son de
> referencia: la H100 del sponsor está 24/7 y no se apaga (ver arriba). Skill operativa asociada:
> [`agrosat-finops`](../../.claude/skills/agrosat-finops/SKILL.md) (auditoría de costos, alertas
> de presupuesto, verificación de scale-to-zero, monitoreo de precio spot).

## Costo por modelo en el switch A/B (LLM)

El switch A/B de variantes del reasoner (US-054) persiste por sesión qué modelo se usa, y la
instrumentación de chat (paraguas US-059, reportada en US-065) emite los logs que permiten cruzar
**costo por modelo × latencia/tokens observados**. Esta sección define la estructura y el método;
los números observados se pueblan con tráfico real (regla de datos reales: no se fabrican).

**Fuente real de los datos** — `backend/app/services/chat_service.py` ya loggea con structlog:

- `chat_model_resolved` (`session_id`, `variant`, `model`, `latency_ms`): qué variante/modelo
  resolvió el reasoner para el turno y cuánto tardó la resolución.
- `chat_stream_finished` (`session_id`, `duration_ms`): duración total del turno.
- El evento terminal `done` del reasoner propaga `usage` cuando el backend lo expone
  (`prompt_tokens` / `completion_tokens` / `total_tokens`, ver `ml/agent/events.py::DoneEvent`).
  Gemini SÍ entrega tokens reales (`response.usage_metadata`); vLLM/Qwen en streaming devuelve
  `None` salvo que se setee `stream_options={"include_usage": True}` (pendiente, ver
  [`docs/blockers/epic10-notas.md`](../blockers/epic10-notas.md) B6). No se inventan tokens.

**Variantes reales del switch** (`ml/agent/llm_routing.py`, `VARIANTS`):
`gemini` (default), `qwen-api`, `qwen-onprem`, `gemma`.

| Variante | Modelo (real) | Precio entrada / salida | Costo marginal | Notas |
|----------|---------------|-------------------------|----------------|-------|
| `gemini` (default) | `gemini-3.5-flash` (deviation consciente de Arthur por costo/latencia frente a Pro) | SKU Gemini API (flash, centavos por turno) | bajo | El plan fija Gemini 2.5 Pro como reasoner de referencia ($1.25/$10 por M tokens entrada/salida); Flash se usa en dev por costo/latencia |
| `gemini` (referencia Pro) | Gemini 2.5 Pro | $1.25 / $10 por M tokens | — | Cifra del plan v8; usar solo si se conmuta a Pro |
| `qwen-onprem` | Qwen3.5-35B-A3B vLLM GPTQ-Int4 (single-GPU) | $0 marginal | $0 (salvo energía) | Corre en la H100 del sponsor 24/7; sin costo de API para el equipo |
| `qwen-api` | Qwen vía endpoint OpenAI-compatible | según endpoint | variable | Solo si se apunta a un proveedor de API externo |
| `gemma` | Gemma 4 LoRA (Ollama) | — | — | FUTURE (ADR-009/ADR-011); fuera de alcance operativo actual |

**Método de cálculo del dashboard:** por cada turno, `costo_turno = (prompt_tokens/1e6 *
precio_in) + (completion_tokens/1e6 * precio_out)`; se agrega por `variant` y se cruza con el p50/p95
de `latency_ms` / `duration_ms` del mismo `variant`. Así el dashboard de FinOps del switch A/B
muestra, lado a lado, **costo/1k tokens vs latencia observada** por modelo, para decidir el
trade-off (Flash barato y rápido vs Pro caro y de mayor calidad vs Qwen on-prem $0 marginal).

> **Estado a la fecha (HONESTO):** la celda de tokens de Qwen en streaming queda `None` hasta
> aplicar el fix de `include_usage` (B6) y el dashboard se puebla con tráfico `/chat` real
> (aterriza con US-052, se reporta en US-065; ver B4/B8 en
> [`docs/blockers/epic10-notas.md`](../blockers/epic10-notas.md)). No se rellena con valores
> sintéticos. La estructura y el método quedan listos para poblarse con tráfico real.
