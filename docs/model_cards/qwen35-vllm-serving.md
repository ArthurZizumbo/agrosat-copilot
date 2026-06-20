# Model Card — Qwen3.5-35B-A3B (serving on-prem, variante B)

**Serving** (no fine-tune) del reasoner on-premise del copiloto: el LLM Qwen
MoE-A3B Int4 servido en la H100 NVL 96GB del sponsor, endpoint
OpenAI-compatible, intercambiable con Gemini desde el mismo cliente
(`/llm/switch`). Sustenta la historia de **soberania de datos** (cooperativas
que no exportan datos a la nube). Esta card **reemplaza** la de Gemma 4
fine-tuned (Gemma LoRA OUT, [ADR-011](../decisions/ADR-011-gemma4-lora-future.md)).
En espanol neutro.

---

## 1. Model Details

- **Modelo servido**: `Qwen/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4` (MoE 30B
  totales / ~3B activos por token, cuantizado GPTQ-Int4 single-GPU).
- **Tipo**: **serving**, no fine-tune. No hay pesos LoRA propios; se sirve el
  modelo cuantizado tal cual. El id comercial "Qwen3.5-35B-A3B" referencia esta
  familia MoE-A3B del reasoner on-prem (variante B del switch A/B).
- **Rol**: reasoner alternativo a Gemini (variante A cloud), seleccionable por
  sesion desde `/llm/switch` (US-054).
- **Endpoint**: OpenAI-compatible `/v1/chat/completions`, consumido por
  `ml/agent/backends.py::VLLMOpenAIBackend` (US-047). El cliente del agente
  funciona con cualquier servidor OpenAI-compatible — solo cambia la URL.

## 2. Intended Use

Reasoner on-prem del agente conversacional cuando el cliente no debe exportar
datos a la nube (soberania de datos). Plan-and-React con las 9 FunctionTools
geoespaciales. **Fuera de alcance**: cargas de fine-tune (es serving puro) y uso
sin GPU compatible.

## 3. Serving — dos vias

| Via | Estado en la VM actual | Endpoint |
|---|---|---|
| **llama.cpp** (nativo Windows + CUDA) | **OPERATIVA** para la demo 27-jun (GGUF Q4_K_M ~18.6GB, `--parallel N`) | `:8002/v1/chat/completions` |
| **vLLM** (Linux, GPTQ-Int4) | **BLOQUEADA** en la VM Windows del sponsor (sin nested virt: ni WSL2 ni Docker) | (cuando se desbloquee) |

Para la Presentacion Final del 27-jun la via operativa es **llama.cpp nativo**
(`llama-server.exe` con CUDA contra la H100). Detalle de setup y arranque en
[docs/serving/qwen35.md](../serving/qwen35.md) (US-048).

## 4. Hardware / FinOps

- **GPU**: H100 NVL 96GB, VM `gjcamacho-gpuh1`, **prestada por el sponsor, 24/7,
  sin costo al equipo** (no apagar). El training historico unico costo entre
  $262 (spot) y $602 (on-demand); ahora el computo intensivo corre en la H100 sin
  costo. Objetivo operativo ~$115 USD/mes (Cloud Run scale-to-zero).

## 5. Metrics

No aplica una metrica de calidad de modelo en esta card: es **serving** de un
modelo de terceros, no un entrenamiento propio. Las metricas de **observabilidad
de chat** (latencia p95, tool-call success, modelo activo) se instrumentan en
US-065. **Honestidad**: el backend vLLM/Qwen OpenAI-compatible NO emite `usage`
(tokens) en streaming salvo que se envie `stream_options={"include_usage": True}`
(hoy no seteado); en esa ruta `chat_turn_metrics.tokens_*` queda `None`
(no se inventa). Gemini si entrega tokens reales via `usage_metadata`. Ver
blocker [§ B6](../blockers/epic10-notas.md).

## 6. Limitations & Ethical Considerations

- vLLM esta bloqueado en la VM Windows del sponsor; la demo usa llama.cpp.
- Tokens de Qwen no disponibles en streaming sin `include_usage` (fix pendiente,
  blocker B6).
- Modelo de terceros; las consideraciones eticas del modelo base aplican.

## 7. Licenses & Attribution

- **Qwen3 (familia MoE-A3B)**: licencia **Apache 2.0**.
- Atribucion: Qwen Team (Alibaba). Sin fine-tune propio que requiera atribucion
  adicional.

## 8. Reproducibility / MLflow

- Es serving: no produce runs de entrenamiento MLflow. La reproducibilidad es la
  del despliegue (GGUF + flags de `llama-server` documentados en
  [docs/serving/qwen35.md](../serving/qwen35.md)).
- Gemma 4 LoRA fine-tune queda explicitamente **future**
  ([ADR-011](../decisions/ADR-011-gemma4-lora-future.md)); esta card NO lo cubre.
