# Serving on-prem del reasoner Qwen (vLLM, H100) — US-048

Variante B del copiloto: el reasoner Qwen MoE-A3B servido on-premise con vLLM en
la H100 NVL 96GB, endpoint OpenAI-compatible, intercambiable con Gemini desde el
mismo cliente (`/llm/switch`). Sustenta la historia de **soberania de datos**
(cooperativas que no exportan datos a la nube).

## Modelo

> **Sustitucion documentada.** El plan v8 nombraba `Qwen3.5-35B-A3B-GPTQ-Int4`.
> Ese id **no existe en HuggingFace** (verificado jun-2026). El checkpoint real
> mas cercano de la familia MoE-A3B con cuantizacion Int4 es
> **`Qwen/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4`** (30B totales / 3B activos,
> Apache 2.0). La sustitucion 35B -> 30B-A3B queda ratificada en
> [ADR-009](../decisions/ADR-009-h100-reactivacion-pivote-farslip-alcance-v8.md)
> (US-050).

Serving **GPTQ-Int4 single-GPU** (sin `--tensor-parallel-size`, no BF16 ~70GB):
los pesos cuantizados son ~18-20GB y caben holgados en 96GB con espacio para el
KV cache. `--enable-prefix-caching` acelera los prefijos repetidos de tool calls.

## BLOQUEO actual (15-jun-2026): no ejecutable en la VM H100

vLLM requiere **Linux**. La VM H100 es **Windows**; su unica via Linux es WSL2,
que **no arranca** porque la VM es un **guest Hyper-V sin virtualizacion anidada
expuesta**:

- `Win32_Processor.VMMonitorModeExtensions = False`,
  `SecondLevelAddressTranslationExtensions = False`.
- Import/arranque WSL2 falla con `HCS_E_HYPERV_NOT_INSTALLED`
  (event Hyper-V-Compute id 11008, `result 0x80370102`).
- Es la **misma causa raiz** por la que Docker Desktop se queda en "Starting".

Pese a que `VirtualMachinePlatform`, `Hyper-V` y `HypervisorPresent` aparecen
habilitados, el guest no puede crear VMs hijas. El codigo de esta US esta
**verificado en sintaxis y listo para correr sin cambios** en cualquier host
Linux + GPU (WSL2 una vez desbloqueado, o una VM Linux con GPU).

### Desbloqueo (accion del sponsor / admin del host Azure)

Sobre la VM **apagada**, en el host Hyper-V:

```powershell
Stop-VM -VMName <thisVM>
Set-VMProcessor -VMName <thisVM> -ExposeVirtualizationExtensions $true
Start-VM -VMName <thisVM>
```

Tras eso, en la VM (WSL2 ya importable, rootfs base ya exportado en
`F:\wsl\ubuntu-base.tar`):

```powershell
wsl --import AgrosatGPU F:\wsl\AgrosatGPU F:\wsl\ubuntu-base.tar --version 2
# verificar GPU: dentro de WSL2 debe existir /dev/dxg y /usr/lib/wsl/lib/nvidia-smi -L lista la H100
```

Alternativa si el host no puede habilitar nested virt: servir en una **GPU sobre
Linux nativo** (L4/A100 GCP), no en la VM Windows-guest.

## Arranque (en un host Linux + GPU)

```bash
# 1. Descargar los pesos (token HF en el entorno; todo en F:/disco grande)
poetry run python scripts/download_qwen35.py --dest /mnt/f/models

# 2. Levantar vLLM (single-GPU, GPTQ-Int4, health check incluido)
HF_HOME=/mnt/f/hf_cache bash scripts/serve_qwen35.sh
#   -> endpoint http://127.0.0.1:8002/v1/chat/completions, served-model-name qwen35

# 3. Smoke + benchmark de latencia (registra en MLflow :5010)
poetry run python scripts/benchmark_qwen35.py \
    --base-url http://127.0.0.1:8002/v1 --model qwen35 --n 10
```

`serve_qwen35.sh` hace pre-flight de GPU (rechaza arrancar si hay >4GB en uso,
respetando el orden de prioridad **FarSLIP -> TSViT -> ensambles -> serving**) y
espera el `/health` antes de declarar el endpoint listo.

## Apagado

```bash
# El script imprime el PID; tambien:
pkill -f "vllm serve" || true
```

En la VM (cuando aplique), si se lanzo via tarea programada, detener la tarea.

## Integracion con el agente

El `VLLMOpenAIBackend` (US-047, `ml/agent/backends.py`) apunta a este endpoint.
Tras `POST /llm/switch` a `qwen35`, `make_backend` selecciona el backend vLLM y
las queries de `/chat` responden por Qwen — sin tocar el loop del agente.

## Objetivos de latencia (rubrica)

- p50 < 2 s / p95 < 5 s en query simple de un turno.
- p95 < 15 s en multi-turno con 3-5 tool calls.

Se miden con `scripts/benchmark_qwen35.py` y quedan en MLflow
(`us048_qwen_serving`, tags `code_version` + `data_version`).

## Fine-tune LoRA (FUTURE)

El fine-tune LoRA de trazas de tool calls queda diferido (ADR-009). La H100
prioriza FarSLIP -> TSViT -> ensambles -> este serving, en ese orden estricto.
