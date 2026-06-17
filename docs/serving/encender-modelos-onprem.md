# Encender los modelos on-prem en la H100 (Qwen + Gemma) — guia operativa

Los dos modelos locales del copiloto corren en la VM H100 y se exponen por
endpoints OpenAI-compatibles. Esta guia explica como **encenderlos**, verificarlos
y exponerlos a una maquina local (para el harness de evaluacion o la demo). Cero
costo de API: ambos corren en la H100 propia.

| Modelo | Servidor | Donde viven los pesos | Puerto en la VM |
|--------|----------|------------------------|-----------------|
| Qwen3-30B-A3B-Instruct-2507 (GGUF Q4_K_M) | llama.cpp (`llama-server.exe`) | `F:\models\` | `:8002` |
| Gemma 4 26B-A4B-it (MoE, Q4_K_M) | Ollama | `F:\ollama\models\` | `:11434` |

> La VM y sus servicios base (sshd :50022, tunel cloudflared, MLflow :5010) se
> dejan SIEMPRE encendidos. Esta guia solo enciende/apaga los dos **procesos de
> inferencia** que ocupan VRAM, para liberar la GPU compartida cuando no se usan.

---

## 1. Conectarse a la VM

El tunel cloudflared y el SSH ya estan descritos en
`docs/general/CONEXION-VM-H100-ACTUAL.local.md` (URL del tunel + llave). Resumen:

```bash
# (en tu PC) levantar el tunel local al SSH de la VM
cloudflared access tcp --hostname <URL>.trycloudflare.com --url localhost:50022
# conectar (USA 127.0.0.1, nunca localhost -> bug IPv6)
ssh -p 50022 -i ~/.ssh/agrosat_h100 -o IdentitiesOnly=yes User1@127.0.0.1
```

---

## 2. Encender Qwen (llama.cpp)

En la VM, via tarea programada (sobrevive al cierre del SSH):

```bat
REM la tarea ya existe; solo relanzarla
schtasks /run /tn qwen_serve
```

Si la tarea no existe (VM recien reiniciada), recrearla:

```bat
schtasks /create /tn qwen_serve /tr "F:\run_serve.bat" /sc once /st 23:59 /rl highest /f
schtasks /run /tn qwen_serve
```

`run_serve.bat` -> `F:\serve_qwen_llamacpp.bat` (en el repo:
`scripts/serve_qwen_llamacpp.bat`). Verificar (tarda ~40s en cargar el GGUF):

```bash
curl http://127.0.0.1:8002/health     # 200 = listo (503 = cargando)
```

Endpoint: `http://127.0.0.1:8002/v1/chat/completions`, served-model `qwen35`.

---

## 3. Encender Gemma (Ollama)

Ollama debe arrancar con `OLLAMA_MODELS=F:\ollama\models` (los pesos viven en F:,
no en C: que esta casi lleno). Hay un `.bat` que lo hace:

```bat
REM matar cualquier ollama previo que use la ruta por defecto
taskkill /im ollama.exe /f
REM arrancar Ollama apuntando a F: (la tarea ya existe)
schtasks /run /tn ollama_serve
```

`ollama_serve` -> `F:\serve_ollama.bat`, que hace:

```bat
set OLLAMA_MODELS=F:\ollama\models
set OLLAMA_HOST=127.0.0.1:11434
ollama serve
```

Verificar que ve los modelos Gemma (deben aparecer `gemma4:26b-a4b-it-q4_K_M` y
`gemma4:31b-it-q8_0`):

```bash
curl http://127.0.0.1:11434/api/tags        # debe listar gemma4:*
```

> **Gotcha:** si `api/tags` devuelve `{"models":[]}`, hay un Ollama viejo corriendo
> con la ruta por defecto. Matar TODOS (`taskkill /im ollama.exe /f`) y relanzar la
> tarea. El puerto 11434 solo admite un proceso.

El modelo recomendado para evaluacion es **`gemma4:26b-a4b-it-q4_K_M`** (MoE, 4B
activos, ~118 tok/s — 7x mas rapido que el `31b-it-q8_0` denso, 17 tok/s).
Endpoint OpenAI-compatible: `http://127.0.0.1:11434/v1/chat/completions`.

---

## 4. Exponer ambos a una maquina local (port-forward SSH)

El harness de evaluacion (y la demo) corren en tu PC y necesitan alcanzar los dos
endpoints de la VM. Como tu PC puede tener su propio Ollama en :11434, usa un
puerto local distinto para el de la VM (ej. :11435):

```bash
# Qwen: local 8002 -> VM 8002
ssh -p 50022 -i ~/.ssh/agrosat_h100 -o IdentitiesOnly=yes -N -L 8002:127.0.0.1:8002 User1@127.0.0.1 &
# Gemma (Ollama): local 11435 -> VM 11434
ssh -p 50022 -i ~/.ssh/agrosat_h100 -o IdentitiesOnly=yes -N -L 11435:127.0.0.1:11434 User1@127.0.0.1 &
```

Verificar desde tu PC:

```bash
curl http://127.0.0.1:8002/health        # Qwen -> 200
curl http://127.0.0.1:11435/api/tags     # Gemma -> lista gemma4:*
```

El harness toma las URLs por variable de entorno (lo hace
`scripts/run_us049_eval.py`):
`VLLM_QWEN35_URL=http://127.0.0.1:8002/v1` y
`OLLAMA_BASE_URL=http://127.0.0.1:11435/v1`.

---

## 5. Apagar (liberar la GPU compartida)

Al terminar de evaluar/demostrar, bajar SOLO los procesos de inferencia (no la VM):

```bat
REM en la VM
taskkill /im llama-server.exe /f      REM apaga Qwen
REM Gemma: descargar el modelo de VRAM sin matar Ollama (mas elegante):
REM   curl http://127.0.0.1:11434/api/generate -d "{\"model\":\"gemma4:26b-a4b-it-q4_K_M\",\"keep_alive\":0}"
REM o apagar Ollama entero:
taskkill /im ollama.exe /f
```

Verificar que la GPU quedo libre:

```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # ~2 MiB = libre
```

Los port-forwards locales (los `ssh -L`) se cierran matando esos procesos ssh en
tu PC; no afectan la VM.

---

## Resumen de scripts (en el repo)

- `scripts/serve_qwen_llamacpp.bat` / `.sh` — arranque de Qwen (llama.cpp).
- `scripts/setup_llamacpp_vm.ps1` — instala llama.cpp CUDA en la VM (una vez).
- `scripts/download_qwen_gguf.py` — descarga el GGUF de Qwen a F:.
- `scripts/run_us049_eval.py` — corre el benchmark apuntando a ambos endpoints.
- Gemma: gestionado por Ollama (`ollama pull gemma4:26b-a4b-it-q4_K_M`); el `.bat`
  de arranque (`F:\serve_ollama.bat`) vive en la VM.
