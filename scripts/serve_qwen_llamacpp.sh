#!/usr/bin/env bash
# US-048 (llama.cpp variant): serve Qwen on-prem via llama.cpp on a Linux + GPU host.
#
# Linux counterpart of scripts/serve_qwen_llamacpp.bat. llama-server exposes an
# OpenAI-compatible /v1/chat/completions endpoint, so the agent's
# VLLMOpenAIBackend (US-047) works unchanged -- only the base URL differs.
#
# --parallel N gives N concurrent request slots; -ngl 99 offloads all layers to
# the GPU. Single GGUF (~18.6GB Q4_K_M) fits on the H100.
#
# Override paths/params via environment. Stop by killing llama-server.
set -euo pipefail

LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
QWEN_GGUF="${QWEN_GGUF:-/mnt/f/models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf}"
QWEN_PORT="${QWEN_PORT:-8002}"
QWEN_CTX="${QWEN_CTX:-32768}"
QWEN_PARALLEL="${QWEN_PARALLEL:-4}"
QWEN_NGL="${QWEN_NGL:-99}"
QWEN_ALIAS="${QWEN_ALIAS:-qwen35}"

echo "[serve_qwen_llamacpp] model=${QWEN_GGUF} port=${QWEN_PORT} parallel=${QWEN_PARALLEL} ngl=${QWEN_NGL}"

if [ ! -f "${QWEN_GGUF}" ]; then
  echo "[serve_qwen_llamacpp] ERROR: GGUF not found at ${QWEN_GGUF}" >&2
  echo "[serve_qwen_llamacpp] Run: poetry run python scripts/download_qwen_gguf.py" >&2
  exit 2
fi

# GPU pre-flight: refuse to start if the shared H100 is already busy (>4GB).
if command -v nvidia-smi >/dev/null 2>&1; then
  used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  echo "[serve_qwen_llamacpp] GPU memory used before launch: ${used_mib} MiB"
  if [ "${used_mib:-0}" -gt 4000 ]; then
    echo "[serve_qwen_llamacpp] GPU already in use (>4GB). Coordinate with the team." >&2
    exit 3
  fi
fi

# --jinja: use the model's chat template (Qwen3) for correct tool/role formatting.
# --flash-attn: enable FlashAttention kernels on the GPU.
exec "${LLAMA_SERVER}" \
  -m "${QWEN_GGUF}" \
  --host 127.0.0.1 \
  --port "${QWEN_PORT}" \
  -ngl "${QWEN_NGL}" \
  --ctx-size "${QWEN_CTX}" \
  --parallel "${QWEN_PARALLEL}" \
  --flash-attn on \
  --jinja \
  --alias "${QWEN_ALIAS}"
