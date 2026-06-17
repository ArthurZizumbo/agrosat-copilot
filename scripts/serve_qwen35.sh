#!/usr/bin/env bash
# US-048: serve the on-prem reasoner variant (Qwen MoE A3B, GPTQ-Int4) with vLLM
# on the H100 NVL 96GB, single-GPU, OpenAI-compatible endpoint.
#
# Model: the plan named "Qwen3.5-35B-A3B-GPTQ-Int4", which does NOT exist on
# HuggingFace (verified jun-2026). The real closest MoE-A3B Int4 checkpoint is
# Qwen/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4 (30B total / 3B active). The
# substitution 35B -> 30B-A3B is documented in ADR-009 (US-050).
#
# Single-GPU GPTQ-Int4 (no --tensor-parallel-size, no BF16 ~70GB): the quantized
# weights are ~18-20GB and fit comfortably in 96GB with room for the KV cache.
# --enable-prefix-caching speeds up repeated tool-call prefixes.
#
# RUNTIME NOTE (US-048 blocker): vLLM requires Linux. The H100 VM is Windows; its
# only Linux path is WSL2, which cannot start because the VM is a Hyper-V guest
# WITHOUT nested virtualization exposed (HCS_E_HYPERV_NOT_INSTALLED). Until the
# Azure host admin enables `ExposeVirtualizationExtensions` on the VM (same
# blocker that stops Docker), this script cannot run on that VM. It is verified
# for syntax and is ready to run unchanged on any Linux+GPU host (WSL2 once
# unblocked, or a Linux GPU VM). See docs/serving/qwen35.md.
set -euo pipefail

# --- Configuration (override via environment) -------------------------------
MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4}"
SERVED_NAME="${QWEN_SERVED_NAME:-qwen35}"
PORT="${QWEN_PORT:-8002}"
MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${QWEN_GPU_MEM_UTIL:-0.90}"
# Keep all weights/cache on F: (the VM's large disk); C: is nearly full.
export HF_HOME="${HF_HOME:-/mnt/f/hf_cache}"
HEALTH_TIMEOUT_S="${QWEN_HEALTH_TIMEOUT_S:-600}"

echo "[serve_qwen35] model=${MODEL_ID} served-as=${SERVED_NAME} port=${PORT}"
echo "[serve_qwen35] HF_HOME=${HF_HOME} max_model_len=${MAX_MODEL_LEN}"

# --- GPU pre-flight: the H100 is shared; refuse to start if it is busy --------
if command -v nvidia-smi >/dev/null 2>&1; then
  used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  echo "[serve_qwen35] GPU memory used before launch: ${used_mib} MiB"
  if [ "${used_mib:-0}" -gt 4000 ]; then
    echo "[serve_qwen35] GPU already in use (>4GB). Coordinate with the team before serving." >&2
    echo "[serve_qwen35] Priority order: FarSLIP -> TSViT -> ensembles -> this serving." >&2
    exit 3
  fi
else
  echo "[serve_qwen35] WARNING: nvidia-smi not found; cannot verify free VRAM." >&2
fi

# --- Launch vLLM (single-GPU, GPTQ-Int4, OpenAI-compatible) -------------------
vllm serve "${MODEL_ID}" \
  --served-model-name "${SERVED_NAME}" \
  --quantization gptq \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --enable-prefix-caching \
  --port "${PORT}" &
VLLM_PID=$!
echo "[serve_qwen35] vLLM started (pid ${VLLM_PID}); waiting for health on :${PORT}"

# --- Health wait: poll /health until the model is loaded ---------------------
deadline=$(( $(date +%s) + HEALTH_TIMEOUT_S ))
until curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[serve_qwen35] vLLM process exited before becoming healthy." >&2
    exit 1
  fi
  if [ "$(date +%s)" -ge "${deadline}" ]; then
    echo "[serve_qwen35] health check timed out after ${HEALTH_TIMEOUT_S}s." >&2
    kill "${VLLM_PID}" 2>/dev/null || true
    exit 1
  fi
  sleep 5
done

echo "[serve_qwen35] HEALTHY. Endpoint: http://127.0.0.1:${PORT}/v1/chat/completions"
echo "[serve_qwen35] Served model name: ${SERVED_NAME}"
echo "[serve_qwen35] To stop: kill ${VLLM_PID}"
wait "${VLLM_PID}"
