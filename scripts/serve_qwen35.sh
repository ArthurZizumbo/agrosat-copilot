#!/usr/bin/env bash
# vLLM serving para Qwen3.5-35B-A3B en H100 NVL 96GB
set -euo pipefail

vllm serve Qwen/Qwen3.5-35B-A3B \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --max-num-batched-tokens 8192 \
  --port 8000 \
  --served-model-name agrosat-qwen35
