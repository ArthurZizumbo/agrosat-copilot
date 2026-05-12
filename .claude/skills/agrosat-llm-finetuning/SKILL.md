---
name: agrosat-llm-finetuning
description: Fine-tune Gemma 4 26B-MoE and Qwen3-VL-30B-A3B with LoRA rank 16 BF16, deploy Qwen3.5-35B-A3B with vLLM on Azure H100 NVL 96GB for AgroSatCopilot. Use for EPIC 6 fine-tuning and EPIC 7 LLM serving with vLLM (OpenAI-compatible).
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot LLM Fine-tuning Skill

## Rules — NON-NEGOTIABLE

- LoRA rank 16 BF16 con FSDP + FlashAttention-2 + gradient checkpointing
- Target modules: attention (q/k/v/o_proj) + MLP (gate/up/down_proj), excluir expertos MoE
- Validar VRAM ANTES de lanzar (Gemma 4: ~82 GB / 96; Qwen3-VL: ~92 GB / 96; Qwen3.5 serving: ~91 GB / 96)
- Checkpoint cada 30 min a Azure Blob
- AgroMind 28482 QA pairs + AgroMind-IT/ES 500 pares + synthetic augmentation
- vLLM continuous batching para serving (OpenAI-compatible)
- vLLM args: `--max-model-len 65536 --gpu-memory-utilization 0.92 --enable-prefix-caching`

## Modelos (IDs verificados HF 24-abr-2026)

| Modelo | HF ID | Licencia | Uso |
|--------|-------|----------|-----|
| Gemma 4 26B-MoE | `google/gemma-4-26b-it` | Apache 2.0 | VLM principal EPIC 6 |
| Gemma 4 E4B | `google/gemma-4-e4b-it` | Apache 2.0 | Fallback L4 |
| Qwen3-VL 30B-A3B | `Qwen/Qwen3-VL-30B-A3B-Instruct` | Apache 2.0 | VLM comparativo |
| Qwen3.5 35B-A3B | `Qwen/Qwen3.5-35B-A3B` | Apache 2.0 | Orquestador on-prem (**sin -Instruct**) |

## Training Script Gemma 4

```python
# ml/train/train_gemma4_lora.py
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

MODEL_ID = "google/gemma-4-26b-it"

def main(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text())
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg["grad_accum"],
        mixed_precision="bf16",
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        # Exclude MoE experts
        modules_to_save=None,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_loader = build_agromind_loader(cfg["batch"])
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    model, optim, train_loader = accelerator.prepare(model, optim, train_loader)

    for epoch in range(cfg["epochs"]):
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                out = model(**batch)
                accelerator.backward(out.loss)
                optim.step(); optim.zero_grad()
            if step % 500 == 0 and accelerator.is_main_process:
                accelerator.save_state(f"azure-blob://checkpoints/gemma4-ep{epoch}-s{step}")
                mlflow.log_metric("train_loss", out.loss.item(), step=step)
```

## Config H100

```yaml
# configs/gemma4_h100.yaml
model_id: google/gemma-4-26b-it
batch: 2
grad_accum: 8     # effective 16
lr: 1.0e-4
epochs: 3
max_seq_len: 32768
lora_rank: 16
lora_alpha: 32
checkpoint_every_steps: 500
checkpoint_path: az://agrosat-checkpoints/gemma4/
flash_attention: 2
gradient_checkpointing: true
fsdp:
  sharding_strategy: FULL_SHARD
  cpu_offload: false
```

## Config L4 Fallback (Gemma 4 E4B)

```yaml
# configs/gemma4_l4_fallback.yaml
model_id: google/gemma-4-e4b-it
batch: 1
grad_accum: 16
quantization: 4bit_nf4
lora_rank: 16
epochs: 3
```

## vLLM Serving (Qwen3.5)

```bash
# scripts/serve_qwen35.sh
#!/usr/bin/env bash
set -euo pipefail

vllm serve Qwen/Qwen3.5-35B-A3B \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching \
  --max-num-batched-tokens 8192 \
  --port 8000 \
  --served-model-name agrosat-qwen35
```

## Validación VRAM

```python
def validate_vram_budget(model_id: str, batch: int, ctx_len: int, h100_total_gb: float = 96):
    """Aborta si presupuesto estimado >94 GB."""
    weights = {"gemma-4-26b": 52, "qwen3-vl-30b": 60, "qwen3.5-35b": 70}[model_id]
    kv_cache_per_ctx = {"gemma-4-26b": 0.00025, "qwen3.5-35b": 0.0002}
    kv = batch * ctx_len * kv_cache_per_ctx[model_id]
    activations = 15  # con grad ckpt
    overhead = 8
    total = weights + 1.5 + kv + activations + overhead
    assert total < 94, f"VRAM exceeded: {total:.1f} GB > 94 GB safe limit"
    return total
```

## QA Checklist Fine-tuning

- [ ] VRAM validada antes de launch
- [ ] LoRA rank 16, target modules correctos
- [ ] FlashAttn-2 + grad ckpt + FSDP
- [ ] Checkpoints cada 30 min a Azure Blob
- [ ] MLflow run con todos los hiperparámetros
- [ ] Auto-shutdown VM H100
- [ ] Eval post-train con AgroMind + AgroMind-IT/ES
- [ ] Atribución Apache 2.0 (Gemma 4, Qwen) en DATA_LICENSE.md
