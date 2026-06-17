# ADR-011 — Gemma 4 26B-A4B LoRA como trabajo FUTURE (perceiver fine-tuned diferido)

**Status**: Aceptada (FUTURE — NO se entrena antes de la presentacion del 27-jun-2026)
**Fecha**: 2026-06-15
**Decisores**: Arthur Zizumbo (MLOps Lead) · equipo Equipo 17
**US relacionada**: US-050 (EPIC 7 — documentar la decision Gemma FUTURE, 2 SP documentales)
**Extiende**: [ADR-009](ADR-009-h100-reactivacion-pivote-farslip-alcance-v8.md) D-1 (orden de prioridad estricto de la H100: FarSLIP -> TSViT -> ensambles -> serving Qwen; Gemma NO recibe horas H100 antes del 27-jun).
**Fundamento**: [`context/RefinamientoPlaneacionAgroSatCopilot_v8.md`](../../context/RefinamientoPlaneacionAgroSatCopilot_v8.md) (US-050) · verificacion en HuggingFace y docs de PEFT (jun-2026).
**Skill afectado**: [`.claude/skills/agrosat-llm-finetuning/SKILL.md`](../../.claude/skills/agrosat-llm-finetuning/SKILL.md) (corregido por esta US).

---

## Contexto

El v6 contemplaba a **Gemma 4 26B-MoE LoRA** como perceiver fine-tuned, tratado
como decision irrevocable del stack. Con la H100 ya disponible (ADR-009), la
restriccion deja de ser de computo y pasa a ser de **tiempo** (~3 semanas a la
presentacion) y de **viabilidad tecnica del fine-tune MoE**. Una verificacion
honesta en HuggingFace y en la documentacion de PEFT (jun-2026) revela varios
supuestos del skill que NO se sostienen. Este ADR los corrige y formaliza que
Gemma 4 LoRA queda **FUTURE**: reactivable post-presentacion sin re-investigar.

## Hechos verificados (jun-2026)

### 1. El id del modelo del skill no existe; el real es otro

- El skill usaba `google/gemma-4-26b-it`. **Ese id no existe en HuggingFace.**
- El real de la familia MoE ~26B es **`google/gemma-4-26B-A4B-it`** (instruct) /
  `google/gemma-4-26B-A4B` (base). "A4B" = ~4B parametros **activos**; la pagina
  oficial reporta **8 expertos activos / 128 totales + 1 compartido**, ~3.8B
  activos sobre ~25.2B totales. Apache 2.0.

### 2. Los expertos MoE son tensores 3D fused -> QLoRA bloqueado

- En la familia Gemma 4 MoE los expertos se almacenan como **tensores 3D
  fusionados** (`nn.Parameter`, no `nn.Linear`), agrupando todos los expertos en
  un solo parametro cuya dimension 0 es el indice de experto.
- **`bitsandbytes` no cuantiza ese layout 3D fused** -> la via **QLoRA queda
  bloqueada** para los expertos (se podria cuantizar solo attention, no los MLP
  de expertos, lo que anula gran parte del ahorro de memoria que justifica QLoRA).

### 3. `target_modules=[gate/up/down_proj]` no matchea los expertos

- Un `LoraConfig(target_modules=["gate_proj","up_proj","down_proj"])` (lo que tenia
  el skill) **no engancha** los expertos MoE: esos nombres apuntan a modulos
  `nn.Linear` que en el MoE fused no existen como tales. El resultado es que LoRA
  solo toca attention (~0.91% de params entrenables) y deja los expertos —donde
  vive la capacidad del modelo— intactos.

### 4. La via real es `target_parameters` (PEFT >= 0.17), bleeding-edge

- PEFT permite aplicar LoRA directamente a `nn.Parameter` via **`target_parameters`**
  (un `nn.Parameter` no tiene `forward` que PEFT pueda envolver, por eso se targetea
  el parametro). Nombres reales del MoE: **`mlp.experts.gate_up_proj`** y
  **`mlp.experts.down_proj`**. Soporta tensores 2D/3D (la dim 0 es el experto), con
  `rank_pattern` para mantener el presupuesto de rank por experto.
- **Caveats documentados** que lo hacen bleeding-edge para una entrega con fecha:
  overhead de inferencia significativo (PEFT materializa la contribucion LoRA por
  experto aunque se activen pocos; mitigable con `merge_and_unload`), y **no se
  pueden cargar multiples adapters `target_parameters` a la vez**.

### 5. AgroMind es eval-only -> "fine-tune sobre AgroMind" = leakage

- **AgroMind** (~28,482 QA, HF `AgroMind/AgroMind`) **no tiene train split**: es un
  benchmark de evaluacion. Entrenar sobre el e informar metricas en el contamina la
  evaluacion (**leakage**). Lo mismo aplica al **AgroMind-IT/ES** propio (500 pares):
  es eval, no train.
- Por tanto cualquier fine-tune de Gemma necesitaria un **SFT sintetico propio**
  (trazas de tool calls generadas), nunca AgroMind. Eso es justo lo que US-049
  respeta: evalua las variantes, no las entrena.

## Decision

1. **Gemma 4 26B-A4B LoRA queda FUTURE.** No se entrena en el horizonte de la
   presentacion (27-jun). La H100 sigue el orden estricto de ADR-009 D-1
   (FarSLIP -> TSViT -> ensambles -> serving Qwen).
2. **Reasoner del copiloto = Gemini 2.5-pro** (cloud, default). **Variante on-prem
   = Qwen3.5-A3B / Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4** servido con vLLM
   (US-048). Gemma LoRA = expert-LoRA con SFT sintetico propio, **post-presentacion**.
3. **Se corrige el skill `agrosat-llm-finetuning`**: id real `gemma-4-26B-A4B-it`;
   QLoRA bloqueado por el layout 3D; via real `target_parameters=["mlp.experts.gate_up_proj","mlp.experts.down_proj"]` (PEFT >= 0.17); AgroMind = eval-only.
4. **Se marca Gemma 4 LoRA como FUTURE en `CLAUDE.md` / `AGENTS.md`** (Decisiones
   Irrevocables: viable con H100 pero diferido por tiempo + complejidad MoE-LoRA).

## Consecuencias

- **Positivas**: la decision es trazable y reactivable sin re-investigar; el equipo
  no quema dias de H100 (escasos) en un fine-tune MoE bleeding-edge cuya viabilidad
  con la fecha es dudosa; la evaluacion (US-049) queda limpia de leakage.
- **Negativas / asumidas**: el copiloto no tiene un perceiver-VLM fine-tuned propio
  para la presentacion; se apoya en el patron Be My Eyes (perceiver = modelos del
  equipo emiten texto; reasoner Gemini/Qwen frozen razona sobre ese texto), que es
  precisamente la arquitectura de US-046/047 y no requiere fine-tune de Gemma.

## Reactivacion (post-presentacion / Paper Track)

1. Descargar `google/gemma-4-26B-A4B-it`; entorno con PEFT >= 0.17.
2. `LoraConfig(target_modules=["q_proj","v_proj"], target_parameters=["mlp.experts.gate_up_proj","mlp.experts.down_proj"], rank_pattern=...)` (sin QLoRA sobre los expertos 3D).
3. SFT sintetico PROPIO (trazas de tool calls), nunca AgroMind/AgroMind-IT-ES (eval-only).
4. `merge_and_unload` antes de servir (elimina el overhead por-experto); evaluar con el harness de US-049 (ya soporta una variante extra).
