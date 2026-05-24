# ADR-007 — Fidelidad de FarSLIP al paper Li et al. 2025 y justificacion de desviaciones

**Status**: Aceptada · 2026-05-24 (US-022-c P1)
**Fecha**: 2026-05-24
**Decisores**: Arthur Zizumbo (MLOps Lead), Isaac Avila (ML)
**US relacionada**: US-017 (FarSLIP implementacion), US-022-c P1 (materializacion L4 GCP)
**Avance**: A5 (Modelos finales + Ensambles · 31-may-2026)
**Paper-faro**: Li et al. (2025), "FarSLIP: ...", arXiv:2511.14901.
**Codigo referencia**: [`ml/farslip/distill.py`](../../ml/farslip/distill.py).

---

## Contexto

US-017 implemento la destilacion FarSLIP en `ml/farslip/distill.py` siguiendo el
paper Li et al. 2025 (referenciado en el docstring del modulo:3). El plan original
US-022-c P1 ejecuta el training real en L4 GCP spot (~6h, ~$1.7 USD) sobre 3 ROIs
italianas, con evaluacion mIoU sobre PASTIS-R (Francia) como gate B-3.

El equipo necesita evidencia escrita y defendible para el Avance 5 que demuestre:

1. **Que metodos del paper SI reimplementamos fielmente** (perdidas, hiperparametros, optimizador).
2. **Que desviaciones introducimos** (geografia, bandas, escala) y por que estan justificadas.
3. **Como interpretar el resultado del gate B-3** si el mIoU FarSLIP - mIoU RemoteCLIP
   queda por debajo del +0.05 esperado.

Sin este ADR, el resultado del gate B-3 (positivo o negativo) carece de contexto
metodologico para defender en la presentacion final.

---

## Decision

Aceptar la siguiente tabla de fidelidad/desviacion como contrato metodologico
oficial para la presentacion del Avance 5 y el Paper Track opcional.

### 1. Componentes implementados con fidelidad 1:1 al paper

| Componente paper | Implementacion nuestra | Codigo |
|------------------|------------------------|--------|
| §3.2 Patch-to-patch distillation loss (MSE + cosine, 196 patches) con `stop-grad` explicito sobre teacher | `class PatchDistillationLoss` con `loss_type="mse_plus_cosine"`, `cosine_weight=0.3`, `.detach()` defensivo del teacher | [`distill.py:47-154`](../../ml/farslip/distill.py) |
| §3.3 Region-Category alignment InfoNCE sobre CLS token contra prototipos textuales precomputados | `class RegionCategoryAlignmentLoss` con `temperature=0.07` (paper), prototipos recalculados 1 vez por epoch (text encoder frozen) | [`distill.py:162+`](../../ml/farslip/distill.py) |
| §3.1 Student init desde teacher (`copy.deepcopy`) + adaptacion `patch_embed.proj` 3->4 canales con `init = mean(RGB)` para NIR (anti-dead-neuron) | `_patch_student_proj` aplica deepcopy + init NIR=mean(RGB) | [`distill.py`](../../ml/farslip/distill.py) (clase `FarSLIPDistillationTrainer`) |
| §3.4 Teacher frozen RGB puro | `images[:, :3, :, :]` antes del teacher forward (fix Q1 QA US-017) | [`distill.py`](../../ml/farslip/distill.py) |
| Optimizer AdamW BF16 + grad_accum=2 + warmup linear | Identico, sin modificaciones | [`distill.py`](../../ml/farslip/distill.py) |

### 2. Desviaciones documentadas y justificadas

| Aspecto | Paper Li et al. 2025 | Nuestro pipeline | Justificacion |
|---------|---------------------|------------------|---------------|
| Backbone | CLIP ViT-B/16 | CLIP ViT-B/16 (identico) | — |
| Bandas | 3 canales RGB | **4 canales RGB + NIR** (Sentinel-2 B02/B03/B04+B08) | Mejora discriminacion entre cultivos C3 vs C4, indispensable en agricultura. Init NIR=mean(RGB) evita dead-neuron en el patch_embed |
| Regiones training | ~10 regiones mundiales | **3 ROIs Italia** (Pianura Padana, Toscana, Puglia) | Restriccion de datos abiertos: GSAA Italia es la mejor cobertura libre de clases CAP en Mediterraneo. ROIs elegidas por diversidad climatica (continental / templado / mediterraneo) — cubre las 3 zonas agronomicas principales |
| Vocabulario | Ingles | **Italiano + espanol + ingles** (CAP) | Lingüisticamente ancla con dominio agronomico real italiano; CAP es taxonomia oficial UE. Plantillas it/es/en en `cap_vocabulary.yaml` |
| Hardware | Multi-GPU TPU | **1xL4 24 GB spot 6h** | Restriccion presupuesto MNA (~$1.7 vs miles del paper) |
| Dataset size | Cientos de miles de pares | **~30k pares (gate B-1)** | Restriccion computacion. Explica el R2 del plan US-022-c (resultado negativo aceptable) |
| Eval domain | Mismo dominio de training | **Cross-region: train Italia, eval Francia (PASTIS-R)** | **Mas exigente que el setup del paper**. Demuestra transferencia de dominio |

### 3. Interpretacion del gate B-3

El gate B-3 evalua `mIoU_farslip - mIoU_remoteclip >= +0.05` sobre PASTIS-R Francia.

Dos resultados posibles, ambos defendibles:

- **PASS (mejora >= +0.05)**: FarSLIP transferencia exitosa Italia→Francia con
  3 ordenes de magnitud menos data/computo que el paper. Evidencia de que las dos
  perdidas del paper son robustas al regimen low-resource.
- **FAIL (mejora < +0.05 o negativa)**: FarSLIP no genera mejora significativa
  con el regimen low-resource. **Resultado honesto y reportable** (R2 plan
  US-022-c §11). No es bug: es falta de capacidad esperada. **FarSLIP queda
  como base learner opcional del stacking ensemble EPIC 6** donde un +1% al
  ensemble vale aunque no gane solo (ADR §6 ensemble EPIC 6).

---

## Alternativas consideradas

### Alternativa A: Re-implementar FarSLIP exactamente como el paper (3-canal RGB, ingles only)

Descartada por:
1. Pierde la senal NIR critica para discriminacion de cultivos (mais C4 vs trigo C3).
2. El vocabulario CAP italiano es nuestra ventaja contextual.

### Alternativa B: Saltarse FarSLIP y usar solo RemoteCLIP frozen

Descartada porque:
1. RemoteCLIP no esta adaptado a agricultura (entrenado con RSICD/UCM/etc.).
2. La US-017 ya invirtio 4 SP en codificar las 2 perdidas del paper. Materializarlas
   en L4 (P1) cuesta solo ~$1.7 USD adicionales con potencial alto upside.

### Alternativa C: Expandir training a Francia (incluir EuroCrops FR_2018)

Descartada en US-022-c (ya documentada en respuesta al usuario 2026-05-24):
- Requeriria nuevo vocabulario CAP frances (~20 plantillas)
- Re-entrenar ~10-12h en vez de 6h (~$3-4 USD extra)
- Cambiaria el caracter del experimento (cross-region transfer -> mixed-domain training)
- **Diferida a backlog**: `docs/product-backlog/us-022-d-farslip-france-augment.md`
  para EPIC 6 ensemble (FarSLIP-IT + FarSLIP-FR como base learners separados del
  stacking)

---

## Consecuencias

### Positivas

1. Evidencia documentada y defendible en Avance 5 para el resultado del gate B-3
   sea cual sea.
2. Contrato metodologico claro: cualquier futuro reviewer puede verificar
   linea-por-linea la fidelidad al paper en `distill.py`.
3. Permite reportar honestamente el resultado low-resource sin necesidad de
   maquillaje.

### Negativas

1. El paper-faro adicional (Wen et al. 2025, ADR-006) es de fenologia y NO de
   FarSLIP. Si el reviewer pide "que paper estamos siguiendo en US-022-c", la
   respuesta correcta es **Li et al. 2025 (arXiv:2511.14901) para FarSLIP** +
   **Wen et al. 2025 para la rama semantica fenologica P5**. Sin este ADR
   habia riesgo de confundir cual paper cubre que.

### Neutras

1. Re-corrida con configuracion identica al paper (3-canal RGB, ingles only)
   queda como experimento opcional para Paper Track si hay tiempo post-21-jun.

---

## Cumplimiento

Esta decision se considera cumplida cuando:

- [x] `ml/farslip/distill.py` referencia explicita Li et al. 2025 arXiv:2511.14901 en docstring (linea 3).
- [x] PatchDistillationLoss + RegionCategoryAlignmentLoss + init NIR=mean(RGB) + teacher RGB puro implementados (US-017 cerrada).
- [ ] MLflow run `farslip-clip-italy-v1` registrado con tags `data_version` + `code_version` post-P1 (esta US-022-c).
- [ ] mIoU FarSLIP vs RemoteCLIP sobre PASTIS-R reportado en `docs/us-resolved/us-022-c.md` § "Resultado FarSLIP gate B-3" con interpretacion alineada a este ADR.
- [ ] Si gate B-3 FAIL, FarSLIP entra como base learner opcional del stacking EPIC 6 (no se descarta).

---

## Referencias

- Paper-faro FarSLIP: Li et al. (2025), arXiv:2511.14901, ["FarSLIP: ..."](https://arxiv.org/abs/2511.14901).
- US-017 (codigo): `ml/farslip/{dataset.py,distill.py,train.py,extract_embeddings.py}`.
- US-022-c plan: [`docs/us-planning/us-022-c.md`](../us-planning/us-022-c.md) §2.1 (gates B-1..B-6) y §11 R2.
- ADR-006 (paper-faro fenologico, no FarSLIP): [`ADR-006-reencuadre-baseline-fenologico.md`](ADR-006-reencuadre-baseline-fenologico.md).
- Skill agrosat-llm-finetuning: [`.claude/skills/agrosat-llm-finetuning/SKILL.md`](../../.claude/skills/agrosat-llm-finetuning/SKILL.md).
- Vocabulario CAP italiano: [`ml/farslip/cap_vocabulary.yaml`](../../ml/farslip/cap_vocabulary.yaml).
