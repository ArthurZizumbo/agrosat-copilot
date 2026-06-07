# ADR-009 — Reactivación de H100, pivote FarSLIP del sponsor y alcance v8

**Status**: Aceptada · 2026-06-06
**Fecha**: 2026-06-06
**Decisores**: Arthur Zizumbo (MLOps Lead) · equipo Equipo 17 · directiva del sponsor/evaluador Dr. Camacho (junta 6-jun)
**Reemplaza/extiende**: el alcance técnico que [ADR-008](ADR-008-rediseno-calendario-presentacion-27jun.md) dejó explícitamente abierto ("la decisión de recorte de alcance queda pendiente") y la narrativa "sin H100 / FarSLIP fuera" de [`context/RefinamientoPlaneacionAgroSatCopilot_v7.md`](../../context/RefinamientoPlaneacionAgroSatCopilot_v7.md).
**Fundamento**: [`context/RefinamientoPlaneacionAgroSatCopilot_v8.md`](../../context/RefinamientoPlaneacionAgroSatCopilot_v8.md) · [`docs/STATUS.md`](../STATUS.md) · [`docs/audit/analisis_disruptivo.md`](../audit/analisis_disruptivo.md)

---

## Contexto

A 6-jun-2026 ocurrieron **tres cambios de realidad** que invalidan supuestos del v7 y de ADR-008:

1. **La H100 NVL 96GB ya está disponible y accesible.** VM `gjcamacho-gpuh1` (Windows, túnel Cloudflare, entorno micromamba `agrosat`, repo en `F:\projects\agrosat-copilot`, 1×H100 NVL 96GB, AMD EPYC 96-core, 320GB RAM). El v7 había soltado Gemma 4 LoRA, Qwen3.5 vLLM y Swin-UNETR por ausencia de H100. Acceso documentado en [`docs/infra/acceso-vm-h100-tunnel.md`](../infra/acceso-vm-h100-tunnel.md).

2. **Directiva del sponsor sobre FarSLIP (junta 6-jun).** El evaluador ordena: usar FarSLIP contrastivo con descripciones fenológicas generadas por Gemini Flash → fine-tune; prueba incremental de 4 clases hacia 18; filtro 3:1 de dominancia de Meadow **por patch**; luego ensamble TSViT-pheno + FarSLIP; ver el desempeño de FarSLIP **primero**, antes del ensamble. Esto contradice la recomendación del v7 de descartar FarSLIP (que perdía 0.163 vs 0.233 contra AlphaEarth) — pero por una razón honesta: FarSLIP perdía porque `ml/farslip/train.py:~184` usa prototipos de texto **aleatorios** (`torch.randn`), no descripciones reales. La directiva llena exactamente ese gap.

3. **El profesor extendió el Avance 5 a miércoles 10-jun** (desde dom 7-jun).

## Decisión

Adoptar el **alcance v8** ([`context/RefinamientoPlaneacionAgroSatCopilot_v8.md`](../../context/RefinamientoPlaneacionAgroSatCopilot_v8.md)) como plan oficial vigente, con las siguientes decisiones concretas:

### D-1 — Reactivar H100 con regla de asignación estricta

La H100 se reactiva, pero **el cuello de botella es tiempo (~3 semanas), no cómputo**. Orden de prioridad de la GPU (una sola, coordinar `nvidia-smi`):
1. FarSLIP-fenológico (ablación de bandas) → 2. TSViT full retrain → 3. ensambles OOF → 4. Qwen3.5 vLLM serving para eval.
**No** se asignan horas H100 a Gemma 4 LoRA antes del 27-jun.

### D-2 — Ejecutar el pivote FarSLIP del sponsor

Implementar FarSLIP-fenológico: reemplazar prototipos aleatorios por embeddings de descripciones fenológicas reales (Gemini Flash → text-encoder → `set_text_prototypes()`), ablación de bandas (RGB / falso-color NIR-R-G / 4-band), protocolo incremental 4→18 con filtro 3:1, y luego ensamble E-a (TSViT-pheno + FarSLIP). Detalle en v8 §4.

### D-3 — Reconciliar el filtro de patches con la rama de Isaac

El [`ml/data/pastis_filter.py`](../../ml/data/pastis_filter.py) de `origin/user/iavila/pastis-preparation-for-farslip` implementa filtro por **cobertura** (≥50% píxeles de clases objetivo), no la regla 3:1 del sponsor. Extender ese módulo con modo `dominance_ratio` (mantener patch si Meadow ≤ 3× la 2ª clase del patch) + parámetro `n_classes`, preservando el modo `coverage` legacy. NO mezclar con `abocanegra/semana-5` (84 commits detrás, 11 archivos en conflicto).

### D-4 — Gemma 4 LoRA queda FUTURE; LLM = Gemini API + Qwen vLLM

Gemma 4 26B LoRA se documenta como trabajo futuro (US-058), no se entrena antes de la presentación: bloqueador técnico real (experts MoE 3D fused → QLoRA bloqueado, `target_modules` no matchea, vía `target_parameters` bleeding-edge) + AgroMind es eval-only (fine-tunearlo sería leakage). Reasoner = `gemini-2.5-pro` (GA); variante-B on-prem = Qwen3.5-35B-A3B vLLM (GPTQ-Int4, single-GPU, ~1 día) para el switch A/B y la narrativa de soberanía de datos.

### D-5 — Extender el objetivo a multi-región con transfer learning real

Agregar la historia "no solo Francia": Sen4AgriNet Catalonia (denso, mismo paradigma que PASTIS, demoable en 3 sem), EuroCropsML (few-shot k-shot tabular), WorldCereal/Harmonized Global Crops (escala tropical, futuro/paper). Armonización vía HCAT v3. Encuadrar como **metodología demostrada**, no exactitud validada. Detalle en v8 §7.

### D-6 — Correcciones factuales obligatorias

Gemini 3.1 Pro = **1M ctx** (no 2M); AlphaEarth = **V1/ANNUAL v1.1** (no v2.1); skill Spatial-RAG "−30% alucinación" es incorrecto (mide ranking). Sincronizar en `CLAUDE.md`/`AGENTS.md`.

## Consecuencias

- **Positiva:** el proyecto recupera ambición (H100) y ejecuta la directiva del sponsor con un gap técnico real identificado y un plan concreto para cerrarlo.
- **Positiva:** la historia multi-región cierra la brecha "solo Francia" con datasets y benchmarks reales (no hype), elevando el proyecto sobre un capstone típico.
- **Riesgo:** una sola GPU + 3 semanas; el alcance v8 (~174 SP nominales) excede la capacidad (150 SP) — los items FUTURE/diseño-only se difieren para caber en ~145 SP comprometidos. La normalización de métricas (v8 §10) es prerequisito de todo ensamble y debe protegerse.
- **Riesgo:** convergencia del incremental FarSLIP 4→18 incierta (mitigado con POC 2-epoch + fallback 18-desde-cero).

## Relacionado

- [ADR-006](ADR-006-reencuadre-baseline-fenologico.md) — reencuadre fenológico del baseline (Gemini Flash ya integrado).
- [ADR-007](ADR-007-farslip-fidelity-paper.md) — fidelidad de FarSLIP al paper Li 2025 (la infra que ahora se completa con prototipos reales).
- [ADR-008](ADR-008-rediseno-calendario-presentacion-27jun.md) — calendario (este ADR cierra el recorte de alcance que aquel dejó pendiente).
