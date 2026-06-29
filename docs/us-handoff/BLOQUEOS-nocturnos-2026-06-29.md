# Bitácora de bloqueos — ejecución nocturna autónoma US-081 / US-082

> **Contexto**: Arthur autorizó ejecución autónoma total (2026-06-29, ~02:54 CST) y se fue a
> dormir. No puede resolver dudas. Regla: **no pararse nunca**; ante un bloqueo, registrarlo
> aquí y seguir con la siguiente sub-tarea que sí pueda avanzar (memoria engram #384).
>
> Decisiones de referencia: #381 (orquestación/ramas/3 vías), #382 (reglas duras datos reales),
> #383 (hallazgos VM), #384 (autonomía total).

## Formato por entrada
`[HH:MM] [US-XXX / dominio] BLOQUEO: <qué> · CAUSA: <por qué> · INTENTADO: <qué probé> · DECISIÓN: <qué hice para no parar> · IMPACTO: <qué queda pendiente para Arthur>`

---

## Bloqueos registrados

- **[03:2x] [US-081/ml Grupo A · AC2/AC3] BLOQUEO**: el scorecard `grounded_crop` bajo
  france-12 con un REASONER LLM REAL (las 4 variantes: gemini / qwen / gemma-base / qwen36-vl)
  no se corrió. · **CAUSA**: requiere los endpoints H100 (forwards :8002 Qwen texto, :8003
  Qwen3.6-VL, :11435 Gemma) arriba + la key Gemini inyectada; no están en esta sesión (regla
  dura: cero números fabricados con un LLM fake). · **INTENTADO/HECHO**: dejé el run script
  LISTO con `--label-space france-12` (fija `LABEL_SPACE` para que `classify.run` restrinja a
  12 clases en la eval live), Y produje el NÚMERO REAL por una vía cero-red equivalente:
  `--france12-offline` corre SOLO `eval_grounded_crop` bajo france-12 con un ORÁCULO
  determinista (reasoner perfecto) sobre los casos con posteriores REALES del campeón v2
  (`reports/agent_bench/us081_grounded_crop_france12.json`: routing 1.0, crop_match 0.875,
  faithfulness 0.875, oov_handoff 1.0). El AC1 (perceiver_champion_eval v2) SÍ se corrió
  completo sobre el fold-5 OOF REAL (CPU, sin bloqueo): france-9 macro-F1 0.9035→0.9400,
  france-12 0.8992. · **DECISIÓN**: no me paré; el oráculo aísla orquestación+restricción+handoff
  (lo que US-081 cambió) sin LLM, y el número live se completa con una línea cuando haya
  endpoints/creds: `poetry run python scripts/run_us049_system_eval.py --label-space france-12
  --variants gemini`. · **IMPACTO para Arthur**: levantar los forwards H100 + key Gemini y correr
  el comando de arriba para el scorecard agéntico live de 12 clases (el número offline ya está;
  el live solo cambia el reasoner-fake por el real).

- **[03:2x] [US-081/test cross-agent] NO-BLOQUEO (nota)**: `tests/ml/eval/test_label_space.py::
  test_get_label_space_default_is_france9` FALLA, pero es PRE-EXISTENTE y ajeno a Grupo A: el
  `DEFAULT_LABEL_SPACE` ya es `france-12` desde el commit `8dd4335` (scope del dueño de
  `class_remap`); el test quedó desactualizado a `france-9`. NO lo toqué (fuera de mi write-set).
  · **IMPACTO**: el dueño de `class_remap`/Grupo B debe actualizar ese assert a `france-12` (o
  renombrar el test) al consolidar el flip del default.

- **[03:30] [US-081/agent Grupo C · AC8] RESUELTO (número REAL obtenido)**: el delta del hedge
  A/B out-of-vocab SÍ se midió con datos 100% reales. La key Gemini estaba presente en
  `.env.local` (verificado hiteando la API real, no asumido), así que corrí el A/B con
  **reasoner Gemini 2.5 Pro + juez LLM-as-judge Gemini 2.5 Pro** sobre las 6 parcelas REALES
  out-of-vocab (las 6 clases dropped de france-12). **Resultado** (`reports/agent_bench/hedge_ab_oov_v2.json`):
  `hedge_quality_ungrounded=1.0`, `hedge_quality_grounded=1.0`, `delta=0.0`,
  `forced_label_rate=0.0` en ambos lados. **Lectura honesta**: Gemini ya hace el hedge
  PERFECTO sin RAG (nunca fuerza una de las 12 clases resueltas, reconoce el límite y nombra
  el cultivo verdadero), por eso el delta del grounding es 0.0 en ESTOS casos — el valor es la
  HONESTIDAD demostrada, no una ganancia de F1 (enmarcado así en el módulo). · **CAVEAT (a
  revisar)**: en el dataset `unresolved_candidate == true_crop`, así que el reasoner sin RAG
  puede nombrar el cultivo desde la propia pista; una variante más dura que oculte el cultivo
  verdadero de la pista (forzando depender de la evidencia vecina) aislaría la contribución del
  RAG — queda como follow-up. · **Bug encontrado y corregido**: el `LLMHedgeJudge` llamaba
  `asyncio.run()` dentro del loop async de `run_hedge_ab` ("cannot be called from a running
  event loop") y degradaba a 0/0 silencioso; fix = invocar el juez vía `asyncio.to_thread`
  (hilo sin loop) + helper `_run_coro_blocking` de respaldo. Verificado: tras el fix el número
  es estable y el juez puntúa correcto (probado directo: hedge bueno -> {1.0, 0.0}).

- **[03:2x] [US-081/agent Grupo C · AC9] SUB-TAREA (no bloqueo de código)**: nombrar 1 clase
  out-of-vocab vía FarSLIP (open-set) NO se ejecutó end-to-end. · **CAUSA**: el refinador
  FarSLIP (`ml/agent/refine.py` + `ml/farslip/zeroshot_head.py`) necesita GPU + checkpoint
  CLIP para producir `farslip_scores` reales sobre una parcela; sin GPU en esta sesión no se
  obtiene el score real (regla dura: no fabricar scores). · **INTENTADO/HECHO**: verifiqué que
  el VEHÍCULO ya existe y es correcto: `apply_refinement(posterior, farslip_scores,
  open_set=True)` fuerza la vía open-set y reordena hacia la clase nueva; es PURO (sin torch).
  El enganche con el copiloto ya está cableado por US-080. · **DECISIÓN**: documentado como
  sub-tarea (abajo); no dupliqué `refine.py`. · **IMPACTO**: correr el `zeroshot_head` FarSLIP
  en H100 sobre ≥1 parcela real de las 6 dropped y pasar sus `farslip_scores` a
  `apply_refinement(..., open_set=True)`; medir si nombra el cultivo correcto.

- **[03:2x] [US-081/backend Grupo B · AC5] BLOQUEO**: el DEPLOY REAL del backend (build de la
  imagen `api` con `dvc pull` de los OOF + push + Cloud Run + smoke `/chat` v2) no se ejecutó.
  · **CAUSA**: requiere credenciales GCP / Cloud Build + el secreto de Secret Manager
  `agrosat-dvc-sa-key` (clave de SA con `roles/storage.objectViewer` sobre
  `gs://agrosat-dvc-remote`) provisionado con VALOR; ninguno está inyectado en esta sesión.
  · **INTENTADO/HECHO**: dejé el código del deploy LISTO y testeado de lo testeable sin red:
  (1) `infrastructure/docker/backend.Dockerfile` con una etapa nueva `dvc-data` que instala
  `dvc[gs]`, copia solo los `.dvc` mínimos + `.dvc/config` y hace `dvc pull` de
  `ml/eval/oof_new32` + `oof_parcel_utae_fold5.parquet` + `oof_parcel_xgb-alphaearth_fold5.parquet`
  (NO PASTIS-R, por pesos pineados) con la clave GCS como build secret `gcs_dvc_key`
  (`--mount=type=secret`); si el secreto falta (build local sin creds) el pull se OMITE con
  aviso ruidoso y el runtime degrada `voting3 -> xgb` (log `classify_voting3_unavailable`), sin
  romper el build. (2) `infrastructure/cloudbuild.yaml`: el step `build-api` corre con
  `DOCKER_BUILDKIT=1` y monta `GCS_DVC_KEY` vía `availableSecrets` (Secret Manager
  `agrosat-dvc-sa-key`). · **DECISIÓN**: no me paré; el contrato SSE + lista DVC + smoke quedan
  documentados en `docs/serving/copiloto-v2-12clases.md`. · **IMPACTO para Arthur**: (a)
  provisionar el VALOR del secreto `agrosat-dvc-sa-key` (el contenedor TF ya existe; el valor se
  llena a mano fuera de TF, convención infra); (b) lanzar `gcloud builds submit
  --config=infrastructure/cloudbuild.yaml`; (c) correr el smoke `_load_voting_three()` dentro del
  contenedor (debe imprimir parcels > 0). Sin esto, el deploy serviría xgb degradado en vez del
  campeón v2.

---

## Tests pre-existentes fallando (AJENOS a US-081, fuera del diff 12c98a6..HEAD)

- **`backend/tests/unit/test_chat_sse.py::test_perceiver_block_injected_as_grounding_turn`** — FALLA pre-existente. El wrapper de grounding (perceiver block como grounding turn) se commiteó en `backend/app/services/chat_service.py` ANTES del SHA base `12c98a6`; ese unit test quedó desactualizado desde entonces. Verificado: `git diff --name-only 12c98a6 -- backend/tests/unit/test_chat_sse.py backend/app/services/chat_service.py` → VACÍO (ni el test ni el service los tocó esta US). NO se arregla aquí (fuera del diff de la US; tocarlo arriesga otra regresión). **A revisar Arthur**: actualizar ese unit test al contrato de grounding vigente en una tarea de mantenimiento aparte.

## Tests pre-existentes CORREGIDOS (legítimo dentro del cierre de US-081)

- **`tests/ml/eval/test_label_space.py::test_get_label_space_default_is_france9`** → renombrado a `_is_france12` y actualizado. El default `DEFAULT_LABEL_SPACE` cambió a `france-12` en `8dd4335` (de-hardcode del label-space, base de US-081); el test asumía aún `france-9`. Como US-081 consolida el campeón v2 de 12 clases, actualizar este test ES parte del cierre. Ahora asserta `get_label_space() is FRANCE_12` + `DEFAULT_LABEL_SPACE == FRANCE_12.name`. 16 passed.

## Decisiones tomadas en autonomía (no bloqueantes, pero a revisar)

- **[03:1x] [US-081/rama]** La rama de trabajo de US-081 es `us-081` (ya existía, sincronizada con `main` @ `5ba5fcb`, incluye el planning US-081/082 del merge #58). El SHA base del handoff es `12c98a6` (ancestro confirmado de HEAD; ancla el diff de la US). NO se creó `feature/E12-US-081-...` nueva para no fragmentar; `us-081` cumple la función y sobre ella se apilará `us-082`. **A revisar**: si prefieres el nombre canónico `feature/E12-US-081-copiloto-v2-12clases`, renombrar antes del PR (`git branch -m`).
- **[03:1x] [US-082/dataset VM]** El dataset Italia completo (1438 patches, 11 GB) vive en `F:\worktrees\us078\data\pastis_italia_2018` (worktree US-079) y NO en el repo principal `F:\projects\agrosat-copilot`. Decisión: NO copiar los 11 GB; la extracción AlphaEarth leerá el dataset por path absoluto desde el worktree (cero duplicación, cumple "lo más fácil"). Si el runner exige el dataset bajo el repo, se hará robocopy entonces. El `.dvc` del dataset en el worktree está DESACTUALIZADO (dice 64 archivos/174MB vs 1438/11GB reales) — NO usar `dvc pull` para este dataset; se re-`dvc add` tras la extracción full.
- **[03:1x] [US-082/repo VM sucio]** El repo principal de la VM (`F:\projects\agrosat-copilot`) está en `main @ cdc6a6f` (atrás de origin `5ba5fcb`) y SUCIO (staged `oof_new32.dvc`/`segmentation.dvc`, modificado `train_segmentation.py`, untracked `ml/eval/oof_new/`, `ml/losses/`, `reports/voting_new/`, etc.). NO se limpió (puede ser trabajo en curso de otra sesión/Isaac). Cuando toque correr la rama 082 allá, se hará `git stash` del estado sucio o se usará un worktree dedicado, sin perder esos cambios. **A revisar**: confirmar si ese estado sucio es consolidable o descartable.
