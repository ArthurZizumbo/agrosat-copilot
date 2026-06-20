# Analisis de Riesgos — AgroSatCopilot (Avance 7)

> **Entregable**: cubre el criterio "Riesgos" (20 pts) de la rubrica del Avance 7.
> **US**: US-062 (EPIC 10 — Observabilidad y Documentacion). **Sprint**: S9 (15-21 jun).
> **Owner**: Arthur Zizumbo. **Plan vigente**: `context/RefinamientoPlaneacionAgroSatCopilot_v8.md`
> (§4 "Gestion de Riesgos", linea 2530; §5 "Criterios de Exito", linea 2544), ADR-009.
>
> **Regla de datos reales**: las cifras, estados y referencias de este documento provienen de
> fuentes verificables del repositorio (plan v8, ADRs, migraciones aplicadas, blockers,
> resultados de EPIC 12 ya cerrada, memoria del proyecto). La probabilidad y el impacto son
> juicio de ingenieria documentado, no placeholders. Las afirmaciones que no se pudieron
> verificar contra el repo quedan anotadas en `docs/blockers/epic10-notas.md`.

---

## 0. Metodologia

Cada riesgo recibe un identificador estable (`R-DAT-01`, `R-ATK-01`, `R-CNF-01`, `R-CMP-01`,
`R-EJE-01`, ...) para referenciarlo en la matriz y en otros documentos (cierres de US, model
cards, `security.md`).

**Escala de probabilidad**

- **Alta**: esperable dentro del horizonte del proyecto.
- **Media**: plausible bajo ciertas condiciones operativas.
- **Baja**: poco probable, pero con impacto suficiente para vigilarlo.

**Escala de impacto**

- **Alto**: bloquea un entregable, provoca fuga de datos o incumplimiento legal.
- **Medio**: degrada calidad o cronograma sin bloquear la entrega.
- **Bajo**: molestia o coste menor, absorbible.

**Severidad** = funcion de Probabilidad x Impacto. Se usa para ordenar la matriz (seccion 7)
y derivar el top-5 del resumen ejecutivo. Convencion de severidad:

| | Impacto Alto | Impacto Medio | Impacto Bajo |
|--|--------------|---------------|--------------|
| **Prob Alta** | Critica | Alta | Media |
| **Prob Media** | Alta | Media | Baja |
| **Prob Baja** | Media | Baja | Baja |

**Fuentes declaradas**: plan v8 (§4 y §5), ADR-009, las 6 migraciones aplicadas en
`db/migrations/` (en particular `20260620000418_rls_multi_tenant.sql`, US-051), los blockers de
EPIC 10/12 (`docs/blockers/`), los resultados de EPIC 12 ya cerrada (US-074/075/076/077,
`reports/segmentation/sen4agrinet_transfer_result.json`), `docs/licenses/DATA_LICENSE.md` y la
memoria operativa del proyecto (gotchas MLflow, FinOps, H100).

> **Nota de fidelidad sobre la referencia del AC.** El criterio de aceptacion cita "§11 del v8";
> la seccion vigente de gestion de riesgos de ejecucion es el **§4 "Gestion de Riesgos"**
> (linea 2530 del plan v8). Este documento usa el §4 como fuente canonica y deja constancia de
> la correccion de referencia.

> **Nota sobre `docs/STATUS.md`.** El plan de la US referencia `docs/STATUS.md` como fuente del
> estado de RLS; ese archivo **no existe** en el repositorio a la fecha. En su lugar, la fuente
> de verdad usada aqui son las **migraciones aplicadas** en `db/migrations/` (verificables con
> `dbmate status`). El faltante queda anotado en `docs/blockers/epic10-notas.md`.

---

## 1. Resumen ejecutivo (top-5 por severidad)

1. **R-EJE-01 — H100 una sola GPU, cola consume dias** (Prob Alta x Impacto Alto = **critica**).
   La unica GPU de entrenamiento serializa FarSLIP, TSViT, ensambles OOF y serving Qwen.
   Mitigado por orden estricto y por que la H100 NVL 96GB del sponsor (VM `gjcamacho-gpuh1`)
   esta prestada 24/7 sin coste al equipo; fallback L4 (`agrosat-farslip-trainer-dev`, spot)
   para la ablacion de bandas.
2. **R-DAT-04 — transferibilidad espacial limitada de AlphaEarth** (Prob Alta x Impacto Alto).
   Evidencia REAL de EPIC 12: el transfer FR -> Catalonia da zero-shot mIoU 0.0000 y few-shot
   mIoU 0.2468 (Delta +0.2468). El gap espacial existe; se cierra con few-shot, nunca con
   zero-shot.
3. **R-CNF-04 — sobre-afirmar F1 >= 0.80 en Mexico sin ground-truth** (Prob Media x Impacto
   Alto). Postura adoptada: la demo de Mexico (US-077, aguacate/guayaba) es zero-shot
   **cualitativa, sin claim de F1**. Riesgo reputacional/academico, mitigado por enmarcado
   explicito.
4. **R-ATK-04 — exfiltracion de datos entre inquilinos (multi-tenant)** (Prob Media x Impacto
   Alto). Mitigado por RLS por `session_id` ya aplicado (US-051) con rol `agrosat_app`
   NOBYPASSRLS y politica fail-closed; queda deuda de backfill de `parcels.session_id` a
   NOT NULL.
5. **R-DAT-02 — calidad de labels (partial-label / null-class del crosswalk HCAT)** (Prob Media
   x Impacto Medio-Alto). Mitigado por el crosswalk PASTIS-18 -> HCAT de US-074, que trata las
   clases fuera del mapa como `null-class` y no como error duro.

---

## 2. Riesgos de DATOS (`R-DAT-*`)

| id | Riesgo | Prob | Impacto | Mitigacion concreta y accionable |
|----|--------|------|---------|----------------------------------|
| R-DAT-01 | Disponibilidad / caidas de CDSE (Copernicus Data Space Ecosystem) | Media | Medio | Fuente primaria de embeddings es GEE AlphaEarth (`SATELLITE_EMBEDDING/V1/ANNUAL`), que no depende de CDSE; reintentos con backoff y cache parquet/COG local en `data/cache/gee/`. Disparador: timeout o 5xx repetido -> servir desde cache y degradar a la mediana anual ya disponible. |
| R-DAT-02 | Calidad de labels: partial-label y clases fuera del crosswalk HCAT (PASTIS-R) | Media | Medio | El crosswalk PASTIS-18 -> HCAT v3 de US-074 mapea explicitamente y trata lo no mapeado como `null-class` (no error duro); label-space `hcat-macro` documentado. Disparador: aparicion de clase nueva -> revisar tabla de crosswalk antes de re-entrenar. |
| R-DAT-03 | Cobertura de nubes en Sentinel-2 degrada features | Media | Medio | AlphaEarth agrega anualmente, reduciendo el efecto puntual de nubes; compositing por mediana temporal en la ingesta S2. Disparador: porcentaje de nubes por escena por encima de umbral -> excluir escena del composite. |
| R-DAT-04 | **Transferibilidad espacial limitada de AlphaEarth** (arXiv:2601.00857) | Alta | Alto | Evidencia REAL E12 (US-075): FR -> Catalonia zero-shot mIoU 0.0000 vs few-shot mIoU 0.2468 (Delta +0.2468, `reports/segmentation/sen4agrinet_transfer_result.json`). Mitigacion: transferir SIEMPRE con finetune few-shot por region, nunca prometer zero-shot fuera de Francia; reportar curva k-shot. |
| R-DAT-05 | Descarga lenta o fallida de subsets multi-region (Sen4AgriNet ~943 MB, EuroCropsML via Zenodo) | Media | Medio | Versionado DVC sobre GCS (`data/sen4agrinet.dvc`, md5 `e292ab8c...`, 88 files/~943 MB/40 patches; `data/transfer/eurocropsml.dvc`); plan B escalonado a subset minimo documentado en blockers E12. Disparador: fallo de `dvc pull` -> usar subset minimo y anotar en blockers. |

> AlphaEarth se referencia como **`SATELLITE_EMBEDDING/V1/ANNUAL`, data v1.1, 64-dim,
> CC-BY-4.0** (cobertura global incl. Mexico). La denominacion "v2.1" no existe; la entrada de
> `docs/licenses/DATA_LICENSE.md` ya corrige esta mencion muerta.

---

## 3. Riesgos de ATAQUES (`R-ATK-*`)

| id | Riesgo | Prob | Impacto | Mitigacion concreta y accionable |
|----|--------|------|---------|----------------------------------|
| R-ATK-01 | Adversarial attacks sobre clasificacion / segmentacion (perturbaciones de entrada) | Baja | Medio | Validacion de entradas, monitoreo de drift con Evidently (US-060/066, asset `drift_check` semanal) que detecta cambios de distribucion; el agente nunca auto-aplica acciones agronomicas sin revision humana. Disparador: `drift_score > 0.3` -> alerta y revision. |
| R-ATK-02 | DDoS / abuso de la API publica | Media | Medio | Rate limiting slowapi ya cableado en `create_app()` (`app.state.limiter`, handler 429 con `Retry-After` / `X-RateLimit-*`), por sesion (US-052); scale-to-zero (Cloud Run `min_instances=0`) acota el coste de un abuso; Cloud Load Balancer + CDN planeado (US-064). |
| R-ATK-03 | Prompt injection / tool abuse en el agente ADK | Media | Medio | Allowlist de FunctionTools en `ml/agent/tools/` (no en routers); el agente no ejecuta acciones destructivas; aislamiento por `session_id`; el reasoner observa salidas de tools, no ejecuta texto del usuario como comando. |
| R-ATK-04 | Exfiltracion de datos entre inquilinos (multi-tenant) | Media | Alto | **RLS por `session_id` YA aplicado** (US-051, `20260620000418_rls_multi_tenant.sql`): rol de aplicacion `agrosat_app` NOSUPERUSER/NOBYPASSRLS, `FORCE ROW LEVEL SECURITY` y politica fail-closed (`current_setting('app.current_session', true)::uuid`, NULL -> 0 filas) en `chat_sessions`/`aois`/`parcels`/`features_parcels`. Deuda: backfill de `parcels.session_id` a NOT NULL y columna `session_id` denormalizada en `features_parcels` (hoy via EXISTS al padre). |

---

## 4. Riesgos de CONFIANZA (`R-CNF-*`)

| id | Riesgo | Prob | Impacto | Mitigacion concreta y accionable |
|----|--------|------|---------|----------------------------------|
| R-CNF-01 | Hallucinations del LLM reasoner (Gemini 2.5/3.5, Qwen3.5-35B-A3B) | Media | Medio | Spatial-RAG hibrido (PostGIS ST_DWithin + pgvector) provee grounding y reduce alucinacion; el reasoner cita fuentes y observa salidas de tools en lugar de inventar cifras. Nota honesta: el "-30%" del patron GeoAnalystBench mide ranking, no es una garantia medida en este proyecto. |
| R-CNF-02 | Sesgos regionales (modelo entrenado en Francia/Europa aplicado a Mexico) | Alta | Medio | Documentado como dominio fuera de distribucion; se aplica transfer few-shot por region, nunca zero-shot fuera de Francia (coherente con R-DAT-04). Las descripciones de Mexico se enmarcan como cualitativas. |
| R-CNF-03 | Falsas alarmas / falsos positivos en clasificacion | Media | Medio | Reportar confianza/intervalos junto a la prediccion; human-in-the-loop antes de cualquier accion; umbral calibrado por clase. |
| R-CNF-04 | **Sobre-afirmar F1 >= 0.80 en Mexico sin ground-truth** | Media | Alto | Postura REAL: US-077 (aguacate/guayaba) es zero-shot **cualitativa, sin claim de F1** (ADR-009; §4 v8 "Transfer Mexico sin ground-truth -> demo metodologico cualitativo, no F1 validado"; §11.2 v8). F1 >= 0.80 mexicano validado queda explicitamente FUTURE (requiere muestras curadas). Disparador: cualquier slide/figura con numero de exactitud en Mexico -> bloquear hasta tener ground-truth. |

---

## 5. Riesgos de CUMPLIMIENTO (`R-CMP-*`)

| id | Riesgo | Prob | Impacto | Mitigacion concreta y accionable |
|----|--------|------|---------|----------------------------------|
| R-CMP-01 | GDPR / proteccion de datos (parcelas y usuarios europeos) | Baja | Alto | Minimizacion de datos, agregacion a nivel parcela, sin PII en logs structlog; aislamiento por sesion (RLS US-051); secretos en Secret Manager (GCP) / Key Vault (Azure), nunca hardcodeados. |
| R-CMP-02 | Licencias multi-region (share-alike y atribucion) | Media | Medio | Atribucion y share-alike donde aplica: Sen4AgriNet CC-BY-SA-4.0, EuroCropsML CC-BY-SA-4.0 (Zenodo DOI 10.5281/zenodo.15095445), AlphaEarth V1/ANNUAL v1.1 CC-BY-4.0, Sentinel/Copernicus CC-BY-SA equivalente, ODbL donde corresponda. Registro en `docs/licenses/DATA_LICENSE.md`; las data sheets (US-064) declaran licencia por dataset. |
| R-CMP-03 | Politicas Copernicus / terminos de uso Sentinel | Baja | Medio | Uso conforme a Copernicus Open Access; atribucion "Contains modified Copernicus Sentinel data 2017-2025"; cuotas EE respetadas para uso no comercial. |
| R-CMP-04 | Malinterpretar el credito cloud Trial GenAI App Builder ($17,178) | Media | Bajo | Aclaracion REAL: ese credito es de Vertex AI Search/Agent Builder y **NO cubre** la SKU de generacion de texto de Gemini API; el gasto Gemini real es de centavos (~$0.0001/descripcion FarSLIP). Documentado por FinOps (US-061). |

---

## 6. Riesgos de EJECUCION — realidad v8 (transversal, §4 del plan, linea 2530) (`R-EJE-*`)

Incorpora los riesgos de ejecucion del §4 "Gestion de Riesgos" del plan v8 con su probabilidad
y mitigacion REAL, mas el estado actual conocido del repositorio.

| id | Riesgo (v8 §4) | Prob (v8) | Impacto | Mitigacion (v8) + estado real |
|----|----------------|-----------|---------|-------------------------------|
| R-EJE-01 | H100 una sola GPU, cola consume dias | Alta | Alto | Orden estricto: FarSLIP-pheno ablacion -> TSViT full retrain -> ensambles OOF -> Qwen serving; fallback L4 para la ablacion de bandas. Estado: H100 NVL 96GB del sponsor 24/7 sin coste; daemon de auto-shutdown por idle en la L4 spot (pararlo antes de runs manuales largos). |
| R-EJE-02 | RLS migracion falla -> data leak | Media | Alto | "Test en docker-compose local antes de exponer endpoints" (US-051). Estado REAL: la migracion `20260620000418_rls_multi_tenant.sql` YA esta aplicada (rol `agrosat_app` NOBYPASSRLS, `FORCE ROW LEVEL SECURITY`, fail-closed). Riesgo residual: la app DEBE conectar como `agrosat_app` (no como superusuario, que bypassa RLS); deuda de backfill `parcels.session_id` NOT NULL. |
| R-EJE-03 | OOF loop 10-20h | Media | Medio | US-030/031 (harness + OOF) primero; correr en H100; 4 ensambles base = MVP. Sub-riesgo de trazabilidad: el lineage MLflow vive en el server Docker `:5010` (no en `./mlruns`); runs lanzados por subprocess pueden quedar RUNNING contra el server equivocado -> usar `track_experiment` (resuelve URI con fallback `:5010 -> file:./mlruns`). |
| R-EJE-04 | FarSLIP band mismatch 4 vs 10 (incremental 4 -> 18 no converge) | Media | Medio | POC de 2 epochs antes del full; fallback a 18-clase desde cero (`load_state_dict(strict=False)` para init Stage-1 -> Stage-2). |
| R-EJE-05 | Gemini rate-limit | Media | Medio | Reasoner con fallback a on-prem Qwen3.5-35B-A3B (vLLM GPTQ-Int4 single-GPU) + backoff/reintentos. Flash es deviation consciente de Arthur por coste/latencia. Gemma 4 LoRA OUT (ADR-009, future). |
| R-EJE-06 | Incremental Stage-1 (FarSLIP) no converge | Media | Medio | POC corto + fallback desde cero; metricas en MLflow para detener temprano. Confianza ~70% de convergencia con set pequeno (~500-700 patches/fold). |

---

## 7. Matriz probabilidad x impacto (3x3)

Filas = Impacto; columnas = Probabilidad. Cada celda lista los ids de riesgo que caen ahi. La
esquina **Prob Alta x Impacto Alto** (severidad critica) esta destacada.

| | Prob Alta | Prob Media | Prob Baja |
|--|-----------|------------|-----------|
| **Impacto Alto** | **R-EJE-01, R-DAT-04** (critica) | R-EJE-02, R-ATK-04, R-CNF-04 | R-CMP-01 |
| **Impacto Medio** | R-CNF-02 | R-DAT-01, R-DAT-02, R-DAT-03, R-DAT-05, R-ATK-02, R-ATK-03, R-CNF-01, R-CNF-03, R-CMP-02, R-EJE-03, R-EJE-04, R-EJE-05, R-EJE-06 | R-ATK-01, R-CMP-03 |
| **Impacto Bajo** | — | R-CMP-04 | — |

Lectura: la esquina critica concentra el riesgo de cronograma (una sola GPU) y el riesgo
cientifico central (transferibilidad espacial de AlphaEarth). El cuadrante Prob Media x Impacto
Alto reune los tres riesgos de gobernanza/seguridad (RLS, multi-tenant, no sobre-afirmar F1),
todos con mitigacion ya implementada o con postura adoptada.

---

## 8. Plan de monitoreo y reevaluacion

- **Responsable**: Arthur Zizumbo (Tech Lead). **Cadencia**: revision al cierre de cada sprint y
  antes de cada Avance del curso.
- **R-EJE-01 (GPU)**: revisar la cola de entrenamiento y el estado de la VM H100 antes de cada
  run largo; no apagar la H100 del sponsor; parar el daemon idle de la L4 antes de runs manuales.
- **R-DAT-04 / R-CNF-02 / R-CNF-04 (transferibilidad y claims)**: cualquier figura con metrica
  fuera de Francia se valida contra la regla "few-shot, no zero-shot; sin F1 en Mexico".
- **R-ATK-04 / R-EJE-02 (RLS)**: re-verificar que la app conecte como `agrosat_app` (no
  superusuario) antes de exponer endpoints; correr el test de aislamiento cross-session.
- **R-ATK-01 (drift)**: el asset `drift_check` (schedule semanal `0 6 * * 1`) alerta cuando
  `drift_score > 0.3`.
- **R-CMP-02 (licencias)**: actualizar `docs/licenses/DATA_LICENSE.md` y las data sheets (US-064)
  al incorporar cualquier dataset nuevo.

---

## 9. Referencias

- Plan v8: `context/RefinamientoPlaneacionAgroSatCopilot_v8.md` — §4 "Gestion de Riesgos"
  (linea 2530), §5 "Criterios de Exito" (linea 2544), §11.2 (claims defendibles, Mexico
  cualitativo).
- ADR-009: `docs/decisions/ADR-009-h100-reactivacion-pivote-farslip-alcance-v8.md` (reactivacion
  H100, AlphaEarth V1/ANNUAL v1.1 no v2.1, Gemma 4 LoRA future).
- RLS multi-tenant: `db/migrations/20260620000418_rls_multi_tenant.sql` (US-051).
- Transfer multi-region (EPIC 12 cerrada): US-074 (crosswalk HCAT), US-075
  (`reports/segmentation/sen4agrinet_transfer_result.json`), US-076 (few-shot EuroCropsML),
  US-077 (demo Mexico cualitativa).
- Licencias: `docs/licenses/DATA_LICENSE.md`.
- Blockers: `docs/blockers/epic10-notas.md`, `docs/blockers/epic12-vm-setup.md`.
- Rate limiting / API: `backend/app/main.py` (`create_app()`, slowapi), `backend/app/core/rate_limit.py`.
- Drift: `ml/monitoring/drift.py`, asset `drift_check` (US-060).
- Paper de transferibilidad: "Harvesting AlphaEarth", arXiv:2601.00857.
- US relacionadas: US-051 (RLS), US-061 (FinOps), US-064 (security.md / data sheets / model
  cards), US-066 (drift Evidently), US-075/076/077 (transfer multi-region).
