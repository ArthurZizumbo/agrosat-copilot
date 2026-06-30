# Serving del copiloto sobre el campeón Voting-3 v2 (12 clases)

> US-081 Grupo B (backend / serving). Documenta (1) el contrato SSE de los dos
> campos nuevos de honestidad de vocabulario y (2) la lista mínima de artefactos
> DVC que el deploy del backend debe materializar para servir el campeón.

## 1. Modelo servido por defecto

Desde US-081 AC4a el copiloto sirve por DEFECTO el campeón **Voting-3 v2**
(`ml/agent/schemas.py::ClassifyParcelInput.model = "voting3"`):

- Para una parcela PASTIS **fold-5** (resuelta en el OOF cacheado) se sirve el voto
  ponderado v2 `tsvit-pheno-v2 (32 timesteps) + utae + xgb-alphaearth` con pesos
  **pineados** `0.902 / 0.0 / 0.098` (`reports/voting_new/cardinalidad.json`):
  france-10 F1-macro `0.9069 -> 0.9264`, **france-12 0.9001**.
- Para una **AOI fresca** sin fila OOF (un polígono nuevo) el voto degrada LIMPIO a
  `xgb-alphaearth` con un warning estructurado `classify_voting3_unavailable` —
  nunca fabrica un posterior. Es exactamente el comportamiento seguro que tenía el
  default histórico `xgb`.
- El flag legacy `use_stacking` solo promueve a `stacking5` cuando `model="xgb"` se
  pasa explícito; con el nuevo default `voting3` el flag legacy se ignora
  (`ClassifyParcelInput.resolved_model`).

El **label-space** activo (qué clases reporta) no se hardcodea: sale de
`Settings.label_space` (env `LABEL_SPACE`) → si `None`, de
`ml.eval.class_remap.DEFAULT_LABEL_SPACE` (= `france-12`). El `/chat` ya inyecta
`settings` al `ToolContext` (`backend/app/services/chat_service.py`), así que el
deploy respeta `france-12` / `LABEL_SPACE=france-9` SIN cambio de código.

## 2. Contrato SSE: campos nuevos en el evento `tool_result`

El reasoner llama `classify_new_parcel`; su resultado fluye como un evento SSE
`tool_result` cuyo `result` es el `ClassificationResult.model_dump(mode="json")`
(serializado por `ml/agent/agent.py::_dump_output` y reenviado verbatim por
`ChatService`). El evento tiene esta forma en el stream:

```
event: tool_result
data: {"name":"classify_new_parcel","ok":true,"result":{ ... }}
```

donde `result` lleva, **además** de los campos históricos (`crop_class`,
`confidence`, `class_probabilities`), los **dos campos nuevos** (US-080 schema,
surface garantizado en US-081 AC6):

| Campo | Tipo | Semántica |
|-------|------|-----------|
| `out_of_vocabulary_classes` | `list[str]` (default `[]`) | Nombres de cultivos que el label-space activo NO resuelve de forma fiable (su conjunto descartado). Poblado al restringir, para que el reasoner sepa el límite del vocabulario y pueda derivar un cultivo fuera de alcance a RAG + fenología en vez de forzar una etiqueta. Para `france-12` son **6** (Winter triticale, Fruits/veg/flowers, Potatoes, Leguminous fodder, Mixed cereal, Sorghum); para `france-9` son **9**. |
| `unresolved_candidate` | `str \| None` (default `None`) | Cuando, al restringir, la clase top RAW (sin restringir) cae FUERA del vocabulario resuelto, el nombre de ese cultivo out-of-vocabulary hacia el que se inclinó la señal cruda. Es la pista explícita para el reasoner de que `crop_class` puede ser un artefacto de renormalización y debe matizarse con anclaje en parcelas vecinas, no reportarse como confiado. `None` cuando la clase top RAW está en vocabulario. |

### Ejemplo de `result` (france-12, top RAW fuera de vocabulario)

```json
{
  "crop_class": "Corn",
  "confidence": 0.62,
  "class_probabilities": {"Corn": 0.62, "Meadow": 0.21, "Grapevine": 0.17},
  "out_of_vocabulary_classes": [
    "Winter triticale", "Fruits, vegetables, flowers", "Potatoes",
    "Leguminous fodder", "Mixed cereal", "Sorghum"
  ],
  "unresolved_candidate": "Potatoes"
}
```

### Retrocompatibilidad

Ambos campos son **opcionales con default** (`[]` / `None`): un `tool_result`
antiguo sin ellos sigue validando. El frontend (US-057) puede renderizar el hedge
condicionando a `unresolved_candidate != null`; si no los conoce, los ignora. Esto
está pineado por el test `test_classification_result_roundtrips_new_fields_through_sse`.

### Tests del contrato

- `backend/tests/integration/test_chat_v2_label_space.py`:
  - `test_chat_classify_surfaces_france12_crop_and_new_fields` (AC4b + AC6): el
    `/chat` corre el `classify` REAL (xgb sobre `features_fused_pastis.parquet`) y
    el `tool_result` SSE lleva un cultivo `france-12` + los 6 out-of-vocab.
  - `test_chat_label_space_override_narrows_to_france9` (AC4b): con
    `LABEL_SPACE=france-9` el posterior se reduce a (≤) 9 clases y 9 out-of-vocab.
  - `test_classification_result_roundtrips_new_fields_through_sse` (AC6): round-trip
    puro `model_dump -> SSE -> validate` sin depender del parquet.

## 3. Lista mínima de artefactos DVC del deploy (AC5)

Por los **pesos pineados**, el voto NO necesita PASTIS-R (ni GT ni geometría) en el
contenedor del backend. La lista mínima de artefactos DVC a materializar en la
imagen `api` es:

| Artefacto DVC | Puntero `.dvc` | Para qué |
|---------------|----------------|----------|
| `ml/eval/oof_new32/` (dir, 3 ficheros) | `ml/eval/oof_new32.dvc` | OOF de `tsvit-pheno-v2` @ 32 timesteps (miembro dominante del voto). |
| `ml/eval/oof/oof_parcel_utae_fold5.parquet` | `ml/eval/oof/oof_parcel_utae_fold5.parquet.dvc` | OOF del miembro U-TAE (peso 0.0, pero necesario para el tensor alineado de miembros). |
| `ml/eval/oof/oof_parcel_xgb-alphaearth_fold5.parquet` | `ml/eval/oof/oof_parcel_xgb-alphaearth_fold5.parquet.dvc` | OOF del miembro xgb-alphaearth + fallback de degradación. |

Comando de referencia (idéntico al que ejecuta la imagen):

```bash
dvc pull ml/eval/oof_new32 \
         ml/eval/oof/oof_parcel_utae_fold5.parquet \
         ml/eval/oof/oof_parcel_xgb-alphaearth_fold5.parquet
```

Remote DVC: `gs://agrosat-dvc-remote` (`.dvc/config`, remote `gcs-remote`).

> **NO** se incluye `data/PASTIS-R` en el deploy del voto: los pesos pineados
> eliminaron la dependencia de GT/geometría (simplificación real del contenedor).

### Cómo lo hace la imagen (`infrastructure/docker/backend.Dockerfile`)

La etapa `dvc-data` (entre `builder` y `runtime`) instala `dvc[gs]`, copia solo los
`.dvc` necesarios + `.dvc/config`, y ejecuta el `dvc pull` con la clave GCS montada
como **build secret** `gcs_dvc_key` (BuildKit `--mount=type=secret`). La etapa
`runtime` copia los parquets resultantes encima del árbol `ml/eval` traído de git
(que solo lleva los punteros `.dvc`).

- **Cloud Build** (`infrastructure/cloudbuild.yaml`): el step `build-api` corre con
  `DOCKER_BUILDKIT=1` y monta el secreto `GCS_DVC_KEY` (Secret Manager
  `agrosat-dvc-sa-key`, una SA con `roles/storage.objectViewer` sobre el bucket DVC)
  vía `availableSecrets`.
- **Build local sin credenciales**: el `dvc pull` se **omite con aviso ruidoso** y
  la imagen queda sin los parquets → el copiloto degrada `voting3 -> xgb-alphaearth`
  en runtime (log `classify_voting3_unavailable`). El build NO falla.

### Smoke del deploy

Tras desplegar la imagen `api`, validar que el voto carga dentro del contenedor:

```bash
# Dentro del contenedor / Cloud Run job de smoke:
python -c "from ml.agent.tools.classify import _load_voting_three; \
           v = _load_voting_three(); print('voting3 parcels:', len(v.member_probs_by_id))"
# Debe imprimir > 0; un FileNotFoundError indica que faltó el dvc pull (AC5 falla ruidoso).
```

## 4. Bloqueos conocidos (deploy real)

El **deploy real** (build + push + Cloud Run) requiere credenciales GCP / Cloud
Build / el secreto `agrosat-dvc-sa-key` provisionado con valor. El código del
`Dockerfile` y del `cloudbuild.yaml` queda LISTO y documentado; la ejecución real y
su smoke se anotan en `docs/us-handoff/BLOQUEOS-nocturnos-2026-06-29.md`.
