# US-023-preview v2 — Manual de ejecucion de notebooks baseline

> Actualizado: 2026-05-27
> Owner: Arthur Zizumbo
> Estado: listo para ejecutar localmente o en VM L4 GCP.

Esta US-023-preview v2 reconstruye desde cero los 6 notebooks de
`notebooks/baseline/` siguiendo el estandar `notebooks/CLAUDE.md` y
eliminando todos los `skip` silenciosos, mocks y datos sinteticos que
plagaba la corrida anterior. Cada notebook ahora:

- Materializa sus propios datos faltantes (FarSLIP path canonico, PASTIS
  eval subset, S2 anchors GEE, pheno_text Gemini, RemoteCLIP HF) llamando
  a los modulos `ml/ingest/*` desde celdas explicitas.
- Persiste sus artefactos (parquet, joblib, PNG) en `reports/baseline/<nb>`
  y `paper/figures/us-023-preview/<nb>`.
- Reusa funciones de `ml/utils/*`, `ml/eval/*`, `ml/train/*` (cero codigo
  inline de logica nueva).
- Imprime graficas inline con `display(fig)` mas las persiste como PNG.

## Prerequisitos en disco

| Recurso | Ruta | Como obtener |
|---|---|---|
| Subset US-018 features | `data/test_fixtures/feature_selection_parcels_subset.parquet` | `dvc pull` |
| Geoparquet parcelas Italia | `data/processed/pastis_parcels_full.geoparquet` | `make build-parcels-geoparquet` o `dvc pull` |
| FarSLIP v2 path canonico | `data/farslip/embeddings_italy.parquet` | `dvc pull data/farslip/embeddings_italy.parquet.dvc` |
| PASTIS-R full | `data/PASTIS-R/` | `dvc pull data/PASTIS-R.dvc` o descargar de Zenodo |
| Class mapping JSON | `data/reference/pastis_class_mapping.json` | Ya commiteado en git |

## Prerequisitos en `.env.local`

```dotenv
# Earth Engine (US-023-preview P5 firma espectral)
GEE_PROJECT_ID=agrosat-copilot
GEE_SERVICE_ACCOUNT_PATH=/abs/path/to/sa.json   # opcional; si ausente usa ADC

# Gemini 3.5 Flash (US-023-preview P4 pheno_text)
GEMINI_API_KEY=...                              # generar en https://aistudio.google.com/apikey
# o alternativamente
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=agrosat-copilot
GOOGLE_CLOUD_LOCATION=us-central1
```

Si `GEMINI_API_KEY` falta y `ENFORCE_GEMINI_API_KEY=True` (default en
`05_reencuadre_fenologico.ipynb`), el notebook **falla explicitamente**
con instrucciones, no salta silenciosamente.

## Orden recomendado de ejecucion

| Orden | Notebook | Tiempo esperado | Compute |
|---|---|---|---|
| 1 | `04b_baseline.ipynb` | 5-10 min | CPU + GPU XGB |
| 2 | `04_baseline.ipynb` | 30-60 min | CPU + GPU XGB/LGBM |
| 3 | `04c_baseline.ipynb` | 20-40 min | CPU + GPU XGB |
| 4 | `04_farslip_eval_pastis.ipynb` | 30-90 min | GPU (RemoteCLIP) + PASTIS-R |
| 5 | `05_reencuadre_fenologico.ipynb` | 1-3 horas | GPU + GEE + Gemini ($8-50 USD) |
| 6 | `Avance3.Equipo17.ipynb` | 5-10 min | CPU (lectura de artefactos) |

**Costos esperados** (`05_reencuadre_fenologico`):

- Gemini 3.5 Flash sobre 85951 parcelas: `~$8.6 USD` (`~$0.0001/parcela`).
- GEE quota: gratis con SA + proyecto activo.
- RemoteCLIP: descarga de pesos ~700 MB primera vez.

## Como ejecutar

### Opcion A — Local Run All en Jupyter / VS Code

1. Abrir el notebook en VS Code.
2. Click en "Run All" arriba.
3. El bootstrap (celda 3) muestra el estado de credenciales y warnings si
   alguna falta.
4. Los datos faltantes se materializan automaticamente en celdas
   posteriores.

### Opcion B — Papermill end-to-end (CI-friendly)

```bash
# Reconstruir los 6 notebooks desde el builder unificado (no los ejecuta)
make baseline-notebooks-v2-build

# Ejecutar los 6 notebooks en orden (requiere ~3 horas y GEMINI_API_KEY)
make baseline-notebooks-v2-run
```

### Opcion C — Solo uno

```bash
poetry run python scripts/build_baseline_notebooks_v2.py --only 04b_baseline
MPLBACKEND=Agg poetry run papermill \
  notebooks/baseline/04b_baseline.ipynb \
  notebooks/baseline/04b_baseline.ipynb \
  --no-progress-bar
```

## Como conectar VS Code al kernel L4 GCP

Si quieres ejecutar las celdas pesadas (GEE muestreo + Gemini full +
RemoteCLIP) en la VM L4 GCP en lugar de tu laptop, sigue uno de los dos
patrones.

### Patron 1 — SSH tunnel + Jupyter Lab remoto (recomendado)

En la VM L4:

```bash
# Una sola vez: levantar Jupyter Lab en localhost:8888 con token
poetry run jupyter lab --no-browser --port 8888 \
  --IdentityProvider.token=SOMETOKEN_LARGO_Y_ALEATORIO
```

En tu laptop:

```bash
# SSH tunnel del puerto remoto al local
gcloud compute ssh agrosat-l4 --zone us-central1-a -- -L 8888:localhost:8888
```

En VS Code:

1. `Ctrl+Shift+P` → "Notebook: Select Kernel".
2. "Existing Jupyter Server" → ingresar
   `http://localhost:8888/?token=SOMETOKEN_LARGO_Y_ALEATORIO`.
3. El kernel ahora corre en la L4, pero los archivos los editas en tu
   laptop via "Remote Repositories" o `gcloud compute scp`.

### Patron 2 — VS Code Remote-SSH directo

1. Instalar la extension "Remote - SSH" en VS Code.
2. `Ctrl+Shift+P` → "Remote-SSH: Connect to Host" → ingresar
   `usuario@<IP-L4>`.
3. Abrir el proyecto `agro_sat_copilot/` directamente desde la VM.
4. `Ctrl+Shift+P` → "Notebook: Select Kernel" → "Python Environments" →
   `agrosatcopilot-Kq8fUqSH-py3.12` (el venv de poetry en la VM).

Ambos patrones requieren que la VM tenga el repo clonado + `poetry install
--with ml,geo` ejecutado.

## QA — checklist post-ejecucion

Despues de correr los 6 notebooks, verificar:

- [ ] `reports/baseline/04_baseline/model_comparison_04.parquet` con 3
      filas (RF, XGB, LGBM) y `f1_macro > 0.30` para al menos uno.
- [ ] `reports/baseline/04c_baseline/ablation_table.parquet` con `>= 6`
      filas y `alphaearth_only` con `n_features > 0` (fix del bug NaN).
- [ ] `reports/baseline/05_reencuadre/ablation_table.parquet` con `>= 8`
      filas incluyendo `with_farslip`, `with_pheno_text`,
      `with_spectral_signature`, `geom_only`.
- [ ] `data/features/features_fused_winning_italy.parquet` existe (output
      de `Avance3.Equipo17.ipynb`) + su `.manifest.json` con la lista
      nominal de columnas ganadoras.
- [ ] `paper/figures/us-023-preview/*/` contiene PNGs para cada notebook
      (ablation, comparison, confusion_matrix, etc.).
- [ ] Todas las celdas codigo de los HTML regenerados imprimen tablas y
      figuras (cero `WARN: ... no existe; generando sintetico ...`).

## Troubleshooting

### "GEMINI_API_KEY ausente"

El notebook 05 hace `assert env.has_gemini_api_key` por defecto. Para
correr solo las ablaciones base sin Gemini:

```bash
papermill 05_reencuadre_fenologico.ipynb out.ipynb -p ENABLE_PHENO_TEXT False
```

### "FarSLIP parquet no encontrado"

`data/farslip/embeddings_italy.parquet` se publica via DVC. Ejecuta:

```bash
dvc pull data/farslip/embeddings_italy.parquet.dvc
```

### "PASTIS-R no encontrado"

`data/PASTIS-R/` pesa ~30 GB. Si no esta en disco, descargar desde Zenodo
(https://zenodo.org/records/5012942) o `dvc pull data/PASTIS-R.dvc` si tu
remote DVC lo tiene.

### "schemas incompatibles parcel_id String vs Int64"

Este bug se corrigio en US-023-preview v2 — `parcel_id` ahora es siempre
`pl.Utf8` en toda la stack (`ml/utils/parcel_id.canonical_parcel_id`). Si
vuelve a aparecer, significa que alguna funcion downstream no esta
pasando por `canonical_parcel_id`. Reportar en el issue tracker.

### "alphaearth_only n_features=0 NaN"

Idem — corregido en US-023-preview v2 con regex generalizado
`^(?:ae|emb|alphaearth|dim)_\d{2,3}$` en `ml/eval/feature_ablation.py`. Si
el dataset realmente no tiene columnas AlphaEarth, el log emite
`ae_cols_empty` warning sin romper.

## Referencias

- Estandar de notebooks: `notebooks/CLAUDE.md`
- Plan US-023-preview original: `docs/us-planning/us-023-preview.md`
- ADR-006 (reencuadre fenologico): `docs/decisions/ADR-006-reencuadre-baseline-fenologico.md`
- Paper-faro fenologico: Wen et al. (2025), *Phenology description is all you need!*
- Paper-faro firma espectral: Frampton et al. (2013), *Sentinel-2 biophysical*.
