# Backlog · US-016.1 — DVC tracking de outputs multisensor + observaciones diferidas

**Origen**: US-016 cerró con estado `ready-to-close` (Fase 4.3, 2026-05-17) cubriendo
AC-1..AC-13 + AC-15 verdes. **AC-14 (DVC push real al remoto GCS) quedó
explícitamente deferido** por dos motivos:

1. El bucket `gs://agrosat-dvc-remote` aún no estaba provisionado al cierre
   de US-016 (riesgo R9 documentado en `docs/us-planning/us-016.md` §9).
2. Los outputs reales de Italia (no solo el fixture demo) requieren ejecutar
   la fusión con GEE real para las 3 regiones — operación que se planificó
   como dependencia downstream cuando el bucket estuviera listo.

Esta US-016.1 cierra el AC-14 pendiente, recoge la observación residual de
calibración detectada en el smoke real GEE (Obs-Bug-6.1) y propone un
refactor menor de DRY identificado en el code review de Fase 4.3.

**Estado**: pendiente — desbloqueada cuando Arthur provisione el bucket
`gs://agrosat-dvc-remote` (acción de plataforma, no requiere US adicional).

**Prioridad**: media-alta — bloquea reproducibilidad cross-checkout para
US-019/US-020/US-021 (baselines tabulares que consumen
`features_fused_v1.parquet`). Sin DVC, esos sprints quedan acoplados a
re-ejecuciones manuales del CLI con cuota GEE.

**Estimación**: 4–6 horas distribuidas en 1 PR principal + 1 opcional.

---

## 1. Motivación

### Por qué cerrar AC-14 ahora

1. **Reproducibilidad cross-machine**: sin DVC, el `features_fused_v1.parquet`
   de Italia solo existe en la máquina donde corrió `make features-fuse-italy`.
   Cualquier devolver al equipo (Aaron, Arthur, sponsor) requiere regenerar
   con GEE (cuota + tiempo).
2. **Tag semántico**: `git checkout fused-features-italy-v1` debe traer el
   parquet exacto vía `dvc pull` — patrón documentado en `CLAUDE.md`
   "Versionado: DVC para datos".
3. **Lineage MLflow**: cuando US-019 baseline registre el run, el tag
   `data_version=fused-features-italy-v1` solo es accionable si el parquet
   asociado vive en el remoto.
4. **Cierre formal de US-016**: AC-14 era criterio de aceptación canonico.
   Mantenerlo abierto deja la US en estado "ready-to-close con deuda" en
   lugar de "closed".

### Por qué NO se hizo en US-016

- Bucket no provisionado (R9). El `dvc add` local sin remoto solo genera
  `.dvc` files pero no garantiza recuperabilidad — mejor diferir para
  hacerlo de una vez con el push real.
- El CLI y el asset Dagster ya producen los outputs determinísticos
  (MD5 byte-equal entre re-ejecuciones, demostrado en QA Fase 4.2).
  Versionar es operación de "publicación", separable del coding.

---

## 2. Scope funcional

### 2.1 AC-14 — DVC push real

Versionar los 3 outputs canónicos de US-016 contra `gs://agrosat-dvc-remote`:

| Artefacto | Tag git | Tamaño estimado |
|-----------|---------|-----------------|
| `data/features/features_fused_v1.parquet` (Italia 3 regiones, año 2024) | `fused-features-italy-v1` | ~5-15 MB |
| `data/splits/spatial_kfold_v1/` (5 folds, train/val/test) | `spatial-kfold-italy-v1` | <1 MB |
| `artifacts/scaler_v1.pkl` (StandardScaler joblib, fit train Fold-0) | `scaler-v1` | <10 KB |

Los comandos exactos ya están documentados en
[`docs/manual-test/us-016.md` §"Comandos exactos para versionar al cierre"](../manual-test/us-016.md).

### 2.2 Obs-Bug-6.1 — Calibración precipitación ERA5

Detectada en smoke real GEE post Bug-6 fix (Pianura Padana 2024):

- `era5_tmean_*` correctos (jan 3.19°C, jul 24.27°C, dentro del rango histórico Po Valley).
- `era5_prec_*` anual acumulado = **1764 mm** vs rango histórico típico **800-1200 mm**.

**Hipótesis del residuo**: el `scale=1000` en `reduceRegions(reducer=mean)` sobre el
`sum()` mensual puede no escalar la dimensión espacial igual que `scale=11132`
nativo. El `mean` espacial de un `sum` temporal es matemáticamente equivalente
solo si el área de la parcela <= 1 píxel ERA5 — para parcelas que cubren múltiples
píxeles oversampleados a 1 km, hay sobre-conteo proporcional.

**Trabajo**:
1. Reproducir el cálculo manualmente con `reduceRegion` (singular) sobre 5 parcelas
   conocidas (Pianura Padana, Toscana, Puglia) y comparar con datos in-situ ARPA.
2. Si el residuo se confirma: ajustar el factor en `ml/ingest/gee_sampler.py:1080`
   (línea del `pval * 1000.0`) o cambiar el reducer a `ee.Reducer.first()` para
   parcelas sub-píxel.
3. Re-ejecutar smoke + actualizar test de regresión con tolerancia física.

**No bloquea US-019** — los stats relativos entre parcelas siguen siendo válidos
para feature importance del baseline tabular. La calibración exacta es nice-to-have
para interpretabilidad agronómica (cuando el sponsor pregunte "¿cuánta lluvia recibió
esa parcela?", la respuesta debe ser fiable a nivel absoluto).

### 2.3 Refactor DRY menor — Loader de parcelas compartido

Detectado en code review Fase 4.3. Hay duplicación de ~20 LoC del loader de
parcelas en:

- `scripts/build_parcel_features.py::_load_parcels` (3 formatos: GeoParquet,
  parquet plano con `geom` WKT, GeoJSON — Bug-5 fix)
- `dagster_project/assets/features.py::_load_parcels_geodataframe` (solo
  parquet plano con `geom` WKT)

**Propuesta**: extraer a `ml/features/_io.py::load_parcels(path: Path) -> gpd.GeoDataFrame`
con soporte de los 3 formatos. CLI y asset Dagster lo consumen.

**Beneficio**: cuando se agregue un 4° formato (ej. GeoPackage, PostGIS connection
string), un solo cambio actualiza ambos puntos de entrada.

**Costo**: bajo (~30 min refactor + actualizar 2 imports + tests existentes
deberían seguir pasando sin cambios).

---

## 3. Plan de implementación

### PR 1 — DVC tracking + push real (~3 h)

**Branch**: `feature/E3-US-016.1-dvc-push`

**Pre-requisito de plataforma** (NO US): Arthur provisiona
`gs://agrosat-dvc-remote` con permisos para la service account configurada
en `DVC_GCS_CREDENTIALS_PATH`. Verificar con:

```bash
gsutil ls gs://agrosat-dvc-remote/    # debe responder, vacío o con contenido
poetry run dvc remote list             # debe listar `origin` apuntando al bucket
```

**Workflow** (replica el §"Comandos exactos" del manual-test US-016):

```bash
# 0. Generar outputs reales de Italia (NO el demo)
# Requiere creds GEE — Isaac corre esto desde su máquina
make features-fuse-italy

# 1. Verificar shapes esperados
poetry run python -c "
import polars as pl
df = pl.read_parquet('data/features/features_fused_v1.parquet')
assert df.height > 100, f'pocos rows: {df.height}'
assert df.width == 191, f'cols inesperadas: {df.width}'
print(f'OK: {df.shape}')
"

# 2. Versionar con DVC
dvc add data/features/features_fused_v1.parquet
dvc add data/splits/spatial_kfold_v1/
dvc add artifacts/scaler_v1.pkl

# 3. Commit de los .dvc files + .gitignore actualizado
git add data/features/features_fused_v1.parquet.dvc \
        data/splits/spatial_kfold_v1.dvc \
        artifacts/scaler_v1.pkl.dvc \
        .gitignore
git commit -m "data(E3): track US-016 multisensor outputs via DVC"

# 4. Push al remoto
make dvc-push   # alias de `dvc push`

# 5. Tags semánticos
git tag fused-features-italy-v1 -m "US-016 fused features v1 (189 cols sin FarSLIP)"
git tag spatial-kfold-italy-v1 -m "US-016 spatial K-fold v1 (5 folds, h3 res 5, buffer 1km)"
git tag scaler-v1 -m "US-016 StandardScaler v1 (fold_0 train)"
git push origin --tags

# 6. Validación pull en checkout limpio (opcional pero recomendado)
cd /tmp && git clone <repo> agrosat-validate
cd agrosat-validate
git checkout fused-features-italy-v1
make dvc-pull
md5sum data/features/features_fused_v1.parquet  # debe coincidir con el original
```

**Actualizar handoff US-016**: marcar AC-14 ✅, cambiar estado `ready-to-close`
→ `closed`.

**Tests**: agregar `tests/dvc/test_pull_roundtrip.py` con un test marcado
`@pytest.mark.integration` que clone en `tmp_path`, haga `dvc pull` y compare
MD5 (requiere `DVC_GCS_CREDENTIALS_PATH` en CI Linux — opcional, documentado).

### PR 2 — Refactor loader compartido (~1 h, opcional)

**Branch**: `feature/E3-US-016.1-loader-refactor`

```python
# ml/features/_io.py (nuevo)
from pathlib import Path
import geopandas as gpd
import polars as pl
from shapely import wkt


def load_parcels(path: Path) -> gpd.GeoDataFrame:
    """Carga parcels desde GeoParquet, parquet plano con WKT, o GeoJSON.

    Path único de carga para CLI (`scripts/build_parcel_features.py`) y
    asset Dagster (`dagster_project/assets/features.py`). Centraliza la
    detección de formato y la conversión WKT → shapely.

    Raises:
        FileNotFoundError: si `path` no existe.
        ValueError: si el parquet no tiene `parcel_id` ni columna geométrica
            reconocible (`geom` WKT o GeoParquet metadata).
    """
    if not path.exists():
        raise FileNotFoundError(f"Parcels path no encontrado: {path}")

    if path.suffix in (".geoparquet", ".parquet"):
        try:
            gdf = gpd.read_parquet(path)
        except Exception:
            frame = pl.read_parquet(path).to_pandas()
            geom_col = "geom" if "geom" in frame.columns else "geometry"
            if geom_col not in frame.columns:
                raise ValueError(
                    f"Parcels en {path} no contiene columna `geom` ni `geometry`."
                ) from None
            geoms = [wkt.loads(g) if isinstance(g, str) else g for g in frame[geom_col].tolist()]
            gdf = gpd.GeoDataFrame(
                frame.drop(columns=[geom_col]), geometry=geoms, crs="EPSG:4326",
            )
    elif path.suffix in (".geojson", ".json"):
        gdf = gpd.read_file(path)
    else:
        raise ValueError(f"Formato no soportado: {path.suffix}. Usa .parquet o .geojson.")

    if "parcel_id" not in gdf.columns:
        raise ValueError(f"Parcels en {path} no contiene la columna `parcel_id`.")
    return gdf
```

Actualizar:
- `scripts/build_parcel_features.py::_load_parcels` → `from ml.features._io import load_parcels`
- `dagster_project/assets/features.py::_load_parcels_geodataframe` → idem
- Re-export en `ml/features/__init__.py`

Tests existentes deberían pasar sin cambios; agregar `tests/ml/features/test_io.py`
con casos por formato (~30 LoC).

### PR 3 — Recalibración ERA5 (~2 h, opcional, post-US-019)

**Branch**: `feature/E3-US-016.1-era5-calibration`

Solo si Obs-Bug-6.1 se manifiesta como problema en US-019 baseline
(feature importance ERA5_prec_* irrealista o el sponsor pide explicabilidad
agronómica fiable a nivel absoluto).

- Reproducir cálculo con datos in-situ ARPA Lombardia 2024 (5 estaciones).
- Si residuo confirmado: cambiar a `ee.Reducer.first()` o ajustar factor.
- Re-correr smoke + actualizar test regresión con tolerancia física
  (`100 < annual_prec_mm < 2000`).
- Documentar en `docs/spectral_indices.md` § "Multisensor fusion vector layout"
  la limitación residual del bloque ERA5.

---

## 4. Tareas para retomar (checklist)

Cuando se arranque esta US:

- [ ] **PRE**: confirmar con Arthur que `gs://agrosat-dvc-remote` está
      provisionado + permisos service account OK
- [ ] Crear branch `feature/E3-US-016.1-dvc-push` desde `develop`
- [ ] Ejecutar `make features-fuse-italy` con creds GEE (Isaac)
- [ ] Verificar shapes de los 3 outputs antes del `dvc add`
- [ ] `dvc add` × 3 + commit `.dvc` files + push al remoto
- [ ] Crear 3 tags semánticos y `git push --tags`
- [ ] Validar pull en checkout limpio (paso 6 del workflow)
- [ ] Actualizar [`docs/us-handoff/us-016.md`](../us-handoff/us-016.md):
      marcar AC-14 ✅, cambiar estado `ready-to-close` → `closed`
- [ ] Crear [`docs/us-resolved/us-016.md`](../us-resolved/) cerrando la US
      formalmente
- [ ] (Opcional PR2) Refactor loader compartido si el costo cabe en el
      mismo sprint
- [ ] (Opcional PR3) Recalibración ERA5 solo si US-019 lo demanda

---

## 5. Riesgos y mitigaciones

| Riesgo | Mitigación |
|--------|------------|
| Service account GCP no tiene `roles/storage.objectAdmin` sobre el bucket | Arthur configura en el pre-requisito; validación con `gsutil cp test.txt gs://agrosat-dvc-remote/` |
| Cuota GEE agotada al regenerar features Italia | El smoke real consumió ~5k elementos para 1 parcela; Italia 3 regiones estimado ~2M (margen 5× sobre cuota diaria 10M) |
| Tag `fused-features-italy-v1` ya existe (alguien lo creó manualmente) | `git tag -l fused-features-italy-v1` antes de crear; usar `-f` solo con consenso del equipo |
| Refactor loader (PR2) rompe el contrato del fixture demo | Tests `test_extract_demo_3regions_end_to_end` y `test_parcel_features_fused_materializes` deben seguir verdes — son el contrato de regresión |
| `dvc pull` lento desde Europa | Documentado; el equipo está en LATAM/EU, latencia GCS aceptable (~1-3 min para 15 MB) |

---

## 6. Criterios de éxito

- [ ] `gs://agrosat-dvc-remote/` contiene los blobs de los 3 outputs
- [ ] `git checkout fused-features-italy-v1 && make dvc-pull` en checkout limpio
      reconstruye `data/features/features_fused_v1.parquet` con MD5 idéntico
- [ ] `docs/us-handoff/us-016.md` muestra AC-14 ✅ y estado `closed`
- [ ] `docs/us-resolved/us-016.md` existe con resumen ejecutivo
- [ ] (PR2 opcional) Tests US-016 siguen pasando 140 + 3 skipped sin
      regresión tras el refactor del loader
- [ ] (PR3 opcional) Smoke ERA5 reporta `annual_prec_mm` dentro del rango
      físico Po Valley `[800, 1200]` con tolerancia ±20%

---

## 7. Referencias

- US-016 handoff (estado `ready-to-close`):
  [`docs/us-handoff/us-016.md`](../us-handoff/us-016.md)
- US-016 manual test (incluye §"Comandos exactos para versionar al cierre"):
  [`docs/manual-test/us-016.md`](../manual-test/us-016.md)
- US-016 planning original (AC-14 + riesgo R9):
  [`docs/us-planning/us-016.md`](../us-planning/us-016.md)
- Convenciones DVC del proyecto: `CLAUDE.md` §"Reglas Globales NON-NEGOTIABLE"
  punto 10 ("Versionado: DVC para datos, MLflow para experimentos")
- Skills involucrados: `agrosat-dvc-mlflow` · `agrosat-dagster-mlops` ·
  `agrosat-ml-features` (refactor loader) · `agrosat-gee-alphaearth`
  (recalibración ERA5)
