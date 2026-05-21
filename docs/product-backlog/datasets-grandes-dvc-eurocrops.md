# Backlog · Versionar datasets grandes en DVC — EuroCrops FR + subset parcel-level

**Origen**: durante el trabajo de investigacion adicional (mayo 2026, post
US-018) se descargaron e integraron datasets que exceden el tamano razonable
para Git. DVC aun **no esta configurado** en el repo (el remoto
`gs://agrosat-dvc-remote` esta pendiente de provisionar — ver
[`us-016-1-dvc-multisensor-outputs.md`](us-016-1-dvc-multisensor-outputs.md)).
Mientras tanto, estos archivos quedan **excluidos de Git via `data/.gitignore`**
y existen solo en la maquina local. Este item recoge la deuda para versionarlos
formalmente cuando DVC este operativo.

**Estado**: pendiente — bloqueado por la provision del bucket
`gs://agrosat-dvc-remote` (mismo pre-requisito que US-016.1).

**Prioridad**: media — no bloquea el Avance 3 (baseline) ni el Avance 2; los
datasets son para validacion cruzada y feature engineering ampliado. Se vuelve
importante cuando el equipo necesite reproducir estos datos en otra maquina.

**Estimacion**: 1-2 horas (depende de US-016.1 que provisiona el bucket).

---

## 1. Archivos pendientes de versionar

| Artefacto | Tamano | Origen | Uso |
|-----------|--------|--------|-----|
| `data/reference/eurocrops/FR_2018/` (`.shp` + `.dbf` + `.shx` + `.prj` + `.cpg`) | ~11 GB | EuroCrops v11, Zenodo `zenodo.org/records/14094196`, archivo `FR_2018.zip` | Ground truth vectorial Francia 2018 — 9.5 M parcelas con etiqueta HCAT. Validacion cruzada de etiquetas de cultivo PASTIS-R vs catastro oficial frances |
| `data/test_fixtures/feature_selection_parcels_subset.parquet` | ~76 MB | Generado por el pipeline de feature engineering sobre PASTIS-R parcel-level | Subset de features (~85k parcelas) consumido por `notebooks/feature_engineering/03b_fe_spectral_temporal_pastis.ipynb` |

**Ya excluidos de Git** (estado actual): ambos estan en `data/.gitignore`. El
ZIP original `FR_2018.zip` (2.5 GB) se elimino tras la extraccion del shapefile.

**Lo que SI esta commiteado** (referencia ligera, no requiere DVC): las tablas
de taxonomia HCAT en `data/reference/eurocrops/` — `HCAT2.csv`, `HCAT3.csv`,
`fr_2018.csv` (52 KB total). De estas depende `ml/features/encoding.py`
(`derive_crop_group_from_class_id`), por eso se versionan en Git directo.

---

## 2. Por que esta diferido

1. **DVC no esta configurado en el repo.** No existe `.dvc/config` con remoto
   ni el bucket `gs://agrosat-dvc-remote` provisionado. Hacer `dvc add` local
   sin remoto genera `.dvc` files pero no garantiza recuperabilidad cross-machine.
2. **No bloquea entregas inmediatas.** El Avance 3 (baseline) usa las features
   de PASTIS-R ya disponibles; EuroCrops FR es para validacion futura.
3. **Conviene hacerlo de una vez con US-016.1.** Esa US ya provisiona el bucket
   y versiona los outputs multisensor — agregar estos dos artefactos al mismo
   PR evita duplicar el setup de DVC.

---

## 3. Plan de implementacion

**Pre-requisito** (compartido con US-016.1): Arthur provisiona
`gs://agrosat-dvc-remote` + permisos de la service account. Verificar con
`poetry run dvc remote list`.

```bash
# 1. Versionar el shapefile EuroCrops FR (carpeta completa)
dvc add data/reference/eurocrops/FR_2018/

# 2. Versionar el subset parcel-level
dvc add data/test_fixtures/feature_selection_parcels_subset.parquet

# 3. Commit de los .dvc files
git add data/reference/eurocrops/FR_2018.dvc \
        data/test_fixtures/feature_selection_parcels_subset.parquet.dvc
git commit -m "data(E3): track EuroCrops FR + parcel subset via DVC"

# 4. Push al remoto
make dvc-push

# 5. Tags semanticos
git tag eurocrops-fr-2018-v1 -m "EuroCrops FR_2018 shapefile (9.5M parcelas, HCAT)"
git tag feature-selection-parcels-v1 -m "Subset PASTIS-R parcel-level feature engineering"
git push origin --tags
```

**Nota sobre `data/.gitignore`**: tras `dvc add`, DVC reemplaza la entrada
manual en `data/.gitignore` por su propia gestion via los `.dvc` files. Al
versionar, limpiar las lineas manuales de `FR_2018/` y
`feature_selection_parcels_subset.parquet` que se agregaron como parche
temporal — DVC las maneja despues.

---

## 4. Tareas para retomar (checklist)

- [ ] **PRE**: bucket `gs://agrosat-dvc-remote` provisionado (US-016.1)
- [ ] `dvc add data/reference/eurocrops/FR_2018/`
- [ ] `dvc add data/test_fixtures/feature_selection_parcels_subset.parquet`
- [ ] Commit `.dvc` files + limpiar lineas manuales temporales de `data/.gitignore`
- [ ] `make dvc-push` al remoto
- [ ] Tags `eurocrops-fr-2018-v1` y `feature-selection-parcels-v1`
- [ ] Validar `dvc pull` en checkout limpio
- [ ] Actualizar [`docs/research/datasets-investigacion-adicional.md`](../research/datasets-investigacion-adicional.md)
      §1.1 confirmando el versionado DVC

---

## 5. Riesgos

| Riesgo | Mitigacion |
|--------|------------|
| El shapefile de 11 GB satura el remoto GCS | Verificar cuota del bucket; ~11 GB es aceptable para Standard Storage. Si preocupa el costo, versionar solo `.shp`+`.dbf`+`.shx`+`.prj`+`.cpg` sin recomprimir |
| `dvc pull` lento del shapefile de 11 GB | Documentar; quien necesite EuroCrops FR hace pull selectivo. Para validaciones puntuales, leer con `pyogrio` permite `max_features` sin descargar todo |
| El subset parcel-level se regenera con el pipeline | Es deterministico; alternativamente NO versionar y documentar el comando de regeneracion. Decision del equipo al retomar |

---

## 6. Referencias

- Informe de investigacion: [`docs/research/datasets-investigacion-adicional.md`](../research/datasets-investigacion-adicional.md) §1.1
- Backlog DVC hermano (provisiona el bucket): [`us-016-1-dvc-multisensor-outputs.md`](us-016-1-dvc-multisensor-outputs.md)
- Convenciones DVC: `CLAUDE.md` §"Reglas Globales NON-NEGOTIABLE" punto 10
- Atribucion de licencia EuroCrops (CC-BY-SA 4.0): `docs/licenses/DATA_LICENSE.md`
