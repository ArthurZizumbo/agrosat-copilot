# Investigacion adicional — datasets y metodos para EDA + FE

**Estado**: informe para revision del equipo
**Fecha**: 2026-05-18
**Autor**: equipo AgroSatCopilot (ML/Data Science)
**Contexto**: el sponsor Dr. Camacho recomendo (18-may-2026) evaluar dos
articulos/datasets — Rußwurm & Körner 2018 y Tarasiou et al. (Context-Self
Contrastive). Este documento amplia esa recomendacion con un barrido de diez
recursos complementarios para las fases de EDA (EPIC 2) y Feature Engineering
(EPIC 3), priorizados por aporte real a la rubrica y por reproducibilidad en
una laptop de desarrollo sin GPU.

> **Regla de uso.** Este informe es la fuente para decidir que entra al plan
> canonico `context/RefinamientoPlaneacionAgroSatCopilot_v6.md` y a
> `docs/licenses/DATA_LICENSE.md`. Solo lo aprobado por el equipo se integra.
> Lo ya aplicado se marca explicitamente.

---

## 1. Resumen ejecutivo

De la lectura completa de cuatro papers (los dos del sponsor + dos recientes
de 2025) se extrajeron ocho metodos de EDA/FE, **ya implementados** en
[`ml/analysis/paper_methods.py`](../../ml/analysis/paper_methods.py) y
demostrados en `notebooks/eda/02e_eda_metodos_paper.ipynb`.

Adicionalmente se evaluaron diez datasets/recursos. La recomendacion es **no
descargarlos todos** — eso es scope creep — sino:

- **Integrar ya** (alto ROI): taxonomia HCAT de EuroCrops para el encoding
  de cultivos; BreizhCrops (ya integrado por el equipo).
- **Citar como antecedente** (medio ROI): EuroCropsML, TimeSen2Crop,
  TimeMatch, BavarianCrops.
- **Contexto / trabajo futuro** (bajo ROI inmediato): ECOSTRESS, GHISA,
  USGS Phenology Metrics, H2Crop, Sen4AgriNet.

> **Aclaracion sobre el shapefile de EuroCrops.** El record EuroCrops v11 en
> Zenodo (`zenodo.org/records/14094196`) contiene shapefiles vectoriales de
> 19 paises de la UE pero **NO incluye Italia** — Italia no participa en el
> dataset armonizado. El hueco de ground truth italiano (GSAA/LPIS, US-006/007
> diferidos en el plan v6) NO se resuelve con EuroCrops. El shapefile **no
> mejora** `derive_crop_group_from_class_id` — esa funcion solo necesita la
> tabla de taxonomia HCAT3 (ya integrada). El shapefile es ground truth
> geoespacial, util para validacion vectorial cruzada en EPIC 1/3.

### 1.1 EuroCrops FR_2018 — descargado e inspeccionado (2026-05-20)

Se descargo **FR_2018** (Francia 2018, mismo pais y año que PASTIS-R) y se
verifico su estructura. Resultado de la inspeccion con `pyogrio`:

| Atributo | Valor |
|----------|-------|
| Ruta | `data/reference/eurocrops/FR_2018/FR_2018_EC21.shp` |
| Parcelas | **9,517,878** poligonos agricolas |
| CRS | EPSG:2154 (Lambert-93) — **identico al CRS de PASTIS-R** |
| Tamaño | ~11 GB descomprimido (.shp 4.3 GB + .dbf 6.7 GB) |
| Licencia | CC-BY-SA 4.0 |
| Columnas clave | `EC_hcat_n` (nombre HCAT), `EC_hcat_c` (codigo HCAT3), `EC_trans_n` (cultivo traducido), `CODE_CULTU` (codigo RPG frances) |

**Aporte concreto.** EuroCrops FR ya trae cada parcela mapeada a la taxonomia
HCAT3 (columnas `EC_hcat_n` / `EC_hcat_c`). Esto permite **validar la
integracion HCAT** de `ml/features/encoding.py` contra el catastro agricola
oficial frances: las mismas clases de cultivo que PASTIS-R, con su codigo HCAT
verificado por una fuente independiente. Ejemplos: `Winter barley` ->
`winter_barley` -> `3301010401`; `Soft winter wheat` ->
`winter_common_soft_wheat` -> `3301010101`.

**Versionado.** El `.shp`/`.dbf` (~11 GB) NO se commitea a Git (`data/.gitignore`
lo excluye); se versiona en DVC. El ZIP original se elimina tras la extraccion.
Las tablas de taxonomia HCAT (`HCAT2.csv`, `HCAT3.csv`, `fr_2018.csv`, 52 KB
total) si se commitean — son referencia ligera de la que depende `encoding.py`.

---

## 2. Papers leidos y metodos implementados

Los cuatro papers se leyeron completos. Los metodos extraidos viven en
`ml/analysis/paper_methods.py` (8 funciones publicas, cobertura 87 %, 24 tests).

| Paper | Cita | Metodo implementado | Funcion |
|-------|------|---------------------|---------|
| A (sponsor) | Rußwurm & Körner 2018, ISPRS IJGI 7(4):129, arXiv:1802.02080 | Analisis de irregularidad de revisita Sentinel-2 | `temporal_sampling_stats` |
| A (sponsor) | idem | Descomposicion confusiones simetricas vs asimetricas | `confusion_symmetry_analysis` |
| A (sponsor) | idem | Agregacion de clases raras por umbral de frecuencia | `aggregate_rare_classes` |
| B (sponsor) | Tarasiou et al. 2022, IEEE TGRS, arXiv:2104.04310 | Analisis interior-vs-frontera de parcela (Fig. 2) | `boundary_interior_stats`, `boundary_pixel_mask` |
| B (sponsor) | idem | Feature nuevo: ratio de pixeles de frontera por parcela | `compute_boundary_ratio` |
| C | Phenology-Aware Transformer (PVM) 2025, Remote Sensing 17(14):2346 | Calendario fenologico — etapas de crecimiento por DOY | `phenology_calendar_features` |
| D | Qin et al. 2025, STCLN, Int. J. Appl. Earth Obs. Geoinf. | Enmascaramiento espaciotemporal — auditoria de robustez de features | `cloud_gap_robustness` |

**Estos ocho metodos respondene directamente a la recomendacion del sponsor:**
no quedaron en cita, se implementaron como codigo reproducible con tests.

---

## 3. Diez datasets/recursos adicionales evaluados

Cada fila indica: tamaño aproximado, si es reproducible en laptop sin GPU,
a que criterio de rubrica abona (Construccion / Normalizacion / Seleccion /
Conclusiones del Avance 2, o EDA del Avance 1) y el enlace de descarga.

### 3.1 Prioridad ALTA — integrar en codigo

| # | Recurso | Tamaño | Reproducible laptop | Abona a | Enlace |
|---|---------|--------|---------------------|---------|--------|
| 1 | **EuroCrops + taxonomia HCAT** | Shapefiles vectoriales por pais; Italia ~cientos de MB (descarga selectiva) | Si — solo vectores, sin imagenes | **FE / Construccion** — estandar oficial UE de jerarquia de cultivos; reemplaza el mapping inline de `derive_crop_group_from_class_id` en `encoding.py` por la taxonomia HCAT armonizada | https://github.com/maja601/EuroCrops · Zenodo: https://zenodo.org/records/10118572 |
| 2 | **BreizhCrops** (Bretaña, Francia) | ~pocos GB, series pixel-level pre-extraidas | Si | **EDA + FE** — validacion cross-regional Francia (PASTIS) vs Bretaña; lo pidio implicitamente el sponsor via Rußwurm | https://github.com/dl4sits/BreizhCrops · `bash scripts/download_breizhcrops.sh` |

> **Nota.** BreizhCrops **ya esta integrado** por el equipo (PR #18,
> `ml/ingest/breizhcrops_loader.py` + `notebooks/eda/02d_eda_breizhcrops.ipynb`
> + `data/breizhcrops.dvc`). La taxonomia HCAT (#1) es la unica integracion de
> codigo pendiente recomendada — cambio chico en `encoding.py`, alto valor de
> justificacion en la rubrica.

### 3.2 Prioridad MEDIA — citar como antecedente / trabajo relacionado

| # | Recurso | Tamaño | Reproducible laptop | Abona a | Enlace |
|---|---------|--------|---------------------|---------|--------|
| 3 | **EuroCropsML** | 706k puntos, time-series S2 pre-extraido (parquet/npy) | Si — sin imagenes pesadas | **EDA** — benchmark few-shot cross-regional Europa | https://zenodo.org/records/13871419 · arXiv:2407.17458 |
| 4 | **TimeSen2Crop** | ~1M series pixel-level S2 (Austria), 16 clases | Si — ligero | **EDA** — analisis de huecos/nubes a gran escala; valida features temporales | https://zenodo.org/records/4715631 · paper: Weikmann et al. 2021 IEEE JSTARS |
| 5 | **TimeMatch** | 4 regiones EU, series pre-extraidas | Si | **EDA** — desfase temporal (temporal shift) entre regiones; justifica el positional encoding por fecha de TSViT | https://zenodo.org/records/5636422 · arXiv:2111.02682 |
| 6 | **BavarianCrops / MTLCC** | ~pocos GB, series pixel-level (Baviera) | Si | **EDA + FE** — el dataset del paper del sponsor (Rußwurm & Körner); validacion cross-pais Francia↔Alemania | https://github.com/MarcCoru/MTLCC |

### 3.3 Prioridad CONTEXTO — referencia, trabajo futuro

| # | Recurso | Tamaño | Reproducible laptop | Abona a | Enlace |
|---|---------|--------|---------------------|---------|--------|
| 7 | **Sen4AgriNet (S4A)** | Multi-pais (Francia/Cataluña), descargable por streaming | Si — `datasets.load_dataset` | **EDA** — validacion cross-regional; ya listado en v6 §9.1 | https://huggingface.co/datasets/orion-ai-lab/S4A |
| 8 | **ECOSTRESS Spectral Library v1.0** | ~3400 espectros de laboratorio, pocos MB | Si — muy ligero | **FE** — validacion de firmas espectrales de cultivos; interpretacion agronomica del Factor Analysis | https://speclib.jpl.nasa.gov/ |
| 9 | **USGS GHISA** (hyperspectral crops) | Espectros de referencia de cultivos, ligero | Si | **FE** — justificar indices espectrales por fenologia del cultivo | https://www.usgs.gov/centers/wgsc/science/global-hyperspectral-imaging-spectral-library-agricultural-crops-ghisa |
| 10 | **USGS Remote Sensing Phenology Metrics 2024** | Metricas SOST/EOST/MAXN/DUR/AMP/TIN, ligero | Si | **FE** — valida las 8 features fenologicas de US-015 contra nomenclatura USGS estandar | https://www.usgs.gov/special-topics/remote-sensing-phenology/science/deriving-phenology-metrics |
| — | **H2Crop** (hierarchical EnMAP + S2) | Dataset grande (hiperespectral) | Parcial — pesado | FE — jerarquia 4-tier de cultivos; ya citado en v6 ref #27 | https://github.com/flyakon/H2Crop · arXiv:2506.06155 |

---

## 4. Recomendacion final de integracion

| Accion | Recurso | Estado |
|--------|---------|--------|
| Integrar en codigo | BreizhCrops | **Hecho** (PR #18 del equipo) |
| Integrar en codigo | Metodos de los 4 papers (`paper_methods.py`) | **Hecho** (8 funciones, 24 tests, notebook 02e) |
| Integrar en codigo | Taxonomia HCAT de EuroCrops en `encoding.py` | **Hecho** — `derive_crop_group_from_class_id` ahora mapea las 20 clases PASTIS a 8 grupos HCAT3 oficiales; tablas en `data/reference/eurocrops/` (HCAT3.csv, HCAT2.csv, fr_2018.csv) |
| Citar en v6 | Rußwurm & Körner + Tarasiou CSCL (§3.5, §3.6, refs #39-#40) | **Hecho** |
| Citar en v6 | PVM + STCLN (refs #41-#42) | **Hecho** |
| Citar en v6 | BavarianCrops + BreizhCrops en §9.1 datasets | **Hecho** |
| Documentar licencia | BreizhCrops, BavarianCrops | **Hecho** en `DATA_LICENSE.md` |
| Trabajo futuro | EuroCropsML, TimeSen2Crop, TimeMatch, Sen4AgriNet | Solo cita — descargar si hay tiempo en Paper Track |

---

## 5. Enlaces de descarga (consolidado)

Para el equipo — los recursos que conviene tener a mano:

- **BreizhCrops** (ya en uso): `bash scripts/download_breizhcrops.sh` o
  https://github.com/dl4sits/BreizhCrops
- **EuroCrops + HCAT** (recomendado integrar):
  https://github.com/maja601/EuroCrops — la taxonomia HCAT esta en
  `csvs/HCAT2.csv` del repo; el shapefile de Italia se descarga selectivo
  desde https://zenodo.org/records/10118572
- **BavarianCrops / MTLCC** (referencia sponsor): https://github.com/MarcCoru/MTLCC
- **EuroCropsML**: https://zenodo.org/records/13871419
- **TimeSen2Crop**: https://zenodo.org/records/4715631
- **TimeMatch**: https://zenodo.org/records/5636422
- **Sen4AgriNet (S4A)**: https://huggingface.co/datasets/orion-ai-lab/S4A
- **ECOSTRESS Spectral Library**: https://speclib.jpl.nasa.gov/
- **USGS GHISA**: https://www.usgs.gov/centers/wgsc/science/global-hyperspectral-imaging-spectral-library-agricultural-crops-ghisa
- **USGS Phenology Metrics**: https://www.usgs.gov/special-topics/remote-sensing-phenology

### Papers (PDF abierto)

- Rußwurm & Körner 2018: https://arxiv.org/pdf/1802.02080
- Tarasiou et al. 2022 (Context-Self): https://arxiv.org/pdf/2104.04310
- Tarasiou et al. 2023 (TSViT, ya en v6 §3.1): https://arxiv.org/pdf/2301.04944
- PVM 2025: https://www.mdpi.com/2072-4292/17/14/2346
- STCLN 2025: https://www.sciencedirect.com/science/article/pii/S1569843225000731
