# Catálogo canónico de los 17 índices espectrales (AgroSatCopilot)

Esta es la tabla de referencia académica del catálogo implementado en
[`ml/features/spectral_indices.py`](../ml/features/spectral_indices.py).
Cada índice tiene una **única versión canónica fijada en este documento** y
no debe revisitarse sin abrir un ADR.

## Convenciones

- **Bandas Sentinel-2**: orden canónico `PASTIS_S2_BANDS` =
  (`B02`, `B03`, `B04`, `B05`, `B06`, `B07`, `B08`, `B8A`, `B11`, `B12`).
  Mapeo a nomenclatura spyndex en `_BAND_TO_SPYNDEX` del módulo:
  `B02→B`, `B03→G`, `B04→R`, `B05→RE1`, `B06→RE2`, `B07→RE3`, `B08→N`,
  `B8A→RE4`, `B11→S1`, `B12→S2`.
- **Reflectancia**: el caller es responsable de escalar DN a [0, 1]
  (dividir por 10000) y aplicar máscara SCL antes de invocar `compute_index`.
  Sin máscara, el EDA del Avance 1 mostró saturación NDVI a 1.0 en 75 % de
  parcelas.
- **Backends**: `spyndex` para 14/17 (11 literal + 3 alias) y fórmula custom
  para los 3 restantes (LAI/FAPAR/CCCI) con DOI propio.

## Tabla académica

| # | Nombre | Fórmula | Autor + año | DOI / Referencia | Uso agronómico | Rango esperado | Bandas S2 | Backend |
|---|--------|---------|-------------|------------------|----------------|----------------|-----------|---------|
| 1 | NDVI | (N − R)/(N + R) | Rouse et al. 1974 | NASA SP-351 (1974) | Vigor vegetativo general; satura a partir de LAI ~3 | [−1, 1] | B04, B08 | spyndex `NDVI` |
| 2 | NDWI | (G − N)/(G + N) | McFeeters 1996 | 10.1080/01431169608948714 | Cuerpos de agua y agua foliar | [−1, 1] | B03, B08 | spyndex `NDWI` |
| 3 | NDMI | (N − S1)/(N + S1) | Gao 1996 | 10.1016/S0034-4257(96)00067-3 | Humedad canopy / estrés hídrico | [−1, 1] | B08, B11 | spyndex `NDMI` |
| 4 | EVI | g·(N − R)/(N + C1·R − C2·B + L), g=2.5, C1=6, C2=7.5, L=1 | Huete et al. 2002 | 10.1016/S0034-4257(02)00096-2 | Vigor canopy denso; corrige aerosoles | [−1, 1] | B02, B04, B08 | spyndex `EVI` |
| 5 | SAVI | (1+L)·(N − R)/(N + R + L), L=0.5 | Huete 1988 | 10.1016/0034-4257(88)90106-X | Vigor con suelo expuesto | [−1.5, 1.5] | B04, B08 | spyndex `SAVI` |
| 6 | MSAVI2 | 0.5·(2N+1 − √((2N+1)² − 8(N−R))) | Qi et al. 1994 | 10.1016/0034-4257(94)90134-1 | SAVI auto-calibrado | [−1, 1] | B04, B08 | spyndex `MSAVI` (alias) |
| 7 | NBR | (N − S2)/(N + S2) | Key & Benson 2006 | USGS Tech. Rep. RMRS-GTR-164 | Severidad de fuego | [−1, 1] | B08, B12 | spyndex `NBR` |
| 8 | MCARI | ((RE1 − R) − 0.2·(RE1 − G))·(RE1/R) | Daughtry et al. 2000 | 10.1016/S0034-4257(00)00113-9 | Clorofila en hoja | [−2, 2] | B03, B04, B05 | spyndex `MCARI` |
| 9 | CCCI | NDRE / NDVI | Barnes et al. 2000 | Proc. 5th Int. Conf. Precision Agric. | Clorofila corregida por canopy (N status) | [−2, 2] | B04, B05, B08 | **custom** `_ccci_barnes_2000` |
| 10 | LAI | −ln(1 − (NDVI − 0.05)/0.95) / 0.5 | Boegh et al. 2002 | 10.1016/S0034-4257(01)00342-X | Índice de área foliar | [0, 8] | B04, B08 | **custom** `_lai_boegh_2002` |
| 11 | FAPAR | 1.24·NDVI − 0.168 | Myneni & Williams 1994 | 10.1016/0034-4257(94)90016-7 | Fracción PAR absorbida (fotosíntesis) | [−0.2, 1.1] | B04, B08 | **custom** `_fapar_myneni_1997` |
| 12 | PSRI | (R − B)/RE2 | Merzlyak et al. 1999 | 10.1034/j.1399-3054.1999.106119.x | Senescencia / pigmentos | [−1, 2] | B02, B04, B06 | spyndex `PSRI` |
| 13 | NDCI | (RE1 − R)/(RE1 + R) | Mishra & Mishra 2012 | 10.1016/j.rse.2011.10.016 | Clorofila-a en aguas continentales | [−1, 1] | B04, B05 | spyndex `NDCI` |
| 14 | GCVI | N/G − 1 | Gitelson et al. 2003 | 10.1029/2002GL016450 | Clorofila verde en canopy | [0, 20] | B03, B08 | spyndex `CIG` (alias) |
| 15 | RENDVI | (RE2 − RE1)/(RE2 + RE1) | Gitelson & Merzlyak 1994 | 10.1016/S0176-1617(11)81633-0 | NDVI red-edge; estrés temprano | [−1, 1] | B05, B06 | spyndex `RENDVI` |
| 16 | NDRE | (N − RE1)/(N + RE1) | Barnes et al. 2000 | Proc. 5th Int. Conf. Precision Agric. | Cultivos densos donde NDVI satura | [−1, 1] | B05, B08 | spyndex `NDREI` (alias) |
| 17 | TSAVI | sla·(N − sla·R − slb)/(sla·N + R − sla·slb), sla=1, slb=0 | Baret & Guyot 1991 | 10.1016/0034-4257(91)90009-U | SAVI calibrado con línea de suelo | [−1.5, 1.5] | B04, B08 | spyndex `TSAVI` |

## Alias documentados (3)

| Nombre del proyecto | Alias spyndex 0.10 | Justificación |
|---------------------|--------------------|---------------|
| MSAVI2 | `MSAVI` | La fórmula spyndex de `MSAVI` coincide literal con MSAVI2 de Qi 1994; el nombre histórico `MSAVI1` no se popularizó |
| NDRE | `NDREI` | Mismo cálculo (Normalized Difference Red-Edge Index); spyndex usa el sufijo `I` |
| GCVI | `CIG` | spyndex documenta este índice como **Chlorophyll Index Green** (Gitelson) |

## Fórmulas custom auditadas (3)

Para los 3 índices sin equivalente en spyndex 0.10 fijamos una versión
canónica única con DOI. Estos índices tienen ≥3 variantes en la literatura;
la elección se justifica abajo y no se revisita sin ADR.

### LAI (Boegh et al. 2002)

```
LAI = -ln(1 - (NDVI - 0.05) / 0.95) / 0.5
```

Forma logarítmica derivada de la ley de Beer-Lambert aplicada al NDVI con
constantes calibradas sobre cultivos europeos (trigo, cebada, remolacha).
Alternativas descartadas: Su 2002 (válida solo bosques tropicales), Asner
2003 (requiere LAI in-situ para calibrar α/β regionales).

**Referencia**: Boegh, E., Soegaard, H., Broge, N., Hasager, C.B.,
Jensen, N.O., Schelde, K., Thomsen, A. (2002). *Airborne multispectral
data for quantifying leaf area index, nitrogen concentration, and
photosynthetic efficiency in agriculture*. Remote Sensing of Environment
81(2-3), 179-193. DOI [10.1016/S0034-4257(01)00342-X](https://doi.org/10.1016/S0034-4257(01)00342-X).

### FAPAR (Myneni & Williams 1994)

```
FAPAR = 1.24 · NDVI − 0.168
```

Ajuste lineal sobre bosques templados, frecuentemente citado como
"Myneni 1997". Alternativas descartadas: Sellers 1985 (no lineal,
requiere LUT), MODIS-FAPAR (depende de inversión 3D del radiative
transfer no replicable a nivel de pixel S2 aislado).

**Referencia**: Myneni, R.B., Williams, D.L. (1994). *On the
relationship between FAPAR and NDVI*. Remote Sensing of Environment
49(3), 200-211. DOI [10.1016/0034-4257(94)90016-7](https://doi.org/10.1016/0034-4257(94)90016-7).

### CCCI (Barnes et al. 2000)

```
CCCI = NDRE / NDVI
```

Ratio normalizado del Canopy Chlorophyll Content Index. NDRE captura
clorofila en hoja (sensible vía red-edge) y NDVI normaliza por densidad
de canopy, eliminando confusión "más verde porque más denso" vs "más
verde porque más clorofila".

**Referencia**: Barnes, E.M., Clarke, T.R., Richards, S.E., Colaizzi, P.D.,
Haberland, J., Kostrzewski, M., Waller, P., Choi, C., Riley, E.,
Thompson, T., Lascano, R.J., Li, H., Moran, M.S. (2000). *Coincident
detection of crop water stress, nitrogen status and canopy density using
ground-based multispectral data*. Proceedings of the 5th International
Conference on Precision Agriculture, Bloomington MN.

## Hallazgos del EDA Avance 1 que validan el catálogo

`notebooks/eda/Avance1.Equipo17.ipynb` §3 (bivariado) y §5 (conclusiones):

- **Redundancia detectada** dentro de `{NDVI, NDRE, NDWI, SAVI}` con Pearson
  0.95-0.97. La selección final (US-018) retendrá `{NDVI, NDMI, EVI}` por
  diversidad espectral (NDMI usa SWIR B11, dimensión independiente).
- **NDVI satura a 1.0** en 75 % de parcelas sin máscara SCL. El módulo
  documenta el contrato del caller en su docstring.
- **DN vs reflectancia**: las bandas S2 distribuidas como DN (0-10000)
  deben dividirse por 10000 antes de invocar `compute_index`. El módulo no
  hace esta conversión para no acoplarse a una fuente de datos específica.

## Temporal aggregation (US-015)

El módulo [`ml/features/temporal_features.py`](../ml/features/temporal_features.py)
agrega cada uno de los 17 índices a lo largo del ciclo agrícola anual y
deriva features fenológicos a partir del NDVI. La salida es un
`polars.DataFrame` con ~187 columnas por `(parcel_id, year)` consumido por
la tabla `features_parcels`.

### Estadísticos descriptivos (9 stats × 17 índices = 153 columnas)

Por cada índice se calculan: `mean`, `std`, `min`, `max` y los percentiles
`p05`, `p25`, `p50`, `p75`, `p95`. Las muestras inválidas (`NaN`/`null`) se
descartan antes de agregar (el caller es responsable del filtrado SCL aguas
arriba). El cálculo usa `pl.LazyFrame.group_by().agg().collect(engine="streaming")`
(Polars 1.x) para escalar a ~30 k parcelas Italia sin desbordar memoria.

### Descomposición harmónica FFT (4 amplitudes + 4 fases × 3 índices = 24 columnas)

Sobre los 3 índices clave (NDVI, NDWI, EVI) se aplica `np.fft.rfft` a la
serie pre-interpolada linealmente a una rejilla diaria — Sentinel-2 tiene
revisita irregular (~5 días con huecos por nubes) y FFT asume muestreo
regular. Se conservan **4 componentes** (DC + 3 armónicos):

- **k=0 (DC)**: equivale a la media; amplitud = `|X[0]| / N`. Fase reportada
  como 0 (carece de interpretación física).
- **k=1 (anual)**: captura la estacionalidad principal del cultivo.
- **k=2 (bimestral / semestral)**: distingue ciclos doble-pico (raros en
  Italia) y siembra-cosecha desfasadas.
- **k=3 (mensual)**: residuo de alta frecuencia útil para detectar
  perturbaciones cortas (riegos, eventos meteorológicos).

Convención single-sided: amplitud de armónicos = `|X[k]| * 2 / N`; fase en
radianes en `(-π, π]`. Validado por test `test_fft_synthetic_sin` con
`sin(2πt/365)` recuperando amplitud `1±0.05` y fase `-π/2±0.1`.

### Features fenológicos NDVI (8 columnas)

Implementados según el criterio de umbral fijo de White et al. 1997 con
`sog_threshold = 0.3`:

- `sog_doy` (start of greenness, día del año en que NDVI cruza 0.3 ascendente).
- `peak_doy`, `peak_value` (máximo de la curva interpolada diaria).
- `senescence_doy` (primer cruce descendente del umbral tras el peak).
- `ndvi_auc` (integral trapezoidal de la curva, proxy de productividad
  primaria bruta — Reed et al. 2003).
- `ndvi_slope_pre_peak`, `ndvi_slope_post_peak` (pendientes de regresión
  lineal SOG→peak y peak→senescencia, en unidades NDVI/día).
- `maturity_duration_days` (longitud de la racha contigua en torno al peak
  con NDVI ≥ `maturity_pct × peak_value`, default 80 %; métrica
  TIMESAT-like — Jönsson & Eklundh 2002).

**Manejo graceful**: si NDVI nunca cruza el umbral (parcela sin vegetación o
con peak por debajo de 0.3) las 8 métricas son `None` (NULL en Postgres) sin
lanzar excepción.

### Referencias académicas (temporal aggregation)

- White, M.A., Thornton, P.E., Running, S.W. (1997). *A continental
  phenology model for monitoring vegetation responses to interannual
  climatic variability*. Global Biogeochemical Cycles 11(2), 217-234. DOI
  [10.1029/97GB00993](https://doi.org/10.1029/97GB00993).
- Reed, B.C., White, M., Brown, J.F. (2003). *Remote sensing phenology*.
  En *Phenology: An Integrative Environmental Science*, Springer. DOI
  [10.1007/978-94-007-0632-3](https://doi.org/10.1007/978-94-007-0632-3).
- Jönsson, P., Eklundh, L. (2002). *Seasonality extraction by function
  fitting to time-series of satellite sensor data*. IEEE Transactions on
  Geoscience and Remote Sensing 40(8), 1824-1832. DOI
  [10.1109/TGRS.2002.802519](https://doi.org/10.1109/TGRS.2002.802519).
- Eklundh, L., Jönsson, P. (2017). *TIMESAT 3.3 with seasonal trend
  decomposition and parallel processing — Software Manual*. Lund and
  Malmö Universities. ISBN 978-91-87983-19-0.

## Multisensor fusion vector layout (US-016)

El módulo [`ml/features/fusion.py`](../ml/features/fusion.py) fusiona seis bloques
heterogéneos por `(parcel_id, year)` produciendo un vector tabular de **189
columnas** (cuando el bloque opcional FarSLIP está deshabilitado) o **701
columnas** con FarSLIP. El frame resultante se persiste en
`data/features/features_fused_v1.parquet` y se versiona con DVC.

El subset de estadísticos temporales `FUSION_STATS = (mean, std, p25, p50, p95)`
es **distinto** al de US-015 (`mean, std, min, max, p05, p25, p50, p75, p95`)
— US-016 reduce a 5 stats por economía cuando combinamos 6 sensores; los 9 stats
completos siguen disponibles en la tabla `features_parcels` de US-015 para
modelos puramente ópticos.

### Tabla de bloques

| # | Bloque | Cols | Prefijo / convención | Fuente | Reducción |
|---|--------|------|----------------------|--------|-----------|
| 1 | AlphaEarth | 64 | `ae_00 .. ae_63` (`float32`) | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` vía `sample_alphaearth_at_coords` | media de las 64 dims sobre los pixeles de la parcela |
| 2 | Indices×Stats | 85 | `{idx}_{stat}` con 17 indices × 5 stats | `ml/features/spectral_indices.py` (US-014) sobre S2 L2A | NDVI lee `features_parcels.ndvi_stats` JSONB (US-015); resto recomputado on-the-fly con cache Redis |
| 3 | Sentinel-1 | 10 | `s1_vv_{stat}` (5) + `s1_vh_{stat}` (5) | `COPERNICUS/S1_GRD` IW GRDH asc+desc, despeckle Lee 7×7, sigma0 en dB | stats anuales sobre ~60 timesteps por parcela |
| 4 | SRTM | 3 | `srtm_elev_mean`, `srtm_slope_mean`, `srtm_aspect_dominant` | `USGS/SRTMGL1_003` + `ee.Terrain.slope` + `ee.Terrain.aspect` | elev / slope: media de pixeles; aspect: bin dominante de 8 cuadrantes (N..NW) |
| 5 | ERA5 mensual | 24 | `era5_tmean_m01..m12` (12) + `era5_prec_m01..m12` (12) | `ECMWF/ERA5_LAND/DAILY_AGGR` agrupado server-side por mes | tmean en °C; prec acumulado mm |
| 6 | Geometría | 3 | `geom_area_ha`, `geom_perimeter_m`, `geom_elongation` | columna `parcels.geom` (POSTGIS / shapely) | area: `ST_Area(geom::geography)/10000`; elongation: `perimeter² / (4·π·area)` (Polsby-Popper inverso) |
| 7 | FarSLIP (opcional) | +512 | `farslip_000..farslip_511` (`float32`) | `data/farslip/embeddings_italy.parquet` (US-016b) | LEFT join por `parcel_id`; si el parquet no existe, el log emite `farslip_block_skipped=True` y el bloque se omite |

### Layout exacto del frame

```
parcel_id (i64) | year (i16) |
ae_00 ... ae_63 (64) |
ndvi_mean, ndvi_std, ndvi_p25, ndvi_p50, ndvi_p95,
... (17 indices × 5 stats = 85 cols, orden indice outer × stat inner) ...
tsavi_mean, tsavi_std, tsavi_p25, tsavi_p50, tsavi_p95 |
s1_vv_mean, s1_vv_std, s1_vv_p25, s1_vv_p50, s1_vv_p95 (5) |
s1_vh_mean, s1_vh_std, s1_vh_p25, s1_vh_p50, s1_vh_p95 (5) |
srtm_elev_mean, srtm_slope_mean, srtm_aspect_dominant (3) |
era5_tmean_m01 ... era5_tmean_m12 (12) |
era5_prec_m01 ... era5_prec_m12 (12) |
geom_area_ha, geom_perimeter_m, geom_elongation (3) |
[farslip_000 ... farslip_511 (512)]  # opcional
```

Total: **2 + 64 + 85 + 10 + 3 + 24 + 3 = 191 columnas** (189 features +
`parcel_id` + `year`). Con FarSLIP: **703 columnas**.

Constantes del módulo:

- `EXPECTED_COL_COUNT_NO_FARSLIP = 189`
- `EXPECTED_COL_COUNT_WITH_FARSLIP = 701`
- `FUSION_STATS = ("mean", "std", "p25", "p50", "p95")`
- `BLOCK_NAMES = ("alphaearth", "indices_stats", "sentinel1", "srtm",
  "era5_monthly", "geometry", "farslip")`

### Orquestación dual

El mismo punto de entrada `ml.features.fusion.build_fused_features` es
consumido por:

- el script CLI `scripts/build_parcel_features.py` (Typer); y
- los assets Dagster `parcel_features_fused`, `parcel_splits_spatial_kfold`,
  `parcel_features_scaler` definidos en
  [`dagster_project/assets/features.py`](../dagster_project/assets/features.py).

DRY enforced — ver `docs/us-planning/us-016.md` §2.1.

## Atribución

Spyndex (MIT) — Montero, D., Aybar, C., Mahecha, M.D. et al. (2023).
*A standardized catalogue of spectral indices to advance the use of
remote sensing in Earth system research*. Scientific Data 10, 197.
DOI [10.1038/s41597-023-02096-0](https://doi.org/10.1038/s41597-023-02096-0).
Repositorio: [awesome-spectral-indices](https://github.com/awesome-spectral-indices/awesome-spectral-indices).

## Feature selection findings (US-018, Avance 2)

Tras ejecutar `notebooks/feature_engineering/03b_fe_spectral_temporal_pastis.ipynb`
sobre el subset PASTIS-R estratificado (77 muestras / 17 clases / 187 columnas):

- **Cluster espectral redundante confirmado**: el filtro
  `drop_correlated_features(|r|>0.95, Pearson)` reduce el cluster
  `{NDVI, NDRE, NDWI, SAVI}` retenido por US-014 a un representante por grupo,
  validando la hipotesis empirica del Avance 1 (US-013).
- **Senial temporal dominante**: el ranking ANOVA F y la importance RF/XGB
  exploratorios muestran que los componentes `*_fft_amp_*` y stats de indices
  vegetativos (NDWI_p05, GCVI_p95, MSAVI2_min) ocupan los primeros lugares,
  por encima de stats puntuales por banda.
- **PCA 0.95 con compresion fuerte**: la varianza explicada acumulada alcanza
  95 % con un numero de componentes muy inferior al total (`pca_scree.png`),
  evidencia de subespacio espectral compartido entre indices.
- **Normalizacion por familia de modelo**: `select_normalizer` rutea
  StandardScaler para modelos lineales, MinMaxScaler para NN, Yeo-Johnson
  para features sesgadas (acepta NDVI negativo en agua/sombras, D10) y
  log1p para LAI/biomasa. Decisiones por feature en
  `reports/feature_selection/normalization_decisions.csv`.
- **Tabla comparativa antes/despues**: el RF baseline con folds PASTIS 1-5
  (split espacial oficial Sainte-Fare-Garnot 2021, D1) cuantifica el costo en
  F1-macro/mIoU de cada estrategia de seleccion. Reporte en
  `reports/feature_selection/before_after.{csv,md}`.

Artefactos visuales commiteados: `reports/feature_selection/correlation_matrix.png`
y `reports/feature_selection/umap_2d.png`. El resto se regenera con
`make feature-selection-notebook`.
