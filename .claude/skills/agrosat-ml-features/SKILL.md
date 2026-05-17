---
name: agrosat-ml-features
description: Compute spectral indices (NDVI, NDWI, EVI, MSAVI2, MCARI, CCCI, NDRE, etc.), temporal features (FFT harmonics, phenology), and multisensor fusion at parcel level using Polars 1.x for AgroSatCopilot. Use for Feature Engineering (EPIC 3).
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# AgroSatCopilot ML Features Skill

## Rules — NON-NEGOTIABLE

- Polars 1.x para DataFrames; jamás pandas salvo conversión final
- Índices con `eemont` (GEE) o `spyndex` (local xarray)
- Cache en Redis con key `{scene_id}:{index_name}` para evitar recómputo
- Tests con valores conocidos: NDVI bosque caducifolio jun ∈ [0.7, 0.9]
- Polars LazyFrame para datasets grandes (>5 GB)
- Outputs como Parquet versionado con DVC

## 17 Índices Espectrales — implementados en `ml/features/spectral_indices.py`

Catálogo canónico (US-014). Tabla académica completa con DOIs en
[`docs/spectral_indices.md`](../../../docs/spectral_indices.md).

**Backends** (verificado con spyndex 0.10.0):
- 11 índices delegan literal a spyndex.
- 3 alias documentados: `MSAVI2 → MSAVI`, `NDRE → NDREI`, `GCVI → CIG`.
- 3 fórmulas custom auditadas con DOI: `LAI` Boegh 2002, `FAPAR` Myneni 1997, `CCCI` Barnes 2000.

| Índice | Fórmula resumida | Backend | Uso agronómico |
|--------|------------------|---------|----------------|
| NDVI   | (N - R) / (N + R) | spyndex `NDVI` | Vigor vegetativo |
| NDWI   | (G - N) / (G + N) | spyndex `NDWI` | Contenido de agua en hoja |
| NDMI   | (N - S1) / (N + S1) | spyndex `NDMI` | Humedad canopy |
| EVI    | g·(N-R)/(N+C1·R-C2·B+L) | spyndex `EVI` | Vigor en canopy denso |
| SAVI   | ((N-R)/(N+R+L))·(1+L), L=0.5 | spyndex `SAVI` | Vigor con suelo desnudo |
| MSAVI2 | 0.5·(2N+1 - √((2N+1)² - 8(N-R))) | spyndex `MSAVI` (alias) | SAVI auto-calibrado |
| NBR    | (N - S2) / (N + S2) | spyndex `NBR` | Estrés / fuego |
| MCARI  | ((RE1-R)-0.2·(RE1-G))·(RE1/R) | spyndex `MCARI` | Clorofila |
| CCCI   | NDRE / NDVI | **custom** Barnes 2000 | Clorofila + canopy |
| LAI    | -ln(1-(NDVI-0.05)/0.95)/0.5 | **custom** Boegh 2002 | Densidad foliar |
| FAPAR  | 1.24·NDVI - 0.168 | **custom** Myneni 1997 | Radiación absorbida |
| PSRI   | (R - B) / RE2 | spyndex `PSRI` | Senescencia |
| NDCI   | (RE1 - R) / (RE1 + R) | spyndex `NDCI` | Clorofila acuática |
| GCVI   | N/G - 1 | spyndex `CIG` (alias) | Clorofila verde |
| RENDVI | (RE2 - RE1) / (RE2 + RE1) | spyndex `RENDVI` | Red-edge NDVI |
| NDRE   | (N - RE1) / (N + RE1) | spyndex `NDREI` (alias) | Cultivos densos |
| TSAVI  | sla·(N-sla·R-slb)/(sla·N+R-sla·slb) | spyndex `TSAVI` | SAVI calibrado |

## API canónica (importar desde `ml.features`)

```python
from ml.features import (
    INDEX_NAMES,                # lista canónica de los 17 índices
    compute_index,              # API principal sobre xr.DataArray
    compute_index_timeseries,   # con eje time + reduce opcional
    compute_index_cached,       # cache Redis opcional (best-effort)
    compute_index_ee,           # wrapper server-side GEE via eemont
)
import xarray as xr
from ml.ingest.pastis_loader import PASTIS_S2_BANDS

# Pre-requisitos (responsabilidad del caller):
# 1. Reflectancia escalada [0, 1] (dividir DN /10000)
# 2. Máscara SCL aplicada (sin nubes/sombras/nieve)
# 3. Dimensión 'band' con labels PASTIS_S2_BANDS

da: xr.DataArray  # dims (band, y, x) o (time, band, y, x)
ndvi = compute_index(da, "NDVI")
lai_max = compute_index_timeseries(da, "LAI", reduce="p95")
```

El mapeo `PASTIS_S2_BANDS → spyndex` (B/G/R/RE1.../S1/S2) y la elección
spyndex vs custom-formula viven internamente en `_BAND_TO_SPYNDEX` y
`_INDEX_REGISTRY`. No reimplementar fuera del módulo.

## Features Temporales con Polars

```python
import polars as pl
import numpy as np

def extract_temporal_features(ts: pl.DataFrame) -> pl.DataFrame:
    """ts schema: parcel_id, date, ndvi, ndwi, ...

    Returns features por (parcel_id, year): media, std, percentiles, FFT, fenología.
    """
    return (
        ts.group_by(["parcel_id", pl.col("date").dt.year().alias("year")])
        .agg([
            pl.col("ndvi").mean().alias("ndvi_mean"),
            pl.col("ndvi").std().alias("ndvi_std"),
            pl.col("ndvi").quantile(0.05).alias("ndvi_p05"),
            pl.col("ndvi").quantile(0.95).alias("ndvi_p95"),
            pl.col("ndvi").max().alias("ndvi_peak"),
            # Fecha del pico
            pl.col("date").get(pl.col("ndvi").arg_max()).alias("ndvi_peak_date"),
            # AUC integral
            pl.col("ndvi").sum().alias("ndvi_auc"),
        ])
    )
```

## FFT Harmonic Decomposition

```python
def fft_harmonics(ndvi_series: np.ndarray, n_harmonics: int = 3) -> dict:
    """Devuelve amplitudes y fases de las primeras n_harmonics componentes."""
    fft = np.fft.rfft(ndvi_series)
    amps = np.abs(fft)[:n_harmonics + 1]
    phases = np.angle(fft)[:n_harmonics + 1]
    return {f"amp_{i}": amps[i] for i in range(n_harmonics + 1)} | \
           {f"phase_{i}": phases[i] for i in range(n_harmonics + 1)}
```

## Phenology Detection

```python
def detect_phenology(ndvi_series: pl.Series, dates: pl.Series) -> dict:
    """Detecta start of greenness (SOG), peak, senescence."""
    # SOG: primer día ascendente cruzando 0.3
    above = (ndvi_series > 0.3).to_numpy()
    if not above.any():
        return {"sog": None, "peak_date": None, "senescence": None}
    sog_idx = np.argmax(above)
    peak_idx = ndvi_series.arg_max()
    # Senescence: pos-peak descendente cruzando 0.3
    post_peak = ndvi_series[peak_idx:].to_numpy()
    sen_idx = peak_idx + np.argmax(post_peak < 0.3) if (post_peak < 0.3).any() else None
    return {
        "sog": dates[sog_idx],
        "peak_date": dates[peak_idx],
        "senescence": dates[sen_idx] if sen_idx else None,
    }
```

## Fusión Multisensor

```python
def fuse_parcel_features(
    alphaearth: pl.DataFrame,  # parcel_id, year, embedding_64dims
    spectral: pl.DataFrame,    # parcel_id, year, ndvi_*, ndwi_*, ...
    dinov3: pl.DataFrame,      # parcel_id, year, dinov3_1024
    gsaa: pl.DataFrame,        # parcel_id, crop_class_label
) -> pl.DataFrame:
    return (
        alphaearth
        .join(spectral, on=["parcel_id", "year"], how="left")
        .join(dinov3, on=["parcel_id", "year"], how="left")
        .join(gsaa, on="parcel_id", how="left")
    )
```

## QA Checklist

- [ ] Polars 1.x exclusivo (DataFrame / LazyFrame)
- [ ] Cache Redis para índices recurrentes
- [ ] Tests con valores conocidos por bioma/estación
- [ ] Outputs en Parquet versionado con DVC
- [ ] Documentación agronómica en docstrings
