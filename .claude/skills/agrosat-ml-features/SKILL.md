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

## 17 Índices Espectrales

| Índice | Fórmula resumida | Uso agronómico |
|--------|------------------|----------------|
| NDVI   | (NIR - R) / (NIR + R) | Vigor vegetativo |
| NDWI   | (G - NIR) / (G + NIR) | Contenido de agua en hoja |
| NDMI   | (NIR - SWIR1) / (NIR + SWIR1) | Humedad canopy |
| EVI    | 2.5*(NIR-R)/(NIR+6R-7.5B+1) | Vigor en canopy denso |
| SAVI   | ((NIR-R)/(NIR+R+L))*(1+L), L=0.5 | Vigor con suelo desnudo |
| MSAVI2 | versión auto-calibrada SAVI | Idem mejorado |
| NBR    | (NIR - SWIR2) / (NIR + SWIR2) | Estrés / fuego |
| MCARI  | ((RE - R) - 0.2(RE - G)) * (RE/R) | Clorofila |
| CCCI   | normalizado MCARI/NDVI | Clorofila + canopy |
| LAI    | leaf area index empírico | Densidad foliar |
| FAPAR  | radiación absorbida | Fotosíntesis |
| PSRI   | (R - G) / RE | Senescencia |
| NDCI   | (RE - R) / (RE + R) | Clorofila acuática |
| GCVI   | NIR/G - 1 | Clorofila verde |
| RENDVI | (NIR - RE) / (NIR + RE) | Red-edge NDVI |
| NDRE   | idem RENDVI alt | Cultivos densos |
| TSAVI  | a*(NIR-a*R-b)/(R+a*(NIR-b)) | SAVI calibrado |

## Implementación con spyndex

```python
import xarray as xr
import spyndex

def compute_index(da: xr.DataArray, index: str) -> xr.DataArray:
    """Computa índice espectral sobre xarray.DataArray.

    Args:
        da: DataArray con dimensión 'band' (B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12).
        index: nombre del índice (NDVI, NDWI, EVI, etc.).

    Returns:
        DataArray con índice computado, conserva tiempo y espacio.
    """
    params = {
        "B": da.sel(band="B02"),
        "G": da.sel(band="B03"),
        "R": da.sel(band="B04"),
        "RE1": da.sel(band="B05"),
        "RE2": da.sel(band="B06"),
        "RE3": da.sel(band="B07"),
        "N": da.sel(band="B08"),
        "S1": da.sel(band="B11"),
        "S2": da.sel(band="B12"),
        "L": 0.5,
    }
    return spyndex.computeIndex(index=index, params=params)
```

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
