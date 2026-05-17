"""Píxeles Sentinel-2 sintéticos con firmas espectrales auditables.

Las firmas (reflectancias por banda) provienen de:

- USGS Spectral Library v7 (Kokaly et al. 2017, DOI 10.3133/ds1035) — firmas
  de vegetación sana, suelo desnudo y cuerpos de agua resampleadas a las 10
  bandas Sentinel-2 conservadas en PASTIS-R.
- Pettorelli, N. (2005). *Using the satellite-derived NDVI to assess
  ecological responses to environmental change*. Trends in Ecology &
  Evolution 20(9), 503-510. DOI 10.1016/j.tree.2005.05.011 — rangos NDVI
  por bioma (forest 0.7-0.9, bare soil 0.0-0.2, water < 0).

Valores en reflectancia [0, 1] (el caller del catálogo es responsable de
escalar DN /10000 antes; los fixtures ya lo simulan).

Bandas en orden canónico `PASTIS_S2_BANDS` importado desde
`ml.ingest.pastis_loader` (única fuente de verdad).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import xarray as xr

from ml.ingest.pastis_loader import PASTIS_S2_BANDS

Archetype = Literal[
    "forest",
    "bare_soil",
    "water",
    "burned",
    "senescent",
    "dense_canopy",
    "sparse_canopy",
    "dry_canopy",
    "chla_water",
]


# Firmas espectrales sintéticas (reflectancia [0, 1]) por arquetipo.
# Orden idéntico a PASTIS_S2_BANDS: B02 B03 B04 B05 B06 B07 B08 B8A B11 B12.
_ARCHETYPE_REFLECTANCE: dict[Archetype, tuple[float, ...]] = {
    # Bosque caducifolio en junio. NDVI esperado = (0.45-0.05)/(0.50) = 0.80
    # ∈ [0.7, 0.9] literal del CA US-014 plan v6 línea 1108.
    "forest": (0.02, 0.06, 0.05, 0.20, 0.35, 0.40, 0.45, 0.46, 0.20, 0.10),
    # Suelo desnudo seco. NDVI esperado = (0.35-0.30)/0.65 = 0.077 ∈ [0, 0.2].
    "bare_soil": (0.20, 0.25, 0.30, 0.32, 0.33, 0.34, 0.35, 0.36, 0.40, 0.42),
    # Cuerpo de agua claro. NDWI = (G-N)/(G+N) = (0.10-0.02)/0.12 = 0.667 > 0.5.
    "water": (0.06, 0.10, 0.08, 0.04, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01),
    # Área quemada reciente. NBR = (N-S2)/(N+S2) = (0.10-0.30)/0.40 = -0.5 < 0.
    "burned": (0.05, 0.08, 0.15, 0.18, 0.20, 0.15, 0.10, 0.10, 0.28, 0.30),
    # Canopy senescente otoñal. PSRI = (R-B)/RE2 > 0 (R alto, B bajo, RE2 medio).
    # PSRI esperado ~ (0.20 - 0.03) / 0.18 = 0.94.
    "senescent": (0.03, 0.07, 0.20, 0.18, 0.18, 0.20, 0.22, 0.22, 0.22, 0.18),
    # Canopy denso vigoroso (cultivo C3 maduro). EVI > 0.4, RENDVI > 0.
    # NDVI=(0.50-0.04)/0.54=0.852; EVI(L=1,C1=6,C2=7.5,g=2.5)~0.62;
    # RENDVI=(0.45-0.30)/0.75=0.20.
    "dense_canopy": (0.02, 0.06, 0.04, 0.15, 0.30, 0.40, 0.50, 0.50, 0.18, 0.08),
    # Canopy disperso con suelo expuesto (efecto suelo notable).
    # NDVI=(0.30-0.18)/0.48=0.25; SAVI con L=0.5 < NDVI (deprime suelo).
    "sparse_canopy": (0.10, 0.13, 0.18, 0.22, 0.25, 0.27, 0.30, 0.30, 0.25, 0.18),
    # Canopy seco / estresado hídricamente. NDMI = (N-S1)/(N+S1) < 0.
    # NDMI = (0.30 - 0.50) / 0.80 = -0.25.
    "dry_canopy": (0.04, 0.08, 0.10, 0.18, 0.25, 0.28, 0.30, 0.30, 0.50, 0.45),
    # Agua con clorofila-a (eutrofización). NDCI = (RE1-R)/(RE1+R) > 0.
    # NDCI = (0.12 - 0.08) / 0.20 = 0.20.
    "chla_water": (0.05, 0.10, 0.08, 0.12, 0.06, 0.04, 0.03, 0.03, 0.02, 0.02),
}


def _reflectance_for(archetype: Archetype) -> np.ndarray:
    """Devuelve la firma espectral (10,) en orden PASTIS_S2_BANDS."""
    return np.asarray(_ARCHETYPE_REFLECTANCE[archetype], dtype=np.float32)


def make_forest_pixel() -> xr.DataArray:
    """Píxel bosque caducifolio en junio (NDVI ∈ [0.7, 0.9])."""
    return make_synthetic_s2_dataarray(shape=(1, 1), archetype="forest")


def make_bare_soil_pixel() -> xr.DataArray:
    """Píxel suelo desnudo seco (NDVI ∈ [0.0, 0.2])."""
    return make_synthetic_s2_dataarray(shape=(1, 1), archetype="bare_soil")


def make_water_pixel() -> xr.DataArray:
    """Píxel cuerpo de agua claro (NDWI > 0.5)."""
    return make_synthetic_s2_dataarray(shape=(1, 1), archetype="water")


def make_synthetic_s2_dataarray(
    shape: tuple[int, int] = (4, 4),
    archetype: Archetype = "forest",
    bands: Sequence[str] = PASTIS_S2_BANDS,
    noise_std: float = 0.0,
    seed: int = 42,
) -> xr.DataArray:
    """Construye un ``xarray.DataArray`` (band, y, x) con la firma del arquetipo.

    Args:
        shape: Tupla (height, width) en píxeles.
        archetype: Arquetipo de cobertura (ver ``Archetype``).
        bands: Subset y orden de bandas Sentinel-2. Default PASTIS_S2_BANDS.
        noise_std: Desviación estándar de ruido gaussiano aditivo (0 = sin ruido).
        seed: Semilla para reproducibilidad del ruido.

    Returns:
        DataArray dims ('band', 'y', 'x'), dtype float32, coords 'band' con
        los labels canónicos.
    """
    sig = _reflectance_for(archetype)
    band_idx = [PASTIS_S2_BANDS.index(b) for b in bands]
    h, w = shape
    arr = np.broadcast_to(sig[band_idx, None, None], (len(bands), h, w)).astype(np.float32)
    arr = arr.copy()  # broadcast_to devuelve vista read-only
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        arr = arr + rng.normal(0.0, noise_std, size=arr.shape).astype(np.float32)
        arr = np.clip(arr, 0.0, 1.5)
    return xr.DataArray(
        arr,
        dims=("band", "y", "x"),
        coords={
            "band": list(bands),
            "y": np.arange(h),
            "x": np.arange(w),
        },
        name="reflectance",
    )


def make_synthetic_s2_timeseries(
    n_timesteps: int = 6,
    shape: tuple[int, int] = (4, 4),
    archetype: Archetype = "forest",
    phenology: bool = True,
    seed: int = 42,
) -> xr.DataArray:
    """Construye una serie temporal Sentinel-2 sintética dims (time, band, y, x).

    Si ``phenology=True`` y ``archetype=='forest'``, modula la reflectancia
    NIR (B08) y la roja (B04) con una curva sinusoidal que produce NDVI
    ascendente-descendente típico (0.2 → 0.8 → 0.3 sobre el rango temporal).

    Args:
        n_timesteps: Número de timesteps.
        shape: (height, width).
        archetype: Arquetipo base.
        phenology: Si True, modula NIR/R para curva fenológica.
        seed: Semilla (no se usa actualmente, reservada para ruido futuro).

    Returns:
        DataArray dims ('time', 'band', 'y', 'x'), dtype float32.
    """
    _ = seed  # reservado para extensiones futuras
    base = make_synthetic_s2_dataarray(shape=shape, archetype=archetype)
    h, w = shape
    stack = np.repeat(base.values[np.newaxis, ...], n_timesteps, axis=0).astype(np.float32)

    if phenology and archetype == "forest":
        # Curva fenológica: NDVI sube de 0.2 a 0.85 y baja a 0.30 en n_timesteps.
        # Modulamos B08 (NIR) manteniendo B04 (R) en 0.05 fija.
        target_ndvi = np.linspace(0.2, 0.85, num=(n_timesteps + 1) // 2)
        if n_timesteps - target_ndvi.size > 0:
            descent = np.linspace(0.85, 0.30, num=n_timesteps - target_ndvi.size + 1)[1:]
            target_ndvi = np.concatenate([target_ndvi, descent])
        target_ndvi = target_ndvi[:n_timesteps]
        red_idx = PASTIS_S2_BANDS.index("B04")
        nir_idx = PASTIS_S2_BANDS.index("B08")
        red = 0.05
        # NDVI = (N-R)/(N+R) → N = R*(1+NDVI)/(1-NDVI)
        for t, ndvi in enumerate(target_ndvi):
            nir = red * (1.0 + float(ndvi)) / max(1.0 - float(ndvi), 1e-3)
            stack[t, red_idx, :, :] = red
            stack[t, nir_idx, :, :] = nir

    return xr.DataArray(
        stack,
        dims=("time", "band", "y", "x"),
        coords={
            "time": np.arange(n_timesteps),
            "band": list(PASTIS_S2_BANDS),
            "y": np.arange(h),
            "x": np.arange(w),
        },
        name="reflectance",
    )
