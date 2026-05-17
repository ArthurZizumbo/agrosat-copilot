"""Test de integracion cross-US: US-014 (indices) -> US-015 (features temporales).

Verifica que la salida de :func:`ml.features.spectral_indices.compute_index`
(US-014) puede componerse en un ``xarray.DataArray`` con dimension ``band`` y
alimentarse directamente a
:func:`ml.features.temporal_features.extract_temporal_features` (US-015) sin
adaptadores intermedios. Este es el contrato implicito que el pipeline de
Feature Engineering del Avance 2 asume (plan US-015 §13).

Estrategia
----------

1. Construir un ``xr.DataArray`` Sentinel-2-like ``(time=10, band=10)`` con las
   bandas canonicas :data:`ml.ingest.pastis_loader.PASTIS_S2_BANDS` y attrs
   ``parcel_id=42, year=2024``. Los valores son reflectancias sinteticas
   deterministas (rng seed 42) en [0, 1].
2. Calcular NDVI, NDWI y EVI via ``compute_index`` (3 de los 17 indices reales).
3. Para los 14 indices restantes en :data:`DEFAULT_INDICES`, generar ruido
   sintetico determinista (mismo rng) — son rellenos para que
   ``extract_temporal_features`` no falte de bandas. Su valor numerico no se
   audita aqui (eso ya lo cubre ``test_spectral_indices.py``).
4. Apilar todo en un solo ``xr.DataArray`` con dim ``band`` ordenada segun
   ``DEFAULT_INDICES`` y attrs preservados.
5. Llamar ``extract_temporal_features`` y validar que el DataFrame resultante
   tiene exactamente 1 fila y que las columnas escalares fenologicas existen.

Notas
-----

- No se aserta sobre valores numericos concretos de NDVI/NDWI/EVI; los unit
  tests de cada modulo ya lo hacen. Aqui solo validamos el contrato de tipos
  y forma entre ambas etapas.
- No se llama a Vertex AI, vLLM, GEE, Postgres ni AlphaEarth.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import xarray as xr

from ml.features.spectral_indices import compute_index
from ml.features.temporal_features import DEFAULT_INDICES, extract_temporal_features
from ml.ingest.pastis_loader import PASTIS_S2_BANDS


def _make_synthetic_s2(
    *,
    n_steps: int = 10,
    year: int = 2024,
    parcel_id: int = 42,
    seed: int = 42,
) -> xr.DataArray:
    """Genera un DataArray Sentinel-2 sintetico ``(time, band)`` en [0, 1]."""
    rng = np.random.default_rng(seed)
    start = np.datetime64(f"{year}-03-01", "ns")
    times = np.array(
        [start + np.timedelta64(i * 30, "D") for i in range(n_steps)],
        dtype="datetime64[ns]",
    )
    bands = list(PASTIS_S2_BANDS)
    # Reflectancias sinteticas: base por banda + componente temporal suave.
    base = rng.uniform(0.05, 0.4, size=len(bands))
    season = 0.2 * np.sin(np.linspace(0, np.pi, n_steps))
    values = (
        base[None, :] + season[:, None] + rng.normal(0.0, 0.01, size=(n_steps, len(bands)))
    ).clip(0.01, 0.99)

    return xr.DataArray(
        data=values,
        dims=("time", "band"),
        coords={"time": times, "band": bands},
        attrs={"parcel_id": parcel_id, "year": year},
    )


def test_compute_indices_then_extract_temporal_features() -> None:
    """Pipeline US-014 -> US-015: indices reales + relleno -> features temporales."""
    s2_da = _make_synthetic_s2()
    n_steps = s2_da.sizes["time"]

    # 1. Indices reales via compute_index (US-014).
    real_indices = {name: compute_index(s2_da, name) for name in ("NDVI", "NDWI", "EVI")}

    # 2. Relleno determinista para el resto (mismo rng -> reproducible).
    rng = np.random.default_rng(42)
    fill = {
        name: xr.DataArray(
            data=rng.uniform(0.0, 1.0, size=(n_steps,)).astype(np.float64),
            dims=("time",),
            coords={"time": s2_da.coords["time"]},
        )
        for name in DEFAULT_INDICES
        if name not in real_indices
    }

    # 3. Apilar segun DEFAULT_INDICES.
    by_name = {**real_indices, **fill}
    stacked = np.stack(
        [np.asarray(by_name[name].values, dtype=np.float64) for name in DEFAULT_INDICES],
        axis=-1,
    )
    composed = xr.DataArray(
        data=stacked,
        dims=("time", "band"),
        coords={"time": s2_da.coords["time"], "band": list(DEFAULT_INDICES)},
        attrs=dict(s2_da.attrs),
    )

    # 4. Extraer features temporales (US-015).
    df = extract_temporal_features(composed)

    # 5. Asserts de contrato (no de valores).
    assert isinstance(df, pl.DataFrame)
    assert df.shape[0] == 1
    expected_scalar_cols = {
        "parcel_id",
        "year",
        "sog_doy",
        "peak_doy",
        "peak_value",
        "senescence_doy",
        "ndvi_auc",
        "ndvi_slope_pre_peak",
        "ndvi_slope_post_peak",
        "maturity_duration_days",
    }
    missing = expected_scalar_cols - set(df.columns)
    assert not missing, f"Faltan columnas fenologicas escalares: {missing}"
    assert df["parcel_id"].to_list() == [42]
    assert df["year"].to_list() == [2024]
