"""Configuracion pytest comun para los tests de la capa ML.

Fuerza el backend `Agg` de matplotlib antes de que cualquier modulo importe
pyplot, para evitar errores `_tkinter.TclError` en Windows cuando varios tests
graficos corren en la misma sesion (cross-test pollution con Tk).

Adiciona fixtures compartidos US-016:

- ``parcels_fixture_3regions``: GeoDataFrame de las 9 parcelas sinteticas
  (3 regiones italianas, 7 cultivos distintos) cargadas desde
  ``data/test_fixtures/parcels_demo_3regions.parquet``.
- ``synthetic_alphaearth_64d``: ``pl.DataFrame`` con los 64 dims AlphaEarth
  rellenos con valores deterministicos por parcela (seed fija). Las dims
  se llaman ``ae_00..ae_63`` para encajar con el contrato de
  ``ml.features.fusion._build_ae_block``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import polars as pl
import pytest

if TYPE_CHECKING:  # pragma: no cover - solo tipado
    import geopandas as gpd

matplotlib.use("Agg")


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PARCELS_FIXTURE_PATH = _REPO_ROOT / "data" / "test_fixtures" / "parcels_demo_3regions.parquet"


@pytest.fixture
def parcels_fixture_3regions() -> "gpd.GeoDataFrame":
    """Carga el fixture de 9 parcelas sinteticas (3 regiones italianas).

    Devuelve un GeoDataFrame con ``parcel_id``, ``year``, ``geometry``
    (POLYGON EPSG:4326), ``crop_class`` y ``region``.
    """
    import geopandas as gpd
    from shapely import wkt

    if not _PARCELS_FIXTURE_PATH.exists():
        pytest.skip(
            f"Fixture demo no encontrado: {_PARCELS_FIXTURE_PATH}. "
            "Genera primero el parquet sintetico US-016."
        )
    raw = pl.read_parquet(_PARCELS_FIXTURE_PATH).to_pandas()
    geoms = [wkt.loads(g) for g in raw["geom"].tolist()]
    gdf = gpd.GeoDataFrame(
        raw.drop(columns=["geom"]),
        geometry=geoms,
        crs="EPSG:4326",
    )
    return gdf


@pytest.fixture
def synthetic_alphaearth_64d(parcels_fixture_3regions: "gpd.GeoDataFrame") -> pl.DataFrame:
    """``pl.DataFrame`` con 64 dims AlphaEarth deterministas por parcela.

    Valor de la dim ``j`` para la parcela ``i`` = ``round((i * 0.13 + j * 0.011),
    6)`` — finito, sin NaN/Inf, reproducible para asserts en tests.
    Columnas: ``parcel_id`` (Int64), ``year`` (Int16), ``ae_00..ae_63`` (Float32).
    """
    n = len(parcels_fixture_3regions)
    pids = parcels_fixture_3regions["parcel_id"].astype("int64").tolist()
    year_val = int(parcels_fixture_3regions["year"].iloc[0])
    cols: dict[str, object] = {
        "parcel_id": pl.Series("parcel_id", pids, dtype=pl.Int64),
        "year": pl.Series("year", [year_val] * n, dtype=pl.Int16),
    }
    for j in range(64):
        col_name = f"ae_{j:02d}"
        values = [round((i + 1) * 0.13 + j * 0.011, 6) for i in range(n)]
        cols[col_name] = pl.Series(col_name, values, dtype=pl.Float32)
    return pl.DataFrame(cols)
