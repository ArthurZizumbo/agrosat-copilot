"""Backend test fixtures.

PROJ FIX (US-055 / Riesgo R1): on the Windows dev box a PostGIS install exports
``PROJ_LIB`` pointing at an ancient ``proj.db`` that breaks ``CRS.from_epsg`` and
the import of ``rio_tiler`` / ``titiler.core``. ``ensure_proj_data`` pins
``PROJ_DATA`` to rasterio's bundled DB *before* any rasterio import. This runs at
collection time so every test that touches CRS sees the fix.
"""

from __future__ import annotations

# Must run before rasterio / rio-tiler / titiler load their C extension.
from backend.app.services.proj_env import ensure_proj_data

ensure_proj_data()

from collections.abc import Iterator  # noqa: E402
from pathlib import Path  # noqa: E402

import polars as pl  # noqa: E402
import pytest  # noqa: E402

#: Real farslip corpus (Sentinel-2 pixels) + manifest with real lat/lon.
_FARSLIP_DIR = Path(__file__).resolve().parents[2] / "data" / "farslip_pairs" / "pianura_padana"
_MANIFEST = _FARSLIP_DIR / "manifest.parquet"


def _build_real_cog(dest_dir: Path) -> Path:
    """Build a valid georeferenced COG from REAL farslip S2 pixels + REAL lat/lon.

    The pixels come untouched from a farslip crop (4-band uint16 256x256
    Sentinel-2). Only a CRS + transform centred on the crop's real lat/lon (from
    the manifest) is assigned -- a legitimate georeferencing operation, not data
    fabrication. There is no ``np.random`` anywhere.

    Args:
        dest_dir: Directory to materialise the intermediate raster and COG into.

    Returns:
        Path to a ``cog_validate``-valid COG (tiled + overviews).
    """
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import transform as warp_transform
    from rio_cogeo.cogeo import cog_translate
    from rio_cogeo.profiles import cog_profiles

    manifest = pl.read_parquet(_MANIFEST)
    row = manifest.row(0, named=True)  # real crop + its real lat/lon.
    with rasterio.open(row["crop_path"]) as src:
        data = src.read()  # (4, 256, 256) REAL pixels, untouched.

    lon, lat = float(row["lon"]), float(row["lat"])
    cx, cy = warp_transform("EPSG:4326", "EPSG:3857", [lon], [lat])
    res = 10.0  # Sentinel-2 = 10 m/px.
    transform = from_origin(cx[0] - 128 * res, cy[0] + 128 * res, res, res)

    raw = dest_dir / "real_s2.tif"
    profile = {
        "driver": "GTiff",
        "height": 256,
        "width": 256,
        "count": 4,
        "dtype": "uint16",
        "crs": "EPSG:3857",
        "transform": transform,
    }
    with rasterio.open(raw, "w", **profile) as dst:
        dst.write(data)  # same real pixels.

    out = dest_dir / "real_s2_cog.tif"
    cog_translate(raw, out, cog_profiles.get("deflate"), web_optimized=True, quiet=True)
    return out


@pytest.fixture(scope="session")
def real_cog(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Session-scoped real COG fixture (skips cleanly if the corpus is absent).

    Args:
        tmp_path_factory: pytest temp-dir factory.

    Returns:
        Path to the built COG.
    """
    if not _MANIFEST.exists():
        pytest.skip(f"farslip manifest not found at {_MANIFEST}")
    dest = tmp_path_factory.mktemp("cog")
    return _build_real_cog(dest)


@pytest.fixture()
def fake_redis() -> Iterator[object]:
    """Yield a ``fakeredis`` async client for the tile cache (no network).

    Yields:
        A ``fakeredis.aioredis.FakeRedis`` instance.
    """
    import fakeredis.aioredis

    client = fakeredis.aioredis.FakeRedis()
    yield client
