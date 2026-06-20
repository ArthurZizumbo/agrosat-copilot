"""PROJ data bootstrap -- MUST be imported before rasterio / rio-tiler / titiler.

On the Windows dev box an unrelated PostgreSQL/PostGIS install exports
``PROJ_LIB`` pointing at an ancient ``proj.db`` (``DATABASE.LAYOUT.VERSION.MINOR=2``).
GDAL's PROJ then refuses to resolve ``EPSG:3857`` and the import of
``titiler.core`` / ``rio_tiler`` fails with ``CRSError``. The env var has to be
fixed *before* the rasterio C-extension initialises GDAL, so this runs at module
import time and any module that needs CRS support imports it first.

On Linux (Cloud Run) ``PROJ_LIB``/``PROJ_DATA`` are unset, the bundled rasterio
``proj_data`` is already on the search path, and this is a harmless no-op that
just pins the correct directory.
"""

from __future__ import annotations

import os
import sysconfig

__all__ = ["BUNDLED_PROJ_DATA", "ensure_proj_data"]


def _bundled_proj_data() -> str | None:
    """Return the path to rasterio's bundled ``proj_data`` dir, if present.

    Computed via ``sysconfig`` (no ``import rasterio``) so PROJ can be pinned
    *before* the GDAL C-extension initialises.

    Returns:
        Absolute path to ``<site-packages>/rasterio/proj_data`` when it exists,
        otherwise ``None``.
    """
    purelib = sysconfig.get_paths().get("purelib")
    if not purelib:
        return None
    candidate = os.path.join(purelib, "rasterio", "proj_data")
    return candidate if os.path.isdir(candidate) else None


def ensure_proj_data() -> str | None:
    """Pin ``PROJ_DATA``/``PROJ_LIB`` to rasterio's bundled DB when needed.

    Idempotent. If ``PROJ_DATA`` already points at a directory containing a
    ``proj.db`` it is left untouched. Otherwise both ``PROJ_DATA`` and the legacy
    ``PROJ_LIB`` are overridden to the bundled directory (overriding any hijacked
    value such as a PostGIS install).

    Returns:
        The pinned ``proj_data`` path, or ``None`` if no bundled DB was found
        (the caller then relies on the platform default).
    """
    current = os.environ.get("PROJ_DATA")
    if current and os.path.isfile(os.path.join(current, "proj.db")):
        return current
    bundled = _bundled_proj_data()
    if bundled is None:
        return None
    os.environ["PROJ_DATA"] = bundled
    os.environ["PROJ_LIB"] = bundled
    return bundled


#: Pinned at import time so importing this module (before rasterio) fixes PROJ.
BUNDLED_PROJ_DATA: str | None = ensure_proj_data()
