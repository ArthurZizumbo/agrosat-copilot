"""Mapa espacial folium (ROIs Italia + tiles PASTIS-R Francia).

Aisla la unica dependencia opcional pesada del dashboard (folium / geopandas).
Si no estan instaladas, las funciones degradan con un mensaje en vez de
romper el import del resto del paquete. El mapa NO se cachea: cachear el
objeto ``folium.Map`` provoca colisiones de IDs en el DOM al re-renderizar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from app.dashboard.loaders import load_yaml

try:  # pragma: no cover - import guard
    import folium
    from streamlit_folium import st_folium

    _HAS_FOLIUM = True
except ImportError:  # pragma: no cover - import guard
    folium = None  # type: ignore[assignment]
    st_folium = None  # type: ignore[assignment]
    _HAS_FOLIUM = False

try:  # pragma: no cover - import guard
    import geopandas as gpd

    _HAS_GEOPANDAS = True
except ImportError:  # pragma: no cover - import guard
    gpd = None  # type: ignore[assignment]
    _HAS_GEOPANDAS = False

# Cap defensivo de tiles en el mapa folium (top-N por area dissolved).
_FOLIUM_TOP_TILES = 10


def build_folium_map(rois: list[dict[str, Any]], pastis_geojson_path: str | None) -> Any:
    """Construye el mapa folium con ROIs italianas + tiles PASTIS dissolved.

    Args:
        rois: Lista de ROIs cargadas desde ``config/rois.yaml``.
        pastis_geojson_path: Ruta al ``metadata.geojson`` dissolved o ``None``.

    Returns:
        Objeto ``folium.Map`` o ``None`` si folium no esta instalado.
    """
    if not _HAS_FOLIUM or folium is None:  # pragma: no cover - import guard
        return None

    fmap = folium.Map(location=[44.5, 11.0], zoom_start=5, tiles="OpenStreetMap")
    italy_fg = folium.FeatureGroup(name="ROIs Italia", show=True)
    france_fg = folium.FeatureGroup(name="PASTIS-R tiles (Francia)", show=True)

    for roi in rois:
        name = str(roi.get("name", "roi"))
        region = str(roi.get("region", ""))
        bbox = roi.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        west, south, east, north = bbox
        bounds = [[south, west], [north, east]]
        color = "#2563EB" if region == "Italy" else "#F97316"
        target = italy_fg if region == "Italy" else france_fg
        folium.Rectangle(
            bounds=bounds,
            color=color,
            weight=2,
            fill=True,
            fill_opacity=0.15,
            popup=f"{name} ({region})",
            tooltip=name,
        ).add_to(target)

    if pastis_geojson_path and _HAS_GEOPANDAS and gpd is not None:
        _add_pastis_tiles(france_fg, Path(pastis_geojson_path))

    italy_fg.add_to(fmap)
    france_fg.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def _add_pastis_tiles(feature_group: Any, path: Path) -> None:
    """Agrega los tiles PASTIS-R dissolved al feature group de Francia."""
    if not path.exists():
        return
    try:
        gdf = gpd.read_file(path)  # type: ignore[union-attr]
        tile_col = next((c for c in ("TILE", "tile", "Tile") if c in gdf.columns), None)
        dissolved = (
            gdf.dissolve(by=tile_col).head(_FOLIUM_TOP_TILES)
            if tile_col is not None
            else gdf.head(_FOLIUM_TOP_TILES)
        )
        if dissolved.crs is not None and dissolved.crs.to_epsg() != 4326:
            dissolved = dissolved.to_crs(epsg=4326)
        folium.GeoJson(  # type: ignore[union-attr]
            dissolved.__geo_interface__,
            name="PASTIS tiles dissolved",
            style_function=lambda _f: {
                "color": "#F97316",
                "weight": 1.5,
                "fillOpacity": 0.25,
            },
        ).add_to(feature_group)
    except (OSError, ValueError, RuntimeError):
        return


def render_spatial_tab(rois_yaml: Path, pastis_metadata: Path) -> None:
    """Renderiza el tab espacial con ROIs Italia + PASTIS-R Francia.

    Args:
        rois_yaml: Ruta a ``config/rois.yaml``.
        pastis_metadata: Ruta al GeoJSON dissolved de PASTIS-R (o full).
    """
    st.markdown(
        '<h2 style="margin-top:0.5rem;color:#1E293B;font-weight:700;">'
        "Mapa espacial - ROIs Italia y PASTIS-R Francia</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#475569;font-size:1rem;line-height:1.6;">'
        "Mapa interactivo con las tres regiones italianas (Pianura "
        "Padana, Toscana central y Apulia) en azul y los patches del "
        "dataset PASTIS-R en Francia en ambar (agregados por TILE). "
        "Usa el control de capas en la esquina superior derecha del "
        "mapa para alternar visibilidad.</p>",
        unsafe_allow_html=True,
    )

    if not _HAS_FOLIUM:
        st.info(
            "streamlit-folium no esta instalado. Ejecuta "
            "`poetry install --with paper` para habilitar este tab."
        )
        return

    if not rois_yaml.exists():
        st.warning(f"No existe `{rois_yaml}` - el mapa se renderiza vacio.")
        rois_data: list[dict[str, Any]] = []
    else:
        rois_data = list(load_yaml(rois_yaml).get("rois", []))

    pastis_path = str(pastis_metadata) if pastis_metadata.exists() else None
    if pastis_path is None:
        st.caption("PASTIS-R metadata no disponible - se muestran solo las regiones italianas.")

    fmap = build_folium_map(rois_data, pastis_path)
    if fmap is None:
        st.info("Mapa no disponible - folium no inicializo.")
        return

    # ``key`` unico por render evita colisiones de IDs en el DOM cuando
    # Streamlit re-renderiza el componente.
    st_folium(fmap, width=None, height=560, returned_objects=[], key="eda_spatial_map")
