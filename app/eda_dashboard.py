"""Dashboard Streamlit de EDA — Avance 1 · AgroSatCopilot (US-013).

Dashboard file-driven que consolida los resultados de US-010 (Sentinel-2
univariado), US-011 (AlphaEarth + cross-region) y US-012 (bivariado, temporal,
espacial) en 6 tabs navegables. No recomputa nada: lee figuras y tablas ya
exportadas por los notebooks a ``paper/figures/us-{010,011,012}/``.

Para arrancar localmente::

    poetry run streamlit run app/eda_dashboard.py --server.port 8501

El dashboard degrada de forma graceful cuando faltan figuras: cada tab muestra
``st.info`` con la US esperada en lugar de fallar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import streamlit as st
import yaml

# streamlit-folium es opcional: si no está instalado, la pestaña espacial
# muestra un mensaje en vez de crashear. Esto permite que `make eda-dashboard`
# funcione en entornos donde solo se quiere ver tabs 1-4 + 6.
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


# Rutas canónicas relativas a la raíz del repositorio.
REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_ROOT = REPO_ROOT / "paper" / "figures"
ROIS_YAML = REPO_ROOT / "config" / "rois.yaml"
PASTIS_METADATA = REPO_ROOT / "data" / "PASTIS-R" / "metadata.geojson"

# Cap defensivo de filas a renderizar en st.dataframe para no saturar al cliente.
_DATAFRAME_HEAD_ROWS = 200
# Cap defensivo de clases en el mapa folium (top-N por área dissolved).
_FOLIUM_TOP_TILES = 10


# ---------------------------------------------------------------------------
# Loaders cacheados
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pl.DataFrame:
    """Carga un CSV como ``polars.DataFrame`` con cache de Streamlit.

    Args:
        path: Ruta absoluta o relativa al CSV en disco.

    Returns:
        DataFrame de Polars. Si la lectura falla devuelve un DataFrame vacío
        (no propaga excepción para no romper el render del tab).
    """
    path = Path(path)
    if not path.exists():
        return pl.DataFrame()
    try:
        return pl.read_csv(path)
    except (pl.exceptions.ComputeError, OSError, ValueError):
        return pl.DataFrame()


@st.cache_data(show_spinner=False)
def load_yaml(path: Path) -> dict[str, Any]:
    """Carga un YAML como diccionario con cache de Streamlit.

    Args:
        path: Ruta al archivo YAML.

    Returns:
        Diccionario parseado. Diccionario vacío si el archivo no existe.
    """
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    return loaded if isinstance(loaded, dict) else {}


def _list_pngs(directory: Path, pattern: str = "*.png") -> list[Path]:
    """Lista PNGs ordenados alfabéticamente filtrando por glob."""
    if not directory.exists():
        return []
    return sorted(directory.glob(pattern))


def _list_csvs(directory: Path, pattern: str = "*.csv") -> list[Path]:
    """Lista CSVs ordenados alfabéticamente filtrando por glob."""
    if not directory.exists():
        return []
    return sorted(directory.glob(pattern))


def _render_pngs(pngs: list[Path], caption_fn: Any = None) -> None:
    """Renderiza una lista de PNGs con ``st.image`` y un caption derivado.

    Args:
        pngs: Lista de rutas a PNG.
        caption_fn: Función opcional ``Path -> str`` para generar caption.
    """
    for png in pngs:
        caption = caption_fn(png) if caption_fn else png.stem.replace("_", " ")
        st.image(str(png), caption=caption, use_container_width=True)


def _render_csvs(csvs: list[Path]) -> None:
    """Renderiza CSVs como ``st.dataframe`` con head capeado a 200 filas."""
    for csv_path in csvs:
        st.markdown(f"**Tabla**: `{csv_path.name}`")
        df = load_csv(csv_path)
        if df.is_empty():
            st.caption("Tabla vacía o ilegible.")
            continue
        st.dataframe(df.head(_DATAFRAME_HEAD_ROWS).to_pandas(), use_container_width=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


def render_tab_univariate(figures_dir: Path) -> None:
    """Renderiza el tab 1 — Univariado Sentinel-2 (US-010).

    Lee ``paper/figures/us-010/*.png`` y ``*.csv`` y los expone en columnas.
    Si el directorio no existe o está vacío, muestra placeholder ``st.info``.

    Args:
        figures_dir: Directorio que contiene las figuras de US-010.
    """
    st.header("Univariado · Sentinel-2 (US-010)")
    st.markdown(
        "Estadísticas por banda, máscara SCL, outliers y distribuciones empíricas "
        "del muestreo de Sentinel-2 L2A sobre las 3 ROIs italianas."
    )

    pngs = _list_pngs(figures_dir)
    csvs = _list_csvs(figures_dir)

    if not pngs and not csvs:
        st.info("Pendiente — esperando US-010 (figuras Sentinel-2 univariado).")
        return

    _render_pngs(pngs)

    if csvs:
        st.subheader("Tablas")
        _render_csvs(csvs)


def render_tab_alphaearth(figures_dir: Path) -> None:
    """Renderiza el tab 2 — AlphaEarth 64-dim + cross-region (US-011).

    Agrupa las figuras por sección (``sec1_*`` Italia, ``sec2_*`` Francia,
    ``sec3_*`` cross-region consistency).

    Args:
        figures_dir: Directorio con figuras ``sec{1,2,3}_*.png``.
    """
    st.header("AlphaEarth Foundations 64-dim + cross-region (US-011)")
    st.markdown(
        "Reducción dimensional (t-SNE / UMAP) sobre embeddings AlphaEarth, "
        "comparativa Italia (Dynamic World) vs Francia (PASTIS-R) y tabla de "
        "consistencia cross-region."
    )

    pngs_sec1 = _list_pngs(figures_dir, "sec1_*.png")
    pngs_sec2 = _list_pngs(figures_dir, "sec2_*.png")
    pngs_sec3 = _list_pngs(figures_dir, "sec3_*.png")
    csvs = _list_csvs(figures_dir)

    if not (pngs_sec1 or pngs_sec2 or pngs_sec3 or csvs):
        st.info("Pendiente — esperando US-011 (figuras AlphaEarth).")
        return

    if pngs_sec1:
        st.subheader("Sección 1 — Italia x Dynamic World")
        _render_pngs(pngs_sec1)
    if pngs_sec2:
        st.subheader("Sección 2 — Francia x PASTIS-R")
        _render_pngs(pngs_sec2)
    if pngs_sec3:
        st.subheader("Sección 3 — Cross-region consistency")
        _render_pngs(pngs_sec3)

    if csvs:
        st.subheader("Tablas")
        _render_csvs(csvs)


def render_tab_bivariate(figures_dir: Path) -> None:
    """Renderiza el tab 3 — Bivariado / Multivariado (US-012).

    Carga heatmaps de correlación + tabla VIF de multicolinealidad.

    Args:
        figures_dir: Directorio con ``correlation_*.png`` y ``vif_*.csv``.
    """
    st.header("Bivariado / Multivariado (US-012)")
    st.markdown(
        "Heatmaps de correlación entre bandas Sentinel-2, índices espectrales "
        "(NDVI, NDWI, EVI, NBR, etc.) y embeddings AlphaEarth. Tabla VIF para "
        "detectar multicolinealidad antes de feature engineering."
    )

    pngs = _list_pngs(figures_dir, "correlation_*.png")
    pairplots = _list_pngs(figures_dir, "pairplot_*.png")
    csvs = _list_csvs(figures_dir, "vif_*.csv")

    if not (pngs or pairplots or csvs):
        st.info("Pendiente — esperando US-012 (figuras bivariado/multivariado).")
        return

    if pngs:
        st.subheader("Correlación")
        _render_pngs(pngs)
    if pairplots:
        st.subheader("Pairplots")
        _render_pngs(pairplots)
    if csvs:
        st.subheader("Tablas VIF")
        _render_csvs(csvs)


def render_tab_temporal(figures_dir: Path) -> None:
    """Renderiza el tab 4 — Análisis temporal (US-012).

    Muestra pico NDVI, ACF/PACF, centroides DTW y cruces ERA5 x NDVI.

    Args:
        figures_dir: Directorio con figuras temporales US-012.
    """
    st.header("Temporal — fenología, ACF y clima (US-012)")
    st.markdown(
        "Análisis temporal de la respuesta espectral por clase: pico NDVI, "
        "autocorrelación (ACF/PACF), agrupamiento DTW de series temporales y "
        "cruces con ERA5 (precipitación anual) sobre las ROIs italianas."
    )

    keys = ("peak_ndvi", "acf", "dtw", "era5")
    pngs: list[Path] = []
    for key in keys:
        pngs.extend(_list_pngs(figures_dir, f"{key}*.png"))

    csvs = _list_csvs(figures_dir, "peak_ndvi*.csv") + _list_csvs(figures_dir, "dtw*.csv")

    if not (pngs or csvs):
        st.info("Pendiente — esperando US-012 (figuras temporales).")
        return

    _render_pngs(pngs)
    if csvs:
        st.subheader("Tablas")
        _render_csvs(csvs)


# ---------------------------------------------------------------------------
# Tab espacial (folium)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def build_folium_map(rois: list[dict[str, Any]], pastis_geojson_path: str | None) -> Any:
    """Construye el mapa folium con ROIs italianas + tiles PASTIS dissolved.

    Args:
        rois: Lista de ROIs cargadas desde ``config/rois.yaml`` (campo ``rois``).
        pastis_geojson_path: Ruta al ``metadata.geojson`` de PASTIS-R, o ``None``.

    Returns:
        Objeto ``folium.Map`` listo para ``st_folium`` o ``None`` si folium no
        está instalado.
    """
    if not _HAS_FOLIUM or folium is None:  # pragma: no cover - import guard
        return None

    # Centro inicial sobre Italia continental (Pianura Padana).
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
        color = "#2E86AB" if region == "Italy" else "#1B998B"
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

    # Capa PASTIS dissolved por TILE — cap defensivo a TOP-N para no saturar
    # el render (PASTIS tiene 2,433 patches en 4 super-tiles).
    if pastis_geojson_path and _HAS_GEOPANDAS and gpd is not None:
        path = Path(pastis_geojson_path)
        if path.exists():
            try:
                gdf = gpd.read_file(path)
                tile_col = next((c for c in ("TILE", "tile", "Tile") if c in gdf.columns), None)
                if tile_col is not None:
                    dissolved = gdf.dissolve(by=tile_col).head(_FOLIUM_TOP_TILES)
                else:
                    dissolved = gdf.head(_FOLIUM_TOP_TILES)
                if dissolved.crs is not None and dissolved.crs.to_epsg() != 4326:
                    dissolved = dissolved.to_crs(epsg=4326)
                folium.GeoJson(
                    dissolved.__geo_interface__,
                    name="PASTIS tiles dissolved",
                    style_function=lambda _f: {
                        "color": "#1B998B",
                        "weight": 1.5,
                        "fillOpacity": 0.25,
                    },
                ).add_to(france_fg)
            except (OSError, ValueError, RuntimeError):
                # Falla silenciosa: el mapa sigue siendo útil sin tiles.
                pass

    italy_fg.add_to(fmap)
    france_fg.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def render_tab_spatial(rois_yaml: Path, pastis_metadata: Path) -> None:
    """Renderiza el tab 5 — Mapa folium con ROIs Italia + PASTIS Francia.

    Args:
        rois_yaml: Ruta a ``config/rois.yaml``.
        pastis_metadata: Ruta a ``data/PASTIS-R/metadata.geojson``.
    """
    st.header("Espacial — ROIs Italia + PASTIS-R Francia")
    st.markdown(
        "Mapa interactivo con las 3 ROIs italianas (Pianura Padana, Toscana "
        "centrale, Apulia) y los tiles agregados (dissolved por TILE) del "
        "dataset PASTIS-R en Francia. Usar el control de capas para alternar."
    )

    if not _HAS_FOLIUM:
        st.info(
            "streamlit-folium no instalado; ejecutar "
            "`poetry install --with paper` para habilitar este tab."
        )
        return

    if not rois_yaml.exists():
        st.warning(f"No existe `{rois_yaml}` — el mapa se renderiza vacío.")
        rois_data: list[dict[str, Any]] = []
    else:
        loaded = load_yaml(rois_yaml)
        rois_data = list(loaded.get("rois", []))

    pastis_path = str(pastis_metadata) if pastis_metadata.exists() else None
    if pastis_path is None:
        st.caption("PASTIS-R metadata no disponible — se muestran solo ROIs italianas.")

    fmap = build_folium_map(rois_data, pastis_path)
    if fmap is None:
        st.info("Mapa no disponible — folium no inicializó.")
        return

    st_folium(fmap, width=900, height=560, returned_objects=[])


# ---------------------------------------------------------------------------
# Tab 6 — Conclusiones CRISP-ML(Q)
# ---------------------------------------------------------------------------


def render_tab_conclusions() -> None:
    """Renderiza el tab 6 — Conclusiones CRISP-ML(Q) en 6 sub-secciones."""
    st.header("Conclusiones CRISP-ML(Q) — Avance 1")
    st.markdown(
        "Mapeo explícito de los hallazgos del EDA a las 6 fases del marco "
        "CRISP-ML(Q). Cada sección sintetiza decisiones para el Avance 2."
    )

    st.subheader("1. Business Understanding")
    st.markdown(
        "AgroSatCopilot resuelve el monitoreo de cultivos en Italia y Francia "
        "para sostenibilidad, gestión hídrica y trazabilidad agroalimentaria. "
        "El Avance 1 valida si la combinación AlphaEarth + Sentinel-2 + PASTIS-R "
        "ofrece señal suficiente para alcanzar F1-macro ≥ 0.80 al cierre del MVP."
    )

    st.subheader("2. Data Understanding")
    st.markdown(
        "- **Fuentes**: Sentinel-2 L2A (Copernicus), AlphaEarth 64-dim (GEE), "
        "PASTIS-R (INRAE), Dynamic World (Google + WRI), ERA5-Land (C3S).\n"
        "- **Calidad**: missingness SCL por ROI/estación documentada en US-010; "
        "outliers controlados con stretch 2-98 por banda.\n"
        "- **Integrabilidad**: GEE permite muestreo on-the-fly sobre todas las "
        "fuentes con CRS armonizado (EPSG:4326 input, proyección métrica por "
        "ROI para análisis)."
    )

    st.subheader("3. Data Preparation")
    st.markdown(
        "Decisiones que alimentan US-014 (feature engineering): (a) descartar "
        "bandas con missingness > 30% por temporada; (b) usar embeddings "
        "AlphaEarth como features de alta señal — VIF aceptable en US-012; "
        "(c) NDVI/NDWI/EVI/NBR como features derivadas; (d) DTW clusters como "
        "feature temporal sintética para clases con fenología distintiva."
    )

    st.subheader("4. Modeling")
    st.markdown(
        "Baseline (US-017) será XGBoost sobre AlphaEarth + S2 features (F1 ≥ 0.60). "
        "Modelos densos en EPIC 5 (U-Net, DeepLabv3+, SegFormer-B2, U-TAE, "
        "TSViT, Swin-UNETR). Ensambles en EPIC 6 (voting top-3, bagging XGB, "
        "stacking con Gemma 4 LoRA, blending Optuna)."
    )

    st.subheader("5. Evaluation")
    st.markdown(
        "Métricas mínimas: F1-macro ≥ 0.80, mIoU ≥ 0.70 en segmentación densa, "
        "AgroMind ≥ 0.70 (Qwen3.5) y ≥ 0.75 (Gemini 3.1 Pro), GeoAnalystBench "
        "pass-rate ≥ 0.65. Cross-region consistency Italia ↔ Francia "
        "documentada en US-011 sec 3."
    )

    st.subheader("6. Deployment")
    st.markdown(
        "Agente Google ADK con 9 tools geoespaciales + Spatial-RAG sobre "
        "pgvector. Backend FastAPI + TiTiler en Cloud Run scale-to-zero (~$115 "
        "USD/mes). Frontend Nuxt 4 SSR con MapLibre + deck.gl. Inferencia pesada "
        "vía Pub/Sub a worker GPU L4 para no bloquear `/chat` SSE."
    )


# ---------------------------------------------------------------------------
# Sidebar + footer + entry point
# ---------------------------------------------------------------------------


_TAB_LABELS: tuple[str, ...] = (
    "1 · Univariado S2",
    "2 · AlphaEarth",
    "3 · Bivariado",
    "4 · Temporal",
    "5 · Espacial",
    "6 · Conclusiones",
)


def _render_sidebar() -> None:
    """Renderiza la sidebar con título + lista de los 6 tabs."""
    with st.sidebar:
        st.title("Avance 1 — EDA · AgroSatCopilot")
        st.markdown("Dashboard file-driven · cierre EDA CRISP-ML(Q).")
        st.markdown("**Navegación**")
        for label in _TAB_LABELS:
            st.markdown(f"- {label}")
        st.divider()
        st.caption("Owner: Aaron Bocanegra · Contenido: Isaac Ávila")
        st.caption("Sprint S2-recovery · 2026-05-12")


def _render_footer() -> None:
    """Renderiza el footer con atribuciones de datasets y modelos."""
    st.divider()
    st.markdown(
        "**Atribuciones**: "
        "PASTIS-R (Sainte-Fare-Garnot et al. 2021, CC-BY-SA 4.0) · "
        "Sentinel-2 (Copernicus, modified data 2017-2025) · "
        "AlphaEarth Foundations (Google DeepMind, GEE ToS) · "
        "Dynamic World (Google + WRI, CC-BY-4.0) · "
        "ERA5-Land (Copernicus C3S) · "
        "HCAT3 / EuroCrops (TUM, CC-BY-4.0)."
    )
    st.caption("Ver `docs/licenses/DATA_LICENSE.md` para detalle completo de licencias.")


def main() -> None:
    """Punto de entrada Streamlit: configura página, sidebar y 6 tabs."""
    st.set_page_config(
        page_title="AgroSatCopilot · EDA Avance 1",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _render_sidebar()
    st.title("AgroSatCopilot — Análisis Exploratorio de Datos (Avance 1)")
    st.markdown(
        "Consolidación de US-010 (Sentinel-2 univariado), US-011 (AlphaEarth "
        "cross-region) y US-012 (bivariado + temporal + espacial)."
    )

    tabs = st.tabs(list(_TAB_LABELS))

    with tabs[0]:
        render_tab_univariate(FIGURES_ROOT / "us-010")
    with tabs[1]:
        render_tab_alphaearth(FIGURES_ROOT / "us-011")
    with tabs[2]:
        render_tab_bivariate(FIGURES_ROOT / "us-012")
    with tabs[3]:
        render_tab_temporal(FIGURES_ROOT / "us-012")
    with tabs[4]:
        render_tab_spatial(ROIS_YAML, PASTIS_METADATA)
    with tabs[5]:
        render_tab_conclusions()

    _render_footer()


if __name__ == "__main__":  # pragma: no cover - streamlit entry
    main()
