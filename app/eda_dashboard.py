"""Dashboard Streamlit de EDA - Avance 1 AgroSatCopilot (US-013).

Design system: Data-Dense Dashboard (Fira Sans + Fira Code, paleta azul +
ambar, fondo #F8FAFC). Cada figura va acompanada de una narrativa
interpretativa que explica que muestra, como se construyo y que implica
para los siguientes Avances.

Para arrancar localmente::

    poetry run streamlit run app/eda_dashboard.py --server.port 8501

El dashboard degrada de forma graceful cuando faltan figuras o narrativas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import streamlit as st
import yaml

from ml.report.figure_narratives import FigureNarrative, get_narrative
from ml.report.notebook_content import (
    ALPHAEARTH_CARD,
    BIVARIATE_CARD,
    CARDS,
    GLOBAL_CARD,
    PASTIS_CARD,
    SENTINEL2_CARD,
    NotebookCard,
    list_figures,
)

# streamlit-folium es opcional: si no esta instalado, el tab espacial muestra
# un mensaje en vez de crashear.
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


# Rutas canonicas relativas a la raiz del repositorio.
REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_ROOT = REPO_ROOT / "paper" / "figures"
ROIS_YAML = REPO_ROOT / "config" / "rois.yaml"
# Para PASTIS preferimos el subset compacto (~500 KB, dissolved por tile)
# que sí está commiteado al repo y funciona en Streamlit Cloud. Si no existe
# (entornos antiguos sin el subset), caemos al metadata completo de 19 MB
# que solo está disponible en máquinas con DVC sincronizado.
PASTIS_METADATA_COMPACT = REPO_ROOT / "data" / "reference" / "pastis_tiles_dissolved.geojson"
PASTIS_METADATA_FULL = REPO_ROOT / "data" / "PASTIS-R" / "metadata.geojson"
PASTIS_METADATA = (
    PASTIS_METADATA_COMPACT if PASTIS_METADATA_COMPACT.exists() else PASTIS_METADATA_FULL
)

# Cap defensivo de filas a renderizar en st.dataframe para no saturar al cliente.
_DATAFRAME_HEAD_ROWS = 200
# Cap defensivo de clases en el mapa folium (top-N por area dissolved).
_FOLIUM_TOP_TILES = 10


# ---------------------------------------------------------------------------
# Design System - CSS injection (Data-Dense Dashboard)
# ---------------------------------------------------------------------------


_DESIGN_SYSTEM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --color-primary: #2563EB;
    --color-secondary: #3B82F6;
    --color-accent: #F97316;
    --color-bg: #F8FAFC;
    --color-surface: #FFFFFF;
    --color-border: #E2E8F0;
    --color-text: #1E293B;
    --color-text-muted: #64748B;
    --color-success: #10B981;
    --color-warning: #F59E0B;
    --shadow-sm: 0 1px 2px 0 rgba(0,0,0,0.04);
    --shadow-md: 0 2px 8px -1px rgba(15,23,42,0.06), 0 1px 4px -1px rgba(15,23,42,0.04);
    --shadow-lg: 0 8px 24px -4px rgba(15,23,42,0.08), 0 4px 12px -2px rgba(15,23,42,0.04);
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
}

html, body, [class*="css"] {
    font-family: 'Fira Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--color-text);
}

code, pre, .stCodeBlock {
    font-family: 'Fira Code', 'Courier New', monospace !important;
}

.stApp {
    background-color: var(--color-bg) !important;
}

/* Hero header */
.hero-banner {
    background: linear-gradient(135deg, #1E40AF 0%, #2563EB 50%, #3B82F6 100%);
    color: white;
    padding: 2rem 2.5rem;
    border-radius: var(--radius-lg);
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-lg);
}

.hero-banner h1 {
    color: white !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    margin: 0 0 0.5rem 0 !important;
    letter-spacing: -0.02em;
}

.hero-banner p {
    color: rgba(255,255,255,0.92) !important;
    font-size: 1.05rem !important;
    margin: 0 !important;
    max-width: 900px;
    line-height: 1.5;
}

.hero-meta {
    margin-top: 1rem;
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
    font-size: 0.85rem;
    color: rgba(255,255,255,0.8);
}

.hero-meta strong {
    color: #FED7AA;
    font-weight: 500;
}

/* KPI cards row */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin: 1rem 0 1.5rem 0;
}

.kpi-card {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 200ms ease, transform 200ms ease;
}

.kpi-card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-1px);
}

.kpi-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--color-text-muted);
    font-weight: 500;
    margin-bottom: 0.4rem;
}

.kpi-value {
    font-size: 1.75rem;
    font-weight: 600;
    color: var(--color-primary);
    font-family: 'Fira Code', monospace;
    line-height: 1.1;
}

.kpi-delta {
    font-size: 0.8rem;
    color: var(--color-text-muted);
    margin-top: 0.3rem;
}

/* Figure cards */
.figure-card {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    margin: 1rem 0;
    box-shadow: var(--shadow-sm);
}

.figure-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--color-text);
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.figure-title::before {
    content: '';
    width: 4px;
    height: 16px;
    background: var(--color-accent);
    border-radius: 2px;
    display: inline-block;
}

.narrative-block {
    background: #F1F5F9;
    border-left: 3px solid var(--color-primary);
    border-radius: var(--radius-sm);
    padding: 0.85rem 1rem;
    margin: 0.6rem 0 0.8rem 0;
    font-size: 0.92rem;
    line-height: 1.6;
    color: #334155;
}

.method-block {
    font-size: 0.82rem;
    color: var(--color-text-muted);
    background: #FAFAFA;
    border: 1px dashed var(--color-border);
    border-radius: var(--radius-sm);
    padding: 0.55rem 0.85rem;
    margin-top: 0.4rem;
    font-style: italic;
}

.method-block strong {
    color: var(--color-text);
    font-style: normal;
    font-weight: 500;
}

/* Conclusion cards */
.conclusion-card {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-left: 4px solid var(--color-primary);
    border-radius: var(--radius-md);
    padding: 1rem 1.2rem;
    margin: 0.7rem 0;
    box-shadow: var(--shadow-sm);
}

.conclusion-card.accent {
    border-left-color: var(--color-accent);
}

.conclusion-card.success {
    border-left-color: var(--color-success);
}

.conclusion-heading {
    font-size: 1rem;
    font-weight: 600;
    color: var(--color-text);
    margin-bottom: 0.45rem;
}

.conclusion-body {
    font-size: 0.92rem;
    line-height: 1.6;
    color: #334155;
    margin: 0;
}

/* Section dividers */
.section-divider {
    margin: 2rem 0 1.25rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--color-border);
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.section-divider h3 {
    margin: 0 !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: var(--color-text) !important;
}

.section-divider-badge {
    background: var(--color-primary);
    color: white;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.2rem 0.55rem;
    border-radius: 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Source link pill */
.source-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #EFF6FF;
    color: var(--color-primary);
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-family: 'Fira Code', monospace;
    margin: 0.4rem 0;
    border: 1px solid #DBEAFE;
}

/* Streamlit overrides */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
    background: var(--color-surface);
    padding: 0.4rem;
    border-radius: var(--radius-md);
    border: 1px solid var(--color-border);
    box-shadow: var(--shadow-sm);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: var(--radius-sm);
    padding: 0.5rem 1rem;
    font-weight: 500;
    color: var(--color-text-muted);
    transition: background 150ms ease, color 150ms ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background: #F1F5F9;
    color: var(--color-text);
}

.stTabs [aria-selected="true"] {
    background: var(--color-primary) !important;
    color: white !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
}

[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: white !important;
}

[data-testid="stExpander"] {
    border: 1px solid var(--color-border) !important;
    border-radius: var(--radius-md) !important;
    background: var(--color-surface) !important;
    box-shadow: var(--shadow-sm);
}

/* Footer */
.footer-attributions {
    margin-top: 2.5rem;
    padding: 1.5rem;
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    font-size: 0.85rem;
    color: var(--color-text-muted);
    line-height: 1.7;
}

.footer-attributions strong {
    color: var(--color-text);
}

/* Reduce motion */
@media (prefers-reduced-motion: reduce) {
    .kpi-card, .stTabs [data-baseweb="tab"] {
        transition: none !important;
    }
}
</style>
"""


# ---------------------------------------------------------------------------
# Loaders cacheados
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pl.DataFrame:
    """Carga un CSV como ``polars.DataFrame`` con cache de Streamlit.

    Args:
        path: Ruta absoluta o relativa al CSV en disco.

    Returns:
        DataFrame de Polars. Si la lectura falla devuelve un DataFrame vacio.
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
        Diccionario parseado. Vacio si el archivo no existe.
    """
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    return loaded if isinstance(loaded, dict) else {}


def _list_csvs(directory: Path, pattern: str = "*.csv") -> list[Path]:
    """Lista CSVs ordenados alfabeticamente filtrando por glob."""
    if not directory.exists():
        return []
    return sorted(directory.glob(pattern))


# ---------------------------------------------------------------------------
# KPI cards por ficha (fuente unica: NotebookCard.kpis)
# ---------------------------------------------------------------------------


def _render_kpi_row(card: NotebookCard) -> None:
    """Renderiza una fila de KPI cards para la ficha dada."""
    if not card.kpis:
        return
    cards_html = "".join(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{kpi.label}</div>'
        f'<div class="kpi-value">{kpi.value}</div>'
        f'<div class="kpi-delta">{kpi.delta}</div>'
        f"</div>"
        for kpi in card.kpis
    )
    st.markdown(f'<div class="kpi-row">{cards_html}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Render unificado de ficha (uno por notebook)
# ---------------------------------------------------------------------------


def _render_card_header(card: NotebookCard) -> None:
    """Renderiza titulo, subtitulo, pill del notebook fuente y KPIs."""
    st.markdown(
        f'<h2 style="margin-top:0.5rem;color:#1E293B;font-weight:700;">{card.title}</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:#475569;font-size:1rem;line-height:1.6;'
        f'margin-bottom:0.6rem;">{card.subtitle}</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="source-pill">Notebook fuente: {card.notebook_path}</div>',
        unsafe_allow_html=True,
    )

    _render_kpi_row(card)

    if card.sections:
        with st.expander("Índice del notebook (secciones)", expanded=False):
            for section in card.sections:
                st.markdown(f"- {section}")


def _render_figure_with_narrative(png_path: Path, narrative: FigureNarrative | None) -> None:
    """Renderiza una figura PNG con su narrativa interpretativa al lado."""
    title = narrative.title if narrative is not None else png_path.stem.replace("_", " ").title()

    st.markdown(
        f'<div class="figure-card"><div class="figure-title">{title}</div></div>',
        unsafe_allow_html=True,
    )

    col_img, col_text = st.columns([3, 2], gap="medium")
    with col_img:
        st.image(str(png_path), use_container_width=True)
    with col_text:
        if narrative is not None:
            st.markdown(
                f'<div class="narrative-block">{narrative.narrative}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="method-block">'
                f"<strong>Cómo se construyó:</strong> {narrative.method}"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="method-block">'
                f"<strong>Figura:</strong> {png_path.name}. "
                f"Narrativa interpretativa pendiente de redacción."
                f"</div>",
                unsafe_allow_html=True,
            )


def _render_section_divider(label: str, badge: str | None = None) -> None:
    """Renderiza un divisor de seccion con label y badge opcional."""
    badge_html = f'<span class="section-divider-badge">{badge}</span>' if badge else ""
    st.markdown(
        f'<div class="section-divider">{badge_html}<h3>{label}</h3></div>',
        unsafe_allow_html=True,
    )


def _render_card_figures(card: NotebookCard) -> None:
    """Renderiza las figuras de la ficha con narrativa por figura."""
    pngs = list_figures(card, FIGURES_ROOT)
    csvs = _list_csvs(FIGURES_ROOT / card.figures_dir) if card.figures_dir else []

    if not pngs and not csvs:
        if card.figures_dir:
            st.info(
                f"Pendiente: no se encontraron figuras en "
                f"`paper/figures/{card.figures_dir}/`. Ejecutá el notebook "
                f"fuente para poblarlas."
            )
        return

    if pngs:
        _render_section_divider("Figuras del análisis", badge=f"{len(pngs)} figuras")
        for png in pngs:
            narrative = get_narrative(card.notebook_id, png.name)
            _render_figure_with_narrative(png, narrative)

    if csvs:
        _render_section_divider("Tablas asociadas", badge=f"{len(csvs)} tablas")
        for csv_path in csvs:
            st.markdown(
                f'<div class="figure-card"><div class="figure-title">{csv_path.name}</div></div>',
                unsafe_allow_html=True,
            )
            df = load_csv(csv_path)
            if df.is_empty():
                st.caption("Tabla vacía o ilegible.")
                continue
            st.dataframe(
                df.head(_DATAFRAME_HEAD_ROWS).to_pandas(),
                use_container_width=True,
            )


def _render_card_conclusions(card: NotebookCard) -> None:
    """Renderiza las conclusiones interpretadas como cards alternadas."""
    if not card.conclusions:
        return
    _render_section_divider(
        "Conclusiones e interpretación",
        badge=f"{len(card.conclusions)} hallazgos",
    )
    for idx, (heading, body) in enumerate(card.conclusions):
        # Alterna accent color para que se distingan visualmente
        css_class = "conclusion-card"
        if idx % 3 == 1:
            css_class += " accent"
        elif idx % 3 == 2:
            css_class += " success"
        st.markdown(
            f'<div class="{css_class}">'
            f'<div class="conclusion-heading">{heading}</div>'
            f'<p class="conclusion-body">{body}</p>'
            f"</div>",
            unsafe_allow_html=True,
        )


def render_card(card: NotebookCard) -> None:
    """Renderiza una ficha completa: header + KPIs + figuras + conclusiones."""
    _render_card_header(card)
    _render_card_figures(card)
    _render_card_conclusions(card)


# Aliases historicos para retrocompatibilidad de tests
def render_tab_sentinel2(figures_dir: Path | None = None) -> None:
    """Tab 1 - Sentinel-2 univariado (firma legacy)."""
    _ = figures_dir
    render_card(SENTINEL2_CARD)


def render_tab_alphaearth(figures_dir: Path | None = None) -> None:
    """Tab 2 - AlphaEarth (firma legacy)."""
    _ = figures_dir
    render_card(ALPHAEARTH_CARD)


def render_tab_bivariate(figures_dir: Path | None = None) -> None:
    """Tab 3 - Bivariado/temporal (firma legacy)."""
    _ = figures_dir
    render_card(BIVARIATE_CARD)


def render_tab_pastis(figures_dir: Path | None = None) -> None:
    """Tab 4 - PASTIS-R consolidado (firma legacy)."""
    _ = figures_dir
    render_card(PASTIS_CARD)


def render_tab_global() -> None:
    """Tab 5 - Conclusiones globales del Avance 1."""
    render_card(GLOBAL_CARD)


# ---------------------------------------------------------------------------
# Tab espacial (folium)
# ---------------------------------------------------------------------------


def build_folium_map(rois: list[dict[str, Any]], pastis_geojson_path: str | None) -> Any:
    """Construye el mapa folium con ROIs italianas + tiles PASTIS dissolved.

    No se cachea: cachear el objeto ``folium.Map`` con ``@st.cache_resource``
    causa colisiones de IDs DOM al re-renderizar (``feature_group_X is not
    defined``, ``layer_control_div_Y already declared``).

    Args:
        rois: Lista de ROIs cargadas desde ``config/rois.yaml``.
        pastis_geojson_path: Ruta al ``metadata.geojson`` o ``None``.

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
                        "color": "#F97316",
                        "weight": 1.5,
                        "fillOpacity": 0.25,
                    },
                ).add_to(france_fg)
            except (OSError, ValueError, RuntimeError):
                pass

    italy_fg.add_to(fmap)
    france_fg.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def render_tab_spatial(rois_yaml: Path, pastis_metadata: Path) -> None:
    """Tab espacial — mapa folium con ROIs Italia + PASTIS Francia.

    Args:
        rois_yaml: Ruta a ``config/rois.yaml``.
        pastis_metadata: Ruta a ``data/PASTIS-R/metadata.geojson``.
    """
    st.markdown(
        '<h2 style="margin-top:0.5rem;color:#1E293B;font-weight:700;">'
        "Mapa espacial — ROIs Italia y PASTIS-R Francia</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#475569;font-size:1rem;line-height:1.6;">'
        "Mapa interactivo con las tres regiones italianas (Pianura "
        "Padana, Toscana central y Apulia) en azul y los patches del "
        "dataset PASTIS-R en Francia en ámbar (agregados por TILE). "
        "Usá el control de capas en la esquina superior derecha del "
        "mapa para alternar visibilidad.</p>",
        unsafe_allow_html=True,
    )

    if not _HAS_FOLIUM:
        st.info(
            "streamlit-folium no está instalado. Ejecutá "
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
        st.caption("PASTIS-R metadata no disponible — se muestran solo las regiones italianas.")

    fmap = build_folium_map(rois_data, pastis_path)
    if fmap is None:
        st.info("Mapa no disponible — folium no inicializó.")
        return

    # `key` único por render evita colisiones de IDs en el DOM cuando
    # Streamlit re-renderiza el componente.
    st_folium(
        fmap,
        width=None,
        height=560,
        returned_objects=[],
        key="eda_spatial_map",
    )


# ---------------------------------------------------------------------------
# Sidebar + hero + footer + entry point
# ---------------------------------------------------------------------------


_TAB_LABELS: tuple[str, ...] = (
    "Sentinel-2",
    "AlphaEarth",
    "Bivariado / Temporal",
    "PASTIS-R Consolidado",
    "Conclusiones Globales",
    "Mapa Espacial",
)


def _render_hero() -> None:
    """Renderiza el hero banner con título, subtítulo y meta."""
    st.markdown(
        '<div class="hero-banner">'
        "<h1>AgroSatCopilot — Análisis Exploratorio de Datos</h1>"
        "<p>Reporte consolidado del Avance 1: cuatro notebooks de EDA "
        "(Sentinel-2 univariado, AlphaEarth Foundations, bivariado y "
        "temporal, y PASTIS-R) sintetizados en una vista única con "
        "narrativa por figura, KPIs y conclusiones por fase.</p>"
        '<div class="hero-meta">'
        "<span><strong>Curso:</strong> MNA — Tec de Monterrey</span>"
        "<span><strong>Sprint:</strong> S2-recovery</span>"
        "<span><strong>Avance:</strong> A1 — EDA (2026-05-13)</span>"
        "<span><strong>Sponsor:</strong> Dr. Camacho</span>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_sidebar() -> None:
    """Renderiza la sidebar con navegación y metadata del proyecto."""
    with st.sidebar:
        st.title("AgroSatCopilot")
        st.markdown("**EDA · Avance 1**")
        st.markdown("---")
        st.markdown("**Equipo**")
        st.markdown("- Arthur Zizumbo · MLOps Lead")
        st.markdown("- Aaron Bocanegra · Backend / Full-Stack")
        st.markdown("- Isaac Ávila · ML / Data Scientist")
        st.markdown("---")
        st.markdown("**Datasets**")
        st.markdown("- PASTIS-R · Sentinel-2")
        st.markdown("- AlphaEarth v2.1")
        st.markdown("- Dynamic World · ERA5")
        st.markdown("- HCAT3 / EuroCrops")
        st.markdown("---")
        st.markdown("**Navegación**")
        for label in _TAB_LABELS:
            st.markdown(f"- {label}")
        st.markdown("---")
        st.caption("Sprint S2-recovery · 2026-05-13")
        st.caption("Owner: Aaron Bocanegra")


def _render_footer() -> None:
    """Renderiza el footer con atribuciones de datasets y modelos."""
    st.markdown(
        '<div class="footer-attributions">'
        "<strong>Atribuciones</strong><br>"
        "PASTIS-R (Sainte-Fare-Garnot et al. 2021, CC-BY-SA 4.0) · "
        "Sentinel-2 (Copernicus, datos modificados 2017–2025) · "
        "AlphaEarth Foundations (Google DeepMind, términos del GEE) · "
        "Dynamic World (Google + WRI, CC-BY-4.0) · "
        "ERA5-Land (Copernicus C3S) · "
        "HCAT3 / EuroCrops (TUM, CC-BY-4.0).<br><br>"
        "<span style='font-size:0.78rem;'>Detalle completo en "
        "<code>docs/licenses/DATA_LICENSE.md</code>.</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    """Punto de entrada Streamlit: design system + sidebar + hero + tabs."""
    st.set_page_config(
        page_title="AgroSatCopilot - EDA Avance 1",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Design system CSS injection
    st.markdown(_DESIGN_SYSTEM_CSS, unsafe_allow_html=True)

    _render_sidebar()
    _render_hero()

    tabs = st.tabs(list(_TAB_LABELS))

    cards_in_order: tuple[NotebookCard, ...] = (
        SENTINEL2_CARD,
        ALPHAEARTH_CARD,
        BIVARIATE_CARD,
        PASTIS_CARD,
        GLOBAL_CARD,
    )
    assert len(cards_in_order) == len(CARDS), "CARDS y cards_in_order desincronizados"

    for tab, card in zip(tabs[:5], cards_in_order, strict=True):
        with tab:
            render_card(card)
    with tabs[5]:
        render_tab_spatial(ROIS_YAML, PASTIS_METADATA)

    _render_footer()


if __name__ == "__main__":  # pragma: no cover - streamlit entry
    main()
