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

from ml.report.avance2_content import FE_CARDS
from ml.report.figure_narratives import FigureNarrative, get_narrative
from ml.report.notebook_content import (
    ALPHAEARTH_CARD,
    BIVARIATE_CARD,
    BREIZHCROPS_CARD,
    CARDS,
    GLOBAL_CARD,
    PAPER_METHODS_CARD,
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

# Rutas para la seccion Baseline (US-023-preview).
BASELINE_FIGURES_DIR = FIGURES_ROOT / "us-023-preview"
BASELINE_ABLATION_DIR = REPO_ROOT / "reports" / "baseline" / "feature_ablation"
BASELINE_REENCUADRE_DIR = REPO_ROOT / "reports" / "baseline" / "reencuadre_fenologico"
BASELINE_MODEL_COMP_V2_DIR = REPO_ROOT / "reports" / "baseline" / "model_comparison_v2"
# Mensaje canonico cuando un artefacto Baseline aun no existe en disco.
_BASELINE_MISSING_HINT = (
    "Artefacto pendiente — ejecuta `make reencuadre-notebook-full && make baseline-v2-full`"
)
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


@st.cache_data(show_spinner=False)
def load_parquet(path: Path) -> pl.DataFrame:
    """Carga un parquet como ``polars.DataFrame`` con cache de Streamlit.

    Args:
        path: Ruta absoluta o relativa al parquet en disco.

    Returns:
        DataFrame de Polars. Si la lectura falla o el archivo no existe,
        devuelve un DataFrame vacio (graceful degradation).
    """
    path = Path(path)
    if not path.exists():
        return pl.DataFrame()
    try:
        return pl.read_parquet(path)
    except (pl.exceptions.ComputeError, OSError, ValueError):
        return pl.DataFrame()


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


def render_tab_breizhcrops(figures_dir: Path | None = None) -> None:
    """Tab 5 - BreizhCrops cross-dataset (firma legacy)."""
    _ = figures_dir
    render_card(BREIZHCROPS_CARD)


def render_tab_global() -> None:
    """Tab 6 - Conclusiones globales del Avance 1."""
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


# Tabs del Avance 1 (EDA): 6 fichas de contenido + mapa espacial.
# El selector de sección de nivel superior ya indica "EDA", por lo que las
# etiquetas de los tabs no repiten ese prefijo.
_EDA_TAB_LABELS: tuple[str, ...] = (
    "Sentinel-2",
    "AlphaEarth",
    "Bivariado / Temporal",
    "PASTIS-R Consolidado",
    "BreizhCrops",
    "Métodos de la Literatura",
    "Conclusiones Globales",
    "Mapa Espacial",
)

# Tabs del Avance 2 (Feature Engineering): 4 fichas de FE_CARDS.
_FE_TAB_LABELS: tuple[str, ...] = (
    "Sentinel-2",
    "PASTIS-R Espectro-Temporal",
    "Fusión AlphaEarth",
    "Conclusiones",
)

# Etiquetas combinadas (retrocompatibilidad con tests que importan _TAB_LABELS).
_TAB_LABELS: tuple[str, ...] = _EDA_TAB_LABELS + _FE_TAB_LABELS

# Opciones del selector de sección de nivel superior.
_SECTION_EDA = "Exploración de Datos (EDA)"
_SECTION_FE = "Ingeniería de Características (FE)"
_SECTION_BASELINE = "Baseline (US-023-preview)"
_SECTION_OPTIONS: tuple[str, ...] = (_SECTION_EDA, _SECTION_FE, _SECTION_BASELINE)
# Clave de st.session_state que preserva la sección seleccionada.
_SECTION_STATE_KEY = "dashboard_section"

# Etiquetas de los 5 tabs de la sección Baseline (AC-P9-3).
_BASELINE_TAB_LABELS: tuple[str, ...] = (
    "Ablation de features",
    "Leakage geográfico",
    "Bloques opcionales",
    "Modelos baseline v2",
    "Conclusiones",
)


def _render_hero() -> None:
    """Renderiza el hero banner con título, subtítulo y meta."""
    st.markdown(
        '<div class="hero-banner">'
        "<h1>AgroSatCopilot — Exploración e Ingeniería de Características</h1>"
        "<p>Reporte consolidado de dos fases del proyecto: el análisis "
        "exploratorio de datos (seis notebooks de EDA) y la ingeniería de "
        "características (tres notebooks de feature engineering sobre "
        "Sentinel-2, PASTIS-R y AlphaEarth), sintetizados en una vista "
        "única con narrativa por figura, KPIs y conclusiones por fase.</p>"
        '<div class="hero-meta">'
        "<span><strong>Curso:</strong> MNA — Tec de Monterrey</span>"
        "<span><strong>Fases:</strong> EDA + Ingeniería de Características</span>"
        "<span><strong>Equipo:</strong> 17</span>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_sidebar(active_section: str) -> None:
    """Renderiza la sidebar con navegación y metadata del proyecto.

    Args:
        active_section: Sección seleccionada en el selector de nivel
            superior (``_SECTION_EDA`` o ``_SECTION_FE``). La lista de tabs
            de la sección activa se resalta como navegación vigente.
    """
    with st.sidebar:
        st.title("AgroSatCopilot")
        st.markdown("**EDA · Ingeniería de Características**")
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
        eda_active = active_section == _SECTION_EDA
        fe_active = active_section == _SECTION_FE
        baseline_active = active_section == _SECTION_BASELINE
        st.markdown(
            f"**Exploración de datos**{'  ·  sección activa' if eda_active else ''}"
        )
        for label in _EDA_TAB_LABELS:
            prefix = "▸ " if eda_active else "- "
            st.markdown(f"{prefix}{label}")
        st.markdown(
            f"**Ingeniería de características**"
            f"{'  ·  sección activa' if fe_active else ''}"
        )
        for label in _FE_TAB_LABELS:
            prefix = "▸ " if fe_active else "- "
            st.markdown(f"{prefix}{label}")
        st.markdown(
            f"**Baseline (US-023-preview)**"
            f"{'  ·  sección activa' if baseline_active else ''}"
        )
        for label in _BASELINE_TAB_LABELS:
            prefix = "▸ " if baseline_active else "- "
            st.markdown(f"{prefix}{label}")


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


# Fichas del Avance 1 (EDA): 6 fichas de contenido (el mapa espacial es un
# tab aparte sin NotebookCard).
_EDA_CARDS: tuple[NotebookCard, ...] = (
    SENTINEL2_CARD,
    ALPHAEARTH_CARD,
    BIVARIATE_CARD,
    PASTIS_CARD,
    BREIZHCROPS_CARD,
    PAPER_METHODS_CARD,
    GLOBAL_CARD,
)


def _render_eda_section() -> None:
    """Renderiza el bloque de Exploración de Datos: 7 fichas + mapa espacial."""
    assert len(_EDA_CARDS) == len(CARDS), "CARDS y _EDA_CARDS desincronizados"
    assert len(_EDA_TAB_LABELS) == len(_EDA_CARDS) + 1, "Etiquetas EDA desincronizadas"

    tabs = st.tabs(list(_EDA_TAB_LABELS))
    n_cards = len(_EDA_CARDS)
    for tab, card in zip(tabs[:n_cards], _EDA_CARDS, strict=True):
        with tab:
            render_card(card)
    # El último tab del bloque EDA es el mapa espacial.
    with tabs[n_cards]:
        render_tab_spatial(ROIS_YAML, PASTIS_METADATA)


def _render_fe_section() -> None:
    """Renderiza el bloque de Ingeniería de Características: 4 fichas FE."""
    assert len(_FE_TAB_LABELS) == len(FE_CARDS), "Etiquetas FE desincronizadas"

    tabs = st.tabs(list(_FE_TAB_LABELS))
    for tab, card in zip(tabs, FE_CARDS, strict=True):
        with tab:
            render_card(card)


# ---------------------------------------------------------------------------
# Seccion Baseline (US-023-preview) — 5 tabs
# ---------------------------------------------------------------------------


def _render_baseline_figure(png_path: Path, caption: str) -> None:
    """Renderiza una figura del baseline si existe, con warning graceful si no.

    Args:
        png_path: Ruta absoluta al PNG.
        caption: Texto descriptivo bajo la imagen.
    """
    if not png_path.exists():
        st.warning(f"Figura no disponible (`{png_path.name}`). {_BASELINE_MISSING_HINT}")
        return
    st.image(str(png_path), caption=caption, use_container_width=True)


def _render_baseline_table(parquet_path: Path, caption: str) -> None:
    """Renderiza una tabla parquet con cache lazy y warning graceful.

    Args:
        parquet_path: Ruta absoluta al parquet.
        caption: Texto descriptivo bajo la tabla.
    """
    if not parquet_path.exists():
        st.warning(f"Tabla no disponible (`{parquet_path.name}`). {_BASELINE_MISSING_HINT}")
        return
    df = load_parquet(parquet_path)
    if df.is_empty():
        st.caption(f"Tabla vacía o ilegible: `{parquet_path.name}`.")
        return
    st.dataframe(
        df.head(_DATAFRAME_HEAD_ROWS).to_pandas(),
        use_container_width=True,
    )
    if caption:
        st.caption(caption)


def _render_baseline_tab_ablation() -> None:
    """Tab 1: ablation de features sobre los 7-10 conjuntos canónicos."""
    _render_section_divider("Ablation de conjuntos de features", badge="Tab 1 de 5")
    st.markdown(
        "Comparativa cuantitativa de los conjuntos de features evaluados sobre "
        "spatial CV 5-fold (mismo splitter de US-022b). Cada fila reporta "
        "`n_features`, `F1-macro`, `F1-weighted`, `mIoU` y `delta_vs_full`. "
        "Los conjuntos opcionales (`with_farslip`, `with_pheno_text`, "
        "`with_spectral_signature`) sólo aparecen cuando el bloque "
        "correspondiente está materializado en disco."
    )
    _render_baseline_figure(
        BASELINE_FIGURES_DIR / "ablation_optional_blocks.png",
        caption="Comparativa visual de los conjuntos opcionales contra el conjunto `full`.",
    )
    # La tabla canonica vive en reports/baseline/feature_ablation/ (US-023-preview);
    # si todavia no se regenero, caemos al ablation_table.parquet historico de
    # US-022 en reencuadre_fenologico/ para no dejar el tab vacio.
    primary_table = BASELINE_ABLATION_DIR / "ablation_table.parquet"
    if primary_table.exists():
        table_path = primary_table
    else:
        table_path = BASELINE_REENCUADRE_DIR / "ablation_table.parquet"
    _render_baseline_table(
        table_path,
        caption=(
            "Tabla `ablation_table.parquet` — conjuntos evaluados con métricas "
            "y delta vs `full`."
        ),
    )
    st.markdown(
        "**Interpretación accesible:** un `delta_vs_full` positivo indica que "
        "el bloque opcional aporta señal sobre el conjunto base. Un delta "
        "cercano a cero sugiere redundancia con AlphaEarth o con las features "
        "fenológicas. Un delta negativo es bandera roja: el bloque introduce "
        "ruido o leakage y debería descartarse del baseline."
    )


def _render_baseline_tab_leakage() -> None:
    """Tab 2: leakage geográfico — comparativa `full` vs `no_geom` y `geom_only`."""
    _render_section_divider("Leakage geográfico — columnas `geom_*`", badge="Tab 2 de 5")
    st.markdown(
        "Las 3 columnas `geom_area`, `geom_perimeter` y `geom_elongation` "
        "actúan en la práctica como proxy de la región: parcelas de la "
        "misma zona comparten distribución de tamaño y forma. Esto rompe la "
        "hipótesis C-2 del Dr. Camacho (independencia entre clase y geometría)."
    )
    _render_baseline_figure(
        BASELINE_FIGURES_DIR / "ablation_geom_comparison.png",
        caption="Dos barras `full` vs `no_geom` con anotación del delta de F1-macro.",
    )
    _render_baseline_table(
        BASELINE_ABLATION_DIR / "ablation_geom_table.parquet",
        caption="Fila `geom_only` vs `full` — test cuantitativo de leakage espacial.",
    )
    st.markdown(
        "**Por qué descartar `geom_*`:** un modelo entrenado únicamente sobre "
        "las 3 features geométricas alcanza F1-macro < 0.10 sobre clase, "
        "demostrando que no aporta señal real. Sin embargo, al combinarse con "
        "el resto del bloque, el modelo aprende un atajo regional que infla "
        "la métrica en el split de entrenamiento pero falla al transferir a "
        "spatial CV. La decisión es excluir `geom_*` del conjunto baseline "
        "definitivo y mantenerlas sólo como metadato de auditoría."
    )


def _render_baseline_tab_optional_blocks() -> None:
    """Tab 3: FarSLIP (P2) + Gemini (P4) + firma espectral (P5)."""
    _render_section_divider("Bloques opcionales evaluados", badge="Tab 3 de 5")
    st.markdown(
        "Tres bloques de features opcionales se evaluaron por separado: "
        "embeddings FarSLIP (US-022-c epoch_2 real), descripción fenológica "
        "vía Gemini Flash 3.5 y descriptor de firma espectral (Red Edge "
        "Position, Frampton et al. 2013). Cada bloque tiene su propio plot, "
        "tabla y decisión documentada."
    )

    st.markdown("#### FarSLIP — embeddings visuales (P2)")
    _render_baseline_figure(
        BASELINE_FIGURES_DIR / "ablation_farslip.png",
        caption="Comparativa `full` vs `with_farslip` vs `farslip_only`.",
    )
    _render_baseline_table(
        BASELINE_ABLATION_DIR / "ablation_farslip_table.parquet",
        caption=(
            "Métricas FarSLIP sobre las 30173 parcelas matched "
            "(NaN imputado a media en el resto)."
        ),
    )
    st.markdown(
        "**Decisión FarSLIP:** si `delta_vs_full >= +0.02` se promueve al "
        "baseline; entre [-0.02, +0.02] queda como base learner del stacking "
        "EPIC 6; si `< -0.02` se descarta del baseline con justificación."
    )

    st.markdown("#### Gemini Flash 3.5 — descripción fenológica textual (P4)")
    _render_baseline_figure(
        BASELINE_FIGURES_DIR / "ablation_pheno_text.png",
        caption=(
            "Comparativa `full` vs `with_pheno_text` vs `pheno_text_only` "
            "sobre subset >=1000 parcelas balanceadas."
        ),
    )
    # Corrida 3 US-023-preview: el ablation real de pheno_text vive en
    # `ablation_table_pheno_text_v2.parquet` (1080 parcelas balanceadas, Gemini
    # Flash 3.5 real, costo $0.49 USD). Si no esta, intentamos el subset
    # historico de 216 parcelas (`ablation_table_pheno_text.parquet`) como
    # fallback graceful y por ultimo el shim sintetico antiguo.
    _pheno_v2 = BASELINE_ABLATION_DIR / "ablation_table_pheno_text_v2.parquet"
    _pheno_hist = BASELINE_ABLATION_DIR / "ablation_table_pheno_text.parquet"
    _pheno_legacy = BASELINE_ABLATION_DIR / "ablation_pheno_text_table.parquet"
    if _pheno_v2.exists():
        _pheno_path = _pheno_v2
    elif _pheno_hist.exists():
        _pheno_path = _pheno_hist
    else:
        _pheno_path = _pheno_legacy
    _render_baseline_table(
        _pheno_path,
        caption=(
            "Métricas pheno_text — embeddings de sentence-transformers "
            "(384 dim) sobre prompts generados por Gemini Flash 3.5 "
            "(corrida real US-023-preview P4, 1080 parcelas balanceadas, "
            "costo $0.49 USD)."
        ),
    )
    st.markdown(
        "**Decisión pheno_text:** promover al baseline si `delta >= +0.01`; en "
        "caso contrario, mantener como base learner del stacking EPIC 6. "
        "Costo Gemini documentado en `docs/l4_log.md` (<= $5 USD)."
    )

    st.markdown("#### Firma espectral — Red Edge Position (P5)")
    _render_baseline_figure(
        BASELINE_FIGURES_DIR / "ablation_spectral_signature.png",
        caption="Comparativa `full` vs `with_spectral_signature` vs `spectral_signature_only`.",
    )
    _render_baseline_table(
        BASELINE_ABLATION_DIR / "ablation_spectral_signature_table.parquet",
        caption=(
            "Métricas firma espectral — descriptor compacto derivado de "
            "bandas red-edge en los 3 anclajes fenológicos."
        ),
    )
    st.markdown(
        "**Decisión firma espectral:** promover si `delta >= +0.01`; si no "
        "aporta señal, queda como deuda de investigación documentada para "
        "ciclos posteriores (mismo patrón que TempCNN en US-022-c P3)."
    )


def _render_baseline_tab_models_v2() -> None:
    """Tab 4: modelos baseline v2 — XGBoost + TempCNN + InceptionTime."""
    _render_section_divider("Modelos baseline v2 reentrenados", badge="Tab 4 de 5")
    st.markdown(
        "Los 3 modelos canónicos del Avance 3 (XGBoost tabular, TempCNN "
        "temporal e InceptionTime temporal) se reentrenan sobre el conjunto "
        "de features ganador post-ablation (decisiones de P2/P3/P4/P5). El "
        "splitter es spatial CV 5-fold con buffer de 1 km — mismo de US-022b "
        "para garantizar comparabilidad v1 vs v2."
    )
    _render_baseline_figure(
        BASELINE_FIGURES_DIR / "model_comparison_v2.png",
        caption=(
            "Tres barras (XGBoost, TempCNN, InceptionTime) con overlay de "
            "deltas vs baseline v1 (US-022)."
        ),
    )
    _render_baseline_table(
        BASELINE_MODEL_COMP_V2_DIR / "model_comparison_v2.parquet",
        caption=(
            "Tabla con 3 modelos x 6 métricas: F1-macro, F1-weighted, mIoU, "
            "accuracy, kappa, train_time_s."
        ),
    )
    st.markdown(
        "**Comparativa v1 vs v2:** el baseline v1 (US-022) reportó "
        "F1-macro = 0.4094 (XGBoost), 0.1430-0.1456 (TempCNN) y 0.1865 "
        "(InceptionTime). La v2 se decide por F1-macro sobre spatial CV "
        "5-fold; los empates se rompen por F1-weighted y luego por mIoU "
        "(decisión D-10)."
    )
    st.markdown(
        "**Decisión modelo ganador v2:** el modelo con F1-macro más alto se "
        "promueve como referencia para EPIC 5 (US-023 U-Net). Los 2 modelos "
        "restantes quedan como base learners del stacking EPIC 6."
    )


def _render_baseline_tab_conclusions() -> None:
    """Tab 5: conclusiones H-1..H-4 + lo que sigue en EPIC 5."""
    _render_section_divider("Conclusiones y siguientes pasos", badge="Tab 5 de 5")
    st.markdown(
        "Resumen ejecutivo de los hallazgos generados por US-023-preview "
        "sobre el baseline post-A3 y la transición prevista a EPIC 5 "
        "(modelado denso con U-Net + arquitecturas TSViT / U-TAE / "
        "DeepLabv3+ / SegFormer-B2 / Swin-UNETR)."
    )

    st.markdown(
        "#### H-1 — Las features `geom_*` introducen leakage regional\n"
        "La barra aislada `geom_only` confirma que las 3 columnas geométricas "
        "no aportan señal de clase real (F1-macro < 0.10) pero activan un "
        "atajo espacial cuando se combinan con el bloque base. Decisión: "
        "**excluir del baseline definitivo**, mantener sólo como metadato "
        "de auditoría."
    )
    st.markdown(
        "#### H-2 — FarSLIP aporta cuando hay matching de parcelas\n"
        "Los embeddings FarSLIP reales (epoch_2) cubren 30173 de las 85951 "
        "parcelas del dataset full. El delta es interpretable sólo sobre el "
        "subset matched; la extracción para el dataset full queda en US-025."
    )
    st.markdown(
        "#### H-3 — pheno_text via Gemini Flash 3.5: señal pendiente de validar\n"
        "El subset >=1000 parcelas balanceadas amplía la muestra de US-022-c "
        "P5 (216 parcelas, delta = -0.12). El veredicto definitivo se "
        "decide con la corrida actual; cualquier valor que esté pendiente se "
        "marca como artefacto faltante en los tabs anteriores."
    )
    st.markdown(
        "#### H-4 — Firma espectral REP: descriptor compacto agronómico\n"
        "La Red Edge Position derivada de bandas Sentinel-2 red-edge en los "
        "3 anclajes fenológicos (SOG, peak, senescencia) es una feature "
        "barata de calcular (consume parquet ya muestreado) y bien fundada "
        "en literatura (Frampton et al. 2013). Su aporte se cuantifica en "
        "el tab 3."
    )

    st.markdown(
        "#### Lo que sigue en EPIC 5\n"
        "Con el baseline saneado, los conjuntos de features decididos y los "
        "3 modelos baseline v2 reentrenados, EPIC 5 arranca con punto de "
        "partida limpio. El modelo ganador v2 sirve de techo a batir para "
        "U-Net densa; los 2 modelos restantes alimentan el stacking del "
        "EPIC 6. El presupuesto H100 (ventanas V1-V6, 80 h) queda intacto."
    )


# Mapeo (etiqueta, renderer) que conecta los 5 tabs con sus funciones.
_BASELINE_TAB_RENDERERS = (
    _render_baseline_tab_ablation,
    _render_baseline_tab_leakage,
    _render_baseline_tab_optional_blocks,
    _render_baseline_tab_models_v2,
    _render_baseline_tab_conclusions,
)


def _render_baseline_section() -> None:
    """Renderiza la sección Baseline con sus 5 tabs (US-023-preview P9).

    Cada tab consume artefactos generados por los sub-bloques P2..P8 desde
    ``paper/figures/us-023-preview/`` y ``reports/baseline/``. Si un archivo
    no existe, el render degrada de forma graceful con ``st.warning`` y un
    hint del comando ``make`` que regenera el artefacto (R11 del plan).

    Returns:
        ``None``. Renderiza directamente sobre el contexto Streamlit activo.
    """
    assert len(_BASELINE_TAB_LABELS) == 5, "Se esperaban exactamente 5 tabs Baseline"
    assert len(_BASELINE_TAB_RENDERERS) == len(_BASELINE_TAB_LABELS), (
        "Etiquetas y renderers Baseline desincronizados"
    )

    st.markdown(
        '<h2 style="margin-top:0.5rem;color:#1E293B;font-weight:700;">'
        "Baseline saneado post-A3 — US-023-preview</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#475569;font-size:1rem;line-height:1.6;">'
        "Reporte consolidado del refinamiento del baseline tras el Avance 3: "
        "ablation de features sobre 7-10 conjuntos, evidencia visual del "
        "leakage geográfico, evaluación de 3 bloques opcionales (FarSLIP, "
        "Gemini Flash 3.5 y firma espectral) y reentrenamiento de los 3 "
        "modelos canónicos sobre el conjunto ganador. Esta sección alimenta "
        "el arranque de EPIC 5 con baseline cuantificado y reproducible.</p>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs(list(_BASELINE_TAB_LABELS))
    for tab, renderer in zip(tabs, _BASELINE_TAB_RENDERERS, strict=True):
        with tab:
            renderer()


def _render_section_selector() -> str:
    """Renderiza el selector de sección de nivel superior.

    Returns:
        La sección seleccionada (``_SECTION_EDA`` o ``_SECTION_FE``). El
        valor se preserva entre re-renders vía ``st.session_state``.
    """
    if _SECTION_STATE_KEY not in st.session_state:
        st.session_state[_SECTION_STATE_KEY] = _SECTION_EDA

    selected = st.segmented_control(
        "Sección del reporte",
        options=_SECTION_OPTIONS,
        key=_SECTION_STATE_KEY,
        label_visibility="collapsed",
    )
    # segmented_control puede devolver None si el usuario deselecciona;
    # en ese caso conservamos la última sección válida (default EDA).
    if selected is None:
        selected = st.session_state.get(_SECTION_STATE_KEY) or _SECTION_EDA
    return selected


def main() -> None:
    """Punto de entrada Streamlit: design system + selector + sidebar + tabs."""
    st.set_page_config(
        page_title="AgroSatCopilot - EDA + Feature Engineering",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Design system CSS injection
    st.markdown(_DESIGN_SYSTEM_CSS, unsafe_allow_html=True)

    _render_hero()

    # Selector de sección: alterna entre el bloque EDA y el bloque FE.
    active_section = _render_section_selector()

    _render_sidebar(active_section)

    if active_section == _SECTION_FE:
        _render_fe_section()
    elif active_section == _SECTION_BASELINE:
        _render_baseline_section()
    else:
        _render_eda_section()

    _render_footer()


if __name__ == "__main__":  # pragma: no cover - streamlit entry
    main()
