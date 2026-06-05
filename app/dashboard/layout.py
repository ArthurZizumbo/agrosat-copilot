"""Page chrome: configuration, hero, sidebar and footer.

The sidebar is data-driven: it iterates the section registry instead of
hardcoding the navigation, so adding an Avance does not force editing this
module. The hero presents the project identity and its evolution narrative
(not only Avance 1).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:  # avoids circular import at runtime
    from app.dashboard.registry import Section


def configure_page() -> None:
    """Apply the global configuration of the Streamlit page."""
    st.set_page_config(
        page_title="AgroSatCopilot - Evolucion del proyecto",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_hero() -> None:
    """Render the hero banner with project identity and narrative."""
    st.markdown(
        '<div class="hero-banner">'
        "<h1>AgroSatCopilot - Cuantificacion de cultivos por satelite</h1>"
        "<p>Recorrido por la evolucion del proyecto, del Analisis Exploratorio "
        "de Datos (A1) a la segmentacion semantica densa (A4): seis notebooks "
        "de EDA, tres de ingenieria de caracteristicas, un baseline tabular "
        "fenologico y seis arquitecturas de segmentacion, narrados figura a "
        "figura con sus metricas reales y las decisiones que conectan cada "
        "fase con la siguiente.</p>"
        '<div class="hero-meta">'
        "<span><strong>Curso:</strong> MNA - Tec de Monterrey</span>"
        "<span><strong>Recorrido:</strong> A1 EDA -> A2 FE -> A3 Baseline -> A4 Segmentacion</span>"
        "<span><strong>Equipo:</strong> 17</span>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_sidebar(sections: Sequence[Section], active_key: str) -> None:
    """Render the sidebar with data-driven navigation and metadata.

    Args:
        sections: Registry of dashboard sections.
        active_key: ``key`` of the active section, highlighted in the navigation.
    """
    with st.sidebar:
        st.title("AgroSatCopilot")
        st.markdown("**Historia · EDA · FE · Baseline · Segmentacion**")
        st.markdown("---")
        st.markdown("**Equipo**")
        st.markdown("- Arthur Zizumbo · MLOps Lead")
        st.markdown("- Aaron Bocanegra · Backend / Full-Stack")
        st.markdown("- Isaac Avila · ML / Data Scientist")
        st.markdown("---")
        st.markdown("**Datasets**")
        st.markdown("- PASTIS-R · Sentinel-2")
        st.markdown("- AlphaEarth v2.1")
        st.markdown("- Dynamic World · ERA5")
        st.markdown("- HCAT3 / EuroCrops")
        st.markdown("---")
        st.markdown("**Navegacion**")
        for section in sections:
            is_active = section.key == active_key
            suffix = "  ·  seccion activa" if is_active else ""
            st.markdown(f"**{section.label}**{suffix}")
            for label in section.tab_labels:
                prefix = "▸ " if is_active else "- "
                st.markdown(f"{prefix}{label}")


def render_footer() -> None:
    """Render the footer with dataset and model attributions."""
    st.markdown(
        '<div class="footer-attributions">'
        "<strong>Atribuciones</strong><br>"
        "PASTIS-R (Sainte-Fare-Garnot et al. 2021, CC-BY-SA 4.0) · "
        "Sentinel-2 (Copernicus, datos modificados 2017-2025) · "
        "AlphaEarth Foundations (Google DeepMind, terminos del GEE) · "
        "Dynamic World (Google + WRI, CC-BY-4.0) · "
        "ERA5-Land (Copernicus C3S) · "
        "HCAT3 / EuroCrops (TUM, CC-BY-4.0).<br><br>"
        "<span style='font-size:0.78rem;'>Detalle completo en "
        "<code>docs/licenses/DATA_LICENSE.md</code>.</span>"
        "</div>",
        unsafe_allow_html=True,
    )
