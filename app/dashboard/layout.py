"""Chrome de pagina: configuracion, hero, sidebar y footer.

La sidebar es data-driven: itera el registro de secciones en lugar de
hardcodear la navegacion, de modo que agregar un Avance no obliga a editar
este modulo. El hero presenta la identidad del proyecto y su narrativa de
evolucion (no solo el Avance 1).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:  # evita import circular en runtime
    from app.dashboard.registry import Section


def configure_page() -> None:
    """Aplica la configuracion global de la pagina Streamlit."""
    st.set_page_config(
        page_title="AgroSatCopilot - Evolucion del proyecto",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_hero() -> None:
    """Renderiza el hero banner con identidad y narrativa del proyecto."""
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
    """Renderiza la sidebar con navegacion data-driven y metadata.

    Args:
        sections: Registro de secciones del dashboard.
        active_key: ``key`` de la seccion activa, resaltada en la navegacion.
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
    """Renderiza el footer con atribuciones de datasets y modelos."""
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
