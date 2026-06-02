"""Seccion Avance 1 - Analisis Exploratorio de Datos (EDA).

Renderiza las fichas de EDA (seis notebooks consolidados) mas el mapa
espacial folium como tab final. El contenido editorial vive en
``ml.report.notebook_content`` (``CARDS`` + ``CARDS_EXTRA``).
"""

from __future__ import annotations

import streamlit as st

from app.dashboard.components import render_card
from app.dashboard.paths import PAPER_FIGURES_ROOT, PASTIS_METADATA, ROIS_YAML
from app.dashboard.spatial import render_spatial_tab
from ml.report.notebook_content import EDA_DISPLAY_CARDS

# Fichas EDA en orden de presentacion (las 7 fichas; el mapa es un tab aparte).
_EDA_CARDS = EDA_DISPLAY_CARDS

EDA_TAB_LABELS: tuple[str, ...] = (
    "Sentinel-2",
    "AlphaEarth",
    "Bivariado / Temporal",
    "PASTIS-R Consolidado",
    "BreizhCrops",
    "Metodos de la Literatura",
    "Conclusiones Globales",
    "Mapa Espacial",
)


def render_eda_section() -> None:
    """Renderiza el bloque EDA: 7 fichas de contenido + mapa espacial."""
    assert len(EDA_TAB_LABELS) == len(_EDA_CARDS) + 1, "Etiquetas EDA desincronizadas"

    tabs = st.tabs(list(EDA_TAB_LABELS))
    n_cards = len(_EDA_CARDS)
    for tab, card in zip(tabs[:n_cards], _EDA_CARDS, strict=True):
        with tab:
            render_card(card, PAPER_FIGURES_ROOT)
    with tabs[n_cards]:
        render_spatial_tab(ROIS_YAML, PASTIS_METADATA)
