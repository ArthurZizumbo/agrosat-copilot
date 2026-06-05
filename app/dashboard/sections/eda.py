"""Avance 1 section - Exploratory Data Analysis (EDA).

Renders the EDA cards (six consolidated notebooks) plus the spatial folium
map as the final tab. The editorial content lives in
``ml.report.notebook_content`` (``CARDS`` + ``CARDS_EXTRA``).
"""

from __future__ import annotations

import streamlit as st

from app.dashboard.components import render_card
from app.dashboard.paths import PAPER_FIGURES_ROOT, PASTIS_METADATA, ROIS_YAML
from app.dashboard.spatial import render_spatial_tab
from ml.report.notebook_content import EDA_DISPLAY_CARDS

# EDA cards in presentation order (the 7 cards; the map is a separate tab).
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
    """Render the EDA block: 7 content cards + spatial map."""
    assert len(EDA_TAB_LABELS) == len(_EDA_CARDS) + 1, "Etiquetas EDA desincronizadas"

    tabs = st.tabs(list(EDA_TAB_LABELS))
    n_cards = len(_EDA_CARDS)
    for tab, card in zip(tabs[:n_cards], _EDA_CARDS, strict=True):
        with tab:
            render_card(card, PAPER_FIGURES_ROOT)
    with tabs[n_cards]:
        render_spatial_tab(ROIS_YAML, PASTIS_METADATA)
