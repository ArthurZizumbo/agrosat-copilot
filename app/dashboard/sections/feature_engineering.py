"""Seccion Avance 2 - Ingenieria de Caracteristicas (FE).

Renderiza las cuatro fichas de feature engineering (Sentinel-2, PASTIS-R
espectro-temporal, fusion AlphaEarth y conclusiones). El contenido editorial
vive en ``ml.report.avance2_content`` (``FE_CARDS``).
"""

from __future__ import annotations

import streamlit as st

from app.dashboard.components import render_card
from app.dashboard.paths import PAPER_FIGURES_ROOT
from ml.report.avance2_content import FE_CARDS

FE_TAB_LABELS: tuple[str, ...] = (
    "Sentinel-2",
    "PASTIS-R Espectro-Temporal",
    "Fusion AlphaEarth",
    "Conclusiones",
)


def render_fe_section() -> None:
    """Renderiza el bloque de Ingenieria de Caracteristicas: 4 fichas."""
    assert len(FE_TAB_LABELS) == len(FE_CARDS), "Etiquetas FE desincronizadas"

    tabs = st.tabs(list(FE_TAB_LABELS))
    for tab, card in zip(tabs, FE_CARDS, strict=True):
        with tab:
            render_card(card, PAPER_FIGURES_ROOT)
