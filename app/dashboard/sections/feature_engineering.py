"""Avance 2 section - Feature Engineering (FE).

Renders the four feature engineering cards (Sentinel-2, PASTIS-R
spectro-temporal, AlphaEarth fusion and conclusions). The editorial content
lives in ``ml.report.avance2_content`` (``FE_CARDS``).
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
    """Render the Feature Engineering block: 4 cards."""
    assert len(FE_TAB_LABELS) == len(FE_CARDS), "Etiquetas FE desincronizadas"

    tabs = st.tabs(list(FE_TAB_LABELS))
    for tab, card in zip(tabs, FE_CARDS, strict=True):
        with tab:
            render_card(card, PAPER_FIGURES_ROOT)
