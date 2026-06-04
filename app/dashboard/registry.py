"""Data-driven registry of dashboard sections.

A single source of truth (``SECTIONS``) feeds the top-level selector, the
sidebar navigation and the ``main()`` dispatch. Adding an Avance reduces to
creating its content module + renderer and adding an entry to ``SECTIONS``;
neither the selector nor the sidebar nor ``main()`` require editing
(Open/Closed principle).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import streamlit as st

from app.dashboard.sections.baseline import BASELINE_TAB_LABELS, render_baseline_section
from app.dashboard.sections.eda import EDA_TAB_LABELS, render_eda_section
from app.dashboard.sections.feature_engineering import FE_TAB_LABELS, render_fe_section
from app.dashboard.sections.segmentation import render_segmentation_section
from app.dashboard.timeline import TIMELINE_TAB_LABELS, render_timeline_section

# Labels of the Avance 4 tabs (one consolidated sheet per architecture).
_A4_TAB_LABELS: tuple[str, ...] = (
    "Comparativa",
    "TSViT-pheno",
    "U-TAE",
    "Baselines densos",
    "Optuna",
)

# st.session_state key that preserves the selected section.
SECTION_STATE_KEY = "dashboard_section"


@dataclass(frozen=True)
class Section:
    """Navigable section of the dashboard.

    Attributes:
        key: Stable identifier (e.g. ``"a1-eda"``).
        label: Text of the selector and the sidebar.
        renderer: Argument-less function that renders the section.
        tab_labels: Tab labels for the sidebar navigation.
    """

    key: str
    label: str
    renderer: Callable[[], None]
    tab_labels: tuple[str, ...] = field(default_factory=tuple)


SECTIONS: tuple[Section, ...] = (
    Section("historia", "Historia del proyecto", render_timeline_section, TIMELINE_TAB_LABELS),
    Section("a1-eda", "Exploracion de Datos (EDA)", render_eda_section, EDA_TAB_LABELS),
    Section("a2-fe", "Ingenieria de Caracteristicas (FE)", render_fe_section, FE_TAB_LABELS),
    Section("a3-baseline", "Baseline (A3)", render_baseline_section, BASELINE_TAB_LABELS),
    Section("a4-seg", "Segmentacion (A4)", render_segmentation_section, _A4_TAB_LABELS),
)

# Legacy compatibility constants for tests that reference them.
_SECTION_HISTORIA = SECTIONS[0].label
_SECTION_EDA = SECTIONS[1].label
_SECTION_FE = SECTIONS[2].label
_SECTION_BASELINE = SECTIONS[3].label
_SECTION_SEGMENTATION = SECTIONS[4].label
_SECTION_OPTIONS: tuple[str, ...] = tuple(section.label for section in SECTIONS)

_LABEL_TO_SECTION: dict[str, Section] = {section.label: section for section in SECTIONS}


def render_section_selector(sections: Sequence[Section]) -> Section:
    """Render the section selector and return the active section.

    Args:
        sections: Available sections (presentation order).

    Returns:
        The selected ``Section``. The value is preserved across re-renders via
        ``st.session_state``; if the user deselects, the last valid section is
        kept (default: the first).
    """
    if SECTION_STATE_KEY not in st.session_state:
        st.session_state[SECTION_STATE_KEY] = sections[0].label

    selected_label = st.segmented_control(
        "Seccion del reporte",
        options=[section.label for section in sections],
        key=SECTION_STATE_KEY,
        label_visibility="collapsed",
    )
    if selected_label is None:
        selected_label = st.session_state.get(SECTION_STATE_KEY) or sections[0].label
    return _LABEL_TO_SECTION.get(selected_label, sections[0])
