"""Seccion Avance 3 - Baseline tabular y fenologico (data-driven).

Recorre ``A3_TABS`` (definidos como datos en ``ml.report.avance3_content``) con
un unico renderer generico: por cada tab muestra la ficha editorial y resuelve
sus artefactos (figura o tabla parquet), probando rutas de fallback en orden y
degradando con ``st.warning`` cuando ninguna existe. Agregar o cambiar un tab
no implica escribir codigo de render nuevo.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.dashboard.components import (
    render_card_conclusions,
    render_card_header,
    render_optional_figure,
    render_parquet_table,
    render_section_divider,
)
from app.dashboard.paths import REPO_ROOT
from ml.report.avance3_content import (
    A3_TABS,
    BASELINE_MISSING_HINT,
    BaselineArtifact,
    BaselineTab,
)

# Labels and renderers derived from A3_TABS for test compatibility.
BASELINE_TAB_LABELS: tuple[str, ...] = tuple(tab.label for tab in A3_TABS)


def _resolve_artifact_path(artifact: BaselineArtifact) -> Path:
    """Devuelve la primera ruta existente entre relpath y fallbacks.

    Args:
        artifact: Descriptor del artefacto con ruta principal y fallbacks.

    Returns:
        Path absoluto a la primera ruta existente; si ninguna existe,
        devuelve el path principal (el renderer mostrara el warning).
    """
    for candidate in (artifact.relpath, *artifact.fallbacks):
        path = REPO_ROOT / candidate
        if path.exists():
            return path
    return REPO_ROOT / artifact.relpath


def _render_artifact(artifact: BaselineArtifact) -> None:
    """Renderiza un artefacto (figura o tabla) resolviendo su ruta."""
    path = _resolve_artifact_path(artifact)
    if artifact.kind == "figure":
        render_optional_figure(path, artifact.caption, BASELINE_MISSING_HINT)
    else:
        render_parquet_table(path, artifact.caption, BASELINE_MISSING_HINT)


def _render_baseline_tab(tab: BaselineTab, index: int, total: int) -> None:
    """Renderiza un tab del Avance 3: divisor + ficha + artefactos."""
    render_section_divider(tab.card.title, badge=f"Tab {index} de {total}")
    render_card_header(tab.card)
    for artifact in tab.artifacts:
        _render_artifact(artifact)
    render_card_conclusions(tab.card)


# Per-tab renderers (closures over each BaselineTab) for compatibility with
# the tests that count ``_BASELINE_TAB_RENDERERS``.
def _make_tab_renderer(tab: BaselineTab, index: int, total: int):
    """Crea un renderer sin argumentos para un tab dado."""

    def _renderer() -> None:
        _render_baseline_tab(tab, index, total)

    return _renderer


BASELINE_TAB_RENDERERS: tuple = tuple(
    _make_tab_renderer(tab, i + 1, len(A3_TABS)) for i, tab in enumerate(A3_TABS)
)


def render_baseline_section() -> None:
    """Renderiza la seccion Baseline (A3) con sus tabs data-driven."""
    st.markdown(
        '<h2 style="margin-top:0.5rem;color:#1E293B;font-weight:700;">'
        "Baseline saneado post-A3 (Avance 3)</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#475569;font-size:1rem;line-height:1.6;">'
        "Reporte consolidado del refinamiento del baseline tras el Avance 3: "
        "ablation de features, evidencia visual del leakage geografico, "
        "evaluacion de bloques opcionales (FarSLIP, Gemini Flash 3.5 y firma "
        "espectral) y reentrenamiento de los modelos canonicos sobre el "
        "conjunto ganador. Esta seccion alimenta el arranque del Avance 4 con "
        "baseline cuantificado y reproducible.</p>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs(list(BASELINE_TAB_LABELS))
    for tab_ui, renderer in zip(tabs, BASELINE_TAB_RENDERERS, strict=True):
        with tab_ui:
            renderer()
