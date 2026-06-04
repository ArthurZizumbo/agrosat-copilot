"""Avance 3 section - tabular and phenological baseline (data-driven).

Iterates over ``A3_TABS`` (defined as data in ``ml.report.avance3_content``)
with a single generic renderer: for each tab it shows the editorial card and
resolves its artifacts (figure or parquet table), trying fallback paths in
order and degrading with ``st.warning`` when none exists. Adding or changing a
tab does not require writing new render code.
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
    """Return the first existing path among relpath and fallbacks.

    Args:
        artifact: Artifact descriptor with main path and fallbacks.

    Returns:
        Absolute Path to the first existing path; if none exists, returns the
        main path (the renderer will show the warning).
    """
    for candidate in (artifact.relpath, *artifact.fallbacks):
        path = REPO_ROOT / candidate
        if path.exists():
            return path
    return REPO_ROOT / artifact.relpath


def _render_artifact(artifact: BaselineArtifact) -> None:
    """Render an artifact (figure or table) resolving its path."""
    path = _resolve_artifact_path(artifact)
    if artifact.kind == "figure":
        render_optional_figure(path, artifact.caption, BASELINE_MISSING_HINT)
    else:
        render_parquet_table(path, artifact.caption, BASELINE_MISSING_HINT)


def _render_baseline_tab(tab: BaselineTab, index: int, total: int) -> None:
    """Render an Avance 3 tab: divider + card + artifacts."""
    render_section_divider(tab.card.title, badge=f"Tab {index} de {total}")
    render_card_header(tab.card)
    for artifact in tab.artifacts:
        _render_artifact(artifact)
    render_card_conclusions(tab.card)


# Per-tab renderers (closures over each BaselineTab) for compatibility with
# the tests that count ``_BASELINE_TAB_RENDERERS``.
def _make_tab_renderer(tab: BaselineTab, index: int, total: int):
    """Create an argument-less renderer for a given tab."""

    def _renderer() -> None:
        _render_baseline_tab(tab, index, total)

    return _renderer


BASELINE_TAB_RENDERERS: tuple = tuple(
    _make_tab_renderer(tab, i + 1, len(A3_TABS)) for i, tab in enumerate(A3_TABS)
)


def render_baseline_section() -> None:
    """Render the Baseline (A3) section with its data-driven tabs."""
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
