"""Avance 4 section - dense semantic segmentation.

Renders the consolidated Avance 4 card: comparative table of the six
architectures with real metrics, per-architecture figures (predictions,
confusion matrix, per-class IoU and curves) with narrative, and the Optuna
convergence. The figures live in ``reports/segmentation/figures/`` (not in
``paper/figures/``), so they are resolved by direct name.
"""

from __future__ import annotations

import streamlit as st

from app.dashboard.components import (
    render_card_conclusions,
    render_card_header,
    render_figure_with_narrative,
    render_optional_figure,
    render_section_divider,
)
from app.dashboard.paths import SEGMENTATION_FIGURES_ROOT
from ml.report.avance4_content import (
    SEGMENTATION_CARD,
    SEGMENTATION_FIGURE_KINDS,
    SEGMENTATION_MODELS,
)
from ml.report.figure_narratives import get_narrative

# Hint shown when segmentation figures are missing on disk.
_MISSING_HINT = "Artefacto pendiente - ejecuta el notebook Avance4.Equipo17.ipynb"


def _render_comparison_table() -> None:
    """Render the comparative table of the 6 architectures with real metrics."""
    badge = f"{len(SEGMENTATION_MODELS)} modelos"
    render_section_divider("Comparativa de arquitecturas", badge=badge)
    rows = []
    for model in SEGMENTATION_MODELS:
        pixel = f"{model.pixel_accuracy:.3f}".replace(".", ",") if model.pixel_accuracy else "-"
        rows.append(
            {
                "Arquitectura": model.name,
                "mIoU": f"{model.miou:.4f}".replace(".", ","),
                "F1-macro": f"{model.f1_macro:.4f}".replace(".", ","),
                "Pixel-acc": pixel,
                "Nota": model.note,
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption(
        "Metricas flat (18 clases) sobre el split de validacion espacial. "
        "Fuente: reports/segmentation/metrics/model_comparison_avance4_*.parquet."
    )


def _render_model_figures() -> None:
    """Render, per architecture, the available figures with narrative."""
    for model in SEGMENTATION_MODELS:
        available = [
            (kind, caption, SEGMENTATION_FIGURES_ROOT / f"{kind}_{model.slug}.png")
            for kind, caption in SEGMENTATION_FIGURE_KINDS
        ]
        present = [(kind, caption, path) for kind, caption, path in available if path.exists()]
        if not present:
            continue
        render_section_divider(model.name, badge=f"{len(present)} figuras")
        for _kind, _caption, path in present:
            narrative = get_narrative(SEGMENTATION_CARD.notebook_id, path.name)
            render_figure_with_narrative(path, narrative)


def render_segmentation_section() -> None:
    """Render the Avance 4 block: card + comparative table + figures."""
    render_card_header(SEGMENTATION_CARD)
    _render_comparison_table()
    _render_model_figures()

    render_section_divider("Ajuste fino con Optuna")
    optuna_path = SEGMENTATION_FIGURES_ROOT / "optuna_convergence.png"
    narrative = get_narrative(SEGMENTATION_CARD.notebook_id, optuna_path.name)
    if optuna_path.exists() and narrative is not None:
        render_figure_with_narrative(optuna_path, narrative)
    else:
        render_optional_figure(
            optuna_path,
            "Convergencia del estudio Optuna sobre los top-2.",
            _MISSING_HINT,
        )

    render_card_conclusions(SEGMENTATION_CARD)
