"""Paquete de evaluacion: metricas e interpretabilidad del baseline (US-019+)."""

from __future__ import annotations

from ml.eval.interpretability import (
    ShapResult,
    alphaearth_dominance_table,
    compute_shap_values,
    feature_importance_table,
    is_alphaearth_dim,
    shap_dependence_plots,
    shap_summary_plot,
    shap_waterfall_plot,
)
from ml.eval.metrics import (
    classification_report_text,
    compute_baseline_metrics,
    confusion_matrix_figure,
)

__all__ = [
    "ShapResult",
    "alphaearth_dominance_table",
    "classification_report_text",
    "compute_baseline_metrics",
    "compute_shap_values",
    "confusion_matrix_figure",
    "feature_importance_table",
    "is_alphaearth_dim",
    "shap_dependence_plots",
    "shap_summary_plot",
    "shap_waterfall_plot",
]
