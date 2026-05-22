"""Paquete de evaluacion: metricas, interpretabilidad, curvas y comparativa (US-019+)."""

from __future__ import annotations

from ml.eval.comparison import (
    ComparisonResult,
    build_comparison_table,
    export_comparison_latex,
)
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
from ml.eval.learning_curves import (
    FitDiagnosis,
    LearningCurveResult,
    ValidationCurveResult,
    diagnose_fit,
    plot_learning_curve,
    plot_validation_curve,
)
from ml.eval.metrics import (
    classification_report_text,
    compute_baseline_metrics,
    confusion_matrix_figure,
)

__all__ = [
    "ComparisonResult",
    "FitDiagnosis",
    "LearningCurveResult",
    "ShapResult",
    "ValidationCurveResult",
    "alphaearth_dominance_table",
    "build_comparison_table",
    "classification_report_text",
    "compute_baseline_metrics",
    "compute_shap_values",
    "confusion_matrix_figure",
    "diagnose_fit",
    "export_comparison_latex",
    "feature_importance_table",
    "is_alphaearth_dim",
    "plot_learning_curve",
    "plot_validation_curve",
    "shap_dependence_plots",
    "shap_summary_plot",
    "shap_waterfall_plot",
]
