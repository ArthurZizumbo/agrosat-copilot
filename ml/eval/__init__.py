"""Paquete de evaluacion: metricas y artefactos del baseline (US-019+)."""

from __future__ import annotations

from ml.eval.metrics import (
    classification_report_text,
    compute_baseline_metrics,
    confusion_matrix_figure,
)

__all__ = [
    "classification_report_text",
    "compute_baseline_metrics",
    "confusion_matrix_figure",
]
