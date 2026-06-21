"""Data and prediction drift monitoring for AgroSatCopilot (US-060).

Reusable, framework-free drift detection pipeline built on Evidently 0.7.x
(modern ``Report`` / ``Dataset`` / ``DataDefinition`` API). The Dagster asset
``drift_check`` (``dagster_project/assets/drift.py``) orchestrates this module
on a weekly schedule; the functions here are pure and unit-tested in isolation.

Drift families (plan v8 §US-060):

- Sentinel-2 spectral bands / indices: Kolmogorov-Smirnov (KS) two-sample test.
- AlphaEarth embeddings (``SATELLITE_EMBEDDING/V1/ANNUAL`` v1.1, 64-dim,
  CC-BY-4.0) and FarSLIP embeddings: Maximum Mean Discrepancy (MMD) via
  Evidently's ``EmbeddingsDrift`` metric.
- Predicted classes (18-class contiguous space from US-030, or HCAT macro from
  US-074 in the multi-region pipeline): Chi-squared categorical drift test.
"""

from ml.monitoring.drift import (
    DRIFT_SCORE_THRESHOLD,
    DriftSummary,
    build_drift_report,
    embedding_columns,
    exceeds_threshold,
    extract_drift_score,
)

__all__ = [
    "DRIFT_SCORE_THRESHOLD",
    "DriftSummary",
    "build_drift_report",
    "embedding_columns",
    "exceeds_threshold",
    "extract_drift_score",
]
