"""Canonical paths to repository artifacts.

Centralizes the paths to figures, reports and configuration consumed by the
dashboard sections, relative to the repository root. For PASTIS-R the compact
dissolved subset is preferred (committed to the repo, works on Streamlit Cloud);
if it does not exist it falls back to the full metadata available only on
machines with DVC synced.
"""

from __future__ import annotations

from pathlib import Path

# app/dashboard/paths.py -> repo root (three levels up).
REPO_ROOT = Path(__file__).resolve().parents[2]

PAPER_FIGURES_ROOT = REPO_ROOT / "paper" / "figures"
REPORTS_ROOT = REPO_ROOT / "reports"
ROIS_YAML = REPO_ROOT / "config" / "rois.yaml"

# Segmentation (Avance 4): figures and metrics in reports/segmentation/.
SEGMENTATION_FIGURES_ROOT = REPORTS_ROOT / "segmentation" / "figures"
SEGMENTATION_METRICS_ROOT = REPORTS_ROOT / "segmentation" / "metrics"

# Baseline (Avance 3 / US-023-preview).
BASELINE_FIGURES_DIR = PAPER_FIGURES_ROOT / "us-023-preview"
BASELINE_ABLATION_DIR = REPORTS_ROOT / "baseline" / "feature_ablation"
BASELINE_REENCUADRE_DIR = REPORTS_ROOT / "baseline" / "reencuadre_fenologico"
BASELINE_MODEL_COMP_V2_DIR = REPORTS_ROOT / "baseline" / "model_comparison_v2"
BASELINE_A3_DIR = REPORTS_ROOT / "baseline" / "Avance3"

_PASTIS_METADATA_COMPACT = REPO_ROOT / "data" / "reference" / "pastis_tiles_dissolved.geojson"
_PASTIS_METADATA_FULL = REPO_ROOT / "data" / "PASTIS-R" / "metadata.geojson"
PASTIS_METADATA = (
    _PASTIS_METADATA_COMPACT if _PASTIS_METADATA_COMPACT.exists() else _PASTIS_METADATA_FULL
)
