"""US-030 closure: real apples-to-apples re-score of the 6 segmentation checkpoints.

Runs ``rescore_all_checkpoints()`` over the full held-out PASTIS fold-5 with the
unified 18-class schema, persists the comparison table (CSV) and the comparison
figure, and prints the resulting table. Invoked once for the US-030 closure on
real data (RTX 4070 local); not part of the test suite.

Usage (from repo root):
    poetry run python scripts/run_us030_rescore_fold5.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import structlog

from ml.eval.comparison import build_fold5_comparison_table, fold5_barplot_figure
from ml.eval.dense_metrics import rescore_all_checkpoints

logger = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_METRICS_DIR = _REPO_ROOT / "reports" / "segmentation" / "metrics"
_FIGURES_DIR = _REPO_ROOT / "reports" / "segmentation" / "figures"


def main() -> int:
    """Run the full fold-5 re-score and persist CSV + figure. Returns exit code."""
    logger.info("us030_rescore_start", fold=5)
    df = rescore_all_checkpoints(fold=5, device="auto", skip_missing=True)

    _METRICS_DIR.mkdir(parents=True, exist_ok=True)
    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = build_fold5_comparison_table(df, _METRICS_DIR)
    fig = fold5_barplot_figure(df)
    fig_path = _FIGURES_DIR / "barplot_comparison_fold5.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")

    n_ok = df.filter(df["status"] == "ok").height
    logger.info(
        "us030_rescore_done",
        csv=str(csv_path),
        figure=str(fig_path),
        models_ok=n_ok,
        models_total=df.height,
    )
    # Print the table as CSV (UTF-8 safe, avoids Windows cp1252 issues).
    sys.stdout.buffer.write(df.write_csv().encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
