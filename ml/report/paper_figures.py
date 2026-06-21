"""Reproducible paper figures for AgroSatCopilot (US-070, EPIC 11).

Single matplotlib template (scientific CVPR/ISPRS style: serif font, 300 DPI,
column-width sizing) plus the SVG+PNG exporter used by every paper figure. The
module follows two DRY principles:

1. **Style lives once** (:func:`set_paper_style`): rcParams + a fixed seed so any
   figure is byte-reproducible.
2. **Plots come from real artifacts**: figures are either *recomposed* from the
   numeric CSV/JSON under ``reports/**`` (barplots, sweep curves, transfer
   deltas) or *promoted* (copied + re-exported as SVG/PNG) from an already
   generated PNG (UMAP, confusion matrices, training curves, spatial residuals).
   There is **no fabricated data**: a figure whose source is missing is skipped
   and recorded in ``docs/blockers/epic11-notas.md``.

Captions/labels carry the project's factual corrections: AlphaEarth =
``SATELLITE_EMBEDDING/V1/ANNUAL`` v1.1 (not "v2.1"), SegFormer = B0 RGB 3-band,
AnySat substitutes the never-trained Swin-UNETR, Gemini 2.5 Pro is a frozen
reasoner (Be My Eyes pattern). Prose visible to the reader is in Spanish;
identifiers and docstrings are in English per the language policy.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "FIGURES_DIR",
    "PAPER_SEED",
    "REPORTS_DIR",
    "build_all_figures",
    "fig_benchmark_barplot",
    "fig_farslip_sweep_curve",
    "fig_llm_benchmark_barplot",
    "fig_transfer_catalonia",
    "promote_png",
    "save_fig_svg_png",
    "set_paper_style",
]

#: Fixed seed for any stochastic step (kept for reproducibility even though the
#: current figures are deterministic reads).
PAPER_SEED = 17

#: Repository ``reports/`` root (figure data sources).
REPORTS_DIR = Path("reports")

#: Default output directory for paper figures.
FIGURES_DIR = Path("paper/figures/us-070")


def set_paper_style() -> None:
    """Apply the scientific (CVPR/ISPRS) matplotlib style and fix the seed.

    Sets serif fonts, 300 DPI, tight column-width defaults and a deterministic
    seed for NumPy. Idempotent; safe to call at the top of every figure.
    """
    np.random.seed(PAPER_SEED)
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.constrained_layout.use": True,
        }
    )


def save_fig_svg_png(
    fig: plt.Figure, stem: str, *, out_dir: Path = FIGURES_DIR
) -> dict[str, Path]:
    """Export a figure as both SVG (vector) and PNG (300 DPI raster).

    Args:
        fig: Figure to export.
        stem: File stem (without extension).
        out_dir: Destination directory (created if missing).

    Returns:
        Mapping ``{"svg": path, "png": path}``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    svg = out_dir / f"{stem}.svg"
    png = out_dir / f"{stem}.png"
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("paper_figure_saved", stem=stem, svg=str(svg), png=str(png))
    return {"svg": svg, "png": png}


def promote_png(
    source_png: Path, stem: str, *, out_dir: Path = FIGURES_DIR
) -> dict[str, Path] | None:
    """Promote an already generated PNG into the paper figure set.

    Re-renders the existing raster through the paper style frame and exports it
    as SVG+PNG so the paper figure carries a consistent border/DPI. Returns
    ``None`` if the source artifact does not exist (a blocked figure) -- the plot
    is never fabricated.

    Args:
        source_png: Existing PNG under ``reports/**``.
        stem: Output file stem.
        out_dir: Destination directory.

    Returns:
        ``save_fig_svg_png`` mapping, or ``None`` if source is missing.
    """
    if not source_png.exists():
        logger.warning("paper_figure_source_missing", stem=stem, source=str(source_png))
        return None
    set_paper_style()
    img = mpimg.imread(source_png)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.imshow(img)
    ax.axis("off")
    ax.grid(False)
    return save_fig_svg_png(fig, stem, out_dir=out_dir)


# --------------------------------------------------------------------------- #
# F7-seg -- benchmark barplot recomposed from fold-5 metrics
# --------------------------------------------------------------------------- #
def fig_benchmark_barplot(
    metrics_csv: Path = REPORTS_DIR
    / "segmentation"
    / "metrics"
    / "model_comparison_fold5.csv",
    *,
    out_dir: Path = FIGURES_DIR,
) -> dict[str, Path] | None:
    """Recompose the EPIC 5 benchmark barplot from fold-5 mIoU/F1.

    Reads the re-scored fold-5 metrics and draws a grouped bar chart (mIoU and
    F1-macro) per model, sorted by mIoU. TSViT-pheno is the top individual.

    Args:
        metrics_csv: ``model_comparison_fold5.csv`` source.
        out_dir: Destination directory.

    Returns:
        ``save_fig_svg_png`` mapping, or ``None`` if source is missing.
    """
    if not metrics_csv.exists():
        logger.warning("paper_figure_source_missing", stem="benchmark_barplot")
        return None
    df = pl.read_csv(metrics_csv).sort("miou", descending=True)
    set_paper_style()
    models = df["model"].to_list()
    miou = df["miou"].to_list()
    f1 = df["f1_macro"].to_list()
    x = np.arange(len(models))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.bar(x - width / 2, miou, width, label="mIoU", color="#2c6fbb")
    ax.bar(x + width / 2, f1, width, label="F1-macro", color="#e08214")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Modelos EPIC 5 (re-score fold-5, harness US-030, 18 clases)")
    ax.legend()
    return save_fig_svg_png(fig, "benchmark_barplot_fold5", out_dir=out_dir)


# --------------------------------------------------------------------------- #
# Fx -- FarSLIP parcel cardinality sweep curve
# --------------------------------------------------------------------------- #
def fig_farslip_sweep_curve(
    sweep_csv: Path = REPORTS_DIR / "farslip" / "metrics" / "parcel_sweep.csv",
    *,
    out_dir: Path = FIGURES_DIR,
) -> dict[str, Path] | None:
    """Recompose the FarSLIP parcel cardinality sweep curve.

    Reads ``parcel_sweep.csv`` and plots macro-F1 / macro-IoU vs the number of
    classes. Shows the difficulty growth as the label space widens.

    Args:
        sweep_csv: ``parcel_sweep.csv`` source.
        out_dir: Destination directory.

    Returns:
        ``save_fig_svg_png`` mapping, or ``None`` if source is missing.
    """
    if not sweep_csv.exists():
        logger.warning("paper_figure_source_missing", stem="farslip_sweep")
        return None
    df = pl.read_csv(sweep_csv).sort("n_classes")
    set_paper_style()
    n = df["n_classes"].to_list()
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(n, df["macro_f1"].to_list(), "o-", label="macro-F1", color="#2c6fbb")
    ax.plot(n, df["macro_iou"].to_list(), "s--", label="macro-IoU", color="#e08214")
    ax.set_xlabel("Numero de clases")
    ax.set_ylabel("Score")
    ax.set_title("Ablacion FarSLIP: cardinalidad de clases (nivel parcela)")
    ax.legend()
    return save_fig_svg_png(fig, "farslip_sweep_curve", out_dir=out_dir)


# --------------------------------------------------------------------------- #
# F (transfer) -- FR -> Catalonia transfer delta
# --------------------------------------------------------------------------- #
def fig_transfer_catalonia(
    transfer_json: Path = REPORTS_DIR / "segmentation" / "sen4agrinet_transfer_result.json",
    *,
    out_dir: Path = FIGURES_DIR,
) -> dict[str, Path] | None:
    """Plot the FR->Catalonia transfer delta (zero-shot vs few-shot).

    Reads ``sen4agrinet_transfer_result.json`` (US-075) and draws the zero-shot
    vs few-shot mIoU/F1/pixel-accuracy bars, evidencing the limited spatial
    transferability that few-shot fine-tuning recovers.

    Args:
        transfer_json: Sen4AgriNet transfer result source.
        out_dir: Destination directory.

    Returns:
        ``save_fig_svg_png`` mapping, or ``None`` if source is missing.
    """
    if not transfer_json.exists():
        logger.warning("paper_figure_source_missing", stem="transfer_catalonia")
        return None
    data = json.loads(transfer_json.read_text(encoding="utf-8"))
    zs = data["zero_shot_metrics"]
    fs = data["few_shot_metrics"]
    set_paper_style()
    labels = ["mIoU", "F1-macro", "Pix-acc"]
    zero = [zs["miou"], zs["f1_macro"], zs["pixel_accuracy"]]
    few = [fs["miou"], fs["f1_macro"], fs["pixel_accuracy"]]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.bar(x - width / 2, zero, width, label="zero-shot", color="#9e9e9e")
    ax.bar(x + width / 2, few, width, label="few-shot (k-shot FT)", color="#2c6fbb")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Transferencia FR->Cataluna (Sen4AgriNet, US-075)")
    ax.legend()
    return save_fig_svg_png(fig, "transfer_fr_catalonia", out_dir=out_dir)


# --------------------------------------------------------------------------- #
# F7-LLM -- LLM benchmark barplot recomposed from us049 eval
# --------------------------------------------------------------------------- #
def fig_llm_benchmark_barplot(
    eval_json: Path = REPORTS_DIR / "agent_bench" / "us049_system_eval.json",
    *,
    out_dir: Path = FIGURES_DIR,
) -> dict[str, Path] | None:
    """Recompose the LLM benchmark barplot (Gemini vs Qwen) from real eval.

    Reads ``us049_system_eval.json`` and draws grouped bars for tool selection,
    argument match, crop grounding and routing for the models present.

    Args:
        eval_json: ``us049_system_eval.json`` source.
        out_dir: Destination directory.

    Returns:
        ``save_fig_svg_png`` mapping, or ``None`` if source is missing.
    """
    if not eval_json.exists():
        logger.warning("paper_figure_source_missing", stem="llm_benchmark")
        return None
    data = json.loads(eval_json.read_text(encoding="utf-8"))

    def _m(block: dict, sub: str, metric: str) -> float:
        entry = block.get(sub, {}).get(metric, {})
        val = entry.get("mean") if isinstance(entry, dict) else None
        return float(val) if val is not None and not _is_nan(val) else 0.0

    label_map = {"gemini": "Gemini 2.5 Pro", "qwen": "Qwen3.5-35B-A3B"}
    metrics = [
        ("Tool-sel", "tool_calling", "tool_selection_accuracy"),
        ("Arg-match", "tool_calling", "arg_match_accuracy"),
        ("Crop-match", "grounded_crop", "crop_match_accuracy"),
        ("Routing", "grounded_crop", "routing_accuracy"),
    ]
    present = [k for k in ("gemini", "qwen") if k in data]
    set_paper_style()
    x = np.arange(len(metrics))
    width = 0.8 / max(len(present), 1)
    colors = {"gemini": "#2c6fbb", "qwen": "#e08214"}
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for i, key in enumerate(present):
        vals = [_m(data[key], sub, metric) for _, sub, metric in metrics]
        ax.bar(
            x + (i - (len(present) - 1) / 2) * width,
            vals,
            width,
            label=label_map.get(key, key),
            color=colors.get(key, None),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics])
    ax.set_ylabel("Score")
    ax.set_title("Benchmark LLMs del copiloto (US-049, AgroMind-IT/ES pendiente)")
    ax.legend()
    return save_fig_svg_png(fig, "llm_benchmark_barplot", out_dir=out_dir)


def _is_nan(value: object) -> bool:
    """Return whether a value is a float NaN.

    Args:
        value: Any value.

    Returns:
        ``True`` if ``value`` is a float NaN.
    """
    return isinstance(value, float) and value != value


# --------------------------------------------------------------------------- #
# Promoted figures (existing PNGs re-exported as SVG+PNG)
# --------------------------------------------------------------------------- #
#: Mapping ``stem -> existing PNG`` for figures promoted as-is. Each is a real
#: artifact already generated by the pipeline. Public so notebooks can iterate it.
PROMOTED_FIGURES: dict[str, Path] = {
    "umap_alphaearth": REPORTS_DIR / "baseline" / "pheno_umap_no_coords.png",
    "curves_tsvit": REPORTS_DIR / "segmentation" / "figures" / "curves_tsvit.png",
    "confusion_tsvit": REPORTS_DIR / "segmentation" / "figures" / "confusion_tsvit.png",
    "confusion_stacking": REPORTS_DIR / "ensemble" / "figures" / "confusion_stacking.png",
    "spatial_residuals": REPORTS_DIR
    / "ensemble"
    / "figures"
    / "spatial_residuals_blending.png",
    "per_class_iou_tsvit": REPORTS_DIR
    / "segmentation"
    / "figures"
    / "per_class_iou_tsvit.png",
}


def export_conversational_examples(
    *,
    out_dir: Path = FIGURES_DIR,
    traces_dir: Path = REPORTS_DIR / "agent_bench" / "traces",
    n_examples: int = 2,
) -> Path | None:
    """Export real ES/EN conversational traces as a JSON snippet for F5.

    Reads the real agent traces (``trace_gemini_*.jsonl``) and dumps the first
    ``n_examples`` task/prompt/prediction triples to a JSON file the LaTeX
    listing consumes. The Italian variant depends on AgroMind-IT (US-068, native
    review) and is left pending -- no synthetic IT trace is fabricated (B-070-4).

    Args:
        out_dir: Destination directory.
        traces_dir: Directory with the real ``*.jsonl`` traces.
        n_examples: Number of examples to extract.

    Returns:
        Path to the written JSON, or ``None`` if no trace exists.
    """
    candidates = sorted(traces_dir.glob("trace_gemini_*.jsonl"))
    if not candidates:
        logger.warning("paper_figure_source_missing", stem="conversational_examples")
        return None
    examples: list[dict[str, str]] = []
    for trace in candidates:
        with trace.open(encoding="utf-8") as fh:
            for line in fh:
                if len(examples) >= n_examples:
                    break
                rec = json.loads(line)
                examples.append(
                    {
                        "benchmark": str(rec.get("benchmark", "")),
                        "lang": "es/en",
                        "task": str(rec.get("task", ""))[:200],
                        "prediction": str(rec.get("prediction", ""))[:400],
                    }
                )
        if len(examples) >= n_examples:
            break
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "conversational_examples.json"
    path.write_text(
        json.dumps(
            {"note_it": "pendiente AgroMind-IT US-068", "examples": examples},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("conversational_examples_written", path=str(path), n=len(examples))
    return path


def build_all_figures(out_dir: Path = FIGURES_DIR) -> dict[str, dict[str, Path] | None]:
    """Generate every paper figure whose real source exists.

    Recomposed figures (barplot, sweep, transfer, LLM bench) and promoted PNGs
    (UMAP, curves, confusion, residuals). Missing-source figures return ``None``
    and are logged + documented in ``docs/blockers/epic11-notas.md``; none is
    fabricated.

    Args:
        out_dir: Output directory for all figures.

    Returns:
        Mapping ``stem -> save mapping`` (``None`` for blocked figures).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Path] | None] = {}
    results["benchmark_barplot_fold5"] = fig_benchmark_barplot(out_dir=out_dir)
    results["farslip_sweep_curve"] = fig_farslip_sweep_curve(out_dir=out_dir)
    results["transfer_fr_catalonia"] = fig_transfer_catalonia(out_dir=out_dir)
    results["llm_benchmark_barplot"] = fig_llm_benchmark_barplot(out_dir=out_dir)
    for stem, source in PROMOTED_FIGURES.items():
        results[stem] = promote_png(source, stem, out_dir=out_dir)
    conv = export_conversational_examples(out_dir=out_dir)
    results["conversational_examples"] = {"json": conv} if conv else None
    return results


if __name__ == "__main__":  # pragma: no cover - manual entry point
    structlog.configure(processors=[structlog.dev.ConsoleRenderer()])
    produced = build_all_figures()
    for name, mapping in produced.items():
        logger.info("figure", name=name, produced=mapping is not None)
