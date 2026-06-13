"""Generate the real-data figures for the US-030..US-040 batch notebooks.

Runs the inference and writes every PNG (segmenter triptychs + confusion
matrices, FarSLIP faithful per-patch predictions with Gemma captions, the
ensemble comparison bar) under ``reports/lote_us030_040/figures/`` plus a
``manifest.json`` the notebooks read to lay out the figures with their captions.

Designed to run on the H100 VM (real PASTIS-R, real checkpoints, real Gemma
captions). It is idempotent: re-running overwrites the PNGs deterministically.

Usage (on the VM, env ``agrosat``)::

    python -m scripts.build_lote_figures \
        --pastis-root data/PASTIS-R \
        --out reports/lote_us030_040/figures \
        --segmenter-examples 3 --farslip-examples 6

Project conventions: Polars, structlog, type hints, English docstrings, Spanish
visible prose, no emojis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import matplotlib

matplotlib.use("Agg")  # headless on the VM
import matplotlib.pyplot as plt
import structlog
import typer

from ml.report import lote_figures as lf

logger = structlog.get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)


def _save(fig: Any, path: Path, *, dpi: int = 120) -> None:
    """Save a matplotlib figure and close it (frees memory on the VM)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


@app.callback()
def _root() -> None:
    """AgroSatCopilot: real-data figure builder for the US-030..040 notebooks."""


@app.command()
def run(
    pastis_root: Annotated[Path, typer.Option("--pastis-root")] = Path("data/PASTIS-R"),
    out: Annotated[Path, typer.Option("--out")] = lf.DEFAULT_FIGURES_DIR,
    segmenter_examples: Annotated[int, typer.Option("--segmenter-examples")] = 3,
    farslip_examples: Annotated[int, typer.Option("--farslip-examples")] = 6,
    max_patches: Annotated[
        int, typer.Option("--max-patches", help="Cap patches per CM (0=full)")
    ] = 0,
    device: Annotated[str, typer.Option("--device")] = "auto",
    do_segmenters: Annotated[bool, typer.Option("--segmenters/--no-segmenters")] = True,
    do_farslip: Annotated[bool, typer.Option("--farslip/--no-farslip")] = True,
    do_ensemble: Annotated[bool, typer.Option("--ensemble/--no-ensemble")] = True,
) -> None:
    """Build all batch figures and write the manifest under ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    cap = None if max_patches <= 0 else int(max_patches)
    # Merge with an existing manifest so partial runs (e.g. --no-segmenters)
    # preserve the sections produced by previous runs instead of wiping them.
    manifest: dict[str, Any] = {"segmenters": {}, "farslip": [], "ensemble": {}}
    existing_manifest = out / "manifest.json"
    if existing_manifest.is_file():
        try:
            prev = json.loads(existing_manifest.read_text(encoding="utf-8"))
            manifest.update(prev)
        except json.JSONDecodeError:
            logger.warning("manifest_unreadable_starting_fresh", path=str(existing_manifest))

    if do_segmenters:
        seg_indices = list(range(0, segmenter_examples * 4, 4))[:segmenter_examples]
        for kind in lf.SEGMENTER_KINDS:
            try:
                figs = lf.build_segmenter_triptychs(
                    kind, indices=seg_indices, pastis_root=pastis_root, device=device
                )
                trip_paths: list[str] = []
                for i, fig in enumerate(figs):
                    p = out / f"seg_{kind}_triptych_{i}.png"
                    _save(fig, p)
                    trip_paths.append(p.name)
                metrics, _cm, cm_fig = lf.build_segmenter_confusion(
                    kind, pastis_root=pastis_root, max_patches=cap, device=device
                )
                cm_path = out / f"seg_{kind}_confusion.png"
                _save(cm_fig, cm_path)
                per_class = lf.per_class_iou_table(metrics)
                pc_path = out / f"seg_{kind}_per_class.csv"
                per_class.write_csv(pc_path)
                manifest["segmenters"][kind] = {
                    "label": lf.SEGMENTER_LABELS.get(kind, kind),
                    "triptychs": trip_paths,
                    "confusion": cm_path.name,
                    "per_class_csv": pc_path.name,
                    "miou": round(float(metrics["miou"]), 4),
                    "f1_macro": round(float(metrics["f1_macro"]), 4),
                    "pixel_acc": round(float(metrics["pixel_acc"]), 4),
                }
                logger.info("segmenter_done", kind=kind, miou=metrics["miou"])
            except Exception as exc:  # noqa: BLE001 - report and continue per model
                logger.error("segmenter_failed", kind=kind, error=str(exc))
                manifest["segmenters"][kind] = {"error": str(exc)}

    if do_farslip:
        try:
            farslip_figs = lf.build_farslip_prediction_figures(
                n_examples=farslip_examples, pastis_root=pastis_root, device=device
            )
            for i, (fig, info) in enumerate(farslip_figs):
                p = out / f"farslip_pred_{i}.png"
                _save(fig, p)
                info["figure"] = p.name
                manifest["farslip"].append(info)
            logger.info("farslip_done", n=len(farslip_figs))
        except Exception as exc:  # noqa: BLE001 - report and continue
            logger.error("farslip_failed", error=str(exc))
            manifest["farslip_error"] = str(exc)

    if do_ensemble:
        try:
            fig, df = lf.build_ensemble_comparison_figure()
            p = out / "ensemble_comparison.png"
            _save(fig, p)
            manifest["ensemble"] = {
                "figure": p.name,
                "rows": df.to_dicts(),
            }
            logger.info("ensemble_done")
        except Exception as exc:  # noqa: BLE001 - report and continue
            logger.error("ensemble_failed", error=str(exc))
            manifest["ensemble"] = {"error": str(exc)}

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info("manifest_written", path=str(manifest_path))


if __name__ == "__main__":
    app()
