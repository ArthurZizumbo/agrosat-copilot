"""Build the WorldCereal tropical transfer figure (Experimento 3, EPIC 12).

Reads ONLY the real artefacts produced by ``ml.transfer.worldcereal_tropical``
(no hand-typed numbers) and emits a two-panel figure under
``paper/figures/us-073-transfer/``:

- Left panel: the few-shot F1-macro-vs-k curve over the LOCAL tropical
  WorldCereal classes for Brazil (Cerrado) and India (Karnataka), with per-seed
  error bars, plus a dashed line at the fully-supervised in-domain ceiling.
- Right panel: the honest zero-shot bar -- the European PASTIS-18 AlphaEarth
  classifier applied to each tropical region, scored only on the SINGLE shared
  concept (maize / Corn), versus the maize base-rate (the trivial reference).

Inputs (all REAL, versioned under ``data/transfer/``):
- ``worldcereal_brazil_cerrado.parquet`` / ``worldcereal_india_karnataka.parquet``
  (joined WorldCereal label + AlphaEarth 64-dim).
- ``worldcereal_fewshot_results.parquet`` (Brazil curve, 3 seeds).
- ``worldcereal_fewshot_india.parquet`` (India curve, 3 seeds).

The zero-shot and in-domain numbers are recomputed from the datasets at build
time so the figure can never drift from a stale cached scalar. Deterministic
(Agg backend, fixed seeds), idempotent. Spanish prose in the figure, English
identifiers, no emojis. If an input is missing the script raises explicitly and
fabricates nothing.

Usage::

    python -m scripts.build_worldcereal_tropical_figure
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl
import structlog

from ml.transfer.worldcereal_tropical import (
    summarize_curve,
    zero_shot_europe_to_tropics,
    zero_shot_separability,
)

logger = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRANSFER = _REPO_ROOT / "data" / "transfer"
_OUT_DIR = _REPO_ROOT / "paper" / "figures" / "us-073-transfer"

#: The two tropical regions plotted (label, dataset parquet, curve parquet, colour).
_REGIONS: tuple[tuple[str, str, str, str], ...] = (
    (
        "Brasil (Cerrado)",
        "worldcereal_brazil_cerrado.parquet",
        "worldcereal_fewshot_results.parquet",
        "#1b7837",
    ),
    (
        "India (Karnataka)",
        "worldcereal_india_karnataka.parquet",
        "worldcereal_fewshot_india.parquet",
        "#762a83",
    ),
)


def _require(path: Path) -> Path:
    """Return ``path`` or raise if it does not exist (no fabricated inputs).

    Args:
        path: Expected input file.

    Returns:
        The same path when it exists.

    Raises:
        FileNotFoundError: if the input is absent.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Required real artefact missing: {path}. Run "
            "ml.transfer.worldcereal_tropical.build_dataset first."
        )
    return path


def build_figure(*, dpi: int = 150) -> tuple[Path, Path]:
    """Render the two-panel WorldCereal tropical transfer figure.

    Args:
        dpi: Raster resolution for the PNG output.

    Returns:
        Tuple ``(png_path, svg_path)`` of the written figure files.

    Raises:
        FileNotFoundError: if any required input parquet is absent.
    """
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax_curve, ax_zs) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    zs_labels: list[str] = []
    zs_f1: list[float] = []
    zs_base: list[float] = []
    zs_colours: list[str] = []

    for label, ds_name, curve_name, colour in _REGIONS:
        dataset = pl.read_parquet(_require(_TRANSFER / ds_name))
        curve = pl.read_parquet(_require(_TRANSFER / curve_name))
        summary = summarize_curve(curve).sort("k")

        ks = summary.get_column("k").to_list()
        means = summary.get_column("f1_mean").to_list()
        stds = summary.get_column("f1_std").to_list()
        ax_curve.errorbar(
            ks,
            means,
            yerr=stds,
            marker="o",
            capsize=3,
            color=colour,
            label=label,
        )

        ceiling = zero_shot_separability(dataset)["f1_macro_cv"]
        ax_curve.axhline(
            ceiling, ls="--", lw=1.0, color=colour, alpha=0.6
        )

        zs = zero_shot_europe_to_tropics(dataset)
        zs_labels.append(label)
        zs_f1.append(zs["maize_f1_zero_shot"])
        zs_base.append(zs["base_rate"])
        zs_colours.append(colour)
        logger.info(
            "worldcereal_figure_region",
            region=label,
            ceiling=round(ceiling, 4),
            zero_shot_maize_f1=round(zs["maize_f1_zero_shot"], 4),
        )

    ax_curve.set_xscale("log")
    ax_curve.set_xlabel("Muestras locales por clase (k, escala log)")
    ax_curve.set_ylabel("F1-macro (clases tropicales locales)")
    ax_curve.set_title(
        "Few-shot sobre clases WorldCereal tropicales\n"
        "(linea discontinua = techo supervisado in-domain)"
    )
    ax_curve.set_ylim(0.0, 0.85)
    ax_curve.grid(True, alpha=0.3)
    ax_curve.legend(loc="lower right")

    # Right panel: zero-shot maize-detection bars vs base rate.
    x = range(len(zs_labels))
    width = 0.38
    ax_zs.bar(
        [i - width / 2 for i in x],
        zs_f1,
        width=width,
        color=zs_colours,
        label="F1 zero-shot (Europa -> tropico, solo maiz)",
    )
    ax_zs.bar(
        [i + width / 2 for i in x],
        zs_base,
        width=width,
        color="#bbbbbb",
        label="Tasa base de maiz (referencia trivial)",
    )
    ax_zs.set_xticks(list(x))
    ax_zs.set_xticklabels(zs_labels)
    ax_zs.set_ylabel("F1 binario de deteccion de maiz")
    ax_zs.set_ylim(0.0, 0.6)
    ax_zs.set_title(
        "Zero-shot: clasificador europeo PASTIS-18\n"
        "aplicado al tropico (unica clase compartida: maiz)"
    )
    ax_zs.grid(True, axis="y", alpha=0.3)
    ax_zs.legend(loc="upper left", fontsize=8)
    for i, v in enumerate(zs_f1):
        ax_zs.text(i - width / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    fig.suptitle(
        "Experimento 3 -- Transfer multi-region a zona tropical (ESA WorldCereal, CC-BY-4.0)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    png = _OUT_DIR / "worldcereal_tropical_transfer.png"
    svg = _OUT_DIR / "worldcereal_tropical_transfer.svg"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    logger.info("worldcereal_figure_saved", png=str(png), svg=str(svg))
    return png, svg


if __name__ == "__main__":
    build_figure()
