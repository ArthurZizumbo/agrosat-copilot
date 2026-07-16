"""Build the 4 reproducible paper notebooks for US-070 (permanent Typer CLI).

Reusable permanent operational tool: regenerates the ``paper/notebooks/*.ipynb``
from a declarative cell definition so the canonical structure stays consistent.
The notebooks **only orchestrate** the ``ml.report.paper_figures`` /
``ml.report.paper_tables`` modules over real artifacts under ``reports/**`` --
they do not reimplement any plot or hardcode any number. Blocked figures/tables
(H100/GEE/human review) print their blocker id and skip, never fabricating data.

Execution (populating outputs) is done afterwards with papermill (target
``make paper-figures``):

    MPLBACKEND=Agg poetry run papermill paper/notebooks/01_figures_segmentation.ipynb \\
        paper/notebooks/01_figures_segmentation.ipynb --no-progress-bar
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
import typer

app = typer.Typer(add_completion=False)

NB_DIR = Path("paper/notebooks")

_HEADER = """\
import structlog

structlog.configure(processors=[structlog.dev.ConsoleRenderer()])
from ml.report import paper_figures as pf
from ml.report import paper_tables as pt

# Seed fijo y estilo cientifico (CVPR/ISPRS) para reproducibilidad byte-a-byte.
pf.set_paper_style()
print("paper style aplicado, seed =", pf.PAPER_SEED)
"""


def _md(source: str) -> nbf.NotebookNode:
    """Create a markdown cell."""
    return nbf.v4.new_markdown_cell(source)


def _code(source: str, tags: list[str] | None = None) -> nbf.NotebookNode:
    """Create a code cell with optional tags."""
    cell = nbf.v4.new_code_cell(source.strip("\n"))
    if tags:
        cell.metadata["tags"] = tags
    return cell


def _notebook(cells: list[nbf.NotebookNode]) -> nbf.NotebookNode:
    """Wrap cells into a notebook with the project kernel metadata."""
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }
    return nb


# --------------------------------------------------------------------------- #
# Notebook 01 -- segmentation figures + table (F4, F6, F7-seg, T2)
# --------------------------------------------------------------------------- #
_NB01 = [
    _md(
        "# US-070 - Figuras y tablas de segmentacion (EPIC 5)\n\n"
        "Reproducible. Orquesta `ml.report.paper_figures` y `ml.report.paper_tables`\n"
        "sobre artefactos reales de `reports/segmentation/**`. Cifras leidas del\n"
        "artefacto, nunca hardcodeadas. Notas de bloqueos: `docs/blockers/epic11-notas.md`.\n\n"
        "- F4 curvas de entrenamiento TSViT (full-config H100 = B-070-2).\n"
        "- F6 matriz de confusion TSViT.\n"
        "- F7-seg barplot de benchmark fold-5 (re-score harness US-030).\n"
        "- T2 modelos individuales fold-5.\n\n"
        "AlphaEarth = SATELLITE_EMBEDDING/V1/ANNUAL v1.1; SegFormer = B0 RGB 3-banda;\n"
        "AnySat sustituye a Swin-UNETR (nunca entrenado)."
    ),
    _code(_HEADER),
    _code(
        "# F7-seg: barplot de benchmark recompuesto desde model_comparison_fold5.csv.\n"
        "# Bilingue: base EN (<stem>.png) + variante ES (<stem>_es.png).\n"
        "for lang in pf.LANGS:\n"
        '    print("F7-seg", lang, pf.fig_benchmark_barplot(lang=lang))'
    ),
    _code(
        "# F4 curvas TSViT y F6 confusion TSViT (PNG reales promovidos a SVG+PNG).\n"
        "# Bilingue: el raster no lleva texto matplotlib; solo cambia el sufijo _es.\n"
        'for stem in ["curves_tsvit", "confusion_tsvit", "per_class_iou_tsvit"]:\n'
        "    for lang in pf.LANGS:\n"
        "        print(stem, lang, pf.promote_png(pf.PROMOTED_FIGURES[stem], stem, lang=lang))"
    ),
    _code(
        "# T2: modelos individuales EPIC 5 re-scoreados fold-5\n"
        "from pathlib import Path\n"
        "tex = pt.build_segmentation_table()\n"
        "out = Path('paper/tables/us-070')\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        '(out / "segmentation_individual_fold5.tex").write_text(tex, encoding="utf-8")\n'
        "print(tex)"
    ),
]


# --------------------------------------------------------------------------- #
# Notebook 02 -- ensemble + FarSLIP figures/tables (F8, T3, Fx, Tx)
# --------------------------------------------------------------------------- #
_NB02 = [
    _md(
        "# US-070 - Figuras y tablas de ensambles y FarSLIP (EPIC 6)\n\n"
        "Orquesta sobre `reports/ensemble/**` y `reports/farslip/**`. Campeon:\n"
        "Stacking-5 (+FarSLIP). Ablacion de bandas completa (3 variantes) = B-070-5.\n\n"
        "- F8 mapa de error espacial (residuals blending).\n"
        "- T3 ensambles EPIC 6 (4 rubrica + E-a/E-b + grid FarSLIP).\n"
        "- Fx curva de ablacion FarSLIP (cardinalidad de clases).\n"
        "- Tx tabla de ablacion de bandas FarSLIP (lo materializado)."
    ),
    _code(_HEADER),
    _code(
        "# F8 mapa de error espacial + F6-ens confusion stacking (bilingue).\n"
        'for stem in ["spatial_residuals", "confusion_stacking"]:\n'
        "    for lang in pf.LANGS:\n"
        "        print(stem, lang, pf.promote_png(pf.PROMOTED_FIGURES[stem], stem, lang=lang))"
    ),
    _code(
        "# Fx curva de ablacion FarSLIP recompuesta desde parcel_sweep.csv (bilingue).\n"
        "for lang in pf.LANGS:\n"
        '    print("Fx farslip_sweep", lang, pf.fig_farslip_sweep_curve(lang=lang))\n'
        "# Ablacion de bandas FarSLIP (fig objetivo de la US): base EN + _es.\n"
        "for lang in pf.LANGS:\n"
        '    print("Fx farslip_band_ablation", lang, pf.fig_farslip_band_ablation(lang=lang))'
    ),
    _code(
        "# T3 ensambles + Tx ablacion bandas FarSLIP\n"
        "from pathlib import Path\n"
        'Path("paper/tables/us-070").mkdir(parents=True, exist_ok=True)\n'
        "t3 = pt.build_ensemble_table()\n"
        'Path("paper/tables/us-070/ensembles_e6.tex").write_text(t3, encoding="utf-8")\n'
        "tx = pt.build_farslip_band_ablation_table()\n"
        'Path("paper/tables/us-070/farslip_band_ablation.tex").write_text(tx, encoding="utf-8")\n'
        "print(t3)\nprint(tx)"
    ),
]


# --------------------------------------------------------------------------- #
# Notebook 03 -- embeddings + FM comparison (F3 UMAP, T1, transfer)
# --------------------------------------------------------------------------- #
_NB03 = [
    _md(
        "# US-070 - Figuras y tablas de embeddings y modelos fundacionales\n\n"
        "- F3 UMAP de AlphaEarth (PNG real promovido).\n"
        "- T1 comparativa de FMs (AlphaEarth v1.1 vs Sentinel-2 vs FarSLIP fiel).\n"
        "- Transferencia FR->Cataluna (Sen4AgriNet, US-075): zero vs few-shot.\n\n"
        "F2 mapas AOI Italia = B-070-1 (requiere auth GEE). La celda de AOI se deja\n"
        "lista para completar con GEE; no se fabrica el mapa."
    ),
    _code(_HEADER),
    _code(
        "# F3 UMAP AlphaEarth + transferencia FR->Cataluna (bilingue).\n"
        'umap_src = pf.PROMOTED_FIGURES["umap_alphaearth"]\n'
        "for lang in pf.LANGS:\n"
        '    print("F3 umap", lang, pf.promote_png(umap_src, "umap_alphaearth", lang=lang))\n'
        '    print("transfer FR->Cataluna", lang, pf.fig_transfer_catalonia(lang=lang))'
    ),
    _code(
        "# T1 comparativa de modelos fundacionales\n"
        "from pathlib import Path\n"
        'Path("paper/tables/us-070").mkdir(parents=True, exist_ok=True)\n'
        "t1 = pt.build_fm_comparison_table()\n"
        'Path("paper/tables/us-070/fm_comparison.tex").write_text(t1, encoding="utf-8")\n'
        "print(t1)"
    ),
    _md(
        "## B-070-1: F2 Mapas AOI Italia (pendiente, requiere auth GEE)\n\n"
        "Celda lista para completar tras autenticar Google Earth Engine y exportar la\n"
        "AOI real de Italia a GeoJSON. No se ejecuta aqui (sin auth GEE en este entorno);\n"
        "no se fabrica un mapa sintetico. Ver `docs/blockers/epic11-notas.md` (B-070-1)."
    ),
    _code(
        "# B-070-1 (NO ejecutar sin auth GEE): render de la AOI real de Italia.\n"
        "# import geopandas as gpd, contextily as cx\n"
        "# aoi = gpd.read_file('data/aoi/italy_aois.geojson')\n"
        "# ax = aoi.to_crs(3857).plot(facecolor='none', edgecolor='red')\n"
        "# cx.add_basemap(ax)\n"
        "# pf.save_fig_svg_png(ax.figure, 'aoi_italy')\n"
        'print("B-070-1 pendiente: requiere auth GEE (ver blockers)")'
    ),
]


# --------------------------------------------------------------------------- #
# Notebook 04 -- agent / LLM figures/tables (F5, F7-LLM, T4, T5)
# --------------------------------------------------------------------------- #
_NB04 = [
    _md(
        "# US-070 - Figuras y tablas del agente y LLMs\n\n"
        "Orquesta sobre `reports/agent_bench/**` (corrida real US-049).\n\n"
        "- F5 ejemplos conversacionales ES/EN (IT = B-070-4, AgroMind-IT US-068).\n"
        "- F7-LLM barplot de benchmark (Gemini vs Qwen reales).\n"
        "- T4 benchmark LLMs (AgroMind-IT/ES = B-070-3, pendiente).\n"
        "- T5 ablacion de uso de herramientas (Gemini real; on-prem pendiente).\n\n"
        "Gemini 2.5-pro = reasoner FROZEN (patron Be My Eyes); Qwen3.5-35B-A3B vLLM\n"
        "GPTQ-Int4 single-GPU = variante on-prem (soberania de datos)."
    ),
    _code(_HEADER),
    _code(
        "# F7-LLM barplot (bilingue: base EN + _es) + F5 ejemplos conversacionales.\n"
        "for lang in pf.LANGS:\n"
        '    print("F7-LLM", lang, pf.fig_llm_benchmark_barplot(lang=lang))\n'
        "# F5: IT pendiente (B-070-4, AgroMind-IT US-068)\n"
        'print("F5 conversational ES/EN:", pf.export_conversational_examples())'
    ),
    _code(
        "# T4 benchmark LLMs + T5 ablacion de tools\n"
        "from pathlib import Path\n"
        'Path("paper/tables/us-070").mkdir(parents=True, exist_ok=True)\n'
        "t4 = pt.build_llm_benchmark_table()\n"
        'Path("paper/tables/us-070/llm_benchmark.tex").write_text(t4, encoding="utf-8")\n'
        "t5 = pt.build_tool_ablation_table()\n"
        'Path("paper/tables/us-070/tool_ablation.tex").write_text(t5, encoding="utf-8")\n'
        "print(t4)\nprint(t5)"
    ),
]


_NOTEBOOKS: dict[str, list[nbf.NotebookNode]] = {
    "01_figures_segmentation.ipynb": _NB01,
    "02_figures_ensemble_farslip.ipynb": _NB02,
    "03_figures_embeddings_fm.ipynb": _NB03,
    "04_figures_agent_llm.ipynb": _NB04,
}


@app.command()
def main(out_dir: Path = NB_DIR) -> None:
    """Write the 4 US-070 paper notebooks (without executing them).

    Args:
        out_dir: Output directory (``paper/notebooks`` by default).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ".gitkeep").touch()
    for name, cells in _NOTEBOOKS.items():
        nb = _notebook(cells)
        path = out_dir / name
        nbf.write(nb, str(path))
        typer.echo(f"wrote {path}")


if __name__ == "__main__":
    app()
