"""Genera el notebook integrador Avance2.Equipo17.ipynb a partir de
``ml.report.avance2_content``.

El notebook consolidado del Avance 2 reune el trabajo de ingenieria de
caracteristicas realizado sobre tres fuentes de datos complementarias
(Sentinel-2, PASTIS-R espectro-temporal y embeddings AlphaEarth) en un
unico entregable coherente, sin re-ejecutar el feature engineering inline.
Sigue el mismo patron que ``scripts/build_avance1_notebook.py``.

Estructura del notebook generado:
    1. Portada (titulo, equipo, fecha, datasets)
    2. Resumen ejecutivo + indice general
    3-5. Capitulos (uno por notebook fuente): titulo, subtitulo, indice del
        notebook fuente, figuras con narrativa, conclusiones interpretadas
    6. Conclusiones globales de la fase de preparacion de datos
    7. Atribuciones de licencias

Las figuras se embeben como base64 en las celdas de codigo, de modo que el
notebook se commitea con las figuras pobladas sin necesidad de re-ejecutarlo
(no requiere papermill). Las figuras se extraen previamente de los tres
notebooks fuente con ``ml.report.extract_notebook_figures``.

Uso:
    poetry run python scripts/build_avance2_notebook.py
    poetry run python scripts/build_avance2_notebook.py \\
        --out notebooks/feature_engineering/Avance2.Equipo17.ipynb

Se ejecuta una sola vez por sprint cuando cambia el contenido editorial
(``ml/report/avance2_content.py``).
"""

from __future__ import annotations

import base64
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import typer

from ml.report.notebook_conclusions import A2_CONCLUSIONS
from ml.report.notebook_cover import build_cover_markdown, build_team_conclusions_markdown


def _new_id() -> str:
    """ID estable para cada celda (nbformat >= 4.5 lo exige)."""
    return uuid.uuid4().hex[:12]


def _visual_cover_cell() -> dict[str, Any]:
    """Portada visual homologada (badges + header) antes del resumen ejecutivo."""
    return _md_cell(
        build_cover_markdown(
            "Avance 2",
            "Ingenieria de Caracteristicas",
            "Construccion, seleccion y normalizacion de caracteristicas sobre "
            "Sentinel-2, PASTIS-R y AlphaEarth, con fusion multisensor a nivel "
            "de parcela y validacion cross-region.",
            "2026-05-17",
        )
    )


def _team_conclusions_cell() -> dict[str, Any]:
    """Celda de conclusiones individuales por integrante."""
    return _md_cell(build_team_conclusions_markdown(A2_CONCLUSIONS))


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.report.avance2_content import FE_CARDS  # noqa: E402
from ml.report.figure_narratives import get_narrative  # noqa: E402
from ml.report.notebook_content import NotebookCard, list_figures  # noqa: E402

FIGURES_ROOT = REPO_ROOT / "paper" / "figures"

app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Helpers to build Jupyter nbformat v4 cells
# ---------------------------------------------------------------------------


def _md_cell(source: str) -> dict[str, Any]:
    """Celda markdown nbformat v4.5."""
    return {
        "cell_type": "markdown",
        "id": _new_id(),
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def _code_cell(
    source: str,
    outputs: list[dict[str, Any]] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Celda code nbformat v4.5 con outputs embebidos y tags opcionales."""
    metadata: dict[str, Any] = {}
    if tags:
        metadata["tags"] = tags
    return {
        "cell_type": "code",
        "id": _new_id(),
        "execution_count": None,
        "metadata": metadata,
        "source": source.splitlines(keepends=True),
        "outputs": outputs or [],
    }


def _image_output(png_path: Path) -> dict[str, Any]:
    """Output display_data con PNG embebido en base64.

    Embebe la imagen como output de la celda code que la renderiza, para que
    el notebook se commitee con las figuras pobladas y se vea sin tener que
    re-ejecutarlo.
    """
    png_bytes = png_path.read_bytes()
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return {
        "output_type": "display_data",
        "data": {
            "image/png": b64,
            "text/plain": [f"<Figure: {png_path.name}>"],
        },
        "metadata": {"image/png": {}},
    }


# ---------------------------------------------------------------------------
# Construction of the different sections of the notebook
# ---------------------------------------------------------------------------


def _parameters_cell() -> dict[str, Any]:
    """Celda code con tag 'parameters'.

    El notebook es file-driven (no consume datos externos), pero exponemos
    ``figures_dir`` para que pueda apuntarse a un directorio alternativo si se
    ejecuta en otra maquina.
    """
    src = (
        "# Parametros configurables del notebook integrador:\n"
        'figures_dir = "paper/figures"\n'
    )
    return _code_cell(src, tags=["parameters"])


def _bootstrap_cell() -> dict[str, Any]:
    """Celda code con sys.path, autoreload y resolucion del directorio de figuras."""
    src = (
        "from __future__ import annotations\n"
        "\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "from IPython.display import Image, Markdown, display\n"
        "\n"
        "# Bootstrap sys.path para que el notebook pueda importar desde ml/*\n"
        "_REPO = Path.cwd().resolve()\n"
        "for _candidate in (_REPO, *_REPO.parents):\n"
        '    if (_candidate / "pyproject.toml").is_file():\n'
        "        _REPO = _candidate\n"
        "        break\n"
        "if str(_REPO) not in sys.path:\n"
        "    sys.path.insert(0, str(_REPO))\n"
        "\n"
        "FIGURES = _REPO / figures_dir\n"
        "\n"
        "# Autoreload para captar cambios en ml/*.py sin restart de kernel\n"
        "%load_ext autoreload\n"
        "%autoreload 2\n"
        "\n"
        'display(Markdown(f"**Configuracion lista** - repo = `{_REPO.name}` - '
        'figuras = `{FIGURES.relative_to(_REPO)}`"))\n'
    )
    return _code_cell(src)


def _kpi_table_cell(card: NotebookCard) -> dict[str, Any] | None:
    """Tabla markdown de KPIs por capitulo."""
    if not card.kpis:
        return None
    lines = [
        "**Indicadores principales**\n",
        "\n",
        "| Indicador | Valor | Detalle |\n",
        "| --- | --- | --- |\n",
    ]
    for kpi in card.kpis:
        lines.append(f"| {kpi.label} | **{kpi.value}** | {kpi.delta} |\n")
    lines.append("\n")
    return _md_cell("".join(lines))


def _cover_cell() -> dict[str, Any]:
    """Portada del notebook + resumen ejecutivo."""
    return _md_cell(
        "# Avance 2 — Ingenieria de Caracteristicas\n"
        "## Equipo 17 · AgroSatCopilot · Proyecto Integrador MNA\n"
        "\n"
        "**Equipo 17**\n"
        "- Carlos Isaac Ávila Gutiérrez — A01796035\n"
        "- Carlos Aaron Bocanegra Buitrón — A01796345\n"
        "- Arthur Jafed Zizumbo Velasco — A01796363\n"
        "\n"
        "**Curso:** MNA — Tec de Monterrey · 20-abr → 3-jul-2026\n"
        "**Fecha de entrega:** 2026-05-17\n"
        "\n"
        "**Datasets utilizados:** Sentinel-2 L2A (Copernicus), PASTIS-R "
        "(Sainte-Fare-Garnot et al. 2021), AlphaEarth Foundations (Google "
        "DeepMind), SRTM, ERA5 y BreizhCrops.\n"
        "\n"
        "---\n"
        "\n"
        "## Resumen ejecutivo\n"
        "\n"
        "Este notebook consolida el trabajo de ingenieria de caracteristicas "
        "que realizamos sobre tres fuentes de datos complementarias y lo "
        "presenta como un unico recorrido coherente.\n"
        "\n"
        "La ingenieria de caracteristicas es el puente entre los datos crudos "
        "y los modelos: transforma imagenes satelitales y series temporales "
        "en tablas numericas listas para entrenar, decidiendo que construir, "
        "que transformar y que conservar. Lo abordamos desde tres angulos:\n"
        "\n"
        "- **Imagen cruda de Sentinel-2** — construccion de indices "
        "espectrales, discretizacion y codificacion de variables a nivel de "
        "pixel.\n"
        "- **Parcelas de PASTIS-R** — seleccion, extraccion y normalizacion "
        "sobre 185 caracteristicas espectro-temporales por parcela.\n"
        "- **Embeddings AlphaEarth** — fusion multisensor sobre el vector de "
        "64 dimensiones aprendido por un modelo de base.\n"
        "\n"
        "Cada capitulo resume uno de los tres notebooks de trabajo detallado, "
        "con sus figuras y sus conclusiones interpretadas con numeros reales. "
        "Un cuarto capitulo cruza los hallazgos y los traduce en decisiones "
        "concretas para la fase de modelado base.\n"
        "\n"
        "Los notebooks de trabajo detallado viven en:\n"
        "- [`03a_fe_sentinel2.ipynb`](03a_fe_sentinel2.ipynb)\n"
        "- [`03b_fe_spectral_temporal_pastis.ipynb`](03b_fe_spectral_temporal_pastis.ipynb)\n"
        "- [`03c_fe_alphaearth_pastis.ipynb`](03c_fe_alphaearth_pastis.ipynb)\n"
    )


def _toc_cell() -> dict[str, Any]:
    """Indice general del notebook."""
    lines = ["# Indice\n", "\n"]
    for idx, card in enumerate(FE_CARDS, start=1):
        anchor = card.notebook_id.replace("-", "")
        lines.append(f"{idx}. [{card.title}](#{anchor})\n")
    lines.append(f"{len(FE_CARDS) + 1}. [Atribuciones de licencias](#atribuciones)\n")
    return _md_cell("".join(lines))


def _chapter_header_cell(idx: int, card: NotebookCard) -> dict[str, Any]:
    """Header del capitulo: titulo, subtitulo, notebook fuente, indice."""
    anchor = card.notebook_id.replace("-", "")
    lines = [
        f'<a id="{anchor}"></a>\n',
        f"## {idx}. {card.title}\n",
        "\n",
        f"_{card.subtitle}_\n",
        "\n",
    ]
    if not card.notebook_path.startswith("("):
        lines.append(f"**Notebook de trabajo:** `{card.notebook_path}`\n\n")
    if card.sections:
        label = (
            "**Contenido de este capitulo:**\n\n"
            if card.notebook_path.startswith("(")
            else "**Indice del notebook de trabajo:**\n\n"
        )
        lines.append(label)
        for section in card.sections:
            lines.append(f"- {section}\n")
        lines.append("\n")
    return _md_cell("".join(lines))


def _figure_cells(card: NotebookCard) -> list[dict[str, Any]]:
    """Celdas markdown + code con figuras embebidas y narrativa por figura.

    Cada figura genera:
        1. Markdown con titulo + narrativa interpretativa + metodo.
        2. Code cell con ``display(Image(...))`` + el PNG embebido como output.

    Replica el patron del notebook integrador del Avance 1: la narrativa
    se resuelve con :func:`ml.report.figure_narratives.get_narrative` usando
    el ``notebook_id`` de la ficha. Si una figura no tiene narrativa
    registrada, se usa un fallback con el nombre del archivo.
    """
    cells: list[dict[str, Any]] = []
    pngs = list_figures(card, FIGURES_ROOT)
    if not pngs:
        return cells

    cells.append(_md_cell(f"### Figuras del analisis ({len(pngs)} figuras)\n"))

    for png in pngs:
        narrative = get_narrative(card.notebook_id, png.name)
        if narrative is not None:
            md_lines = [
                f"**{narrative.title}**\n",
                "\n",
                f"{narrative.narrative}\n",
                "\n",
                f"> _Como se construyo: {narrative.method}_\n",
            ]
        else:
            stem = png.stem.replace("_", " ").capitalize()
            md_lines = [
                f"**{stem}**\n",
                "\n",
                f"_Figura: `{png.name}`. Narrativa interpretativa pendiente._\n",
            ]
        cells.append(_md_cell("".join(md_lines)))

        rel_to_figures = png.relative_to(FIGURES_ROOT).as_posix()
        code_src = f'display(Image(str(FIGURES / "{rel_to_figures}")))\n'
        cells.append(_code_cell(code_src, outputs=[_image_output(png)]))

    return cells


def _conclusions_cells(card: NotebookCard) -> list[dict[str, Any]]:
    """Celdas markdown con las conclusiones interpretadas del capitulo."""
    if not card.conclusions:
        return []

    cells: list[dict[str, Any]] = [
        _md_cell(
            f"### Conclusiones e interpretacion ({len(card.conclusions)} hallazgos)\n"
        )
    ]
    for heading, body in card.conclusions:
        cells.append(_md_cell(f"**{heading}**\n\n{body}\n"))
    return cells


def _attributions_cell() -> dict[str, Any]:
    """Cierre con atribuciones de licencias."""
    return _md_cell(
        '<a id="atribuciones"></a>\n'
        "## Atribuciones de licencias\n"
        "\n"
        "- **PASTIS-R** — Sainte-Fare-Garnot et al. 2021 · CC-BY-SA 4.0\n"
        "- **Sentinel-2 L2A / Sentinel-1 GRD** — Copernicus Programme "
        "(European Union / ESA) · términos Copernicus\n"
        "- **AlphaEarth Foundations** — Google DeepMind vía Google Earth "
        "Engine · términos GEE\n"
        "- **SRTM** — NASA / USGS · dominio público\n"
        "- **ERA5** — Copernicus Climate Change Service (C3S) · licencia "
        "Copernicus\n"
        "- **BreizhCrops** — Rußwurm et al. 2020 · licencia abierta de "
        "investigación\n"
        "\n"
        "Detalle completo en `docs/licenses/DATA_LICENSE.md`.\n"
        "\n"
        "---\n"
        "\n"
        "_Notebook generado por_ `scripts/build_avance2_notebook.py` _a "
        "partir de_ `ml/report/avance2_content.py`."
    )


def build_notebook() -> dict[str, Any]:
    """Construye el dict del notebook listo para escribir a JSON."""
    cells: list[dict[str, Any]] = [
        _visual_cover_cell(),
        _cover_cell(),
        _parameters_cell(),
        _bootstrap_cell(),
        _toc_cell(),
    ]

    for idx, card in enumerate(FE_CARDS, start=1):
        cells.append(_chapter_header_cell(idx, card))
        kpi_cell = _kpi_table_cell(card)
        if kpi_cell is not None:
            cells.append(kpi_cell)
        cells.extend(_figure_cells(card))
        cells.extend(_conclusions_cells(card))

    cells.append(_team_conclusions_cell())
    cells.append(_attributions_cell())

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


@app.command()
def build(
    out: Path = typer.Option(
        REPO_ROOT / "notebooks" / "feature_engineering" / "Avance2.Equipo17.ipynb",
        "--out",
        "-o",
        help="Ruta destino del .ipynb generado.",
    ),
) -> None:
    """Genera el notebook integrador Avance 2 con figuras embebidas."""
    nb = build_notebook()
    out_path = out if out.is_absolute() else Path.cwd() / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)

    size_kb = out_path.stat().st_size / 1024
    n_cells = len(nb["cells"])
    n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    typer.echo(f"OK notebook generado: {out_path}")
    typer.echo(
        f"   {n_cells} celdas ({n_md} markdown + {n_code} code) - {size_kb:.1f} KB"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
