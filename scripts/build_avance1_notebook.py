"""Genera el notebook integrador Avance1.Equipo17.ipynb a partir de
``ml.report.notebook_content`` y ``ml.report.figure_narratives``.

El notebook consolidado del Avance 1 consume las mismas fichas que el
dashboard Streamlit y el reporte PDF, garantizando que los tres canales
muestren el mismo contenido (DRY).

Estructura del notebook generado:
    1. Portada (título, equipo, fecha, sponsor, datasets)
    2. Resumen ejecutivo + índice general
    3-7. Cinco capítulos (uno por ficha): título, subtítulo, índice del
        notebook fuente, figuras con narrativa + método, conclusiones
    8. Atribuciones de licencias

Uso:
    poetry run python scripts/build_avance1_notebook.py
    poetry run python scripts/build_avance1_notebook.py \\
        --output notebooks/eda/Avance1.Equipo17.ipynb

Se ejecuta una sola vez por sprint cuando cambia el contenido editorial
(``ml/report/notebook_content.py`` o ``ml/report/figure_narratives.py``).
"""

from __future__ import annotations

import base64
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import typer


def _new_id() -> str:
    """ID estable para cada celda (nbformat >= 4.5 lo exige)."""
    return uuid.uuid4().hex[:12]


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.report.figure_narratives import get_narrative  # noqa: E402
from ml.report.notebook_content import (  # noqa: E402
    CARDS,
    NotebookCard,
    list_figures,
)

FIGURES_ROOT = REPO_ROOT / "paper" / "figures"

app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Helpers para construir celdas Jupyter nbformat v4
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

    Embebe la imagen como output de la celda code que la "renderizó", para
    que el notebook se commitee con figuras pobladas y se vea sin tener
    que re-ejecutarlo.
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
# Construcción de las distintas secciones del notebook
# ---------------------------------------------------------------------------


def _parameters_cell() -> dict[str, Any]:
    """Celda code con tag 'parameters' (papermill).

    Aunque el notebook es file-driven (no consume datos externos), exponemos
    ``figures_dir`` para que papermill pueda apuntarlo a un directorio
    alternativo si se ejecuta en otra máquina.
    """
    src = (
        "# Parámetros papermill — pueden sobrescribirse al ejecutar:\n"
        "#   poetry run papermill notebook.ipynb out.ipynb \\\n"
        '#     -p figures_dir "paper/figures"\n'
        'figures_dir = "paper/figures"\n'
    )
    return _code_cell(src, tags=["parameters"])


def _bootstrap_cell() -> dict[str, Any]:
    """Celda code con sys.path, autoreload, configs polars/matplotlib.

    Sigue la plantilla recomendada en ``notebooks/CLAUDE.md`` aunque este
    notebook solo consume figuras pre-generadas (file-driven). Mantener el
    bootstrap homogéneo facilita reutilizar la plantilla en notebooks futuros.
    """
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
        'display(Markdown(f"**Bootstrap completo** · repo = `{_REPO.name}` · '
        'figuras = `{FIGURES.relative_to(_REPO)}`"))\n'
    )
    return _code_cell(src)


def _kpi_table_cell(card: NotebookCard) -> dict[str, Any] | None:
    """Tabla markdown de KPIs por capítulo (replica los KPI cards del dashboard)."""
    if not card.kpis:
        return None
    lines = [
        "**KPIs principales**\n",
        "\n",
        "| Indicador | Valor | Detalle |\n",
        "| --- | --- | --- |\n",
    ]
    for kpi in card.kpis:
        lines.append(f"| {kpi.label} | **{kpi.value}** | {kpi.delta} |\n")
    lines.append("\n")
    return _md_cell("".join(lines))


def _cover_cell() -> dict[str, Any]:
    """Portada del notebook."""
    return _md_cell(
        "# Análisis Exploratorio de Datos — Avance 1\n"
        "## AgroSatCopilot · Proyecto Integrador MNA\n"
        "\n"
        "**Equipo 17**\n"
        "- Carlos Isaac Ávila Gutiérrez — A01796035\n"
        "- Carlos Aaron Bocanegra Buitrón — A01796345\n"
        "- Arthur Jafed Zizumbo Velasco — A01796363\n"
        "\n"
        "**Sponsor:** Dr. Gerardo Camacho · gjcamacho@tec.mx\n"
        "**Curso:** MNA — Tec de Monterrey · 20-abr → 3-jul-2026\n"
        "**Fecha de entrega:** 2026-05-13\n"
        "\n"
        "---\n"
        "\n"
        "## Resumen ejecutivo\n"
        "\n"
        "Este notebook integra el trabajo de los cuatro análisis exploratorios "
        "que produjimos para el Avance 1 — Sentinel-2 univariado, AlphaEarth "
        "Foundations, bivariado/multivariado/temporal y PASTIS-R consolidado — "
        "más un capítulo de conclusiones globales que cruza los hallazgos y "
        "los traduce en decisiones concretas para los próximos Avances "
        "(Feature Engineering, Baseline y Modelos).\n"
        "\n"
        "Los notebooks individuales por sección viven en:\n"
        "- [`02a_eda_sentinel2.ipynb`](02a_eda_sentinel2.ipynb)\n"
        "- [`02b_eda_alphaearth.ipynb`](02b_eda_alphaearth.ipynb)\n"
        "- [`02c_eda_bivariado_temporal.ipynb`](02c_eda_bivariado_temporal.ipynb)\n"
        "- [`02c_eda_pastis.ipynb`](02c_eda_pastis.ipynb)\n"
        "\n"
        "Las mismas fichas se sirven en el dashboard Streamlit "
        "(`make eda-dashboard`) y en el reporte PDF (`make eda-pdf`). El "
        "contenido editorial (narrativas, conclusiones, índices) vive en "
        "`ml/report/notebook_content.py` y `ml/report/figure_narratives.py` "
        "para garantizar consistencia entre los tres canales.\n"
    )


def _toc_cell() -> dict[str, Any]:
    """Índice general del notebook."""
    lines = ["# Índice\n", "\n"]
    for idx, card in enumerate(CARDS, start=1):
        anchor = card.notebook_id.replace("-", "")
        lines.append(f"{idx}. [{card.title}](#{anchor})\n")
    lines.append("6. [Atribuciones de licencias](#atribuciones)\n")
    return _md_cell("".join(lines))


def _chapter_header_cell(idx: int, card: NotebookCard) -> dict[str, Any]:
    """Header del capítulo: título, subtítulo, notebook fuente, índice."""
    anchor = card.notebook_id.replace("-", "")
    lines = [
        f'<a id="{anchor}"></a>\n',
        f"## {idx}. {card.title}\n",
        "\n",
        f"_{card.subtitle}_\n",
        "\n",
        f"**Notebook fuente:** `{card.notebook_path}`\n",
        "\n",
    ]
    if card.sections:
        lines.append("**Índice del notebook fuente:**\n\n")
        for section in card.sections:
            lines.append(f"- {section}\n")
        lines.append("\n")
    return _md_cell("".join(lines))


def _figure_cells(card: NotebookCard) -> list[dict[str, Any]]:
    """Celdas markdown + code con figuras embebidas y narrativa.

    Cada figura genera:
        1. Markdown con título + narrativa + método.
        2. Code cell con ``display(Image(...))`` + el PNG embebido como output.
    """
    cells: list[dict[str, Any]] = []
    pngs = list_figures(card, FIGURES_ROOT)
    if not pngs:
        if card.figures_dir:
            cells.append(
                _md_cell(
                    f"> _Pendiente: no se encontraron figuras en_ "
                    f"`paper/figures/{card.figures_dir}/`. _Ejecutar el "
                    f"notebook fuente para poblarlas._\n"
                )
            )
        return cells

    cells.append(_md_cell(f"### Figuras del análisis ({len(pngs)} figuras)\n"))

    for png in pngs:
        narrative = get_narrative(card.notebook_id, png.name)
        if narrative is not None:
            md_lines = [
                f"**{narrative.title}**\n",
                "\n",
                f"{narrative.narrative}\n",
                "\n",
                f"> _Cómo se construyó: {narrative.method}_\n",
            ]
        else:
            stem = png.stem.replace("_", " ").capitalize()
            md_lines = [
                f"**{stem}**\n",
                "\n",
                f"_Figura: `{png.name}`. Narrativa interpretativa pendiente._\n",
            ]
        cells.append(_md_cell("".join(md_lines)))

        # Celda code que "renderizaría" la figura. Embebemos el PNG como
        # output para que el notebook se vea con figuras pobladas sin
        # re-ejecutar.
        # Path relativo a paper/figures (sin el prefijo) — la celda lo
        # resuelve contra `FIGURES` (variable definida en bootstrap),
        # garantizando que funciona sin importar el CWD desde el que
        # se abra el notebook (Jupyter typically usa el dir del notebook).
        rel_to_figures = png.relative_to(FIGURES_ROOT).as_posix()
        code_src = f'display(Image(str(FIGURES / "{rel_to_figures}")))\n'
        cells.append(_code_cell(code_src, outputs=[_image_output(png)]))

    return cells


def _conclusions_cells(card: NotebookCard) -> list[dict[str, Any]]:
    """Celdas markdown con las conclusiones interpretadas de la ficha."""
    if not card.conclusions:
        return []

    cells: list[dict[str, Any]] = [
        _md_cell(f"### Conclusiones e interpretación ({len(card.conclusions)} hallazgos)\n")
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
        "- **Sentinel-2** — Copernicus Programme (European Union / ESA) · "
        "términos Copernicus\n"
        "- **AlphaEarth Foundations v2.1** — Google DeepMind vía Google Earth "
        "Engine · términos GEE\n"
        "- **Dynamic World** — Google + World Resources Institute · CC-BY-4.0\n"
        "- **ERA5** — Copernicus Climate Change Service (C3S) · licencia "
        "Copernicus\n"
        "- **EuroCrops / HCAT3** — TUM (Technische Universität München) · "
        "CC-BY-4.0\n"
        "\n"
        "Detalle completo en `docs/licenses/DATA_LICENSE.md`.\n"
        "\n"
        "---\n"
        "\n"
        "_Notebook generado por_ `scripts/build_avance1_notebook.py` _a "
        "partir de_ `ml/report/notebook_content.py` _y_ "
        "`ml/report/figure_narratives.py`."
    )


def build_notebook() -> dict[str, Any]:
    """Construye el dict del notebook listo para escribir a JSON."""
    cells: list[dict[str, Any]] = [
        _cover_cell(),
        _parameters_cell(),
        _bootstrap_cell(),
        _toc_cell(),
    ]

    for idx, card in enumerate(CARDS, start=1):
        cells.append(_chapter_header_cell(idx, card))
        kpi_cell = _kpi_table_cell(card)
        if kpi_cell is not None:
            cells.append(kpi_cell)
        cells.extend(_figure_cells(card))
        cells.extend(_conclusions_cells(card))

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
def main(
    output: Path = typer.Option(  # noqa: B008
        REPO_ROOT / "notebooks" / "eda" / "Avance1.Equipo17.ipynb",
        "--output",
        "-o",
        help="Path destino del notebook generado.",
    ),
) -> None:
    """Genera el notebook integrador Avance 1 con figuras embebidas."""
    nb = build_notebook()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)

    size_kb = output.stat().st_size / 1024
    n_cells = len(nb["cells"])
    n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    typer.echo(f"OK notebook generado: {output}")
    typer.echo(f"   {n_cells} celdas ({n_md} markdown + {n_code} code) · {size_kb:.1f} KB")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
