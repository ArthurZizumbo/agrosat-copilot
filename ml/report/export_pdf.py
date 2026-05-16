"""CLI para exportar el reporte PDF del Avance 1 (EDA).

Renderiza un template Jinja2 con cinco fichas (Sentinel-2 univariado,
AlphaEarth Foundations, bivariado/temporal, PASTIS-R consolidado y
conclusiones globales) y lo convierte a PDF mediante WeasyPrint. Cumple
AC-8 y AC-9 de US-013: PDF generado + estructura coherente con los
notebooks reales del Avance 1.

Uso:
    python -m ml.report.export_pdf
    python -m ml.report.export_pdf --output paper/avance1_eda_report.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import typer
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ml.report.figure_narratives import get_narrative
from ml.report.notebook_content import CARDS, NotebookCard, list_figures

try:
    import structlog

    logger: Any = structlog.get_logger(__name__)
except ImportError:  # pragma: no cover - fallback simple
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "paper" / "avance1_eda_report.pdf"
DEFAULT_FIGURES_DIR = REPO_ROOT / "paper" / "figures"
DEFAULT_TEMPLATE = REPO_ROOT / "ml" / "report" / "templates" / "avance1_eda.html.j2"
DEFAULT_TITLE = "Avance 1 - Analisis Exploratorio de Datos - AgroSatCopilot"


app = typer.Typer(add_completion=False, help="Exporta el reporte PDF del Avance 1 EDA.")


def _collect_card_figures(
    cards: tuple[NotebookCard, ...],
    figures_dir: Path,
) -> dict[str, list[Path]]:
    """Agrupa las figuras por ``notebook_id`` segun la configuracion de cada ficha.

    Args:
        cards: Tupla de fichas a procesar (orden de presentacion del reporte).
        figures_dir: Directorio raiz que contiene los subdirectorios por ficha
            (``us-010``, ``us-011``, ``us-012``, ``avance1``, ...).

    Returns:
        Diccionario ``{notebook_id: [paths_png]}`` con listas ordenadas. La
        lista esta vacia para fichas sin ``figures_dir`` o sin PNGs disponibles.
    """
    return {card.notebook_id: list_figures(card, figures_dir) for card in cards}


def _render_html(template_path: Path, context: dict[str, Any]) -> str:
    """Renderiza el template Jinja2 con autoescape activo.

    Args:
        template_path: Path al template .html.j2.
        context: Diccionario de variables que se inyectan al template.

    Returns:
        HTML como string listo para WeasyPrint.
    """
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        autoescape=select_autoescape(["html", "j2", "html.j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_path.name)
    return template.render(**context)


def _html_to_pdf(html_str: str, output: Path, css_path: Path, base_url: Path) -> None:
    """Convierte HTML a PDF mediante WeasyPrint.

    Args:
        html_str: HTML completo a renderizar.
        output: Path destino del PDF.
        css_path: Path al stylesheet CSS.
        base_url: Path base para resolver imagenes referenciadas en el HTML.

    Raises:
        RuntimeError: Si WeasyPrint no esta instalado o si faltan dependencias
            nativas GTK/cairo/pango en Windows.
    """
    try:
        from weasyprint import CSS, HTML
    except ImportError as exc:
        raise RuntimeError(
            "WeasyPrint no esta instalado. Ejecute 'poetry install --with paper' "
            "para anadir el grupo paper que incluye weasyprint y jinja2."
        ) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        html_doc = HTML(string=html_str, base_url=str(base_url))
        stylesheets = [CSS(filename=str(css_path))] if css_path.exists() else []
        html_doc.write_pdf(str(output), stylesheets=stylesheets)
    except OSError as exc:
        gtk_url = "https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer"
        raise RuntimeError(
            "WeasyPrint fallo al renderizar el PDF. En Windows requiere GTK runtime "
            f"(libpango/libcairo). Instale GTK3 ({gtk_url}) o ejecute dentro de WSL2/Linux "
            f"donde las dependencias nativas estan disponibles. Error original: {exc}"
        ) from exc


@app.command()
def main(
    output: Path = typer.Option(  # noqa: B008
        DEFAULT_OUTPUT,
        "--output",
        "-o",
        help="Path destino del PDF generado.",
    ),
    figures_dir: Path = typer.Option(  # noqa: B008
        DEFAULT_FIGURES_DIR,
        "--figures-dir",
        "-f",
        help="Directorio raiz con subdirectorios us-010/us-011/us-012/avance1.",
    ),
    template: Path = typer.Option(  # noqa: B008
        DEFAULT_TEMPLATE,
        "--template",
        "-t",
        help="Path al template Jinja2 .html.j2.",
    ),
    title: str = typer.Option(
        DEFAULT_TITLE,
        "--title",
        help="Titulo de portada del reporte.",
    ),
) -> None:
    """Genera el reporte PDF del Avance 1 a partir de las cinco fichas configuradas."""
    figures_dir = figures_dir.resolve()
    template = template.resolve()
    output = output.resolve()

    if not template.exists():
        typer.echo(f"ERROR: template no existe: {template}", err=True)
        raise typer.Exit(code=2)

    card_figures = _collect_card_figures(CARDS, figures_dir)
    total = sum(len(v) for v in card_figures.values())
    logger.info(
        "report_collect_figures",
        figures_dir=str(figures_dir),
        per_card={k: len(v) for k, v in card_figures.items()},
        total=total,
    )

    context = {
        "title": title,
        "report_date": "2026-05-13",
        "team": [
            "Arthur Zizumbo (MLOps / Platform Lead)",
            "Aaron Bocanegra (Full-Stack / Backend Lead)",
            "Isaac Ávila (ML / Data Scientist)",
        ],
        "sponsor": "Dr. Gerardo Camacho (gjcamacho@tec.mx)",
        "course": "MNA — Tec de Monterrey",
        "cards": CARDS,
        "card_figures": card_figures,
        "figures_base_url": str(figures_dir.parent) + "/",
        "get_narrative": get_narrative,
    }

    html_str = _render_html(template, context)

    css_path = template.parent / "styles.css"
    base_url = figures_dir.parent.parent

    try:
        _html_to_pdf(html_str, output, css_path, base_url)
    except RuntimeError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    size_bytes = output.stat().st_size if output.exists() else 0
    logger.info(
        "report_pdf_generated",
        output=str(output),
        size_bytes=size_bytes,
        size_mb=round(size_bytes / 1024 / 1024, 2),
    )
    typer.echo(f"OK PDF generado: {output} ({size_bytes / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
