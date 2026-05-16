"""Extrae figuras PNG inline de un notebook Jupyter a un directorio destino.

Recorre las salidas de cada celda buscando ``image/png`` (base64) y las exporta
con un nombre estable ``cell_{idx:03d}_{output_idx}.png``. Sirve para alimentar
el reporte PDF del Avance 1 desde notebooks que generan sus figuras inline
(sin ``plt.savefig`` explicito), como ``Avance1.Equipo17.ipynb``.

Uso::

    python -m ml.report.extract_notebook_figures \\
        notebooks/eda/Avance1.Equipo17.ipynb \\
        --output paper/figures/avance1/
"""

from __future__ import annotations

import base64
import json
import re
import sys
import unicodedata
from pathlib import Path

import typer

REPO_ROOT = Path(__file__).resolve().parents[2]

app = typer.Typer(add_completion=False, help="Extrae figuras PNG inline de notebooks Jupyter.")


def _slugify(text: str, max_len: int = 60) -> str:
    """Convierte un heading markdown a un slug ASCII seguro para nombres de archivo.

    Args:
        text: Texto fuente (puede contener acentos, ``#``, numeros de seccion).
        max_len: Longitud maxima del slug resultante.

    Returns:
        Slug en minusculas con guiones bajos. Vacio si el input es vacio.
    """
    text = text.lstrip("#").strip()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text[:max_len]


def _nearest_heading(cells: list[dict], cell_idx: int) -> str:
    """Devuelve el heading markdown mas cercano (hacia atras) a ``cell_idx``.

    Args:
        cells: Lista de celdas del notebook.
        cell_idx: Indice de la celda con figura.

    Returns:
        Primera linea no vacia del markdown previo, o cadena vacia si no hay.
    """
    for j in range(cell_idx, -1, -1):
        cell = cells[j]
        if cell.get("cell_type") != "markdown":
            continue
        src = "".join(cell.get("source", []))
        for line in src.splitlines():
            line = line.strip()
            if line:
                return line
    return ""


def extract_png_outputs(notebook_path: Path, output_dir: Path) -> list[Path]:
    """Extrae todas las imagenes PNG inline de un notebook a ``output_dir``.

    Args:
        notebook_path: Path al archivo .ipynb.
        output_dir: Directorio destino donde se escriben los PNG.

    Returns:
        Lista de paths PNG creados, ordenados por celda y output_idx.

    Raises:
        FileNotFoundError: Si el notebook no existe.
        ValueError: Si el JSON del notebook esta corrupto.
    """
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook no existe: {notebook_path}")

    with notebook_path.open("r", encoding="utf-8") as fh:
        try:
            nb = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Notebook JSON invalido: {notebook_path}") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    cells: list[dict] = nb.get("cells", [])
    used_slugs: dict[str, int] = {}

    for cell_idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs") or []
        for output in outputs:
            data = output.get("data") or {}
            png_b64 = data.get("image/png")
            if not png_b64:
                continue
            if isinstance(png_b64, list):
                png_b64 = "".join(png_b64)
            try:
                png_bytes = base64.b64decode(png_b64)
            except (ValueError, TypeError):
                continue
            heading = _nearest_heading(cells, cell_idx)
            slug = _slugify(heading) or "figure"
            count = used_slugs.get(slug, 0)
            used_slugs[slug] = count + 1
            target = output_dir / f"cell_{cell_idx:03d}_{slug}_{count}.png"
            if count == 0:
                # Para evitar siempre el sufijo _0 en el caso comun.
                target = output_dir / f"cell_{cell_idx:03d}_{slug}.png"
            target.write_bytes(png_bytes)
            created.append(target)

    return created


@app.command()
def main(
    notebook: Path = typer.Argument(  # noqa: B008
        ...,
        help="Path al notebook .ipynb a procesar.",
    ),
    output: Path = typer.Option(  # noqa: B008
        REPO_ROOT / "paper" / "figures" / "avance1",
        "--output",
        "-o",
        help="Directorio destino para las figuras PNG.",
    ),
) -> None:
    """Extrae figuras PNG inline de un notebook al directorio destino."""
    notebook = notebook.resolve()
    output = output.resolve()

    try:
        created = extract_png_outputs(notebook, output)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"OK {len(created)} figuras extraidas a {output}")
    for path in created:
        typer.echo(f"  - {path.name}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
