"""Tests para scripts/build_avance1_notebook.py.

El script genera el notebook integrador commiteado como entregable A1; sin
test, una refactor de NotebookCard o figure_narratives puede romper la
generacion y solo se detectaria al regenerar manualmente.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_avance1_notebook.py"


@pytest.fixture(scope="module")
def build_module():
    """Carga build_avance1_notebook.py como modulo importable."""
    spec = importlib.util.spec_from_file_location("build_avance1_notebook", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_avance1_notebook"] = module
    spec.loader.exec_module(module)
    return module


def test_build_notebook_returns_valid_nbformat_v4(build_module) -> None:
    """build_notebook() retorna un dict con nbformat == 4 y minor == 5."""
    nb = build_module.build_notebook()
    assert isinstance(nb, dict)
    assert nb["nbformat"] == 4
    assert nb["nbformat_minor"] == 5
    assert "metadata" in nb
    assert "cells" in nb


def test_build_notebook_has_cells_and_kernelspec(build_module) -> None:
    """El notebook generado tiene celdas y declara kernel Python 3."""
    nb = build_module.build_notebook()
    assert len(nb["cells"]) > 0
    kernel = nb["metadata"]["kernelspec"]
    assert kernel["language"] == "python"
    assert kernel["name"] == "python3"


def test_build_notebook_first_cell_is_cover_markdown(build_module) -> None:
    """La primera celda es la portada (markdown con titulo del Avance 1)."""
    nb = build_module.build_notebook()
    cover = nb["cells"][0]
    assert cover["cell_type"] == "markdown"
    source = "".join(cover["source"])
    assert "Avance 1" in source
    assert "AgroSatCopilot" in source
    assert "Equipo 17" in source


def test_build_notebook_includes_all_five_chapters(build_module) -> None:
    """El notebook integra los 5 capitulos (uno por NotebookCard)."""
    from ml.report.notebook_content import CARDS

    nb = build_module.build_notebook()
    full_text = "".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "markdown"
    )
    for card in CARDS:
        assert card.title in full_text, f"Falta el capitulo: {card.title}"


def test_build_notebook_renders_kpis_from_notebookcard(build_module) -> None:
    """Las tablas de KPIs consumen card.kpis (fuente unica, no listas duplicadas)."""
    from ml.report.notebook_content import CARDS

    nb = build_module.build_notebook()
    full_text = "".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "markdown"
    )
    for card in CARDS:
        for kpi in card.kpis:
            assert kpi.label in full_text, f"Falta KPI '{kpi.label}' de {card.notebook_id}"


def test_build_notebook_closes_with_attributions(build_module) -> None:
    """La ultima celda es la atribucion de licencias."""
    nb = build_module.build_notebook()
    last = nb["cells"][-1]
    assert last["cell_type"] == "markdown"
    source = "".join(last["source"])
    assert "Atribuciones" in source
    assert "PASTIS-R" in source
    assert "Sentinel-2" in source


def test_parameters_cell_tagged_for_papermill(build_module) -> None:
    """La celda parameters tiene el tag 'parameters' que papermill espera."""
    cell = build_module._parameters_cell()
    assert cell["cell_type"] == "code"
    assert cell["metadata"]["tags"] == ["parameters"]
    source = "".join(cell["source"])
    assert "figures_dir" in source


def test_kpi_table_cell_returns_none_when_card_has_no_kpis(build_module) -> None:
    """_kpi_table_cell devuelve None para fichas sin KPIs (degradacion graceful)."""
    from ml.report.notebook_content import NotebookCard

    empty_card = NotebookCard(
        notebook_id="empty",
        notebook_path="(test)",
        title="Empty",
        subtitle="Empty",
        sections=(),
        figures_dir="",
    )
    assert build_module._kpi_table_cell(empty_card) is None
