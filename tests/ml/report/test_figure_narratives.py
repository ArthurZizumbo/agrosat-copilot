"""Tests para ml.report.figure_narratives.

Verifica que cada figura existente en ``paper/figures/<dir>/`` tenga su
narrativa asignada y que ``get_narrative`` devuelva ``None`` para
filenames desconocidos.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.report.figure_narratives import (
    ALPHAEARTH_NARRATIVES,
    BIVARIATE_NARRATIVES,
    NARRATIVES_BY_NOTEBOOK,
    PASTIS_NARRATIVES,
    SENTINEL2_NARRATIVES,
    FigureNarrative,
    get_narrative,
)
from ml.report.notebook_content import CARDS

REPO_ROOT = Path(__file__).resolve().parents[3]
FIGURES_ROOT = REPO_ROOT / "paper" / "figures"


def test_narratives_by_notebook_covers_all_cards() -> None:
    """``NARRATIVES_BY_NOTEBOOK`` define entrada para cada ficha de ``CARDS``."""
    notebook_ids = {card.notebook_id for card in CARDS}
    assert set(NARRATIVES_BY_NOTEBOOK.keys()) == notebook_ids


@pytest.mark.parametrize(
    "narratives",
    [SENTINEL2_NARRATIVES, ALPHAEARTH_NARRATIVES, BIVARIATE_NARRATIVES, PASTIS_NARRATIVES],
)
def test_narratives_have_required_fields(narratives: tuple) -> None:
    """Cada narrativa define filename, title, narrative, method no vacios."""
    assert narratives, "narrativas no pueden estar vacias"
    for narr in narratives:
        assert isinstance(narr, FigureNarrative)
        assert narr.filename.endswith(".png")
        assert narr.title.strip()
        assert narr.narrative.strip()
        assert narr.method.strip()


def test_get_narrative_returns_match() -> None:
    """``get_narrative`` retorna la narrativa correcta cuando coincide filename."""
    narr = get_narrative("sentinel2", "band_distributions.png")
    assert narr is not None
    assert "10 bandas" in narr.narrative or "infrarrojo" in narr.narrative


def test_get_narrative_returns_none_for_unknown_filename() -> None:
    """``get_narrative`` retorna None si filename no esta registrado."""
    assert get_narrative("sentinel2", "no_existe.png") is None


def test_get_narrative_returns_none_for_unknown_notebook() -> None:
    """``get_narrative`` retorna None si notebook_id no existe."""
    assert get_narrative("inexistente", "x.png") is None


def test_get_narrative_globales_has_no_narratives() -> None:
    """La ficha globales no tiene figuras propias - get_narrative siempre None."""
    assert NARRATIVES_BY_NOTEBOOK["globales"] == ()
    assert get_narrative("globales", "x.png") is None


@pytest.mark.parametrize(
    "subdir,narratives_key",
    [
        ("us-010", "sentinel2"),
        ("us-011", "alphaearth"),
        ("us-012", "bivariate-temporal"),
        ("avance1", "pastis-consolidado"),
    ],
)
def test_existing_figures_have_narratives(subdir: str, narratives_key: str) -> None:
    """Toda figura presente en disco debe tener narrativa registrada.

    Si faltan narrativas, este test indica que el contenido editorial quedo
    desincronizado con la salida real de los notebooks. No falla si el
    directorio no existe (CI sin figuras pobladas).
    """
    figures_dir = FIGURES_ROOT / subdir
    if not figures_dir.is_dir():
        pytest.skip(f"Directorio {figures_dir} no existe (figuras no pobladas)")

    pngs = sorted(figures_dir.glob("*.png"))
    if not pngs:
        pytest.skip(f"No hay PNGs en {figures_dir}")

    missing = [png.name for png in pngs if get_narrative(narratives_key, png.name) is None]
    assert not missing, (
        f"Figuras sin narrativa en {subdir}: {missing}. "
        f"Agregar en ml.report.figure_narratives.{narratives_key.upper()}_NARRATIVES."
    )
