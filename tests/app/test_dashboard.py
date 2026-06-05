"""Smoke tests para ``app/eda_dashboard.py``.

Cobertura objetivo: >= 70 % sobre ``app/eda_dashboard.py``.

Por limitaciones de ``AppTest.from_function`` (extrae solo el source de la
funcion y lo ejecuta en sandbox sin imports), los tests de render por ficha
se ejecutan a traves de ``AppTest.from_file`` que carga el dashboard
completo. Los helpers privados (``_render_card_*``) se prueban
indirectamente via las salidas del dashboard cargado.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

# Streamlit es opcional para el repo base; se instala via grupo `paper`.
streamlit = pytest.importorskip("streamlit", reason="Streamlit no instalado (grupo paper).")
AppTest = pytest.importorskip(
    "streamlit.testing.v1", reason="streamlit.testing.v1 requiere Streamlit >= 1.28."
).AppTest

# Importes del modulo bajo prueba - protegidos por importorskip arriba.
from app import eda_dashboard  # noqa: E402

DASHBOARD_PATH = Path(__file__).resolve().parents[2] / "app" / "eda_dashboard.py"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_csv(tmp_path: Path) -> Path:
    """Crea un CSV de 2 filas dummy."""
    target = tmp_path / "sample.csv"
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["band", "mean", "std"])
        writer.writerow(["B04", "1234.5", "456.7"])
        writer.writerow(["B08", "2345.6", "567.8"])
    return target


@pytest.fixture(scope="module")
def loaded_app() -> object:
    """Carga el dashboard con la landing Historia (seccion por defecto)."""
    return AppTest.from_file(str(DASHBOARD_PATH)).run(timeout=45)


@pytest.fixture(scope="module")
def eda_app() -> object:
    """Carga el dashboard y selecciona la seccion EDA via session_state."""
    at = AppTest.from_file(str(DASHBOARD_PATH)).run(timeout=45)
    at.session_state[eda_dashboard._SECTION_STATE_KEY] = eda_dashboard._SECTION_EDA  # type: ignore[attr-defined]
    return at.run(timeout=45)


def _markdown_blob(at: object) -> str:
    """Concatena el markdown renderizado del AppTest en un solo string."""
    return " ".join(getattr(node, "value", "") for node in at.markdown)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Landing por defecto: Historia del proyecto (linea de tiempo A0 -> A4)
# ---------------------------------------------------------------------------


def test_dashboard_loads(loaded_app: object) -> None:
    """El dashboard arranca sin excepcion y la landing expone 6 hitos."""
    at = loaded_app
    assert not at.exception, f"Excepcion al cargar: {at.exception}"  # type: ignore[attr-defined]
    assert len(at.tabs) == 6, f"Se esperaban 6 hitos en la landing, hay {len(at.tabs)}"  # type: ignore[attr-defined]


def test_dashboard_sidebar_has_team(loaded_app: object) -> None:
    """La sidebar muestra titulo y miembros del equipo."""
    at = loaded_app
    sidebar_titles = " ".join(getattr(node, "value", "") for node in at.sidebar.title)  # type: ignore[attr-defined]
    sidebar_md = " ".join(getattr(node, "value", "") for node in at.sidebar.markdown)  # type: ignore[attr-defined]
    blob = sidebar_titles + " " + sidebar_md
    assert "AgroSatCopilot" in blob
    assert "Arthur Zizumbo" in blob or "Isaac" in blob


def test_dashboard_hero_present(loaded_app: object) -> None:
    """El hero presenta la identidad del proyecto y su narrativa de evolucion."""
    blob = _markdown_blob(loaded_app)
    assert "AgroSatCopilot" in blob
    assert "Analisis Exploratorio" in blob


def test_landing_shows_evolution_kpis(loaded_app: object) -> None:
    """La landing Historia muestra KPIs que evolucionan A0 -> A4."""
    blob = _markdown_blob(loaded_app)
    for needle in ("Historia del proyecto", "Mejor mIoU", "0,625", "Reencuadre"):
        assert needle in blob, f"Falta hito/KPI en la landing: {needle}"


# ---------------------------------------------------------------------------
# Seccion EDA (seleccionada via session_state)
# ---------------------------------------------------------------------------


def test_dashboard_renders_all_card_titles(eda_app: object) -> None:
    """La seccion EDA muestra los titulos de las 7 fichas."""
    blob = _markdown_blob(eda_app)
    expected_titles = (
        "EDA Univariado",
        "AlphaEarth Foundations",
        "Bivariado, Multivariado y Temporal",
        "PASTIS-R Consolidado",
        "BreizhCrops",
        "Conclusiones Globales del Avance 1",
    )
    missing = [t for t in expected_titles if t not in blob]
    assert not missing, f"Titulos faltantes: {missing}"


def test_dashboard_renders_kpi_cards(eda_app: object) -> None:
    """La seccion EDA inyecta KPI cards con valores clave de cada ficha."""
    blob = _markdown_blob(eda_app)
    expected_kpis = (
        "Bandas analizadas",
        "OOB Italia",
        "VIF mínimo",
        "Parches",
        "Notebooks integrados",
    )
    missing = [k for k in expected_kpis if k not in blob]
    assert not missing, f"KPIs faltantes: {missing}"


def test_dashboard_renders_global_conclusions(eda_app: object) -> None:
    """La ficha de globales muestra conclusiones interpretadas."""
    blob = _markdown_blob(eda_app)
    for needle in (
        "AlphaEarth es la mejor base",
        "especialización por región",
        "máscara de calidad",
        "Lo que sigue",
    ):
        assert needle in blob, f"Falta conclusion: {needle}"


def test_footer_has_attributions(loaded_app: object) -> None:
    """AC-12: el footer contiene atribuciones de al menos 6 datasets."""
    at = loaded_app
    markdown_blob = " ".join(getattr(node, "value", "") for node in at.markdown)  # type: ignore[attr-defined]
    expected_datasets = (
        "PASTIS-R",
        "Sentinel-2",
        "AlphaEarth",
        "Dynamic World",
        "ERA5",
        "HCAT3",
    )
    missing = [name for name in expected_datasets if name not in markdown_blob]
    assert not missing, f"Datasets faltantes en footer: {missing}"


# ---------------------------------------------------------------------------
# Cache loaders (no requieren AppTest)
# ---------------------------------------------------------------------------


def test_load_csv_missing(tmp_path: Path) -> None:
    """``load_csv`` devuelve DataFrame vacio para rutas inexistentes."""
    eda_dashboard.load_csv.clear()
    df = eda_dashboard.load_csv(tmp_path / "missing.csv")
    assert df.is_empty()


def test_load_csv_valid(dummy_csv: Path) -> None:
    """``load_csv`` lee un CSV valido con las columnas esperadas."""
    eda_dashboard.load_csv.clear()
    df = eda_dashboard.load_csv(dummy_csv)
    assert not df.is_empty()
    assert set(df.columns) >= {"band", "mean", "std"}


def test_load_yaml_missing(tmp_path: Path) -> None:
    """``load_yaml`` devuelve dict vacio para rutas inexistentes."""
    eda_dashboard.load_yaml.clear()
    data = eda_dashboard.load_yaml(tmp_path / "missing.yaml")
    assert data == {}


def test_load_yaml_valid(tmp_path: Path) -> None:
    """``load_yaml`` lee y parsea correctamente un YAML simple."""
    eda_dashboard.load_yaml.clear()
    path = tmp_path / "config.yaml"
    path.write_text("rois:\n  - name: x\n    region: Italy\n", encoding="utf-8")
    data = eda_dashboard.load_yaml(path)
    assert "rois" in data and isinstance(data["rois"], list)


# ---------------------------------------------------------------------------
# KPI lookup (helper publico testeable sin Streamlit context)
# ---------------------------------------------------------------------------


def test_all_cards_expose_kpis_from_notebook_content() -> None:
    """Los 5 NotebookCard publicos exponen KPIs (fuente unica DRY)."""
    from ml.report.notebook_content import CARDS

    expected_ids = {
        "sentinel2",
        "alphaearth",
        "bivariate-temporal",
        "pastis-consolidado",
        "globales",
    }
    assert {card.notebook_id for card in CARDS} == expected_ids
    for card in CARDS:
        assert card.kpis, f"sin KPIs en {card.notebook_id}"


def test_kpi_entries_have_required_fields() -> None:
    """Cada KPI tiene label, value, delta no vacios."""
    from ml.report.notebook_content import CARDS

    for card in CARDS:
        for kpi in card.kpis:
            assert kpi.label.strip(), f"label vacio en {card.notebook_id}"
            assert kpi.value.strip(), f"value vacio en {card.notebook_id}"
            assert kpi.delta.strip(), f"delta vacio en {card.notebook_id}"
