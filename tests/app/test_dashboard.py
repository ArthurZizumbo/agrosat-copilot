"""Smoke tests para ``app/eda_dashboard.py`` con ``streamlit.testing.v1.AppTest``.

Cobertura objetivo: >= 70 % sobre ``app/eda_dashboard.py``.

Estos tests cargan el dashboard en modo headless con la API oficial
``streamlit.testing.v1.AppTest`` (estable desde Streamlit 1.28). Si Streamlit
no está instalado en el entorno (caso bootstrap inicial sin grupo ``paper``),
todos los tests se skipean en lugar de fallar.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

# Streamlit es opcional para el repo base; se instala via grupo `paper`.
# Si no está disponible, skip toda la suite con un mensaje claro.
streamlit = pytest.importorskip("streamlit", reason="Streamlit no instalado (grupo paper).")
AppTest = pytest.importorskip(
    "streamlit.testing.v1", reason="streamlit.testing.v1 requiere Streamlit >= 1.28."
).AppTest

# Importes del módulo bajo prueba — protegidos por importorskip arriba.
from app import eda_dashboard  # noqa: E402

DASHBOARD_PATH = Path(__file__).resolve().parents[2] / "app" / "eda_dashboard.py"


# ---------------------------------------------------------------------------
# Fixtures sintéticos
# ---------------------------------------------------------------------------

# PNG mínimo válido (1x1 px, RGBA). Suficiente para que st.image lo acepte.
_PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


@pytest.fixture
def dummy_png(tmp_path: Path) -> Path:
    """Crea un PNG mínimo válido en ``tmp_path/sample.png``."""
    target = tmp_path / "sample.png"
    target.write_bytes(_PNG_1X1)
    return target


@pytest.fixture
def dummy_csv(tmp_path: Path) -> Path:
    """Crea un CSV de 2 filas dummy en ``tmp_path/sample.csv``."""
    target = tmp_path / "sample.csv"
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["band", "mean", "std"])
        writer.writerow(["B04", "1234.5", "456.7"])
        writer.writerow(["B08", "2345.6", "567.8"])
    return target


# ---------------------------------------------------------------------------
# Tests del dashboard completo (AppTest)
# ---------------------------------------------------------------------------


def test_dashboard_loads() -> None:
    """AC-1: el dashboard arranca sin excepción y expone los 6 tabs."""
    at = AppTest.from_file(str(DASHBOARD_PATH)).run(timeout=30)
    assert not at.exception, f"Excepción al cargar: {at.exception}"
    # 6 tabs canónicos (AC-2 a AC-7).
    assert len(at.tabs) == 6, f"Se esperaban 6 tabs, hay {len(at.tabs)}"


def test_dashboard_sidebar_has_title() -> None:
    """La sidebar muestra el título canónico del Avance 1."""
    at = AppTest.from_file(str(DASHBOARD_PATH)).run(timeout=30)
    sidebar_text = " ".join(getattr(node, "value", "") for node in at.sidebar.title)
    assert "Avance 1" in sidebar_text
    assert "AgroSatCopilot" in sidebar_text


def test_footer_has_attributions() -> None:
    """AC-12: el footer contiene atribuciones de al menos 6 datasets."""
    at = AppTest.from_file(str(DASHBOARD_PATH)).run(timeout=30)
    # Concatenar todo el texto markdown renderizado.
    markdown_blob = " ".join(getattr(node, "value", "") for node in at.markdown)
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
# Tests directos de cada render_tab_X — placeholder vacío
# ---------------------------------------------------------------------------


def test_render_tab_univariate_empty(tmp_path: Path) -> None:
    """``render_tab_univariate`` con dir vacío muestra placeholder sin crash."""
    at = AppTest.from_function(eda_dashboard.render_tab_univariate, args=(tmp_path,)).run(
        timeout=15
    )
    assert not at.exception
    info_blob = " ".join(getattr(node, "value", "") for node in at.info)
    assert "US-010" in info_blob


def test_render_tab_alphaearth_empty(tmp_path: Path) -> None:
    """``render_tab_alphaearth`` con dir vacío muestra placeholder US-011."""
    at = AppTest.from_function(eda_dashboard.render_tab_alphaearth, args=(tmp_path,)).run(
        timeout=15
    )
    assert not at.exception
    info_blob = " ".join(getattr(node, "value", "") for node in at.info)
    assert "US-011" in info_blob


def test_render_tab_bivariate_empty(tmp_path: Path) -> None:
    """``render_tab_bivariate`` con dir vacío muestra placeholder US-012."""
    at = AppTest.from_function(eda_dashboard.render_tab_bivariate, args=(tmp_path,)).run(timeout=15)
    assert not at.exception
    info_blob = " ".join(getattr(node, "value", "") for node in at.info)
    assert "US-012" in info_blob


def test_render_tab_temporal_empty(tmp_path: Path) -> None:
    """``render_tab_temporal`` con dir vacío muestra placeholder US-012."""
    at = AppTest.from_function(eda_dashboard.render_tab_temporal, args=(tmp_path,)).run(timeout=15)
    assert not at.exception
    info_blob = " ".join(getattr(node, "value", "") for node in at.info)
    assert "US-012" in info_blob


def test_render_tab_spatial_empty(tmp_path: Path) -> None:
    """``render_tab_spatial`` con rutas inexistentes no crashea.

    Si streamlit-folium no está instalado, debe mostrar st.info; si lo está,
    debe mostrar un warning sobre el YAML ausente.
    """
    rois = tmp_path / "rois.yaml"
    pastis = tmp_path / "metadata.geojson"
    at = AppTest.from_function(eda_dashboard.render_tab_spatial, args=(rois, pastis)).run(
        timeout=15
    )
    assert not at.exception


def test_render_tab_conclusions_renders() -> None:
    """``render_tab_conclusions`` renderiza las 6 sub-secciones CRISP-ML(Q)."""
    at = AppTest.from_function(eda_dashboard.render_tab_conclusions).run(timeout=15)
    assert not at.exception
    subheaders = " ".join(getattr(node, "value", "") for node in at.subheader)
    expected = (
        "Business Understanding",
        "Data Understanding",
        "Data Preparation",
        "Modeling",
        "Evaluation",
        "Deployment",
    )
    missing = [phase for phase in expected if phase not in subheaders]
    assert not missing, f"Faltan fases CRISP-ML(Q): {missing}"


# ---------------------------------------------------------------------------
# Tests con figuras dummy (asegura el path "con contenido")
# ---------------------------------------------------------------------------


def _seed_figures(directory: Path, *, prefix: str = "") -> None:
    """Crea figuras y CSVs dummy en ``directory`` para satisfacer el render."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{prefix}sample.png").write_bytes(_PNG_1X1)
    csv_path = directory / f"{prefix}sample.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["k", "v"])
        writer.writerow(["a", "1"])


def test_render_tab_univariate_with_figures(tmp_path: Path) -> None:
    """``render_tab_univariate`` con 1 PNG + 1 CSV dummy renderiza sin error."""
    _seed_figures(tmp_path)
    at = AppTest.from_function(eda_dashboard.render_tab_univariate, args=(tmp_path,)).run(
        timeout=15
    )
    assert not at.exception


def test_render_tab_alphaearth_with_figures(tmp_path: Path) -> None:
    """``render_tab_alphaearth`` con sec1_ / sec2_ / sec3_ PNGs renderiza."""
    _seed_figures(tmp_path, prefix="sec1_")
    _seed_figures(tmp_path, prefix="sec2_")
    _seed_figures(tmp_path, prefix="sec3_")
    at = AppTest.from_function(eda_dashboard.render_tab_alphaearth, args=(tmp_path,)).run(
        timeout=15
    )
    assert not at.exception


def test_render_tab_bivariate_with_figures(tmp_path: Path) -> None:
    """``render_tab_bivariate`` con ``correlation_*.png`` + ``vif_*.csv``."""
    (tmp_path / "correlation_heat.png").write_bytes(_PNG_1X1)
    csv_path = tmp_path / "vif_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["feature", "vif"])
        writer.writerow(["ndvi", "2.1"])
    at = AppTest.from_function(eda_dashboard.render_tab_bivariate, args=(tmp_path,)).run(timeout=15)
    assert not at.exception


def test_render_tab_temporal_with_figures(tmp_path: Path) -> None:
    """``render_tab_temporal`` con peak_ndvi/acf/dtw/era5 PNGs renderiza."""
    for stem in ("peak_ndvi", "acf_grid", "dtw_centroids", "era5_ndvi_dual"):
        (tmp_path / f"{stem}.png").write_bytes(_PNG_1X1)
    at = AppTest.from_function(eda_dashboard.render_tab_temporal, args=(tmp_path,)).run(timeout=15)
    assert not at.exception


# ---------------------------------------------------------------------------
# Tests de utilidades cacheadas
# ---------------------------------------------------------------------------


def test_load_csv_missing(tmp_path: Path) -> None:
    """``load_csv`` devuelve DataFrame vacío para rutas inexistentes."""
    eda_dashboard.load_csv.clear()
    df = eda_dashboard.load_csv(tmp_path / "missing.csv")
    assert df.is_empty()


def test_load_csv_valid(dummy_csv: Path) -> None:
    """``load_csv`` lee un CSV válido con las columnas esperadas."""
    eda_dashboard.load_csv.clear()
    df = eda_dashboard.load_csv(dummy_csv)
    assert not df.is_empty()
    assert set(df.columns) >= {"band", "mean", "std"}


def test_load_yaml_missing(tmp_path: Path) -> None:
    """``load_yaml`` devuelve dict vacío para rutas inexistentes."""
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
