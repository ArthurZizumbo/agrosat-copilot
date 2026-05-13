"""Tests para ml.report.export_pdf - cobertura objetivo >= 70 %.

Verifica:
    - Agrupacion de figuras por subdirectorio (AC-8).
    - Renderizado Jinja2 con autoescape.
    - Generacion de PDF valido si WeasyPrint esta disponible (AC-8).
    - Manejo limpio de error cuando GTK falta en Windows.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from ml.report import export_pdf
from ml.report.export_pdf import (
    _collect_figures,
    _render_html,
    app,
)

WEASYPRINT_AVAILABLE = importlib.util.find_spec("weasyprint") is not None
PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff"
    b"\xff?\x00\x05\xfe\x02\xfe\xa3R\xafG\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.fixture
def figures_dir(tmp_path: Path) -> Path:
    """Crea un figures_dir con 3 subdirs (us-010/011/012) y un PNG en cada uno."""
    base = tmp_path / "figures"
    for us in ("us-010", "us-011", "us-012"):
        sub = base / us
        sub.mkdir(parents=True)
        (sub / f"{us}_plot.png").write_bytes(PNG_BYTES)
    return base


def test_collect_figures_groups_by_us(figures_dir: Path) -> None:
    """Verifica que _collect_figures agrupa por subdirectorio us-XXX."""
    figures = _collect_figures(figures_dir)

    assert set(figures.keys()) == {"us_010", "us_011", "us_012"}
    assert len(figures["us_010"]) == 1
    assert len(figures["us_011"]) == 1
    assert len(figures["us_012"]) == 1
    assert figures["us_010"][0].name == "us-010_plot.png"
    assert figures["us_011"][0].name == "us-011_plot.png"


def test_collect_figures_sorted_alphabetically(tmp_path: Path) -> None:
    """Verifica orden alfabetico para reproducibilidad del PDF."""
    base = tmp_path / "figures"
    sub = base / "us-010"
    sub.mkdir(parents=True)
    for name in ["zeta.png", "alpha.png", "mango.png"]:
        (sub / name).write_bytes(PNG_BYTES)

    figures = _collect_figures(base)
    names = [p.name for p in figures["us_010"]]
    assert names == ["alpha.png", "mango.png", "zeta.png"]


def test_collect_figures_missing_subdir_returns_empty(tmp_path: Path) -> None:
    """Subdirs ausentes producen listas vacias, no errores."""
    base = tmp_path / "figures"
    base.mkdir()
    (base / "us-010").mkdir()
    (base / "us-010" / "x.png").write_bytes(PNG_BYTES)

    figures = _collect_figures(base)
    assert len(figures["us_010"]) == 1
    assert figures["us_011"] == []
    assert figures["us_012"] == []


def test_render_html_uses_template(tmp_path: Path) -> None:
    """Verifica renderizado Jinja2 con un template inline minimo."""
    template = tmp_path / "mini.html.j2"
    template.write_text(
        "<html><body><h1>{{ title }}</h1>"
        "{% for f in figures.us_010 %}<img src='{{ f.name }}' />{% endfor %}"
        "</body></html>",
        encoding="utf-8",
    )
    context = {
        "title": "Test Report",
        "figures": {"us_010": [Path("a.png"), Path("b.png")]},
    }
    html = _render_html(template, context)

    assert "Test Report" in html
    assert "a.png" in html
    assert "b.png" in html
    assert html.count("<img") == 2


def test_render_html_autoescape_active(tmp_path: Path) -> None:
    """Confirma que autoescape Jinja2 escapa HTML en variables (seguridad XSS)."""
    template = tmp_path / "x.html.j2"
    template.write_text("<p>{{ title }}</p>", encoding="utf-8")
    html = _render_html(template, {"title": "<script>alert(1)</script>"})

    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_cli_template_not_found(tmp_path: Path) -> None:
    """CLI debe fallar con exit code 2 si el template no existe."""
    runner = CliRunner()
    output = tmp_path / "out.pdf"
    missing_template = tmp_path / "missing.html.j2"

    result = runner.invoke(
        app,
        [
            "--output",
            str(output),
            "--figures-dir",
            str(tmp_path),
            "--template",
            str(missing_template),
        ],
    )
    assert result.exit_code == 2


@pytest.mark.skipif(not WEASYPRINT_AVAILABLE, reason="weasyprint no instalado")
def test_export_pdf_generates_valid_pdf(tmp_path: Path, figures_dir: Path) -> None:
    """Genera un PDF real con WeasyPrint y verifica tamano minimo (AC-8)."""
    repo_root = Path(__file__).resolve().parents[3]
    template = repo_root / "ml" / "report" / "templates" / "avance1_eda.html.j2"
    output = tmp_path / "out.pdf"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--output",
            str(output),
            "--figures-dir",
            str(figures_dir),
            "--template",
            str(template),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert output.stat().st_size > 1000


def test_html_to_pdf_handles_missing_gtk(tmp_path: Path, figures_dir: Path) -> None:
    """Mockea OSError de WeasyPrint (GTK ausente Windows) y verifica mensaje claro."""
    repo_root = Path(__file__).resolve().parents[3]
    template = repo_root / "ml" / "report" / "templates" / "avance1_eda.html.j2"
    output = tmp_path / "out.pdf"

    fake_weasyprint = MagicMock()
    fake_html_instance = MagicMock()
    fake_html_instance.write_pdf.side_effect = OSError("cannot load library 'libgobject-2.0-0'")
    fake_weasyprint.HTML.return_value = fake_html_instance
    fake_weasyprint.CSS.return_value = MagicMock()

    runner = CliRunner()
    with patch.dict(
        "sys.modules",
        {"weasyprint": fake_weasyprint},
    ):
        result = runner.invoke(
            app,
            [
                "--output",
                str(output),
                "--figures-dir",
                str(figures_dir),
                "--template",
                str(template),
            ],
        )

    assert result.exit_code != 0
    assert "GTK" in result.output or "WeasyPrint" in result.output


def test_html_to_pdf_runtime_when_weasyprint_missing(tmp_path: Path) -> None:
    """_html_to_pdf debe lanzar RuntimeError si weasyprint no es importable."""
    with patch.dict("sys.modules", {"weasyprint": None}):
        with pytest.raises(RuntimeError, match="WeasyPrint no esta instalado"):
            export_pdf._html_to_pdf(
                html_str="<html></html>",
                output=tmp_path / "x.pdf",
                css_path=tmp_path / "styles.css",
                base_url=tmp_path,
            )
