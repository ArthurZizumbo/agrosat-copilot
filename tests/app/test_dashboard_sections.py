"""Smoke tests para las secciones nuevas del dashboard (A4 + Historia).

Verifica que el registro de secciones sea data-driven (5 secciones) y que
seleccionar la landing Historia o la seccion Segmentacion no rompa el render.
"""

from __future__ import annotations

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit", reason="Streamlit no instalado (grupo paper).")
AppTest = pytest.importorskip(
    "streamlit.testing.v1", reason="streamlit.testing.v1 requiere Streamlit >= 1.28."
).AppTest

from app import eda_dashboard  # noqa: E402

DASHBOARD_PATH = Path(__file__).resolve().parents[2] / "app" / "eda_dashboard.py"


def test_registry_has_five_sections() -> None:
    """El registro de secciones expone Historia + 4 Avances (5 secciones)."""
    labels = [section.label for section in eda_dashboard.SECTIONS]
    assert len(labels) == 5, f"Se esperaban 5 secciones, hay {len(labels)}: {labels}"
    assert eda_dashboard._SECTION_HISTORIA == labels[0], "Historia debe ser la landing"
    assert eda_dashboard._SECTION_SEGMENTATION in labels, "Falta la seccion Segmentacion"


def test_all_sections_have_callable_renderer() -> None:
    """Cada seccion del registro tiene un renderer invocable."""
    for section in eda_dashboard.SECTIONS:
        assert callable(section.renderer), f"renderer no invocable en {section.key}"
        assert section.key, "key vacio en una seccion"
        assert section.label, "label vacio en una seccion"


def test_segmentation_section_renders_via_apptest() -> None:
    """Seleccionar la seccion Segmentacion (A4) no levanta excepcion."""
    at = AppTest.from_file(str(DASHBOARD_PATH)).run(timeout=60)
    assert not at.exception, f"Excepcion al cargar dashboard: {at.exception}"  # type: ignore[attr-defined]

    at.session_state[eda_dashboard._SECTION_STATE_KEY] = eda_dashboard._SECTION_SEGMENTATION
    at.run(timeout=60)
    assert not at.exception, f"Excepcion al seleccionar Segmentacion: {at.exception}"  # type: ignore[attr-defined]

    blob = " ".join(getattr(node, "value", "") for node in at.markdown)  # type: ignore[attr-defined]
    assert "Segmentacion semantica densa" in blob, "Falta el titulo del Avance 4"


def test_landing_timeline_renders_via_apptest() -> None:
    """La landing por defecto (Historia) muestra los hitos del proyecto."""
    at = AppTest.from_file(str(DASHBOARD_PATH)).run(timeout=60)
    assert not at.exception, f"Excepcion al cargar landing: {at.exception}"  # type: ignore[attr-defined]
    blob = " ".join(getattr(node, "value", "") for node in at.markdown)  # type: ignore[attr-defined]
    assert "Historia del proyecto" in blob
