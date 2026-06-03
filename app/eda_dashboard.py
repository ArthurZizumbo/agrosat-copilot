"""Entry point del dashboard Streamlit de AgroSatCopilot.

Shim delgado que ensambla ``main()`` a partir del paquete ``app.dashboard`` y
reexporta la API publica que consumen los tests y el deploy. La logica vive en
modulos con responsabilidad unica (``theme``, ``loaders``, ``components``,
``layout``, ``spatial``, ``timeline``, ``registry`` y ``sections/``).

El dashboard recorre la evolucion completa del proyecto (A1 EDA -> A2 FE ->
A3 Baseline -> A4 Segmentacion) con narrativa por figura y metricas reales.

Para arrancar localmente::

    poetry run streamlit run app/eda_dashboard.py --server.port 8501
"""

from __future__ import annotations

from app.dashboard.components import render_card  # noqa: F401  (reexport)
from app.dashboard.layout import (
    configure_page,
    render_footer,
    render_hero,
    render_sidebar,
)
from app.dashboard.loaders import list_csvs as _list_csvs  # noqa: F401  (reexport)
from app.dashboard.loaders import load_csv, load_parquet, load_yaml  # noqa: F401  (reexport)
from app.dashboard.paths import PAPER_FIGURES_ROOT as FIGURES_ROOT  # noqa: F401  (reexport)
from app.dashboard.paths import PASTIS_METADATA, ROIS_YAML  # noqa: F401  (reexport)
from app.dashboard.registry import (
    _SECTION_BASELINE,  # noqa: F401  (reexport)
    _SECTION_EDA,  # noqa: F401  (reexport)
    _SECTION_FE,  # noqa: F401  (reexport)
    _SECTION_HISTORIA,  # noqa: F401  (reexport)
    _SECTION_OPTIONS,  # noqa: F401  (reexport)
    _SECTION_SEGMENTATION,  # noqa: F401  (reexport)
    SECTION_STATE_KEY,
    SECTIONS,
    render_section_selector,
)
from app.dashboard.sections.baseline import (
    BASELINE_TAB_LABELS as _BASELINE_TAB_LABELS,  # noqa: F401  (reexport)
)
from app.dashboard.sections.baseline import (
    BASELINE_TAB_RENDERERS as _BASELINE_TAB_RENDERERS,  # noqa: F401  (reexport)
)
from app.dashboard.sections.baseline import (
    render_baseline_section as _render_baseline_section,  # noqa: F401  (reexport)
)
from app.dashboard.spatial import (
    build_folium_map,  # noqa: F401  (reexport)
    render_spatial_tab,
)
from app.dashboard.theme import inject_design_system
from ml.report.avance3_content import (
    BASELINE_MISSING_HINT as _BASELINE_MISSING_HINT,  # noqa: F401  (reexport)
)

# Historical alias of the state key for legacy tests.
_SECTION_STATE_KEY = SECTION_STATE_KEY

# Historical alias of the spatial tab (legacy signature of the tests).
render_tab_spatial = render_spatial_tab


def main() -> None:
    """Punto de entrada Streamlit: design system + selector + sidebar + seccion."""
    configure_page()
    inject_design_system()
    render_hero()

    selected = render_section_selector(SECTIONS)
    render_sidebar(SECTIONS, selected.key)
    selected.renderer()

    render_footer()


if __name__ == "__main__":  # pragma: no cover - streamlit entry point
    main()
