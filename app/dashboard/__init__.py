"""Paquete del dashboard Streamlit de AgroSatCopilot.

Separa el monolito historico ``app/eda_dashboard.py`` en modulos con
responsabilidad unica (design system, loaders, componentes de render,
secciones por Avance y registro data-driven). El entry point
``app/eda_dashboard.py`` queda como un shim delgado que ensambla ``main()``
y reexporta la API publica que consumen los tests.
"""
