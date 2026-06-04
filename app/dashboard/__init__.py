"""AgroSatCopilot Streamlit dashboard package.

Splits the historic monolith ``app/eda_dashboard.py`` into single-
responsibility modules (design system, loaders, render components,
per-Avance sections and data-driven registry). The entry point
``app/eda_dashboard.py`` remains a thin shim that assembles ``main()``
and re-exports the public API consumed by the tests.
"""
