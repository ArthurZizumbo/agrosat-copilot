"""Per-Avance sections of the dashboard.

Each module exposes an argument-less ``render_*_section()`` that the registry
(``app.dashboard.registry``) associates with a selector entry. Adding a new
Avance reduces to creating a content module and registering its renderer.
"""
