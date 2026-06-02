"""Secciones por Avance del dashboard.

Cada modulo expone un ``render_*_section()`` sin argumentos que el registro
(``app.dashboard.registry``) asocia a una entrada del selector. Agregar un
Avance nuevo se reduce a crear un modulo de contenido y registrar su renderer.
"""
