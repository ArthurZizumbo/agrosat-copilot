"""Configuracion pytest comun para los tests de la capa ML.

Fuerza el backend `Agg` de matplotlib antes de que cualquier modulo importe
pyplot, para evitar errores `_tkinter.TclError` en Windows cuando varios tests
graficos corren en la misma sesion (cross-test pollution con Tk).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
