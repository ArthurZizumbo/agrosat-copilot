"""Paquete Dagster del proyecto AgroSatCopilot.

Expone ``defs`` a nivel de paquete para que ``dagster dev`` y el
``workspace.yaml`` puedan cargarlo via ``python_package: dagster_project``
sin necesidad de la opción ``-m dagster_project.definitions``.
"""

from dagster_project.definitions import defs

__all__ = ["defs"]
