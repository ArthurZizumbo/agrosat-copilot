"""Dagster package of the AgroSatCopilot project.

Exposes ``defs`` at the package level so that ``dagster dev`` and the
``workspace.yaml`` can load it via ``python_package: dagster_project``
without needing the ``-m dagster_project.definitions`` option.
"""

from dagster_project.definitions import defs

__all__ = ["defs"]
