"""Pytest config: añade la raíz del repo al sys.path para imports `ml.*`.

Sin un `src layout` ni instalación editable (`pip install -e .`), pytest no
encuentra paquetes top-level del repo. Esta configuración inyecta la raíz
del repositorio al `sys.path` antes de la colección de tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
