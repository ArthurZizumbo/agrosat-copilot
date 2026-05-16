"""Entry point para Streamlit Community Cloud.

Streamlit Cloud lee este archivo + ``requirements.txt`` + ``runtime.txt`` +
``packages.txt`` desde el mismo directorio. Este entry point ajusta
``sys.path`` para que el dashboard pueda importar desde la raíz del
repositorio y luego ejecuta el script real (``app/eda_dashboard.py``).

Configuración en Streamlit Cloud:
    Repository:     ArthurZizumbo/agrosat-copilot
    Branch:         us-013
    Main file path: deploy/streamlit/streamlit_app.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

# Subimos dos niveles: deploy/streamlit/streamlit_app.py -> raíz del repo
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Ejecuta el dashboard real con ``__name__ == "__main__"`` para que
# Streamlit lo interprete como script de entrada.
_DASHBOARD = _REPO_ROOT / "app" / "eda_dashboard.py"
runpy.run_path(str(_DASHBOARD), run_name="__main__")
