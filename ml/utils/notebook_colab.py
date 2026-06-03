"""Bootstrap de notebooks de segmentacion que corren en Colab o en local.

Centraliza la celda de setup que de otra forma se duplica en cada notebook de
segmentacion (`04d`, `04e`, `04g`, `04h`, `Avance4.Equipo17`):

- Monta Google Drive (donde vive el dataset compartido del equipo) cuando se
  ejecuta en Colab; en local es un no-op.
- Localiza el repo por su `pyproject.toml`, lo agrega a `sys.path` y fija el CWD.
- Instala bajo demanda las dependencias que Colab no trae por defecto.

El clone inicial del repo en Colab NO vive aqui: precede al `import`, asi que se
queda en una celda minima del notebook. Una vez clonado el repo (o en local),
``bootstrap_colab_run`` hace el resto del trabajo.

Uso tipico en la celda de setup del notebook:

```python
from ml.utils.notebook_colab import bootstrap_colab_run

env = bootstrap_colab_run(pip_packages=("segmentation-models-pytorch", "polars"))
shared_folder_path = env.shared_folder_path
```
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ColabEnv", "bootstrap_colab_run"]

#: Drive path of the team shared folder mounted in Colab.
_SHARED_FOLDER = "/content/drive/MyDrive/Integrador/"


@dataclass(frozen=True)
class ColabEnv:
    """Entorno resuelto por el bootstrap de la celda de setup.

    Attributes:
        repo: Ruta absoluta al repo root (resuelto via `pyproject.toml`).
        in_colab: True si el notebook corre dentro de Google Colab.
        shared_folder_path: Prefijo del Drive compartido (`""` en local).
    """

    repo: Path
    in_colab: bool
    shared_folder_path: str


def _mount_drive() -> tuple[bool, str]:
    """Monta Google Drive si estamos en Colab.

    Returns:
        Par ``(in_colab, shared_folder_path)``. En local devuelve
        ``(False, "")`` sin efectos secundarios.
    """
    try:
        from google.colab import drive  # type: ignore[import-not-found]
    except ImportError:
        return False, ""
    drive.mount("/content/drive")
    return True, _SHARED_FOLDER


def _locate_repo(in_colab: bool) -> Path:
    """Localiza el repo por su `pyproject.toml`, lo agrega a `sys.path` y fija CWD.

    Args:
        in_colab: Si True antepone `/content/agrosat-copilot` (destino del clone).

    Returns:
        Ruta absoluta al repo root.

    Raises:
        RuntimeError: Si no se encuentra `pyproject.toml` en ningun candidato.
    """
    search = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    if in_colab:
        search = [Path("/content/agrosat-copilot"), *search]
    for cand in search:
        if (cand / "pyproject.toml").is_file():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            os.chdir(cand)
            return cand
    raise RuntimeError(
        "No se encontro el repo agrosat-copilot (pyproject.toml). "
        "Clonalo en /content/agrosat-copilot o sincronizalo desde VS Code."
    )


def bootstrap_colab_run(
    *,
    pip_packages: tuple[str, ...] = (),
    require_repo: bool = True,
) -> ColabEnv:
    """Aplica el bootstrap de la celda de setup y devuelve el entorno.

    Args:
        pip_packages: Paquetes a instalar via pip solo cuando se corre en Colab
            (en local se asume el venv del repo). Lista vacia = no instala nada.
        require_repo: Si True (default, notebooks de modelo) levanta
            ``RuntimeError`` cuando no encuentra el repo. Si False (notebook
            integrador) usa el CWD actual como fallback sin romper.

    Returns:
        ``ColabEnv`` con repo root, flag de Colab y prefijo del Drive compartido.
    """
    in_colab, shared_folder_path = _mount_drive()

    try:
        repo = _locate_repo(in_colab)
    except RuntimeError:
        if require_repo:
            raise
        repo = Path.cwd().resolve()

    if in_colab and pip_packages:
        subprocess.run(  # noqa: S603 - fixed argv, sys.executable, no shell
            [sys.executable, "-m", "pip", "-q", "install", *pip_packages],
            check=False,
        )

    return ColabEnv(repo=repo, in_colab=in_colab, shared_folder_path=shared_folder_path)
