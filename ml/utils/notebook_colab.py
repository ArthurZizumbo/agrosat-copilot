"""Bootstrap for segmentation notebooks that run on Colab or locally.

Centralizes the setup cell that would otherwise be duplicated in each
segmentation notebook (`04d`, `04e`, `04g`, `04h`, `Avance4.Equipo17`):

- Mounts Google Drive (where the team shared dataset lives) when running
  in Colab; locally it is a no-op.
- Locates the repo by its `pyproject.toml`, adds it to `sys.path` and sets the CWD.
- Installs on demand the dependencies that Colab does not bring by default.

The initial repo clone on Colab does NOT live here: it precedes the `import`,
so it stays in a minimal notebook cell. Once the repo is cloned (or locally),
``bootstrap_colab_run`` does the rest of the work.

Typical usage in the notebook setup cell:

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
    """Environment resolved by the bootstrap of the setup cell.

    Attributes:
        repo: Absolute path to the repo root (resolved via `pyproject.toml`).
        in_colab: True if the notebook runs inside Google Colab.
        shared_folder_path: Prefix of the shared Drive (`""` locally).
    """

    repo: Path
    in_colab: bool
    shared_folder_path: str


def _mount_drive() -> tuple[bool, str]:
    """Mount Google Drive if we are in Colab.

    Returns:
        Pair ``(in_colab, shared_folder_path)``. Locally returns
        ``(False, "")`` with no side effects.
    """
    try:
        from google.colab import drive  # type: ignore[import-not-found]
    except ImportError:
        return False, ""
    drive.mount("/content/drive")
    return True, _SHARED_FOLDER


def _locate_repo(in_colab: bool) -> Path:
    """Locate the repo by its `pyproject.toml`, add it to `sys.path` and set CWD.

    Args:
        in_colab: If True prepends `/content/agrosat-copilot` (the clone target).

    Returns:
        Absolute path to the repo root.

    Raises:
        RuntimeError: If `pyproject.toml` is not found in any candidate.
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
        "Could not find the agrosat-copilot repo (pyproject.toml). "
        "Clone it into /content/agrosat-copilot or sync it from VS Code."
    )


def bootstrap_colab_run(
    *,
    pip_packages: tuple[str, ...] = (),
    require_repo: bool = True,
) -> ColabEnv:
    """Apply the bootstrap of the setup cell and return the environment.

    Args:
        pip_packages: Packages to install via pip only when running in Colab
            (locally the repo venv is assumed). Empty list = installs nothing.
        require_repo: If True (default, model notebooks) raises
            ``RuntimeError`` when it does not find the repo. If False (the
            integrator notebook) uses the current CWD as a fallback without breaking.

    Returns:
        A ``ColabEnv`` with repo root, Colab flag and shared Drive prefix.
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
