"""Reusable helpers for the EDA notebooks.

Centralizes three patterns that would otherwise be duplicated in each `.ipynb`:

- Robust repo root resolution via `pyproject.toml` (independent of the CWD).
- Loading `.env.local` without the `python-dotenv` dependency.
- Configuration of credentials for Google Earth Engine (ADC vs service account).

The functions are pure (input -> output, no global state) and follow the style
of the rest of `ml/utils/`.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up levels from `start` until `pyproject.toml` is found.

    Useful for notebooks: works from any subdirectory of the repo
    (`notebooks/`, `notebooks/eda/`, `scripts/`, etc.) without assuming a
    specific folder name.

    Args:
        start: Starting point. If None uses `Path.cwd()`.

    Returns:
        Absolute path to the repo root. If `pyproject.toml` is not found in
        any ancestor, returns `start.resolve()` as fallback (degraded mode
        so the notebook does not break).
    """
    base = (start or Path.cwd()).resolve()
    for candidate in (base, *base.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return base


def load_env_local(repo_root: Path) -> None:
    """Load `KEY=VALUE` pairs from `.env.local` into `os.environ`.

    Does not overwrite variables already present. Ignores empty lines and
    comments. No `python-dotenv` dependency.

    Args:
        repo_root: Absolute path to the repo root (use `find_repo_root()`).
    """
    env_file = repo_root / ".env.local"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.split("#", 1)[0].strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def configure_ee_from_env(repo_root: Path) -> tuple[str | None, Path | None]:
    """Read `.env.local` and prepare credentials for Earth Engine.

    Does three things:

    1. Calls `load_env_local(repo_root)` to populate `os.environ`.
    2. If `GOOGLE_APPLICATION_CREDENTIALS` points to a non-existent file
       (placeholder from the `.env.local` template), removes it so
       `google-auth` falls back cleanly to the gcloud ADC.
    3. Returns `(GEE_PROJECT_ID, service_account_path_or_None)` ready to
       pass to `init_ee(service_account_json=..., project=...)`.

    Args:
        repo_root: Path to the repo root.

    Returns:
        Tuple `(gee_project, sa_json_path)`. Both may be None if the
        variables are not set or the SA file does not exist.
    """
    load_env_local(repo_root)

    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac and not Path(gac).is_file():
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

    gee_project = os.environ.get("GEE_PROJECT_ID") or None
    gee_sa_path = os.environ.get("GEE_SERVICE_ACCOUNT_PATH")
    sa_json = Path(gee_sa_path) if gee_sa_path and Path(gee_sa_path).is_file() else None

    return gee_project, sa_json


def show_saved_png(path: Path, caption: str | None = None) -> None:
    """Display inline in Jupyter a PNG already written to disk.

    Useful when a plotter function does `fig.savefig(...) + plt.close(fig)`:
    `display(fig)` no longer renders because the matplotlib backend closed the
    figure, but the PNG exists. This function loads it via
    `IPython.display.Image` and shows an optional bold caption as preceding
    Markdown.

    Args:
        path: Path to the PNG.
        caption: Bold text to show before the image.
    """
    from IPython.display import Image, Markdown, display

    if not path.is_file():
        display(Markdown(f"> Figura `{path.name}` no disponible (no se genero)."))
        return
    if caption:
        display(Markdown(f"**{caption}**"))
    display(Image(filename=str(path)))
