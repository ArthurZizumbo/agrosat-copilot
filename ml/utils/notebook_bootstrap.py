"""Bootstrap canonico para notebooks del proyecto AgroSatCopilot.

Centraliza el patron que de otra forma se duplica en cada `.ipynb`:

- Resolucion robusta del repo root (re-export de `notebook_setup.find_repo_root`).
- Carga de `.env.local`.
- Configuracion de `sys.path` para que `import ml.*` funcione desde cualquier
  subcarpeta de `notebooks/`.
- Configuracion de Polars (rendering rico HTML), matplotlib (DPI, inline) y
  autoreload (`%autoreload 2`).
- Creacion del directorio de figuras del notebook (`paper/figures/{slug}`).

Devuelve un dataclass `NotebookEnv` con los paths utiles para que cada notebook
no tenga que reconstruirlos.

Uso tipico en la celda 3 del notebook:

```python
from ml.utils.notebook_bootstrap import setup_notebook

env = setup_notebook(figures_subdir="us-023-preview/04_baseline")
display(env.summary_markdown())
```
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ml.utils.notebook_setup import (
    configure_ee_from_env,
    find_repo_root,
    load_env_local,
)

if TYPE_CHECKING:
    from IPython.core.interactiveshell import InteractiveShell

__all__ = ["NotebookEnv", "setup_notebook"]


@dataclass(frozen=True)
class NotebookEnv:
    """Paths y configuracion derivados del bootstrap del notebook.

    Attributes:
        repo: Ruta absoluta al repo root (resuelto via `pyproject.toml`).
        figures_dir: Directorio donde el notebook persiste plots PNG.
        reports_dir: Directorio donde el notebook persiste tablas y parquets.
        data_dir: Atajo a `repo / "data"`.
        cache_dir: Atajo a `repo / "data/cache"`.
        has_gemini_api_key: True si `GEMINI_API_KEY` o `GOOGLE_API_KEY` o
            `GOOGLE_GENAI_USE_VERTEXAI=true` estan presentes en el env.
        has_ee_credentials: True si Earth Engine puede iniciarse con SA o ADC.
        gee_project: Proyecto GCP para EE (puede ser None).
        gee_sa_path: Path al JSON de la service account de EE (puede ser None).
        env_warnings: Lista de mensajes accionables para el usuario.
    """

    repo: Path
    figures_dir: Path
    reports_dir: Path
    data_dir: Path
    cache_dir: Path
    has_gemini_api_key: bool
    has_ee_credentials: bool
    gee_project: str | None
    gee_sa_path: Path | None
    env_warnings: list[str] = field(default_factory=list)

    def summary_markdown(self) -> str:
        """Construye un resumen Markdown legible para `display(Markdown(...))`.

        Returns:
            Texto Markdown con tabla de paths y estado de credenciales.
        """
        rows = [
            "| Recurso | Estado |",
            "|---|---|",
            f"| Repo root | `{self.repo}` |",
            f"| Figures dir | `{self.figures_dir.relative_to(self.repo)}` |",
            f"| Reports dir | `{self.reports_dir.relative_to(self.repo)}` |",
            f"| Gemini API key | {'presente' if self.has_gemini_api_key else 'ausente'} |",
            f"| Earth Engine | {'configurado' if self.has_ee_credentials else 'no configurado'} |",
        ]
        if self.gee_project:
            rows.append(f"| GEE project | `{self.gee_project}` |")
        text = "\n".join(rows)
        if self.env_warnings:
            text += "\n\n**Avisos**:\n" + "\n".join(f"- {w}" for w in self.env_warnings)
        return text


def setup_notebook(
    figures_subdir: str = "default",
    reports_subdir: str = "default",
    *,
    enable_autoreload: bool = True,
    matplotlib_inline: bool = True,
    polars_rich_html: bool = True,
    load_dotenv: bool = True,
    ipython: InteractiveShell | None = None,
) -> NotebookEnv:
    """Aplica el bootstrap canonico y devuelve el entorno listo para usar.

    Sigue el orden documentado en `notebooks/CLAUDE.md` Seccion "Estructura
    estandar de notebook" Celda 3.

    Args:
        figures_subdir: Subcarpeta bajo `paper/figures/` para los PNG. La
            ruta efectiva queda en `env.figures_dir`.
        reports_subdir: Subcarpeta bajo `reports/` para tablas/parquets.
        enable_autoreload: Si True ejecuta `%load_ext autoreload` y
            `%autoreload 2` (cambios en `ml/*.py` se reflejan sin reiniciar
            el kernel).
        matplotlib_inline: Si True ejecuta `%matplotlib inline`.
        polars_rich_html: Si True configura Polars para render HTML formateado
            (`ASCII_MARKDOWN`, 20 filas, 60 chars).
        load_dotenv: Si True (default) carga `.env.local` en `os.environ`.
            Tests deterministas pueden ponerlo a False para no sobrescribir
            sus monkeypatches.
        ipython: Shell de IPython (auto-detectada si None). Se usa para los
            magics `%load_ext`, `%autoreload`, `%matplotlib`. Si no estamos
            dentro de IPython, los magics se omiten silenciosamente.

    Returns:
        `NotebookEnv` con paths resueltos y status de credenciales.
    """
    repo = find_repo_root()

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    if load_dotenv:
        load_env_local(repo)
        gee_project, gee_sa_path = configure_ee_from_env(repo)
    else:
        gee_project = os.environ.get("GEE_PROJECT_ID") or None
        gee_sa_env = os.environ.get("GEE_SERVICE_ACCOUNT_PATH")
        gee_sa_path = (
            Path(gee_sa_env) if gee_sa_env and Path(gee_sa_env).is_file() else None
        )

    figures_dir = repo / "paper" / "figures" / figures_subdir
    reports_dir = repo / "reports" / reports_subdir
    data_dir = repo / "data"
    cache_dir = data_dir / "cache"

    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if polars_rich_html:
        _configure_polars()

    if matplotlib_inline:
        _configure_matplotlib(ipython)

    if enable_autoreload:
        _enable_autoreload(ipython)

    has_gemini_api_key = _detect_gemini_credentials()
    has_ee_credentials = gee_project is not None or gee_sa_path is not None

    env_warnings: list[str] = []
    if not has_gemini_api_key:
        env_warnings.append(
            "Gemini API key ausente. Exporta `GEMINI_API_KEY` en `.env.local` "
            "antes de ejecutar celdas que llamen a `materialize_phenology_text`."
        )
    if not has_ee_credentials:
        env_warnings.append(
            "Earth Engine no configurado. Define `GEE_PROJECT_ID` y opcionalmente "
            "`GEE_SERVICE_ACCOUNT_PATH` en `.env.local`, o ejecuta "
            "`earthengine authenticate` localmente."
        )

    return NotebookEnv(
        repo=repo,
        figures_dir=figures_dir,
        reports_dir=reports_dir,
        data_dir=data_dir,
        cache_dir=cache_dir,
        has_gemini_api_key=has_gemini_api_key,
        has_ee_credentials=has_ee_credentials,
        gee_project=gee_project,
        gee_sa_path=gee_sa_path,
        env_warnings=env_warnings,
    )


def _configure_polars() -> None:
    """Configura Polars para rendering rico en notebooks."""
    import polars as pl

    pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
    pl.Config.set_tbl_rows(20)
    pl.Config.set_fmt_str_lengths(60)


def _configure_matplotlib(ipython: InteractiveShell | None) -> None:
    """Configura matplotlib (DPI alta + backend inline) para notebooks."""
    import matplotlib.pyplot as plt

    shell = ipython or _get_ipython()
    if shell is not None:
        try:
            shell.run_line_magic("matplotlib", "inline")
        except (ValueError, AttributeError):
            pass

    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 200


def _enable_autoreload(ipython: InteractiveShell | None) -> None:
    """Activa `%autoreload 2` si estamos dentro de IPython."""
    shell = ipython or _get_ipython()
    if shell is None:
        return
    try:
        shell.run_line_magic("load_ext", "autoreload")
        shell.run_line_magic("autoreload", "2")
    except (ValueError, AttributeError):
        pass


def _get_ipython() -> InteractiveShell | None:
    """Devuelve el shell de IPython activo o None si no estamos en notebook."""
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    return get_ipython()


def _detect_gemini_credentials() -> bool:
    """Detecta si alguna variable de entorno habilita la llamada a Gemini.

    Returns:
        True si al menos una de las siguientes esta presente: GEMINI_API_KEY,
        GOOGLE_API_KEY, o (GOOGLE_GENAI_USE_VERTEXAI=true con GOOGLE_CLOUD_PROJECT).
    """
    if os.environ.get("GEMINI_API_KEY"):
        return True
    if os.environ.get("GOOGLE_API_KEY"):
        return True
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() == "true"
    has_project = bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))
    return use_vertex and has_project
