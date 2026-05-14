"""Helpers reutilizables por los notebooks de EDA.

Centraliza tres patrones que de otra forma se duplicarian en cada `.ipynb`:

- Resolucion robusta del repo root via `pyproject.toml` (independiente del CWD).
- Carga de `.env.local` sin dependencia `python-dotenv`.
- Configuracion de credenciales para Google Earth Engine (ADC vs service account).

Las funciones son puras (input -> output, sin estado global) y siguen el estilo
del resto de `ml/utils/`.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Sube niveles desde `start` hasta encontrar `pyproject.toml`.

    Util para notebooks: funciona desde cualquier subdirectorio del repo
    (`notebooks/`, `notebooks/eda/`, `scripts/`, etc.) sin asumir un nombre
    de carpeta concreto.

    Args:
        start: Punto de partida. Si None usa `Path.cwd()`.

    Returns:
        Path absoluto al repo root. Si no se encuentra `pyproject.toml` en
        ningun ancestro, retorna `start.resolve()` como fallback (modo
        degradado para que el notebook no rompa).
    """
    base = (start or Path.cwd()).resolve()
    for candidate in (base, *base.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return base


def load_env_local(repo_root: Path) -> None:
    """Carga claves `KEY=VALUE` de `.env.local` en `os.environ`.

    No sobreescribe variables ya presentes. Ignora lineas vacias y comentarios.
    Sin dependencia de `python-dotenv`.

    Args:
        repo_root: Ruta absoluta al repo root (usar `find_repo_root()`).
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
    """Lee `.env.local` y prepara credenciales para Earth Engine.

    Hace tres cosas:

    1. Llama `load_env_local(repo_root)` para popular `os.environ`.
    2. Si `GOOGLE_APPLICATION_CREDENTIALS` apunta a un archivo inexistente
       (placeholder del template `.env.local`), lo remueve para que
       `google-auth` caiga limpio al ADC del gcloud.
    3. Retorna `(GEE_PROJECT_ID, service_account_path_or_None)` listo para
       pasar a `init_ee(service_account_json=..., project=...)`.

    Args:
        repo_root: Ruta al repo root.

    Returns:
        Tupla `(gee_project, sa_json_path)`. Ambos pueden ser None si las
        variables no estan seteadas o el archivo SA no existe.
    """
    load_env_local(repo_root)

    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac and not Path(gac).is_file():
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

    gee_project = os.environ.get("GEE_PROJECT_ID") or None
    gee_sa_path = os.environ.get("GEE_SERVICE_ACCOUNT_PATH")
    sa_json = Path(gee_sa_path) if gee_sa_path and Path(gee_sa_path).is_file() else None

    return gee_project, sa_json
