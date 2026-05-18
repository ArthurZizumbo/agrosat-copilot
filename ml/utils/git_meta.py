"""Helpers de metadatos git para tags MLflow / Dagster (US-017+).

Centraliza la lectura del SHA HEAD para que la misma cadena `code_version`
aparezca en MLflow tags, en metadata de assets Dagster y en cualquier otro
sink de versionado. Tolerante a entornos sin git (contenedores efimeros).
"""

from __future__ import annotations

import shutil
import subprocess

import structlog

_log = structlog.get_logger(__name__)


def git_sha(short: bool = False) -> str:
    """Devuelve el SHA del ``HEAD`` o ``"unknown"`` si no es un repo git.

    Args:
        short: si ``True``, devuelve los primeros 7 caracteres (formato
            consistente con ``git log --oneline``). Default ``False``
            (SHA completo de 40 chars para MLflow tags).

    Returns:
        SHA hex o ``"unknown"`` si git no esta instalado, el directorio
        no es repo, o la ejecucion falla por cualquier motivo.
    """
    git_bin = shutil.which("git")
    if git_bin is None:  # pragma: no cover
        return "unknown"
    args = [git_bin, "rev-parse"]
    if short:
        args.append("--short=7")
    args.append("HEAD")
    try:
        out = subprocess.check_output(  # noqa: S603 - git_bin es path absoluto
            args, stderr=subprocess.DEVNULL
        )
        return out.decode().strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):  # pragma: no cover
        return "unknown"


def dvc_data_version(dvc_path: str) -> str:
    """Lee el hash del .dvc file para usar como ``data_version`` en MLflow.

    Args:
        dvc_path: ruta al archivo ``.dvc`` (e.g. ``"data/farslip_pairs.dvc"``)
            o al directorio rastreado por DVC (se busca ``{path}.dvc``).

    Returns:
        Hash MD5 del outs[0] del .dvc file, prefijado con el path para
        contexto (``"data/farslip_pairs@<md5>"``). Devuelve ``"<path>@untracked"``
        si el .dvc file no existe (modo desarrollo sin push DVC).
    """
    from pathlib import Path

    path = Path(dvc_path)
    if not path.suffix == ".dvc":
        path = Path(f"{dvc_path}.dvc")
    if not path.exists():
        _log.warning("dvc file ausente, data_version=untracked", path=str(path))
        return f"{dvc_path}@untracked"
    try:
        import yaml

        meta = yaml.safe_load(path.read_text(encoding="utf-8"))
        outs = meta.get("outs", [])
        if not outs:
            return f"{dvc_path}@no_outs"
        md5 = outs[0].get("md5", "unknown")
        return f"{dvc_path}@{md5}"
    except (OSError, ValueError, KeyError) as exc:  # pragma: no cover
        _log.warning("dvc file malformado", path=str(path), error=str(exc))
        return f"{dvc_path}@malformed"


__all__ = ["dvc_data_version", "git_sha"]
