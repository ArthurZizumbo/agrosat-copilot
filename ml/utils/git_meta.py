"""Git metadata helpers for MLflow / Dagster tags (US-017+).

Centralizes reading the HEAD SHA so the same `code_version` string appears
in MLflow tags, in Dagster asset metadata and in any other versioning sink.
Tolerant of environments without git (ephemeral containers).
"""

from __future__ import annotations

import shutil
import subprocess

import structlog

_log = structlog.get_logger(__name__)


def git_sha(short: bool = False) -> str:
    """Return the ``HEAD`` SHA or ``"unknown"`` if it is not a git repo.

    Args:
        short: if ``True``, returns the first 7 characters (format
            consistent with ``git log --oneline``). Default ``False``
            (full 40-char SHA for MLflow tags).

    Returns:
        Hex SHA or ``"unknown"`` if git is not installed, the directory
        is not a repo, or the execution fails for any reason.
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
    """Read the .dvc file hash to use as ``data_version`` in MLflow.

    Args:
        dvc_path: path to the ``.dvc`` file (e.g. ``"data/farslip_pairs.dvc"``)
            or to the directory tracked by DVC (``{path}.dvc`` is looked up).

    Returns:
        MD5 hash of the .dvc file's outs[0], prefixed with the path for
        context (``"data/farslip_pairs@<md5>"``). Returns ``"<path>@untracked"``
        if the .dvc file does not exist (development mode without DVC push).
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
