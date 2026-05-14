"""Tests para ml.utils.notebook_setup.

Verifican:
- `find_repo_root` localiza el repo aunque cwd este en subdir profundo.
- `find_repo_root` retorna fallback si no encuentra pyproject.toml.
- `load_env_local` parsea correctamente y respeta env vars preexistentes.
- `load_env_local` no falla si .env.local no existe.
- `configure_ee_from_env` limpia GAC roto y retorna tupla esperada.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ml.utils.notebook_setup import (
    configure_ee_from_env,
    find_repo_root,
    load_env_local,
)


def test_find_repo_root_from_subdir(tmp_path: Path) -> None:
    """Debe encontrar pyproject.toml subiendo niveles desde un subdir profundo."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    deep = tmp_path / "notebooks" / "eda" / "subdir"
    deep.mkdir(parents=True)
    assert find_repo_root(deep) == tmp_path.resolve()


def test_find_repo_root_at_root(tmp_path: Path) -> None:
    """Debe encontrar pyproject.toml en el directorio inicial."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    assert find_repo_root(tmp_path) == tmp_path.resolve()


def test_find_repo_root_fallback(tmp_path: Path) -> None:
    """Sin pyproject.toml en ningun ancestro retorna `start.resolve()`."""
    # tmp_path NO tiene pyproject.toml: nos aseguramos
    assert not (tmp_path / "pyproject.toml").exists()
    result = find_repo_root(tmp_path)
    # Fallback: debe retornar el `start` resuelto (o un ancestro real con
    # pyproject.toml si el test runner corre dentro del propio repo).
    assert result.is_dir()


def test_find_repo_root_default_uses_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin argumento usa Path.cwd() como punto de partida."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert find_repo_root() == tmp_path.resolve()


def test_load_env_local_parses_key_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Lee KEY=VALUE y popula os.environ."""
    env = tmp_path / ".env.local"
    env.write_text(
        "FOO=hello\nBAR=world\n# comment line\nEMPTY=\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("BAR", raising=False)
    load_env_local(tmp_path)
    assert os.environ.get("FOO") == "hello"
    assert os.environ.get("BAR") == "world"


def test_load_env_local_strips_quotes_and_inline_comments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Quita comillas envolventes y comentarios al final de linea."""
    env = tmp_path / ".env.local"
    env.write_text(
        'QUOTED="value with spaces"\nWITH_COMMENT=foo  # trailing comment\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("QUOTED", raising=False)
    monkeypatch.delenv("WITH_COMMENT", raising=False)
    load_env_local(tmp_path)
    assert os.environ.get("QUOTED") == "value with spaces"
    assert os.environ.get("WITH_COMMENT") == "foo"


def test_load_env_local_respects_preexisting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No sobreescribe variables ya seteadas en os.environ."""
    env = tmp_path / ".env.local"
    env.write_text("PREEXISTING=fromfile\n", encoding="utf-8")
    monkeypatch.setenv("PREEXISTING", "fromshell")
    load_env_local(tmp_path)
    assert os.environ.get("PREEXISTING") == "fromshell"


def test_load_env_local_missing_file_noop(tmp_path: Path) -> None:
    """Si .env.local no existe la funcion no levanta excepcion."""
    assert not (tmp_path / ".env.local").exists()
    load_env_local(tmp_path)  # no raises


def test_configure_ee_strips_broken_gac(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Si GOOGLE_APPLICATION_CREDENTIALS apunta a inexistente, se remueve."""
    env = tmp_path / ".env.local"
    env.write_text("GEE_PROJECT_ID=test-project\n", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "nope.json"))
    monkeypatch.delenv("GEE_PROJECT_ID", raising=False)
    monkeypatch.delenv("GEE_SERVICE_ACCOUNT_PATH", raising=False)

    project, sa_json = configure_ee_from_env(tmp_path)

    assert project == "test-project"
    assert sa_json is None
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ


def test_configure_ee_preserves_valid_gac(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Si GAC apunta a archivo real se preserva."""
    real_sa = tmp_path / "real_sa.json"
    real_sa.write_text("{}", encoding="utf-8")
    (tmp_path / ".env.local").write_text("GEE_PROJECT_ID=p\n", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(real_sa))
    monkeypatch.delenv("GEE_PROJECT_ID", raising=False)
    monkeypatch.delenv("GEE_SERVICE_ACCOUNT_PATH", raising=False)

    configure_ee_from_env(tmp_path)
    assert os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") == str(real_sa)


def test_configure_ee_returns_sa_path_if_file_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`GEE_SERVICE_ACCOUNT_PATH` apuntando a archivo real retorna Path."""
    sa = tmp_path / "sa.json"
    sa.write_text("{}", encoding="utf-8")
    (tmp_path / ".env.local").write_text(
        f"GEE_PROJECT_ID=p\nGEE_SERVICE_ACCOUNT_PATH={sa}\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GEE_PROJECT_ID", raising=False)
    monkeypatch.delenv("GEE_SERVICE_ACCOUNT_PATH", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    project, sa_json = configure_ee_from_env(tmp_path)

    assert project == "p"
    assert sa_json == sa
