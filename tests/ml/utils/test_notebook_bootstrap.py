"""Tests para `ml.utils.notebook_bootstrap.setup_notebook`."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.utils.notebook_bootstrap import NotebookEnv, setup_notebook


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Limpia variables de entorno LLM/EE para tests deterministas."""
    for key in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GEE_PROJECT_ID",
        "GEE_SERVICE_ACCOUNT_PATH",
    ):
        monkeypatch.delenv(key, raising=False)


def test_setup_notebook_returns_notebook_env(clean_env: None) -> None:
    env = setup_notebook(
        figures_subdir="test-bootstrap/figs",
        reports_subdir="test-bootstrap/reports",
        enable_autoreload=False,
        matplotlib_inline=False,
        polars_rich_html=False,
        load_dotenv=False,
    )
    assert isinstance(env, NotebookEnv)
    assert env.repo.is_dir()
    assert (env.repo / "pyproject.toml").is_file()
    assert env.figures_dir.is_dir()
    assert env.reports_dir.is_dir()
    assert env.data_dir == env.repo / "data"


def test_setup_notebook_detects_no_credentials(clean_env: None) -> None:
    env = setup_notebook(
        figures_subdir="test-bootstrap/no-creds",
        enable_autoreload=False,
        matplotlib_inline=False,
        polars_rich_html=False,
        load_dotenv=False,
    )
    assert env.has_gemini_api_key is False
    assert env.has_ee_credentials is False
    assert any("Gemini" in w for w in env.env_warnings)
    assert any("Earth Engine" in w for w in env.env_warnings)


def test_setup_notebook_detects_gemini_api_key(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    env = setup_notebook(
        figures_subdir="test-bootstrap/with-key",
        enable_autoreload=False,
        matplotlib_inline=False,
        polars_rich_html=False,
        load_dotenv=False,
    )
    assert env.has_gemini_api_key is True
    assert not any("Gemini API key ausente" in w for w in env.env_warnings)


def test_setup_notebook_detects_vertex_setup(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
    env = setup_notebook(
        figures_subdir="test-bootstrap/vertex",
        enable_autoreload=False,
        matplotlib_inline=False,
        polars_rich_html=False,
        load_dotenv=False,
    )
    assert env.has_gemini_api_key is True


def test_summary_markdown_contains_keys(clean_env: None) -> None:
    env = setup_notebook(
        figures_subdir="test-bootstrap/md",
        enable_autoreload=False,
        matplotlib_inline=False,
        polars_rich_html=False,
        load_dotenv=False,
    )
    md = env.summary_markdown()
    assert "Repo root" in md
    assert "Figures dir" in md
    assert "Gemini API key" in md
    assert "Earth Engine" in md


def test_setup_notebook_creates_missing_directories(
    clean_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No cambiamos cwd para no romper find_repo_root; solo verificamos que la
    # subcarpeta unica de este test no existiera antes y se crea ahora.
    unique = f"test-bootstrap/{tmp_path.name}/sub"
    env = setup_notebook(
        figures_subdir=unique,
        reports_subdir=unique,
        enable_autoreload=False,
        matplotlib_inline=False,
        polars_rich_html=False,
        load_dotenv=False,
    )
    assert env.figures_dir.is_dir()
    assert env.reports_dir.is_dir()
