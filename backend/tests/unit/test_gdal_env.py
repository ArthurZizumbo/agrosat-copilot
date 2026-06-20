"""Unit tests for the GDAL/GCS environment dependency (US-055 AC-1/AC-6)."""

from __future__ import annotations

import pytest

from backend.app.core.config import Settings, get_settings
from backend.app.services import gdal_env


@pytest.fixture()
def _clear_settings_cache() -> None:
    """Clear the cached settings so a monkeypatched instance is picked up."""
    get_settings.cache_clear()


def test_base_vars_present_without_gcs(
    monkeypatch: pytest.MonkeyPatch, _clear_settings_cache: None
) -> None:
    """Dev (no SA): COG-friendly vars present, no GCS auth added (AC-6)."""
    monkeypatch.setattr(
        gdal_env, "get_settings", lambda: Settings(google_application_credentials="")
    )
    env = gdal_env.gdal_gcs_environment()

    assert env["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR"
    assert env["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] == ".tif,.tiff,.TIF"
    assert env["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] == "YES"
    assert env["VSI_CACHE"] == "TRUE"
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in env
    assert "CPL_GS_ENDPOINT" not in env


def test_gcs_auth_added_with_service_account(
    monkeypatch: pytest.MonkeyPatch, _clear_settings_cache: None
) -> None:
    """Staging/prod (SA set): GCS auth vars are appended (AC-1)."""
    monkeypatch.setattr(
        gdal_env,
        "get_settings",
        lambda: Settings(google_application_credentials="/secrets/sa.json"),
    )
    env = gdal_env.gdal_gcs_environment()

    assert env["GOOGLE_APPLICATION_CREDENTIALS"] == "/secrets/sa.json"
    assert env["CPL_GS_ENDPOINT"] == "https://storage.googleapis.com/"
    # Base tuning still present.
    assert env["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR"
