"""GDAL/VSI environment for reading COGs from GCS and tuning range reads.

TiTiler applies the dict returned by :func:`gdal_gcs_environment` inside a
per-request ``rasterio.Env`` (wired through ``TilerFactory(environment_dependency=...)``,
which supersedes the deprecated ``gdal_config=`` argument). The same dict is
applied by the ``/tiles`` adapter so both tile surfaces read COGs identically.

The dict stays GCS-agnostic in dev (no service account configured): local /
``file://`` / ``http(s)://`` COGs work without any GCS auth (AC-6). GCS auth
(``vsigs``) is added only when ``settings.google_application_credentials`` is set
(staging/prod), where ``url=gs://bucket/...`` resolves through ``/vsigs/`` using
the service account (or ADC on Cloud Run).
"""

from __future__ import annotations

import structlog

from backend.app.core.config import get_settings

logger = structlog.get_logger(__name__)

__all__ = ["gdal_gcs_environment"]


def gdal_gcs_environment() -> dict[str, str]:
    """Return the GDAL/VSI env vars applied per request when reading a COG.

    The base set is COG-friendly range-read tuning that is safe everywhere:

    - ``GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR`` -- critical for object stores;
      stops GDAL from listing the (potentially huge) bucket directory on every
      ``open`` (latency + cost).
    - ``CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif,.tiff,.TIF`` -- only fetch raster
      sidecars, not arbitrary neighbours.
    - HTTP range tuning (``MERGE_CONSECUTIVE_RANGES``, ``MULTIPLEX``) + a 64 MiB
      VSI block cache to make partial COG reads efficient.

    GCS auth is appended only when a service account path is configured, so the
    same callable serves dev (local COG, no GCS) and prod (``gs://`` via vsigs).

    Returns:
        Mapping of GDAL/VSI configuration option names to their string values,
        suitable as the body of a per-request ``rasterio.Env``.
    """
    settings = get_settings()
    env: dict[str, str] = {
        # Do not list the bucket/dir on open -- the single most important COG /
        # object-store setting (latency + per-LIST cost).
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.TIF",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_HTTP_MULTIPLEX": "YES",
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": "67108864",  # 64 MiB block cache.
    }
    # GCS auth: only when a service account is configured (staging/prod). In dev
    # (no GCS) the dict stays GCS-agnostic so local / http(s) COGs work (AC-6).
    if settings.google_application_credentials:
        env["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials
        # GDAL resolves gs:// and /vsigs/ once credentials resolve via SA / ADC.
        env["CPL_GS_ENDPOINT"] = "https://storage.googleapis.com/"
    else:
        logger.debug("gdal_env_gcs_agnostic", reason="no_service_account")
    return env
