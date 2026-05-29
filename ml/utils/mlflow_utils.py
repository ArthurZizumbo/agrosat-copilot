"""Utilidades de tracking MLflow para experimentos del proyecto (US-019).

Centraliza la apertura de runs MLflow con los tags obligatorios
``data_version`` y ``code_version`` (regla CLAUDE.md 10). Reutiliza
:mod:`ml.utils.git_meta` para resolver el SHA git y el hash DVC en lugar
de re-implementar la llamada a ``subprocess`` (DRY, decision D7).

Resolucion del tracking URI (decision D8/D14, AC-14):

1. ``override`` explicito pasado por el llamador.
2. La variable de entorno ``MLFLOW_TRACKING_URI``.
3. El servidor MLflow Docker local ``http://localhost:5010`` si responde
   a ``/health`` dentro del timeout.
4. ``file:./mlruns`` como fallback para que un dev sin Docker arriba (o el
   CI) no quede bloqueado.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import mlflow
import structlog

from ml.utils.git_meta import dvc_data_version, git_sha

if TYPE_CHECKING:  # pragma: no cover - solo para anotaciones de tipo
    from mlflow import ActiveRun

logger = structlog.get_logger(__name__)

__all__ = [
    "resolve_tracking_uri",
    "server_is_reachable",
    "track_experiment",
]

_DEFAULT_SERVER_URL = "http://localhost:5010"
_DEFAULT_FILE_STORE = "file:./mlruns"
_HEALTH_TIMEOUT_S = 2.0


def server_is_reachable(server_url: str, *, timeout: float = _HEALTH_TIMEOUT_S) -> bool:
    """Hace un probe HTTP al endpoint ``/health`` del servidor MLflow.

    Args:
        server_url: URL base del servidor MLflow (sin ``/health``).
        timeout: Timeout en segundos del request HTTP.

    Returns:
        ``True`` si ``/health`` responde con codigo 2xx dentro del
        timeout; ``False`` ante cualquier error de red o timeout.
    """
    health_url = f"{server_url.rstrip('/')}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as response:  # noqa: S310 - URL local de dev
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


# Alias retro-compatible (modulos internos lo usan).
_server_is_reachable = server_is_reachable


def resolve_tracking_uri(
    override: str | None = None,
    *,
    server_url: str = _DEFAULT_SERVER_URL,
    probe_server: bool = True,
) -> str:
    """Resuelve el tracking URI de MLflow con fallback gradual.

    Prioridad: ``override`` > ``$MLFLOW_TRACKING_URI`` > ``server_url`` (si
    responde ``/health``) > ``file:./mlruns``.

    Args:
        override: URI explicito; si se pasa, se usa sin mas comprobaciones.
        server_url: URL del servidor MLflow Docker local a probar.
        probe_server: Si ``True`` (default) hace un probe a ``/health``
            antes de elegir ``server_url``; si el servidor no responde,
            degrada al file store con un ``log.warning``. Si ``False`` no
            se contacta al servidor (util para tests deterministas).

    Returns:
        El tracking URI resuelto como cadena.
    """
    if override:
        return override

    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri

    if not probe_server:
        return _DEFAULT_FILE_STORE

    if _server_is_reachable(server_url):
        logger.info("mlflow_tracking_uri_resolved", uri=server_url, source="docker_server")
        return server_url

    logger.warning(
        "mlflow_server_unreachable_fallback_file_store",
        server_url=server_url,
        fallback=_DEFAULT_FILE_STORE,
        note="Servidor MLflow Docker no responde; usar `make mlflow-up` para levantarlo.",
    )
    return _DEFAULT_FILE_STORE


@contextmanager
def track_experiment(
    experiment_name: str,
    *,
    run_name: str | None = None,
    tracking_uri: str | None = None,
    dvc_path: str | None = None,
    probe_server: bool = True,
) -> Iterator[ActiveRun]:
    """Context manager que abre un run MLflow con tags de versionado.

    Resuelve el tracking URI, fija el experimento, abre un run y le inyecta
    los tags ``code_version`` (SHA git via :func:`git_sha`) y
    ``data_version`` (hash DVC via :func:`dvc_data_version` si se pasa
    ``dvc_path``, o ``"untracked"`` en caso contrario).

    Args:
        experiment_name: Nombre del experimento MLflow (se crea si no
            existe).
        run_name: Nombre legible del run; ``None`` deja que MLflow genere
            uno aleatorio.
        tracking_uri: Override del tracking URI; si es ``None`` se delega
            en :func:`resolve_tracking_uri`.
        dvc_path: Ruta al dataset rastreado por DVC para resolver el
            ``data_version``. Si es ``None`` el tag queda como
            ``"untracked"``.
        probe_server: Se reenvia a :func:`resolve_tracking_uri`; ponerlo
            en ``False`` en tests para no contactar al servidor Docker.

    Yields:
        El :class:`mlflow.ActiveRun` activo, para loggear params, metricas
        y artefactos dentro del bloque ``with``.
    """
    resolved_uri = resolve_tracking_uri(tracking_uri, probe_server=probe_server)
    mlflow.set_tracking_uri(resolved_uri)
    mlflow.set_experiment(experiment_name)

    code_version = git_sha()
    data_version = dvc_data_version(dvc_path) if dvc_path else "untracked"

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.set_tag("code_version", code_version)
        mlflow.set_tag("data_version", data_version)
        logger.info(
            "mlflow_run_started",
            experiment=experiment_name,
            run_name=run_name,
            run_id=active_run.info.run_id,
            tracking_uri=resolved_uri,
            code_version=code_version,
            data_version=data_version,
        )
        yield active_run
