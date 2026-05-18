"""Clasificacion de errores GCS para fallback graceful (US-017+).

Centraliza la deteccion de excepciones de autenticacion / permisos / red
contra ``google.cloud.storage`` para que tanto el extractor en runtime
(``ml/extractors/farslip_extractor.py``) como los assets Dagster
(``dagster_project/assets/farslip.py``) compartan un solo contrato.

Cubre tres familias:

- ``google.auth.exceptions``: credenciales por defecto ausentes o
  caducadas (``DefaultCredentialsError``, ``GoogleAuthError``,
  ``RefreshError``).
- ``google.api_core.exceptions``: respuestas HTTP de denegacion
  (``Forbidden`` 403, ``Unauthenticated`` 401, ``PermissionDenied``,
  ``NotFound`` 404 — esta ultima cubre bucket/objeto inexistente).
- Stubs en tests sin las libs google.* instaladas: deteccion por nombre
  de clase canonico.

Cualquier otra excepcion (``AttributeError``, ``KeyError``, ``ValueError``,
errores reales del extractor) NO se clasifica como auth y debe burbujear
para no enmascarar bugs.
"""

from __future__ import annotations

#: Nombres canonicos usados como fallback cuando las libs google.* no estan
#: disponibles (CI minimo, tests con stubs). Mantener sincronizado con los
#: bloques try/except de abajo.
_GCS_AUTH_EXC_NAMES = frozenset(
    {
        "DefaultCredentialsError",
        "GoogleAuthError",
        "RefreshError",
        "Forbidden",
        "Unauthenticated",
        "PermissionDenied",
        "NotFound",
    }
)


def is_gcs_auth_error(exc: BaseException) -> bool:
    """Devuelve ``True`` si ``exc`` indica un fallo de auth/permiso/red GCS.

    Args:
        exc: excepcion a clasificar.

    Returns:
        ``True`` para errores google.auth / google.api_core que justifican
        degradar a modo offline (cache local o teacher fallback);
        ``False`` para todo lo demas, que debe burbujear.
    """
    try:
        from google.auth.exceptions import (  # type: ignore[import-not-found]
            DefaultCredentialsError,
            GoogleAuthError,
            RefreshError,
        )

        if isinstance(exc, (DefaultCredentialsError, GoogleAuthError, RefreshError)):
            return True
    except ImportError:  # pragma: no cover
        pass
    try:
        from google.api_core.exceptions import (  # type: ignore[import-not-found]
            Forbidden,
            NotFound,
            PermissionDenied,
            Unauthenticated,
        )

        if isinstance(exc, (Forbidden, NotFound, PermissionDenied, Unauthenticated)):
            return True
    except ImportError:  # pragma: no cover
        pass
    return type(exc).__name__ in _GCS_AUTH_EXC_NAMES


__all__ = ["is_gcs_auth_error"]
