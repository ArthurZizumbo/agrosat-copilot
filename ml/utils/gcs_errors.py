"""GCS error classification for graceful fallback (US-017+).

Centralizes the detection of authentication / permission / network exceptions
against ``google.cloud.storage`` so that both the runtime extractor
(``ml/extractors/farslip_extractor.py``) and the Dagster assets
(``dagster_project/assets/farslip.py``) share a single contract.

Covers three families:

- ``google.auth.exceptions``: missing or expired default credentials
  (``DefaultCredentialsError``, ``GoogleAuthError``, ``RefreshError``).
- ``google.api_core.exceptions``: HTTP denial responses (``Forbidden`` 403,
  ``Unauthenticated`` 401, ``PermissionDenied``, ``NotFound`` 404 — the latter
  covers a nonexistent bucket/object).
- Stubs in tests without the google.* libs installed: detection by canonical
  class name.

Any other exception (``AttributeError``, ``KeyError``, ``ValueError``, real
errors from the extractor) is NOT classified as auth and must bubble up so as
not to mask bugs.
"""

from __future__ import annotations

#: Canonical names used as fallback when the google.* libs are not
#: available (minimal CI, tests with stubs). Keep in sync with the
#: try/except blocks below.
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
    """Return ``True`` if ``exc`` indicates a GCS auth/permission/network failure.

    Args:
        exc: exception to classify.

    Returns:
        ``True`` for google.auth / google.api_core errors that justify degrading
        to offline mode (local cache or teacher fallback); ``False`` for
        everything else, which must bubble up.
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
