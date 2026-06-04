"""Dataset path resolution with naming-convention compatibility.

The feature artifacts were originally generated with the ``_italy`` suffix
(inherited name), but their content is **French PASTIS-R**: the ``parcel_id``
have the format ``{patch_id}_{instance_id}`` (e.g. ``10000_1``), not Italian
parcels. The correct naming is ``_pastis``.

This module allows migrating the code to ``_pastis`` without breaking the
``_italy`` artifacts already materialized on disk (nor the executed notebooks
that reference them, which are kept with their outputs). The strategy is a
compatibility alias: new code requests the canonical ``_pastis`` path and
:func:`resolve_dataset_path` returns the first one that exists, preferring
``_pastis`` and falling back to ``_italy`` (legacy) if the first is not present.

Rename migration: documented in
``docs/product-backlog/rename-italy-to-pastis.md``.
"""

from __future__ import annotations

from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

#: Legacy suffix (inherited name, PASTIS content) and the canonical one.
_LEGACY_SUFFIX = "_italy"
_CANONICAL_SUFFIX = "_pastis"


def to_pastis_name(path: Path | str) -> Path:
    """Returns the canonical ``_pastis`` variant of an ``_italy`` path.

    If the path does not contain ``_italy`` it is returned unchanged.

    Args:
        path: Path with a possible ``_italy`` suffix in the file name.

    Returns:
        ``Path`` with ``_italy`` replaced by ``_pastis`` in the ``stem``.
    """
    p = Path(path)
    if _LEGACY_SUFFIX not in p.name:
        return p
    return p.with_name(p.name.replace(_LEGACY_SUFFIX, _CANONICAL_SUFFIX))


def to_legacy_name(path: Path | str) -> Path:
    """Returns the legacy ``_italy`` variant of a ``_pastis`` path.

    If the path does not contain ``_pastis`` it is returned unchanged.

    Args:
        path: Path with a possible ``_pastis`` suffix in the file name.

    Returns:
        ``Path`` with ``_pastis`` replaced by ``_italy`` in the ``stem``.
    """
    p = Path(path)
    if _CANONICAL_SUFFIX not in p.name:
        return p
    return p.with_name(p.name.replace(_CANONICAL_SUFFIX, _LEGACY_SUFFIX))


def resolve_dataset_path(path: Path | str) -> Path:
    """Resolves a dataset path preferring ``_pastis`` over ``_italy``.

    Resolution rules (for reading existing artifacts):

    1. The input is normalized to its canonical ``_pastis`` form.
    2. If the ``_pastis`` file exists, it is returned.
    3. If not, and the legacy ``_italy`` variant exists, that one is returned
       (with an informative log that the fallback was used).
    4. If neither exists, the canonical ``_pastis`` form is returned (so that the
       caller materializes with the correct name).

    This allows the migrated code to always request ``_pastis`` and still find the
    already-generated ``_italy`` data, without renaming it on disk.

    Args:
        path: Dataset path (may carry ``_italy`` or ``_pastis``).

    Returns:
        ``Path`` resolved to the first existing variant, or the canonical
        ``_pastis`` if neither exists.
    """
    canonical = to_pastis_name(path)
    if canonical.exists():
        return canonical
    legacy = to_legacy_name(canonical)
    if legacy.exists():
        logger.info(
            "dataset_path_legacy_fallback",
            requested=str(canonical),
            resolved=str(legacy),
            hint="artefacto _italy legacy; rename pendiente (ver backlog)",
        )
        return legacy
    return canonical
