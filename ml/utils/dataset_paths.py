"""Resolucion de rutas de datasets con compatibilidad de nomenclatura.

Los artefactos de features se generaron originalmente con el sufijo
``_italy`` (nombre heredado), pero su contenido es **PASTIS-R frances**: los
``parcel_id`` tienen formato ``{patch_id}_{instance_id}`` (ej ``10000_1``),
no parcelas italianas. La nomenclatura correcta es ``_pastis``.

Este modulo permite migrar el codigo a ``_pastis`` sin romper los artefactos
``_italy`` ya materializados en disco (ni los notebooks ejecutados que los
referencian, que se conservan con sus salidas). La estrategia es un alias de
compatibilidad: el codigo nuevo pide la ruta canonica ``_pastis`` y
:func:`resolve_dataset_path` devuelve la primera que exista, prefiriendo
``_pastis`` y cayendo a ``_italy`` (legacy) si la primera no esta presente.

Migracion del rename: documentada en
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
    """Devuelve la variante canonica ``_pastis`` de una ruta ``_italy``.

    Si la ruta no contiene ``_italy`` se devuelve sin cambios.

    Args:
        path: Ruta con posible sufijo ``_italy`` en el nombre de archivo.

    Returns:
        ``Path`` con ``_italy`` reemplazado por ``_pastis`` en el ``stem``.
    """
    p = Path(path)
    if _LEGACY_SUFFIX not in p.name:
        return p
    return p.with_name(p.name.replace(_LEGACY_SUFFIX, _CANONICAL_SUFFIX))


def to_legacy_name(path: Path | str) -> Path:
    """Devuelve la variante legacy ``_italy`` de una ruta ``_pastis``.

    Si la ruta no contiene ``_pastis`` se devuelve sin cambios.

    Args:
        path: Ruta con posible sufijo ``_pastis`` en el nombre de archivo.

    Returns:
        ``Path`` con ``_pastis`` reemplazado por ``_italy`` en el ``stem``.
    """
    p = Path(path)
    if _CANONICAL_SUFFIX not in p.name:
        return p
    return p.with_name(p.name.replace(_CANONICAL_SUFFIX, _LEGACY_SUFFIX))


def resolve_dataset_path(path: Path | str) -> Path:
    """Resuelve una ruta de dataset prefiriendo ``_pastis`` sobre ``_italy``.

    Reglas de resolucion (para lectura de artefactos existentes):

    1. Se normaliza la entrada a su forma canonica ``_pastis``.
    2. Si el archivo ``_pastis`` existe, se devuelve.
    3. Si no, y existe la variante legacy ``_italy``, se devuelve esa (con un
       log informativo de que se uso el fallback).
    4. Si ninguna existe, se devuelve la forma canonica ``_pastis`` (para que
       el caller materialice con el nombre correcto).

    Esto permite que el codigo migrado pida siempre ``_pastis`` y siga
    encontrando los datos ``_italy`` ya generados, sin renombrarlos en disco.

    Args:
        path: Ruta del dataset (puede traer ``_italy`` o ``_pastis``).

    Returns:
        ``Path`` resuelta a la primera variante existente, o la canonica
        ``_pastis`` si no existe ninguna.
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
