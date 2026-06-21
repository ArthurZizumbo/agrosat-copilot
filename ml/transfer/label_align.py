"""Align EuroCropsML ``EC_hcat_c`` codes to the US-074 HCAT macro label-space.

EuroCropsML labels every parcel with ``EC_hcat_c``, a 10-digit HCAT (Harmonized
Crop and Agricultural Types) leaf code -- the **same** taxonomy the US-074
crosswalk (``data/reference/hcat_crosswalk.parquet``) and the HCAT v3 reference
(``data/reference/eurocrops_hcat3.csv``, 384 nodes) speak. This module collapses
those leaf codes into the 11 macro groups of the ``hcat-macro`` label-space
registered by US-074, so the EuroCropsML few-shot curve is reported on the exact
same label-space as the rest of EPIC 12.

The collapse rule mirrors US-074 section 3.2:

1. Truncate the 10-digit ``EC_hcat_c`` to its HCAT **group** code (the 6 leading
   significant digits, i.e. ``XXXXXX0000``). HCAT is hierarchical: a leaf such as
   ``3301010101`` (winter common soft wheat) shares the group ``3301010000``
   (cereal) with its siblings.
2. Look the group code up in the crosswalk's ``hcat_group_code`` -> ``macro_hcat_group``
   map. The crosswalk only carries the 18 PASTIS-R leaves, but its group codes are
   the canonical HCAT groups, so every EuroCropsML code whose group is present
   resolves to a macro group.
3. Codes whose group is absent from the crosswalk are tagged ``null-class``: an
   honest "outside the PASTIS-18 crosswalk" marker (partial-label, US-074 section 7),
   never a fabricated class.

This module only *consumes* the parquet/CSV; it never mutates
:mod:`ml.eval.class_remap` nor :mod:`ml.agent.tools.classify` (the ``hcat-macro``
space is already registered there by US-074).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import polars as pl
import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "NULL_CLASS",
    "align_codes_to_hcat_macro",
    "align_labels_to_hcat_macro",
    "load_hcat_macro_map",
    "to_group_code",
]

#: Sentinel macro group for EuroCropsML codes outside the PASTIS-18 crosswalk.
NULL_CLASS: str = "null-class"

_REPO_ROOT = Path(__file__).resolve().parents[2]
#: US-074 crosswalk: ``hcat_group_code`` -> ``macro_hcat_group`` (canonical source).
_CROSSWALK_PATH = _REPO_ROOT / "data" / "reference" / "hcat_crosswalk.parquet"
#: HCAT v3 reference (384 nodes); used to validate that a code is a known HCAT node.
_HCAT3_PATH = _REPO_ROOT / "data" / "reference" / "eurocrops_hcat3.csv"

#: HCAT codes are 10-digit; the group level keeps the 6 leading significant
#: digits and zero-pads the trailing 4 (e.g. 3301010101 -> 3301010000).
_GROUP_TRAILING_ZEROS: int = 4


def to_group_code(hcat_code: int) -> int:
    """Collapse a 10-digit HCAT leaf code to its 6-significant-digit group code.

    HCAT is hierarchical with a fixed 10-digit width; the crop *group* lives in
    the 6 leading digits, the leaf refinement in the trailing 4. Truncating the
    trailing 4 to zero yields the group code used by the US-074 crosswalk
    (``hcat_group_code``).

    Args:
        hcat_code: A 10-digit ``EC_hcat_c`` value (e.g. ``3301010101``).

    Returns:
        The group code with the trailing four digits zeroed
        (e.g. ``3301010000``). Non-positive or malformed codes are returned
        unchanged so the lookup downstream maps them to ``null-class``.
    """
    if hcat_code <= 0:
        return hcat_code
    factor = 10**_GROUP_TRAILING_ZEROS
    return int((hcat_code // factor) * factor)


@lru_cache(maxsize=1)
def load_hcat_macro_map(
    crosswalk_path: str | None = None,
) -> dict[int, str]:
    """Load the ``hcat_group_code -> macro_hcat_group`` map from the US-074 crosswalk.

    The crosswalk carries one row per PASTIS-18 leaf, but its ``hcat_group_code``
    column holds the canonical HCAT group code, so collapsing it to a unique
    ``group_code -> macro_group`` mapping covers every EuroCropsML code whose
    group is part of the crosswalk.

    Args:
        crosswalk_path: Optional override of the crosswalk parquet path
            (defaults to ``data/reference/hcat_crosswalk.parquet``).

    Returns:
        A mapping ``{hcat_group_code: macro_hcat_group}``.

    Raises:
        FileNotFoundError: if the crosswalk parquet does not exist.
    """
    path = Path(crosswalk_path) if crosswalk_path is not None else _CROSSWALK_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Crosswalk HCAT no encontrado en {path}. Es un artefacto de US-074 "
            "(data/reference/hcat_crosswalk.parquet)."
        )
    df = pl.read_parquet(path).select("hcat_group_code", "macro_hcat_group").unique()
    mapping = {
        int(code): str(macro)
        for code, macro in zip(
            df.get_column("hcat_group_code").to_list(),
            df.get_column("macro_hcat_group").to_list(),
            strict=True,
        )
    }
    logger.info("hcat_macro_map_loaded", n_groups=len(mapping), path=str(path))
    return mapping


def align_codes_to_hcat_macro(
    hcat_codes: list[int],
    *,
    crosswalk_path: str | None = None,
) -> list[str]:
    """Map a list of raw ``EC_hcat_c`` codes to macro HCAT groups.

    Each code is collapsed to its group level (:func:`to_group_code`) and looked
    up in the US-074 crosswalk. Codes whose group is not present resolve to
    :data:`NULL_CLASS`.

    Args:
        hcat_codes: Raw 10-digit ``EC_hcat_c`` values.
        crosswalk_path: Optional override of the crosswalk parquet path.

    Returns:
        A list (same length/order as ``hcat_codes``) of macro group names, with
        :data:`NULL_CLASS` for codes outside the crosswalk.
    """
    macro_map = load_hcat_macro_map(crosswalk_path)
    return [macro_map.get(to_group_code(int(c)), NULL_CLASS) for c in hcat_codes]


def align_labels_to_hcat_macro(
    hcat_codes: pl.Series,
    *,
    crosswalk_path: str | None = None,
) -> pl.Series:
    """Map a Polars series of ``EC_hcat_c`` codes to macro HCAT groups.

    Vectorized counterpart of :func:`align_codes_to_hcat_macro` for the Polars
    pipeline: returns a ``pl.Utf8`` series of macro group names aligned to the
    US-074 ``hcat-macro`` label-space, with :data:`NULL_CLASS` for unmapped codes.

    Args:
        hcat_codes: Series of raw 10-digit ``EC_hcat_c`` values.
        crosswalk_path: Optional override of the crosswalk parquet path.

    Returns:
        A ``pl.Series`` (named ``"macro_hcat_group"``) of the same length as the
        input, holding the macro group per parcel.
    """
    macro_map = load_hcat_macro_map(crosswalk_path)
    factor = 10**_GROUP_TRAILING_ZEROS
    group = (hcat_codes.cast(pl.Int64) // factor) * factor
    macro = group.replace_strict(
        old=list(macro_map.keys()),
        new=list(macro_map.values()),
        default=NULL_CLASS,
        return_dtype=pl.Utf8,
    )
    result: pl.Series = macro.alias("macro_hcat_group")
    return result
