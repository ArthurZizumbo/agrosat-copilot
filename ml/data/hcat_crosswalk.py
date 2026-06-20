"""HCAT v3 crosswalk loader for the PASTIS-18 label space (US-074).

This module is the single Python entry point for the taxonomic crosswalk that
maps the 18 contiguous ``semantic18`` PASTIS-R classes to HCAT v3 leaf codes
(10 digits, ``33`` ``crop_type`` prefix) and collapses them into the canonical
macro-classes used by EPIC 12 multi-region transfer.

The crosswalk itself is materialized as a lightweight Polars parquet at
``data/reference/hcat_crosswalk.parquet`` (< 50 KB, versioned in Git, NOT DVC).
It is DERIVED from the real reference CSVs already in the repo
(``data/reference/eurocrops_hcat3.csv`` for the canonical HCAT name<->code
dictionary and ``data/reference/pastis_class_mapping.json`` for the PASTIS names
and real parcel counts) -- no HCAT code is invented here.

Two public helpers:

- :func:`build_crosswalk` re-derives the 18-row table from the source CSV/JSON
  and validates every leaf code against ``eurocrops_hcat3.csv`` (used by the
  notebook and the writer; raises if a code drifts).
- :func:`load_crosswalk` reads the materialized parquet (used by adapters /
  consumers that must not pay the CSV-join cost on every call).

Both return a Polars :class:`polars.DataFrame` with the fixed schema documented
in ``docs/us-planning/us-074.md`` §4. Nothing here trains or serves a model: it
is a pure mapping artifact, and it does NOT touch ``ml.agent.tools.classify``.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import structlog

logger = structlog.get_logger(__name__)

#: Repo-root-relative location of the real reference inputs / output parquet.
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
EUROCROPS_HCAT3_CSV: Path = _REPO_ROOT / "data" / "reference" / "eurocrops_hcat3.csv"
PASTIS_MAPPING_JSON: Path = (
    _REPO_ROOT / "data" / "reference" / "pastis_class_mapping.json"
)
CROSSWALK_PARQUET: Path = _REPO_ROOT / "data" / "reference" / "hcat_crosswalk.parquet"

#: Fixed parquet schema (see US-074 §4). ``*_code`` are Utf8 to preserve the
#: leading zeros of the 10-digit HCAT codes.
CROSSWALK_SCHEMA: dict[str, type[pl.DataType]] = {
    "pastis_id": pl.Int64,
    "semantic18_id": pl.Int64,
    "pastis_name": pl.Utf8,
    "hcat_leaf_name": pl.Utf8,
    "hcat_leaf_code": pl.Utf8,
    "hcat_group_code": pl.Utf8,
    "macro_hcat_group": pl.Utf8,
    "macro_hcat_l1_6": pl.Utf8,
    "n_parcels": pl.Int64,
    "match_quality": pl.Utf8,
    "void_convention": pl.Utf8,
}

#: PASTIS id (1..18) -> HCAT v3 leaf, resolved by agronomic name against the real
#: ``eurocrops_hcat3.csv`` (verified, NOT the approximate codes in the JSON). The
#: three ``approx`` rows have no perfect 1:1 leaf and map to the nearest group
#: node (documented in docs/data/hcat_crosswalk.md).
_PASTIS_TO_HCAT_LEAF: dict[int, tuple[str, str, str]] = {
    # pastis_id: (hcat_leaf_name, hcat_leaf_code, match_quality)
    1: ("pasture_meadow_grassland_grass", "3302000000", "exact"),
    2: ("winter_common_soft_wheat", "3301010101", "exact"),
    3: ("grain_maize_corn_popcorn", "3301010600", "exact"),
    4: ("winter_barley", "3301010401", "exact"),
    5: ("winter_rapeseed_rape", "3301060401", "exact"),
    6: ("spring_barley", "3301010402", "exact"),
    7: ("sunflower", "3301060500", "exact"),
    8: ("vineyards_wine_vine_rebland_grapes", "3303060000", "exact"),
    9: ("sugar_beet", "3301290700", "exact"),
    10: ("winter_triticale", "3301010801", "exact"),
    11: ("winter_durum_hard_wheat", "3301010201", "exact"),
    12: ("fresh_vegetables", "3301070000", "approx"),
    13: ("potatoes", "3301030000", "exact"),
    14: ("legumes_harvested_green", "3301090300", "approx"),
    15: ("soy_soybeans", "3301160000", "exact"),
    16: ("orchards_fruits", "3303010000", "exact"),
    17: ("cereal", "3301010000", "approx"),
    18: ("millet_sorghum", "3301010900", "exact"),
}

#: HCAT group node (L2, 6 significant digits; L1 for pasture/permanent) each leaf
#: collapses into, plus the readable macro label. Derived deterministically from
#: the leaf code's ancestor in the HCAT hierarchy. Yields 10 distinct group codes
#: over the 18 crop classes; with the ``void`` partial-label macro (§7) the full
#: macro vocabulary is 11.
_LEAF_CODE_TO_GROUP: dict[str, tuple[str, str]] = {
    # hcat_leaf_code: (hcat_group_code, macro_hcat_group)
    "3302000000": ("3302000000", "grassland"),
    "3301010101": ("3301010000", "cereals"),
    "3301010600": ("3301010000", "cereals"),
    "3301010401": ("3301010000", "cereals"),
    "3301060401": ("3301060000", "oilseed_industrial"),
    "3301010402": ("3301010000", "cereals"),
    "3301060500": ("3301060000", "oilseed_industrial"),
    "3303060000": ("3303060000", "vineyard"),
    "3301290700": ("3301290000", "sugar_beet"),
    "3301010801": ("3301010000", "cereals"),
    "3301010201": ("3301010000", "cereals"),
    "3301070000": ("3301070000", "vegetables"),
    "3301030000": ("3301030000", "potato"),
    "3301090300": ("3301090000", "legumes_fodder"),
    "3301160000": ("3301160000", "soybean"),
    "3303010000": ("3303010000", "orchard"),
    "3301010000": ("3301010000", "cereals"),
    "3301010900": ("3301010000", "cereals"),
}

#: Legacy 6-family HCAT L1 macro (pastis_class_mapping.json groupings.hcat_l1_6),
#: keyed by PASTIS id 1..18. Kept for E4/E6 compatibility (XGBoost 0.6535 F1-macro,
#: product finding v8); the parquet carries both this and the finer 10-group view.
_PASTIS_TO_HCAT_L1_6: dict[int, str] = {
    1: "OTHER",
    2: "CEREALS",
    3: "CEREALS",
    4: "CEREALS",
    5: "OILSEEDS",
    6: "CEREALS",
    7: "OILSEEDS",
    8: "PERMANENT_WOODY",
    9: "ROOT_CROPS",
    10: "CEREALS",
    11: "CEREALS",
    12: "OTHER",
    13: "ROOT_CROPS",
    14: "LEGUMES",
    15: "LEGUMES",
    16: "PERMANENT_WOODY",
    17: "CEREALS",
    18: "CEREALS",
}

#: The 11 canonical macro-classes (10 HCAT crop groups + the ``void`` partial-label
#: macro that absorbs background/out-of-nomenclature pixels, §7). Exposed so tests
#: and the registry agree on the vocabulary without re-deriving it.
MACRO_HCAT_GROUPS: tuple[str, ...] = (
    "grassland",
    "cereals",
    "oilseed_industrial",
    "vineyard",
    "sugar_beet",
    "vegetables",
    "potato",
    "legumes_fodder",
    "soybean",
    "orchard",
    "void",
)


def _load_hcat_name_to_code() -> dict[str, str]:
    """Load the canonical HCAT v3 name->code dictionary from the real CSV.

    Returns:
        Mapping ``{HCAT3_name: HCAT3_code}`` read from
        :data:`EUROCROPS_HCAT3_CSV` with codes kept as zero-preserving strings.

    Raises:
        FileNotFoundError: if the reference CSV is missing.
    """
    if not EUROCROPS_HCAT3_CSV.is_file():
        raise FileNotFoundError(
            f"eurocrops_hcat3.csv not found at {EUROCROPS_HCAT3_CSV}; the HCAT "
            "name<->code dictionary is required to validate leaf codes."
        )
    df = pl.read_csv(
        EUROCROPS_HCAT3_CSV,
        schema_overrides={"HCAT3_code": pl.Utf8, "HCAT3_name": pl.Utf8},
    )
    return dict(zip(df["HCAT3_name"], df["HCAT3_code"], strict=True))


def _load_pastis_names_and_counts() -> dict[int, tuple[str, int]]:
    """Load PASTIS crop names and real parcel counts from the mapping JSON.

    Returns:
        Mapping ``{pastis_id: (pastis_name, n_parcels)}`` for the 18 crop classes
        (ids 1..18), read from :data:`PASTIS_MAPPING_JSON`.

    Raises:
        FileNotFoundError: if the mapping JSON is missing.
    """
    if not PASTIS_MAPPING_JSON.is_file():
        raise FileNotFoundError(
            f"pastis_class_mapping.json not found at {PASTIS_MAPPING_JSON}; "
            "PASTIS names and real parcel counts are required."
        )
    raw = json.loads(PASTIS_MAPPING_JSON.read_text(encoding="utf-8"))
    classes = raw["classes"]
    out: dict[int, tuple[str, int]] = {}
    for pid in range(1, 19):
        entry = classes[str(pid)]
        out[pid] = (str(entry["name"]), int(entry["n_parcels"]))
    return out


def build_crosswalk() -> pl.DataFrame:
    """Re-derive the 18-row PASTIS-18 -> HCAT v3 crosswalk from the real sources.

    Joins :data:`_PASTIS_TO_HCAT_LEAF` with the PASTIS names/counts (JSON) and the
    macro-group / legacy 6-family mappings, then VALIDATES every ``hcat_leaf_code``
    against the canonical ``eurocrops_hcat3.csv`` so no invented code can slip in.

    Returns:
        A Polars :class:`polars.DataFrame` with the fixed :data:`CROSSWALK_SCHEMA`
        (18 rows, ``*_code`` as Utf8), sorted by ``semantic18_id``.

    Raises:
        ValueError: if a leaf code is absent from ``eurocrops_hcat3.csv`` (drift
            guard) or if a PASTIS id lacks a group mapping.
    """
    name_to_code = _load_hcat_name_to_code()
    valid_codes = set(name_to_code.values())
    pastis = _load_pastis_names_and_counts()

    rows: list[dict[str, object]] = []
    for pid in range(1, 19):
        leaf_name, leaf_code, quality = _PASTIS_TO_HCAT_LEAF[pid]
        if leaf_code not in valid_codes:
            raise ValueError(
                f"HCAT leaf code {leaf_code!r} (PASTIS id {pid}, "
                f"{leaf_name!r}) is not present in eurocrops_hcat3.csv; "
                "refusing to materialize an invented code."
            )
        if leaf_code not in _LEAF_CODE_TO_GROUP:
            raise ValueError(
                f"no HCAT group node mapped for leaf code {leaf_code!r} "
                f"(PASTIS id {pid})."
            )
        group_code, macro = _LEAF_CODE_TO_GROUP[leaf_code]
        pastis_name, n_parcels = pastis[pid]
        rows.append(
            {
                "pastis_id": pid,
                "semantic18_id": pid - 1,
                "pastis_name": pastis_name,
                "hcat_leaf_name": leaf_name,
                "hcat_leaf_code": leaf_code,
                "hcat_group_code": group_code,
                "macro_hcat_group": macro,
                "macro_hcat_l1_6": _PASTIS_TO_HCAT_L1_6[pid],
                "n_parcels": n_parcels,
                "match_quality": quality,
                "void_convention": "crop",
            }
        )

    df = pl.DataFrame(rows, schema=CROSSWALK_SCHEMA).sort("semantic18_id")
    logger.info(
        "hcat_crosswalk_built",
        n_rows=df.height,
        n_macro_groups=df["macro_hcat_group"].n_unique(),
        n_approx=df.filter(pl.col("match_quality") == "approx").height,
    )
    return df


def write_crosswalk(path: Path | None = None) -> Path:
    """Materialize the crosswalk to the lightweight reference parquet.

    Args:
        path: Output parquet path; defaults to :data:`CROSSWALK_PARQUET`.

    Returns:
        The path the parquet was written to.
    """
    out_path = path or CROSSWALK_PARQUET
    df = build_crosswalk()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    logger.info("hcat_crosswalk_written", path=str(out_path), n_rows=df.height)
    return out_path


def load_crosswalk(path: Path | None = None) -> pl.DataFrame:
    """Read the materialized crosswalk parquet (consumer entry point).

    Falls back to :func:`build_crosswalk` (re-derivation from the CSV/JSON) when
    the parquet does not yet exist, so adapters never crash on a fresh checkout.

    Args:
        path: Parquet path to read; defaults to :data:`CROSSWALK_PARQUET`.

    Returns:
        A Polars :class:`polars.DataFrame` with :data:`CROSSWALK_SCHEMA`.
    """
    in_path = path or CROSSWALK_PARQUET
    if not in_path.is_file():
        logger.warning(
            "hcat_crosswalk_parquet_missing_rebuilding", path=str(in_path)
        )
        return build_crosswalk()
    return pl.read_parquet(in_path)
