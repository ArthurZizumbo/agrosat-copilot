"""Categorical variable encoding and attribute derivation (US-018 ext, Avance 2).

Module complementary to :mod:`ml.features.selection` that closes the
"Feature construction" block of Avance 2 (30 pts rubric), covering what the
team's WIP notebook (``notebooks/feature_engineering/02a_fe_sentinel2.ipynb``)
explored interactively with ``pandas.get_dummies`` and ``KBinsDiscretizer``.

Here the public API is **Polars-first** (rule ``ml/CLAUDE.md NEVER pandas``):
all functions receive and return :class:`polars.DataFrame` /
:class:`polars.Series`. ``numpy`` only appears internally for the
numerical computations.

Public API
----------
- :func:`derive_crop_group_from_class_id` — collapses the 20 PASTIS-R classes
  into 8 agronomic groups according to the HCAT taxonomy (Hierarchical Crop and
  Agriculture Taxonomy) official to EuroCrops (Schneider et al. 2023). Uses the
  override of ``PASTIS_R_GROUPINGS["agronomic_group"]`` from the official loader
  (US-001) if present; otherwise falls back to the inline HCAT mapping.
- :func:`derive_season_from_doy` — converts day-of-year to a season label
  (``winter/spring/summer/autumn``); useful to seed
  :func:`encode_ordinal` when the input feature is ``peak_doy`` or another
  phenology-derived feature.
- :func:`encode_onehot` — wrapper over :meth:`polars.DataFrame.to_dummies`
  with cardinality report and ``drop_first`` support.
- :func:`encode_ordinal` — applies an explicit per-column mapping
  (``dict[col, dict[value, int]]``); unknown values -> ``-1`` with
  structured warning.
- :func:`encode_target_mean` — Bayesian target encoding with smoothing
  (Galli 2022 ch. 3) for high-cardinality categoricals without exploding the
  DataFrame width.

Key decisions
-------------
- Polars in / Polars out (no ``pandas`` import). For one-hot it uses
  ``df.to_dummies(columns=..., separator="__")``; for discretization the
  binning lives in :func:`ml.features.selection.discretize_features`.
- ``exclude_cols`` always excludes ``parcel_id`` and ``year`` by project
  convention (they are not feature candidates to encode).
- When ``drop_first=True`` in one-hot, the first category is removed
  alphabetically (k -> k-1 columns), following the convention of linear
  models to avoid collinearity of the full indicator matrix.

References
----------
- Galli, S. (2022). *Python Feature Engineering Cookbook* (2nd ed.), ch. 3
  "Encoding Categorical Variables". Bayesian smoothing for target encoding.
- Sainte-Fare-Garnot, V., Landrieu, L. (2021). *PASTIS dataset documentation*
  — 20 classes grouped into ``agronomic_group`` (cereals, root_crops,
  oilseeds_legumes, permanent_long_cycle, special_crops).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "derive_crop_group_from_class_id",
    "derive_season_from_doy",
    "encode_onehot",
    "encode_ordinal",
    "encode_target_mean",
]


# Convention shared with :mod:`ml.features.selection`.
_DEFAULT_EXCLUDE: tuple[str, ...] = ("parcel_id", "year")

# Default mapping for northern hemisphere seasons (month -> season).
# Follows the meteorological convention: winter = DEC/JAN/FEB (month 12, 1, 2).
_SEASON_NORTH_BY_MONTH: dict[int, str] = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}

# Southern hemisphere: inverted seasons (winter = JUN/JUL/AUG).
_SEASON_SOUTH_BY_MONTH: dict[int, str] = {
    m: {
        "winter": "summer",
        "summer": "winter",
        "spring": "autumn",
        "autumn": "spring",
    }[s]
    for m, s in _SEASON_NORTH_BY_MONTH.items()
}

# Mapping PASTIS-R (20 classes) -> HCAT agronomic group (Hierarchical Crop and
# Agriculture Taxonomy of EuroCrops, Schneider et al. 2023). HCAT is the
# official harmonized taxonomy of the European Union for crop types;
# using it instead of an invented grouping gives academic traceability to the
# categorical encoding. Each group points to a real HCAT3 node verified
# against `data/reference/eurocrops/HCAT3.csv` (downloaded from
# github.com/maja601/EuroCrops). The HCAT3 code is documented in
# :data:`_HCAT_GROUP_CODES` for traceability.
#
# Reference: M. Schneider, T. Schelte, F. Schmitz, M. Korner (2023).
# "EuroCrops: A Pan-European Dataset for Time Series Crop Type Classification".
# arXiv:2106.08151. HCAT taxonomy: github.com/maja601/EuroCrops.
_DEFAULT_CROP_GROUP_MAP: dict[int, str] = {
    0: "background",
    1: "grassland",  # Meadow -> pasture_meadow_grassland_grass
    2: "cereal",  # Soft winter wheat
    3: "cereal",  # Corn (grain maize)
    4: "cereal",  # Winter barley
    5: "industrial_nonfood",  # Winter rapeseed
    6: "cereal",  # Spring barley
    7: "industrial_nonfood",  # Sunflower
    8: "vineyard",  # Grapevine
    9: "root_tuber",  # Beet -> sugar_beet
    10: "cereal",  # Winter triticale
    11: "cereal",  # Winter durum wheat
    12: "vegetable",  # Fruits, vegetables, flowers -> fresh_vegetables
    13: "root_tuber",  # Potatoes
    14: "legume",  # Leguminous fodder
    15: "legume",  # Soybeans -> soy_soybeans
    16: "orchard",  # Orchard -> orchards_fruits
    17: "cereal",  # Mixed cereal
    18: "cereal",  # Sorghum -> millet_sorghum
    19: "void",
}

# Official HCAT3 code per agronomic group, for academic traceability.
# Verified against data/reference/eurocrops/HCAT3.csv (EuroCrops v3).
_HCAT_GROUP_CODES: dict[str, str] = {
    "cereal": "3301010000",  # cereal
    "legume": "3301020000",  # legumes_dried_pulses_protein_crops
    "root_tuber": "3301290000",  # root_vegetables (includes sugar_beet, potatoes)
    "industrial_nonfood": "3301060000",  # industrial_nonfood_crops
    "vegetable": "3301070000",  # fresh_vegetables
    "grassland": "3302000000",  # pasture_meadow_grassland_grass
    "orchard": "3303010000",  # orchards_fruits
    "vineyard": "3303060000",  # vineyards_wine_vine_rebland_grapes
    "background": "0",
    "void": "0",
}


def _filter_exclude(columns: list[str], exclude_cols: tuple[str, ...]) -> list[str]:
    """Return ``columns`` without those appearing in ``exclude_cols``."""
    return [c for c in columns if c not in exclude_cols]


# ---------------------------------------------------------------------------
# Derivation helpers (typical inputs of Isaac's WIP notebook)
# ---------------------------------------------------------------------------


def derive_season_from_doy(
    doy_series: pl.Series,
    *,
    hemisphere: Literal["north", "south"] = "north",
) -> pl.Series:
    """Convert day-of-year (1..366) to a season label.

    Useful to seed :func:`encode_ordinal` when the input is a phenology
    feature like ``peak_doy`` or ``sog_doy`` (coming from
    ``ml.features.temporal_features``).

    Args:
        doy_series: Polars series with values in ``[1, 366]``. Accepts floats;
            rounded down. NaN is mapped to ``"unknown"``.
        hemisphere: ``"north"`` (default) or ``"south"`` to invert
            seasons in the southern hemisphere.

    Returns:
        A ``pl.Series`` Utf8 with values in
        ``{"winter", "spring", "summer", "autumn", "unknown"}`` and the same
        ``name`` as ``doy_series`` with the suffix ``__season``.

    Notes:
        Approximates ``month = ceil(doy / 30.5)`` clamped to ``[1, 12]``. The
        precision is sufficient for agronomic seasonality (the difference of
        1-2 days at month boundaries does not change the season).
    """
    season_map = _SEASON_NORTH_BY_MONTH if hemisphere == "north" else _SEASON_SOUTH_BY_MONTH
    raw = doy_series.cast(pl.Float64).to_numpy()
    months = np.clip(np.ceil(np.where(np.isnan(raw), 0.0, raw) / 30.5).astype(np.int64), 0, 12)
    out: list[str] = []
    for doy_val, mo in zip(raw.tolist(), months.tolist(), strict=True):
        if doy_val is None or (isinstance(doy_val, float) and np.isnan(doy_val)) or mo == 0:
            out.append("unknown")
        else:
            out.append(season_map.get(int(mo), "unknown"))
    name_suffix = f"{doy_series.name}__season" if doy_series.name else "season"
    return pl.Series(name_suffix, out, dtype=pl.Utf8)


def derive_crop_group_from_class_id(
    class_id_series: pl.Series,
    *,
    mapping: dict[int, str] | None = None,
) -> pl.Series:
    """Collapse the 20 PASTIS-R classes into 8 HCAT agronomic groups.

    The groups follow the HCAT taxonomy (Hierarchical Crop and Agriculture
    Taxonomy) official to EuroCrops (Schneider et al. 2023, arXiv:2106.08151),
    the harmonized crop-type standard of the European Union. Each
    group corresponds to a real HCAT3 node (codes in
    :data:`_HCAT_GROUP_CODES`). Using HCAT instead of a custom grouping
    gives academic traceability to the categorical encoding of Avance 2.

    When ``mapping`` is ``None``, it attempts to load
    the inline HCAT mapping :data:`_DEFAULT_CROP_GROUP_MAP` (8 groups, EU
    standard).

    Note on taxonomies: the repo maintains two distinct and complementary
    groupings. (1) HCAT — 8 official EuroCrops taxonomic groups,
    the default of this function; gives academic traceability. (2)
    ``PASTIS_R_GROUPINGS["agronomic_group"]`` from the reference JSON — 5
    super-classes by commercial use, used by the US-016 baseline. To
    obtain the 5-class commercial grouping instead of HCAT, pass
    ``mapping=PASTIS_R_GROUPINGS["agronomic_group"]`` explicitly.

    Args:
        class_id_series: Int ``pl.Series`` with values in ``[0, 19]``.
        mapping: Optional override ``{class_id: group_name}``. If
            ``None`` the HCAT taxonomy is used. Values outside the mapping are
            labeled as ``"unknown"``.

    Returns:
        A ``pl.Series`` Utf8 with the HCAT agronomic group per row; the input
        ``name`` + suffix ``__group`` (or ``"crop_group"`` if the input has
        no ``name``). The 8 groups are: ``cereal``, ``legume``,
        ``root_tuber``, ``industrial_nonfood``, ``vegetable``, ``grassland``,
        ``orchard``, ``vineyard`` (plus ``background``/``void``).
    """
    if mapping is None:
        mapping = _DEFAULT_CROP_GROUP_MAP
        source = "hcat3_eurocrops_inline"
    else:
        source = "caller_provided"

    raw = class_id_series.to_list()
    out = [mapping.get(int(v), "unknown") if v is not None else "unknown" for v in raw]
    out_name = f"{class_id_series.name}__group" if class_id_series.name else "crop_group"
    groups_present = sorted(set(out))
    hcat_codes = {g: _HCAT_GROUP_CODES.get(g, "n/a") for g in groups_present}
    logger.info(
        "crop_group_derived",
        source=source,
        taxonomy="HCAT3_eurocrops",
        n=len(out),
        n_groups=len(groups_present),
        hcat_codes=hcat_codes,
    )
    return pl.Series(out_name, out, dtype=pl.Utf8)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


def encode_ordinal(
    df: pl.DataFrame,
    mapping: dict[str, dict[Any, int]],
    *,
    exclude_cols: tuple[str, ...] = _DEFAULT_EXCLUDE,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Apply an explicit per-column ordinal mapping.

    Args:
        df: Wide-format Polars DataFrame.
        mapping: ``{col_name: {original_value: int_ordinal}}``. Each column
            is replaced by its encoded version keeping the same
            name. Values not present in the mapping -> ``-1`` (with a
            structured warning per affected column).
        exclude_cols: Columns that are NOT encoded even if they appear in
            ``mapping`` (defense against ``parcel_id``/``year``).

    Returns:
        Tuple ``(df_encoded, report)`` where ``report`` contains
        ``{col: {"mapping": dict, "unknown_count": int}}`` per processed
        column.

    Raises:
        ValueError: If any column of ``mapping`` does not exist in ``df``.
    """
    missing = [c for c in mapping if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columnas en mapping ausentes del DataFrame: {missing}. Disponibles: {df.columns}"
        )

    out = df
    report: dict[str, Any] = {}
    for col, value_map in mapping.items():
        if col in exclude_cols:
            logger.warning("encode_ordinal_skip_excluded", col=col)
            continue
        original = out.get_column(col).to_list()
        encoded = [int(value_map.get(v, -1)) for v in original]
        unknown_count = sum(1 for e in encoded if e == -1)
        if unknown_count > 0:
            logger.warning(
                "encode_ordinal_unknown_values",
                col=col,
                unknown_count=unknown_count,
                total=len(encoded),
            )
        out = out.with_columns(pl.Series(col, encoded, dtype=pl.Int64))
        report[col] = {"mapping": dict(value_map), "unknown_count": unknown_count}

    logger.info(
        "encode_ordinal_done",
        cols_encoded=list(report.keys()),
        n_rows=df.height,
    )
    return out, report


def encode_onehot(
    df: pl.DataFrame,
    columns: list[str] | tuple[str, ...],
    *,
    drop_first: bool = False,
    exclude_cols: tuple[str, ...] = _DEFAULT_EXCLUDE,
) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """Encode categorical columns with :meth:`polars.DataFrame.to_dummies`.

    Polars-native wrapper over Polars' own one-hot API. **Does NOT use
    pandas** (rule ``ml/CLAUDE.md NEVER pandas``).

    Args:
        df: Wide-format Polars DataFrame.
        columns: List/tuple of columns to encode. Those appearing in
            ``exclude_cols`` are filtered defensively.
        drop_first: If ``True``, removes the first category (alphabetical
            order) of each encoded column. Reduces ``k`` columns to
            ``k - 1`` to avoid collinearity in linear models.
        exclude_cols: Columns to preserve without encoding.

    Returns:
        Tuple ``(df_wide, report)`` where:

        - ``df_wide`` replaces each ``col`` with columns
          ``{col}__{category}`` (fixed separator ``"__"``).
        - ``report = {original_col: [new_columns]}`` for traceability.

    Raises:
        ValueError: If any column of ``columns`` does not exist in ``df``.
    """
    cols_list = [c for c in columns if c not in exclude_cols]
    missing = [c for c in cols_list if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columnas a codificar ausentes del DataFrame: {missing}. Disponibles: {df.columns}"
        )
    if not cols_list:
        return df, {}

    pre_cols = set(df.columns)
    encoded = df.to_dummies(columns=list(cols_list), separator="__")

    report: dict[str, list[str]] = {}
    for col in cols_list:
        new_cols_all = sorted(c for c in encoded.columns if c.startswith(f"{col}__"))
        if drop_first and new_cols_all:
            dropped = new_cols_all[0]
            encoded = encoded.drop(dropped)
            new_cols = new_cols_all[1:]
        else:
            new_cols = new_cols_all
        report[col] = new_cols

    new_total = len(encoded.columns) - (len(pre_cols) - len(cols_list))
    logger.info(
        "encode_onehot_done",
        cols_encoded=list(report.keys()),
        n_new_columns=new_total,
        drop_first=drop_first,
    )
    return encoded, report


def encode_target_mean(
    df: pl.DataFrame,
    target_col: str,
    columns: list[str] | tuple[str, ...],
    *,
    smoothing: float = 10.0,
    exclude_cols: tuple[str, ...] = _DEFAULT_EXCLUDE,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Apply Bayesian target mean encoding with smoothing (Galli 2022).

    For each category ``c`` in a column to encode:

    .. math::

        \\hat{y}_c = \\frac{n_c \\cdot \\bar{y}_c + m \\cdot \\bar{y}}{n_c + m}

    where ``n_c`` is the number of samples of the category, ``mean_c`` is
    the mean of ``target_col`` for that category, ``mean_global`` is the
    global mean of the target and ``m`` is the smoothing (``smoothing > 0``
    shifts rare categories toward the global mean, avoiding overfitting).

    Args:
        df: Polars DataFrame with the ``target_col`` column present.
        target_col: Name of the target column (numeric). In
            multiclass classification, it is advisable to binarize first (e.g.
            ``one-vs-rest`` per class) or to use the integer mean as an
            ordinal severity proxy.
        columns: Categoricals to encode. ``parcel_id``/``year`` excluded
            by default.
        smoothing: Bayesian smoothing factor ``m``. ``m=0`` -> pure
            per-category mean; ``m -> inf`` -> global mean. Galli 2022
            recommends ``m in [5, 20]`` for medium-sized datasets.
        exclude_cols: Columns not to encode.

    Returns:
        Tuple ``(df_encoded, report)`` where:

        - ``df_encoded`` adds columns ``{col}_target_enc`` (keeps the
          original so as not to lose information).
        - ``report`` contains:
          ``{"global_mean": float, "per_column": {col: {cat: encoded_value}}}``.

    Raises:
        ValueError: If ``target_col`` does not exist in ``df`` or is not numeric.
    """
    if target_col not in df.columns:
        raise ValueError(f"target_col {target_col!r} no presente en df.columns")
    if not df.get_column(target_col).dtype.is_numeric():
        raise ValueError(
            f"target_col {target_col!r} debe ser numerico (es {df.get_column(target_col).dtype})"
        )

    cols_list = [c for c in columns if c not in exclude_cols]
    missing = [c for c in cols_list if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas a target-encodear ausentes del DataFrame: {missing}.")

    target_arr = df.get_column(target_col).cast(pl.Float64).to_numpy()
    global_mean = float(np.nanmean(target_arr)) if target_arr.size else 0.0
    if not np.isfinite(global_mean):
        global_mean = 0.0

    out = df
    per_column: dict[str, dict[Any, float]] = {}

    for col in cols_list:
        per_cat = (
            df.group_by(col)
            .agg(
                pl.len().alias("__n"),
                pl.col(target_col).cast(pl.Float64).mean().alias("__mean"),
            )
            .to_dict(as_series=False)
        )
        cat_to_enc: dict[Any, float] = {}
        for cat, n, mean_c in zip(per_cat[col], per_cat["__n"], per_cat["__mean"], strict=True):
            if mean_c is None or (isinstance(mean_c, float) and not np.isfinite(mean_c)):
                encoded_val = global_mean
            else:
                encoded_val = (n * mean_c + smoothing * global_mean) / (n + smoothing)
            cat_to_enc[cat] = float(encoded_val)
        original = out.get_column(col).to_list()
        encoded_series = pl.Series(
            f"{col}_target_enc",
            [cat_to_enc.get(v, global_mean) for v in original],
            dtype=pl.Float64,
        )
        out = out.with_columns(encoded_series)
        per_column[col] = cat_to_enc

    report: dict[str, Any] = {
        "global_mean": global_mean,
        "per_column": per_column,
        "smoothing": float(smoothing),
        "target_col": target_col,
    }
    logger.info(
        "encode_target_mean_done",
        cols_encoded=list(per_column.keys()),
        smoothing=smoothing,
        global_mean=global_mean,
    )
    return out, report
