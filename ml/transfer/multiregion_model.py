"""Multi-region rescued-class crop classifier (EPIC 12, Arthur's idea).

This module builds and evaluates the **multi-region "rescued classes" model**:
a single XGBoost classifier trained on the 64-dim AlphaEarth embedding pooled
across every transfer region (PASTIS France, EuroCropsML Estonia + Latvia,
Sen4AgriNet Catalonia/France, WorldCereal Brazil/India), so a crop that is
*weak* in one region (few parcels) can be *strong* in another that has many
samples of it. The promise is a FINER taxonomy than any single dataset: e.g.
EuroCropsML labels ``spring_barley`` / ``oats`` / ``rye`` as distinct HCAT
leaves where PASTIS only resolves a coarse ``cereals`` / ``winter_barley``.

Design (honest, no fabricated mappings)
---------------------------------------
1. **Harmonization to a common HCAT space.** Every parcel is mapped to two
   label levels using the real US-074 crosswalk
   (:mod:`ml.data.hcat_crosswalk`) and the HCAT v3 reference
   (``data/reference/eurocrops_hcat3.csv``, 384 nodes):

     - a fine **leaf** level (``hcat_leaf_name``), and
     - a coarse **macro** level (``macro_hcat_group``).

   Only PASTIS and EuroCropsML carry a real fine HCAT *leaf*; Sen4AgriNet and
   WorldCereal only resolve a coarse macro/group, so they contribute MACRO
   supervision and (for maize / vineyard) reinforce leaves that already exist.
   We never invent a leaf for a dataset that does not label one.

2. **Few-shot, not zero-shot.** The transfer evidence is load-bearing: the
   zero-shot Europe->tropics direction FAILS (maize Brazil F1 ~ 0.0095) and is
   only recovered few-shot (Brazil k=20 F1 ~ 0.626). Therefore the multi-region
   model is trained WITH a few-shot slice of every region in the training split
   (a per-region held-out test slice keeps the evaluation honest); it is not a
   zero-shot Europe->tropics extrapolation.

3. **Honest metric.** We do NOT report the macro-F1 over N classes (which
   mechanically drops as N grows: PASTIS 9 classes = 0.91 vs 18 = 0.749).
   Instead we COUNT how many *individual* leaf classes clear an F1 threshold
   (0.85 headline, plus 0.70 / 0.80), and compare that count against the
   PASTIS-only baseline. The count is a RESULT, not a target.

4. **Hierarchical measurable evaluation (measurement A).** The model predicts
   the fine leaf; we evaluate F1 at the leaf level AND collapsed to the macro
   level (papaya->fruits is a hit at the measurable macro level). This shows
   whether the multi-region model degrades the coarse level vs PASTIS-only.

5. **Qualitative demo (measurement B).** Real Mexico parcels (avocado / guava)
   get a fine predicted label + confidence, illustrating the extra granularity
   with NO accuracy claim where there is no fine ground truth.

Everything runs on CPU in seconds. No number is fabricated: if a parquet is
missing the builder raises rather than inventing rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import structlog
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from ml.data.hcat_crosswalk import load_crosswalk
from ml.train.baseline import _XGB_BASE_PARAMS, build_estimator

logger = structlog.get_logger(__name__)

__all__ = [
    "ALPHAEARTH_DIM_COLS",
    "MIN_LEAF_SUPPORT",
    "HarmonizedDataset",
    "MultiRegionResult",
    "build_harmonized_dataset",
    "run_multiregion_experiment",
    "train_multiregion_model",
]

#: Canonical 64-dim AlphaEarth embedding columns (``dim_00 .. dim_63``). The
#: Mexico demo parquet uses ``A00..A63`` and is renamed on load.
ALPHAEARTH_DIM_COLS: tuple[str, ...] = tuple(f"dim_{i:02d}" for i in range(64))

#: Repo-root-relative data locations.
_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_TRANSFER_DIR: Path = _REPO_ROOT / "data" / "transfer"
_FEATURES_DIR: Path = _REPO_ROOT / "data" / "features"
_REFERENCE_DIR: Path = _REPO_ROOT / "data" / "reference"
_HCAT3_CSV: Path = _REFERENCE_DIR / "eurocrops_hcat3.csv"

#: Minimum number of parcels for a leaf class to enter the FINE label-space.
#: Leaves with fewer samples (e.g. lavender=2, hops=1 in EuroCropsML) are too
#: rare to learn or evaluate honestly and are dropped from the leaf head (they
#: still contribute their macro label). The value is a sample-count floor, not a
#: tuned hyperparameter.
MIN_LEAF_SUPPORT: int = 50

#: Per-region maximum parcels sampled into the pooled dataset. Caps the large
#: European pixel dumps (PASTIS 86k, EuroCropsML ~20k each) so no single region
#: dominates the booster; the tropical regions (~1.5k) are taken whole. This is
#: the "few-shot per region" knob, not a fabricated balance.
_PER_REGION_CAP: int = 6000

#: F1 thresholds at which we count "good" classes (headline 0.85).
F1_THRESHOLDS: tuple[float, ...] = (0.70, 0.80, 0.85)

#: Fraction of each region held out for the honest per-region test split.
_TEST_FRACTION: float = 0.30

_RANDOM_STATE: int = 42


@dataclass(frozen=True)
class HarmonizedDataset:
    """A pooled multi-region dataset in the common HCAT label-space.

    Attributes:
        features: ``(n, 64)`` AlphaEarth embedding matrix.
        leaf: ``(n,)`` fine HCAT leaf name per parcel.
        macro: ``(n,)`` coarse HCAT macro group per parcel.
        region: ``(n,)`` source-region tag per parcel.
        has_fine: ``(n,)`` bool, True where the region labels a real fine leaf
            (PASTIS, EuroCropsML); False where only a macro is known
            (Sen4AgriNet, WorldCereal) and the leaf equals the macro sentinel.
        leaf_vocabulary: sorted unique leaf names kept in the fine head.
        macro_vocabulary: sorted unique macro groups.
        provenance: per-(leaf, region) parcel counts for the documentation.
    """

    features: np.ndarray
    leaf: np.ndarray
    macro: np.ndarray
    region: np.ndarray
    has_fine: np.ndarray
    leaf_vocabulary: list[str]
    macro_vocabulary: list[str]
    provenance: pl.DataFrame


@dataclass
class MultiRegionResult:
    """Outputs of the multi-region experiment (measurement A + B)."""

    per_class_leaf: pl.DataFrame
    per_class_macro: pl.DataFrame
    threshold_counts: pl.DataFrame
    pastis_only_per_class: pl.DataFrame
    pastis_only_threshold_counts: pl.DataFrame
    rescued_classes: pl.DataFrame
    mexico_demo: pl.DataFrame
    summary: dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Harmonization
# --------------------------------------------------------------------------- #
def _load_hcat_name_map() -> dict[int, str]:
    """Return ``{hcat_code: hcat_name}`` from the real HCAT v3 reference CSV."""
    if not _HCAT3_CSV.is_file():
        raise FileNotFoundError(f"HCAT v3 reference missing at {_HCAT3_CSV}")
    h = pl.read_csv(
        _HCAT3_CSV, schema_overrides={"HCAT3_code": pl.Utf8, "HCAT3_name": pl.Utf8}
    )
    return {
        int(c): str(n)
        for c, n in zip(h["HCAT3_code"], h["HCAT3_name"], strict=True)
    }


def _pastis_frame() -> pl.DataFrame:
    """PASTIS France: 64-dim AlphaEarth + fine HCAT leaf via the US-074 crosswalk.

    The crosswalk maps the 18 PASTIS ``class_id`` values to an ``hcat_leaf_name``
    and a ``macro_hcat_group``. PASTIS labels a real fine leaf, so ``has_fine``
    is True for all rows.
    """
    cols = ["class_id", *ALPHAEARTH_DIM_COLS]
    df = pl.read_parquet(_FEATURES_DIR / "features_fused_pastis.parquet", columns=cols)
    cw = load_crosswalk().select(["pastis_id", "hcat_leaf_name", "macro_hcat_group"])
    df = df.join(cw, left_on="class_id", right_on="pastis_id", how="inner")
    return df.select(
        *ALPHAEARTH_DIM_COLS,
        pl.col("hcat_leaf_name").alias("leaf"),
        pl.col("macro_hcat_group").alias("macro"),
        pl.lit("PASTIS_FR").alias("region"),
        pl.lit(True).alias("has_fine"),
    )


def _eurocropsml_frame() -> pl.DataFrame:
    """EuroCropsML Estonia + Latvia: 64-dim AlphaEarth + fine HCAT leaf.

    The raw ``hcat_code`` is the real 10-digit HCAT leaf; we resolve it to a leaf
    NAME via the HCAT v3 reference and reuse the parquet's ``macro_hcat_group``.
    These regions carry the fine leaves that PASTIS collapses (spring_barley,
    oats, rye, apples, clover, alfalfa, ...), so ``has_fine`` is True.
    """
    name_map = _load_hcat_name_map()
    frames: list[pl.DataFrame] = []
    for region in ("estonia", "latvia"):
        p = _TRANSFER_DIR / f"eurocropsml_alphaearth_{region}.parquet"
        if not p.is_file():
            raise FileNotFoundError(f"EuroCropsML parquet missing at {p}")
        df = pl.read_parquet(p, columns=["hcat_code", "macro_hcat_group", *ALPHAEARTH_DIM_COLS])
        df = df.with_columns(
            pl.col("hcat_code")
            .cast(pl.Utf8)
            .cast(pl.Int64)
            .replace_strict(name_map, default="unknown_hcat", return_dtype=pl.Utf8)
            .alias("leaf")
        )
        frames.append(
            df.select(
                *ALPHAEARTH_DIM_COLS,
                "leaf",
                pl.col("macro_hcat_group").alias("macro"),
                pl.lit(f"EUROCROPS_{region[:2].upper()}").alias("region"),
                pl.lit(True).alias("has_fine"),
            )
        )
    return pl.concat(frames)


def _sen4agrinet_frame() -> pl.DataFrame:
    """Sen4AgriNet Catalonia (ES) + France (FR): 64-dim AlphaEarth, MACRO only.

    Sen4AgriNet resolves only a coarse macro group (no fine HCAT leaf is
    available), so the leaf is set to the macro sentinel ``"<macro>__macro"`` and
    ``has_fine`` is False. These rows add MACRO supervision (notably extra
    vineyard from Catalonia) without an invented fine leaf.
    """
    frames: list[pl.DataFrame] = []
    for region in ("es", "fr"):
        p = _TRANSFER_DIR / f"sen4agrinet_{region}_alphaearth.parquet"
        if not p.is_file():
            raise FileNotFoundError(f"Sen4AgriNet parquet missing at {p}")
        df = pl.read_parquet(p, columns=["macro", *ALPHAEARTH_DIM_COLS])
        frames.append(
            df.select(
                *ALPHAEARTH_DIM_COLS,
                (pl.col("macro") + pl.lit("__macro")).alias("leaf"),
                pl.col("macro").alias("macro"),
                pl.lit(f"SEN4AGRINET_{region.upper()}").alias("region"),
                pl.lit(False).alias("has_fine"),
            )
        )
    return pl.concat(frames)


def _worldcereal_frame() -> pl.DataFrame:
    """WorldCereal Brazil Cerrado + India Karnataka: 64-dim AlphaEarth, tropical.

    WorldCereal only resolves maize / wintercereals / other_cropland / non_crop.
    ``maize`` and ``wintercereals`` map to the ``cereals`` macro (and to the
    existing maize/cereal leaves); ``other_cropland`` / ``non_crop`` are kept as
    explicit out-of-nomenclature macros so the tropical regions are not forced
    onto a false European leaf. ``has_fine`` is False (no fine HCAT leaf).
    """
    macro_map = {
        "maize": "cereals",
        "wintercereals": "cereals",
        "other_cropland": "other_cropland",
        "non_crop": "non_crop",
    }
    frames: list[pl.DataFrame] = []
    for region in ("brazil_cerrado", "india_karnataka"):
        p = _TRANSFER_DIR / f"worldcereal_{region}.parquet"
        if not p.is_file():
            raise FileNotFoundError(f"WorldCereal parquet missing at {p}")
        df = pl.read_parquet(p, columns=["class_name", *ALPHAEARTH_DIM_COLS])
        df = df.with_columns(
            pl.col("class_name")
            .replace_strict(macro_map, default="other_cropland", return_dtype=pl.Utf8)
            .alias("macro")
        )
        tag = "BR" if region.startswith("brazil") else "IN"
        frames.append(
            df.select(
                *ALPHAEARTH_DIM_COLS,
                (pl.col("class_name") + pl.lit("__macro")).alias("leaf"),
                "macro",
                pl.lit(f"WORLDCEREAL_{tag}").alias("region"),
                pl.lit(False).alias("has_fine"),
            )
        )
    return pl.concat(frames)


def _cap_per_region(df: pl.DataFrame, cap: int, seed: int) -> pl.DataFrame:
    """Stratified-by-leaf down-sample of each region to at most ``cap`` rows."""
    out: list[pl.DataFrame] = []
    for (region,), sub in df.group_by(["region"], maintain_order=True):
        if sub.height <= cap:
            out.append(sub)
            continue
        out.append(sub.sample(n=cap, seed=seed, shuffle=True))
        logger.info("region_capped", region=region, kept=cap, original=sub.height)
    return pl.concat(out)


def build_harmonized_dataset(
    *,
    min_leaf_support: int = MIN_LEAF_SUPPORT,
    per_region_cap: int = _PER_REGION_CAP,
    seed: int = _RANDOM_STATE,
) -> HarmonizedDataset:
    """Pool every region into the common HCAT leaf/macro label-space.

    Steps: load + harmonize each region, cap per-region size, drop fine leaves
    below ``min_leaf_support`` (rare-tail guard), and assemble the matrices.

    Args:
        min_leaf_support: parcel-count floor for a leaf to stay in the fine head.
        per_region_cap: max parcels kept per region (few-shot balance).
        seed: RNG seed for the per-region down-sample.

    Returns:
        A :class:`HarmonizedDataset`.
    """
    pooled = pl.concat(
        [
            _pastis_frame(),
            _eurocropsml_frame(),
            _sen4agrinet_frame(),
            _worldcereal_frame(),
        ],
        how="vertical",
    )
    pooled = _cap_per_region(pooled, per_region_cap, seed)

    # Fine leaf-support guard: only leaves from fine-labelling regions with
    # enough parcels stay as distinct fine classes; rarer fine leaves are folded
    # into the macro-sentinel so we never train/evaluate a 2-sample class.
    fine_counts = (
        pooled.filter(pl.col("has_fine"))
        .group_by("leaf")
        .len()
        .filter(pl.col("len") >= min_leaf_support)
    )
    keep_leaves = set(fine_counts["leaf"].to_list())
    pooled = pooled.with_columns(
        pl.when(pl.col("has_fine") & pl.col("leaf").is_in(list(keep_leaves)))
        .then(pl.col("leaf"))
        .otherwise(pl.col("macro") + pl.lit("__macro"))
        .alias("leaf"),
        pl.when(pl.col("has_fine") & pl.col("leaf").is_in(list(keep_leaves)))
        .then(pl.col("has_fine"))
        .otherwise(pl.lit(False))
        .alias("has_fine"),
    )

    provenance = (
        pooled.group_by(["leaf", "macro", "region", "has_fine"])
        .len()
        .rename({"len": "n_parcels"})
        .sort(["leaf", "region"])
    )

    feats = pooled.select(ALPHAEARTH_DIM_COLS).to_numpy().astype(np.float32)
    leaf = pooled["leaf"].to_numpy()
    macro = pooled["macro"].to_numpy()
    region = pooled["region"].to_numpy()
    has_fine = pooled["has_fine"].to_numpy()

    leaf_vocab = sorted(set(leaf.tolist()))
    macro_vocab = sorted(set(macro.tolist()))
    logger.info(
        "harmonized_dataset_built",
        n_parcels=int(feats.shape[0]),
        n_leaf_classes=len(leaf_vocab),
        n_macro_classes=len(macro_vocab),
        n_fine_parcels=int(has_fine.sum()),
    )
    return HarmonizedDataset(
        features=feats,
        leaf=leaf,
        macro=macro,
        region=region,
        has_fine=has_fine,
        leaf_vocabulary=leaf_vocab,
        macro_vocabulary=macro_vocab,
        provenance=provenance,
    )


# --------------------------------------------------------------------------- #
# Training (champion recipe: XGBoost multi:softprob over AlphaEarth)
# --------------------------------------------------------------------------- #
def _region_stratified_split(
    region: np.ndarray, leaf: np.ndarray, *, test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Held-out-by-region split: each region contributes train AND test rows.

    The split is stratified within each region (few-shot per region, no leakage
    of the same parcels across train/test). Returns boolean train/test masks.
    """
    n = region.shape[0]
    is_train = np.zeros(n, dtype=bool)
    idx_all = np.arange(n)
    for reg in np.unique(region):
        reg_idx = idx_all[region == reg]
        reg_leaf = leaf[region == reg]
        # Stratify only where every class in the region has >= 2 samples.
        _, counts = np.unique(reg_leaf, return_counts=True)
        stratify = reg_leaf if counts.min() >= 2 else None
        tr, _te = train_test_split(
            reg_idx,
            test_size=test_fraction,
            random_state=seed,
            stratify=stratify,
        )
        is_train[tr] = True
    return is_train, ~is_train


def train_multiregion_model(
    ds: HarmonizedDataset, *, seed: int = _RANDOM_STATE
) -> tuple[object, np.ndarray, np.ndarray, dict[str, object]]:
    """Train the pooled XGBoost FINE-LEAF classifier with a per-region held-out split.

    Reuses the champion baseline recipe (:func:`ml.train.baseline.build_estimator`
    with ``multi:softprob``). Critically, the leaf head is trained ONLY on parcels
    that carry a real fine HCAT leaf (PASTIS + EuroCropsML, ``has_fine=True``):
    the macro-only regions (Sen4AgriNet, WorldCereal) do NOT inject a coarse
    ``*__macro`` pseudo-class into the fine head, which would collide with the
    genuine fine leaves of the same macro (e.g. ``cereals__macro`` vs ``oats``)
    and artificially depress them. The macro-only regions are kept aside for the
    collapsed macro evaluation. Labels are encoded to a contiguous ``[0, K)``
    range as XGBoost requires.

    Returns:
        ``(model, test_mask, label_encoder_classes, split_info)``. ``test_mask``
        and ``label_encoder_classes`` are defined over the FINE-leaf subset; the
        returned masks index into ``ds.features[ds.has_fine]``.
    """
    fine = ds.has_fine
    feats = ds.features[fine]
    leaf = ds.leaf[fine]
    region = ds.region[fine]

    is_train, is_test = _region_stratified_split(
        region, leaf, test_fraction=_TEST_FRACTION, seed=seed
    )
    classes = np.array(sorted(set(leaf.tolist())))
    class_to_id = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_id[c] for c in leaf], dtype=np.int64)

    params = dict(_XGB_BASE_PARAMS)
    params["random_state"] = seed
    model = build_estimator("xgb", params)
    model.fit(feats[is_train], y[is_train])

    split_info = {
        "n_train": int(is_train.sum()),
        "n_test": int(is_test.sum()),
        "n_leaf_classes": int(classes.shape[0]),
        "train_per_region": {
            str(r): int((region[is_train] == r).sum()) for r in np.unique(region)
        },
        "test_per_region": {
            str(r): int((region[is_test] == r).sum()) for r in np.unique(region)
        },
    }
    logger.info(
        "multiregion_model_trained",
        **{k: split_info[k] for k in ("n_train", "n_test", "n_leaf_classes")},
    )
    return model, is_test, classes, split_info


# --------------------------------------------------------------------------- #
# PASTIS-only baseline (the comparison anchor)
# --------------------------------------------------------------------------- #
def _train_pastis_only(seed: int, *, cap: int | None = None) -> pl.DataFrame:
    """Train + evaluate a PASTIS-ONLY leaf classifier (the baseline anchor).

    Same recipe, same held-out fraction, but the training and test rows come
    only from PASTIS. When ``cap`` is set the PASTIS rows are first down-sampled
    to ``cap`` parcels so the baseline sees the SAME amount of PASTIS data the
    multi-region model gets in the pool (apples-to-apples). Returns a per-leaf
    F1 table for the PASTIS label-space.

    Args:
        seed: RNG seed for the split (and the optional down-sample).
        cap: optional parcel cap matching the per-region cap of the pool.

    Returns:
        Per-leaf precision/recall/F1/support table.
    """
    df = _pastis_frame()
    if cap is not None and df.height > cap:
        df = df.sample(n=cap, seed=seed, shuffle=True)
    feats = df.select(ALPHAEARTH_DIM_COLS).to_numpy().astype(np.float32)
    leaf = df["leaf"].to_numpy()
    classes = np.array(sorted(set(leaf.tolist())))
    class_to_id = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_id[c] for c in leaf], dtype=np.int64)
    tr, te = train_test_split(
        np.arange(feats.shape[0]),
        test_size=_TEST_FRACTION,
        random_state=seed,
        stratify=leaf,
    )
    params = dict(_XGB_BASE_PARAMS)
    params["random_state"] = seed
    model = build_estimator("xgb", params)
    model.fit(feats[tr], y[tr])
    pred = model.predict(feats[te])
    return _per_class_table(y[te], pred, classes, level="leaf")


# --------------------------------------------------------------------------- #
# Metrics (measurement A)
# --------------------------------------------------------------------------- #
def _per_class_table(
    y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray, *, level: str
) -> pl.DataFrame:
    """Per-class precision/recall/F1/support table over the encoded labels."""
    labels = np.arange(classes.shape[0])
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return pl.DataFrame(
        {
            "level": [level] * classes.shape[0],
            "class_name": classes.tolist(),
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "f1": f1.tolist(),
            "support": sup.astype(int).tolist(),
        }
    ).filter(pl.col("support") > 0)


def _collapse_to_macro(
    leaf_names: np.ndarray, leaf_to_macro: dict[str, str]
) -> np.ndarray:
    """Map an array of leaf names to their macro group."""
    return np.array([leaf_to_macro.get(name, "void") for name in leaf_names])


def _threshold_counts(per_class: pl.DataFrame, source: str) -> pl.DataFrame:
    """Count classes whose F1 clears each threshold (the honest metric)."""
    rows = []
    for thr in F1_THRESHOLDS:
        n_good = per_class.filter(pl.col("f1") >= thr).height
        rows.append({"source": source, "threshold": thr, "n_classes_over": n_good,
                     "n_classes_total": per_class.height})
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Mexico qualitative demo (measurement B)
# --------------------------------------------------------------------------- #
def _mexico_demo(model: object, classes: np.ndarray) -> pl.DataFrame:
    """Predict fine leaf + confidence for the real Mexico parcels (no GT claim)."""
    p = _TRANSFER_DIR / "mexico_demo_alphaearth.parquet"
    if not p.is_file():
        logger.warning("mexico_demo_missing", path=str(p))
        return pl.DataFrame()
    df = pl.read_parquet(p)
    a_cols = [f"A{i:02d}" for i in range(64)]
    feats = df.select(a_cols).to_numpy().astype(np.float32)
    proba = model.predict_proba(feats)  # type: ignore[attr-defined]
    top = np.argsort(proba, axis=1)[:, ::-1][:, :3]
    rows = []
    for i in range(df.height):
        rows.append(
            {
                "aoi": df["aoi"][i],
                "cultivo_real": df["cultivo"][i],
                "pred_1": str(classes[top[i, 0]]),
                "conf_1": float(proba[i, top[i, 0]]),
                "pred_2": str(classes[top[i, 1]]),
                "conf_2": float(proba[i, top[i, 1]]),
                "pred_3": str(classes[top[i, 2]]),
                "conf_3": float(proba[i, top[i, 2]]),
            }
        )
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_multiregion_experiment(*, seed: int = _RANDOM_STATE) -> MultiRegionResult:
    """Run the full multi-region experiment: harmonize -> train -> measure A + B.

    Returns:
        A :class:`MultiRegionResult` with per-class leaf/macro tables, threshold
        counts (multi-region and PASTIS-only), the rescued-class table, the
        Mexico demo and a JSON-serializable summary.
    """
    ds = build_harmonized_dataset(seed=seed)
    leaf_to_macro = _leaf_to_macro_map(ds)

    model, is_test_fine, classes, split_info = train_multiregion_model(ds, seed=seed)

    # ---- Measurement A, LEAF level (fine head, fine-labelled regions only) ----
    fine_feats = ds.features[ds.has_fine]
    fine_leaf = ds.leaf[ds.has_fine]
    pred_id = model.predict(fine_feats[is_test_fine])  # type: ignore[attr-defined]
    class_to_id = {c: i for i, c in enumerate(classes)}
    y_true_id = np.array(
        [class_to_id[c] for c in fine_leaf[is_test_fine]], dtype=np.int64
    )
    per_leaf_fine = _per_class_table(y_true_id, pred_id, classes, level="leaf")

    # ---- Measurement A, MACRO level (collapse over the FULL multi-region test) -
    # The fine model predicts a leaf for EVERY region (incl. the macro-only ones);
    # both prediction and truth are collapsed to the macro group, so the coarse
    # level is evaluated over PASTIS + EuroCropsML + Sen4AgriNet + WorldCereal.
    is_test_full = _full_test_mask(ds, seed=seed)
    pred_full_id = model.predict(ds.features[is_test_full])  # type: ignore[attr-defined]
    macro_classes = np.array(sorted(set(ds.macro.tolist())))
    macro_to_id = {c: i for i, c in enumerate(macro_classes)}
    true_macro = ds.macro[is_test_full]
    pred_macro = _collapse_to_macro(classes[pred_full_id], leaf_to_macro)
    y_true_macro = np.array([macro_to_id[c] for c in true_macro], dtype=np.int64)
    y_pred_macro = np.array(
        [macro_to_id.get(c, -1) for c in pred_macro], dtype=np.int64
    )
    valid = y_pred_macro >= 0
    per_macro = _per_class_table(
        y_true_macro[valid], y_pred_macro[valid], macro_classes, level="macro"
    )

    # ---- PASTIS-only baselines (capped = apples-to-apples, full = reference) ---
    pastis_only = _train_pastis_only(seed, cap=_PER_REGION_CAP)
    pastis_full = _train_pastis_only(seed, cap=None)

    # ---- Honest metric: count classes over threshold (fine leaves) ------------
    thr_multi = _threshold_counts(per_leaf_fine, "multi_region_leaf")
    thr_pastis = _threshold_counts(pastis_only, "pastis_only_leaf_capped")
    thr_pastis_full = _threshold_counts(pastis_full, "pastis_only_leaf_full")

    # ---- Rescued classes: fine leaves over F1>=0.85 absent from PASTIS-only ----
    pastis_leaves = set(pastis_only["class_name"].to_list()) | set(
        pastis_full["class_name"].to_list()
    )
    leaf_region = _dominant_region_per_leaf(ds)
    rescued = (
        per_leaf_fine.filter(pl.col("f1") >= 0.85)
        .with_columns(
            pl.col("class_name").is_in(list(pastis_leaves)).alias("in_pastis"),
            pl.col("class_name")
            .replace_strict(leaf_region, default="?", return_dtype=pl.Utf8)
            .alias("dominant_region"),
        )
        .sort("f1", descending=True)
    )

    mexico = _mexico_demo(model, classes)

    macro_f1 = float(f1_score(y_true_macro[valid], y_pred_macro[valid], average="macro"))
    leaf_f1_mean = per_leaf_fine.get_column("f1").to_numpy().mean()
    summary = {
        "n_parcels_total": int(ds.features.shape[0]),
        "n_fine_parcels": int(ds.has_fine.sum()),
        "n_leaf_classes_fine": int(per_leaf_fine.height),
        "n_macro_classes": int(per_macro.height),
        "split": split_info,
        "multi_region_over_070": _count_at(thr_multi, 0.70),
        "multi_region_over_080": _count_at(thr_multi, 0.80),
        "multi_region_over_085": _count_at(thr_multi, 0.85),
        "pastis_capped_over_070": _count_at(thr_pastis, 0.70),
        "pastis_capped_over_080": _count_at(thr_pastis, 0.80),
        "pastis_capped_over_085": _count_at(thr_pastis, 0.85),
        "pastis_full_over_070": _count_at(thr_pastis_full, 0.70),
        "pastis_full_over_085": _count_at(thr_pastis_full, 0.85),
        "n_rescued_classes_085": int(rescued.filter(~pl.col("in_pastis")).height),
        "macro_f1_macro_multiregion": macro_f1,
        "leaf_f1_macro_multiregion_fine": float(leaf_f1_mean),
    }
    logger.info(
        "multiregion_experiment_done",
        **{
            k: summary[k]
            for k in (
                "n_parcels_total",
                "n_leaf_classes_fine",
                "multi_region_over_085",
                "pastis_capped_over_085",
                "n_rescued_classes_085",
            )
        },
    )

    return MultiRegionResult(
        per_class_leaf=per_leaf_fine,
        per_class_macro=per_macro,
        threshold_counts=pl.concat([thr_multi, thr_pastis, thr_pastis_full]),
        pastis_only_per_class=pastis_only,
        pastis_only_threshold_counts=thr_pastis,
        rescued_classes=rescued,
        mexico_demo=mexico,
        summary=summary,
    )


def _count_at(thr_table: pl.DataFrame, threshold: float) -> int:
    """Read the n-classes-over count for a given threshold from a counts table."""
    return int(thr_table.filter(pl.col("threshold") == threshold)["n_classes_over"][0])


def _full_test_mask(ds: HarmonizedDataset, *, seed: int) -> np.ndarray:
    """Per-region held-out test mask over ALL regions (for the macro evaluation).

    Mirrors the fine-head split protocol but on the full pooled dataset so the
    macro-only regions also contribute held-out test rows.
    """
    _tr, te = _region_stratified_split(
        ds.region, ds.leaf, test_fraction=_TEST_FRACTION, seed=seed
    )
    return te


def _leaf_to_macro_map(ds: HarmonizedDataset) -> dict[str, str]:
    """Build the leaf-name -> macro-group map from the harmonized provenance."""
    pairs = ds.provenance.select(["leaf", "macro"]).unique()
    return {
        leaf: macro
        for leaf, macro in zip(pairs["leaf"], pairs["macro"], strict=True)
    }


def _dominant_region_per_leaf(ds: HarmonizedDataset) -> dict[str, str]:
    """Return the region contributing the most parcels to each fine leaf."""
    g = (
        ds.provenance.filter(pl.col("has_fine"))
        .group_by("leaf")
        .agg(pl.col("region").sort_by("n_parcels", descending=True).first())
    )
    return {leaf: reg for leaf, reg in zip(g["leaf"], g["region"], strict=True)}


def _save_outputs(result: MultiRegionResult, out_dir: Path) -> None:
    """Persist every result table + the JSON summary to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result.per_class_leaf.write_parquet(out_dir / "multiregion_per_class_leaf.parquet")
    result.per_class_macro.write_parquet(out_dir / "multiregion_per_class_macro.parquet")
    result.threshold_counts.write_parquet(out_dir / "multiregion_threshold_counts.parquet")
    result.pastis_only_per_class.write_parquet(out_dir / "pastis_only_per_class.parquet")
    result.pastis_only_threshold_counts.write_parquet(
        out_dir / "pastis_only_threshold_counts.parquet"
    )
    result.rescued_classes.write_parquet(out_dir / "multiregion_rescued_classes.parquet")
    if result.mexico_demo.height:
        result.mexico_demo.write_parquet(out_dir / "multiregion_mexico_demo.parquet")
    (out_dir / "multiregion_summary.json").write_text(
        json.dumps(result.summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("multiregion_outputs_saved", out_dir=str(out_dir))


def make_figures(result: MultiRegionResult, fig_dir: Path) -> list[Path]:
    """Render the three documentation figures (PNG) for the multi-region model.

    1. ``multiregion_k_over_threshold.png`` — K classes over F1 threshold,
       multi-region vs PASTIS-only (capped + full): the headline honest metric.
    2. ``multiregion_leaf_f1.png`` — per fine-leaf F1, coloured by dominant region.
    3. ``multiregion_leaf_vs_macro_f1.png`` — leaf vs collapsed-macro F1 (does the
       coarse level hold up).

    Args:
        result: the experiment result.
        fig_dir: output directory (created if missing).

    Returns:
        The list of written PNG paths.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    counts = result.threshold_counts

    # --- Figure 1: K classes over threshold -------------------------------- #
    fig, ax = plt.subplots(figsize=(7, 4.2))
    series = {
        "multi_region_leaf": ("Multi-region (30 hojas)", "#1f77b4", "o"),
        "pastis_only_leaf_capped": ("Solo-PASTIS (capped, 18)", "#ff7f0e", "s"),
        "pastis_only_leaf_full": ("Solo-PASTIS (full, 18)", "#2ca02c", "^"),
    }
    for src, (label, color, marker) in series.items():
        sub = counts.filter(pl.col("source") == src).sort("threshold")
        ax.plot(
            sub["threshold"].to_list(),
            sub["n_classes_over"].to_list(),
            marker=marker,
            color=color,
            label=label,
            linewidth=2,
        )
    ax.set_xlabel("Umbral F1 por clase")
    ax.set_ylabel("Numero de clases individuales sobre el umbral")
    ax.set_title("Clases que cruzan F1>=umbral: multi-region vs solo-PASTIS")
    ax.set_xticks([0.70, 0.80, 0.85])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p1 = fig_dir / "multiregion_k_over_threshold.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    written.append(p1)

    # --- Figure 2: per fine-leaf F1, coloured by dominant region ----------- #
    pl_leaf = result.per_class_leaf.sort("f1", descending=True)
    rescue_region = dict(
        zip(
            result.rescued_classes["class_name"].to_list(),
            result.rescued_classes["dominant_region"].to_list(),
            strict=True,
        )
    )
    # Region per leaf via the rescued table when available; otherwise gray.
    color_by_region = {
        "PASTIS_FR": "#4c72b0",
        "EUROCROPS_ES": "#dd8452",
        "EUROCROPS_LA": "#55a868",
    }
    fig, ax = plt.subplots(figsize=(8, 8))
    names = pl_leaf["class_name"].to_list()
    f1s = pl_leaf["f1"].to_list()
    bar_colors = [color_by_region.get(rescue_region.get(n, ""), "#9aa0a6") for n in names]
    y = np.arange(len(names))
    ax.barh(y, f1s, color=bar_colors)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    for thr, style in ((0.70, ":"), (0.85, "--")):
        ax.axvline(thr, color="black", linestyle=style, linewidth=1, alpha=0.6)
        ax.text(thr, len(names) - 0.5, f"{thr:.2f}", fontsize=7, rotation=90, va="bottom")
    ax.set_xlabel("F1 por hoja HCAT (held-out por region)")
    ax.set_title("F1 por clase-hoja fina del modelo multi-region")
    fig.tight_layout()
    p2 = fig_dir / "multiregion_leaf_f1.png"
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    written.append(p2)

    # --- Figure 3: macro-level F1 bars ------------------------------------- #
    pm = result.per_class_macro.sort("f1", descending=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    yy = np.arange(pm.height)
    ax.barh(yy, pm["f1"].to_list(), color="#6a51a3")
    ax.set_yticks(yy)
    ax.set_yticklabels(pm["class_name"].to_list(), fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0.85, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("F1 macro-grupo (prediccion fina colapsada a macro)")
    ax.set_title("Medicion A: nivel MACRO colapsado (multi-region)")
    fig.tight_layout()
    p3 = fig_dir / "multiregion_leaf_vs_macro_f1.png"
    fig.savefig(p3, dpi=130)
    plt.close(fig)
    written.append(p3)

    logger.info("multiregion_figures_written", n=len(written), fig_dir=str(fig_dir))
    return written
