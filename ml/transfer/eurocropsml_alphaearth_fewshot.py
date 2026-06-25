"""EuroCropsML few-shot transfer over AlphaEarth embeddings (champion recipe).

Companion to :mod:`ml.transfer.eurocropsml_fewshot`. US-076 ran the k-shot
``LV -> EE`` transfer over a vector derived from the parcel's own raw Sentinel-2
series, because EuroCropsML ships no AlphaEarth embedding. This module closes the
question Arthur raised ("why not the champion recipe?"): it samples the
AlphaEarth Foundations 64-dim annual embedding (``SATELLITE_EMBEDDING/V1/ANNUAL``
v1.1, CC-BY-4.0) at each parcel's ``center`` ``[lon, lat]`` via Google Earth
Engine, caches it, and runs the **same** XGBoost ``multi:softprob`` recipe over
that 64-dim AlphaEarth vector instead of the S2-derived vector.

Honest constraints (real data only, never fabricated):

- EuroCropsML has 175 906 EE parcels + 431 143 LV parcels. Sampling all 607 049
  through GEE ``reduceRegions`` (500 points/batch) is ~1214 batches and would
  dominate the GEE compute budget. We instead draw a **per-macro-class stratified
  subset**: up to ``per_class_cap`` parcels per macro HCAT group per country
  (rare classes -- vineyard, sugar_beet, soybean -- are taken in full). This
  preserves the full label space, gives enough common-class parcels for high k,
  and keeps the GEE cost bounded. The exact counts sampled are logged and written
  to the cache so the entregable can report them.
- The AlphaEarth annual image is the 2021 mosaic, matching the EuroCropsML S2
  acquisition year (the per-parcel ``dates`` span 2021-02 .. 2021-12).
- The label alignment to the US-074 ``hcat-macro`` space and the ``null-class``
  drop are identical to :mod:`ml.transfer.eurocropsml_fewshot`, so the two curves
  are directly comparable on the same label space and the same ``LV -> EE``
  protocol.

If GEE returns no embedding for a parcel (point outside any tile), the row is
dropped, never imputed.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import structlog
from sklearn.metrics import f1_score

from ml.ingest.gee_sampler import (
    ALPHAEARTH_DIM_COLS,
    init_ee,
    sample_alphaearth_at_coords,
)
from ml.train.baseline import build_estimator
from ml.transfer.label_align import NULL_CLASS, align_codes_to_hcat_macro

logger = structlog.get_logger(__name__)

__all__ = [
    "ALPHAEARTH_YEAR",
    "DEFAULT_PER_CLASS_CAP",
    "K_SHOTS_EXTENDED",
    "build_alphaearth_fewshot_splits",
    "load_parcel_index",
    "read_centers_for",
    "run_alphaearth_fewshot_curve",
    "sample_country_alphaearth",
    "summarize_curve",
    "train_xgb_kshot_alphaearth",
]

#: AlphaEarth annual mosaic year matching the EuroCropsML 2021 S2 series.
ALPHAEARTH_YEAR: int = 2021

#: Extended k ladder (US-076 stopped at 500; here we push to 2000 to find where
#: the AlphaEarth champion saturates).
K_SHOTS_EXTENDED: tuple[int, ...] = (1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)

#: Default per-macro-class cap per country for the stratified GEE subset.
DEFAULT_PER_CLASS_CAP: int = 3500

_REGION_CODES: dict[str, str] = {"estonia": "EE", "latvia": "LV", "portugal": "PT"}

#: ``<NUTS3>_<parcel_id>_<EC_hcat_c>.npz`` -- region letters + parcel id + hcat.
_NPZ_NAME_RE = re.compile(r"^(?P<region>[A-Z]{2}[A-Z0-9]*)_(?P<parcel>\d+)_(?P<hcat>\d+)$")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ROOT = _REPO_ROOT / "data" / "transfer" / "eurocropsml"
_CACHE_DIR = _REPO_ROOT / "data" / "transfer"


# ---------------------------------------------------------------------------
# Parcel index: read centers + HCAT labels straight from the npz filenames.
# ---------------------------------------------------------------------------


def load_parcel_index(
    root: Path | str,
    region: str,
) -> pl.DataFrame:
    """Index every parcel of one country from its ``.npz`` **filename** alone.

    The ``center`` ``[lon, lat]`` lives inside each ``.npz`` payload, but opening
    600 k zip archives just to read a 2-float array is pathologically slow on
    Windows (>10 min for one country). Filenames already carry the parcel id and
    the ``EC_hcat_c`` code, which is all we need to assign the macro HCAT label and
    do the stratified selection; the centers are read lazily for the (much
    smaller) selected subset by :func:`read_centers_for`. This function therefore
    does NOT open any ``.npz``.

    Args:
        root: EuroCropsML root directory (holding ``preprocess/``).
        region: Country key (``"estonia"`` / ``"latvia"``).

    Returns:
        A Polars frame with columns ``px_id, parcel_id, npz_name, hcat_code,
        macro_hcat_group`` (no centers yet).

    Raises:
        ValueError: if ``region`` is unknown.
        FileNotFoundError: if no ``.npz`` parcels are found.
    """
    if region not in _REGION_CODES:
        raise ValueError(f"Unknown region {region!r}; expected one of {sorted(_REGION_CODES)}.")
    root = Path(root)
    code = _REGION_CODES[region]
    npz_dir = root / "preprocess"
    if not npz_dir.exists():
        raise FileNotFoundError(f"No preprocess dir under {root}.")

    parcel_ids: list[int] = []
    codes: list[int] = []
    px_ids: list[str] = []
    names: list[str] = []

    for entry in os.scandir(npz_dir):
        name = entry.name
        if not name.endswith(".npz") or not name.startswith(code):
            continue
        match = _NPZ_NAME_RE.match(name[:-4])
        if match is None:
            continue
        pid = int(match.group("parcel"))
        parcel_ids.append(pid)
        codes.append(int(match.group("hcat")))
        px_ids.append(f"{region}_{pid}")
        names.append(name)

    macro = align_codes_to_hcat_macro(codes)
    frame = pl.DataFrame(
        {
            "px_id": px_ids,
            "parcel_id": parcel_ids,
            "npz_name": names,
            "hcat_code": codes,
            "macro_hcat_group": macro,
        }
    )
    logger.info("parcel_index_built", region=region, n_parcels=frame.height)
    return frame


def read_centers_for(root: Path | str, index_subset: pl.DataFrame) -> pl.DataFrame:
    """Read the ``center`` ``[lon, lat]`` from the ``.npz`` of the selected parcels.

    Opens only the ``.npz`` files named in ``index_subset`` (the stratified
    subset, ~tens of thousands, not the full ~600 k), keeping the GEE-bound
    workload bounded and the I/O tractable. Rows whose center is missing or
    non-finite are dropped (never imputed).

    Args:
        root: EuroCropsML root directory.
        index_subset: Subset frame from :func:`load_parcel_index` (must carry
            ``npz_name``).

    Returns:
        ``index_subset`` augmented with ``lon`` and ``lat`` columns (finite only).
    """
    npz_dir = Path(root) / "preprocess"
    lons: list[float | None] = []
    lats: list[float | None] = []
    for name in index_subset.get_column("npz_name").to_list():
        try:
            with np.load(npz_dir / name) as payload:
                center = np.asarray(payload["center"], dtype=np.float64).ravel()
        except (OSError, KeyError, ValueError):
            lons.append(None)
            lats.append(None)
            continue
        if center.size < 2 or not np.all(np.isfinite(center[:2])):
            lons.append(None)
            lats.append(None)
            continue
        lons.append(float(center[0]))
        lats.append(float(center[1]))
    out = index_subset.with_columns(
        pl.Series("lon", lons, dtype=pl.Float64),
        pl.Series("lat", lats, dtype=pl.Float64),
    ).filter(pl.col("lon").is_not_null() & pl.col("lat").is_not_null())
    logger.info("centers_read", region_rows=out.height, requested=index_subset.height)
    return out


def _stratified_subset(
    index: pl.DataFrame,
    per_class_cap: int,
    seed: int,
) -> pl.DataFrame:
    """Draw up to ``per_class_cap`` parcels per macro class (rare classes in full).

    Args:
        index: Frame from :func:`load_parcel_index`.
        per_class_cap: Maximum parcels kept per macro HCAT group.
        seed: Deterministic shuffle seed.

    Returns:
        The stratified subset frame.
    """
    return (
        index.filter(pl.col("macro_hcat_group") != NULL_CLASS)
        .with_columns(
            pl.int_range(pl.len()).shuffle(seed=seed).over("macro_hcat_group").alias("_rank")
        )
        .filter(pl.col("_rank") < per_class_cap)
        .drop("_rank")
    )


# ---------------------------------------------------------------------------
# AlphaEarth sampling per country (cached parquet).
# ---------------------------------------------------------------------------


def sample_country_alphaearth(
    root: Path | str,
    region: str,
    *,
    per_class_cap: int = DEFAULT_PER_CLASS_CAP,
    year: int = ALPHAEARTH_YEAR,
    seed: int = 42,
    batch_size: int = 500,
    cache_dir: Path | None = None,
    project: str = "agrosat-copilot",
) -> pl.DataFrame:
    """Sample the 64-dim AlphaEarth embedding at one country's parcel centers.

    Builds the stratified subset, queries AlphaEarth via
    :func:`ml.ingest.gee_sampler.sample_alphaearth_at_coords` (which itself caches
    per ``(cache_key, year, n_points)``), joins the labels back and writes a
    country-level parquet ``eurocropsml_alphaearth_<region>.parquet``.

    Args:
        root: EuroCropsML root directory.
        region: Country key.
        per_class_cap: Max parcels per macro class for the stratified subset.
        year: AlphaEarth annual mosaic year (default 2021).
        seed: Stratified-subset shuffle seed.
        batch_size: GEE ``reduceRegions`` batch size.
        cache_dir: Output directory (default ``data/transfer``).
        project: GCP project for ``ee.Initialize``.

    Returns:
        A frame ``px_id, parcel_id, lon, lat, hcat_code, macro_hcat_group,
        dim_00..dim_63`` for the parcels GEE returned an embedding for.
    """
    cache_dir = cache_dir or _CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"eurocropsml_alphaearth_{region}.parquet"
    if out_path.exists():
        logger.info("alphaearth_country_cache_hit", region=region, path=str(out_path))
        return pl.read_parquet(out_path)

    index = load_parcel_index(root, region)
    subset_labels = _stratified_subset(index, per_class_cap, seed)
    subset = read_centers_for(root, subset_labels)
    logger.info(
        "alphaearth_subset_selected",
        region=region,
        n_total=index.height,
        n_subset_labels=subset_labels.height,
        n_subset_with_center=subset.height,
        per_class_cap=per_class_cap,
    )

    init_ee(project=project)
    coords = subset.select("px_id", "lon", "lat")
    embeddings = sample_alphaearth_at_coords(
        coords,
        year=year,
        cache_path=_REPO_ROOT / "data" / "cache" / "gee",
        cache_key=f"eurocropsml_{region}",
        batch_size=batch_size,
    )
    if embeddings.is_empty():
        logger.warning("alphaearth_country_empty", region=region)
        return embeddings

    # Drop rows where every embedding dim is null (point fell outside the mosaic).
    embeddings = embeddings.filter(
        pl.any_horizontal(pl.col(c).is_not_null() for c in ALPHAEARTH_DIM_COLS)
    )
    joined = subset.join(
        embeddings.select(["px_id", *ALPHAEARTH_DIM_COLS]),
        on="px_id",
        how="inner",
    )
    joined.write_parquet(out_path)
    logger.info(
        "alphaearth_country_saved",
        region=region,
        n_requested=subset.height,
        n_with_embedding=joined.height,
        path=str(out_path),
    )
    return joined


# ---------------------------------------------------------------------------
# k-shot splits over the AlphaEarth vectors.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FewShotMatrix:
    """AlphaEarth feature matrices + macro labels for one few-shot scenario."""

    x_source: np.ndarray
    y_source: np.ndarray
    x_target_train: np.ndarray
    y_target_train: np.ndarray
    x_target_test: np.ndarray
    y_target_test: np.ndarray


def _frame_to_xy(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Split an AlphaEarth frame into ``(X[64], y_macro)``."""
    x = frame.select(ALPHAEARTH_DIM_COLS).to_numpy().astype(np.float64)
    y = frame.get_column("macro_hcat_group").to_numpy()
    return x, np.asarray(y)


def _sample_k_shot(rng: np.random.Generator, labels: np.ndarray, k: int) -> np.ndarray:
    """Pick up to ``k`` support indices per class."""
    selected: list[int] = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        selected.extend(idx[:k].tolist())
    return np.array(sorted(selected), dtype=np.int64)


def _stratified_holdout(
    rng: np.random.Generator, labels: np.ndarray, test_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class held-out test set + remaining support pool (mirrors US-076)."""
    import math

    test: list[int] = []
    pool: list[int] = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        n_test = math.floor(len(idx) * test_fraction)
        if len(idx) >= 2:
            n_test = max(1, min(n_test, len(idx) - 1))
        test.extend(idx[:n_test].tolist())
        pool.extend(idx[n_test:].tolist())
    return np.array(sorted(test), dtype=np.int64), np.array(sorted(pool), dtype=np.int64)


def build_alphaearth_fewshot_splits(
    source_frame: pl.DataFrame,
    target_frame: pl.DataFrame,
    *,
    k: int,
    seed: int,
    test_fraction: float = 0.3,
) -> FewShotMatrix:
    """Assemble one ``(LV -> EE, k, seed)`` AlphaEarth few-shot scenario.

    Args:
        source_frame: Source-country AlphaEarth frame (LV).
        target_frame: Target-country AlphaEarth frame (EE).
        k: Shots per class drawn from the target support pool.
        seed: Deterministic seed for the support/test partition.
        test_fraction: Fraction of target parcels held out as the query set.

    Returns:
        A :class:`FewShotMatrix` ready for :func:`train_xgb_kshot_alphaearth`.
    """
    rng = np.random.default_rng(seed)
    x_src, y_src = _frame_to_xy(source_frame)
    x_tgt, y_tgt = _frame_to_xy(target_frame)

    test_idx, pool_idx = _stratified_holdout(rng, y_tgt, test_fraction)
    support_local = _sample_k_shot(rng, y_tgt[pool_idx], k)
    support_idx = pool_idx[support_local]

    return FewShotMatrix(
        x_source=x_src,
        y_source=y_src,
        x_target_train=x_tgt[support_idx],
        y_target_train=y_tgt[support_idx],
        x_target_test=x_tgt[test_idx],
        y_target_test=y_tgt[test_idx],
    )


def train_xgb_kshot_alphaearth(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    pretrain_x: np.ndarray | None = None,
    pretrain_y: np.ndarray | None = None,
) -> dict[str, float]:
    """Train the champion XGBoost recipe on a k-shot AlphaEarth set, score F1-macro.

    Identical recipe to :func:`ml.transfer.eurocropsml_fewshot.train_xgb_kshot`
    (same ``SpatialXGBClassifier`` via :func:`ml.train.baseline.build_estimator`,
    same ``multi:softprob`` hyperparameters); the ONLY difference is the input
    vector is the 64-dim AlphaEarth embedding rather than the S2-derived vector.

    Args:
        x_train: k-shot target support matrix ``(n_support, 64)``.
        y_train: k-shot target support labels.
        x_test: Target query matrix.
        y_test: Target query labels.
        pretrain_x: Optional source (LV) pre-train matrix folded ahead of support.
        pretrain_y: Optional source pre-train labels.

    Returns:
        Dict with ``f1_macro``, ``n_train`` and ``n_classes``.

    Raises:
        ValueError: if the support set is empty.
    """
    if x_train.shape[0] == 0:
        raise ValueError("Empty k-shot support set; cannot train.")

    if pretrain_x is not None and pretrain_x.shape[0] > 0:
        x_fit = np.vstack([pretrain_x, x_train])
        y_fit = np.concatenate([np.asarray(pretrain_y), np.asarray(y_train)])
    else:
        x_fit = x_train
        y_fit = np.asarray(y_train)

    label_to_id = {lab: i for i, lab in enumerate(sorted(set(y_fit.tolist())))}
    y_fit_enc = np.array([label_to_id[v] for v in y_fit], dtype=np.int64)

    params = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "objective": "multi:softprob",
        "random_state": 42,
    }
    estimator = build_estimator("xgb", dict(params))
    estimator.fit(x_fit, y_fit_enc)

    id_to_label = {i: lab for lab, i in label_to_id.items()}
    pred_enc = estimator.predict(x_test)
    y_pred = np.array([id_to_label.get(int(p), NULL_CLASS) for p in pred_enc])

    labels = sorted(set(np.asarray(y_test).tolist()) | set(y_pred.tolist()))
    f1 = float(f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0))
    return {
        "f1_macro": f1,
        "n_train": float(x_fit.shape[0]),
        "n_classes": float(len(set(np.asarray(y_test).tolist()))),
    }


def run_alphaearth_fewshot_curve(
    source_frame: pl.DataFrame,
    target_frame: pl.DataFrame,
    *,
    source_label: str = "LV",
    target_label: str = "EE",
    k_shots: Iterable[int] = K_SHOTS_EXTENDED,
    seeds: Sequence[int] = (0, 1, 2),
    use_pretrain: bool = True,
) -> pl.DataFrame:
    """Compute the real F1-macro-vs-k curve over AlphaEarth for ``LV -> EE``.

    Args:
        source_frame: Source (LV) AlphaEarth frame.
        target_frame: Target (EE) AlphaEarth frame.
        source_label: Source label for the output rows.
        target_label: Target label for the output rows.
        k_shots: Extended k ladder.
        seeds: Seeds for error bars.
        use_pretrain: When ``True`` the XGB is trained on LV pool + k EE shots.

    Returns:
        Long frame ``source, target, k, seed, f1_macro, n_classes, use_pretrain,
        feature_space, scenario``.
    """
    rows: list[dict[str, object]] = []
    scenario = (
        f"{source_label}->{target_label}" if use_pretrain else f"sin-pretrain->{target_label}"
    )
    for k in k_shots:
        for seed in seeds:
            split = build_alphaearth_fewshot_splits(
                source_frame, target_frame, k=int(k), seed=int(seed)
            )
            metrics = train_xgb_kshot_alphaearth(
                split.x_target_train,
                split.y_target_train,
                split.x_target_test,
                split.y_target_test,
                pretrain_x=split.x_source if use_pretrain else None,
                pretrain_y=split.y_source if use_pretrain else None,
            )
            rows.append(
                {
                    "source": source_label,
                    "target": target_label,
                    "k": int(k),
                    "seed": int(seed),
                    "f1_macro": metrics["f1_macro"],
                    "n_classes": int(metrics["n_classes"]),
                    "use_pretrain": bool(use_pretrain),
                    "feature_space": "alphaearth",
                    "scenario": scenario,
                }
            )
            logger.info(
                "alphaearth_fewshot_point",
                scenario=scenario,
                k=int(k),
                seed=int(seed),
                f1_macro=round(metrics["f1_macro"], 4),
                n_train=int(metrics["n_train"]),
            )
    return pl.DataFrame(rows)


def summarize_curve(curve: pl.DataFrame) -> pl.DataFrame:
    """Aggregate a raw curve frame into per-``k`` mean/std F1-macro."""
    return (
        curve.group_by("scenario", "k")
        .agg(
            pl.col("f1_macro").mean().alias("f1_mean"),
            pl.col("f1_macro").std(ddof=0).fill_null(0.0).alias("f1_std"),
            pl.len().alias("n_seeds"),
        )
        .sort("scenario", "k")
    )
