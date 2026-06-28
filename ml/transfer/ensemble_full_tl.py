"""Full-ensemble transfer learning: champion members as a representation.

Arthur's question (2026-06-25): does the WHOLE champion ensemble
(``tsvit-pheno`` + ``utae`` + ``xgb-alphaearth`` + FarSLIP), not just the tabular
member, transfer to a new dataset (EuroCropsML Estonia)?

Methodology (Arthur's decision, the honest TL framing)
------------------------------------------------------
Zero-shot is the wrong test: the dense members were trained to predict the 18
PASTIS (France) classes, and Estonia has crops that vocabulary never saw (rye,
spring barley forms...). So a raw forward can only ever answer in French. The
right TL test, with the SAME resources for every contender, is:

    Use each champion member as a frozen FEATURE EXTRACTOR over the target
    parcels, then train ONE lightweight head on a few-shot budget of target
    labels. Compare, on the IDENTICAL target test set and the IDENTICAL k-shot
    budget:
        - baseline : xgb-alphaearth features only (AlphaEarth 64-dim),
        - ensemble : ALL members' features concatenated.
    If the full-ensemble representation beats the tabular one at the SAME k,
    that is evidence the dense temporal/phenology members carry transferable
    signal the annual embedding does not.

This never re-trains a champion member -- only the small target head is fit, which
is exactly what transfer learning permits.

The format bridge (honest about its limit)
------------------------------------------
TSViT/U-TAE are DENSE models: they expect a spatial patch ``(T, 10, H, W)`` of 10
PASTIS-R bands. EuroCropsML ships a per-parcel series ``(T, 13)`` -- a single
pixel, 13 bands, no spatial axis. We bridge by:
  1. mapping the 13 EuroCropsML S2 bands to the 10 PASTIS-R bands by name,
  2. tiling the single pixel into a small ``H x W`` patch so the dense forward
     runs, and
  3. subsampling/padding the series to the model's trained ``n_timesteps`` (37).
We then read the model's per-pixel feature/posterior at the patch centre. This is
a CONTROLLED degradation: a tiled pixel has no real texture, so the dense models
operate out of their training distribution. The result is therefore reported as
"champion members as per-parcel extractors", NOT as dense patch inference. The
honest bound is exactly what we want to measure: how much of the members' signal
survives the per-parcel reduction.

Compute
-------
CPU-light for xgb-alphaearth; GPU for the dense forwards. Runs on a laptop GPU
for a few-hundred-parcel pilot; scale the cap to push it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "EUROCROPS_TO_PASTIS_BAND_INDEX",
    "FullEnsembleTLResult",
    "parcel_series_to_patch",
    "run_full_ensemble_tl",
]

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_OUT_DIR: Path = _REPO_ROOT / "data" / "transfer" / "ensemble_full_tl"
_TRANSFER_DIR: Path = _REPO_ROOT / "data" / "transfer"
_PREPROCESS_DIR: Path = _TRANSFER_DIR / "eurocropsml" / "preprocess"
_HCAT3_CSV: Path = _REPO_ROOT / "data" / "reference" / "eurocrops_hcat3.csv"
#: 64-dim AlphaEarth annual embedding columns in the EuroCropsML parquet.
_ALPHAEARTH_COLS: tuple[str, ...] = tuple(f"dim_{i:02d}" for i in range(64))
#: Rare-tail guard, mirrors the multi-region / temporal modules.
_MIN_LEAF_SUPPORT: int = 50

#: EuroCropsML S2 band order (eurocropsml.acquisition.config.S2_BANDS), 13 bands.
_EUROCROPS_BANDS: tuple[str, ...] = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
)

#: PASTIS-R 10-band order the dense models were trained on (the standard PASTIS
#: S2 selection drops B01, B09, B10 -- the atmospheric/cirrus bands).
_PASTIS_BANDS: tuple[str, ...] = (
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
)

#: Index into the 13-band EuroCropsML series for each of the 10 PASTIS bands, in
#: PASTIS order. Built by name so the mapping is auditable, not magic numbers.
EUROCROPS_TO_PASTIS_BAND_INDEX: tuple[int, ...] = tuple(
    _EUROCROPS_BANDS.index(b) for b in _PASTIS_BANDS
)

#: Trained temporal length of the dense checkpoints (TSVIT_FULLM_CONFIG; U-TAE is
#: length-agnostic but we keep one length for a stackable batch).
_N_TIMESTEPS: int = 37

#: Small patch side for the tiled per-parcel forward. patch_size=8 means an 8x8
#: patch is the minimum that yields a single spatial token in TSViT; we use 8 to
#: keep the dense forward cheap (one token, one pixel of signal).
_PATCH_SIDE: int = 8

#: Reflectance scale of the raw DN.
_DN_SCALE: float = 1e4

_DEFAULT_SOURCE: str = "latvia"
_DEFAULT_TARGET: str = "estonia"
_RANDOM_STATE: int = 42


@dataclass
class FullEnsembleTLResult:
    """Few-shot transfer F1 of the full-ensemble representation vs the tabular one."""

    per_k: pl.DataFrame
    summary: dict[str, object] = field(default_factory=dict)


@dataclass
class _RegionParcels:
    """AlphaEarth annual + raw S2 series + leaf for one region's parcels."""

    annual: np.ndarray  # (n, 64)
    patches: list[np.ndarray]  # each (n_timesteps, 10, P, P)
    leaf: np.ndarray  # (n,)
    n_missing_npz: int


def _load_hcat_name_map() -> dict[int, str]:
    """Return ``{hcat_code: hcat_leaf_name}`` from the HCAT v3 reference CSV."""
    if not _HCAT3_CSV.is_file():
        raise FileNotFoundError(f"HCAT v3 reference missing at {_HCAT3_CSV}")
    h = pl.read_csv(_HCAT3_CSV, schema_overrides={"HCAT3_code": pl.Utf8, "HCAT3_name": pl.Utf8})
    return {int(c): str(n) for c, n in zip(h["HCAT3_code"], h["HCAT3_name"], strict=True)}


def _load_region_parcels(
    region: str,
    *,
    max_parcels: int,
    seed: int,
    stratify_keep: set[str] | None = None,
    per_class: int | None = None,
    patch_side: int = _PATCH_SIDE,
) -> _RegionParcels:
    """Load a region's AlphaEarth + raw S2 series + leaf, bridging series to patches.

    Reads the EuroCropsML AlphaEarth parquet (64-dim + ``npz_name`` + ``hcat_code``),
    resolves the HCAT leaf, applies the rare-tail support guard, caps the parcel
    count, then re-reads each parcel's raw ``(T, 13)`` npz series and bridges it to
    a dense ``(n_timesteps, 10, P, P)`` patch. Parcels whose npz is missing/corrupt
    are dropped and counted.

    This is the LOCAL-npz source (no Sentinel Hub, no paid quota): the per-parcel
    series ships with EuroCropsML, so the dense members can be fed real data for
    free. The patch is pixel-tiled (no spatial texture) -- but for the
    VOCABULARY-correction experiment that is fine: the texture run already showed
    the bottleneck is the label space, not the texture.

    Args:
        region: EuroCropsML region key (e.g. ``"estonia"``).
        max_parcels: Parcel cap applied before the npz pass (global random draw).
        seed: Sampling seed.
        stratify_keep: When set with ``per_class``, take a per-class (stratified)
            sample restricted to this label-space instead of the global random cap.
        per_class: Parcels per leaf for the stratified sample.

    Returns:
        A :class:`_RegionParcels`.

    Raises:
        FileNotFoundError: if the region parquet or preprocess dir is missing.
    """
    parquet = _TRANSFER_DIR / f"eurocropsml_alphaearth_{region}.parquet"
    if not parquet.is_file():
        raise FileNotFoundError(f"EuroCropsML parquet missing at {parquet}")
    if not _PREPROCESS_DIR.is_dir():
        raise FileNotFoundError(f"EuroCropsML preprocess dir missing at {_PREPROCESS_DIR}")

    name_map = _load_hcat_name_map()
    df = pl.read_parquet(
        parquet, columns=["npz_name", "hcat_code", *_ALPHAEARTH_COLS]
    ).with_columns(
        pl.col("hcat_code")
        .cast(pl.Int64)
        .replace_strict(name_map, default="unknown_hcat", return_dtype=pl.Utf8)
        .alias("leaf")
    )
    counts = df.group_by("leaf").len().filter(pl.col("len") >= _MIN_LEAF_SUPPORT)
    df = df.filter(pl.col("leaf").is_in(counts["leaf"].to_list()))
    if stratify_keep is not None and per_class is not None:
        from ml.transfer.finetune_baltico import stratified_parcel_sample

        picked = stratified_parcel_sample(
            df["leaf"].to_list(), keep=stratify_keep, per_class=per_class, seed=seed
        )
        df = df[picked]
    elif df.height > max_parcels:
        df = df.sample(n=max_parcels, seed=seed, shuffle=True)

    npz_names = df["npz_name"].to_list()
    leaves = df["leaf"].to_list()
    annual_mat = df.select(_ALPHAEARTH_COLS).to_numpy().astype(np.float64)

    annual_rows: list[np.ndarray] = []
    patch_rows: list[np.ndarray] = []
    leaf_rows: list[str] = []
    n_missing = 0
    for i, npz_name in enumerate(npz_names):
        path = _PREPROCESS_DIR / npz_name
        if not path.is_file():
            n_missing += 1
            continue
        try:
            payload = np.load(path)  # allow_pickle=False: data/dates son arrays planos
            data = payload["data"]
            dates = payload["dates"]
        except Exception:  # noqa: BLE001 -- corrupt npz dropped, counted
            n_missing += 1
            continue
        if data.ndim != 2 or data.shape[0] < 2:
            n_missing += 1
            continue
        patch_rows.append(parcel_series_to_patch(data, dates, patch_side=patch_side))
        annual_rows.append(annual_mat[i])
        leaf_rows.append(leaves[i])

    logger.info(
        "region_parcels_loaded",
        region=region,
        n_parcels=len(patch_rows),
        n_missing_npz=n_missing,
    )
    return _RegionParcels(
        annual=np.asarray(annual_rows, dtype=np.float64),
        patches=patch_rows,
        leaf=np.asarray(leaf_rows),
        n_missing_npz=n_missing,
    )


def _subsample_time(series: np.ndarray, dates: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Equispaced subsample/pad a ``(T, B)`` series to exactly ``n`` timesteps.

    Args:
        series: Per-parcel series ``(T, B)`` in raw DN.
        dates: ``datetime64`` array ``(T,)``.
        n: Target timestep count.

    Returns:
        ``(series_n, dates_n)`` each with ``n`` rows (last frame repeated if
        ``T < n``; equispaced indices if ``T > n``).
    """
    t = series.shape[0]
    if t == n:
        return series, dates
    if t > n:
        idx = np.linspace(0, t - 1, n).round().astype(int)
        return series[idx], dates[idx]
    pad = np.repeat(series[-1:], n - t, axis=0)
    pad_d = np.repeat(dates[-1:], n - t, axis=0)
    return np.concatenate([series, pad], axis=0), np.concatenate([dates, pad_d], axis=0)


def parcel_series_to_patch(
    series: np.ndarray, dates: np.ndarray, *, patch_side: int = _PATCH_SIDE
) -> np.ndarray:
    """Bridge a per-parcel ``(T, 13)`` series to a dense patch ``(T, 10, P, P)``.

    Maps the 13 EuroCropsML bands to the 10 PASTIS bands by name, scales DN to
    reflectance, subsamples the series to the trained ``n_timesteps``, and tiles
    the single pixel into a ``P x P`` patch so the dense forward runs.

    Args:
        series: EuroCropsML parcel series ``(T, 13)`` in raw DN.
        dates: ``datetime64`` acquisition dates ``(T,)``.
        patch_side: Spatial side ``P`` of the tiled patch. Default 8 (cheap, for
            U-TAE which is grid-agnostic); pass 128 for TSViT-fullm, whose trained
            ``spatial_pos_embedding`` expects a ``128/patch_size`` token grid.

    Returns:
        A ``(n_timesteps, 10, P, P)`` float32 reflectance patch.
    """
    s_n, _d = _subsample_time(series, dates, _N_TIMESTEPS)
    refl = s_n.astype(np.float32) / _DN_SCALE
    sel = refl[:, list(EUROCROPS_TO_PASTIS_BAND_INDEX)]  # (T, 10)
    patch = np.broadcast_to(
        sel[:, :, None, None], (sel.shape[0], sel.shape[1], patch_side, patch_side)
    )
    return np.ascontiguousarray(patch, dtype=np.float32)


def _extract_dense_features(
    patches: list[np.ndarray], model_kinds: tuple[str, ...], *, device: str = "auto"
) -> dict[str, np.ndarray]:
    """Run each dense member over the tiled patches and read the centre posterior.

    For each model kind, loads the checkpoint (cached), forwards every patch, and
    reads the softmax posterior at the patch centre pixel -- a per-parcel feature
    vector of length ``native_num_classes``.

    Args:
        patches: List of ``(T, 10, P, P)`` patches, one per parcel.
        model_kinds: Registry kinds to run (e.g. ``("tsvit-pheno", "utae")``).
        device: Inference device.

    Returns:
        Mapping ``kind -> (n_parcels, n_classes)`` centre-pixel posterior matrix.
    """
    import torch

    from ml.eval.checkpoint_registry import CHECKPOINT_REGISTRY
    from ml.eval.segmentation_inference import (
        _forward_logits,
        load_checkpoint_model,
    )

    out: dict[str, np.ndarray] = {}
    centre = _PATCH_SIDE // 2
    for kind in model_kinds:
        spec = CHECKPOINT_REGISTRY[kind]
        model = load_checkpoint_model(spec, n_timesteps=_N_TIMESTEPS, device=device)
        rows: list[np.ndarray] = []
        with torch.no_grad():
            for patch in patches:
                x = torch.from_numpy(patch)  # (T, 10, P, P)
                logits = _forward_logits(model, x, model_kind=kind)  # (1, K, P, P)
                post = torch.softmax(logits, dim=1)[0, :, centre, centre]
                rows.append(post.float().cpu().numpy())
        out[kind] = np.asarray(rows, dtype=np.float64)
        logger.info(
            "dense_member_extracted",
            kind=kind,
            n_parcels=len(rows),
            n_classes=out[kind].shape[1] if rows else 0,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def _fewshot_curve(
    feats_src: np.ndarray,
    y_src: np.ndarray,
    feats_tgt: np.ndarray,
    y_tgt: np.ndarray,
    *,
    ks: tuple[int, ...],
    n_classes: int,
    seeds: tuple[int, ...],
    label: str,
) -> list[dict[str, object]]:
    """Few-shot curve: train an XGBoost head on k source shots/class, score target.

    For each ``k`` and ``seed``, samples up to ``k`` source parcels per class,
    fits the champion XGBoost head on that budget, and reports the target macro F1.
    The SAME (k, seed) sampling is reused across feature sets by fixing the seed.

    Args:
        feats_src: Source features ``(n_src, d)``.
        y_src: Source labels.
        feats_tgt: Target features ``(n_tgt, d)``.
        y_tgt: Target labels.
        ks: Shots-per-class budgets.
        n_classes: Shared label-space size.
        seeds: RNG seeds to average over.
        label: Feature-set name for the rows ("baseline" | "ensemble").

    Returns:
        List of ``{feature_set, k, seed, macro_f1}`` dicts.
    """
    from sklearn.metrics import f1_score

    from ml.train.baseline import _XGB_BASE_PARAMS, build_estimator

    rows: list[dict[str, object]] = []
    for k in ks:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            pick: list[int] = []
            for cls in range(n_classes):
                idx = np.where(y_src == cls)[0]
                if idx.size == 0:
                    continue
                take = min(k, idx.size)
                pick.extend(rng.choice(idx, size=take, replace=False).tolist())
            if not pick:
                continue
            params = dict(_XGB_BASE_PARAMS)
            params["random_state"] = seed
            model = build_estimator("xgb", params)
            model.fit(feats_src[pick], y_src[pick])
            pred = model.predict(feats_tgt)
            macro = float(f1_score(y_tgt, pred, average="macro"))
            rows.append(
                {"feature_set": label, "k": int(k), "seed": int(seed), "macro_f1": round(macro, 4)}
            )
    return rows


def run_full_ensemble_tl(
    *,
    source: str = _DEFAULT_SOURCE,
    target: str = _DEFAULT_TARGET,
    max_parcels_per_region: int = 600,
    dense_members: tuple[str, ...] = ("tsvit-pheno-fullm", "utae"),
    ks: tuple[int, ...] = (1, 5, 10, 20),
    seeds: tuple[int, ...] = (0, 1, 2),
    device: str = "auto",
    seed: int = _RANDOM_STATE,
) -> FullEnsembleTLResult:
    """Few-shot transfer: full-ensemble representation vs xgb-alphaearth alone.

    Builds, for the SAME source+target parcels, the AlphaEarth annual embedding
    and the dense members' per-parcel centre posteriors, then compares two
    representations on the IDENTICAL few-shot budget and target test:
        - baseline : AlphaEarth 64-dim only,
        - ensemble : AlphaEarth ++ each dense member's posterior.

    Args:
        source: Source region key (trained on).
        target: Target region key (tested on).
        max_parcels_per_region: Per-region parcel cap (the pilot is small).
        dense_members: Champion dense members to add as extractors.
        ks: Shots-per-class budgets for the few-shot curve.
        seeds: Seeds to average the curve over.
        device: Inference device for the dense forwards.
        seed: Seed for the aligned-dataset subsample.

    Returns:
        A :class:`FullEnsembleTLResult`.

    Raises:
        ValueError: if the regions share no leaf class.
    """
    reg_src = _load_region_parcels(source, max_parcels=max_parcels_per_region, seed=seed)
    reg_tgt = _load_region_parcels(target, max_parcels=max_parcels_per_region, seed=seed)

    shared = sorted(set(reg_src.leaf.tolist()) & set(reg_tgt.leaf.tolist()))
    if not shared:
        raise ValueError(f"source={source!r} and target={target!r} share no leaf.")
    cls_id = {c: i for i, c in enumerate(shared)}
    keep = set(shared)
    n_classes = len(shared)

    def _prep(reg: _RegionParcels) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        mask = np.array([leaf in keep for leaf in reg.leaf], dtype=bool)
        annual = reg.annual[mask]
        y = np.array([cls_id[c] for c in reg.leaf[mask]], dtype=np.int64)
        patches = [p for p, m in zip(reg.patches, mask, strict=True) if m]
        return annual, y, patches

    a_src, y_src, p_src = _prep(reg_src)
    a_tgt, y_tgt, p_tgt = _prep(reg_tgt)

    dense_src = _extract_dense_features(p_src, dense_members, device=device)
    dense_tgt = _extract_dense_features(p_tgt, dense_members, device=device)

    ens_src = np.concatenate([a_src, *[dense_src[m] for m in dense_members]], axis=1)
    ens_tgt = np.concatenate([a_tgt, *[dense_tgt[m] for m in dense_members]], axis=1)

    rows = _fewshot_curve(
        a_src, y_src, a_tgt, y_tgt, ks=ks, n_classes=n_classes, seeds=seeds, label="baseline"
    ) + _fewshot_curve(
        ens_src, y_src, ens_tgt, y_tgt, ks=ks, n_classes=n_classes, seeds=seeds, label="ensemble"
    )
    per_k = pl.DataFrame(rows)
    agg = (
        per_k.group_by(["feature_set", "k"])
        .agg(
            pl.col("macro_f1").mean().alias("f1_mean"),
            pl.col("macro_f1").std(ddof=0).fill_null(0.0).alias("f1_std"),
        )
        .sort(["k", "feature_set"])
    )

    # Per-k delta ensemble - baseline.
    deltas: dict[int, float] = {}
    for k in ks:
        base = agg.filter((pl.col("feature_set") == "baseline") & (pl.col("k") == k))
        ens = agg.filter((pl.col("feature_set") == "ensemble") & (pl.col("k") == k))
        if base.height and ens.height:
            deltas[int(k)] = round(float(ens["f1_mean"][0] - base["f1_mean"][0]), 4)

    summary: dict[str, object] = {
        "source": source,
        "target": target,
        "n_source_parcels": int(a_src.shape[0]),
        "n_target_parcels": int(a_tgt.shape[0]),
        "n_shared_leaves": n_classes,
        "dense_members": list(dense_members),
        "baseline_dim": int(a_src.shape[1]),
        "ensemble_dim": int(ens_src.shape[1]),
        "ks": list(ks),
        "seeds": list(seeds),
        "curve": agg.to_dicts(),
        "delta_ensemble_minus_baseline_by_k": deltas,
        "note": (
            "Dense members run as per-parcel extractors over a tiled single pixel "
            "(no real texture); honest lower bound on their transferable signal."
        ),
    }
    logger.info(
        "full_ensemble_tl_done",
        source=source,
        target=target,
        n_shared_leaves=n_classes,
        deltas=deltas,
    )
    return FullEnsembleTLResult(per_k=per_k, summary=summary)


def save_outputs(result: FullEnsembleTLResult, out_dir: Path = _OUT_DIR) -> None:
    """Persist the few-shot curve and JSON summary to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result.per_k.write_parquet(out_dir / "ensemble_full_tl_per_k.parquet")
    (out_dir / "ensemble_full_tl_summary.json").write_text(
        json.dumps(result.summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("full_ensemble_tl_saved", out_dir=str(out_dir))
