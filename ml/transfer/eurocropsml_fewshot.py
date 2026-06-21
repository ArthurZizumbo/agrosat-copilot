"""EuroCropsML transnational few-shot transfer curve (US-076).

Reproduces the *pre-coded* k-shot transfer protocol of EuroCropsML (paper
arXiv 2407.17458, Table II: ``LV+PT -> EE`` and ``LV -> EE``) reusing the
tabular **recipe** of the AlphaEarth baseline (XGBoost ``multi:softprob`` over a
fixed per-parcel feature vector, :mod:`ml.train.baseline`) to produce a real
F1-macro-vs-k curve that quantifies how many local samples of the target country
are needed to close the domain gap.

Honest decisions (see ``docs/us-planning/us-076.md`` sections 1.1 and 3.2):

- **France is NOT in EuroCropsML.** The dataset covers Estonia (EE), Latvia (LV)
  and Portugal (PT); the protocol is ``LV[+PT] -> EE``, not ``France -> Estonia``.
- **EuroCropsML carries NO AlphaEarth embeddings.** A datapoint is an annual
  Sentinel-2 L1C median time series (13 bands, B10 excluded), stored as one
  ``.npz`` per parcel named ``<region>_<parcel_id>_<EC_hcat_c>.npz`` with arrays
  ``data`` (T, n_bands), ``dates`` and ``center``. We therefore reuse the XGBoost
  *recipe* (a fixed vector per parcel) but the vector is derived from that S2
  series (:func:`parcel_feature_vector`), NOT from AlphaEarth. The
  AlphaEarth-via-GEE variant stays FUTURE.
- **Labels are aligned to the US-074 HCAT macro space** via
  :mod:`ml.transfer.label_align` (the ``EC_hcat_c`` code IS an HCAT leaf code),
  so the curve is reported on the same ``hcat-macro`` label-space as the rest of
  EPIC 12.

The curve is **evidence of the measured domain gap**, never a zero-shot accuracy
claim, and is computed strictly over the REAL EuroCropsML splits. If the data is
not present the public functions raise an explicit ``EuroCropsMLDataMissing``
error -- they never fabricate numbers.
"""

from __future__ import annotations

import math
import re
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import structlog
from sklearn.metrics import f1_score

from ml.train.baseline import build_estimator
from ml.transfer.label_align import NULL_CLASS, align_codes_to_hcat_macro

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_REGIONS",
    "K_SHOTS",
    "ZENODO_VERSIONS_URL",
    "EuroCropsMLDataMissing",
    "ParcelSample",
    "align_labels_to_hcat_macro",
    "build_fewshot_splits",
    "download_eurocropsml",
    "load_region_samples",
    "parcel_feature_vector",
    "run_fewshot_curve",
    "train_xgb_kshot",
]

#: k-shot ladder of the EuroCropsML protocol (paper arXiv 2407.17458, Table II).
K_SHOTS: tuple[int, ...] = (1, 5, 10, 20, 100, 200, 500)

#: Minimal viable subset: target Estonia + one source Latvia (Plan B, section 8).
DEFAULT_REGIONS: tuple[str, ...] = ("estonia", "latvia")

#: Zenodo versions endpoint for the EuroCropsML concept record (10629610 ->
#: concept DOI 10.5281/zenodo.10629609). ``download_eurocropsml`` resolves the
#: effective version deterministically against this URL (no interactive prompt).
ZENODO_VERSIONS_URL: str = "https://zenodo.org/api/records/10629610/versions"

#: Map of two-letter region codes used in the ``.npz`` filenames.
_REGION_CODES: dict[str, str] = {
    "estonia": "EE",
    "latvia": "LV",
    "portugal": "PT",
}

#: ``.npz`` filename pattern: ``<NUTS3>_<parcel_id>_<EC_hcat_c>.npz``. The region
#: prefix is the NUTS3 code (e.g. ``EE001``), so we match on the leading country
#: letters. ``EC_hcat_c`` is the trailing integer before the extension.
_NPZ_NAME_RE = re.compile(r"^(?P<region>[A-Z]{2}[A-Z0-9]*)_(?P<parcel>\d+)_(?P<hcat>\d+)$")

#: Per-band percentiles summarising the temporal distribution of each S2 band.
_PERCENTILES: tuple[int, ...] = (10, 25, 50, 75, 90)


class EuroCropsMLDataMissing(FileNotFoundError):
    """Raised when the EuroCropsML subset is absent on disk.

    The public pipeline never fabricates a curve: when the real ``.npz`` parcels
    are not present it raises this so callers (notebook in ``degraded`` mode,
    tests) can branch on a clear, typed condition instead of crashing opaquely.
    """


@dataclass(frozen=True)
class ParcelSample:
    """A single EuroCropsML parcel: its S2 series, centroid and HCAT label.

    Attributes:
        region: Country key (``"estonia"``/``"latvia"``/``"portugal"``).
        parcel_id: EuroCropsML parcel identifier.
        hcat_code: Raw 10-digit ``EC_hcat_c`` HCAT leaf code.
        series: Sentinel-2 series of shape ``(T, n_bands)`` (annual L1C medians).
        center: Parcel centroid ``(x, y)`` in the dataset CRS.
    """

    region: str
    parcel_id: int
    hcat_code: int
    series: np.ndarray
    center: tuple[float, float]


# ---------------------------------------------------------------------------
# Download (non-interactive, idempotent).
# ---------------------------------------------------------------------------


def download_eurocropsml(
    root: Path | str,
    regions: Sequence[str] = DEFAULT_REGIONS,
    *,
    files: Sequence[str] = ("split.zip", "preprocess.zip"),
    min_publication_date: str = "2024-03-14",
    timeout: int = 120,
) -> Path:
    """Download the EuroCropsML subset from Zenodo non-interactively and idempotently.

    Wraps the Zenodo REST API the same way
    :func:`eurocropsml.dataset.download.download_dataset` does, but WITHOUT the
    interactive ``select_version`` / ``get_user_choice`` prompts (which would
    deadlock papermill/CI): the latest valid version (publication date >=
    ``min_publication_date``, excluding the package's blocklisted corrupt
    versions) is selected deterministically and only the requested ``files`` are
    streamed to disk and unzipped. Re-running is a no-op when the unzipped
    payload already exists.

    The ``regions`` argument documents the intended subset (EE + LV by default,
    the minimal viable Plan B); the actual per-region filtering happens at read
    time in :func:`load_region_samples`, because the Zenodo ``preprocess.zip``
    bundles all three countries.

    Args:
        root: Destination directory for the EuroCropsML data
            (``data/transfer/eurocropsml``). Created if absent.
        regions: Country keys the subset is intended for (documentation only;
            the zip bundles all countries).
        files: Zenodo file names to download. ``raw_data.zip`` (~3.3 GB) is
            intentionally omitted; ``preprocess.zip`` (~1.5 GB) holds the
            per-parcel ``.npz`` series and ``split.zip`` the pre-coded k-shot
            splits.
        min_publication_date: Earliest acceptable Zenodo publication date
            (the package discards older, incorrect versions).
        timeout: Per-request timeout in seconds for the API metadata call.

    Returns:
        The ``root`` directory, now containing the unzipped subset.

    Raises:
        RuntimeError: if no valid Zenodo version is found.
        requests.HTTPError: if the Zenodo metadata request fails.
    """
    import requests

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    preprocess_dir = root / "preprocess"
    split_dir = root / "split"
    if preprocess_dir.exists() and any(preprocess_dir.rglob("*.npz")):
        logger.info(
            "eurocropsml_download_skip",
            reason="preprocess .npz already present",
            root=str(root),
        )
        return root

    logger.info("eurocropsml_zenodo_query", url=ZENODO_VERSIONS_URL)
    resp = requests.get(ZENODO_VERSIONS_URL, timeout=timeout)
    resp.raise_for_status()
    versions = resp.json()["hits"]["hits"]
    # Same blocklist as eurocropsml.dataset.download.select_version: drop the
    # pre-2024-03-14 incorrect versions and the known-corrupt 2025 ones.
    valid = [
        v
        for v in versions
        if v["metadata"]["publication_date"] >= min_publication_date
        and v["metadata"]["publication_date"] not in ("2025-02-06", "2025-03-18")
    ]
    if not valid:
        raise RuntimeError(
            "No valid EuroCropsML version found on Zenodo "
            f"(>= {min_publication_date}); see docs/blockers/epic12-vm-setup.md."
        )
    valid.sort(key=lambda v: v["metadata"]["publication_date"])
    selected = valid[-1]
    version_id = selected["metadata"]["relations"]["version"][0]["index"] + 1
    logger.info(
        "eurocropsml_version_selected",
        version=version_id,
        doi=selected["links"]["doi"],
        publication_date=selected["metadata"]["publication_date"],
    )

    wanted = set(files)
    for entry in selected["files"]:
        key = entry["key"]
        if key not in wanted:
            continue
        local_zip = root / key
        _stream_download(entry["links"]["self"], local_zip)
        _unzip(local_zip, root)

    if not (split_dir.exists() or preprocess_dir.exists()):
        raise RuntimeError(
            f"EuroCropsML download produced no usable payload under {root}; "
            "see docs/blockers/epic12-vm-setup.md."
        )
    logger.info("eurocropsml_download_done", root=str(root), regions=tuple(regions))
    return root


def _stream_download(url: str, local_path: Path, *, chunk_size: int = 1 << 20) -> None:
    """Stream a remote file to disk in chunks (idempotent on the zip name).

    Args:
        url: Remote file URL.
        local_path: Local destination path.
        chunk_size: Streaming chunk size in bytes (default 1 MiB).
    """
    import requests

    if local_path.exists():
        logger.info("eurocropsml_zip_exists", path=str(local_path))
        return
    tmp_path = local_path.with_suffix(local_path.suffix + ".part")
    logger.info("eurocropsml_stream_start", url=url, dest=str(local_path))
    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with tmp_path.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
    tmp_path.replace(local_path)
    logger.info("eurocropsml_stream_done", dest=str(local_path), bytes=local_path.stat().st_size)


def _unzip(zip_path: Path, extract_to: Path, *, delete_zip: bool = True) -> None:
    """Extract a zip archive and optionally delete it afterwards.

    Args:
        zip_path: Path to the ``.zip`` archive.
        extract_to: Directory to extract into.
        delete_zip: When ``True`` (default) removes the archive after extraction
            to save disk (the ~1.5 GB zip is redundant once unzipped).
    """
    import zipfile

    logger.info("eurocropsml_unzip", zip=str(zip_path))
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    if delete_zip:
        zip_path.unlink()


# ---------------------------------------------------------------------------
# Reading the real per-parcel .npz series.
# ---------------------------------------------------------------------------


def _find_preprocess_dir(root: Path) -> Path:
    """Locate the directory holding the per-parcel ``.npz`` files.

    The Zenodo ``preprocess.zip`` unzips to ``<root>/preprocess/S2/<year>/`` but
    older layouts differ; this searches for the first directory containing
    ``.npz`` files.

    Args:
        root: EuroCropsML root directory.

    Returns:
        The directory containing the ``.npz`` parcels.

    Raises:
        EuroCropsMLDataMissing: if no ``.npz`` directory exists under ``root``.
    """
    candidates = [
        root / "preprocess" / "S2",
        root / "preprocess",
        root,
    ]
    for base in candidates:
        if base.exists():
            for npz in base.rglob("*.npz"):
                return npz.parent
    raise EuroCropsMLDataMissing(
        f"No EuroCropsML .npz parcels found under {root}. Run "
        "download_eurocropsml(...) first or see docs/blockers/epic12-vm-setup.md."
    )


def load_region_samples(
    root: Path | str,
    region: str,
    *,
    max_parcels: int | None = None,
) -> list[ParcelSample]:
    """Load the real EuroCropsML parcels of one country from disk.

    Reads the per-parcel ``.npz`` files (arrays ``data``, ``dates``, ``center``)
    whose filename region prefix matches ``region``. The ``EC_hcat_c`` label is
    parsed from the filename. No synthetic data is ever produced: an absent
    subset raises :class:`EuroCropsMLDataMissing`.

    Args:
        root: EuroCropsML root directory.
        region: Country key (``"estonia"``/``"latvia"``/``"portugal"``).
        max_parcels: Optional cap on the number of parcels (for quick smoke runs).

    Returns:
        A list of :class:`ParcelSample`, one per matching parcel.

    Raises:
        ValueError: if ``region`` is unknown.
        EuroCropsMLDataMissing: if the subset is absent.
    """
    if region not in _REGION_CODES:
        raise ValueError(f"Unknown region {region!r}; expected one of {sorted(_REGION_CODES)}.")
    root = Path(root)
    code = _REGION_CODES[region]
    npz_dir = _find_preprocess_dir(root)

    samples: list[ParcelSample] = []
    for npz_path in sorted(npz_dir.rglob("*.npz")):
        match = _NPZ_NAME_RE.match(npz_path.stem)
        if match is None or not match.group("region").startswith(code):
            continue
        with np.load(npz_path) as payload:
            series = np.asarray(payload["data"], dtype=np.float64)
            center_arr = np.asarray(payload.get("center", np.array([np.nan, np.nan])))
        center = (float(center_arr.flat[0]), float(center_arr.flat[1]))
        samples.append(
            ParcelSample(
                region=region,
                parcel_id=int(match.group("parcel")),
                hcat_code=int(match.group("hcat")),
                series=series,
                center=center,
            )
        )
        if max_parcels is not None and len(samples) >= max_parcels:
            break
    logger.info("eurocropsml_region_loaded", region=region, n_parcels=len(samples))
    return samples


def featurize_region(
    root: Path | str,
    region: str,
    *,
    max_parcels: int | None = None,
    cache: bool = True,
) -> pl.DataFrame:
    """Featurize all parcels of one region into a cached Polars frame.

    Reads the region's ``.npz`` parcels once, reduces each S2 series to the fixed
    per-parcel vector (:func:`parcel_feature_vector`) and resolves the macro HCAT
    label, returning a frame with columns ``f0..f{d-1}``, ``macro_hcat_group`` and
    ``hcat_code``. The frame is cached to parquet keyed by ``(region, max_parcels)``
    so the k-shot curve (which iterates over many ``(k, seed)`` points) never
    re-reads the tens of thousands of ``.npz`` files.

    Args:
        root: EuroCropsML root directory.
        region: Country key.
        max_parcels: Optional per-region parcel cap (smoke runs); part of the
            cache key.
        cache: When ``True`` (default) reads/writes the parquet feature cache.

    Returns:
        A Polars frame of per-parcel feature vectors + macro labels.

    Raises:
        EuroCropsMLDataMissing: if the subset is absent.
    """
    root = Path(root)
    cache_path = _feature_cache_path(root, region, max_parcels)
    if cache and cache_path.exists():
        logger.info("eurocropsml_feature_cache_hit", region=region, path=str(cache_path))
        return pl.read_parquet(cache_path)

    samples = load_region_samples(root, region, max_parcels=max_parcels)
    vectors = np.vstack([parcel_feature_vector(s.series) for s in samples])
    macro = align_codes_to_hcat_macro([s.hcat_code for s in samples])
    feat_cols = {f"f{i}": vectors[:, i] for i in range(vectors.shape[1])}
    frame = pl.DataFrame(
        {
            **feat_cols,
            "macro_hcat_group": macro,
            "hcat_code": [s.hcat_code for s in samples],
        }
    )
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(cache_path)
        logger.info("eurocropsml_feature_cache_saved", region=region, path=str(cache_path))
    return frame


def _feature_cache_path(root: Path, region: str, max_parcels: int | None) -> Path:
    """Path of the per-region feature cache parquet.

    Args:
        root: EuroCropsML root directory.
        region: Country key.
        max_parcels: Per-region cap (part of the key; ``all`` when ``None``).

    Returns:
        The cache parquet path under ``<root>/_feature_cache``.
    """
    cap = "all" if max_parcels is None else str(max_parcels)
    return root / "_feature_cache" / f"{region}_{cap}.parquet"


def _frame_to_xy(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Split a feature frame into the ``(X, y_macro)`` arrays.

    Args:
        frame: Frame from :func:`featurize_region`.

    Returns:
        Tuple ``(X, y)`` with the float matrix and the macro-label vector.
    """
    feat_cols = [c for c in frame.columns if c.startswith("f") and c[1:].isdigit()]
    feat_cols = sorted(feat_cols, key=lambda c: int(c[1:]))
    x = frame.select(feat_cols).to_numpy().astype(np.float64)
    y = frame.get_column("macro_hcat_group").to_numpy()
    return x, np.asarray(y)


# ---------------------------------------------------------------------------
# Feature engineering: fixed per-parcel vector from the S2 series.
# ---------------------------------------------------------------------------


def parcel_feature_vector(series: np.ndarray) -> np.ndarray:
    """Reduce a parcel's S2 time series ``(T, n_bands)`` to a fixed vector.

    The XGBoost baseline recipe consumes a fixed-width per-parcel vector. As
    EuroCropsML provides no AlphaEarth embedding, we derive that vector from the
    parcel's own Sentinel-2 series: per band we stack the temporal mean, std,
    min, max, the percentiles in :data:`_PERCENTILES`, and a linear-trend slope.
    The output dimensionality depends ONLY on ``n_bands`` (not on ``T``), so the
    vector is comparable across parcels with different revisit counts.

    Args:
        series: S2 series of shape ``(T, n_bands)``. A single-row ``(n_bands,)``
            input is treated as ``T = 1``.

    Returns:
        A 1-D ``float64`` vector of length ``n_bands * (8 + len(_PERCENTILES))``.

    Raises:
        ValueError: if ``series`` is empty or not 1-D/2-D.
    """
    arr = np.asarray(series, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"`series` must be a non-empty (T, n_bands) array; got shape {arr.shape}.")
    # Non-finite revisits (clouds/padding) must not poison the statistics.
    arr = np.where(np.isfinite(arr), arr, np.nan)
    n_steps = arr.shape[0]

    def _nan_stat(fn, axis: int = 0) -> np.ndarray:  # type: ignore[no-untyped-def]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = fn(arr, axis=axis)
        return np.where(np.isfinite(out), out, 0.0)

    mean = _nan_stat(np.nanmean)
    std = _nan_stat(np.nanstd)
    vmin = _nan_stat(np.nanmin)
    vmax = _nan_stat(np.nanmax)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw_pcts = np.nanpercentile(arr, _PERCENTILES, axis=0)
    pcts = [np.where(np.isfinite(p), p, 0.0) for p in raw_pcts]
    rng = vmax - vmin
    last_minus_first = _nan_stat(lambda a, axis: np.nan_to_num(a[-1] - a[0]))

    # Per-band linear-trend slope over the temporal index (cheap phenology proxy).
    if n_steps >= 2:
        t = np.arange(n_steps, dtype=np.float64)
        t_centered = t - t.mean()
        denom = float((t_centered**2).sum()) or 1.0
        filled = np.where(np.isfinite(arr), arr, np.nanmean(arr, axis=0, keepdims=True))
        filled = np.where(np.isfinite(filled), filled, 0.0)
        slope = (t_centered[:, None] * (filled - filled.mean(axis=0, keepdims=True))).sum(
            axis=0
        ) / denom
    else:
        slope = np.zeros(arr.shape[1], dtype=np.float64)

    parts = [mean, std, vmin, vmax, rng, last_minus_first, slope, *pcts]
    vector = np.concatenate(parts).astype(np.float64)
    clean: np.ndarray = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    return clean


def align_labels_to_hcat_macro(hcat_codes: pl.Series) -> pl.Series:
    """Convenience re-export: ``EC_hcat_c`` series -> macro HCAT group series.

    Thin wrapper over :func:`ml.transfer.label_align.align_labels_to_hcat_macro`
    so the pipeline module exposes the label alignment as a single import.

    Args:
        hcat_codes: Series of raw ``EC_hcat_c`` codes.

    Returns:
        A ``pl.Utf8`` series of macro HCAT group names.
    """
    from ml.transfer.label_align import align_labels_to_hcat_macro as _align

    return _align(hcat_codes)


# ---------------------------------------------------------------------------
# Splits: k-shot sampling of the target country.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FewShotMatrix:
    """Feature matrices + macro labels for one few-shot transfer scenario.

    Attributes:
        x_source: Source-country feature matrix ``(n_source, d)`` (pre-train pool).
        y_source: Source-country macro labels ``(n_source,)``.
        x_target_train: k-shot target-country support matrix ``(n_support, d)``.
        y_target_train: k-shot target-country support labels.
        x_target_test: Held-out target-country query matrix.
        y_target_test: Held-out target-country query labels.
        classes: Sorted macro classes present in the test set.
    """

    x_source: np.ndarray
    y_source: np.ndarray
    x_target_train: np.ndarray
    y_target_train: np.ndarray
    x_target_test: np.ndarray
    y_target_test: np.ndarray
    classes: tuple[str, ...]


def _sample_k_shot(
    rng: np.random.Generator,
    labels: np.ndarray,
    k: int,
) -> np.ndarray:
    """Pick up to ``k`` support indices per class (the k-shot support set).

    Args:
        rng: Seeded random generator.
        labels: Per-sample class labels.
        k: Shots per class.

    Returns:
        A 1-D array of selected positional indices (the support set).
    """
    selected: list[int] = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        selected.extend(idx[:k].tolist())
    return np.array(sorted(selected), dtype=np.int64)


def build_fewshot_splits(
    root: Path | str,
    *,
    source: Sequence[str],
    target: str,
    k: int,
    seed: int,
    test_fraction: float = 0.3,
    max_parcels: int | None = None,
    drop_null_class: bool = True,
) -> FewShotMatrix:
    """Assemble a single ``(source -> target, k, seed)`` few-shot scenario.

    Loads the REAL parcels of the source and target countries, featurizes them,
    aligns labels to the HCAT macro space, splits the target into a held-out test
    query set and a k-shot support set, and returns the matrices ready for
    :func:`train_xgb_kshot`.

    Args:
        root: EuroCropsML root directory.
        source: One or more source country keys (the pre-train pool), e.g.
            ``["latvia"]`` or ``["latvia", "portugal"]``.
        target: Target country key (e.g. ``"estonia"``).
        k: Shots per class drawn from the target support pool.
        seed: Deterministic seed for the support/test partition.
        test_fraction: Fraction of target parcels held out as the test query set.
        max_parcels: Optional per-region parcel cap (smoke runs).
        drop_null_class: When ``True`` (default) drops parcels whose label is
            outside the PASTIS-18 crosswalk (``null-class``) from the metrics.

    Returns:
        A :class:`FewShotMatrix` for the scenario.

    Raises:
        EuroCropsMLDataMissing: if the subset is absent.
        ValueError: if the target has too few parcels to form a test set.
    """
    rng = np.random.default_rng(seed)

    src_frames = [featurize_region(root, reg, max_parcels=max_parcels) for reg in source]
    source_frame = pl.concat(src_frames) if len(src_frames) > 1 else src_frames[0]
    target_frame = featurize_region(root, target, max_parcels=max_parcels)

    x_src, y_src_arr = _frame_to_xy(source_frame)
    x_tgt, y_tgt_arr = _frame_to_xy(target_frame)

    if drop_null_class:
        x_src, y_src_arr = _drop_null(x_src, y_src_arr)
        x_tgt, y_tgt_arr = _drop_null(x_tgt, y_tgt_arr)

    if x_tgt.shape[0] < 2:
        raise ValueError(
            f"Target {target!r} has too few parcels ({x_tgt.shape[0]}) to build a few-shot split."
        )

    # Stratified-ish target test/support partition: hold out `test_fraction`
    # per class as the query set, draw k-shot support from the remainder.
    test_idx, pool_idx = _stratified_holdout(rng, y_tgt_arr, test_fraction)
    support_local = _sample_k_shot(rng, y_tgt_arr[pool_idx], k)
    support_idx = pool_idx[support_local]

    classes = tuple(sorted(set(y_tgt_arr[test_idx].tolist())))
    return FewShotMatrix(
        x_source=x_src,
        y_source=y_src_arr,
        x_target_train=x_tgt[support_idx],
        y_target_train=y_tgt_arr[support_idx],
        x_target_test=x_tgt[test_idx],
        y_target_test=y_tgt_arr[test_idx],
        classes=classes,
    )


def _drop_null(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows whose macro label is :data:`NULL_CLASS`.

    Args:
        x: Feature matrix.
        y: Label vector.

    Returns:
        The filtered ``(x, y)``.
    """
    if x.shape[0] == 0:
        return x, y
    keep = y != NULL_CLASS
    return x[keep], y[keep]


def _stratified_holdout(
    rng: np.random.Generator,
    labels: np.ndarray,
    test_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split indices into a per-class held-out test set and a remaining pool.

    Args:
        rng: Seeded generator.
        labels: Per-sample class labels.
        test_fraction: Fraction held out per class (at least one sample stays in
            the pool when a class has >= 2 members).

    Returns:
        Tuple ``(test_idx, pool_idx)`` of positional index arrays.
    """
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


# ---------------------------------------------------------------------------
# Training the k-shot XGBoost (baseline recipe).
# ---------------------------------------------------------------------------


def train_xgb_kshot(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    pretrain_x: np.ndarray | None = None,
    pretrain_y: np.ndarray | None = None,
) -> dict[str, float]:
    """Train the baseline XGBoost recipe on a k-shot set and score F1-macro.

    Reuses :func:`ml.train.baseline.build_estimator` (the ``SpatialXGBClassifier``
    with the documented ``_XGB_BASE_PARAMS``) so the *recipe* is identical to the
    AlphaEarth baseline; only the input vector differs (S2-derived, not
    AlphaEarth). When ``pretrain_x``/``pretrain_y`` are given they are concatenated
    ahead of the k-shot support set, modelling the "pre-train on the source then
    fine-tune on k target shots" protocol with a single fit (XGBoost has no warm
    fine-tune, so the source pool is folded into the training matrix).

    Args:
        x_train: k-shot target support matrix ``(n_support, d)``.
        y_train: k-shot target support labels.
        x_test: Target query matrix.
        y_test: Target query labels.
        pretrain_x: Optional source-country pre-train matrix.
        pretrain_y: Optional source-country pre-train labels.

    Returns:
        A dict with ``f1_macro`` (in ``[0, 1]``), ``n_train`` and ``n_classes``.

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

    # XGB needs contiguous integer labels; map macro group strings to ids.
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

    # Predict on the query set; unseen test classes simply count as errors.
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


# ---------------------------------------------------------------------------
# The k-shot curve.
# ---------------------------------------------------------------------------


def run_fewshot_curve(
    root: Path | str,
    *,
    source: Sequence[str],
    target: str,
    k_shots: Iterable[int] = K_SHOTS,
    seeds: Sequence[int] = (0, 1, 2),
    use_pretrain: bool = True,
    max_parcels: int | None = None,
) -> pl.DataFrame:
    """Compute the real F1-macro-vs-k curve for one transfer scenario.

    Loops over ``k_shots`` x ``seeds``; for each, builds the k-shot split,
    trains the baseline XGBoost recipe (optionally pre-trained on the source
    pool) and records the F1-macro on the target query set. The result is a long
    frame with one row per ``(source, target, k, seed)`` -- the seed dimension
    yields error bars and the curve is strictly REAL (no fabricated numbers; an
    absent subset raises :class:`EuroCropsMLDataMissing`).

    Args:
        root: EuroCropsML root directory.
        source: Source country keys (pre-train pool).
        target: Target country key.
        k_shots: k ladder (defaults to :data:`K_SHOTS`).
        seeds: Seeds for error bars.
        use_pretrain: When ``True`` (default) the XGBoost is trained on the
            source pool + the k target shots ("pre-train -> fine-tune"); when
            ``False`` it is trained on the k target shots only (the
            "no pre-train" reference of the paper).
        max_parcels: Optional per-region parcel cap (smoke runs).

    Returns:
        A Polars frame with columns
        ``(source, target, k, seed, f1_macro, n_classes, use_pretrain)``.

    Raises:
        EuroCropsMLDataMissing: if the subset is absent.
    """
    source_label = "+".join(_REGION_CODES.get(s, s.upper())[:2] for s in source)
    target_label = _REGION_CODES.get(target, target.upper())
    rows: list[dict[str, object]] = []
    for k in k_shots:
        for seed in seeds:
            split = build_fewshot_splits(
                root,
                source=source,
                target=target,
                k=int(k),
                seed=int(seed),
                max_parcels=max_parcels,
            )
            metrics = train_xgb_kshot(
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
                }
            )
            logger.info(
                "fewshot_point",
                source=source_label,
                target=target_label,
                k=int(k),
                seed=int(seed),
                f1_macro=round(metrics["f1_macro"], 4),
            )
    return pl.DataFrame(rows)


def summarize_curve(curve: pl.DataFrame) -> pl.DataFrame:
    """Aggregate a raw curve frame into per-``k`` mean/std F1-macro.

    Args:
        curve: Long frame from :func:`run_fewshot_curve`.

    Returns:
        A frame ``(source, target, k, f1_mean, f1_std, n_seeds)`` sorted by k.
    """
    return (
        curve.group_by("source", "target", "k", "use_pretrain")
        .agg(
            pl.col("f1_macro").mean().alias("f1_mean"),
            pl.col("f1_macro").std(ddof=0).fill_null(0.0).alias("f1_std"),
            pl.len().alias("n_seeds"),
        )
        .sort("source", "target", "use_pretrain", "k")
    )


def _count_labels(samples: Sequence[ParcelSample]) -> dict[str, int]:
    """Count macro HCAT labels across parcels (notebook distribution table).

    Args:
        samples: Parcels to summarize.

    Returns:
        A dict ``{macro_group: count}``.
    """
    macro = align_codes_to_hcat_macro([s.hcat_code for s in samples])
    counts: dict[str, int] = defaultdict(int)
    for label in macro:
        counts[label] += 1
    return dict(counts)
