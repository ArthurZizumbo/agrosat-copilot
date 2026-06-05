"""Loading of BreizhCrops time series from disk into Polars structures.

BreizhCrops (Russwurm et al., ISPRS Archives 2020 — maintained successor of the
Russwurm & Korner dataset, ISPRS IJGI 2018) provides Sentinel-2 time series per
agricultural parcel in Brittany (France). Each parcel is a sequence
``(T, n_bands)`` stored in an HDF5 store ``<region>.h5``, with a tabular index
``<region>.csv`` and a 9-class mapping in ``classmapping.csv``.

Unlike PASTIS-R (dense 128x128 grid), BreizhCrops is a collection of per-object
series: 1 multiband temporal vector per parcel, without a spatial component.
This makes it the natural complement to validate that the temporal features
(FFT / phenology) generalize cross-region.

This module exposes lightweight helpers that reuse the official ``breizhcrops``
package with download DISABLED: if the files are not on disk with the expected
layout, the public functions return Polars DataFrames with a valid EMPTY schema
(degraded mode, mirror of ``pastis_loader.py``), so that any notebook completes
its run without error and without touching the network.

The download is manual and one-time via ``scripts/download_breizhcrops.sh``.

Expected layout (root = ``data/breizhcrops/``)::

    data/breizhcrops/classmapping.csv
    data/breizhcrops/codes.csv
    data/breizhcrops/2017/L2A/frh04.csv
    data/breizhcrops/2017/L2A/frh04.h5
    data/breizhcrops/2017/L2A/frh01.csv
    data/breizhcrops/2017/L2A/frh01.h5
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

# Bands kept by the breizhcrops package at L2A level. The order
# replicates SELECTED_BANDS["L2A"] from breizhcrops.datasets.breizhcrops: column
# 0 of each series is `doa` (date as integer) and the next 10
# are the optical bands; CLD/EDG/SAT (masks) are discarded in EDA.
BREIZHCROPS_L2A_BANDS: list[str] = [
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
]
"""Canonical order of the 10 BreizhCrops Sentinel-2 L2A optical bands.

Mapped to the project nomenclature (B2->B02, etc.) to align with
``PASTIS_S2_BANDS`` and allow direct cross-dataset comparison.
"""

# Positional index of each band within the raw array returned by
# breizhcrops.BreizhCrops.load() for L2A: [doa, B2, B3, B4, B5, B6, B7,
# B8, B8A, B11, B12, CLD, EDG, SAT]. Index 0 (doa) is NOT a band.
_L2A_BAND_OFFSET: int = 1

BREIZHCROPS_CLASSES: dict[int, str] = {
    0: "barley",
    1: "wheat",
    2: "rapeseed",
    3: "corn",
    4: "sunflower",
    5: "orchards",
    6: "nuts",
    7: "permanent meadows",
    8: "temporary meadows",
}
"""Mapping `class_id -> name` of the 9 canonical BreizhCrops classes.

Source: ``classmapping.csv`` distributed with the dataset (public S2 bucket). It
is hardcoded so that the module exposes the taxonomy even in degraded mode
(without the dataset downloaded).
"""

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ROOT = _REPO_ROOT / "data" / "breizhcrops"

_PARCEL_INDEX_SCHEMA: dict[str, Any] = {
    "parcel_id": pl.Utf8,
    "region": pl.Utf8,
    "year": pl.Int64,
    "level": pl.Utf8,
    "code_cultu": pl.Utf8,
    "class_id": pl.Int16,
    "class_name": pl.Utf8,
    "sequence_length": pl.Int64,
}

_PIXEL_SERIES_SCHEMA: dict[str, Any] = {
    "parcel_id": pl.Utf8,
    "t": pl.Int64,
    "date": pl.Int64,
    "doy": pl.Int64,
    "band": pl.Utf8,
    "value": pl.Float64,
    "class_id": pl.Int16,
    "class_name": pl.Utf8,
}


def _required_paths(root: Path, region: str, year: int, level: str) -> dict[str, Path]:
    """Builds the paths the breizhcrops package expects for a region.

    Args:
        root: Root of the dataset (``data/breizhcrops/``).
        region: BreizhCrops region (e.g. ``frh04``).
        year: Cycle year (only 2017 verified).
        level: Sentinel-2 processing level (``L2A``).

    Returns:
        Dictionary with keys ``classmapping``, ``codes``, ``index``, ``h5``.
    """
    level_dir = root / str(year) / level
    return {
        "classmapping": root / "classmapping.csv",
        "codes": root / "codes.csv",
        "index": level_dir / f"{region}.csv",
        "h5": level_dir / f"{region}.h5",
    }


def _dataset_available(root: Path, region: str, year: int, level: str) -> bool:
    """Verifies that ALL the required files exist on disk.

    This guard is what guarantees that a network download is never triggered: we
    only instantiate ``breizhcrops.BreizhCrops`` when the full layout is already
    present locally.

    Args:
        root: Root of the dataset.
        region: BreizhCrops region.
        year: Cycle year.
        level: Processing level.

    Returns:
        ``True`` if classmapping, codes, index and h5 exist and are not empty;
        ``False`` otherwise (activates degraded mode).
    """
    paths = _required_paths(root, region, year, level)
    return all(p.exists() and p.stat().st_size > 0 for p in paths.values())


def _open_dataset(root: Path, region: str, year: int, level: str) -> Any | None:
    """Instantiates ``breizhcrops.BreizhCrops`` WITHOUT download (offline-safe).

    The package does not expose a ``download=False`` flag: it downloads if files
    are missing. That is why we only build the dataset when
    :func:`_dataset_available` confirms that everything is on disk. If the package
    is not installed or the construction fails, we return ``None`` to fall back to
    degraded mode.

    Args:
        root: Root of the dataset.
        region: BreizhCrops region.
        year: Cycle year.
        level: Processing level.

    Returns:
        Instance of ``BreizhCrops`` or ``None`` if it cannot be loaded without
        the network.
    """
    if not _dataset_available(root, region, year, level):
        return None
    try:
        from breizhcrops import BreizhCrops  # type: ignore[import-untyped]
    except ImportError:
        return None
    try:
        return BreizhCrops(
            region=region,
            root=str(root),
            year=year,
            level=level,
            load_timeseries=True,
            verbose=False,
        )
    except Exception:  # noqa: BLE001
        # Construction failed (corrupt h5, incompatible index, etc.):
        # we degrade to an empty schema instead of propagating. No log because
        # breizhcrops is optional and the notebook documents the mode.
        return None


def breizhcrops_parcel_index(
    region: str = "frh04",
    year: int = 2017,
    level: str = "L2A",
    root: Path | None = None,
) -> pl.DataFrame:
    """Returns the flat index of BreizhCrops parcels for a region.

    Equivalent to ``pastis_patch_index`` but for per-object series: one row per
    parcel with its class and the length of its time series. Useful for stratified
    sampling by class before loading the H5 series.

    Args:
        region: BreizhCrops region (``frh01``..``frh04``, ``belle-ile``).
        year: Agricultural cycle year (only 2017 verified).
        level: Sentinel-2 processing level (``L2A`` recommended).
        root: Root of the dataset. If ``None``, uses ``data/breizhcrops/``
            relative to the repo.

    Returns:
        Polars DataFrame with columns ``parcel_id, region, year, level,
        code_cultu, class_id, class_name, sequence_length``. Empty (with a valid
        schema) if the dataset is not downloaded or the ``breizhcrops`` package is
        not available.
    """
    root = root or _DEFAULT_ROOT
    ds = _open_dataset(root, region, year, level)
    if ds is None:
        return pl.DataFrame(schema=_PARCEL_INDEX_SCHEMA)

    idx = ds.index.reset_index()
    rows: list[dict[str, Any]] = []
    for _, r in idx.iterrows():
        class_id = int(r["classid"])
        rows.append(
            {
                "parcel_id": str(r["id"]),
                "region": region,
                "year": int(year),
                "level": level,
                "code_cultu": str(r["CODE_CULTU"]),
                "class_id": class_id,
                "class_name": BREIZHCROPS_CLASSES.get(class_id, str(r.get("classname", "unknown"))),
                "sequence_length": int(r["sequencelength"]),
            }
        )

    if not rows:
        return pl.DataFrame(schema=_PARCEL_INDEX_SCHEMA)
    return pl.DataFrame(rows, schema=_PARCEL_INDEX_SCHEMA)


def _doa_to_date_doy(doa_int: float) -> tuple[int, int]:
    """Converts the `doa` field (datetime64[ns] as int) to (YYYYMMDD, DOY).

    The breizhcrops package stores the acquisition date as
    ``pd.to_datetime(...).astype(int)`` (nanoseconds since epoch). Here we revert
    it to a readable ``YYYYMMDD`` integer and the day of the year.

    Args:
        doa_int: Raw value of column 0 (`doa`) of the series.

    Returns:
        Tuple ``(date_yyyymmdd, doy)``. ``(0, 0)`` if the value is not finite.
    """
    if not np.isfinite(doa_int):
        return 0, 0
    dt = np.datetime64(int(doa_int), "ns")
    day = dt.astype("datetime64[D]")
    year = day.astype("datetime64[Y]").astype(int) + 1970
    months = day.astype("datetime64[M]")
    month = months.astype(int) % 12 + 1
    day_of_month = (day - months).astype(int) + 1
    jan1 = np.datetime64(f"{year:04d}-01-01", "D")
    doy = int((day - jan1).astype(int)) + 1
    return year * 10000 + month * 100 + day_of_month, doy


def breizhcrops_pixel_series(
    region: str = "frh04",
    year: int = 2017,
    level: str = "L2A",
    sample_parcels: int | None = None,
    seed: int = 42,
    root: Path | None = None,
    only_parcel_ids: set[str] | None = None,
) -> pl.DataFrame:
    """Converts BreizhCrops series into a long-format ``pl.DataFrame``.

    Each parcel contributes ``T`` temporal steps x 10 optical bands. The resulting
    long format is directly comparable with the output of ``pastis_to_polars``
    (same semantic columns: ``band``, ``value``, ``class_id``), enabling the
    cross-dataset BreizhCrops vs PASTIS-R analysis.

    Stratified sampling is done per parcel (not per pixel, because BreizhCrops has
    no spatial grid): all observations are expanded from ``sample_parcels``
    parcels chosen with a fixed seed.

    Args:
        region: BreizhCrops region.
        year: Agricultural cycle year.
        level: Processing level (``L2A``).
        sample_parcels: If not ``None``, maximum number of parcels to sample
            (reproducible with ``seed``). ``None`` loads all.
        seed: Seed for the parcel sampling.
        root: Root of the dataset. ``None`` uses ``data/breizhcrops/``.
        only_parcel_ids: If not ``None``, restricts the extraction to the parcels
            whose ``id`` (as string) is in the set. It is the efficient way to
            extract a previously sampled subset without expanding the whole region
            (the full region is hundreds of thousands of parcels). It is applied
            BEFORE ``sample_parcels``.

    Returns:
        Polars DataFrame with columns ``parcel_id, t, date, doy, band, value,
        class_id, class_name``. Empty (valid schema) if the dataset is not
        downloaded or the package is not available.
    """
    root = root or _DEFAULT_ROOT
    ds = _open_dataset(root, region, year, level)
    if ds is None:
        return pl.DataFrame(schema=_PIXEL_SERIES_SCHEMA)

    n_parcels = len(ds)
    if n_parcels == 0:
        return pl.DataFrame(schema=_PIXEL_SERIES_SCHEMA)

    order = np.arange(n_parcels)
    if only_parcel_ids is not None:
        # Filter by positions whose `id` is in the requested set. The
        # breizhcrops index uses a positional RangeIndex, so the
        # position in `order` coincides with `ds.index.iloc[pos]`.
        wanted = {str(p) for p in only_parcel_ids}
        id_series = ds.index["id"].astype(str).to_numpy()
        order = np.where(np.isin(id_series, list(wanted)))[0]
        if order.size == 0:
            return pl.DataFrame(schema=_PIXEL_SERIES_SCHEMA)
    if sample_parcels is not None and sample_parcels < order.size:
        rng = np.random.default_rng(seed)
        order = rng.choice(order, size=sample_parcels, replace=False)

    band_names = BREIZHCROPS_L2A_BANDS
    n_bands = len(band_names)
    frames: list[pl.DataFrame] = []

    # We open the HDF5 ONCE for the entire extraction: reopening the file
    # per parcel dominates the time when `order` has hundreds/thousands of ids.
    try:
        h5_ctx = _h5_open(ds)
    except Exception:  # noqa: BLE001
        return pl.DataFrame(schema=_PIXEL_SERIES_SCHEMA)

    with h5_ctx as h5:
        for i in order:
            try:
                row = ds.index.iloc[int(i)]
                raw = np.asarray(h5[row.path], dtype=np.float64)
            except Exception:  # noqa: BLE001, S112
                # Unreadable series: we skip it without aborting the full load.
                # No log because breizhcrops is optional and the notebook documents
                # the degraded mode (mirror of pastis_loader.py).
                continue
            if raw.ndim != 2 or raw.shape[0] == 0:
                continue

            class_id = int(row["classid"])
            class_name = BREIZHCROPS_CLASSES.get(class_id, "unknown")
            parcel_id = str(row["id"])

            t_steps = raw.shape[0]
            doa_col = raw[:, 0]
            dates = np.empty(t_steps, dtype=np.int64)
            doys = np.empty(t_steps, dtype=np.int64)
            for ti in range(t_steps):
                d, doy = _doa_to_date_doy(doa_col[ti])
                dates[ti] = d
                doys[ti] = doy

            for bi in range(n_bands):
                col = raw[:, _L2A_BAND_OFFSET + bi]
                frames.append(
                    pl.DataFrame(
                        {
                            "parcel_id": [parcel_id] * t_steps,
                            "t": np.arange(t_steps, dtype=np.int64),
                            "date": dates,
                            "doy": doys,
                            "band": [band_names[bi]] * t_steps,
                            "value": col.astype(np.float64),
                            "class_id": np.full(t_steps, class_id, dtype=np.int16),
                            "class_name": [class_name] * t_steps,
                        },
                        schema=_PIXEL_SERIES_SCHEMA,
                    )
                )

    if not frames:
        return pl.DataFrame(schema=_PIXEL_SERIES_SCHEMA)
    return pl.concat(frames, how="vertical_relaxed")


def _h5_open(ds: Any) -> Any:
    """Opens the HDF5 of the BreizhCrops instance in read mode.

    Isolated in its own function so that the caller's ``with`` is readable and so
    it can be mocked in tests without the network or a real h5py.

    Args:
        ds: Instance of ``breizhcrops.BreizhCrops``.

    Returns:
        Context manager of ``h5py.File`` over ``ds.h5path``.
    """
    import h5py  # type: ignore[import-untyped]

    return h5py.File(ds.h5path, "r")


__all__ = [
    "BREIZHCROPS_CLASSES",
    "BREIZHCROPS_L2A_BANDS",
    "breizhcrops_parcel_index",
    "breizhcrops_pixel_series",
]
