"""Deterministic synthetic OOF artifacts for the ensemble base tests (US-040).

The real US-031 OOF parquets are DVC blobs of ~1.5 GB, far too heavy for unit
tests. These factories write *tiny* but format-faithful parquets to a temporary
directory so :meth:`ml.ensemble.base.EnsembleModel.load_oof_members` exercises
the REAL readers (:func:`read_softmax_parquet` for pixel space, plain Polars for
parcel space) on data shaped exactly like the production dump:

- Pixel space: ``oof_{member}_fold5.parquet`` written via
  :func:`ml.eval.oof.parquet_io.write_softmax_parquet` (flat ``softmax`` /
  ``pred`` lists + shape metadata).
- Parcel space: ``oof_parcel_{member}_fold5.parquet`` with
  ``canonical_parcel_id`` + ``prob_000..prob_017`` (post-softmax) + ``pred_class``
  + ``n_pixels``.

Everything is seeded with :func:`numpy.random.default_rng`, so the same ``seed``
yields identical arrays across runs (golden values).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ml.eval.oof.parquet_io import write_softmax_parquet
from ml.utils.parcel_reconcile import PROB_COLUMNS

#: Small spatial side for the synthetic pixel maps (real dump uses 128).
SMALL_SIZE: int = 8
#: Class count of the harness 18-class space.
NUM_CLASSES: int = 18


def make_softmax_map(
    *,
    num_classes: int = NUM_CLASSES,
    size: int = SMALL_SIZE,
    seed: int = 0,
) -> np.ndarray:
    """Build a valid post-softmax map ``(num_classes, size, size)`` summing to 1.

    Args:
        num_classes: Number of class channels.
        size: Spatial side length of the square map.
        seed: Deterministic seed.

    Returns:
        A ``float32`` array ``(num_classes, size, size)`` with each pixel's class
        axis summing to 1.
    """
    rng = np.random.default_rng(seed)
    logits = rng.uniform(-5.0, 5.0, size=(num_classes, size, size)).astype(np.float32)
    shifted = logits - logits.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=0, keepdims=True)).astype(np.float32)


def write_pixel_oof(
    oof_dir: Path,
    member: str,
    *,
    patch_ids: tuple[str, ...] = ("10000", "10001"),
    size: int = SMALL_SIZE,
    seed: int = 0,
) -> Path:
    """Write a tiny per-pixel OOF parquet for ``member`` and return its path.

    Args:
        oof_dir: Directory to write into (created if missing).
        member: Base-learner name used in the file stem.
        patch_ids: Patch ids, one dense softmax row each.
        size: Spatial side of each softmax map.
        seed: Base seed (each patch gets ``seed + i`` for distinct maps).

    Returns:
        Path of the written ``oof_{member}_fold5.parquet``.
    """
    oof_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for i, pid in enumerate(patch_ids):
        sm = make_softmax_map(size=size, seed=seed + i)
        rows.append(
            {
                "patch_id": pid,
                "fold": 5,
                "held_out": True,
                "model": member,
                "status": "ok",
                "softmax": sm,
                "pred": sm.argmax(axis=0).astype(np.int8),
                "code_version": "test",
                "data_version": "test",
            }
        )
    path = oof_dir / f"oof_{member}_fold5.parquet"
    write_softmax_parquet(rows, path, num_classes=NUM_CLASSES, size=size)
    return path


def make_parcel_frame(
    member: str,
    *,
    n_parcels: int = 6,
    patch_id: str = "10000",
    seed: int = 0,
) -> pl.DataFrame:
    """Build a deterministic per-parcel OOF DataFrame (post-softmax rows).

    Args:
        member: Base-learner name stored in the ``model`` column.
        n_parcels: Number of parcel rows.
        patch_id: Patch id used to build canonical ids ``f"{patch_id}_{i}"``.
        seed: Deterministic seed.

    Returns:
        A Polars DataFrame with ``canonical_parcel_id``, ``patch_id``, ``fold``,
        ``held_out``, ``model``, ``prob_000..prob_017`` (sum-to-1), ``pred_class``
        and ``n_pixels``.
    """
    rng = np.random.default_rng(seed)
    logits = rng.uniform(-5.0, 5.0, size=(n_parcels, NUM_CLASSES))
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)

    data: dict[str, object] = {
        "canonical_parcel_id": [f"{patch_id}_{i + 1:03d}" for i in range(n_parcels)],
        "patch_id": [patch_id] * n_parcels,
        "fold": [5] * n_parcels,
        "held_out": [True] * n_parcels,
        "model": [member] * n_parcels,
    }
    for c, col in enumerate(PROB_COLUMNS):
        data[col] = probs[:, c]
    data["pred_class"] = probs.argmax(axis=1).astype(np.int64)
    data["n_pixels"] = rng.integers(50, 500, size=n_parcels).astype(np.int64)
    return pl.DataFrame(data)


def write_parcel_oof(
    oof_dir: Path,
    member: str,
    *,
    n_parcels: int = 6,
    patch_id: str = "10000",
    seed: int = 0,
) -> Path:
    """Write a tiny per-parcel OOF parquet for ``member`` and return its path.

    Args:
        oof_dir: Directory to write into (created if missing).
        member: Base-learner name used in the file stem.
        n_parcels: Number of parcel rows.
        patch_id: Patch id for the canonical ids.
        seed: Deterministic seed.

    Returns:
        Path of the written ``oof_parcel_{member}_fold5.parquet``.
    """
    oof_dir.mkdir(parents=True, exist_ok=True)
    frame = make_parcel_frame(member, n_parcels=n_parcels, patch_id=patch_id, seed=seed)
    path = oof_dir / f"oof_parcel_{member}_fold5.parquet"
    frame.write_parquet(path)
    return path
