"""End-to-end roundtrip: dump_oof -> read_softmax_parquet -> pixel_to_parcel_probs.

The sibling ``test_dump_oof`` always monkeypatches ``load_pastis_parcel_ids`` to
fail, so the per-parcel sidecar branch of :func:`ml.eval.oof.dump_oof.dump_oof`
is never exercised end to end, and the three modules (``dump_oof`` -> ``parquet_io``
-> ``parcel_reconcile``) are never run as a chain over the SAME data. This suite
closes that gap with a real (synthetic) ParcelIDs raster written to ``tmp_path``:

1. ``dump_oof`` runs with a deterministic per-patch softmax and writes BOTH the
   per-pixel parquet and the per-parcel sidecar.
2. ``read_softmax_parquet`` reconstructs the dense ``(18, 128, 128)`` softmax.
3. ``pixel_to_parcel_probs`` re-applied to the reconstructed softmax reproduces
   the sidecar the dump wrote (within float16 storage tolerance), proving the
   two paths agree and that the canonical ids / supports survive serialization.

No checkpoint, ``torch.hub``, MLflow or real PASTIS patch is touched; the dataset,
the model loader and the softmax are mocked, and the ParcelIDs npy is the only
on-disk artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl
import pytest
import torch
from torch import nn

import ml.eval.oof.dump_oof as dump_mod
import ml.eval.segmentation_inference as seg_inf
from ml.eval.checkpoint_registry import CheckpointSpec
from ml.eval.class_remap import HARNESS_SIZE
from ml.eval.oof.parquet_io import read_softmax_parquet
from ml.utils.parcel_reconcile import (
    PROB_COLUMNS,
    load_pastis_parcel_ids,
    pixel_to_parcel_probs,
)
from tests.ml.eval.oof.fixtures.oof_synthetic import make_logits

_F16_TOL = 2e-3

#: The two patch ids the fake dataset serves; each gets its own ParcelIDs raster.
_PATCH_IDS = ["10000", "10001"]


def _patch_softmax(pid: str) -> np.ndarray:
    """Deterministic 18-class softmax for a patch, distinct per patch id."""
    seed = 100 + int(pid)
    logits = make_logits(num_classes=18, size=HARNESS_SIZE, seed=seed)[0]
    shifted = logits - logits.max(axis=0, keepdims=True)
    exp = np.exp(shifted)
    return (exp / exp.sum(axis=0, keepdims=True)).astype(np.float32)


def _write_parcel_rasters(annot_dir: Path) -> dict[str, np.ndarray]:
    """Write a (128,128) ParcelIDs npy per patch and return them by patch id.

    The raster splits the grid into two parcels (101 / 202) plus a Background
    border column (id 0), so the per-parcel supports are known and Background is
    exercised (it must never produce a row).
    """
    annot_dir.mkdir(parents=True, exist_ok=True)
    rasters: dict[str, np.ndarray] = {}
    for pid in _PATCH_IDS:
        grid = np.zeros((HARNESS_SIZE, HARNESS_SIZE), dtype=np.int32)
        half = HARNESS_SIZE // 2
        grid[:, :half] = 101
        grid[:, half : HARNESS_SIZE - 1] = 202
        # last column stays Background (0)
        np.save(annot_dir / f"ParcelIDs_{pid}.npy", grid)
        rasters[pid] = grid.astype(np.int64)
    return rasters


class _FakeSegDataset:
    """PASTIS dataset stub whose ``root`` points at the tmp dir with ParcelIDs."""

    root_override: ClassVar[Path | None] = None
    last_kwargs: ClassVar[dict[str, object]] = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).last_kwargs = dict(kwargs)
        self.folds = tuple(kwargs.get("folds", ()))  # type: ignore[arg-type]
        root = kwargs.get("root") or type(self).root_override
        assert root is not None
        self.root = Path(str(root))
        self.patch_ids = list(_PATCH_IDS)
        self._norm_stats: dict[int, tuple[np.ndarray, np.ndarray]] = {
            f: (np.full(10, float(f)), np.full(10, float(f))) for f in (1, 2, 3)
        }
        self._fold_of: dict[str, int] = dict.fromkeys(self.patch_ids, 5)

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.zeros(10, HARNESS_SIZE, HARNESS_SIZE, dtype=torch.float32)
        y = torch.zeros(HARNESS_SIZE, HARNESS_SIZE, dtype=torch.int64)
        return x, y


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_parameter("w", nn.Parameter(torch.zeros(1)))


@pytest.fixture
def _pastis_root(tmp_path: Path) -> tuple[Path, dict[str, np.ndarray]]:
    """A tmp PASTIS-R root holding only the synthetic ParcelIDs rasters."""
    root = tmp_path / "PASTIS-R"
    rasters = _write_parcel_rasters(root / "ANNOTATIONS")
    return root, rasters


def _install_mocks(monkeypatch: pytest.MonkeyPatch, *, data_root: Path) -> None:
    """Wire dataset / loader / per-patch softmax mocks for the dump."""
    import ml.data.pastis_seg_dataset as ds_mod

    _FakeSegDataset.root_override = data_root
    _FakeSegDataset.last_kwargs = {}
    monkeypatch.setattr(ds_mod, "PASTISSegmentationDataset", _FakeSegDataset)
    monkeypatch.setattr(seg_inf, "load_checkpoint_model", lambda spec, **_kw: _DummyModel())

    # The softmax is keyed by the patch id via a per-call counter so each patch
    # gets its own distribution (the dump iterates patches in order).
    state = {"i": 0}

    def _fake_softmax(model: nn.Module, x: torch.Tensor, *, model_kind: str) -> np.ndarray:
        pid = _PATCH_IDS[state["i"] % len(_PATCH_IDS)]
        state["i"] += 1
        return _patch_softmax(pid)

    monkeypatch.setattr(seg_inf, "softmax_patch_for_kind", _fake_softmax)


def _spec_18(model_kind: str = "deeplabv3plus") -> CheckpointSpec:
    return CheckpointSpec(
        name=model_kind,
        model_kind=model_kind,  # type: ignore[arg-type]
        path=Path(__file__).resolve(),
        native_num_classes=18,
        native_ignore_index=255,
    )


def test_dump_to_parquet_to_parcel_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _pastis_root: tuple[Path, dict[str, np.ndarray]],
) -> None:
    """dump_oof writes both parquets; the sidecar matches re-reduced per-pixel data."""
    data_root, rasters = _pastis_root
    _install_mocks(monkeypatch, data_root=data_root)
    spec = _spec_18()

    manifest = dump_mod.dump_oof(
        {"deeplabv3plus": spec},
        fold=5,
        out_dir=tmp_path,
        data_root=data_root,
        device="cpu",
        write_parcel=True,
    )

    entry = manifest["models"]["deeplabv3plus"]
    assert entry["status"] == "ok"
    assert entry["n_patches"] == len(_PATCH_IDS)
    pixel_path = Path(entry["path"])
    parcel_path = Path(entry["parcel_path"])
    assert pixel_path.exists()
    assert parcel_path.exists()

    # 1) Per-pixel parquet reconstructs to (18, 128, 128) per patch.
    pixel_df = read_softmax_parquet(pixel_path)
    assert pixel_df.height == len(_PATCH_IDS)
    by_pid = {row["patch_id"]: row for row in pixel_df.iter_rows(named=True)}

    # 2) Per-parcel sidecar carries the contract columns + provenance.
    parcel_df = pl.read_parquet(parcel_path)
    for col in (
        "canonical_parcel_id",
        "patch_id",
        "fold",
        "held_out",
        "model",
        *PROB_COLUMNS,
        "pred_class",
        "n_pixels",
        "code_version",
        "data_version",
    ):
        assert col in parcel_df.columns, col
    assert parcel_df["held_out"].to_list() == [True] * parcel_df.height
    assert parcel_df["model"].unique().to_list() == ["deeplabv3plus"]

    # 3) Re-reduce each reconstructed per-pixel softmax with pixel_to_parcel_probs
    #    and confirm it matches the sidecar the dump wrote (within float16 tol).
    for pid in _PATCH_IDS:
        recon = by_pid[pid]["softmax"].astype(np.float32)
        assert recon.shape == (18, HARNESS_SIZE, HARNESS_SIZE)
        parcel_ids = load_pastis_parcel_ids(pid, data_root)
        np.testing.assert_array_equal(parcel_ids, rasters[pid])

        expected = pixel_to_parcel_probs(recon, parcel_ids, patch_id=pid, method="mean")
        # Background border (id 0) yields no row -> only parcels 101, 202.
        assert set(expected["canonical_parcel_id"].to_list()) == {
            f"{pid}_101",
            f"{pid}_202",
        }

        sidecar = (
            parcel_df.filter(pl.col("patch_id") == pid)
            .select("canonical_parcel_id", *PROB_COLUMNS, "pred_class", "n_pixels")
            .sort("canonical_parcel_id")
        )
        exp_sorted = expected.select(
            "canonical_parcel_id", *PROB_COLUMNS, "pred_class", "n_pixels"
        ).sort("canonical_parcel_id")

        assert (
            sidecar["canonical_parcel_id"].to_list() == exp_sorted["canonical_parcel_id"].to_list()
        )
        assert sidecar["n_pixels"].to_list() == exp_sorted["n_pixels"].to_list()
        np.testing.assert_allclose(
            sidecar.select(PROB_COLUMNS).to_numpy(),
            exp_sorted.select(PROB_COLUMNS).to_numpy(),
            atol=_F16_TOL,
        )
        # pred_class is the argmax of a sharp-enough mean -> stable under float16.
        assert sidecar["pred_class"].to_list() == exp_sorted["pred_class"].to_list()


def test_parcel_probs_rows_sum_to_one_after_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _pastis_root: tuple[Path, dict[str, np.ndarray]],
) -> None:
    """Every sidecar row written by the dump is still a valid distribution."""
    data_root, _rasters = _pastis_root
    _install_mocks(monkeypatch, data_root=data_root)

    manifest = dump_mod.dump_oof(
        {"deeplabv3plus": _spec_18()},
        fold=5,
        out_dir=tmp_path,
        data_root=data_root,
        device="cpu",
        write_parcel=True,
    )
    parcel_df = pl.read_parquet(manifest["models"]["deeplabv3plus"]["parcel_path"])
    prob_matrix = parcel_df.select(PROB_COLUMNS).to_numpy()
    assert prob_matrix.shape[0] == 2 * len(_PATCH_IDS)  # 2 parcels x 2 patches
    np.testing.assert_allclose(prob_matrix.sum(axis=1), 1.0, atol=1e-5)


def test_dump_skips_parcel_sidecar_when_rasters_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ParcelIDs on disk -> per-pixel parquet still written, no parcel sidecar.

    Complements the roundtrip: the parcel branch degrades gracefully (debug log,
    no raise) when the ANNOTATIONS rasters were not pulled, while the dense dump
    remains intact.
    """
    empty_root = tmp_path / "PASTIS-R-empty"
    (empty_root / "ANNOTATIONS").mkdir(parents=True)
    _install_mocks(monkeypatch, data_root=empty_root)

    manifest = dump_mod.dump_oof(
        {"deeplabv3plus": _spec_18()},
        fold=5,
        out_dir=tmp_path,
        data_root=empty_root,
        device="cpu",
        write_parcel=True,
    )
    entry = manifest["models"]["deeplabv3plus"]
    assert entry["status"] == "ok"
    assert Path(entry["path"]).exists()
    assert entry["parcel_path"] is None
    assert not list(tmp_path.glob("oof_parcel_*.parquet"))
