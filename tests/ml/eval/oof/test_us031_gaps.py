"""Gap tests for US-031 not covered by ``test_dump_oof`` / ``test_softmax_remap``.

Three families the sibling suites leave open:

1. ``_forward_logits`` REGRESSION across the six ``model_kind`` forward branches.
   ``test_dump_oof`` only exercises a single ``_FixedLogitsModel`` (whose forward
   ignores its input) for the unet/deeplab branch, so the per-architecture
   dispatch (``model(xb)`` vs ``model(xb, positions)`` for U-TAE, the AnySat
   64 px downsample + ``dates``, the TSViT/U-TAE tuple unwrap) is never asserted.
   Here a per-kind spy model records exactly how it was called and confirms that,
   for every kind, ``predict_patch_for_kind`` (post refactor) == ``argmax`` of the
   logits the model actually emitted == ``argmax(softmax_patch_for_kind)``.

2. ANTI-LEAK: ``dump_oof`` with NO explicit ``fold`` defaults to the held-out
   fold-5 and marks ``held_out=True`` on every row (the sibling test always
   passes ``fold=5`` explicitly). Plus a direct check that the train-only
   normalization average excludes the held-out fold's statistics.

3. The fold-4 leakage guard at the ``_train_norm_stats`` level (train folds
   1,2,3 only), independent of the dataset mock.

MOCKED end to end: no checkpoint is loaded, ``torch.hub`` (AnySat) is never
contacted, MLflow is never reached, and no real PASTIS patch is read. Fixtures
are deterministic (fixed seeds / fixed logits).
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import torch
from torch import nn

import ml.eval.oof.dump_oof as dump_mod
import ml.eval.segmentation_inference as seg_inf
from ml.eval.checkpoint_registry import CheckpointSpec
from ml.eval.class_remap import HARNESS_SIZE
from ml.eval.oof.parquet_io import read_softmax_parquet
from tests.ml.eval.oof.fixtures.oof_synthetic import make_logits

# ===========================================================================
# Gap 1: _forward_logits regression over the six model_kind forward branches.
# ===========================================================================

#: The five non-SegFormer kinds whose forward runs through ``_forward_logits``.
#: SegFormer has its own 3-RGB/256 sub-pipeline and is excluded by design (see
#: ``test_dump_oof.test_forward_logits_rejects_segformer``).
_DISPATCH_KINDS: tuple[str, ...] = (
    "unet",
    "deeplabv3plus",
    "tsvit",
    "tsvit-pheno",
    "utae",
    "anysat",
)

#: Native class count per kind (20 for unet/utae/anysat, 18 for the others); the
#: forward dispatch is identical regardless, but the logits keep the native C so
#: ``argmax`` parity is checked in the model's own class space.
_NATIVE_C: dict[str, int] = {
    "unet": 20,
    "deeplabv3plus": 18,
    "tsvit": 18,
    "tsvit-pheno": 18,
    "utae": 20,
    "anysat": 20,
}


class _SpyModel(nn.Module):
    """Records how ``_forward_logits`` invoked it and returns shaped logits.

    Each architecture branch in :func:`ml.eval.segmentation_inference._forward_logits`
    calls the model differently:

    - ``unet`` / ``deeplabv3plus`` / ``tsvit`` / ``tsvit-pheno``: ``model(xb)``.
    - ``utae``: ``model(xb, positions)`` (a second positional arg).
    - ``anysat``: ``model(xb_small, dates)`` where ``xb_small`` has been resized
      to the 64 px encoder grid.

    The spy returns logits whose spatial side matches the tensor it received
    (so AnySat's resized input is observed at 64), letting the regression assert
    BOTH that the dispatch reached the right branch AND that ``argmax`` of the
    emitted logits equals ``predict_patch_for_kind``. ``return_tuple`` wraps the
    logits in a 1-tuple to exercise the ``out[0] if isinstance(out, tuple)``
    unwrap used by the temporal models.
    """

    def __init__(self, *, num_classes: int, return_tuple: bool, seed: int) -> None:
        super().__init__()
        self.register_parameter("w", nn.Parameter(torch.zeros(1)))
        self._num_classes = num_classes
        self._return_tuple = return_tuple
        self._seed = seed
        self.calls: list[dict[str, object]] = []

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor]:
        # x is always the batched (B, ...) tensor; the spatial side is the last
        # dim. AnySat downsamples to 64 before calling, so this is where we see
        # the resize happen.
        spatial = int(x.shape[-1])
        self.calls.append(
            {
                "ndim": x.ndim,
                "spatial": spatial,
                "has_positions": positions is not None,
            }
        )
        rng = np.random.default_rng(self._seed + spatial)
        logits_np = rng.uniform(-5.0, 5.0, size=(1, self._num_classes, spatial, spatial)).astype(
            np.float32
        )
        logits = torch.from_numpy(logits_np)
        return (logits,) if self._return_tuple else logits


def _make_input(kind: str, *, size: int) -> torch.Tensor:
    """Build the architecture-appropriate input the dataset would hand over.

    2D ``(10, H, W)`` for the spatial models, temporal ``(T, 10, H, W)`` for the
    temporal ones (TSViT / U-TAE / AnySat).
    """
    if kind in ("unet", "deeplabv3plus"):
        return torch.zeros(10, size, size, dtype=torch.float32)
    return torch.zeros(4, 10, size, size, dtype=torch.float32)


@pytest.mark.parametrize("kind", _DISPATCH_KINDS)
def test_forward_logits_dispatch_regression(kind: str) -> None:
    """Each kind's forward branch reaches the model and stays argmax-consistent.

    Asserts the three properties the ``_forward_logits`` refactor must preserve:

    - the model is actually invoked (the dispatch did not no-op),
    - ``predict_patch_for_kind`` == ``argmax`` over the emitted logits ==
      ``argmax(softmax_patch_for_kind)``,
    - the architecture-specific call contract holds (U-TAE / AnySat receive a
      second positional arg -- ``positions`` / ``dates`` -- and AnySat is fed at
      the 64 px encoder grid).
    """
    # TSViT/U-TAE unwrap a tuple output; the spatial models return a bare tensor.
    return_tuple = kind in ("tsvit", "tsvit-pheno", "utae")
    model = _SpyModel(num_classes=_NATIVE_C[kind], return_tuple=return_tuple, seed=7)
    x = _make_input(kind, size=16)

    logits = seg_inf._forward_logits(model, x, model_kind=kind)
    pred = seg_inf.predict_patch_for_kind(model, x, model_kind=kind)
    probs = seg_inf.softmax_patch_for_kind(model, x, model_kind=kind)

    assert model.calls, "the model forward was never reached"
    # Determinism: the spy returns the same logits for the same spatial side, so
    # the three calls observe byte-identical logits and argmax must agree.
    np.testing.assert_array_equal(
        pred, logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)
    )
    np.testing.assert_array_equal(pred, probs.argmax(axis=0))

    # Per-architecture call contract. U-TAE passes ``positions`` and AnySat
    # passes ``dates`` as a second positional; the spatial / TSViT branches do
    # not (TSViT relies on its learned ordinal PE, ``doy`` not supplied).
    last = model.calls[-1]
    if kind in ("utae", "anysat"):
        assert last["has_positions"] is True, (
            f"{kind} must receive a second positional (positions/dates)"
        )
    else:
        assert last["has_positions"] is False
    if kind == "anysat":
        # The temporal series was downsampled to the 64 px encoder grid before
        # the forward (the OOM mitigation), so the model saw 64, not 16.
        assert last["spatial"] == 64
    elif kind in ("unet", "deeplabv3plus"):
        assert last["ndim"] == 4  # (B, C, H, W)
        assert last["spatial"] == 16
    else:  # tsvit / tsvit-pheno / utae are temporal (B, T, C, H, W)
        assert last["ndim"] == 5
        assert last["spatial"] == 16


def test_forward_logits_tuple_and_tensor_outputs_agree() -> None:
    """The tuple-unwrap path yields the same prediction as a bare-tensor output.

    A temporal model that returns ``(logits,)`` and one that returns ``logits``
    must reduce to the same class map for identical logits, proving the
    ``out[0] if isinstance(out, tuple)`` unwrap is transparent.
    """
    fixed = make_logits(num_classes=18, size=12, scale=4.0, seed=11)

    class _Bare(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_parameter("w", nn.Parameter(torch.zeros(1)))

        def forward(self, *_a: object, **_k: object) -> torch.Tensor:
            return torch.from_numpy(fixed)

    class _Tupled(_Bare):
        def forward(self, *a: object, **k: object) -> tuple[torch.Tensor]:
            return (super().forward(*a, **k),)

    x = torch.zeros(4, 10, 12, 12)
    pred_bare = seg_inf.predict_patch_for_kind(_Bare(), x, model_kind="tsvit-pheno")
    pred_tupled = seg_inf.predict_patch_for_kind(_Tupled(), x, model_kind="tsvit-pheno")
    np.testing.assert_array_equal(pred_bare, pred_tupled)


# ===========================================================================
# Gap 2: dump_oof default fold is the held-out fold-5 (no explicit fold arg).
# ===========================================================================


class _FakeSegDataset:
    """Minimal PASTIS dataset stub recording the kwargs it was built with."""

    last_kwargs: ClassVar[dict[str, object]] = {}

    def __init__(self, **kwargs: object) -> None:
        type(self).last_kwargs = dict(kwargs)
        self.folds = tuple(kwargs.get("folds", ()))  # type: ignore[arg-type]
        self.root = Path(str(kwargs.get("root", "data/PASTIS-R")))
        self.patch_ids = ["10000", "10001"]
        # Per-fold stats where ONLY the held-out fold (5) is poisoned (value 999);
        # the train folds 1,2,3 carry 1.0/2.0/3.0. If fold-5 leaked into the
        # average the mean would jump well above the train-fold mean of 2.0.
        self._norm_stats: dict[int, tuple[np.ndarray, np.ndarray]] = {
            1: (np.full(10, 1.0), np.full(10, 1.0)),
            2: (np.full(10, 2.0), np.full(10, 2.0)),
            3: (np.full(10, 3.0), np.full(10, 3.0)),
            4: (np.full(10, 50.0), np.full(10, 50.0)),
            5: (np.full(10, 999.0), np.full(10, 999.0)),
        }
        self._fold_of: dict[str, int] = dict.fromkeys(self.patch_ids, 5)

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.zeros(10, HARNESS_SIZE, HARNESS_SIZE, dtype=torch.float32)
        y = torch.zeros(HARNESS_SIZE, HARNESS_SIZE, dtype=torch.int64)
        return x, y


class _DummyModel(nn.Module):
    """Model with a single parameter so ``next(model.parameters())`` works."""

    def __init__(self) -> None:
        super().__init__()
        self.register_parameter("w", nn.Parameter(torch.zeros(1)))


def _install_dump_mocks(monkeypatch: pytest.MonkeyPatch) -> type[_FakeSegDataset]:
    """Patch dataset / loader / softmax / ParcelIDs at their import sites."""
    import ml.data.pastis_seg_dataset as ds_mod

    _FakeSegDataset.last_kwargs = {}
    monkeypatch.setattr(ds_mod, "PASTISSegmentationDataset", _FakeSegDataset)
    monkeypatch.setattr(seg_inf, "load_checkpoint_model", lambda spec, **_kw: _DummyModel())

    def _fake_softmax(model: nn.Module, x: torch.Tensor, *, model_kind: str) -> np.ndarray:
        logits = make_logits(num_classes=18, size=HARNESS_SIZE, seed=3)[0]
        shifted = logits - logits.max(axis=0, keepdims=True)
        exp = np.exp(shifted)
        return (exp / exp.sum(axis=0, keepdims=True)).astype(np.float32)

    monkeypatch.setattr(seg_inf, "softmax_patch_for_kind", _fake_softmax)
    # The parcel sidecar is exercised by the integration test; here ParcelIDs are
    # always absent so the dump degrades gracefully (debug log, no raise).
    monkeypatch.setattr(
        dump_mod,
        "load_pastis_parcel_ids",
        lambda *_a, **_k: (_ for _ in ()).throw(FileNotFoundError("no parcels")),
    )
    return _FakeSegDataset


def _spec_18(model_kind: str) -> CheckpointSpec:
    """An 18-class CheckpointSpec whose path exists (points at this test file)."""
    return CheckpointSpec(
        name=model_kind,
        model_kind=model_kind,  # type: ignore[arg-type]
        path=Path(__file__).resolve(),
        native_num_classes=18,
        native_ignore_index=255,
    )


def test_dump_default_fold_is_held_out_five(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dump_oof with NO explicit fold defaults to fold-5 and marks held_out=True.

    The sibling test always passes ``fold=5``; this asserts the DEFAULT (the
    anti-leak invariant R-OOF: the only genuinely held-out fold is dumped, and
    every row is flagged ``held_out=True`` so the meta-learner can filter on it).
    """
    fake_ds = _install_dump_mocks(monkeypatch)
    spec = _spec_18("deeplabv3plus")

    manifest = dump_mod.dump_oof(
        {"deeplabv3plus": spec},
        out_dir=tmp_path,
        device="cpu",
        max_patches=2,
        # NOTE: fold is intentionally omitted -> must default to 5.
    )

    assert fake_ds.last_kwargs.get("folds") == (5,)
    assert manifest["fold"] == 5
    assert manifest["held_out"] is True
    entry = manifest["models"]["deeplabv3plus"]
    assert entry["held_out"] is True
    df = read_softmax_parquet(Path(entry["path"]))
    assert df["held_out"].to_list() == [True, True]
    assert df["fold"].to_list() == [5, 5]


def test_dump_norm_average_excludes_held_out_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The applied normalization average uses train folds only (fold-5 dropped).

    Spies on the real ``_apply_train_norm`` and asserts that, AFTER the dump
    overwrites the dataset stats, EVERY fold maps to the train-fold (1,2,3)
    average (mean 2.0) and never to the held-out fold-5's poisoned value (999.0).
    This is the R-LEAK-NORM guard end to end, not just the spy call.
    """
    captured: dict[str, _FakeSegDataset] = {}
    _install_dump_mocks(monkeypatch)

    import ml.eval.dense_metrics as dense_metrics

    real_apply = dense_metrics._apply_train_norm

    def _spy(dataset: object) -> None:
        real_apply(dataset)
        captured["ds"] = dataset  # type: ignore[assignment]

    monkeypatch.setattr(dense_metrics, "_apply_train_norm", _spy)

    spec = _spec_18("deeplabv3plus")
    dump_mod.dump_oof({"deeplabv3plus": spec}, out_dir=tmp_path, device="cpu", max_patches=1)

    ds = captured["ds"]
    assert ds._norm_stats, "stats must remain after the train-only overwrite"
    for mean, std in ds._norm_stats.values():  # type: ignore[attr-defined]
        # mean over train folds (1+2+3)/3 == 2.0; the held-out fold-5 (999.0) and
        # the selection fold-4 (50.0) are both absent.
        np.testing.assert_allclose(mean, np.full(10, 2.0))
        np.testing.assert_allclose(std, np.full(10, 2.0))
        assert not np.allclose(mean, np.full(10, 999.0)), "fold-5 leaked into norm"
        assert not np.allclose(mean, np.full(10, 50.0)), "fold-4 leaked into norm"


def test_train_norm_stats_drops_held_out_fold() -> None:
    """`_train_norm_stats` averages only folds 1,2,3 regardless of fold-5 value.

    Unit-level guard on the anti-leak helper the dump reuses verbatim: poisoning
    fold-5 (and fold-4) does not move the train-only average.
    """
    from ml.eval.dense_metrics import _train_norm_stats

    raw = {
        1: (np.full(10, 1.0), np.full(10, 1.0)),
        2: (np.full(10, 2.0), np.full(10, 2.0)),
        3: (np.full(10, 3.0), np.full(10, 3.0)),
        4: (np.full(10, 50.0), np.full(10, 50.0)),
        5: (np.full(10, 999.0), np.full(10, 999.0)),
    }
    mean, std = _train_norm_stats(raw)  # type: ignore[misc]
    np.testing.assert_allclose(mean, np.full(10, 2.0))
    np.testing.assert_allclose(std, np.full(10, 2.0))
