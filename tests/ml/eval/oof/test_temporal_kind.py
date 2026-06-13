"""Regression test: temporal model kinds keep the full S2 time series.

Guards ``ml/eval/oof/dump_oof._is_temporal_kind`` -- the bug where
``tsvit-pheno-fullm`` was omitted from the temporal set, so its dataset collapsed
the time axis to a single frame and the TSViT rearrange raised an EinopsError
(``expected 5 dims. Received 4-dim tensor``) during the E-a OOF re-dump.
"""

from __future__ import annotations

from ml.eval.oof.dump_oof import _is_temporal_kind


def test_fullm_is_temporal() -> None:
    """The US-039 Full-M retrain MUST be temporal (the bug that broke E-a)."""
    assert _is_temporal_kind("tsvit-pheno-fullm") is True


def test_all_temporal_kinds() -> None:
    """Every temporal architecture keeps the time series."""
    for kind in ("tsvit", "tsvit-pheno", "tsvit-pheno-fullm", "utae", "anysat"):
        assert _is_temporal_kind(kind) is True, kind


def test_dense_kinds_collapse_time() -> None:
    """Pixel-only architectures collapse the time axis (median frame)."""
    for kind in ("unet", "deeplabv3plus", "segformer"):
        assert _is_temporal_kind(kind) is False, kind
