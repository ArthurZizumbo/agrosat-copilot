"""CPU/offline tests del pre-encoding de captions y el collate wrapper (T-wiring).

Cubre ``ml/farslip/caption_encoder.py``: el pre-encoding MiniLM (encoder mockeado,
sin descargar el modelo ni red) y el collate wrapper que inyecta ``caption_cls``
``(B, 384)`` al batch en el MISMO orden que ``patch_ids`` (lo que activa L_glo).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.farslip.caption_encoder import (
    MINILM_DIM,
    encode_captions_minilm,
    make_caption_collate,
)


class _FakeST:
    """Stand-in deterministico de SentenceTransformer (sin red ni modelo real)."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device

    def encode(self, texts, **kwargs):
        # Embedding determinista por longitud del texto, dim MINILM_DIM.
        rng = np.random.default_rng(0)
        base = rng.random((len(texts), MINILM_DIM)).astype(np.float32)
        for i, t in enumerate(texts):
            base[i, 0] = float(len(t))
        return base


@pytest.fixture(autouse=True)
def _patch_sentence_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    fake_mod = types.ModuleType("sentence_transformers")
    fake_mod.SentenceTransformer = _FakeST  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)


def test_encode_captions_minilm_shape_and_keys() -> None:
    captions = {"100": "una pradera", "200": "vinedos en hileras", "300": "maiz"}
    emb = encode_captions_minilm(captions)
    assert set(emb) == {"100", "200", "300"}
    for vec in emb.values():
        assert isinstance(vec, torch.Tensor)
        assert vec.shape == (MINILM_DIM,)
        assert vec.dtype == torch.float32


def test_make_caption_collate_injects_caption_cls_in_order() -> None:
    captions = {"100": "aaa", "200": "bbbb", "300": "cc"}
    emb = encode_captions_minilm(captions)

    def _base_collate(items):
        return {
            "images": torch.zeros((len(items), 4, 8, 8)),
            "patch_ids": [it["patch_id"] for it in items],
            "captions": [it["caption"] for it in items],
        }

    collate = make_caption_collate(_base_collate, emb)
    items = [
        {"patch_id": "200", "caption": "bbbb"},
        {"patch_id": "100", "caption": "aaa"},
    ]
    batch = collate(items)
    assert "caption_cls" in batch
    assert batch["caption_cls"].shape == (2, MINILM_DIM)
    # caption_cls[i] corresponde al patch_id[i] (mismo orden que images).
    assert torch.equal(batch["caption_cls"][0], emb["200"])
    assert torch.equal(batch["caption_cls"][1], emb["100"])


def test_make_caption_collate_raises_on_missing_caption() -> None:
    emb = encode_captions_minilm({"100": "aaa"})

    def _base_collate(items):
        return {"patch_ids": [it["patch_id"] for it in items]}

    collate = make_caption_collate(_base_collate, emb)
    with pytest.raises(KeyError, match="no pre-encoded caption"):
        collate([{"patch_id": "999"}])
