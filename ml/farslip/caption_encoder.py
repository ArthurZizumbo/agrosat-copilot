"""Pre-encoding de captions ``L_glo`` a embeddings MiniLM (US-036-a v2, wiring).

El entrenamiento fiel de FarSLIP (``step_faithful_v2``) necesita el ``caption_cls``
(embedding por patch) para activar la perdida global imagen-texto ``L_glo`` (eq.
1-2). El ``collate_region_batch`` entrega las captions como ``list[str]``; este
modulo las **pre-encodea UNA sola vez** con el mismo encoder de texto del proyecto
(``all-MiniLM-L6-v2``, 384-dim, el de los prototipos US-033) y expone un
**collate wrapper** que inyecta ``caption_cls`` ``(B, 384)`` al batch usando los
``patch_ids``. El trainer luego liftea 384 -> 768 via ``_proto_to_clip_proj``.

Pre-encodear una vez (no por epoca) es clave: el encoder de texto queda fuera del
loop, sin coste por step y sin re-cargar el modelo. Convenciones: ``torch`` solo
en el borde, ``structlog``, type hints, docstrings en ingles.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import structlog
import torch

logger = structlog.get_logger(__name__)

#: Text encoder shared with US-033 prototypes (384-dim MiniLM).
SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_DIM = 384


def encode_captions_minilm(
    captions: Mapping[str, str],
    *,
    model_name: str = SENTENCE_MODEL,
    batch_size: int = 64,
    device: str | None = None,
) -> dict[str, torch.Tensor]:
    """Encodes every caption once into a MiniLM-384 embedding map.

    Args:
        captions: ``{patch_id: caption_glo}`` map (the loaded captions parquet).
        model_name: SentenceTransformer model id (default the US-033 MiniLM).
        batch_size: encoding batch size.
        device: torch device for the encoder; ``None`` lets the library decide.

    Returns:
        ``{patch_id: tensor(384,)}`` float32 CPU embeddings, normalized as the
        SentenceTransformer returns them.
    """
    from sentence_transformers import SentenceTransformer

    pids = [str(pid) for pid in captions]
    texts = [str(captions[pid]) for pid in pids]
    model = SentenceTransformer(model_name, device=device)
    matrix = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    embeddings = {
        pid: torch.from_numpy(matrix[i]).to(torch.float32)
        for i, pid in enumerate(pids)
    }
    logger.info(
        "captions_preencoded",
        n_captions=len(embeddings),
        dim=int(matrix.shape[1]) if len(matrix) else 0,
        model=model_name,
    )
    return embeddings


def make_caption_collate(
    base_collate: Callable[[list[dict[str, Any]]], dict[str, Any]],
    caption_embeddings: Mapping[str, torch.Tensor],
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Wraps a collate so the batch carries ``caption_cls`` ``(B, 384)``.

    The wrapper runs ``base_collate`` (e.g.
    :func:`ml.farslip.region_category_dataset.collate_region_batch`) and stacks
    the pre-encoded embedding of every batch ``patch_id`` into ``caption_cls``,
    in the SAME order as ``patch_ids`` / ``images`` (so the global InfoNCE pairs
    image[i] with caption[i]). This is what activates ``L_glo`` in
    ``step_faithful_v2`` (without it the batch has no ``caption_cls`` and L_glo
    is skipped, leaving only MPCL).

    Args:
        base_collate: the underlying collate producing ``patch_ids``.
        caption_embeddings: ``{patch_id: tensor(384,)}`` from
            :func:`encode_captions_minilm`.

    Returns:
        A collate callable returning the base batch plus ``caption_cls``.

    Raises:
        KeyError: if a batch ``patch_id`` has no pre-encoded caption (fail fast,
            same contract as ``_require_captions_for_dataset``).
    """

    def _collate(items: list[dict[str, Any]]) -> dict[str, Any]:
        batch = base_collate(items)
        patch_ids: Sequence[str] = batch["patch_ids"]
        missing = [pid for pid in patch_ids if pid not in caption_embeddings]
        if missing:
            raise KeyError(
                f"{len(missing)} batch patch_ids have no pre-encoded caption "
                f"(e.g. {missing[:5]}); pre-encode every caption before training."
            )
        batch["caption_cls"] = torch.stack(
            [caption_embeddings[pid] for pid in patch_ids], dim=0
        )
        return batch

    return _collate


__all__ = ["MINILM_DIM", "SENTENCE_MODEL", "encode_captions_minilm", "make_caption_collate"]
