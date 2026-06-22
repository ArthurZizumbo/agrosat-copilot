"""Streamlit human-review app for the AgroMind-IT/ES seed (US-068).

Loads the Gemini-2.5-pro seed JSONL, shows each pair (question, options, gold,
and the Sentinel-2 image when present) and lets a native reviewer Accept / Edit
/ Reject it, logging the reviewer id + language + decision. Accepted / edited
pairs are marked ``reviewed=True``, ``source="human-edited"`` and exported to
the published benchmark JSONL. This materialises the eval-only split: only the
human-validated pairs become the benchmark.

The app is the CODE deliverable of US-068. Running it WITH the real native
reviewers (Italian via Scuola Sant'Anna, Spanish via a team member) is blocker
B2 in ``docs/blockers/epic11-notas.md``.

``streamlit`` is an optional dependency (poetry group ``paper``). The module is
import-safe without it: the heavy UI body only runs under
``streamlit run`` (``__main__``), so the test can import it (smoke) with no
``streamlit`` installed.

Project conventions: identifiers / docstrings in English; UI labels in Spanish
(the team member's UI); the pair content stays in its native language; full
type hints; ``structlog`` (never ``print``); no emojis.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from ml.eval.agromind_it_es.generate_seed import DEFAULT_SEED_PATH
from ml.eval.agromind_it_es.schema import QAItem, dump_jsonl, load_jsonl

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_REVIEWED_PATH",
    "accept_item",
    "reject_item",
    "run_app",
]

#: Default destination of the human-validated, published benchmark split.
DEFAULT_REVIEWED_PATH: Path = Path("data/benchmark/agromind_it_es/agromind_it_es_500.jsonl")


def accept_item(item: QAItem, reviewer: str, *, edited: bool = False) -> QAItem:
    """Mark an item as accepted (or edited) by a native reviewer.

    Args:
        item: The seed item under review.
        reviewer: Identifier of the human reviewer.
        edited: Whether the reviewer edited the pair before accepting.

    Returns:
        A new :class:`QAItem` with ``reviewed=True``, the reviewer recorded and
        ``source="human-edited"`` (the only provenance that enters the
        published benchmark).
    """
    accepted = QAItem(
        item_id=item.item_id,
        category=item.category,
        lang=item.lang,
        question=item.question,
        options=dict(item.options),
        answer=item.answer,
        image=item.image,
        is_multimodal=bool(item.image),
        reviewed=True,
        reviewer=reviewer,
        source="human-edited",
    ).with_derived_flags()
    logger.info(
        "agromind_it_es_item_accepted",
        item_id=item.item_id,
        reviewer=reviewer,
        edited=edited,
    )
    return accepted


def reject_item(item: QAItem, reviewer: str) -> None:
    """Log a rejected item (it never enters the published benchmark).

    Args:
        item: The rejected seed item.
        reviewer: Identifier of the human reviewer.
    """
    logger.info(
        "agromind_it_es_item_rejected",
        item_id=item.item_id,
        reviewer=reviewer,
    )


def export_accepted(items: Sequence[QAItem], path: Path = DEFAULT_REVIEWED_PATH) -> int:
    """Export the accepted items to the published benchmark JSONL.

    Args:
        items: The accepted (human-edited) items.
        path: Destination JSONL path for the published split.

    Returns:
        The number of records written.
    """
    return dump_jsonl(items, path)


def run_app(seed_path: Path = DEFAULT_SEED_PATH) -> None:  # pragma: no cover - UI
    """Run the Streamlit review UI (entry point under ``streamlit run``).

    Imported lazily so the module stays import-safe without ``streamlit``
    installed (the test imports the module but never calls this).

    Args:
        seed_path: Path to the seed JSONL to review.
    """
    import streamlit as st

    st.set_page_config(page_title="AgroMind-IT/ES — Revision", layout="wide")
    st.title("AgroMind-IT/ES — Revision humana del seed")
    st.caption(
        "Benchmark eval-only (sin particion de entrenamiento). Acepta, edita o "
        "rechaza cada par; solo los aceptados entran al benchmark publicado."
    )

    reviewer = st.text_input("Revisor (id):", value="")
    if not Path(seed_path).exists():
        st.warning(f"No se encontro el seed en {seed_path}. Genera el seed primero.")
        return

    items = load_jsonl(seed_path)
    if "accepted" not in st.session_state:
        st.session_state.accepted = []
    if "cursor" not in st.session_state:
        st.session_state.cursor = 0

    cursor = st.session_state.cursor
    if cursor >= len(items):
        st.success(f"Revision completa: {len(st.session_state.accepted)} pares aceptados.")
        if st.button("Exportar benchmark aceptado"):
            n = export_accepted(st.session_state.accepted)
            st.info(f"Exportados {n} pares a {DEFAULT_REVIEWED_PATH}.")
        return

    item = items[cursor]
    st.progress((cursor + 1) / len(items), text=f"Par {cursor + 1} / {len(items)}")
    st.write(f"**Familia:** {item.category.value} · **Idioma:** {item.lang}")
    if item.image and Path(item.image).exists():
        st.image(item.image, caption=item.image)
    question = st.text_area("Pregunta:", value=item.question, height=120)
    answer = st.text_input("Respuesta (letra o texto):", value=item.answer)

    col_accept, col_edit, col_reject = st.columns(3)
    if col_accept.button("Aceptar") and reviewer:
        st.session_state.accepted.append(accept_item(item, reviewer))
        st.session_state.cursor += 1
        st.rerun()
    if col_edit.button("Aceptar con edicion") and reviewer:
        edited = QAItem(
            item_id=item.item_id,
            category=item.category,
            lang=item.lang,
            question=question,
            options=dict(item.options),
            answer=answer,
            image=item.image,
        )
        st.session_state.accepted.append(accept_item(edited, reviewer, edited=True))
        st.session_state.cursor += 1
        st.rerun()
    if col_reject.button("Rechazar") and reviewer:
        reject_item(item, reviewer)
        st.session_state.cursor += 1
        st.rerun()


if __name__ == "__main__":  # pragma: no cover - UI shim
    run_app()
