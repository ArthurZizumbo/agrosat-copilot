"""Zenodo deposition metadata builder for AgroMind-IT/ES (US-068).

Builds the ``.zenodo.json`` deposition metadata for the bilingual benchmark
(title, creators, CC-BY-4.0 license, keywords, eval-only description) and writes
it next to the dataset. It does NOT perform the HTTP upload: that needs the
sponsor's Zenodo token (blocker B3 in ``docs/blockers/epic11-notas.md``). The
metadata structure IS exercised by the schema test so it is verifiably valid.

Project conventions: identifiers / docstrings in English; the deposition
description (reader-facing) is bilingual Italian / Spanish / English; full type
hints; ``structlog`` (never ``print``); no emojis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from ml.eval.agromind_it_es.schema import QuestionFamily

logger = structlog.get_logger(__name__)

__all__ = [
    "DATASET_LICENSE",
    "DATASET_TITLE",
    "build_zenodo_metadata",
    "write_zenodo_metadata",
]

#: Target license (Zenodo SPDX-style identifier). CC-BY-4.0 per the US-068 AC.
DATASET_LICENSE: str = "cc-by-4.0"

#: Canonical dataset title used in the deposition and the README.
DATASET_TITLE: str = "AgroMind-IT/ES: a bilingual Italian/Spanish agricultural VQA benchmark"

#: Default creators of the deposition. Edit before the real upload (B3) to the
#: actual reviewer roster (Scuola Sant'Anna for Italian, team member for
#: Spanish); the structure is what the test pins.
_DEFAULT_CREATORS: tuple[dict[str, str], ...] = (
    {"name": "AgroSatCopilot Team", "affiliation": "Tecnologico de Monterrey"},
    {"name": "Scuola Superiore Sant'Anna", "affiliation": "Scuola Superiore Sant'Anna"},
)


def _description() -> str:
    """Build the trilingual eval-only deposition description.

    Returns:
        An HTML-light description string stating the eval-only nature, the
        500-pair bilingual scope, the ten families and the AlphaEarth/Sentinel-2
        grounding -- in Italian, Spanish and English.
    """
    families = ", ".join(family.value for family in QuestionFamily)
    return (
        "<p><strong>AgroMind-IT/ES</strong> es un benchmark bilingue "
        "italiano/espanol de 500 pares de preguntas y respuestas agricolas "
        "(250 it + 250 es) sobre imagenes Sentinel-2 reales de Italia, que "
        "cubre las diez familias de preguntas del copiloto: "
        f"{families}. "
        "Es estrictamente <strong>eval-only</strong>: no existe particion de "
        "entrenamiento; ajustar (fine-tune) un modelo sobre el seria fuga de "
        "datos (leakage), igual que en el AgroMind original.</p>"
        "<p><em>AgroMind-IT/ES e un benchmark bilingue italiano/spagnolo di 500 "
        "coppie domanda-risposta agricole su immagini Sentinel-2 reali "
        "dell'Italia. E rigorosamente eval-only: nessuna partizione di "
        "addestramento.</em></p>"
        "<p>AgroMind-IT/ES is a bilingual Italian/Spanish benchmark of 500 "
        "agricultural QA pairs over real Sentinel-2 imagery of Italy, covering "
        "ten copilot question families. It is strictly eval-only: there is no "
        "training split and fine-tuning on it constitutes leakage.</p>"
    )


def build_zenodo_metadata(
    *,
    version: str = "1.0.0",
    creators: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build the Zenodo deposition metadata dict.

    Args:
        version: Dataset version string for the deposition.
        creators: Optional creators list; defaults to the reviewer roster
            placeholder (edit before the real upload, B3).

    Returns:
        The ``{"metadata": {...}}`` deposition payload (Zenodo schema): an
        ``upload_type`` of ``dataset``, the CC-BY-4.0 license, keywords and the
        trilingual eval-only description.
    """
    resolved_creators = (
        list(creators) if creators is not None else [dict(c) for c in _DEFAULT_CREATORS]
    )
    return {
        "metadata": {
            "upload_type": "dataset",
            "title": DATASET_TITLE,
            "version": version,
            "language": "ita",
            "license": DATASET_LICENSE,
            "access_right": "open",
            "creators": resolved_creators,
            "keywords": [
                "agriculture",
                "remote sensing",
                "Sentinel-2",
                "visual question answering",
                "Italian",
                "Spanish",
                "crop classification",
                "eval-only benchmark",
                "AlphaEarth",
            ],
            "description": _description(),
            "notes": (
                "Eval-only benchmark (no training split). Derived from a "
                "Gemini-2.5-pro seed over real Sentinel-2 imagery of Italy, "
                "validated by native Italian (Scuola Sant'Anna) and Spanish "
                "(team) reviewers. Schema-compatible with the original AgroMind."
            ),
        }
    }


def write_zenodo_metadata(path: Path, *, version: str = "1.0.0") -> Path:
    """Write the deposition metadata to ``path`` as UTF-8 JSON.

    Args:
        path: Destination ``.zenodo.json`` path (parent dirs are created).
        version: Dataset version string.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_zenodo_metadata(version=version)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("agromind_it_es_zenodo_metadata_written", path=str(path), version=version)
    return path
