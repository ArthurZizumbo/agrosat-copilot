"""AgroMind-IT/ES bilingual benchmark scaffold (US-068, EPIC 11 Paper Track).

This package holds the *infrastructure* of the original bilingual Italian/Spanish
AgroMind-IT/ES benchmark (target 500 Q&A pairs: 250 ``it`` + 250 ``es``) used to
evaluate the conversational copilot. It ships:

- :mod:`ml.eval.agromind_it_es.schema` -- the JSONL record schema
  (:class:`~ml.eval.agromind_it_es.schema.QAItem`), the 10 copilot question
  families (:class:`~ml.eval.agromind_it_es.schema.QuestionFamily`), the
  ``to_agromind_item`` compatibility bridge to the real
  :class:`ml.eval.agent_bench.AgroMindItem`, and the eval-only JSONL
  round-trip (``dump_jsonl`` / ``load_jsonl``) that REJECTS any train mark.
- :mod:`ml.eval.agromind_it_es.generate_seed` -- the seed generator scaffold:
  Gemini 2.5-pro prompt templates per family x language, a ``SeedGenerator``
  that reads credentials via ``get_settings()`` (never ``os.environ``), and a
  ``dry_run`` mode that emits the per-family plan WITHOUT calling the API (the
  autonomous mode of US-068).
- :mod:`ml.eval.agromind_it_es.zenodo_metadata` -- the Zenodo metadata builder
  (CC-BY-4.0, eval-only description); it does NOT upload (that is a blocker).

EVAL-ONLY BY DESIGN: AgroMind ships no train split, so any fine-tune over it (or
over this bilingual extension) would be leakage. The schema carries no ``split``
field and the loader rejects any record that smuggles a train mark in.

Project conventions: identifiers and docstrings in English (Google style),
visible prose (prompts, README, CLI) in Italian/Spanish per the dataset; full
type hints; ``structlog`` (never ``print``); no emojis.
"""

from __future__ import annotations

from ml.eval.agromind_it_es.schema import (
    QAItem,
    QuestionFamily,
    dump_jsonl,
    load_jsonl,
    to_agromind_item,
    validate_record,
)

__all__ = [
    "QAItem",
    "QuestionFamily",
    "dump_jsonl",
    "load_jsonl",
    "to_agromind_item",
    "validate_record",
]
