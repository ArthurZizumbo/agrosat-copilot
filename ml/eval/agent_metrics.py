"""Pure scoring metrics for the AgroSatCopilot agent benchmark (US-049).

This module holds the side-effect-free metric functions consumed by the agent
benchmark harness (``ml/eval/agent_bench.py``). They are deliberately decoupled
from the orchestrator (DRY, plan Section 5) so they can be unit-tested in
isolation against known cases without any network or LLM call.

Two benchmarks are scored here:

- **AgroMind** (multiple-choice QA, answer is a letter ``A``-``Z`` -- the real
  subset uses up to ten options ``A``-``J``, not just ``A``-``D``, plus open
  numeric/text golds with no options at all):
  :func:`exact_match`, :func:`f1_squad`, :func:`bertscore_f1`,
  :func:`hallucination_rate` (LLM-as-judge) and :func:`tool_call_accuracy`
  (for the copilot tool-use queries).
- **GeoAnalystBench** (plan-and-react, free-text workflow + Python code):
  :func:`codebleu_score` and :func:`workflow_semantic_similarity`.

Metric provenance and approximations (documented per plan Section 6 / Risks):

- ``bertscore_f1`` is a **semantic proxy**, NOT the canonical BERTScore. There
  is no ``bert-score`` package in the deps, so we embed each sentence with
  ``sentence-transformers/all-MiniLM-L6-v2`` and use mean cosine similarity as a
  semantic-overlap surrogate. It correlates with answer quality but is not
  token-level BERTScore; it must be reported as a proxy.
- ``codebleu_score`` is a **simplified CodeBLEU**. There is no ``codebleu``
  package in the deps, so we combine (a) an n-gram BLEU (NLTK
  ``sentence_bleu`` over Python tokens) with (b) a Python keyword/identifier set
  overlap. It omits the weighted-AST and data-flow components of canonical
  CodeBLEU and must be reported as an approximation.
- ``hallucination_rate`` uses DeepEval (LLM-as-judge) only when an injectable
  ``judge`` is provided. With no judge configured it returns ``float('nan')``
  (reported as ``n/a``) instead of fabricating a score.

Project conventions: pure functions with full type hints and Google-style
docstrings; identifiers and docstrings in English; visible prose elsewhere in
Spanish; ``structlog`` for logging (never ``print``); no emojis; edge cases
(empty inputs) collapse to ``0.0`` unless documented otherwise.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from collections.abc import Sequence

    import numpy as np
    from sentence_transformers import SentenceTransformer

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_SENTENCE_MODEL",
    "HallucinationJudge",
    "bertscore_f1",
    "codebleu_score",
    "exact_match",
    "f1_squad",
    "hallucination_rate",
    "tool_call_accuracy",
    "workflow_semantic_similarity",
]

#: Sentence encoder used for the semantic proxies (same frozen model the rest of
#: the project standardises on, see ``ml/features/phenology_description.py``).
DEFAULT_SENTENCE_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

#: Python keywords + common builtins used by the simplified CodeBLEU keyword
#: overlap component. Kept explicit (not ``keyword.kwlist``) so the set is stable
#: across interpreter versions and includes the geospatial-relevant builtins.
_PYTHON_KEYWORDS: frozenset[str] = frozenset(
    {
        "and", "as", "assert", "async", "await", "break", "class", "continue",
        "def", "del", "elif", "else", "except", "finally", "for", "from",
        "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or",
        "pass", "raise", "return", "try", "while", "with", "yield", "True",
        "False", "None", "self", "print", "len", "range", "list", "dict", "set",
        "tuple", "open", "int", "float", "str", "bool",
    }
)

#: Lazy module-level cache for the sentence encoder so the (heavy) model is
#: loaded at most once per process and tests can monkeypatch
#: ``sentence_transformers.SentenceTransformer`` before first use.
_sentence_model: SentenceTransformer | None = None

#: Token pattern for SQuAD-style F1 (alphanumeric word tokens).
_WORD_RE = re.compile(r"[a-z0-9]+")

#: Token pattern for code (identifiers, numbers and single non-space symbols),
#: a lightweight tokenizer that avoids a hard dependency on ``tokenize`` which
#: raises on syntactically incomplete model output.
_CODE_TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d+\.?\d*|[^\s\w]")

# Multiple-choice option labels recognised in AgroMind answers, tried in order:
# an isolated bare letter (``"B"``), then a labelled form (``"Answer: C"``,
# ``"(D)"``, ``"option A"``), then any standalone letter token. Plain prose words
# like "Answer" must not leak their leading ``A`` into the match.
#
# The real subset has items with up to ten options (``A``-``J``), so the letter
# class spans ``A``-``Z`` rather than the historical ``A``-``D`` (B-5). Callers
# may pass ``valid_letters`` (derived from ``item.options``) to constrain the
# match to the labels that actually exist for the item; the standalone fallback
# is then restricted to that set so a stray capital in prose (e.g. "GIS") is not
# mistaken for a choice when the item only offers ``A``-``C``.
_BARE_LETTER_RE = re.compile(r"^\s*\(?([A-Z])\)?\s*$")
_LABELLED_LETTER_RE = re.compile(
    r"(?:answer|option|choice|respuesta|opcion)\s*[:\-\)\.]?\s*\(?([A-Z])\b",
    re.IGNORECASE,
)
_STANDALONE_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])")


@runtime_checkable
class HallucinationJudge(Protocol):
    """Injectable LLM-as-judge contract for :func:`hallucination_rate`.

    Implemented over DeepEval (``HallucinationMetric``) in the orchestrator, and
    mocked in tests. A judge scores a single sample and returns a hallucination
    score in ``[0.0, 1.0]`` (1.0 == fully hallucinated).
    """

    def score(self, sample: dict[str, Any]) -> float:
        """Return the hallucination score in ``[0.0, 1.0]`` for one sample."""
        ...


def _normalize_text(text: str) -> str:
    """Lowercase, strip articles/punctuation and collapse whitespace.

    Mirrors the SQuAD normalisation used for exact-match and token-F1 so both
    metrics agree on what counts as equal.

    Args:
        text: Raw prediction or gold string.

    Returns:
        The normalised, whitespace-collapsed lowercase string.
    """
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_choice_letter(
    text: str, valid_letters: frozenset[str] | None = None
) -> str | None:
    """Extract a single choice letter (``A``-``Z``) from a model answer.

    Tries, in order: an isolated bare letter (``"B"`` / ``"(C)"``), a labelled
    form (``"Answer: F"``, ``"option D"``), then any standalone letter token.
    This ordering keeps prose words such as ``"Answer"`` from leaking their
    leading ``A``. Returns ``None`` when no letter is present so callers can fall
    back to text comparison.

    The real AgroMind subset has items with up to ten options (``A``-``J``), so
    the recogniser spans the full ``A``-``Z`` range (B-5). When ``valid_letters``
    is given (the labels that exist for the item, e.g. ``frozenset("ABC")``) the
    match is constrained to that set: a bare/labelled letter outside the set is
    rejected and the standalone fallback only fires on an in-set letter, so a
    stray capital from prose (``"GIS"``) is never mistaken for a choice.

    Args:
        text: Raw answer string.
        valid_letters: Optional set of uppercase labels that actually exist for
            the item; ``None`` accepts any ``A``-``Z`` letter.

    Returns:
        The uppercase choice letter, or ``None`` if none is found.
    """
    stripped = text.strip()

    def _accept(letter: str) -> str | None:
        upper = letter.upper()
        if valid_letters is not None and upper not in valid_letters:
            return None
        return upper

    bare = _BARE_LETTER_RE.match(stripped)
    if bare is not None:
        accepted = _accept(bare.group(1))
        if accepted is not None:
            return accepted
    labelled = _LABELLED_LETTER_RE.search(stripped)
    if labelled is not None:
        accepted = _accept(labelled.group(1))
        if accepted is not None:
            return accepted
    # Standalone fallback: scan every standalone capital and return the first one
    # allowed by ``valid_letters`` (or the first capital when unconstrained).
    for match in _STANDALONE_LETTER_RE.finditer(stripped):
        accepted = _accept(match.group(1))
        if accepted is not None:
            return accepted
    return None


def exact_match(
    pred: str, gold: str, valid_letters: frozenset[str] | None = None
) -> float:
    """Exact-match score for an AgroMind multiple-choice answer.

    AgroMind golds are mostly a choice letter (``A``-``J`` in the real subset),
    but the subset also carries open numeric/text golds with no options. When
    both strings reduce to a choice letter they are compared as letters;
    otherwise they fall back to SQuAD-normalised string equality (so a free
    numeric/text answer like ``"10"`` still scores correctly).

    Passing ``valid_letters`` (the item's option labels) keeps the letter path
    from firing on open-ended items and from misreading a stray capital, while a
    correct answer like ``"F"`` or ``"The answer is F"`` still scores ``1.0``
    when ``F`` is a real option (B-5).

    Args:
        pred: Model prediction (may be a bare letter or contain extra prose).
        gold: Gold answer (a letter for multiple-choice items, free text/number
            for open ones).
        valid_letters: Optional set of the item's option labels; constrains the
            letter match to the labels that exist (``None`` accepts any letter).

    Returns:
        ``1.0`` on match, ``0.0`` otherwise. Empty inputs score ``0.0``.
    """
    if not pred or not gold:
        return 0.0
    pred_letter = _extract_choice_letter(pred, valid_letters)
    gold_letter = _extract_choice_letter(gold, valid_letters)
    if pred_letter is not None and gold_letter is not None:
        return 1.0 if pred_letter == gold_letter else 0.0
    return 1.0 if _normalize_text(pred) == _normalize_text(gold) else 0.0


def f1_squad(pred: str, gold: str) -> float:
    """SQuAD-style token-overlap F1 between a prediction and a gold answer.

    Computes precision/recall over the multiset of normalised word tokens and
    returns their harmonic mean ``2 * p * r / (p + r)``. This is the standard
    extractive-QA F1, useful for the free-text portion of AgroMind answers.

    Args:
        pred: Model prediction string.
        gold: Gold answer string.

    Returns:
        Token-overlap F1 in ``[0.0, 1.0]``. Empty inputs (or no token overlap)
        score ``0.0``; two empty strings also score ``0.0`` by convention.
    """
    pred_tokens = _WORD_RE.findall(_normalize_text(pred))
    gold_tokens = _WORD_RE.findall(_normalize_text(gold))
    if not pred_tokens or not gold_tokens:
        return 0.0

    gold_counts: dict[str, int] = {}
    for tok in gold_tokens:
        gold_counts[tok] = gold_counts.get(tok, 0) + 1
    pred_counts: dict[str, int] = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1
    num_same = 0
    for tok, count in pred_counts.items():
        num_same += min(count, gold_counts.get(tok, 0))
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _get_sentence_model() -> SentenceTransformer:
    """Return the cached sentence encoder, loading it lazily on first use.

    The import and instantiation are deferred so importing this module stays
    cheap and CI can monkeypatch ``sentence_transformers.SentenceTransformer``
    before the model is materialised.

    Returns:
        The shared :class:`~sentence_transformers.SentenceTransformer` instance.
    """
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("loading_sentence_model", model=DEFAULT_SENTENCE_MODEL)
        _sentence_model = SentenceTransformer(DEFAULT_SENTENCE_MODEL)
    return _sentence_model


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors, clamped to ``[0.0, 1.0]``.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        The non-negative cosine similarity (negatives are clamped to ``0.0``
        since semantic-overlap scores are reported as non-negative).
    """
    import numpy as np

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    sim = float(np.dot(a, b) / denom)
    return max(0.0, min(1.0, sim))


def bertscore_f1(preds: Sequence[str], golds: Sequence[str]) -> float:
    """Semantic-similarity proxy for BERTScore F1 over paired strings.

    PROXY, not canonical BERTScore (no ``bert-score`` package in deps): each
    prediction and gold is embedded with ``all-MiniLM-L6-v2`` and scored by
    cosine similarity; the returned value is the mean cosine over all pairs.
    Reported as a semantic proxy, not token-level BERTScore.

    Args:
        preds: Predicted answer strings.
        golds: Gold answer strings, aligned 1:1 with ``preds``.

    Returns:
        Mean cosine similarity in ``[0.0, 1.0]``. Returns ``0.0`` for empty
        inputs or length mismatch (logged as a warning).
    """
    if not preds or not golds:
        return 0.0
    if len(preds) != len(golds):
        logger.warning(
            "bertscore_length_mismatch", n_preds=len(preds), n_golds=len(golds)
        )
        return 0.0

    model = _get_sentence_model()
    pred_emb = model.encode(list(preds), normalize_embeddings=False)
    gold_emb = model.encode(list(golds), normalize_embeddings=False)
    scores = [_cosine(pred_emb[i], gold_emb[i]) for i in range(len(preds))]
    return sum(scores) / len(scores) if scores else 0.0


def tool_call_accuracy(
    pred_calls: Sequence[str], gold_calls: Sequence[str]
) -> float:
    """Fraction of expected tool calls present in the prediction.

    For copilot plan-and-react queries: measures recall of the gold tool-call
    set (tool names) within the predicted tool-call set. Order and duplicates
    are ignored; matching is case-insensitive on the stripped tool name.

    Args:
        pred_calls: Tool names the model actually invoked.
        gold_calls: Tool names that were expected.

    Returns:
        ``|gold ∩ pred| / |gold|`` in ``[0.0, 1.0]``. Returns ``1.0`` when no
        tool call is expected (vacuously satisfied) and ``0.0`` when a call is
        expected but none predicted.
    """
    gold_set = {c.strip().lower() for c in gold_calls if c.strip()}
    if not gold_set:
        return 1.0
    pred_set = {c.strip().lower() for c in pred_calls if c.strip()}
    matched = len(gold_set & pred_set)
    return matched / len(gold_set)


def hallucination_rate(
    samples: Sequence[dict[str, Any]], judge: HallucinationJudge | None = None
) -> float:
    """Mean hallucination rate over samples via an injectable LLM-as-judge.

    Each sample is a dict ``{"input", "actual_output", "context"}`` (the
    DeepEval test-case shape). When ``judge`` is provided every sample is scored
    and the mean is returned. When ``judge`` is ``None`` (no LLM-as-judge
    configured) this returns ``float('nan')`` so the caller can render it as
    ``n/a`` rather than a fabricated number, per the eval-only honesty policy.

    Args:
        samples: Sequence of ``{input, actual_output, context}`` dicts.
        judge: Injectable judge implementing :class:`HallucinationJudge`. When
            ``None`` the metric is reported as not-available (NaN).

    Returns:
        Mean hallucination score in ``[0.0, 1.0]`` when a judge is given; an
        empty sample list scores ``0.0``; ``float('nan')`` when no judge.
    """
    if not samples:
        return 0.0
    if judge is None:
        logger.info("hallucination_rate_no_judge", reason="judge_not_configured")
        return math.nan

    scores: list[float] = []
    for sample in samples:
        try:
            scores.append(float(judge.score(sample)))
        except Exception as exc:  # noqa: BLE001 - judge errors must not crash eval
            logger.warning("hallucination_judge_error", error=str(exc))
    if not scores:
        return math.nan
    return sum(scores) / len(scores)


def _tokenize_code(code: str) -> list[str]:
    """Tokenise a (possibly incomplete) Python snippet for scoring.

    Uses a regex tokenizer instead of :mod:`tokenize` because model output may
    be syntactically incomplete, which would make the stdlib tokenizer raise.

    Args:
        code: Python source string.

    Returns:
        The list of token strings (identifiers, numbers, single symbols).
    """
    return _CODE_TOKEN_RE.findall(code)


def _ngram_bleu(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    """Sentence-level n-gram BLEU (up to 4-grams) over token lists.

    Wraps NLTK ``sentence_bleu`` with a smoothing function so short snippets do
    not collapse to ``0.0`` on a single missing higher-order n-gram.

    Args:
        pred_tokens: Predicted code tokens.
        ref_tokens: Reference code tokens.

    Returns:
        BLEU score in ``[0.0, 1.0]``; ``0.0`` if either side is empty.
    """
    if not pred_tokens or not ref_tokens:
        return 0.0
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    smoothing = SmoothingFunction().method1
    weights = (0.25, 0.25, 0.25, 0.25)
    return float(
        sentence_bleu(
            [ref_tokens], pred_tokens, weights=weights, smoothing_function=smoothing
        )
    )


def _keyword_overlap(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    """Jaccard overlap of Python keywords + identifiers between two snippets.

    Approximates CodeBLEU's syntactic/keyword-match signal without an AST: the
    intersection-over-union of the token sets, biased toward Python keywords and
    identifiers (symbols are ignored).

    Args:
        pred_tokens: Predicted code tokens.
        ref_tokens: Reference code tokens.

    Returns:
        Jaccard similarity in ``[0.0, 1.0]``; ``0.0`` when both sets are empty.
    """
    def _ids(tokens: list[str]) -> set[str]:
        return {
            t for t in tokens if t in _PYTHON_KEYWORDS or re.fullmatch(r"[A-Za-z_]\w*", t)
        }

    pred_set = _ids(pred_tokens)
    ref_set = _ids(ref_tokens)
    if not pred_set and not ref_set:
        return 0.0
    union = pred_set | ref_set
    if not union:
        return 0.0
    return len(pred_set & ref_set) / len(union)


def codebleu_score(
    pred_code: str, ref_code: str, *, bleu_weight: float = 0.5
) -> float:
    """Simplified CodeBLEU between predicted and reference Python code.

    APPROXIMATION, not canonical CodeBLEU (no ``codebleu`` package in deps):
    a weighted average of (a) n-gram BLEU over code tokens and (b) Python
    keyword/identifier set overlap. It omits the weighted-AST and data-flow
    components of the published CodeBLEU and must be reported as an
    approximation.

    Args:
        pred_code: Generated Python code.
        ref_code: Reference Python code (GeoAnalystBench ``CodeString``).
        bleu_weight: Weight on the BLEU component in ``[0.0, 1.0]``; the
            keyword-overlap component gets ``1 - bleu_weight``.

    Returns:
        Combined score in ``[0.0, 1.0]``. Empty inputs score ``0.0``.
    """
    if not pred_code or not pred_code.strip():
        return 0.0
    if not ref_code or not ref_code.strip():
        return 0.0
    weight = max(0.0, min(1.0, bleu_weight))
    pred_tokens = _tokenize_code(pred_code)
    ref_tokens = _tokenize_code(ref_code)
    bleu = _ngram_bleu(pred_tokens, ref_tokens)
    keyword = _keyword_overlap(pred_tokens, ref_tokens)
    return weight * bleu + (1.0 - weight) * keyword


def _steps_to_text(steps: str | Sequence[str]) -> str:
    """Coerce a workflow (string or list of steps) into a single string.

    Args:
        steps: A workflow as a single string or a sequence of step strings.

    Returns:
        A newline-joined string (empty string for empty input).
    """
    if isinstance(steps, str):
        return steps.strip()
    return "\n".join(str(s).strip() for s in steps if str(s).strip())


def workflow_semantic_similarity(
    pred_steps: str | Sequence[str], gold_steps: str | Sequence[str]
) -> float:
    """Semantic similarity between a generated and a reference workflow.

    Embeds both workflows (the GeoAnalystBench ``Human Designed Workflow`` is
    the reference) with ``all-MiniLM-L6-v2`` and returns their cosine
    similarity. Accepts either a single string or a list of numbered steps on
    either side.

    Args:
        pred_steps: The model-generated workflow (string or list of steps).
        gold_steps: The reference workflow (string or list of steps).

    Returns:
        Cosine similarity in ``[0.0, 1.0]``. Empty inputs score ``0.0``.
    """
    pred_text = _steps_to_text(pred_steps)
    gold_text = _steps_to_text(gold_steps)
    if not pred_text or not gold_text:
        return 0.0
    model = _get_sentence_model()
    emb = model.encode([pred_text, gold_text], normalize_embeddings=False)
    return _cosine(emb[0], emb[1])
