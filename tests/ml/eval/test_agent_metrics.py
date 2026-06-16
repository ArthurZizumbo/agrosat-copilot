"""Unit tests for the pure agent-benchmark metrics (US-049).

Every metric in :mod:`ml.eval.agent_metrics` is side-effect-free, so these tests
run fully offline with zero network and zero LLM calls. The two metrics that
need a heavy resource are isolated behind doubles:

- The DeepEval LLM-as-judge for :func:`hallucination_rate` is a tiny in-test
  stub exposing ``.score(sample) -> float`` (the injectable
  :class:`~ml.eval.agent_metrics.HallucinationJudge` contract).
- The ``sentence-transformers`` encoder behind :func:`bertscore_f1` and
  :func:`workflow_semantic_similarity` is monkeypatched with a deterministic
  fake encoder (``sentence_transformers.SentenceTransformer``) and the module
  cache ``agent_metrics._sentence_model`` is reset, so no real model is loaded.

Conventions: identifiers and docstrings in English; visible prose elsewhere in
Spanish; no emojis; full type hints.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from ml.eval import agent_metrics


class _FakeEncoder:
    """Deterministic stand-in for ``SentenceTransformer`` (no model download).

    Maps each input string to a small fixed vector by a stable hash so that
    identical strings embed identically (cosine ``1.0``) and clearly different
    strings embed to near-orthogonal vectors. Enough to exercise the cosine
    plumbing of :func:`bertscore_f1` / :func:`workflow_semantic_similarity`
    without the real encoder.
    """

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self._dim = 16

    def encode(self, texts: list[str], **_kwargs: Any) -> np.ndarray:
        """Embed a batch of strings to deterministic vectors.

        Args:
            texts: Input strings.

        Returns:
            A ``(len(texts), dim)`` float array; identical strings map to the
            same row.
        """
        rows = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            rows.append(rng.standard_normal(self._dim))
        return np.asarray(rows, dtype=np.float64)


@pytest.fixture
def fake_sentence_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the sentence encoder to a deterministic fake and reset the cache.

    Tests touching the semantic proxies must not load the real MiniLM model.
    """
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeEncoder)
    monkeypatch.setattr(agent_metrics, "_sentence_model", None, raising=False)
    yield
    monkeypatch.setattr(agent_metrics, "_sentence_model", None, raising=False)


# --------------------------------------------------------------------------- #
# exact_match
# --------------------------------------------------------------------------- #


class TestExactMatch:
    """Exact-match scoring for AgroMind multiple-choice answers."""

    def test_bare_letters_equal(self) -> None:
        assert agent_metrics.exact_match("B", "B") == 1.0

    def test_bare_letters_differ(self) -> None:
        assert agent_metrics.exact_match("A", "C") == 0.0

    def test_letter_case_and_parens_normalised(self) -> None:
        assert agent_metrics.exact_match("(c)", "C") == 1.0

    def test_labelled_prediction_against_letter_gold(self) -> None:
        assert agent_metrics.exact_match("Answer: D", "D") == 1.0

    def test_labelled_prediction_wrong_letter(self) -> None:
        assert agent_metrics.exact_match("The answer is B", "C") == 0.0

    def test_prose_answer_word_does_not_leak_leading_a(self) -> None:
        # The literal word "Answer" must not be parsed as choice "A".
        assert agent_metrics.exact_match("Answer", "A") == 0.0

    def test_free_text_falls_back_to_normalised_equality(self) -> None:
        assert agent_metrics.exact_match("the maize field", "Maize  field") == 1.0

    def test_empty_inputs_score_zero(self) -> None:
        assert agent_metrics.exact_match("", "A") == 0.0
        assert agent_metrics.exact_match("A", "") == 0.0

    # B-5: the real subset has items with up to ten options (A-J), so letters
    # beyond the historical A-D must score correctly, both bare and in prose.
    def test_bare_letter_e_to_j_matches(self) -> None:
        assert agent_metrics.exact_match("F", "F") == 1.0
        assert agent_metrics.exact_match("J", "J") == 1.0

    def test_labelled_high_letter_in_prose_matches(self) -> None:
        # Previously the [A-D] regex could not capture an F wrapped in prose, so
        # this scored 0; now it recovers the choice letter and scores 1.0.
        assert agent_metrics.exact_match("The answer is F", "F") == 1.0

    def test_high_letter_wrong_choice_scores_zero(self) -> None:
        assert agent_metrics.exact_match("G", "H") == 0.0

    # B-5: open items (no options) carry a numeric/text gold; passing the valid
    # letter set keeps the letter path from firing so the normalised text
    # fallback scores them. A correct "10" must score 1.0, a wrong one 0.0.
    def test_open_numeric_item_scores_via_text_fallback(self) -> None:
        assert agent_metrics.exact_match("10", "10", valid_letters=None) == 1.0
        assert agent_metrics.exact_match("2", "10", valid_letters=None) == 0.0

    def test_valid_letters_constrains_the_match(self) -> None:
        # When the item only offers A-C, a stray capital ("D") must not be read
        # as a choice; the bare "D" then falls through to the text comparison.
        valid = frozenset("ABC")
        assert agent_metrics.exact_match("C", "C", valid_letters=valid) == 1.0
        assert agent_metrics.exact_match("D", "C", valid_letters=valid) == 0.0
        # An in-set letter still scores even when wrapped in prose.
        assert (
            agent_metrics.exact_match("The answer is B", "B", valid_letters=valid)
            == 1.0
        )

    def test_valid_letters_skips_out_of_set_capital_in_prose(self) -> None:
        # "GIS" leads with an out-of-set capital G; with valid_letters={A,B}, the
        # parser must not grab G and must reach the real in-set letter A.
        valid = frozenset("AB")
        assert (
            agent_metrics.exact_match("GIS, option A", "A", valid_letters=valid)
            == 1.0
        )


# --------------------------------------------------------------------------- #
# _extract_choice_letter (B-5: A-Z range + valid-letters constraint)
# --------------------------------------------------------------------------- #


class TestExtractChoiceLetter:
    """Letter extraction across the full A-J label range and constraints."""

    def test_extracts_letter_beyond_d(self) -> None:
        assert agent_metrics._extract_choice_letter("F") == "F"
        assert agent_metrics._extract_choice_letter("(I)") == "I"
        assert agent_metrics._extract_choice_letter("Answer: H") == "H"

    def test_returns_none_for_open_text(self) -> None:
        assert agent_metrics._extract_choice_letter("10") is None
        assert agent_metrics._extract_choice_letter("rice paddy") is None

    def test_valid_letters_rejects_out_of_set_letter(self) -> None:
        valid = frozenset("ABC")
        # A bare out-of-set letter is rejected (returns None), an in-set one is
        # accepted.
        assert agent_metrics._extract_choice_letter("E", valid) is None
        assert agent_metrics._extract_choice_letter("B", valid) == "B"

    def test_valid_letters_skips_to_first_in_set_standalone(self) -> None:
        # The leading capital G is out of set; the parser advances to the in-set
        # A rather than giving up.
        assert agent_metrics._extract_choice_letter("GIS A", frozenset("AB")) == "A"


# --------------------------------------------------------------------------- #
# f1_squad
# --------------------------------------------------------------------------- #


class TestF1Squad:
    """SQuAD-style token-overlap F1."""

    def test_identical_strings_score_one(self) -> None:
        assert agent_metrics.f1_squad("rice paddy field", "rice paddy field") == 1.0

    def test_no_overlap_scores_zero(self) -> None:
        assert agent_metrics.f1_squad("rice paddy", "tractor engine") == 0.0

    def test_known_partial_overlap(self) -> None:
        # Tokens are normalised (articles dropped), so use non-article words.
        # pred: {rice, maize, wheat}; gold: {maize, wheat, barley}; shared = 2.
        # precision = recall = 2/3 -> F1 = 2/3.
        score = agent_metrics.f1_squad("rice maize wheat", "maize wheat barley")
        assert score == pytest.approx(2 / 3, abs=1e-9)

    def test_articles_and_punctuation_normalised(self) -> None:
        assert agent_metrics.f1_squad("the maize.", "maize") == 1.0

    def test_empty_inputs_score_zero(self) -> None:
        assert agent_metrics.f1_squad("", "maize") == 0.0
        assert agent_metrics.f1_squad("maize", "") == 0.0
        assert agent_metrics.f1_squad("", "") == 0.0


# --------------------------------------------------------------------------- #
# tool_call_accuracy
# --------------------------------------------------------------------------- #


class TestToolCallAccuracy:
    """Tool-call recall for the plan-and-react copilot queries."""

    def test_all_expected_calls_present(self) -> None:
        score = agent_metrics.tool_call_accuracy(
            ["get_ndvi", "search_stac"], ["search_stac", "get_ndvi"]
        )
        assert score == 1.0

    def test_partial_recall(self) -> None:
        score = agent_metrics.tool_call_accuracy(["get_ndvi"], ["get_ndvi", "search_stac"])
        assert score == pytest.approx(0.5, abs=1e-9)

    def test_case_insensitive_and_dedup(self) -> None:
        score = agent_metrics.tool_call_accuracy(
            ["GET_NDVI", "get_ndvi"], ["get_ndvi"]
        )
        assert score == 1.0

    def test_no_gold_is_vacuously_satisfied(self) -> None:
        assert agent_metrics.tool_call_accuracy([], []) == 1.0
        assert agent_metrics.tool_call_accuracy(["extra"], []) == 1.0

    def test_gold_expected_but_none_predicted(self) -> None:
        assert agent_metrics.tool_call_accuracy([], ["get_ndvi"]) == 0.0


# --------------------------------------------------------------------------- #
# hallucination_rate (LLM-as-judge, mocked)
# --------------------------------------------------------------------------- #


class _ConstantJudge:
    """Mock judge returning a fixed hallucination score per sample."""

    def __init__(self, score: float) -> None:
        self._score = score
        self.seen: list[dict[str, Any]] = []

    def score(self, sample: dict[str, Any]) -> float:
        """Record the sample and return the constant score."""
        self.seen.append(sample)
        return self._score


class _RaisingJudge:
    """Mock judge that raises, to exercise the resilient scoring path."""

    def score(self, sample: dict[str, Any]) -> float:
        raise RuntimeError("judge boom")


class TestHallucinationRate:
    """Mean hallucination rate via the injectable LLM-as-judge."""

    def test_no_judge_returns_nan(self) -> None:
        samples = [{"input": "q", "actual_output": "a", "context": ["c"]}]
        result = agent_metrics.hallucination_rate(samples, judge=None)
        assert math.isnan(result)

    def test_empty_samples_score_zero(self) -> None:
        assert agent_metrics.hallucination_rate([], judge=None) == 0.0
        assert agent_metrics.hallucination_rate([], judge=_ConstantJudge(0.9)) == 0.0

    def test_mocked_judge_mean(self) -> None:
        judge = _ConstantJudge(0.25)
        samples = [
            {"input": "q1", "actual_output": "a1", "context": ["c1"]},
            {"input": "q2", "actual_output": "a2", "context": ["c2"]},
        ]
        result = agent_metrics.hallucination_rate(samples, judge=judge)
        assert result == pytest.approx(0.25, abs=1e-9)
        assert len(judge.seen) == 2
        # The judge receives the DeepEval-shaped sample dict.
        assert set(judge.seen[0]) == {"input", "actual_output", "context"}

    def test_judge_protocol_is_satisfied_by_duck_typed_object(self) -> None:
        assert isinstance(_ConstantJudge(0.1), agent_metrics.HallucinationJudge)

    def test_all_judge_calls_raise_returns_nan(self) -> None:
        samples = [{"input": "q", "actual_output": "a", "context": ["c"]}]
        result = agent_metrics.hallucination_rate(samples, judge=_RaisingJudge())
        assert math.isnan(result)


# --------------------------------------------------------------------------- #
# codebleu_score
# --------------------------------------------------------------------------- #


class TestCodeBleuScore:
    """Simplified CodeBLEU (BLEU n-gram + keyword/identifier Jaccard)."""

    def test_identical_code_scores_near_one(self) -> None:
        code = "def ndvi(red, nir):\n    return (nir - red) / (nir + red)\n"
        score = agent_metrics.codebleu_score(code, code)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_unrelated_code_scores_low(self) -> None:
        pred = "x = 1\nprint(x)\n"
        ref = "def compute_slope(dem):\n    return gradient(dem, axis=0)\n"
        score = agent_metrics.codebleu_score(pred, ref)
        assert score < 0.3

    def test_score_in_unit_interval(self) -> None:
        pred = "def ndvi(red, nir):\n    return (nir - red)\n"
        ref = "def ndvi(red, nir):\n    return (nir - red) / (nir + red)\n"
        score = agent_metrics.codebleu_score(pred, ref)
        assert 0.0 <= score <= 1.0

    def test_bleu_weight_zero_is_pure_keyword_overlap(self) -> None:
        pred = "def ndvi(red, nir):\n    return (nir - red) / (nir + red)\n"
        ref = "def ndvi(red, nir):\n    return (nir - red) / (nir + red)\n"
        # With identical code the keyword Jaccard is 1.0 regardless of weight.
        assert agent_metrics.codebleu_score(pred, ref, bleu_weight=0.0) == pytest.approx(
            1.0, abs=1e-9
        )

    def test_empty_inputs_score_zero(self) -> None:
        assert agent_metrics.codebleu_score("", "x = 1") == 0.0
        assert agent_metrics.codebleu_score("x = 1", "") == 0.0
        assert agent_metrics.codebleu_score("   ", "x = 1") == 0.0


# --------------------------------------------------------------------------- #
# bertscore_f1 (semantic proxy, fake encoder)
# --------------------------------------------------------------------------- #


class TestBertScoreF1:
    """Semantic-proxy BERTScore over paired strings (fake encoder)."""

    def test_identical_strings_score_one(self, fake_sentence_model: None) -> None:
        preds = ["the maize field", "a rice paddy"]
        golds = ["the maize field", "a rice paddy"]
        score = agent_metrics.bertscore_f1(preds, golds)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_length_mismatch_returns_zero(self, fake_sentence_model: None) -> None:
        score = agent_metrics.bertscore_f1(["a"], ["a", "b"])
        assert score == 0.0

    def test_empty_inputs_return_zero(self, fake_sentence_model: None) -> None:
        assert agent_metrics.bertscore_f1([], []) == 0.0
        assert agent_metrics.bertscore_f1(["a"], []) == 0.0

    def test_score_in_unit_interval(self, fake_sentence_model: None) -> None:
        score = agent_metrics.bertscore_f1(["maize"], ["completely other text"])
        assert 0.0 <= score <= 1.0


# --------------------------------------------------------------------------- #
# workflow_semantic_similarity (semantic proxy, fake encoder)
# --------------------------------------------------------------------------- #


class TestWorkflowSemanticSimilarity:
    """Workflow similarity over the GeoAnalystBench plan (fake encoder)."""

    def test_identical_workflow_scores_one(self, fake_sentence_model: None) -> None:
        workflow = "1. load raster\n2. clip to AOI\n3. compute NDVI"
        score = agent_metrics.workflow_semantic_similarity(workflow, workflow)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_accepts_list_of_steps(self, fake_sentence_model: None) -> None:
        steps = ["load raster", "clip to AOI", "compute NDVI"]
        as_text = "load raster\nclip to AOI\ncompute NDVI"
        score = agent_metrics.workflow_semantic_similarity(steps, as_text)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_empty_inputs_return_zero(self, fake_sentence_model: None) -> None:
        assert agent_metrics.workflow_semantic_similarity("", "step") == 0.0
        assert agent_metrics.workflow_semantic_similarity([], "step") == 0.0

    def test_score_in_unit_interval(self, fake_sentence_model: None) -> None:
        score = agent_metrics.workflow_semantic_similarity(
            "load and clip raster", "train a random forest classifier"
        )
        assert 0.0 <= score <= 1.0
