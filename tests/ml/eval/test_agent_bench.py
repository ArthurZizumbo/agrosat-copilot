"""Integration tests for the agent-benchmark harness (US-049).

These tests drive :mod:`ml.eval.agent_bench` with **mock backends only** (no
network, no real LLM) and the **real** datasets shipped under ``data/`` so the
loaders are exercised against the genuine 500-item AgroMind subset and the
50-task GeoAnalystBench CSV.

Key doubles:

- :class:`_FixedBackend` implements the :class:`~ml.agent.backends.LLMBackend`
  contract by exposing an async ``generate_stream`` that yields duck-typed
  chunks with a ``text`` attribute (the only thing the harness reads). It returns
  a canned answer so the parsed letter / workflow / code are deterministic.
- The sentence encoder behind the semantic proxies is monkeypatched with a
  deterministic fake (``sentence_transformers.SentenceTransformer``) and the
  module cache ``agent_metrics._sentence_model`` is reset, so no real model is
  loaded and ``mean_semantic_sim`` / ``bertscore`` stay finite and offline.

The Qwen text-only tension (AC-3) is verified explicitly: a non-multimodal
variant skips every multimodal AgroMind item and reports ``n_skipped`` > 0.

Conventions: identifiers and docstrings in English; visible prose elsewhere in
Spanish; no emojis; full type hints; ``pytest-asyncio`` in auto mode (no
decorator needed for the ``async def`` tests).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml.eval import agent_bench, agent_metrics

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AGROMIND_PATH = _REPO_ROOT / "data" / "agromind" / "agromind_subset_500.json"
_GEO_PATH = _REPO_ROOT / "data" / "geoanalystbench" / "GeoAnalystBench.csv"


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


class _Chunk:
    """Minimal backend chunk: only the ``text`` attribute is read by the harness."""

    def __init__(self, text: str) -> None:
        self.text = text


class _FixedBackend:
    """Backend double returning a canned answer for every turn (no network).

    Implements the slice of :class:`~ml.agent.backends.LLMBackend` the harness
    uses: an async ``generate_stream`` yielding chunks with a ``text`` delta.
    """

    def __init__(self, answer: str) -> None:
        self.model = "mock"
        self._answer = answer
        self.calls = 0

    async def generate_stream(
        self,
        *,
        contents: Any,
        tools: Any,
        system_instruction: str,
    ) -> Any:
        """Yield the canned answer as a single text chunk."""
        self.calls += 1
        yield _Chunk(self._answer)


class _FakeEncoder:
    """Deterministic ``SentenceTransformer`` stand-in (see metrics test)."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self._dim = 16

    def encode(self, texts: list[str], **_kwargs: Any) -> np.ndarray:
        """Embed strings to stable per-string vectors (identical -> identical)."""
        rows = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            rows.append(rng.standard_normal(self._dim))
        return np.asarray(rows, dtype=np.float64)


@pytest.fixture
def fake_sentence_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the sentence encoder and reset the module cache (offline proxies)."""
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeEncoder)
    monkeypatch.setattr(agent_metrics, "_sentence_model", None, raising=False)
    yield
    monkeypatch.setattr(agent_metrics, "_sentence_model", None, raising=False)


def _require_data() -> None:
    """Skip the test when the real datasets are not present locally."""
    if not _AGROMIND_PATH.exists():
        pytest.skip(f"AgroMind subset missing: {_AGROMIND_PATH}")
    if not _GEO_PATH.exists():
        pytest.skip(f"GeoAnalystBench CSV missing: {_GEO_PATH}")


# --------------------------------------------------------------------------- #
# Loaders against the real data
# --------------------------------------------------------------------------- #


class TestLoaders:
    """The loaders parse the genuine shipped datasets."""

    def test_load_agromind_subset_real_500(self) -> None:
        _require_data()
        items = agent_bench.load_agromind_subset(_AGROMIND_PATH)
        assert len(items) == 500
        n_multimodal = sum(1 for it in items if it.is_multimodal)
        # Documented split for this subset: 494 multimodal, 6 purely textual.
        assert n_multimodal == 494
        assert len(items) - n_multimodal == 6
        first = items[0]
        assert first.answer in {"A", "B", "C", "D"}
        assert set("ABCD").issuperset(first.options.keys())

    def test_option_image_paths_property(self) -> None:
        _require_data()
        items = agent_bench.load_agromind_subset(_AGROMIND_PATH)
        # Items whose options are image paths expose them via the property; the
        # property is a subset of options keyed by label.
        multi_opt = next(
            (it for it in items if it.option_image_paths), None
        )
        if multi_opt is not None:
            assert set(multi_opt.option_image_paths).issubset(multi_opt.options)

    def test_load_geoanalystbench_real_50(self) -> None:
        _require_data()
        tasks = agent_bench.load_geoanalystbench(_GEO_PATH)
        assert len(tasks) == 50
        # The trailing blank row (empty id) must have been dropped.
        assert all(t.id for t in tasks)
        first = tasks[0]
        assert first.instruction
        assert first.human_workflow


# --------------------------------------------------------------------------- #
# eval_agromind
# --------------------------------------------------------------------------- #


class TestEvalAgromind:
    """AgroMind evaluation with a mock backend."""

    async def test_multimodal_variant_scores_all_items(
        self, fake_sentence_model: None
    ) -> None:
        _require_data()
        items = agent_bench.load_agromind_subset(_AGROMIND_PATH)
        gold = items[0].answer
        variant = agent_bench.ReasonerVariant(
            name="gemini", model="mock", multimodal=True
        )
        backend = _FixedBackend(answer=gold)
        result = await agent_bench.eval_agromind(
            variant,
            items,
            backend=backend,
            judge=None,
            seed=0,
            image_root=Path("data/agromind/images_does_not_exist"),
        )
        # Multimodal variant evaluates the full subset, skips nothing.
        assert result["n_evaluated"] == 500
        assert result["n_skipped"] == 0
        assert 0.0 <= result["exact_match"] <= 1.0
        # The backend always answers the single letter ``gold``; every item whose
        # gold answer is that same letter is a hit, so exact_match is at least the
        # fraction of items carrying that letter (free-text golds that embed an
        # A-D token may also match, so this is a lower bound, never above 1).
        n_gold = sum(1 for it in items if it.answer == gold)
        assert result["exact_match"] >= n_gold / 500 - 1e-9

    async def test_text_only_variant_skips_multimodal(
        self, fake_sentence_model: None
    ) -> None:
        _require_data()
        items = agent_bench.load_agromind_subset(_AGROMIND_PATH)
        variant = agent_bench.ReasonerVariant(
            name="qwen", model="mock", multimodal=False
        )
        backend = _FixedBackend(answer="A")
        result = await agent_bench.eval_agromind(
            variant, items, backend=backend, judge=None, seed=0
        )
        # Text-only Qwen skips the 494 multimodal items and scores only the 6
        # purely-textual ones; the limitation is reported, never papered over.
        assert result["n_skipped"] == 494
        assert result["n_evaluated"] == 6
        assert backend.calls == 6

    async def test_hallucination_is_nan_without_judge(
        self, fake_sentence_model: None
    ) -> None:
        _require_data()
        items = agent_bench.load_agromind_subset(_AGROMIND_PATH)[:5]
        variant = agent_bench.ReasonerVariant(
            name="gemini", model="mock", multimodal=True
        )
        result = await agent_bench.eval_agromind(
            variant,
            items,
            backend=_FixedBackend(answer="A"),
            judge=None,
            seed=0,
            image_root=Path("data/agromind/images_does_not_exist"),
        )
        import math

        assert math.isnan(float(result["hallucination"]))

    async def test_judge_injected_yields_finite_hallucination(
        self, fake_sentence_model: None
    ) -> None:
        _require_data()
        items = agent_bench.load_agromind_subset(_AGROMIND_PATH)[:4]

        class _Judge:
            def score(self, sample: dict[str, Any]) -> float:
                return 0.1

        variant = agent_bench.ReasonerVariant(
            name="gemini", model="mock", multimodal=True
        )
        result = await agent_bench.eval_agromind(
            variant,
            items,
            backend=_FixedBackend(answer="A"),
            judge=_Judge(),
            seed=0,
            image_root=Path("data/agromind/images_does_not_exist"),
        )
        assert result["hallucination"] == pytest.approx(0.1, abs=1e-9)


# --------------------------------------------------------------------------- #
# eval_geoanalyst
# --------------------------------------------------------------------------- #


class TestEvalGeoanalyst:
    """GeoAnalystBench evaluation with a mock backend."""

    async def test_pass_rate_high_when_workflow_matches(
        self, fake_sentence_model: None
    ) -> None:
        _require_data()
        tasks = agent_bench.load_geoanalystbench(_GEO_PATH)[:5]
        variant = agent_bench.ReasonerVariant(
            name="gemini", model="mock", multimodal=True
        )
        # The mock echoes each task's own gold workflow + code, so the fake
        # encoder embeds identical text (cosine 1.0) -> every task passes.
        first = tasks[0]
        answer = (
            f"WORKFLOW:\n{first.human_workflow}\n"
            f"CODE:\n```python\n{first.code_string}\n```"
        )
        backend = _FixedBackend(answer=answer)
        result = await agent_bench.eval_geoanalyst(
            variant, [first], backend=backend, seed=0
        )
        assert result["n"] == 1
        assert result["pass_rate"] == pytest.approx(1.0, abs=1e-9)
        assert result["mean_semantic_sim"] == pytest.approx(1.0, abs=1e-6)
        assert 0.0 <= result["mean_codebleu"] <= 1.0

    async def test_full_taskset_runs_for_text_only_variant(
        self, fake_sentence_model: None
    ) -> None:
        _require_data()
        tasks = agent_bench.load_geoanalystbench(_GEO_PATH)
        variant = agent_bench.ReasonerVariant(
            name="qwen", model="mock", multimodal=False
        )
        backend = _FixedBackend(answer="WORKFLOW:\n1. do nothing\nCODE:\n```python\npass\n```")
        result = await agent_bench.eval_geoanalyst(
            variant, tasks, backend=backend, seed=0
        )
        # GeoAnalystBench is 100% text: every variant runs the full task set.
        assert result["n"] == 50
        assert backend.calls == 50
        assert 0.0 <= result["pass_rate"] <= 1.0


# --------------------------------------------------------------------------- #
# run_benchmark (aggregation + report)
# --------------------------------------------------------------------------- #


class TestRunBenchmark:
    """End-to-end aggregation over seeds and HTML report generation."""

    def test_run_benchmark_aggregates_and_writes_report(
        self, fake_sentence_model: None, tmp_path: Path
    ) -> None:
        _require_data()
        variants = [
            agent_bench.ReasonerVariant(name="gemini", model="mock", multimodal=True),
            agent_bench.ReasonerVariant(name="qwen", model="mock", multimodal=False),
        ]
        backends = {
            "gemini": _FixedBackend(answer="A"),
            "qwen": _FixedBackend(answer="A"),
        }
        report_path = tmp_path / "agent_bench.html"
        results = agent_bench.run_benchmark(
            variants,
            seeds=(0, 1),
            agromind_path=_AGROMIND_PATH,
            geo_path=_GEO_PATH,
            backends=backends,
            judge=None,
            image_root=tmp_path / "no_images",
            report_path=report_path,
            log_mlflow=False,
            probe_server=False,
        )

        # Nested shape: {variant: {benchmark: {metric: {"mean", "std"}}}}.
        assert set(results) == {"gemini", "qwen"}
        for variant in ("gemini", "qwen"):
            assert set(results[variant]) == {"AgroMind", "GeoAnalystBench"}
            agro = results[variant]["AgroMind"]
            geo = results[variant]["GeoAnalystBench"]
            # Headline keys are populated with the exact expected names.
            assert "exact_match" in agro
            assert "pass_rate" in geo
            for metric_stats in (agro["exact_match"], geo["pass_rate"]):
                assert set(metric_stats) == {"mean", "std"}
                assert isinstance(metric_stats["mean"], float)
                assert isinstance(metric_stats["std"], float)
            # std over the two seeds is 0.0 because the mock is deterministic.
            assert agro["exact_match"]["std"] == pytest.approx(0.0, abs=1e-9)

        # The text-only Qwen carries the n_skipped signal in its AgroMind table.
        assert results["qwen"]["AgroMind"]["n_skipped"]["mean"] == pytest.approx(
            494.0, abs=1e-9
        )

        # The HTML report was written.
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "<table>" in content
        assert "gemini" in content
        assert "qwen" in content
