"""Offline tests for the project-grounded agent system eval (sibling of US-049).

Every external boundary is replaced by a deterministic in-memory double, exactly
as in :mod:`tests.ml.agent.test_agent`:

- the LLM backend is a scripted :class:`ScriptedBackend` (a tiny ``FakeBackend``
  variant) that yields either a ``function_call`` chunk (native-FC path) or a
  text chunk (JSON-fallback path / final answers) -- no ``google-genai`` client,
  no network;
- Eval 2 runs the REAL ``classify_new_parcel.run`` under a stubbed embedding
  fetch + a stubbed classifier whose ``predict_proba_18`` returns the injected
  posterior (so routing + ``ClassificationResult`` plumbing is exercised);
- Eval 3 scores with a deterministic :class:`FakeJudge` so the A/B aggregation
  runs with zero judge-model network.

The tests assert: tool-selection accuracy (correct + wrong tool), partial arg
matching, the JSON-fallback parser, grounded crop match (positive + needs-GEE
control + faithfulness trap) and the RAG A/B two-number output with a delta.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from ml.eval.agent_bench import ReasonerVariant
from ml.eval.agent_system_eval import (
    CropCase,
    RagCase,
    ToolCallCase,
    _augment_query_with_context,
    _parse_json_tool_answer,
    _score_args,
    eval_grounded_crop,
    eval_rag_ab,
    eval_tool_calling,
    load_crop_cases,
    load_rag_cases,
    load_toolcall_cases,
    run_system_eval,
)

#: Test session id (mirrors the agent suite's SESSION_A; the DB is never touched
#: by Eval 2 since the embedding fetch + classifier are stubbed per case).
_SESSION_A = __import__("uuid").UUID("11111111-1111-1111-1111-111111111111")

#: Repo root, so the curated JSONL datasets resolve regardless of the pytest CWD
#: (CI runs the suite from ``ml/``, not from the repo root, so the module-level
#: relative defaults would not be found).
from pathlib import Path  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TOOLCALL_PATH = _REPO_ROOT / "data" / "agent_eval" / "toolcalling_cases.jsonl"
_CROP_PATH = _REPO_ROOT / "data" / "agent_eval" / "grounded_crop_cases.jsonl"
_RAG_PATH = _REPO_ROOT / "data" / "agent_eval" / "rag_hallucination_cases.jsonl"


class _FakeSettings:
    """Lightweight settings stub for the :class:`ToolContext` (no .env.local)."""

    database_url = "postgresql+asyncpg://agrosat:agrosat@localhost:5432/agrosat"
    titiler_host_port = 8001
    rag_enabled = False


@pytest.fixture
def make_ctx():
    """Return a factory building a ``ToolContext`` for Eval 2 (pool unused)."""
    from ml.agent.context import ToolContext

    def _make(session_id=_SESSION_A, defer=None):
        return ToolContext(
            pool=None,  # type: ignore[arg-type]
            settings=_FakeSettings(),  # type: ignore[arg-type]
            session_id=session_id,
            defer=defer,
        )

    return _make


GEMINI = ReasonerVariant(name="gemini", model="gemini-3.5-flash", multimodal=True)
GEMMA = ReasonerVariant(name="gemma-base", model="gemma4:26b", multimodal=True)

_POLY = {
    "type": "Polygon",
    "coordinates": [[[3.1, 43.4], [3.11, 43.4], [3.11, 43.41], [3.1, 43.41], [3.1, 43.4]]],
}


# ---------------------------------------------------------------------------
# Backend doubles
# ---------------------------------------------------------------------------
@dataclass
class _FC:
    """Duck-typed ``BackendFunctionCall`` stand-in (name/args/id)."""

    name: str
    args: dict[str, Any]
    id: str | None = None


@dataclass
class _Chunk:
    """Duck-typed ``BackendChunk`` stand-in (text and/or function_call)."""

    text: str | None = None
    function_call: _FC | None = None


@dataclass
class ScriptedBackend:
    """Scripted backend yielding ``turns[i]`` on the i-th ``generate_stream``.

    ``reset`` rewinds the turn cursor so the same backend instance can be reused
    across seeds (the harness calls ``reset`` before each seed). Beyond the
    script a terminal text chunk is yielded so a runaway loop still terminates.
    """

    turns: list[list[_Chunk]]
    model: str = "scripted"
    _turn: int = 0

    def reset(self) -> None:
        """Rewind the turn cursor to the start of the script."""
        self._turn = 0

    async def generate_stream(
        self, *, contents: list, tools: list, system_instruction: str
    ) -> AsyncIterator[_Chunk]:
        """Yield the next scripted turn's chunks."""
        index = self._turn
        self._turn += 1
        chunks = self.turns[index] if index < len(self.turns) else [_Chunk(text="(fin)")]
        for chunk in chunks:
            yield chunk


@dataclass
class StaticTextBackend:
    """Backend that returns the SAME text answer on every call (for fallbacks)."""

    text: str
    model: str = "static"
    calls: int = field(default=0)

    def reset(self) -> None:
        """No-op reset (the answer is stateless)."""

    async def generate_stream(
        self, *, contents: list, tools: list, system_instruction: str
    ) -> AsyncIterator[_Chunk]:
        """Yield the configured text as a single chunk."""
        self.calls += 1
        yield _Chunk(text=self.text)


class FakeJudge:
    """Deterministic hallucination judge: high score when no grounding context.

    Returns ``0.1`` when the sample's context contains the citation-tagged
    grounding block (the A/grounded run) and ``0.8`` otherwise (the B/ungrounded
    run), so the A/B delta is deterministically positive offline.
    """

    def score(self, sample: dict[str, Any]) -> float:
        """Return a fixed hallucination score keyed on the grounding presence."""
        context = " ".join(sample.get("context") or [])
        return 0.1 if "corpus PASTIS-R" in context else 0.8


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def test_loaders_read_curated_datasets() -> None:
    """The three curated JSONL datasets load and validate."""
    tcs = load_toolcall_cases(_TOOLCALL_PATH)
    crops = load_crop_cases(_CROP_PATH)
    rags = load_rag_cases(_RAG_PATH)
    assert len(tcs) >= 18
    assert {c.expected_tool for c in tcs} >= {
        "list_parcels",
        "get_parcel_timeseries",
        "classify_new_parcel",
        "search_stac",
        "retrieve_context",
    }
    assert any(c.expects_needs_gee for c in crops)
    assert all(r.ungrounded_is_unanswerable for r in rags)


# ---------------------------------------------------------------------------
# Eval 1 -- tool calling
# ---------------------------------------------------------------------------
async def test_tool_selection_accuracy_native_fc() -> None:
    """A native-FC backend that always picks the expected tool scores 1.0."""
    cases = [
        ToolCallCase(
            id="tc-x",
            user_query="que parcelas tengo?",
            expected_tool="list_parcels",
            expected_args_subset={},
            fuzzy_arg_keys=["session_id"],
            category="sync",
            rationale="",
        )
    ]
    backend = ScriptedBackend(turns=[[_Chunk(function_call=_FC(name="list_parcels", args={}))]])
    out = await eval_tool_calling(GEMINI, cases, backend=backend, seed=0)
    assert out["tool_selection_accuracy"] == 1.0
    assert out["tool_calling_native"] == 1.0
    assert out["no_call_rate"] == 0.0


async def test_tool_selection_accuracy_wrong_tool_scores_zero() -> None:
    """Selecting the wrong tool scores 0 and does not contribute arg accuracy."""
    cases = [
        ToolCallCase(
            id="tc-y",
            user_query="serie ndvi de la parcela 12",
            expected_tool="get_parcel_timeseries",
            expected_args_subset={"parcel_id": 12, "index": "ndvi"},
            fuzzy_arg_keys=["session_id"],
            category="sync",
            rationale="",
        )
    ]
    backend = ScriptedBackend(turns=[[_Chunk(function_call=_FC(name="list_parcels", args={}))]])
    out = await eval_tool_calling(GEMINI, cases, backend=backend, seed=0)
    assert out["tool_selection_accuracy"] == 0.0
    # No correctly-selected tool -> arg_match_accuracy is NaN (undefined).
    assert out["arg_match_accuracy"] != out["arg_match_accuracy"]


async def test_arg_match_partial() -> None:
    """Partial arg correctness yields a fractional arg_match_accuracy."""
    cases = [
        ToolCallCase(
            id="tc-z",
            user_query="serie ndvi de la parcela 12",
            expected_tool="get_parcel_timeseries",
            expected_args_subset={"parcel_id": 12, "index": "ndvi"},
            fuzzy_arg_keys=[],
            category="sync",
            rationale="",
        )
    ]
    # Correct tool, correct parcel_id, WRONG index -> 1 of 2 expected keys match.
    backend = ScriptedBackend(
        turns=[
            [
                _Chunk(
                    function_call=_FC(
                        name="get_parcel_timeseries",
                        args={
                            "parcel_id": 12,
                            "index": "ndwi",
                            "start": "2019-01-01",
                            "end": "2019-12-31",
                        },
                    )
                )
            ]
        ]
    )
    out = await eval_tool_calling(GEMINI, cases, backend=backend, seed=0)
    assert out["tool_selection_accuracy"] == 1.0
    assert out["arg_match_accuracy"] == 0.5


async def test_json_fallback_path_for_non_tool_backend() -> None:
    """A non-tool backend is scored via the JSON ``{tool,args}`` text fallback."""
    cases = [
        ToolCallCase(
            id="tc-f",
            user_query="busca escenas con pocas nubes",
            expected_tool="search_stac",
            expected_args_subset={"cloud_cover_max": 10.0},
            fuzzy_arg_keys=[],
            category="deferred",
            rationale="",
        )
    ]
    backend = StaticTextBackend(
        text='Pienso... Aqui va: {"tool": "search_stac", "args": {"cloud_cover_max": 10.0}}'
    )
    out = await eval_tool_calling(GEMMA, cases, backend=backend, seed=0)
    assert out["tool_selection_accuracy"] == 1.0
    assert out["tool_calling_native"] == 0.0
    assert out["parse_failure_rate"] == 0.0
    assert out["arg_match_accuracy"] == 1.0


async def test_json_fallback_parse_failure_buckets() -> None:
    """An unparseable text answer is a no_call + parse failure, not a wrong tool."""
    cases = [
        ToolCallCase(
            id="tc-g",
            user_query="x",
            expected_tool="list_parcels",
            expected_args_subset={},
            fuzzy_arg_keys=[],
            category="sync",
            rationale="",
        )
    ]
    backend = StaticTextBackend(text="No estoy seguro de que herramienta usar.")
    out = await eval_tool_calling(GEMMA, cases, backend=backend, seed=0)
    assert out["tool_selection_accuracy"] == 0.0
    assert out["no_call_rate"] == 1.0
    assert out["parse_failure_rate"] == 1.0


def test_parse_json_tool_answer_extracts_first_block() -> None:
    """The tolerant parser extracts the first ``{...}`` block from reasoning prose."""
    name, args = _parse_json_tool_answer(
        'Razonamiento. {"tool": "get_tiles", "args": {"scene_id": "S2A", "index": "ndvi"}} fin'
    )
    assert name == "get_tiles"
    assert args == {"scene_id": "S2A", "index": "ndvi"}


def test_score_args_fuzzy_keys_count_as_present() -> None:
    """Fuzzy keys are matched on presence, expected-subset on value equality."""
    case = ToolCallCase(
        id="tc-h",
        user_query="x",
        expected_tool="add_aoi",
        expected_args_subset={"name": "Valle Norte"},
        fuzzy_arg_keys=["session_id", "aoi"],
        category="deferred",
        rationale="",
    )
    # name matches; aoi present; session_id injected by the coercer -> 3/3.
    assert _score_args(case, {"name": "Valle Norte", "aoi": _POLY}) == 1.0


def test_augment_query_appends_aoi_only_for_spatial_cases() -> None:
    """Spatial cases get the frontend AOI appended; explicit-id cases are untouched.

    This levels the native-FC and JSON-fallback channels and mirrors what the real
    chat frontend sends, so a native reasoner no longer discovery-calls
    ``list_parcels`` first and is mis-scored (verified live: Gemini 0.55 -> 0.95).
    """
    spatial = ToolCallCase(
        id="tc-s",
        user_query="que cultivo hay en esta zona?",
        expected_tool="classify_new_parcel",
        expected_args_subset={},
        fuzzy_arg_keys=["session_id", "aoi"],
        category="sync",
        rationale="",
    )
    bbox_case = ToolCallCase(
        id="tc-b",
        user_query="busca escenas aqui",
        expected_tool="search_stac",
        expected_args_subset={},
        fuzzy_arg_keys=["bbox", "datetime_range"],
        category="deferred",
        rationale="",
    )
    explicit = ToolCallCase(
        id="tc-e",
        user_query="serie ndvi de la parcela 12",
        expected_tool="get_parcel_timeseries",
        expected_args_subset={"parcel_id": 12},
        fuzzy_arg_keys=["session_id"],
        category="sync",
        rationale="",
    )
    spatial_q = _augment_query_with_context(spatial)
    assert spatial.user_query in spatial_q
    assert "AOI (GeoJSON)" in spatial_q
    assert "AOI (GeoJSON)" in _augment_query_with_context(bbox_case)
    # No spatial fuzzy key -> the query is returned verbatim (no AOI noise).
    assert _augment_query_with_context(explicit) == explicit.user_query


# ---------------------------------------------------------------------------
# Eval 2 -- grounded crop
# ---------------------------------------------------------------------------
def _classify_backend(crop: str) -> ScriptedBackend:
    """Backend that calls ``classify_new_parcel`` then names ``crop`` in prose.

    The function call carries the ``aoi`` geometry (the model supplies it; only
    ``session_id`` is injected by the agent loop) so the strict ``*Input`` model
    validates and the real ``classify.run`` executes under the stubs.
    """
    return ScriptedBackend(
        turns=[
            [_Chunk(function_call=_FC(name="classify_new_parcel", args={"aoi": _POLY}))],
            [_Chunk(text=f"El cultivo de la parcela es {crop}.")],
        ]
    )


async def test_grounded_crop_routing_and_match(monkeypatch: pytest.MonkeyPatch, make_ctx) -> None:
    """The agent routes to classify and faithfully names the injected crop."""
    case = CropCase(
        id="crop-x",
        parcel_id=12,
        true_crop="Corn",
        injected_confidence=0.88,
        injected_class_probabilities={"Corn": 0.88, "Sunflower": 0.12},
        aoi_geometry=_POLY,
        year=2019,
        user_query="que cultivo hay aqui?",
        expects_needs_gee=False,
    )
    out = await eval_grounded_crop(
        GEMINI,
        [case],
        backend=_classify_backend("Corn"),
        make_ctx=make_ctx,
        monkeypatch_target=monkeypatch,
        seed=0,
    )
    assert out["routing_accuracy"] == 1.0
    assert out["crop_match_accuracy"] == 1.0
    assert out["faithfulness_crop"] == 1.0


async def test_grounded_crop_needs_gee_control(monkeypatch: pytest.MonkeyPatch, make_ctx) -> None:
    """A needs-GEE case is satisfied only by a faithful refusal (no crop named)."""
    case = CropCase(
        id="crop-gee",
        parcel_id=99,
        true_crop="needs_gee_sampling",
        injected_confidence=0.0555,
        injected_class_probabilities={"needs_gee_sampling": 1.0},
        aoi_geometry=_POLY,
        year=2019,
        user_query="clasifica esta nueva parcela",
        expects_needs_gee=True,
    )
    backend = ScriptedBackend(
        turns=[
            [_Chunk(function_call=_FC(name="classify_new_parcel", args={"aoi": _POLY}))],
            [_Chunk(text="La parcela requiere muestreo GEE; no puedo determinar el cultivo.")],
        ]
    )
    out = await eval_grounded_crop(
        GEMINI,
        [case],
        backend=backend,
        make_ctx=make_ctx,
        monkeypatch_target=monkeypatch,
        seed=0,
    )
    assert out["routing_accuracy"] == 1.0
    assert out["crop_match_accuracy"] == 1.0


async def test_grounded_crop_faithfulness_trap(monkeypatch: pytest.MonkeyPatch, make_ctx) -> None:
    """Naming a neighbour crop (drift) fails crop_match and faithfulness."""
    # Both ``Soybeans`` and ``Corn`` belong to the resolved france-9 label-space
    # (the default since US-053 / the champion re-wiring), so the restricted tool
    # actually returns ``Soybeans`` and the prose drift to ``Corn`` is a real,
    # detectable infidelity. (Using a crop outside france-9 would be silently
    # remapped by the restriction and the trap would not fire.)
    case = CropCase(
        id="crop-trap",
        parcel_id=21,
        true_crop="Soybeans",
        injected_confidence=0.97,
        injected_class_probabilities={"Soybeans": 0.97, "Corn": 0.03},
        aoi_geometry=_POLY,
        year=2019,
        user_query="cual es el cultivo?",
        expects_needs_gee=False,
    )
    # Tool returns Soybeans, but the prose drifts to Corn.
    out = await eval_grounded_crop(
        GEMINI,
        [case],
        backend=_classify_backend("Corn"),
        make_ctx=make_ctx,
        monkeypatch_target=monkeypatch,
        seed=0,
    )
    assert out["routing_accuracy"] == 1.0
    assert out["crop_match_accuracy"] == 0.0
    assert out["faithfulness_crop"] == 0.0


async def test_grounded_crop_no_routing_fails(monkeypatch: pytest.MonkeyPatch, make_ctx) -> None:
    """Answering the crop WITHOUT calling the tool fails routing (must orchestrate)."""
    case = CropCase(
        id="crop-noroute",
        parcel_id=12,
        true_crop="Corn",
        injected_confidence=0.88,
        injected_class_probabilities={"Corn": 0.88, "Sunflower": 0.12},
        aoi_geometry=_POLY,
        year=2019,
        user_query="que cultivo hay aqui?",
        expects_needs_gee=False,
    )
    backend = ScriptedBackend(turns=[[_Chunk(text="Es maiz (Corn) sin duda.")]])
    out = await eval_grounded_crop(
        GEMINI,
        [case],
        backend=backend,
        make_ctx=make_ctx,
        monkeypatch_target=monkeypatch,
        seed=0,
    )
    assert out["routing_accuracy"] == 0.0
    assert out["crop_match_accuracy"] == 0.0


@dataclass
class SequencedTextBackend:
    """Text backend yielding ``answers[i]`` on the i-th ``generate_stream`` call.

    Models the JSON-fallback path of an Ollama variant (which ignores ``tools``):
    the first call is the routing turn (a ``{"tool", "args"}`` JSON answer) and
    the second is the faithful-reporting turn. ``reset`` rewinds the cursor; the
    harness resets between the two turns of a single case, so the cursor is NOT
    rewound here on ``reset`` (it must keep advancing across the two turns).
    """

    answers: list[str]
    model: str = "sequenced"
    _idx: int = 0

    def reset(self) -> None:
        """No-op: the cursor must persist across the two turns of one case."""

    async def generate_stream(
        self, *, contents: list, tools: list, system_instruction: str
    ) -> AsyncIterator[_Chunk]:
        """Yield the next scripted text answer."""
        index = self._idx
        self._idx += 1
        text = self.answers[index] if index < len(self.answers) else "(fin)"
        yield _Chunk(text=text)


async def test_grounded_crop_fallback_for_non_native_variant(
    monkeypatch: pytest.MonkeyPatch, make_ctx
) -> None:
    """A non-native (Ollama) variant is scored via the JSON-fallback path.

    ``gemma-base`` ignores ``tools``, so ``eval_grounded_crop`` must route via a
    JSON ``{"tool", "args"}`` answer and then, once routed, run the stubbed
    ``classify.run`` and check the model faithfully echoes the returned crop.
    This is the Bug 2 regression test: previously this scored a hard zero because
    the Agent loop dropped the ignored tools.
    """
    case = CropCase(
        id="crop-fb",
        parcel_id=7,
        true_crop="Corn",
        injected_confidence=0.91,
        injected_class_probabilities={"Corn": 0.91, "Sunflower": 0.09},
        aoi_geometry=_POLY,
        year=2019,
        user_query="que cultivo hay en esta parcela?",
        expects_needs_gee=False,
    )
    # Turn 1: routes to classify_new_parcel. Turn 2: faithfully names the crop.
    backend = SequencedTextBackend(
        answers=[
            '{"tool": "classify_new_parcel", "args": {"year": 2019}}',
            "El cultivo de la parcela es Corn.",
        ]
    )
    out = await eval_grounded_crop(
        GEMMA,
        [case],
        backend=backend,
        make_ctx=make_ctx,
        monkeypatch_target=monkeypatch,
        seed=0,
    )
    assert out["routing_accuracy"] == 1.0
    assert out["crop_match_accuracy"] == 1.0
    assert out["faithfulness_crop"] == 1.0


async def test_grounded_crop_fallback_wrong_tool_fails_routing(
    monkeypatch: pytest.MonkeyPatch, make_ctx
) -> None:
    """A non-native variant that picks the wrong tool fails routing (no crop run)."""
    case = CropCase(
        id="crop-fb-wrong",
        parcel_id=8,
        true_crop="Corn",
        injected_confidence=0.91,
        injected_class_probabilities={"Corn": 0.91, "Sunflower": 0.09},
        aoi_geometry=_POLY,
        year=2019,
        user_query="que cultivo hay aqui?",
        expects_needs_gee=False,
    )
    backend = SequencedTextBackend(answers=['{"tool": "list_parcels", "args": {}}'])
    out = await eval_grounded_crop(
        GEMMA,
        [case],
        backend=backend,
        make_ctx=make_ctx,
        monkeypatch_target=monkeypatch,
        seed=0,
    )
    assert out["routing_accuracy"] == 0.0
    assert out["crop_match_accuracy"] == 0.0


# ---------------------------------------------------------------------------
# Eval 3 -- RAG A/B
# ---------------------------------------------------------------------------
async def test_rag_ab_produces_two_numbers_with_positive_delta() -> None:
    """Grounded (A) hallucinates less than ungrounded (B): delta > 0."""
    cases = [
        RagCase(
            id="rag-x",
            question="cual es el cultivo vecino al norte?",
            aoi_geometry=_POLY,
            grounding_text=(
                "Contexto recuperado de parcelas vecinas (corpus PASTIS-R):\n"
                "[phenology_caption:T31_1] maiz NDVI 0.82."
            ),
            gold_grounded_answer="La parcela vecina al norte es maiz.",
            ungrounded_is_unanswerable=True,
        )
    ]
    backend = StaticTextBackend(text="respuesta")
    out = await eval_rag_ab(GEMINI, cases, backend=backend, judge=FakeJudge(), seed=0)
    assert out["hallucination_rate_ungrounded"] == 0.8
    assert out["hallucination_rate_grounded"] == pytest.approx(0.1)
    assert out["hallucination_reduction_delta"] == pytest.approx(0.7)
    assert out["faithfulness_grounded"] == pytest.approx(0.9)


async def test_rag_ab_no_judge_is_nan() -> None:
    """Without a judge the A/B numbers and the delta are NaN (never fabricated)."""
    cases = [
        RagCase(
            id="rag-n",
            question="q",
            aoi_geometry=_POLY,
            grounding_text="Contexto recuperado de parcelas vecinas (corpus PASTIS-R):\n[s:1] x.",
            gold_grounded_answer="x",
            ungrounded_is_unanswerable=True,
        )
    ]
    out = await eval_rag_ab(GEMINI, cases, backend=StaticTextBackend(text="y"), judge=None, seed=0)
    assert out["hallucination_rate_ungrounded"] != out["hallucination_rate_ungrounded"]
    assert out["hallucination_reduction_delta"] != out["hallucination_reduction_delta"]


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
def test_run_system_eval_aggregates_all_three(monkeypatch: pytest.MonkeyPatch, make_ctx) -> None:
    """``run_system_eval`` wires the three evals and aggregates mean+-std."""
    tc_backend = ScriptedBackend(turns=[[_Chunk(function_call=_FC(name="list_parcels", args={}))]])
    crop_backend = _classify_backend("Corn")
    rag_backend = StaticTextBackend(text="r")

    results = run_system_eval(
        [GEMINI],
        seeds=(0, 1),
        toolcall_backends={"gemini": tc_backend},
        crop_backends={"gemini": crop_backend},
        rag_backends={"gemini": rag_backend},
        make_ctx=make_ctx,
        monkeypatch_target=monkeypatch,
        judge=FakeJudge(),
        toolcall_path=_TOOLCALL_PATH,
        crop_path=_CROP_PATH,
        rag_path=_RAG_PATH,
    )

    assert set(results["gemini"]) == {"tool_calling", "grounded_crop", "rag_ab"}
    tc = results["gemini"]["tool_calling"]
    assert "tool_selection_accuracy" in tc
    assert "mean" in tc["tool_selection_accuracy"]
    rag = results["gemini"]["rag_ab"]
    assert rag["hallucination_reduction_delta"]["mean"] == pytest.approx(0.7)
