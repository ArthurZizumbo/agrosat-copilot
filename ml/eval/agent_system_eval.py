"""Project-grounded system eval for the AgroSatCopilot conversational agent.

This is a SIBLING harness to :mod:`ml.eval.agent_bench` (it never imports nor
mutates that module's public API). Where ``agent_bench`` scores the reasoner on
the two public QA benchmarks (AgroMind, GeoAnalystBench), this module scores the
agent SYSTEM against the project's own tool/orchestration contracts, with three
complementary evals:

- **Eval 1 -- tool-calling correctness** (:func:`eval_tool_calling`): given a
  Spanish user turn and the REAL :func:`ml.agent.tools.build_function_declarations`
  schemas, does the reasoner select the right tool and fill the load-bearing
  arguments? Tool-capable backends (Gemini native FC, Qwen via vLLM) are read
  through ``BackendChunk.function_call``; the two Ollama backends
  (``gemma-base``, ``qwen36-vl``) IGNORE the ``tools`` argument, so they take a
  JSON ``{"tool", "args"}`` text-answer fallback parsed leniently. The
  ``tool_calling_native`` flag is reported per variant so the mechanism is
  transparent and the name-match metric stays comparable.

- **Eval 2 -- grounded crop accuracy** (:func:`eval_grounded_crop`): does the
  agent route to ``classify_new_parcel`` and FAITHFULLY report the crop the
  classifier returned? The XGBoost-AlphaEarth ensemble output is MOCKED per
  parcel (the embedding fetch + estimator are stubbed), so the known result is
  deterministic. This measures AGENT ORCHESTRATION + faithful reporting, NOT
  classifier quality (that lives in the EPIC 4/6 MLflow runs).

- **Eval 3 -- RAG A/B hallucination** (:func:`eval_rag_ab`): each question is
  answerable ONLY from neighbouring-parcel grounding. Run twice per variant: B
  (``rag_enabled=False``, ungrounded) and A (``rag_enabled=True``, grounding
  injected). Expect ``hallucination_rate_grounded`` (A) <
  ``hallucination_rate_ungrounded`` (B). Scored via
  :func:`ml.eval.agent_metrics.hallucination_rate` with an injectable judge.

The :func:`run_system_eval` aggregator mirrors
:func:`ml.eval.agent_bench.run_benchmark`: injectable backends per variant,
``mean +- std`` over seeds (reusing ``agent_bench._aggregate``), structlog, full
type hints and zero network in tests (every external boundary is injected or
monkeypatched).

Project conventions: identifiers and docstrings in English (Google style);
visible prose (the curated queries, the report) in Spanish; ``structlog`` (never
``print``); full type hints; no emojis.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from google.genai import types

from ml.agent.schemas import GeoJSONGeometry
from ml.agent.tools import TOOL_REGISTRY, build_function_declarations
from ml.eval import agent_metrics
from ml.eval.agent_bench import ReasonerVariant, _aggregate

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from ml.agent.backends import LLMBackend
    from ml.agent.context import ToolContext
    from ml.eval.agent_metrics import HallucinationJudge

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_CROP_PATH",
    "DEFAULT_RAG_PATH",
    "DEFAULT_TOOLCALL_PATH",
    "CropCase",
    "RagCase",
    "ToolCallCase",
    "eval_grounded_crop",
    "eval_rag_ab",
    "eval_tool_calling",
    "load_crop_cases",
    "load_rag_cases",
    "load_toolcall_cases",
    "run_system_eval",
]

#: The ten canonical tool names, mirrored from the registry for fail-fast checks.
_CANONICAL_TOOLS: frozenset[str] = frozenset(
    {
        "list_parcels",
        "get_parcel_timeseries",
        "get_aoi_stats",
        "search_stac",
        "get_tiles",
        "classify_new_parcel",
        "add_aoi",
        "compare_models",
        "explain_prediction",
        "retrieve_context",
    }
)

#: Default dataset locations (small curated JSONL, committed -- not DVC).
DEFAULT_TOOLCALL_PATH: Path = Path("data/agent_eval/toolcalling_cases.jsonl")
DEFAULT_CROP_PATH: Path = Path("data/agent_eval/grounded_crop_cases.jsonl")
DEFAULT_RAG_PATH: Path = Path("data/agent_eval/rag_hallucination_cases.jsonl")

#: Per-item timeout for a single reasoner call (mirrors agent_bench hardening).
_ITEM_TIMEOUT_S: float = 200.0

#: Variants whose backend ignores ``tools`` (OllamaBackend): they take the JSON
#: text fallback for Eval 1 and are reported with ``tool_calling_native=False``.
_NON_TOOL_VARIANTS: frozenset[str] = frozenset({"gemma-base", "qwen36-vl"})

#: The ``needs_gee_sampling`` sentinel the classify tool emits for a fresh AOI.
_NEEDS_GEE_SENTINEL: str = "needs_gee_sampling"

#: Spanish phrases that signal a faithful needs-GEE refusal in the final answer.
_NEEDS_GEE_PHRASES: tuple[str, ...] = (
    "needs_gee_sampling",
    "muestreo gee",
    "muestreo de gee",
    "requiere muestreo",
    "no tengo el embedding",
    "sin embedding",
)


# ---------------------------------------------------------------------------
# Dataset records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolCallCase:
    """One tool-calling correctness case (Eval 1).

    Attributes:
        id: Stable case id (e.g. ``"tc-001"``).
        user_query: Spanish natural-language user turn.
        expected_tool: One of the ten canonical tool names.
        expected_args_subset: Load-bearing args that must match exactly, keyed by
            the REAL ``*Input`` field names.
        fuzzy_arg_keys: Arg names that only need to be PRESENT, not value-equal
            (geometry/session/bbox injected or unknowable to the model).
        category: ``"sync"`` or ``"deferred"``.
        rationale: Spanish justification used in the report example rows.
    """

    id: str
    user_query: str
    expected_tool: str
    expected_args_subset: dict[str, Any]
    fuzzy_arg_keys: list[str]
    category: str
    rationale: str


@dataclass(frozen=True)
class CropCase:
    """One grounded-crop orchestration case (Eval 2).

    Attributes:
        id: Stable case id (e.g. ``"crop-001"``).
        parcel_id: The persisted parcel referenced (for logging/trace).
        true_crop: The KNOWN/injected crop label (a ``SEMANTIC18_CLASS_NAMES``
            value) the mocked classifier returns as argmax, or the
            ``needs_gee_sampling`` sentinel for negative controls.
        injected_confidence: Confidence of ``true_crop`` in ``[0, 1]``.
        injected_class_probabilities: ``{crop: prob}`` summing ~1.0; argmax MUST
            equal ``true_crop``.
        aoi_geometry: GeoJSON Polygon of the parcel.
        year: Campaign year (default 2019 per ``ClassifyParcelInput``).
        user_query: Spanish user turn.
        expects_needs_gee: When ``True`` the mock returns the needs-GEE sentinel
            and the gold expectation is a faithful refusal (no fabricated crop).
    """

    id: str
    parcel_id: int
    true_crop: str
    injected_confidence: float
    injected_class_probabilities: dict[str, float]
    aoi_geometry: dict[str, Any]
    year: int
    user_query: str
    expects_needs_gee: bool


@dataclass(frozen=True)
class RagCase:
    """One RAG A/B hallucination case (Eval 3).

    Attributes:
        id: Stable case id (e.g. ``"rag-001"``).
        question: Spanish question answerable ONLY from grounding.
        aoi_geometry: GeoJSON Polygon passed to ``retrieve_context``.
        grounding_text: The KNOWN grounding block injected on the A run (same
            citation-tagged shape ``_build_grounding_text`` emits).
        gold_grounded_answer: The faithful answer derivable from grounding (the
            judge's context/expected).
        ungrounded_is_unanswerable: ``True`` when, without grounding, the only
            faithful response is "no tengo ese dato".
    """

    id: str
    question: str
    aoi_geometry: dict[str, Any]
    grounding_text: str
    gold_grounded_answer: str
    ungrounded_is_unanswerable: bool


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file into a list of records (blank lines skipped).

    Args:
        path: The JSONL dataset path.

    Returns:
        The parsed records, order preserved.
    """
    records: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def load_toolcall_cases(path: Path = DEFAULT_TOOLCALL_PATH) -> list[ToolCallCase]:
    """Load and validate the tool-calling cases.

    Args:
        path: Path to ``toolcalling_cases.jsonl``.

    Returns:
        The parsed :class:`ToolCallCase` list.

    Raises:
        ValueError: If any case names a non-canonical ``expected_tool``.
    """
    cases: list[ToolCallCase] = []
    for record in _read_jsonl(path):
        expected_tool = str(record["expected_tool"])
        if expected_tool not in _CANONICAL_TOOLS:
            raise ValueError(
                f"case {record.get('id')!r} expects unknown tool {expected_tool!r}; "
                f"canonical: {sorted(_CANONICAL_TOOLS)}"
            )
        cases.append(
            ToolCallCase(
                id=str(record["id"]),
                user_query=str(record["user_query"]),
                expected_tool=expected_tool,
                expected_args_subset=dict(record.get("expected_args_subset") or {}),
                fuzzy_arg_keys=[str(k) for k in record.get("fuzzy_arg_keys") or []],
                category=str(record.get("category") or "sync"),
                rationale=str(record.get("rationale") or ""),
            )
        )
    logger.info("toolcall_cases_loaded", path=str(path), n=len(cases))
    return cases


def load_crop_cases(path: Path = DEFAULT_CROP_PATH) -> list[CropCase]:
    """Load and validate the grounded-crop cases.

    Args:
        path: Path to ``grounded_crop_cases.jsonl``.

    Returns:
        The parsed :class:`CropCase` list.
    """
    cases: list[CropCase] = []
    for record in _read_jsonl(path):
        cases.append(
            CropCase(
                id=str(record["id"]),
                parcel_id=int(record["parcel_id"]),
                true_crop=str(record["true_crop"]),
                injected_confidence=float(record.get("injected_confidence", 0.0)),
                injected_class_probabilities=dict(record["injected_class_probabilities"]),
                aoi_geometry=dict(record["aoi_geometry"]),
                year=int(record.get("year", 2019)),
                user_query=str(record["user_query"]),
                expects_needs_gee=bool(record.get("expects_needs_gee", False)),
            )
        )
    logger.info("crop_cases_loaded", path=str(path), n=len(cases))
    return cases


def load_rag_cases(path: Path = DEFAULT_RAG_PATH) -> list[RagCase]:
    """Load and validate the RAG A/B cases.

    Args:
        path: Path to ``rag_hallucination_cases.jsonl``.

    Returns:
        The parsed :class:`RagCase` list.
    """
    cases: list[RagCase] = []
    for record in _read_jsonl(path):
        cases.append(
            RagCase(
                id=str(record["id"]),
                question=str(record["question"]),
                aoi_geometry=dict(record["aoi_geometry"]),
                grounding_text=str(record.get("grounding_text") or ""),
                gold_grounded_answer=str(record.get("gold_grounded_answer") or ""),
                ungrounded_is_unanswerable=bool(record.get("ungrounded_is_unanswerable", True)),
            )
        )
    logger.info("rag_cases_loaded", path=str(path), n=len(cases))
    return cases


# ---------------------------------------------------------------------------
# Backend driving helpers (shared)
# ---------------------------------------------------------------------------
def _user_contents(text: str) -> list[types.Content]:
    """Build a single-user-turn ``contents`` list for a backend call.

    Args:
        text: The user prompt text.

    Returns:
        A one-element list with a ``role="user"`` :class:`types.Content`.
    """
    return [types.Content(role="user", parts=[types.Part.from_text(text=text)])]


async def _drive_for_call(
    backend: LLMBackend, prompt: str, declarations: list[types.FunctionDeclaration]
) -> tuple[str | None, dict[str, Any] | None, str]:
    """Drive a tool-capable backend for one turn and read its first call.

    Consumes the chunk stream, returning the first ``function_call`` seen (its
    name + args) and the concatenated text. Used for the native-FC variants
    (Gemini, Qwen).

    Args:
        backend: The injected/constructed :class:`LLMBackend`.
        prompt: The user prompt.
        declarations: Tool declarations to advertise this turn.

    Returns:
        ``(tool_name, args, text)`` -- ``tool_name``/``args`` are ``None`` when
        the turn produced no function call.
    """
    tool_name: str | None = None
    args: dict[str, Any] | None = None
    text_parts: list[str] = []
    async for chunk in backend.generate_stream(
        contents=_user_contents(prompt),
        tools=declarations,
        system_instruction="",
    ):
        text = getattr(chunk, "text", None)
        if text:
            text_parts.append(text)
        call = getattr(chunk, "function_call", None)
        if call is not None and tool_name is None and getattr(call, "name", None):
            tool_name = call.name
            args = dict(getattr(call, "args", None) or {})
    return tool_name, args, "".join(text_parts).strip()


def _declarations_for_generate(
    declarations: list[types.FunctionDeclaration],
) -> list[types.FunctionDeclaration]:
    """Strip the ``behavior`` field so the non-bidi ``generate_content`` accepts them.

    ``build_function_declarations`` tags deferred tools with
    ``Behavior.NON_BLOCKING`` for the real agent's async (bidi) loop, but the
    plain ``generate_content`` API rejects any ``FunctionDeclaration.behavior``
    with ``400 INVALID_ARGUMENT`` (it is only valid for ``BidiGenerateContent``).
    The eval only needs the name + parameter schema for the model to choose a
    tool, so a ``behavior``-less copy is built here.

    Args:
        declarations: The declarations from ``build_function_declarations``.

    Returns:
        Equivalent declarations with ``behavior`` cleared.
    """
    cleaned: list[types.FunctionDeclaration] = []
    for decl in declarations:
        cleaned.append(
            types.FunctionDeclaration(
                name=decl.name,
                description=decl.description,
                parameters=decl.parameters,
            )
        )
    return cleaned


async def _drive_for_text(backend: LLMBackend, prompt: str) -> str:
    """Drive a backend for one non-streaming text answer (no tools).

    Args:
        backend: The injected/constructed backend.
        prompt: The user prompt.

    Returns:
        The concatenated answer text (stripped).
    """
    buffer: list[str] = []
    async for chunk in backend.generate_stream(
        contents=_user_contents(prompt),
        tools=[],
        system_instruction="",
    ):
        text = getattr(chunk, "text", None)
        if text:
            buffer.append(text)
    return "".join(buffer).strip()


#: Lenient extractor for the first balanced ``{...}`` block in a text answer.
_FIRST_BRACE_RE: re.Pattern[str] = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_tool_answer(text: str) -> tuple[str | None, dict[str, Any] | None]:
    """Tolerantly parse a ``{"tool", "args"}`` answer from a text response.

    Gemma is a *thinking* model that may wrap the JSON in reasoning prose, so the
    first balanced ``{...}`` block is extracted and parsed; on any failure the
    result is ``(None, None)`` so the case is bucketed as a parse failure rather
    than mis-scored as a wrong tool.

    Args:
        text: The raw text answer from a non-tool backend.

    Returns:
        ``(tool_name, args)`` -- both ``None`` when no JSON tool object parses.
    """
    if not text:
        return None, None
    candidates: list[str] = []
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)
    match = _FIRST_BRACE_RE.search(text)
    if match is not None:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("tool"):
            args = obj.get("args")
            return str(obj["tool"]), dict(args) if isinstance(args, dict) else {}
    return None, None


def _json_fallback_prompt(query: str) -> str:
    """Build the JSON-answer prompt for backends that ignore tools.

    Lists the ten canonical tools and asks for a strict ``{"tool", "args"}``
    object so the two Ollama variants are scored on the SAME expected tool/args
    as the native-FC variants (handicap-adjusted, never a hard zero).

    Args:
        query: The Spanish user turn.

    Returns:
        The composed prompt string (Spanish instructions).
    """
    tool_lines = "\n".join(f"  - {name}" for name in sorted(_CANONICAL_TOOLS))
    return "\n".join(
        [
            "Eres el planificador de un copiloto agricola satelital. Para la peticion "
            "del usuario, elige UNA herramienta de la lista y devuelve EXCLUSIVAMENTE "
            'un objeto JSON con la forma {"tool": <nombre>, "args": {...}} sin texto '
            "adicional ni explicaciones.",
            "",
            "Herramientas disponibles:",
            tool_lines,
            "",
            "No incluyas session_id ni la geometria del poligono en args (el sistema "
            "los inyecta). Incluye solo los argumentos relevantes (por ejemplo "
            "parcel_id, index, year, scene_id, name, models, cloud_cover_max).",
            "",
            f"Peticion del usuario: {query}",
            "",
            'Responde unicamente con el objeto JSON {"tool": ..., "args": ...}.',
        ]
    )


# ---------------------------------------------------------------------------
# Eval 1 -- tool-calling correctness
# ---------------------------------------------------------------------------
def _coerce_args(tool_name: str, raw_args: dict[str, Any]) -> dict[str, Any]:
    """Pydantic-coerce model args via ``TOOL_REGISTRY[name].input_model``.

    A dummy ``session_id`` and ``aoi`` are injected (as the agent loop does) so
    the strict ``*Input`` model can validate; on failure the raw args are
    returned untouched so the scorer still compares the load-bearing keys.

    Args:
        tool_name: The selected canonical tool name.
        raw_args: The model-provided argument mapping.

    Returns:
        The coerced argument mapping (JSON-mode dump), or ``raw_args`` when the
        strict model rejects the input.
    """
    from uuid import UUID

    spec = TOOL_REGISTRY[tool_name]
    fields = spec.input_model.model_fields
    candidate = dict(raw_args)
    if "session_id" in fields and "session_id" not in candidate:
        candidate["session_id"] = UUID("00000000-0000-0000-0000-000000000000")
    if "aoi" in fields and "aoi" not in candidate:
        candidate["aoi"] = {
            "type": "Polygon",
            "coordinates": [[[0.0, 0.0], [0.01, 0.0], [0.01, 0.01], [0.0, 0.0]]],
        }
    try:
        model = spec.input_model.model_validate(candidate)
    except Exception:  # noqa: BLE001 - coercion is best-effort for scoring
        return dict(raw_args)
    return model.model_dump(mode="json")


def _score_args(case: ToolCallCase, args: dict[str, Any]) -> float:
    """Fraction of the case's expected args satisfied (subset + fuzzy present).

    For each ``expected_args_subset`` key the coerced value must equal the
    expected value; each ``fuzzy_arg_keys`` key counts as matched when merely
    present. With no expected keys the score is ``1.0`` (vacuously satisfied).

    Args:
        case: The tool-calling case.
        args: The model args after coercion.

    Returns:
        Argument-match accuracy in ``[0, 1]``.
    """
    coerced = _coerce_args(case.expected_tool, args)
    checks: list[bool] = []
    for key, expected in case.expected_args_subset.items():
        checks.append(key in coerced and coerced[key] == expected)
    for key in case.fuzzy_arg_keys:
        checks.append(key in args or key in coerced)
    if not checks:
        return 1.0
    return sum(1.0 for ok in checks if ok) / len(checks)


#: Canonical AOI appended to spatial user turns, mirroring the geometry the real
#: chat frontend attaches when the user draws a polygon (a small box near
#: Toulouse, FR). Without it a native-FC reasoner (Gemini, Qwen-text) reasonably
#: calls ``list_parcels`` to DISCOVER the area first and is then mis-scored on the
#: first call; verified live, appending it lifts Gemini tool-selection from 0.55 to
#: 0.95. The JSON-fallback prompt already states the geometry is auto-injected, so
#: this levels the two channels and matches what the frontend really sends.
_FRONTEND_AOI_GEOMETRY: str = (
    '{"type": "Polygon", "coordinates": '
    "[[[1.30, 43.50], [1.31, 43.50], [1.31, 43.51], [1.30, 43.51], [1.30, 43.50]]]}"
)

#: Fuzzy arg keys that mark a case as spatial (the frontend attaches a geometry).
_SPATIAL_ARG_KEYS: frozenset[str] = frozenset({"aoi", "bbox"})


def _augment_query_with_context(case: ToolCallCase) -> str:
    """Append the frontend-provided AOI/bbox geometry for spatial user turns.

    The real chat frontend attaches the drawn polygon to the user message for any
    spatial request; the tool-calling eval previously sent the bare query, so a
    native-FC reasoner (Gemini, Qwen-text) reasonably called ``list_parcels`` to
    discover the area first and was mis-scored on that first call (a measurement
    artifact, not a routing weakness -- ``grounded_crop`` already embeds the AOI
    and the same reasoner routes correctly there). Cases whose ``fuzzy_arg_keys``
    include ``aoi``/``bbox`` get the geometry appended so the metric reflects the
    real agent; non-spatial cases (an explicit ``parcel_id``/``scene_id``) are
    returned unchanged.

    Args:
        case: The tool-calling case.

    Returns:
        The user query, with the AOI line appended for spatial cases.
    """
    if set(case.fuzzy_arg_keys) & _SPATIAL_ARG_KEYS:
        return f"{case.user_query}\nAOI (GeoJSON): {_FRONTEND_AOI_GEOMETRY}"
    return case.user_query


async def eval_tool_calling(
    variant: ReasonerVariant,
    cases: Sequence[ToolCallCase],
    *,
    backend: LLMBackend,
    seed: int = 0,
) -> dict[str, float | int]:
    """Evaluate one variant on tool-calling correctness (Eval 1).

    Native-FC variants (Gemini, Qwen) advertise the REAL
    :func:`build_function_declarations` and the selected tool is read from
    ``BackendChunk.function_call``. The two Ollama variants ignore tools, so they
    take the JSON ``{"tool", "args"}`` text fallback parsed by
    :func:`_parse_json_tool_answer`. Both paths are scored against the SAME
    ``expected_tool`` / ``expected_args_subset`` so the numbers stay comparable.

    Args:
        variant: The reasoner variant under test.
        cases: The tool-calling cases.
        backend: The injected backend (tests/live).
        seed: Seed tag (carried for logging/reproducibility).

    Returns:
        Metric mapping with ``tool_selection_accuracy``, ``arg_match_accuracy``,
        ``no_call_rate``, ``parse_failure_rate``, ``tool_calling_native`` and
        ``n``.
    """
    native = variant.name not in _NON_TOOL_VARIANTS
    declarations = _declarations_for_generate(build_function_declarations())
    selection_scores: list[float] = []
    arg_scores: list[float] = []
    n_no_call = 0
    n_parse_fail = 0

    for case in cases:
        tool_name: str | None
        args: dict[str, Any] | None
        # Attach the AOI/bbox geometry the real frontend sends for spatial turns
        # (no-op for cases with an explicit parcel_id/scene_id). Both channels see
        # the same context so the native-FC and JSON-fallback paths stay comparable.
        query = _augment_query_with_context(case)
        try:
            if native:
                tool_name, args, _text = await asyncio.wait_for(
                    _drive_for_call(backend, query, declarations),
                    timeout=_ITEM_TIMEOUT_S,
                )
            else:
                text = await asyncio.wait_for(
                    _drive_for_text(backend, _json_fallback_prompt(query)),
                    timeout=_ITEM_TIMEOUT_S,
                )
                tool_name, args = _parse_json_tool_answer(text)
        except Exception as exc:  # noqa: BLE001 - one case must not crash the run
            logger.warning(
                "toolcall_case_failed", variant=variant.name, case=case.id, error=str(exc)
            )
            tool_name, args = None, None

        if tool_name is None:
            n_no_call += 1
            if not native:
                n_parse_fail += 1
            selection_scores.append(0.0)
            continue

        # DRY name match: reuse agent_metrics.tool_call_accuracy on a 1-tuple.
        selected = agent_metrics.tool_call_accuracy([tool_name], [case.expected_tool])
        selection_scores.append(selected)
        if selected >= 1.0:
            arg_scores.append(_score_args(case, args or {}))

    n = len(cases)
    tool_sel = sum(selection_scores) / n if n else math.nan
    arg_acc = sum(arg_scores) / len(arg_scores) if arg_scores else math.nan
    logger.info(
        "toolcall_eval_done",
        variant=variant.name,
        seed=seed,
        tool_selection=tool_sel,
        n=n,
        native=native,
    )
    return {
        "tool_selection_accuracy": tool_sel,
        "arg_match_accuracy": arg_acc,
        "no_call_rate": (n_no_call / n) if n else math.nan,
        "parse_failure_rate": (n_parse_fail / n) if n else 0.0,
        "tool_calling_native": 1.0 if native else 0.0,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Eval 2 -- grounded crop accuracy (agent orchestration + faithful reporting)
# ---------------------------------------------------------------------------
class _StubClassifier:
    """Deterministic stand-in for ``_XgbAlphaEarthClassifier`` (Eval 2 mock).

    ``predict_proba_18`` ignores the embedding and returns the injected posterior
    aligned to the 18-class space, so the real ``classify.run`` plumbing runs
    end-to-end while only the estimator is stubbed.

    Attributes:
        class_names: The semantic18 ``{id: name}`` mapping (real names).
        probabilities: The injected ``(18,)`` posterior, in class-id order.
    """

    def __init__(self, class_names: dict[int, str], probabilities: Any) -> None:
        self.class_names = class_names
        self._probabilities = probabilities

    def predict_proba_18(self, embedding: Any) -> Any:
        """Return the injected 18-class posterior (embedding ignored)."""
        del embedding
        return self._probabilities


def _build_stub_classifier(case: CropCase) -> _StubClassifier:
    """Build the stub classifier whose argmax equals ``case.true_crop``.

    Args:
        case: The crop case carrying the injected class probabilities.

    Returns:
        A :class:`_StubClassifier` returning the aligned posterior.
    """
    import numpy as np

    from ml.data.pastis_filter import SEMANTIC18_CLASS_NAMES

    class_names = dict(SEMANTIC18_CLASS_NAMES)
    name_to_id = {name: idx for idx, name in class_names.items()}
    proba = np.zeros(len(class_names), dtype=np.float64)
    for crop, prob in case.injected_class_probabilities.items():
        idx = name_to_id.get(crop)
        if idx is not None:
            proba[idx] = float(prob)
    total = float(proba.sum())
    if total > 1e-12:
        proba = proba / total
    return _StubClassifier(class_names=class_names, probabilities=proba)


def _crop_in_answer(answer: str, crop: str) -> bool:
    """Whether the final answer names ``crop`` (whole-word, normalised).

    Matches on WORD BOUNDARIES, not raw substring, so a short crop token such as
    ``corn`` does not spuriously match inside ``popcorn``/``acorn`` and ``bean``
    not inside ``beanstalk``. Both strings are run through the agent_metrics
    normaliser first (lowercase, punctuation stripped), then the crop must appear
    as a contiguous whole-word phrase.

    Args:
        answer: The agent's final answer prose.
        crop: The crop label to look for.

    Returns:
        ``True`` when the normalised crop appears as a whole word/phrase in the
        normalised answer.
    """
    norm_answer = agent_metrics._normalize_text(answer)
    norm_crop = agent_metrics._normalize_text(crop)
    if not norm_crop:
        return False
    return re.search(rf"\b{re.escape(norm_crop)}\b", norm_answer) is not None


def _answer_signals_needs_gee(answer: str, crop_names: Sequence[str]) -> bool:
    """Whether the answer faithfully signals needs-GEE without naming a crop.

    Args:
        answer: The agent's final answer prose.
        crop_names: The 18 real crop names (must NOT appear in a faithful refusal).

    Returns:
        ``True`` when a needs-GEE phrase is present and no real crop is named.
    """
    lower = answer.lower()
    signals = any(phrase in lower for phrase in _NEEDS_GEE_PHRASES)
    names_crop = any(_crop_in_answer(answer, name) for name in crop_names)
    return signals and not names_crop


def _crop_followup_prompt(query: str, result: dict[str, Any]) -> str:
    """Build the follow-up prompt that injects the classifier output verbatim.

    Used on the JSON-fallback path (the two Ollama variants): once the model has
    routed to ``classify_new_parcel``, the REAL (stubbed) classifier result is
    injected as the tool's response and the model is asked to report it. This
    isolates FAITHFUL REPORTING -- whether the model echoes the crop the tool
    returned -- from tool selection, so the non-native variants are scored on the
    SAME contract as the native path (route, then faithfully name the crop the
    classifier returned) and never penalised for the backend ignoring ``tools``.

    Args:
        query: The Spanish user turn.
        result: The ``ClassificationResult`` dump returned by ``classify.run``
            (carries ``crop_class``, ``confidence`` and ``class_probabilities``).

    Returns:
        The composed Spanish follow-up prompt.
    """
    return "\n".join(
        [
            "Eres un copiloto agricola satelital. La herramienta de clasificacion "
            "ya se ejecuto y devolvio el siguiente resultado:",
            "",
            json.dumps(result, ensure_ascii=False),
            "",
            "Responde al usuario reportando FIELMENTE el cultivo de 'crop_class' tal "
            "como aparece en el resultado. No inventes otro cultivo ni cambies el "
            "valor; si 'crop_class' es 'needs_gee_sampling', explica que la parcela "
            "requiere muestreo GEE y NO menciones ningun cultivo.",
            "",
            f"Peticion original del usuario: {query}",
        ]
    )


async def _score_crop_case_native(
    case: CropCase,
    *,
    variant: ReasonerVariant,
    backend: LLMBackend,
    ctx: ToolContext,
    classify_spec: Any,
) -> tuple[bool, str | None, str]:
    """Drive the REAL :class:`Agent` loop for one crop case (native-FC variants).

    Runs ``Agent.stream_response`` with the single ``classify_new_parcel`` tool
    (its ``run`` executes under the per-case stubs the caller installed). The
    model must route to the tool and then name the returned crop.

    Args:
        case: The crop case under test.
        variant: The reasoner variant (for logging).
        backend: The injected native-FC backend.
        ctx: The per-case tool context.
        classify_spec: The ``classify_new_parcel`` tool spec.

    Returns:
        ``(routed, tool_crop, answer)`` -- ``routed`` is ``True`` when the loop
        emitted a ``classify_new_parcel`` call, ``tool_crop`` is the crop the
        tool returned (``None`` if it never ran), ``answer`` is the final prose.
    """
    from ml.agent.agent import Agent

    agent = Agent(backend=backend, tools=[classify_spec], instruction="")
    if hasattr(backend, "reset"):
        backend.reset()

    routed = False
    tool_crop: str | None = None
    answer_parts: list[str] = []
    # Hand the AOI + year to the model the way the frontend does when the user
    # draws a parcel: without the geometry the reasoner correctly REFUSES and asks
    # for the area (so it would never call classify), which is not what this eval
    # measures. Embedding the AOI lets it route to the tool with the right args;
    # the classifier result is still the injected/stubbed one.
    user_content = (
        f"{case.user_query}\n"
        f"AOI (GeoJSON): {json.dumps(case.aoi_geometry, ensure_ascii=False)}\n"
        f"Anio: {case.year}"
    )
    messages = [{"role": "user", "content": user_content}]
    try:
        async for event in agent.stream_response(messages, ctx.session_id, ctx):
            name = getattr(event, "name", None)
            if name == "classify_new_parcel" and type(event).__name__ == "ToolCallEvent":
                routed = True
            if type(event).__name__ == "ToolResultEvent" and getattr(event, "ok", False):
                result = getattr(event, "result", {}) or {}
                tool_crop = result.get("crop_class")
            if type(event).__name__ == "TextDeltaEvent":
                answer_parts.append(getattr(event, "text", "") or "")
    except Exception as exc:  # noqa: BLE001 - one case must not crash the run
        logger.warning("crop_case_failed", variant=variant.name, case=case.id, error=str(exc))
    return routed, tool_crop, "".join(answer_parts).strip()


async def _score_crop_case_fallback(
    case: CropCase,
    *,
    variant: ReasonerVariant,
    backend: LLMBackend,
    ctx: ToolContext,
    classify_spec: Any,
) -> tuple[bool, str | None, str]:
    """Score one crop case via the JSON tool-selection fallback (Ollama variants).

    The two Ollama backends (``gemma-base``, ``qwen36-vl``) IGNORE the ``tools``
    argument, so the real ``Agent`` loop never routes for them and would score a
    hard zero that measures the backend's missing tool API, not the model. To
    stay HONEST and COMPARABLE with the native path this mirrors
    :func:`eval_tool_calling`'s handicap-adjusted fallback in two turns:

    1. **Routing** -- the model is asked (via :func:`_json_fallback_prompt`, with
       the AOI appended) to emit ``{"tool", "args"}``; routing succeeds iff it
       selects ``classify_new_parcel`` (parsed by :func:`_parse_json_tool_answer`).
    2. **Faithful reporting** -- when (and only when) it routed, the REAL
       (stubbed) ``classify.run`` is executed directly to obtain the injected
       ``ClassificationResult``; that result is injected into a follow-up prompt
       (:func:`_crop_followup_prompt`) and the model's prose is checked for the
       crop the classifier returned. This measures AGENT ORCHESTRATION + faithful
       reporting (never the classifier, which is stubbed).

    Args:
        case: The crop case under test.
        variant: The reasoner variant (for logging).
        backend: The injected non-native backend.
        ctx: The per-case tool context (threaded into ``classify.run``).
        classify_spec: The ``classify_new_parcel`` tool spec (its ``input_model``
            + ``fn`` are reused so the same plumbing runs as on the native path).

    Returns:
        ``(routed, tool_crop, answer)`` with the same contract as
        :func:`_score_crop_case_native`.
    """
    from uuid import UUID

    aoi_line = (
        f"\nAOI (GeoJSON): {json.dumps(case.aoi_geometry, ensure_ascii=False)}\nAnio: {case.year}"
    )
    route_prompt = _json_fallback_prompt(case.user_query) + aoi_line
    if hasattr(backend, "reset"):
        backend.reset()
    try:
        route_text = await asyncio.wait_for(
            _drive_for_text(backend, route_prompt), timeout=_ITEM_TIMEOUT_S
        )
    except Exception as exc:  # noqa: BLE001 - one case must not crash the run
        logger.warning("crop_route_failed", variant=variant.name, case=case.id, error=str(exc))
        return False, None, ""
    tool_name, _args = _parse_json_tool_answer(route_text)
    routed = tool_name == "classify_new_parcel"
    if not routed:
        return False, None, ""

    # Run the REAL (stubbed) classify.run so the injected ensemble result is the
    # SAME deterministic plumbing the native path exercises.
    inp = classify_spec.input_model.model_validate(
        {
            "session_id": ctx.session_id or UUID("00000000-0000-0000-0000-000000000000"),
            "aoi": case.aoi_geometry,
            "year": case.year,
        }
    )
    try:
        result_obj = await classify_spec.fn(inp, ctx)
    except Exception as exc:  # noqa: BLE001 - one case must not crash the run
        logger.warning("crop_classify_failed", variant=variant.name, case=case.id, error=str(exc))
        return True, None, ""
    result = result_obj.model_dump(mode="json")
    tool_crop = result.get("crop_class")

    if hasattr(backend, "reset"):
        backend.reset()
    try:
        answer = await asyncio.wait_for(
            _drive_for_text(backend, _crop_followup_prompt(case.user_query, result)),
            timeout=_ITEM_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 - one case must not crash the run
        logger.warning("crop_report_failed", variant=variant.name, case=case.id, error=str(exc))
        answer = ""
    return True, tool_crop, answer


async def eval_grounded_crop(
    variant: ReasonerVariant,
    cases: Sequence[CropCase],
    *,
    backend: LLMBackend,
    make_ctx: Any,
    monkeypatch_target: Any,
    seed: int = 0,
) -> dict[str, float | int]:
    """Evaluate agent orchestration + faithful crop reporting (Eval 2).

    The REAL ``classify_new_parcel.run`` executes under a stubbed embedding fetch
    + estimator (the known ensemble result is injected per case), so the routing +
    ``ClassificationResult`` plumbing is exercised end-to-end while only the
    embedding + classifier are mocked. The reasoner must (1) route to
    ``classify_new_parcel`` and (2) faithfully name the returned crop.

    Two scoring paths keep the metric COMPARABLE across backends without
    rewarding any of them for a capability they lack:

    - **Native-FC variants** (Gemini; Qwen-text once Bug 1's JSON-Schema fix lets
      vLLM/llama.cpp accept the tools) take the REAL :class:`Agent` loop
      (:func:`_score_crop_case_native`): the model routes via the function-call
      API and the loop feeds the stubbed ``ClassificationResult`` back so the
      model can report the crop.
    - **Non-native variants** (the two Ollama backends in ``_NON_TOOL_VARIANTS``,
      which IGNORE ``tools``) take the JSON tool-selection fallback
      (:func:`_score_crop_case_fallback`), mirroring :func:`eval_tool_calling`:
      the model picks the tool from a JSON ``{"tool", "args"}`` answer (routing),
      then -- once routed -- the same stubbed ``classify.run`` is executed
      directly and its result injected into a follow-up prompt so faithful
      reporting is measured. Both paths score the SAME contract (route, then echo
      the classifier's crop); only the routing CHANNEL differs, transparently.

    This stays HONEST: it measures orchestration + faithful reporting, never the
    classifier (stubbed). The two ``_NON_TOOL_VARIANTS`` previously scored a hard
    zero here because the Agent loop dropped their ignored tools.

    Args:
        variant: The reasoner variant under test.
        cases: The crop cases.
        backend: The injected backend (scripted in CI, real live).
        make_ctx: Factory ``(session_id=..., defer=...) -> ToolContext``.
        monkeypatch_target: An object exposing ``setattr(obj, name, value)``
            (a ``pytest.MonkeyPatch`` in CI) used to stub the classify module's
            embedding fetch and classifier loader per case, then undo it.
        seed: Seed tag.

    Returns:
        Metric mapping with ``routing_accuracy``, ``crop_match_accuracy``,
        ``faithfulness_crop`` and ``n``.
    """
    import ml.agent.tools.classify as classify_mod
    from ml.agent.tools import get_tool
    from ml.data.pastis_filter import SEMANTIC18_CLASS_NAMES

    native = variant.name not in _NON_TOOL_VARIANTS
    crop_names = list(SEMANTIC18_CLASS_NAMES.values())
    classify_spec = get_tool("classify_new_parcel")
    routing_scores: list[float] = []
    crop_scores: list[float] = []
    faithfulness_scores: list[float] = []

    for case in cases:
        ctx = make_ctx()

        # Stub the embedding fetch: a fresh AOI (needs_gee) returns None; a
        # positive case returns a deterministic (64,) vector so classify.run
        # reaches the stubbed estimator.
        async def _fake_fetch(
            _ctx: ToolContext, _year: int, _aoi: Any = None, *, _case: CropCase = case
        ) -> Any:
            import numpy as np

            if _case.expects_needs_gee:
                return None
            return np.ones(64, dtype=np.float64)

        monkeypatch_target.setattr(classify_mod, "_fetch_parcel_embedding", _fake_fetch)
        monkeypatch_target.setattr(
            classify_mod, "_load_classifier", lambda c=case: _build_stub_classifier(c)
        )

        if native:
            routed, tool_crop, answer = await _score_crop_case_native(
                case, variant=variant, backend=backend, ctx=ctx, classify_spec=classify_spec
            )
        else:
            routed, tool_crop, answer = await _score_crop_case_fallback(
                case, variant=variant, backend=backend, ctx=ctx, classify_spec=classify_spec
            )

        routing_scores.append(1.0 if routed else 0.0)

        if case.expects_needs_gee:
            tool_ok = tool_crop == _NEEDS_GEE_SENTINEL
            answer_ok = _answer_signals_needs_gee(answer, crop_names)
            crop_scores.append(1.0 if (routed and tool_ok and answer_ok) else 0.0)
            no_crop_named = not _names_any_crop(answer, crop_names)
            faithfulness_scores.append(1.0 if tool_ok and no_crop_named else 0.0)
        else:
            tool_ok = tool_crop == case.true_crop
            answer_ok = _crop_in_answer(answer, case.true_crop)
            crop_scores.append(1.0 if (routed and tool_ok and answer_ok) else 0.0)
            # Faithfulness = "does the prose name the crop the TOOL returned".
            # It is only defined when the tool actually returned a crop, so the
            # positive branch scores it conditionally on ``tool_crop`` (an agent
            # that never called the tool has no tool output to be faithful to and
            # is already penalised by routing/crop accuracy). The needs-GEE branch
            # always has a tool output (the sentinel), hence its unconditional
            # append -- the two denominators differ on purpose, by definition.
            if tool_crop is not None:
                faithfulness_scores.append(1.0 if _crop_in_answer(answer, tool_crop) else 0.0)

    n = len(cases)
    routing = sum(routing_scores) / n if n else math.nan
    crop_match = sum(crop_scores) / n if n else math.nan
    faithfulness = (
        sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else math.nan
    )
    logger.info(
        "crop_eval_done",
        variant=variant.name,
        seed=seed,
        routing_accuracy=routing,
        crop_match_accuracy=crop_match,
        n=n,
    )
    return {
        "routing_accuracy": routing,
        "crop_match_accuracy": crop_match,
        "faithfulness_crop": faithfulness,
        "n": n,
    }


def _names_any_crop(answer: str, crop_names: Sequence[str]) -> bool:
    """Whether the answer names any of the 18 real crops (drift/fabrication check).

    Args:
        answer: The agent's final answer prose.
        crop_names: The 18 real crop names.

    Returns:
        ``True`` when at least one real crop name appears in the answer.
    """
    return any(_crop_in_answer(answer, name) for name in crop_names)


# ---------------------------------------------------------------------------
# Eval 3 -- RAG A/B hallucination
# ---------------------------------------------------------------------------
@dataclass
class _RagSettings:
    """Minimal settings stub exposing only the ``rag_enabled`` gate.

    Attributes:
        rag_enabled: The A/B factor flipped between the grounded (A) and
            ungrounded (B) runs.
    """

    rag_enabled: bool = False


def _build_rag_prompt(question: str, grounding_text: str) -> str:
    """Compose the reasoner prompt for a RAG run.

    On the A run the injected ``grounding_text`` is prepended as context; on the B
    run it is empty, so the question is unanswerable from grounding and any
    specific claim is a hallucination.

    Args:
        question: The Spanish question.
        grounding_text: The grounding block (empty on the B/ungrounded run).

    Returns:
        The composed prompt string.
    """
    lines = [
        "Eres un copiloto agricola. Responde la pregunta del usuario usando "
        "UNICAMENTE el contexto recuperado. Si el contexto no contiene el dato, "
        "responde exactamente 'No tengo ese dato'. No inventes cifras ni cultivos.",
        "",
    ]
    if grounding_text:
        lines.extend(["Contexto recuperado:", grounding_text, ""])
    else:
        lines.extend(["Contexto recuperado: (vacio)", ""])
    lines.append(f"Pregunta: {question}")
    return "\n".join(lines)


async def _run_rag_side(
    backend: LLMBackend,
    cases: Sequence[RagCase],
    *,
    grounded: bool,
    judge: HallucinationJudge | None,
) -> float:
    """Run one A/B side (grounded or ungrounded) and return its hallucination rate.

    Builds DeepEval-shaped samples ``{input, actual_output, context}`` and scores
    them with :func:`agent_metrics.hallucination_rate` (the injectable judge). On
    the grounded side the context is the record's ``grounding_text``; on the
    ungrounded side the model gets no grounding and the context is the
    gold-grounded answer (so a specific claim with no grounding scores as a
    hallucination against the judge).

    Args:
        backend: The injected reasoner backend.
        cases: The RAG cases.
        grounded: ``True`` for the A (grounding injected) run, ``False`` for B.
        judge: The injectable hallucination judge (``None`` -> NaN).

    Returns:
        The mean hallucination rate for this side (NaN when no judge).
    """
    samples: list[dict[str, Any]] = []
    for case in cases:
        grounding = case.grounding_text if grounded else ""
        prompt = _build_rag_prompt(case.question, grounding)
        if hasattr(backend, "reset"):
            backend.reset()
        try:
            answer = await asyncio.wait_for(
                _drive_for_text(backend, prompt), timeout=_ITEM_TIMEOUT_S
            )
        except Exception as exc:  # noqa: BLE001 - one case must not crash the run
            logger.warning("rag_case_failed", case=case.id, grounded=grounded, error=str(exc))
            answer = ""
        context = case.grounding_text if grounded else case.gold_grounded_answer
        samples.append({"input": case.question, "actual_output": answer, "context": [context]})
    return agent_metrics.hallucination_rate(samples, judge)


async def eval_rag_ab(
    variant: ReasonerVariant,
    cases: Sequence[RagCase],
    *,
    backend: LLMBackend,
    judge: HallucinationJudge | None = None,
    seed: int = 0,
) -> dict[str, float | int]:
    """Evaluate the RAG A/B hallocination delta for one variant (Eval 3).

    Runs each question twice (B = ungrounded, A = grounded with the injected
    ``grounding_text``), scores both with the injectable judge and reports the
    delta ``B - A``. The grounding injection happens at the prompt boundary here
    (the deterministic offline analogue of monkeypatching ``spatial_rag``), so no
    pgvector/PostGIS is touched.

    Honest framing (do NOT over-claim): the delta is the reduction in
    hallucination RELATIVE TO whatever the ungrounded model does. It is only
    positive when the ungrounded side actually fabricates; a reasoner that
    correctly refuses ("No tengo ese dato") when ungrounded yields a small or
    zero delta even though RAG is working as designed. Both raw rates
    (``hallucination_rate_ungrounded`` and ``_grounded``) are therefore reported
    SEPARATELY -- the delta is a derived convenience, not the headline, and must
    be read together with the ungrounded fabrication rate, not alone.

    Args:
        variant: The reasoner variant under test.
        cases: The RAG cases.
        backend: The injected reasoner backend.
        judge: The injectable hallucination judge (``None`` -> NaN, rendered n/a).
        seed: Seed tag.

    Returns:
        Metric mapping with ``hallucination_rate_ungrounded``,
        ``hallucination_rate_grounded``, ``hallucination_reduction_delta``,
        ``faithfulness_grounded`` and ``n``.
    """
    ungrounded = await _run_rag_side(backend, cases, grounded=False, judge=judge)
    grounded = await _run_rag_side(backend, cases, grounded=True, judge=judge)
    delta = (
        ungrounded - grounded if not (math.isnan(ungrounded) or math.isnan(grounded)) else math.nan
    )
    faithfulness = (1.0 - grounded) if not math.isnan(grounded) else math.nan
    logger.info(
        "rag_eval_done",
        variant=variant.name,
        seed=seed,
        hallucination_ungrounded=ungrounded,
        hallucination_grounded=grounded,
        delta=delta,
        n=len(cases),
    )
    return {
        "hallucination_rate_ungrounded": ungrounded,
        "hallucination_rate_grounded": grounded,
        "hallucination_reduction_delta": delta,
        "faithfulness_grounded": faithfulness,
        "n": len(cases),
    }


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
def _validate_geometries(crop_cases: Sequence[CropCase], rag_cases: Sequence[RagCase]) -> None:
    """Fail fast if any dataset geometry is not a valid ``GeoJSONGeometry``.

    Args:
        crop_cases: The crop cases whose ``aoi_geometry`` must validate.
        rag_cases: The RAG cases whose ``aoi_geometry`` must validate.

    Raises:
        ValueError: When a geometry fails ``GeoJSONGeometry`` validation.
    """
    for case in crop_cases:
        try:
            GeoJSONGeometry.model_validate(case.aoi_geometry)
        except Exception as exc:  # re-raised with case context
            raise ValueError(f"crop case {case.id!r} has invalid geometry: {exc}") from exc
    for rag_case in rag_cases:
        try:
            GeoJSONGeometry.model_validate(rag_case.aoi_geometry)
        except Exception as exc:  # re-raised with case context
            raise ValueError(f"rag case {rag_case.id!r} has invalid geometry: {exc}") from exc


def run_system_eval(
    variants: Sequence[ReasonerVariant],
    *,
    seeds: Sequence[int] = (0, 1, 2),
    toolcall_backends: dict[str, LLMBackend] | None = None,
    crop_backends: dict[str, LLMBackend] | None = None,
    rag_backends: dict[str, LLMBackend] | None = None,
    make_ctx: Any = None,
    monkeypatch_target: Any = None,
    judge: HallucinationJudge | None = None,
    toolcall_path: Path = DEFAULT_TOOLCALL_PATH,
    crop_path: Path = DEFAULT_CROP_PATH,
    rag_path: Path = DEFAULT_RAG_PATH,
) -> dict[str, Any]:
    """Run the three system evals for every variant over the seeds and aggregate.

    Mirrors :func:`ml.eval.agent_bench.run_benchmark`: per-variant injectable
    backends, ``mean +- std`` over seeds (reusing ``agent_bench._aggregate``),
    structlog. Backends are injected per eval (so CI scripts a deterministic
    ``FakeBackend`` per variant and live builds them with ``make_backend``); when
    a backend is missing for a (variant, eval) that eval is skipped for the
    variant. Eval 2 additionally requires ``make_ctx`` + ``monkeypatch_target``;
    when absent it is skipped (it runs the real ``classify.run`` under stubs).

    Args:
        variants: The reasoner variants to evaluate.
        seeds: Evaluation seeds for the error bars.
        toolcall_backends: ``{variant_name: backend}`` for Eval 1.
        crop_backends: ``{variant_name: backend}`` for Eval 2.
        rag_backends: ``{variant_name: backend}`` for Eval 3.
        make_ctx: ``ToolContext`` factory required by Eval 2.
        monkeypatch_target: A ``pytest.MonkeyPatch``-like object required by
            Eval 2 to stub the classify module.
        judge: The injectable hallucination judge for Eval 3.
        toolcall_path: Path to the tool-calling dataset.
        crop_path: Path to the grounded-crop dataset.
        rag_path: Path to the RAG dataset.

    Returns:
        Nested mapping ``{variant: {eval: {metric: {"mean", "std"}}}}``.
    """
    toolcall_backends = toolcall_backends or {}
    crop_backends = crop_backends or {}
    rag_backends = rag_backends or {}

    toolcall_cases = load_toolcall_cases(toolcall_path)
    crop_cases = load_crop_cases(crop_path)
    rag_cases = load_rag_cases(rag_path)
    _validate_geometries(crop_cases, rag_cases)

    # Evaluate the paid cloud variant (Gemini) first so it is never recomputed
    # by a later failure (mirrors agent_bench ordering).
    ordered = sorted(variants, key=lambda v: 0 if "gemini" in v.name.lower() else 1)

    results: dict[str, Any] = {}
    for variant in ordered:
        per_variant: dict[str, Any] = {}

        tc_backend = toolcall_backends.get(variant.name)
        if tc_backend is not None:
            seed_metrics: list[dict[str, float]] = []
            for seed in seeds:
                if hasattr(tc_backend, "reset"):
                    tc_backend.reset()
                seed_metrics.append(
                    asyncio.run(
                        eval_tool_calling(variant, toolcall_cases, backend=tc_backend, seed=seed)
                    )
                )
            per_variant["tool_calling"] = _aggregate(seed_metrics)

        crop_backend = crop_backends.get(variant.name)
        if crop_backend is not None and make_ctx is not None and monkeypatch_target is not None:
            seed_metrics = []
            for seed in seeds:
                seed_metrics.append(
                    asyncio.run(
                        eval_grounded_crop(
                            variant,
                            crop_cases,
                            backend=crop_backend,
                            make_ctx=make_ctx,
                            monkeypatch_target=monkeypatch_target,
                            seed=seed,
                        )
                    )
                )
            per_variant["grounded_crop"] = _aggregate(seed_metrics)

        rag_backend = rag_backends.get(variant.name)
        if rag_backend is not None:
            seed_metrics = []
            for seed in seeds:
                if hasattr(rag_backend, "reset"):
                    rag_backend.reset()
                seed_metrics.append(
                    asyncio.run(
                        eval_rag_ab(variant, rag_cases, backend=rag_backend, judge=judge, seed=seed)
                    )
                )
            per_variant["rag_ab"] = _aggregate(seed_metrics)

        results[variant.name] = per_variant
        logger.info("system_eval_variant_done", variant=variant.name, evals=sorted(per_variant))

    logger.info("system_eval_done", variants=[v.name for v in ordered])
    return results
