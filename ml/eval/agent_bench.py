"""Agent benchmark orchestrator for AgroSatCopilot (US-049).

This module is the harness that evaluates the conversational copilot variants
against two public benchmarks, **EVAL-ONLY** (it never trains: AgroMind ships
no train split, so any fine-tune would be leakage, see AC-3 / ADR-009). It only
runs inference, parses the model output and scores it with the pure metrics in
:mod:`ml.eval.agent_metrics`, then renders the comparison report via
:mod:`ml.eval.agent_report`.

Pieces:

- Data models: :class:`AgroMindItem`, :class:`GeoTask`, :class:`ReasonerVariant`.
- Loaders: :func:`load_agromind_subset` (the real 500-item JSON subset) and
  :func:`load_geoanalystbench` (the real 50-task CSV, read with Polars).
- Runners: :func:`eval_agromind` (multiple-choice QA -> letter -> exact match,
  plus the textual proxies and the optional LLM-as-judge hallucination rate) and
  :func:`eval_geoanalyst` (plan-and-react -> workflow + code -> semantic
  similarity vs the human workflow and simplified CodeBLEU vs the reference).
- Aggregator + entry point: :func:`run_benchmark` runs every variant over the
  seeds, aggregates ``mean +- std``, optionally logs to MLflow with the
  ``code_version`` + ``data_version`` tags (lineage on ``:5010``) and builds the
  HTML report; :func:`main` is the argparse CLI.

Multimodality tension (documented, AC-3 / plan Section 3): AgroMind is
multimodal. ``gemini`` and ``gemma-base`` are multimodal and evaluate the full
subset; ``qwen`` is **text-only** (it is the on-prem reasoner, not a VLM), so it
SKIPS the multimodal items (those with a base image or image options) and is
scored only on the purely-textual subset, with ``n_skipped`` reported so the
limitation is explicit and never papered over. GeoAnalystBench is 100% text, so
every variant runs it in full.

Backends are **injectable** (``backends`` / ``backend`` parameters) so the whole
harness runs in tests with mocks and zero network. When no backend is injected
one is built with :func:`ml.agent.backends.make_backend`. The real Qwen run
depends on the on-prem vLLM endpoint of US-048 (currently blocked): when that
endpoint exists the same harness scores Qwen unchanged; Gemini / Gemma run via
their cloud API.

Project conventions: identifiers and docstrings in English (Google style),
visible prose (CLI help, the report) in Spanish; ``structlog`` (never
``print`` in logic); Polars for the tabular load; full type hints; no emojis.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl
import structlog

from ml.eval import agent_metrics
from ml.eval.agent_report import DEFAULT_REPORT_DIR, build_report_html

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from ml.agent.backends import LLMBackend
    from ml.eval.agent_metrics import HallucinationJudge

logger = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_AGROMIND_PATH",
    "DEFAULT_GEO_PATH",
    "AgroMindItem",
    "AgroMindResult",
    "GeoResult",
    "GeoTask",
    "ReasonerVariant",
    "eval_agromind",
    "eval_geoanalyst",
    "load_agromind_subset",
    "load_geoanalystbench",
    "main",
    "run_benchmark",
]

#: Default location of the real AgroMind 500-item stratified subset.
DEFAULT_AGROMIND_PATH: Path = Path("data/agromind/agromind_subset_500.json")

#: Default location of the real GeoAnalystBench 50-task CSV.
DEFAULT_GEO_PATH: Path = Path("data/geoanalystbench/GeoAnalystBench.csv")

#: Default base folder where the subset images are extracted (see
#: ``scripts/download_agromind_images.py``). Used to resolve ``image_path`` for
#: the multimodal variants; absent files degrade to a text-only prompt.
DEFAULT_IMAGE_ROOT: Path = Path("data/agromind/images")

#: Pass threshold for a GeoAnalystBench task: a task counts as passed when its
#: workflow semantic similarity to the human-designed workflow exceeds this.
#: Documented threshold (plan Section 4): the rubric pass-rate target (>= 0.65)
#: is applied on top of the per-task pass decision made here.
GEO_PASS_THRESHOLD: float = 0.5

#: The three default reasoner variants (AC-1). ``multimodal`` gates whether a
#: variant may consume AgroMind images; ``qwen`` is text-only on purpose.
DEFAULT_VARIANTS: tuple[ReasonerVariant, ...]  # populated after the dataclass.

#: Image-file suffixes recognised when deciding whether an option is an image.
_IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

#: MLflow experiment name for this benchmark (AC-6).
_EXPERIMENT_NAME: str = "us049_agent_bench"


@dataclass(frozen=True)
class AgroMindItem:
    """One AgroMind multiple-choice QA item (the real subset schema).

    Attributes:
        image_path: Relative path to the question's base image (e.g.
            ``./Rural/piece_images/x.png``); empty when the item has no base
            image. Resolved against :data:`DEFAULT_IMAGE_ROOT` at eval time.
        question: The question text.
        options: Mapping of choice label (``A``-``D``) to its value. Values are
            either answer text or, for multi-image items, a relative image path.
        answer: The gold choice letter (``A``-``D``).
        type_id: AgroMind question-type id.
        item_id: AgroMind item id within its task file.
        level1_id: Top-level taxonomy id (used for the stratified subset).
        level2_id: Second-level taxonomy id.
        level3_id: Third-level taxonomy id.
        task_file: The source QA task file tag (e.g. ``"BD"``).
        is_multimodal: ``True`` when answering requires an image -- either the
            item has a base ``image_path`` or any option value is an image path.
            Text-only reasoners (Qwen) skip these.
    """

    image_path: str
    question: str
    options: dict[str, str]
    answer: str
    type_id: int
    item_id: int
    level1_id: int
    level2_id: int
    level3_id: int
    task_file: str
    is_multimodal: bool

    @property
    def option_image_paths(self) -> dict[str, str]:
        """Return the subset of options whose value is an image path.

        Returns:
            A mapping ``{label: relative_image_path}`` for image-valued options
            (empty when the options are plain text).
        """
        return {
            label: value
            for label, value in self.options.items()
            if _is_image_path(value)
        }


@dataclass(frozen=True)
class GeoTask:
    """One GeoAnalystBench plan-and-react task (the real CSV schema).

    Attributes:
        id: Task id (``"1"`` .. ``"50"``).
        task: Short task title.
        instruction: The full instruction handed to the reasoner.
        domain_knowledge: Background domain knowledge for the task.
        dataset_description: Description of the available datasets.
        human_workflow: The gold human-designed workflow (numbered steps),
            used as the reference for :func:`workflow_semantic_similarity`.
        code_string: The reference Python code, used for the simplified
            CodeBLEU.
        task_length: The task length (number of expected steps), as a string.
    """

    id: str
    task: str
    instruction: str
    domain_knowledge: str
    dataset_description: str
    human_workflow: str
    code_string: str
    task_length: str


@dataclass(frozen=True)
class ReasonerVariant:
    """A reasoner under evaluation.

    Attributes:
        name: Variant tag, one of ``"gemini"``, ``"qwen"``, ``"gemma-base"``.
        model: The concrete model id passed to :func:`make_backend` (or the
            multimodal API).
        multimodal: Whether the variant can consume images. ``qwen`` is
            text-only, so it is ``False`` and skips multimodal AgroMind items.
    """

    name: str
    model: str
    multimodal: bool


# Populated here (after the dataclass is defined) so the module exposes the
# canonical three variants used by the CLI and the rubric targets.
DEFAULT_VARIANTS = (
    ReasonerVariant(name="gemini", model="gemini-3.5-flash", multimodal=True),
    ReasonerVariant(name="qwen", model="qwen35", multimodal=False),
    ReasonerVariant(name="gemma-base", model="gemma4:26b-a4b-it-q4_K_M", multimodal=True),
)

#: Variant lookup by tag for the CLI.
_VARIANTS_BY_NAME: dict[str, ReasonerVariant] = {v.name: v for v in DEFAULT_VARIANTS}


@dataclass
class AgroMindResult:
    """Per-seed AgroMind scores for one variant.

    Attributes:
        exact_match: Mean exact-match over the evaluated items.
        f1_squad: Mean SQuAD-style token F1 over the textual answers.
        bertscore: Semantic-proxy BERTScore F1 over the textual answers.
        hallucination: Mean hallucination rate (NaN when no judge is given).
        n_evaluated: Number of items actually scored.
        n_skipped: Number of items skipped (text-only variant on multimodal).
    """

    exact_match: float
    f1_squad: float
    bertscore: float
    hallucination: float
    n_evaluated: int
    n_skipped: int

    def as_metrics(self) -> dict[str, float]:
        """Return the per-metric mapping consumed by the aggregator.

        Returns:
            A mapping of metric name to its scalar value for this seed.
        """
        return {
            "exact_match": self.exact_match,
            "f1_squad": self.f1_squad,
            "bertscore": self.bertscore,
            "hallucination": self.hallucination,
            "n_evaluated": float(self.n_evaluated),
            "n_skipped": float(self.n_skipped),
        }


@dataclass
class GeoResult:
    """Per-seed GeoAnalystBench scores for one variant.

    Attributes:
        pass_rate: Fraction of tasks whose workflow similarity passed the
            threshold (the rubric headline metric for this benchmark).
        mean_semantic_sim: Mean workflow semantic similarity over tasks.
        mean_codebleu: Mean simplified CodeBLEU over tasks.
        n: Number of tasks evaluated.
    """

    pass_rate: float
    mean_semantic_sim: float
    mean_codebleu: float
    n: int

    def as_metrics(self) -> dict[str, float]:
        """Return the per-metric mapping consumed by the aggregator.

        Returns:
            A mapping of metric name to its scalar value for this seed.
        """
        return {
            "pass_rate": self.pass_rate,
            "mean_semantic_sim": self.mean_semantic_sim,
            "mean_codebleu": self.mean_codebleu,
            "n": float(self.n),
        }


def _is_image_path(value: str) -> bool:
    """Return whether an option value is an image path rather than answer text.

    Args:
        value: The option value (answer text or a relative image path).

    Returns:
        ``True`` when the lowercased value ends with a known image suffix.
    """
    return isinstance(value, str) and value.lower().endswith(_IMAGE_SUFFIXES)


def _coerce_int(value: Any, default: int = -1) -> int:
    """Best-effort integer coercion for the (mostly-int) AgroMind id fields.

    Args:
        value: The raw value from the JSON (int, str or ``None``).
        default: Value returned when coercion is not possible.

    Returns:
        The integer value, or ``default`` when ``value`` is missing/invalid.
    """
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_agromind_subset(path: Path) -> list[AgroMindItem]:
    """Load and parse the real AgroMind 500-item subset JSON.

    Each record is mapped to an :class:`AgroMindItem`. An item is marked
    :attr:`~AgroMindItem.is_multimodal` when it carries a base ``image_path`` or
    any of its options is an image path, so the text-only variant can skip it.

    Args:
        path: Path to ``agromind_subset_500.json``.

    Returns:
        The list of parsed items (one per record, order preserved).
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items: list[AgroMindItem] = []
    for record in raw:
        options = {
            str(label): str(value)
            for label, value in (record.get("options") or {}).items()
        }
        image_path = str(record.get("image_path") or "").strip()
        has_option_image = any(_is_image_path(v) for v in options.values())
        is_multimodal = bool(image_path) or has_option_image
        items.append(
            AgroMindItem(
                image_path=image_path,
                question=str(record.get("question") or ""),
                options=options,
                answer=str(record.get("answer") or "").strip(),
                type_id=_coerce_int(record.get("type_id")),
                item_id=_coerce_int(record.get("item_id")),
                level1_id=_coerce_int(record.get("level1_id")),
                level2_id=_coerce_int(record.get("level2_id")),
                level3_id=_coerce_int(record.get("level3_id")),
                task_file=str(record.get("_task_file") or ""),
                is_multimodal=is_multimodal,
            )
        )
    n_multimodal = sum(1 for it in items if it.is_multimodal)
    logger.info(
        "agromind_subset_loaded",
        path=str(path),
        n_items=len(items),
        n_multimodal=n_multimodal,
        n_textual=len(items) - n_multimodal,
    )
    return items


def load_geoanalystbench(csv: Path) -> list[GeoTask]:
    """Load and parse the real GeoAnalystBench 50-task CSV with Polars.

    Read with ``infer_schema_length=0`` so every column stays ``Utf8`` (the CSV
    mixes numbered workflows, multiline code and ids). Rows with an empty ``id``
    are dropped (the file carries one trailing blank row), yielding the 50 tasks.

    Args:
        csv: Path to ``GeoAnalystBench.csv``.

    Returns:
        The list of parsed :class:`GeoTask` (one per non-empty row).
    """
    frame = pl.read_csv(csv, infer_schema_length=0)
    tasks: list[GeoTask] = []
    for row in frame.iter_rows(named=True):
        task_id = (row.get("id") or "").strip()
        if not task_id:
            continue
        tasks.append(
            GeoTask(
                id=task_id,
                task=(row.get("Task") or "").strip(),
                instruction=(row.get("Instruction") or "").strip(),
                domain_knowledge=(row.get("Domain Knowledge") or "").strip(),
                dataset_description=(row.get("Dataset Description") or "").strip(),
                human_workflow=(row.get("Human Designed Workflow") or "").strip(),
                code_string=(row.get("CodeString") or "").strip(),
                task_length=(row.get("Task Length") or "").strip(),
            )
        )
    logger.info("geoanalystbench_loaded", path=str(csv), n_tasks=len(tasks))
    return tasks


def _build_agromind_prompt(item: AgroMindItem, *, with_images: bool) -> str:
    """Build the textual prompt for an AgroMind item.

    Renders the question and the labelled options. For multi-image options the
    value is shown as a reference path when ``with_images`` is ``False`` (the
    text-only variant) and as a marker the multimodal path fills in otherwise.
    The instruction pins the output to a single choice letter so the parser can
    recover it deterministically.

    Args:
        item: The AgroMind item.
        with_images: Whether the caller will also attach the images.

    Returns:
        The composed prompt string.
    """
    lines = [
        "Eres un evaluador experto. Responde la siguiente pregunta de opcion "
        "multiple eligiendo UNA sola letra (A, B, C o D).",
        "",
        f"Pregunta: {item.question}",
        "",
        "Opciones:",
    ]
    for label in sorted(item.options):
        value = item.options[label]
        if _is_image_path(value):
            shown = f"[imagen {Path(value).name}]" if with_images else f"[imagen: {value}]"
        else:
            shown = value
        lines.append(f"  {label}. {shown}")
    lines.extend(
        [
            "",
            "Responde unicamente con la letra de la opcion correcta.",
        ]
    )
    return "\n".join(lines)


def _resolve_image(rel_path: str, image_root: Path) -> Path | None:
    """Resolve an AgroMind relative image path under the local image root.

    Args:
        rel_path: The ``./Category/...`` relative path from the subset.
        image_root: The base folder where subset images were extracted.

    Returns:
        The resolved path when the file exists locally, else ``None``.
    """
    if not rel_path:
        return None
    cleaned = rel_path.lstrip("./").replace("\\", "/")
    candidate = image_root / cleaned
    return candidate if candidate.exists() else None


def _image_part(path: Path) -> Any:
    """Build a ``google.genai`` image part from a local image file.

    Args:
        path: Local path to an existing image file.

    Returns:
        A ``types.Part`` carrying the image bytes (PNG/JPEG inferred).
    """
    from google.genai import types

    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime)


def _build_contents(
    prompt: str, image_parts: Sequence[Any]
) -> list[Any]:
    """Build a single-user-turn ``contents`` list for the backend.

    Args:
        prompt: The textual prompt.
        image_parts: Zero or more image parts to attach before the text.

    Returns:
        A one-element list with a ``types.Content`` user turn.
    """
    from google.genai import types

    parts = [*image_parts, types.Part.from_text(text=prompt)]
    return [types.Content(role="user", parts=parts)]


async def _run_backend_text(
    backend: LLMBackend, prompt: str, image_parts: Sequence[Any]
) -> str:
    """Drive a backend for one non-streaming text answer (no tools).

    Consumes the backend's chunk stream and concatenates the text deltas. Tool
    calls are not requested here (the benchmark asks for a direct answer), so any
    function-call chunk is ignored.

    Args:
        backend: The injected or constructed :class:`LLMBackend`.
        prompt: The user prompt.
        image_parts: Image parts to attach (empty for text-only).

    Returns:
        The concatenated answer text (stripped).
    """
    contents = _build_contents(prompt, image_parts)
    buffer: list[str] = []
    async for chunk in backend.generate_stream(
        contents=contents, tools=[], system_instruction=""
    ):
        text = getattr(chunk, "text", None)
        if text:
            buffer.append(text)
    return "".join(buffer).strip()


def _resolve_backend(
    variant: ReasonerVariant, backend: LLMBackend | None
) -> LLMBackend:
    """Return the backend to use for a variant, building one if not injected.

    Args:
        variant: The reasoner variant.
        backend: An injected backend (tests / explicit wiring) or ``None``.

    Returns:
        The injected backend, or one built with :func:`make_backend`.
    """
    if backend is not None:
        return backend
    from ml.agent.backends import make_backend

    # Pass Settings so the Gemini/vLLM credentials from .env.local reach the
    # backend (they are NOT exported to os.environ for the SDK to auto-discover).
    try:
        from backend.app.core.config import get_settings

        settings = get_settings()
    except Exception:  # noqa: BLE001 - settings optional (tests inject the backend)
        settings = None
    return make_backend(variant.model, settings)


async def eval_agromind(
    variant: ReasonerVariant,
    items: Sequence[AgroMindItem],
    *,
    backend: LLMBackend | None = None,
    judge: HallucinationJudge | None = None,
    seed: int = 0,
    image_root: Path = DEFAULT_IMAGE_ROOT,
) -> dict[str, float | int]:
    """Evaluate one variant on AgroMind (multiple-choice QA).

    For each item it builds the prompt (question + options), attaches the images
    when the variant is multimodal and the files are present, runs the backend,
    parses the chosen letter and scores exact-match vs the gold answer. The
    textual proxies (:func:`f1_squad`, :func:`bertscore_f1`) are computed over
    the rendered prediction vs the gold option text, and the optional
    LLM-as-judge :func:`hallucination_rate` over the answer samples.

    Text-only variants (``variant.multimodal`` is ``False``) SKIP every
    multimodal item (base image or image options) and report ``n_skipped`` so
    the limitation is explicit (AC-3 / plan Section 3); they are never scored as
    if they had seen the image.

    Args:
        variant: The reasoner variant under test.
        items: The AgroMind items to evaluate.
        backend: Injected backend (tests / explicit wiring); built lazily when
            ``None`` via :func:`make_backend`.
        judge: Injectable hallucination judge; ``None`` reports hallucination as
            NaN (rendered ``n/a``).
        seed: Seed tag (carried for logging / reproducibility of the run id).
        image_root: Base folder for resolving the subset images.

    Returns:
        A mapping ``{exact_match, f1_squad, bertscore, hallucination,
        n_evaluated, n_skipped}``.
    """
    resolved_backend = _resolve_backend(variant, backend)
    em_scores: list[float] = []
    pred_texts: list[str] = []
    gold_texts: list[str] = []
    judge_samples: list[dict[str, Any]] = []
    n_skipped = 0

    for item in items:
        if item.is_multimodal and not variant.multimodal:
            n_skipped += 1
            continue

        image_parts: list[Any] = []
        if variant.multimodal:
            base = _resolve_image(item.image_path, image_root)
            if base is not None:
                image_parts.append(_image_part(base))
            for opt_path in item.option_image_paths.values():
                resolved = _resolve_image(opt_path, image_root)
                if resolved is not None:
                    image_parts.append(_image_part(resolved))

        prompt = _build_agromind_prompt(item, with_images=bool(image_parts))
        try:
            answer = await _run_backend_text(resolved_backend, prompt, image_parts)
        except Exception as exc:  # noqa: BLE001 - one item must not crash the run
            logger.warning(
                "agromind_item_failed",
                variant=variant.name,
                item_id=item.item_id,
                error=str(exc),
            )
            answer = ""

        em_scores.append(agent_metrics.exact_match(answer, item.answer))
        gold_text = item.options.get(item.answer, item.answer)
        pred_texts.append(answer)
        gold_texts.append(gold_text)
        judge_samples.append(
            {
                "input": item.question,
                "actual_output": answer,
                "context": [gold_text],
            }
        )

    n_evaluated = len(em_scores)
    exact = sum(em_scores) / n_evaluated if n_evaluated else 0.0
    f1 = (
        sum(agent_metrics.f1_squad(p, g) for p, g in zip(pred_texts, gold_texts, strict=True))
        / n_evaluated
        if n_evaluated
        else 0.0
    )
    bert = agent_metrics.bertscore_f1(pred_texts, gold_texts) if n_evaluated else 0.0
    halluc = agent_metrics.hallucination_rate(judge_samples, judge)

    logger.info(
        "agromind_eval_done",
        variant=variant.name,
        seed=seed,
        exact_match=exact,
        n_evaluated=n_evaluated,
        n_skipped=n_skipped,
    )
    return {
        "exact_match": exact,
        "f1_squad": f1,
        "bertscore": bert,
        "hallucination": halluc,
        "n_evaluated": n_evaluated,
        "n_skipped": n_skipped,
    }


def _build_geo_prompt(task: GeoTask) -> str:
    """Build the plan-and-react prompt for a GeoAnalystBench task.

    Asks the reasoner for a numbered workflow followed by a Python code block,
    in two clearly delimited sections so the response can be split for scoring.

    Args:
        task: The GeoAnalystBench task.

    Returns:
        The composed prompt string.
    """
    return "\n".join(
        [
            "Eres un analista geoespacial. Resuelve la siguiente tarea en dos "
            "secciones.",
            "Primero, un flujo de trabajo numerado paso a paso bajo el "
            "encabezado 'WORKFLOW:'.",
            "Despues, el codigo Python completo bajo el encabezado 'CODE:' "
            "dentro de un bloque ```python```.",
            "",
            f"Tarea: {task.task}",
            "",
            f"Instruccion: {task.instruction}",
            "",
            f"Conocimiento de dominio: {task.domain_knowledge}",
            "",
            f"Descripcion de datos: {task.dataset_description}",
        ]
    )


def _split_workflow_and_code(answer: str) -> tuple[str, str]:
    """Split a plan-and-react answer into its workflow and code sections.

    Recognises a fenced ```python``` (or bare ```) code block as the code, and
    treats the remaining text (with any ``WORKFLOW:`` / ``CODE:`` headers
    stripped) as the workflow. Falls back gracefully when the response is not
    perfectly formatted.

    Args:
        answer: The raw model answer.

    Returns:
        A ``(workflow_text, code_text)`` tuple.
    """
    code = ""
    workflow = answer
    fence = "```"
    if fence in answer:
        first = answer.find(fence)
        rest = answer[first + len(fence):]
        end = rest.find(fence)
        block = rest if end == -1 else rest[:end]
        if block.lower().startswith("python"):
            block = block[len("python"):]
        code = block.strip()
        workflow = answer[:first]
    for header in ("WORKFLOW:", "CODE:", "Workflow:", "Code:"):
        workflow = workflow.replace(header, " ")
    return workflow.strip(), code.strip()


async def eval_geoanalyst(
    variant: ReasonerVariant,
    tasks: Sequence[GeoTask],
    *,
    backend: LLMBackend | None = None,
    seed: int = 0,
    pass_threshold: float = GEO_PASS_THRESHOLD,
) -> dict[str, float | int]:
    """Evaluate one variant on GeoAnalystBench (plan-and-react).

    For each task the reasoner receives the instruction and returns a workflow +
    Python code. The workflow is scored against the human-designed workflow with
    :func:`workflow_semantic_similarity` and the code against the reference with
    the simplified :func:`codebleu_score`. A task passes when its workflow
    similarity exceeds ``pass_threshold``; the pass-rate is the headline metric.

    GeoAnalystBench is 100% text, so every variant (including text-only Qwen)
    runs the full task set.

    Args:
        variant: The reasoner variant under test.
        tasks: The GeoAnalystBench tasks.
        backend: Injected backend; built lazily when ``None``.
        seed: Seed tag (logging / reproducibility).
        pass_threshold: Workflow-similarity threshold for the per-task pass.

    Returns:
        A mapping ``{pass_rate, mean_semantic_sim, mean_codebleu, n}``.
    """
    resolved_backend = _resolve_backend(variant, backend)
    sims: list[float] = []
    bleus: list[float] = []
    passes: list[float] = []

    for task in tasks:
        prompt = _build_geo_prompt(task)
        try:
            answer = await _run_backend_text(resolved_backend, prompt, [])
        except Exception as exc:  # noqa: BLE001 - one task must not crash the run
            logger.warning(
                "geoanalyst_task_failed",
                variant=variant.name,
                task_id=task.id,
                error=str(exc),
            )
            answer = ""
        workflow, code = _split_workflow_and_code(answer)
        sim = agent_metrics.workflow_semantic_similarity(workflow, task.human_workflow)
        bleu = agent_metrics.codebleu_score(code, task.code_string)
        sims.append(sim)
        bleus.append(bleu)
        passes.append(1.0 if sim > pass_threshold else 0.0)

    n = len(tasks)
    pass_rate = sum(passes) / n if n else 0.0
    mean_sim = sum(sims) / n if n else 0.0
    mean_bleu = sum(bleus) / n if n else 0.0
    logger.info(
        "geoanalyst_eval_done",
        variant=variant.name,
        seed=seed,
        pass_rate=pass_rate,
        n=n,
    )
    return {
        "pass_rate": pass_rate,
        "mean_semantic_sim": mean_sim,
        "mean_codebleu": mean_bleu,
        "n": n,
    }


def _aggregate(per_seed: Sequence[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Aggregate per-seed metric dicts into ``{metric: {mean, std}}``.

    NaN values (e.g. hallucination with no judge) are excluded from the mean/std
    so a missing metric stays NaN instead of poisoning the aggregate.

    Args:
        per_seed: One metric mapping per seed.

    Returns:
        ``{metric: {"mean": float, "std": float}}`` over the seeds.
    """
    metric_names: set[str] = set()
    for seed_metrics in per_seed:
        metric_names.update(seed_metrics.keys())

    aggregated: dict[str, dict[str, float]] = {}
    for metric in metric_names:
        values = [
            float(seed_metrics[metric])
            for seed_metrics in per_seed
            if metric in seed_metrics and not _is_nan(seed_metrics[metric])
        ]
        if not values:
            aggregated[metric] = {"mean": math.nan, "std": math.nan}
            continue
        mean = sum(values) / len(values)
        if len(values) > 1:
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(variance)
        else:
            std = 0.0
        aggregated[metric] = {"mean": mean, "std": std}
    return aggregated


def _is_nan(value: Any) -> bool:
    """Return whether ``value`` is a float NaN.

    Args:
        value: Any candidate value.

    Returns:
        ``True`` when ``value`` is a float NaN.
    """
    return isinstance(value, float) and math.isnan(value)


def run_benchmark(
    variants: Sequence[ReasonerVariant],
    *,
    seeds: Sequence[int] = (0, 1, 2),
    agromind_path: Path = DEFAULT_AGROMIND_PATH,
    geo_path: Path = DEFAULT_GEO_PATH,
    backends: dict[str, LLMBackend] | None = None,
    judge: HallucinationJudge | None = None,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    report_path: Path | None = None,
    log_mlflow: bool = True,
    probe_server: bool = True,
) -> dict[str, Any]:
    """Run both benchmarks for every variant over the seeds and report.

    Loads the real datasets once, then for each variant and seed evaluates
    AgroMind and GeoAnalystBench, aggregates ``mean +- std`` across seeds
    (error bars, AC-4), optionally logs every metric to MLflow with the
    ``code_version`` + ``data_version`` tags (AC-6, lineage on ``:5010``) and
    finally builds the HTML comparison report (AC-4/AC-5).

    Backends are injectable per variant via ``backends[variant.name]`` so the
    whole run is deterministic and offline in tests; when a variant has no
    injected backend one is built with :func:`make_backend` (real API / vLLM).

    Args:
        variants: The reasoner variants to evaluate.
        seeds: The evaluation seeds (3 by default, AC-4).
        agromind_path: Path to the AgroMind subset JSON.
        geo_path: Path to the GeoAnalystBench CSV.
        backends: Optional ``{variant_name: backend}`` injection map.
        judge: Optional hallucination judge (NaN when absent).
        image_root: Base folder for the AgroMind subset images.
        report_path: Output HTML path; defaults to
            ``reports/agent_bench/agent_bench.html``.
        log_mlflow: Whether to log the run to MLflow (AC-6).
        probe_server: Forwarded to ``track_experiment`` (set ``False`` in tests).

    Returns:
        The nested results mapping
        ``{variant: {benchmark: {metric: {"mean", "std"}}}}`` (also passed to
        :func:`build_report_html`).
    """
    backends = backends or {}
    items = load_agromind_subset(agromind_path)
    tasks = load_geoanalystbench(geo_path)

    results: dict[str, Any] = {}
    for variant in variants:
        backend = backends.get(variant.name)
        agromind_seeds: list[dict[str, float]] = []
        geo_seeds: list[dict[str, float]] = []
        for seed in seeds:
            agromind_seeds.append(
                asyncio.run(
                    eval_agromind(
                        variant,
                        items,
                        backend=backend,
                        judge=judge,
                        seed=seed,
                        image_root=image_root,
                    )
                )
            )
            geo_seeds.append(
                asyncio.run(
                    eval_geoanalyst(variant, tasks, backend=backend, seed=seed)
                )
            )
        results[variant.name] = {
            "AgroMind": _aggregate(agromind_seeds),
            "GeoAnalystBench": _aggregate(geo_seeds),
        }

    if log_mlflow:
        _log_to_mlflow(results, agromind_path, probe_server=probe_server)

    out_path = report_path or (DEFAULT_REPORT_DIR / "agent_bench.html")
    build_report_html(results, out_path)
    logger.info("agent_bench_done", variants=[v.name for v in variants], report=str(out_path))
    return results


def _log_to_mlflow(
    results: dict[str, Any], agromind_path: Path, *, probe_server: bool
) -> None:
    """Log the aggregated results to MLflow with versioning tags (AC-6).

    Opens one run via :func:`track_experiment` (which sets ``code_version`` and
    ``data_version``) and logs every ``{variant}/{benchmark}/{metric}`` mean and
    std as a metric. Logging failures are caught and logged: the benchmark and
    its report must still complete (eval-only, no training side effects).

    Args:
        results: The nested results mapping.
        agromind_path: DVC-tracked subset path used for the ``data_version`` tag.
        probe_server: Forwarded to ``track_experiment``.
    """
    try:
        import mlflow

        from ml.utils.mlflow_utils import track_experiment

        with track_experiment(
            _EXPERIMENT_NAME, dvc_path=str(agromind_path), probe_server=probe_server
        ):
            for variant, benchmarks in results.items():
                for benchmark, metrics in benchmarks.items():
                    for metric, stats in metrics.items():
                        mean = stats.get("mean", math.nan)
                        std = stats.get("std", math.nan)
                        if not _is_nan(mean):
                            mlflow.log_metric(f"{variant}/{benchmark}/{metric}/mean", mean)
                        if not _is_nan(std):
                            mlflow.log_metric(f"{variant}/{benchmark}/{metric}/std", std)
        logger.info("agent_bench_mlflow_logged", experiment=_EXPERIMENT_NAME)
    except Exception as exc:  # noqa: BLE001 - tracking must not break the eval run
        logger.warning("agent_bench_mlflow_failed", error=str(exc))


def _resolve_variants(names: Sequence[str] | None) -> list[ReasonerVariant]:
    """Resolve CLI variant tags to :class:`ReasonerVariant` objects.

    Args:
        names: The variant tags from the CLI, or ``None`` for all three.

    Returns:
        The resolved variants (defaults to all three, order preserved).
    """
    if not names:
        return list(DEFAULT_VARIANTS)
    resolved: list[ReasonerVariant] = []
    for name in names:
        variant = _VARIANTS_BY_NAME.get(name)
        if variant is None:
            valid = sorted(_VARIANTS_BY_NAME)
            raise SystemExit(f"Variante desconocida: {name!r}. Validas: {valid}")
        resolved.append(variant)
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the agent benchmark.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).

    Returns:
        Process exit code (``0`` on success).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evalua el copiloto AgroSat en AgroMind y GeoAnalystBench "
            "(eval-only, sin entrenamiento; US-049)."
        )
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(_VARIANTS_BY_NAME),
        default=None,
        help="Variantes a evaluar (por defecto las tres).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Seeds de evaluacion para las barras de error (por defecto 0 1 2).",
    )
    parser.add_argument(
        "--agromind",
        type=Path,
        default=DEFAULT_AGROMIND_PATH,
        help="Ruta al subset JSON de AgroMind.",
    )
    parser.add_argument(
        "--geo",
        type=Path,
        default=DEFAULT_GEO_PATH,
        help="Ruta al CSV de GeoAnalystBench.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
        help="Carpeta base de las imagenes del subset de AgroMind.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Ruta de salida del reporte HTML.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="No registrar la corrida en MLflow.",
    )
    args = parser.parse_args(argv)

    variants = _resolve_variants(args.variants)
    results = run_benchmark(
        variants,
        seeds=tuple(args.seeds),
        agromind_path=args.agromind,
        geo_path=args.geo,
        image_root=args.image_root,
        report_path=args.report,
        log_mlflow=not args.no_mlflow,
    )
    logger.info("agent_bench_cli_done", n_variants=len(variants), n_seeds=len(args.seeds))
    # Emit a compact JSON summary to stdout for the calling script / operator.
    _print_summary(results)
    return 0


def _print_summary(results: dict[str, Any]) -> None:
    """Write a compact JSON summary of the headline metrics to stdout.

    Args:
        results: The nested results mapping returned by :func:`run_benchmark`.
    """
    summary: dict[str, Any] = {}
    for variant, benchmarks in results.items():
        em = benchmarks.get("AgroMind", {}).get("exact_match", {})
        pr = benchmarks.get("GeoAnalystBench", {}).get("pass_rate", {})
        summary[variant] = {
            "AgroMind/exact_match": em.get("mean"),
            "GeoAnalystBench/pass_rate": pr.get("mean"),
        }
    sys.stdout.write(json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
