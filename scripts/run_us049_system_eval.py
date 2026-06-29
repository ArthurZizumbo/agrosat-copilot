"""US-049: run the PROJECT-GROUNDED agent eval live over the four variants.

This is the eval that measures OUR system on OUR tools/data (tool-calling
correctness, grounded-crop orchestration, RAG A/B hallucination), as opposed to
the external public benchmarks (AgroMind/GeoAnalystBench) which probe base-VLM
perception. It wires the live reasoner backends (Gemini cloud + the three on-prem
endpoints) and the real tool registry, and writes a JSON summary.

It complements ``scripts/run_us049_eval.py`` (the public benchmark). Run with the
H100 endpoints up and the local forwards active (:8002 Qwen text, :8003 Qwen3.6
VL, :11435 Gemma); Gemini reads its key from ``.env.local`` via ``Settings``.

Grounded-crop stubs the classifier seams (``_fetch_parcel_embedding`` +
``_load_classifier``) with a standalone ``pytest.MonkeyPatch`` so the REAL
``classify.run`` plumbing runs while the ensemble output is the injected,
deterministic per-case result -- the eval scores AGENT ORCHESTRATION + faithful
reporting, never the classifier's own accuracy.

Usage:
    poetry run python scripts/run_us049_system_eval.py --seeds 0
    poetry run python scripts/run_us049_system_eval.py --variants gemini qwen36-vl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

# Force UTF-8 on the standard streams BEFORE any logging is configured. On a
# Windows cp1252 console structlog otherwise raises ``UnicodeEncodeError`` while
# emitting accented Spanish answer prose, which the per-case ``except`` records as
# a failed (charmap) case and silently DEPRESSES crop_match -- a measurement
# artifact, not a model error. ``reconfigure`` works even after interpreter start,
# so it fixes runs launched without ``PYTHONUTF8=1``.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]


def _build_live_backends(variant_names: Sequence[str]) -> dict[str, object]:
    """Build one live :class:`LLMBackend` per variant name via ``make_backend``.

    Args:
        variant_names: Variant tags to build backends for.

    Returns:
        A ``{variant_name: backend}`` mapping reused for all three evals.
    """
    from ml.agent.backends import make_backend
    from ml.eval.agent_bench import _VARIANTS_BY_NAME

    try:
        from backend.app.core.config import get_settings

        settings = get_settings()
    except Exception:  # noqa: BLE001 - settings optional outside the app
        settings = None

    backends: dict[str, object] = {}
    for name in variant_names:
        variant = _VARIANTS_BY_NAME[name]
        backends[name] = make_backend(variant.model, settings)
    return backends


def _build_judge() -> object | None:
    """Build a live hallucination judge backed by Gemini, or ``None`` on failure.

    Returns:
        A judge object exposing ``score(sample) -> float`` in ``[0, 1]``, or
        ``None`` when no judge can be built (then RAG hallucination is NaN).
    """
    try:
        from ml.eval.agent_metrics import build_gemini_judge

        return build_gemini_judge()
    except Exception:  # noqa: BLE001 - judge optional; NaN is the honest fallback
        return None


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the project-grounded agent eval.

    Args:
        argv: Optional argument vector.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Run the live US-049 system eval.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["gemini", "qwen", "gemma-base", "qwen36-vl"],
        help="Variantes a evaluar (por defecto las cuatro).",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--qwen-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11435/v1")
    parser.add_argument("--qwen-vl-url", default="http://127.0.0.1:8003/v1")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/agent_bench/us049_system_eval.json"),
        help="Ruta del JSON de resultados.",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="No usar juez de alucinacion (RAG hallucination queda NaN).",
    )
    args = parser.parse_args(argv)

    os.environ["VLLM_QWEN35_URL"] = args.qwen_url
    os.environ.setdefault("VLLM_API_KEY", "EMPTY")
    os.environ["OLLAMA_BASE_URL"] = args.ollama_url
    os.environ["QWEN36_VL_URL"] = args.qwen_vl_url
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUNBUFFERED"] = "1"

    import structlog
    from pytest import MonkeyPatch

    from ml.agent.context import ToolContext
    from ml.eval.agent_bench import _VARIANTS_BY_NAME
    from ml.eval.agent_system_eval import run_system_eval

    logger = structlog.get_logger("run_us049_system_eval")

    variants = [_VARIANTS_BY_NAME[name] for name in args.variants]
    backends = _build_live_backends(args.variants)

    try:
        from backend.app.core.config import get_settings

        settings = get_settings()
    except Exception:  # noqa: BLE001 - settings optional outside the app
        settings = None

    def make_ctx(session_id=None, defer=None):
        """Build a live ToolContext (DB pool unused for the stubbed crop eval)."""
        from uuid import uuid4

        return ToolContext(
            pool=None,  # type: ignore[arg-type]
            settings=settings,  # type: ignore[arg-type]
            session_id=session_id or uuid4(),
            defer=defer,
        )

    judge = None if args.no_judge else _build_judge()
    monkeypatch = MonkeyPatch()
    try:
        results = run_system_eval(
            variants,
            seeds=tuple(args.seeds),
            toolcall_backends=backends,
            crop_backends=backends,
            rag_backends=backends,
            make_ctx=make_ctx,
            monkeypatch_target=monkeypatch,
            judge=judge,
        )
    finally:
        monkeypatch.undo()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("system_eval_written", path=str(args.out), variants=args.variants)
    # Compact summary to stdout.
    print(json.dumps(results, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
