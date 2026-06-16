"""US-049: run the real agent benchmark over the three reasoner variants.

Wires the live endpoints (Gemini 3.5 Flash via the public API, Qwen on-prem via
the llama.cpp endpoint reached through the local SSH forward on :8002) and runs
``run_benchmark`` over AgroMind (500-pair subset, real images) and
GeoAnalystBench (50 tasks). Writes the HTML report and logs to MLflow.

Run with the H100 Qwen serving up and the :8002 forward active. Use ``--seeds 0``
to keep API cost bounded; pass more seeds for error bars when budget allows.

Usage:
    poetry run python scripts/run_us049_eval.py --seeds 0
    poetry run python scripts/run_us049_eval.py --variants gemini --no-mlflow
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: set live endpoints, then delegate to the bench CLI.

    Args:
        argv: Optional argument vector.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Run the real US-049 agent benchmark.")
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--qwen-url",
        default="http://127.0.0.1:8002/v1",
        help="OpenAI-compatible URL of the on-prem Qwen endpoint (local forward).",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11435/v1",
        help="OpenAI-compatible URL of the on-prem Gemma (Ollama) endpoint.",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/agent_bench/us049_report.html"),
    )
    args = parser.parse_args(argv)

    # Point the vLLM-compatible backend at the live Qwen endpoint and the Ollama
    # backend at the live Gemma endpoint for this run (both on-prem, zero cost).
    os.environ["VLLM_QWEN35_URL"] = args.qwen_url
    os.environ.setdefault("VLLM_API_KEY", "EMPTY")
    os.environ["OLLAMA_BASE_URL"] = args.ollama_url
    # Force UTF-8 so structlog never trips on cp1252 when logging accented prose.
    os.environ["PYTHONIOENCODING"] = "utf-8"

    from ml.eval.agent_bench import main as bench_main

    bench_argv: list[str] = ["--seeds", *[str(s) for s in args.seeds]]
    if args.variants:
        bench_argv += ["--variants", *args.variants]
    if args.no_mlflow:
        bench_argv.append("--no-mlflow")
    bench_argv += ["--report", str(args.report)]
    return bench_main(bench_argv)


if __name__ == "__main__":
    sys.exit(main())
