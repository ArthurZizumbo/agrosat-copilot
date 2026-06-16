"""US-048: smoke test + latency benchmark of the on-prem Qwen vLLM endpoint.

Sends a canonical agronomic query (smoke) and then a small batch of
single-turn queries to measure p50/p95 latency, logging the results to MLflow
(server :5010, native on the VM). Targets from the plan: p50 < 2s / p95 < 5s on
a simple single-turn query.

This is the same OpenAI-compatible surface the agent's ``VLLMOpenAIBackend``
uses, so a green smoke here means the agent can switch to Qwen via ``/llm/switch``.

RUNTIME NOTE: requires a live vLLM endpoint. On the H100 VM that endpoint cannot
start yet (nested-virt blocker, see docs/serving/qwen35.md); run this once the
serving is up (WSL2 unblocked or a Linux GPU host).

Usage:
    poetry run python scripts/benchmark_qwen35.py \
        --base-url http://127.0.0.1:8002/v1 --model qwen35 --n 10
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import structlog

logger = structlog.get_logger(__name__)

#: Canonical single-turn smoke query (Spanish: the user-facing language).
SMOKE_QUERY: str = (
    "Eres un analista agronomico. En una frase, explica que indica un pico de "
    "NDVI alto y tardio en la curva fenologica de una parcela de maiz."
)

#: A few varied single-turn queries to estimate latency percentiles.
LATENCY_QUERIES: tuple[str, ...] = (
    "Que cultivo suele tener un pico de NDVI temprano en primavera?",
    "Explica brevemente que es el NDWI y para que sirve en agricultura.",
    "Que diferencia hay entre senescencia temprana y tardia en trigo?",
    "Resume en una linea el valor de un embedding AlphaEarth para clasificar cultivos.",
    "Por que conviene usar validacion cruzada espacial y no aleatoria?",
)


def _client(base_url: str, api_key: str):
    """Build the OpenAI-compatible client pointed at the vLLM endpoint.

    Args:
        base_url: Base URL (``.../v1``).
        api_key: API key (vLLM ignores it but the client needs a value).

    Returns:
        An ``openai.OpenAI`` client.
    """
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key or "EMPTY")


def _ask(client, model: str, query: str) -> tuple[str, float]:
    """Send one chat completion and time it.

    Args:
        client: The OpenAI-compatible client.
        model: Served model name.
        query: User query.

    Returns:
        ``(answer_text, elapsed_seconds)``.
    """
    start = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}],
        max_tokens=256,
        temperature=0.0,
    )
    elapsed = time.perf_counter() - start
    return resp.choices[0].message.content or "", elapsed


def run_benchmark(base_url: str, model: str, api_key: str, n: int) -> dict[str, float]:
    """Run the smoke query and the latency batch.

    Args:
        base_url: vLLM base URL.
        model: Served model name.
        api_key: API key.
        n: Number of latency samples (queries cycle through ``LATENCY_QUERIES``).

    Returns:
        A metrics mapping (p50/p95/mean latency, n).
    """
    client = _client(base_url, api_key)

    smoke_text, smoke_latency = _ask(client, model, SMOKE_QUERY)
    logger.info("qwen_smoke_ok", latency_s=round(smoke_latency, 3), answer=smoke_text[:160])

    latencies: list[float] = []
    for i in range(n):
        query = LATENCY_QUERIES[i % len(LATENCY_QUERIES)]
        _, elapsed = _ask(client, model, query)
        latencies.append(elapsed)
        logger.info("qwen_latency_sample", i=i, latency_s=round(elapsed, 3))

    latencies.sort()
    metrics = {
        "smoke_latency_s": smoke_latency,
        "latency_p50_s": statistics.median(latencies),
        "latency_p95_s": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))],
        "latency_mean_s": statistics.fmean(latencies),
        "n_samples": float(n),
    }
    return metrics


def _log_to_mlflow(metrics: dict[str, float], model_id: str, tracking_uri: str) -> None:
    """Log the benchmark metrics to MLflow with data/code version tags.

    Args:
        metrics: The benchmark metrics.
        model_id: The served model id (logged as a param).
        tracking_uri: MLflow tracking URI (the VM server :5010).
    """
    import subprocess

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("us048_qwen_serving")
    try:
        code_version = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip()
            or "unknown"
        )
    except OSError:
        code_version = "unknown"
    with mlflow.start_run(run_name="qwen35_latency"):
        mlflow.set_tags({"code_version": code_version, "data_version": "n/a-serving"})
        mlflow.log_param("model_id", model_id)
        mlflow.log_metrics(metrics)
    logger.info("qwen_benchmark_logged_mlflow", tracking_uri=tracking_uri)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument vector.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Smoke + latency benchmark of the Qwen vLLM endpoint."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--model", default="qwen35")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--n", type=int, default=10, help="Number of latency samples.")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4",
        help="HF id of the served model (for MLflow logging).",
    )
    parser.add_argument("--mlflow-uri", default="http://127.0.0.1:5010")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging.")
    args = parser.parse_args(argv)
    if args.n < 1:
        parser.error("--n debe ser >= 1 (se necesita al menos una muestra de latencia)")

    try:
        metrics = run_benchmark(args.base_url, args.model, args.api_key, args.n)
    except Exception as exc:  # noqa: BLE001 - surface the real cause to the operator
        logger.error("qwen_benchmark_failed", error=str(exc))
        return 1

    print("Qwen serving benchmark:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")

    if not args.no_mlflow:
        try:
            _log_to_mlflow(metrics, args.model_id, args.mlflow_uri)
        except Exception as exc:  # noqa: BLE001
            logger.warning("qwen_benchmark_mlflow_skipped", error=str(exc))

    return 0


if __name__ == "__main__":
    sys.exit(main())
