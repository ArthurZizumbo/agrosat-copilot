"""US-048: download the on-prem reasoner weights (Qwen MoE-A3B GPTQ-Int4).

The plan named ``Qwen3.5-35B-A3B-GPTQ-Int4``, which does NOT exist on
HuggingFace (verified jun-2026). The real closest MoE-A3B Int4 checkpoint is
``Qwen/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4`` (30B total / 3B active). The
substitution 35B -> 30B-A3B is documented in ADR-009 (US-050).

Weights go to ``F:`` on the VM (the large disk; ``C:`` is nearly full). Run on a
host with network access and the HuggingFace token in the environment.

Usage:
    poetry run python scripts/download_qwen35.py --dest /mnt/f/models
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

#: Real HuggingFace id (see module docstring for the 35B -> 30B-A3B substitution).
DEFAULT_MODEL_ID: str = "Qwen/Qwen3-30B-A3B-Instruct-2507-GPTQ-Int4"


def download(model_id: str, dest: Path, token: str | None) -> Path:
    """Download a model snapshot to ``dest`` via ``huggingface_hub``.

    Args:
        model_id: HuggingFace repo id of the model.
        dest: Destination directory (the snapshot is placed in a subdir).
        token: Optional HuggingFace token (public models do not require it).

    Returns:
        The local path of the downloaded snapshot.
    """
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    logger.info("qwen_download_started", model_id=model_id, dest=str(dest))
    local = snapshot_download(
        repo_id=model_id,
        local_dir=str(dest / model_id.split("/")[-1]),
        token=token,
    )
    logger.info("qwen_download_finished", model_id=model_id, local=local)
    return Path(local)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Download the Qwen MoE-A3B GPTQ-Int4 weights.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HuggingFace repo id.")
    parser.add_argument(
        "--dest",
        default="/mnt/f/models",
        help="Destination directory (keep on F: / the large disk).",
    )
    args = parser.parse_args(argv)

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    try:
        local = download(args.model_id, Path(args.dest), token)
    except Exception as exc:  # noqa: BLE001 - surface the real cause to the operator
        logger.error("qwen_download_failed", model_id=args.model_id, error=str(exc))
        return 1
    print(f"Downloaded {args.model_id} -> {local}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
