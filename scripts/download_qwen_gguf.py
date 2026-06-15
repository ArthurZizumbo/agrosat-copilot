"""US-048 (llama.cpp variant): download the Qwen GGUF weights for on-prem serving.

vLLM cannot run on the Windows H100 VM (Hyper-V guest without nested virt -> no
WSL2/Docker; see docs/serving/qwen35.md). The native Windows alternative is
llama.cpp, which serves an OpenAI-compatible endpoint and runs the same model in
GGUF format. This script downloads a single-file GGUF quant of
``Qwen3-30B-A3B-Instruct-2507`` (a Qwen3 MoE A3B that llama.cpp supports natively).

Default quant: Q4_K_M (~18.6 GB) -- the quality/size sweet spot, the GGUF
equivalent of the GPTQ-Int4 we planned for vLLM. Fits easily on the H100.

Weights go to ``F:`` on the VM (the large disk). Run on a host with network and,
if needed, the HuggingFace token in the environment.

Usage:
    poetry run python scripts/download_qwen_gguf.py --dest F:/models
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

#: Default GGUF repo (unsloth mirror, most-downloaded; bartowski is an alternative).
DEFAULT_REPO_ID: str = "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF"

#: Default single-file quant (Q4_K_M ~18.6 GB). Other valid choices in the repo:
#: Q5_K_M (~21.7 GB, higher quality), Q4_K_S (~17.5 GB), Q6_K (~25 GB).
DEFAULT_GGUF_FILE: str = "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"


def download(repo_id: str, filename: str, dest: Path, token: str | None) -> Path:
    """Download a single GGUF file from a HuggingFace repo.

    Args:
        repo_id: HuggingFace repo id holding the GGUF files.
        filename: Exact GGUF filename to fetch.
        dest: Destination directory.
        token: Optional HuggingFace token (public repos do not require it).

    Returns:
        The local path of the downloaded GGUF.
    """
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    logger.info("qwen_gguf_download_started", repo_id=repo_id, filename=filename, dest=str(dest))
    local = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dest),
        token=token,
    )
    logger.info("qwen_gguf_download_finished", local=local)
    return Path(local)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument vector.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Download the Qwen3-30B-A3B GGUF weights for llama.cpp serving."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HuggingFace GGUF repo id.")
    parser.add_argument("--file", default=DEFAULT_GGUF_FILE, help="GGUF filename to download.")
    parser.add_argument(
        "--dest",
        default="F:/models",
        help="Destination directory (keep on F: / the large disk on the VM).",
    )
    args = parser.parse_args(argv)

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    try:
        local = download(args.repo_id, args.file, Path(args.dest), token)
    except Exception as exc:  # noqa: BLE001 - surface the real cause to the operator
        logger.error("qwen_gguf_download_failed", repo_id=args.repo_id, error=str(exc))
        return 1
    print(f"Downloaded {args.repo_id}/{args.file} -> {local}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
