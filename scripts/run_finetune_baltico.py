"""Runner: fine-tune the dense backbone to the Baltic vocabulary on the H100.

Downloads real-texture Sentinel Hub patches for the source/target Baltic regions,
fine-tunes U-TAE (PASTIS backbone init + kept-class flag) on the per-parcel labels,
and reports the fine + hierarchical (collapsed-to-coarse) macro F1. Writes a JSON
summary so the verdict is auditable.

Run on the VM:
    F:\\tools\\micromamba.exe run -n agrosat python -m scripts.run_finetune_baltico
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import structlog

from backend.app.core.config import get_settings
from ml.ingest.sh_client import sh_client_from_settings
from ml.transfer.finetune_baltico import FineTuneConfig, run_finetune

logger = structlog.get_logger(__name__)

_OUT = Path("data/transfer/finetune_baltico/finetune_baltico_summary.json")
_UTAE_CKPT = "checkpoints/segmentation/utae-isaac/best_model.pt"


def main(argv: list[str] | None = None) -> int:
    """Run the Baltic fine-tune and dump the JSON verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="latvia")
    parser.add_argument("--target", default="estonia")
    parser.add_argument("--max-parcels", type=int, default=600)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--finetune-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-kind", default="utae")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=_OUT)
    args = parser.parse_args(argv)

    config = FineTuneConfig(
        model_kind=args.model_kind,
        head_warmup_epochs=args.warmup_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        max_parcels_per_region=args.max_parcels,
    )
    client = sh_client_from_settings(get_settings())
    summary = run_finetune(
        config,
        sh_client=client,
        pastis_checkpoint=_UTAE_CKPT,
        source=args.source,
        target=args.target,
        device=args.device,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("finetune_runner_done", out=str(args.out), **summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
