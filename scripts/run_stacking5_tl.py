"""Runner: Stacking-5 transfer learning to the Baltic vocabulary on the H100.

Downloads the stratified Baltic parcels (real-texture SH patches), fine-tunes the
dense members + xgb, stacks them with the champion meta-LogReg, and reports the
fine + hierarchical (collapsed-to-coarse) macro F1. Real data only.

Run on the VM:
    F:\\tools\\micromamba.exe run -n agrosat python -m scripts.run_stacking5_tl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import structlog

from backend.app.core.config import get_settings
from ml.ingest.sh_client import sh_client_from_settings
from ml.transfer.stacking5_tl_baltico import Stacking5TLConfig, run_stacking5_tl

logger = structlog.get_logger(__name__)

_OUT = Path("data/transfer/stacking5_tl_baltico")


def main(argv: list[str] | None = None) -> int:
    """Run the Stacking-5 Baltic transfer and dump the JSON verdict."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="latvia")
    parser.add_argument("--target", default="estonia")
    parser.add_argument("--per-class", type=int, default=120)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--finetune-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-farslip", action="store_true")
    parser.add_argument(
        "--sh-texture", action="store_true",
        help="Use Sentinel Hub textured patches (paid quota) instead of local npz.",
    )
    parser.add_argument(
        "--dense-members", nargs="+", default=["utae"],
        help="Dense members to fine-tune (utae works at 8px; tsvit needs 128px SH).",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    config = Stacking5TLConfig(
        source=args.source,
        target=args.target,
        per_class=args.per_class,
        dense_members=tuple(args.dense_members),
        warmup_epochs=args.warmup_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        include_farslip=not args.no_farslip,
        use_local_npz=not args.sh_texture,
        device=args.device,
    )
    client = sh_client_from_settings(get_settings())
    result = run_stacking5_tl(config, sh_client=client)
    out = _OUT / "stacking5_tl_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result.summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "stacking5_runner_done",
        out=str(out),
        **{k: v for k, v in result.summary.items() if not isinstance(v, (list, dict))},
    )
    print(json.dumps(
        {k: v for k, v in result.summary.items() if not isinstance(v, list)},
        indent=2, ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
