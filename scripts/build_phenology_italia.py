"""Runner: build the Italian per-class phenology prototypes (US-079 fix B).

Generates the Mediterranean homologue of
``data/features/phenology_class_prototypes_pastis.parquet``: a per-class mean NDVI
curve computed from the REAL US-078 Italy 2018 patches, a Gemini 3.5 Flash
phenological description grounded in the Italian calendar (durum harvested in
June, perennial olive/vineyard, displaced season), and its 384-dim
``all-MiniLM-L6-v2`` embedding. This replaces the Bretagne/PASTIS prototypes the
TSViT-pheno semantic branch was contrasting against (US-079 root cause B: the
French prototypes aligned Italian pixels with the wrong phenology).

The flow is the SAME molde as :mod:`ml.features.phenology_class_prototypes`
(curve -> description -> embedding), reusing the SHA256 description cache, so a
re-run is deterministic and free for unchanged curves. Nothing is fabricated:
classes absent from the scanned patches are reported and skipped, and if Gemini
has no credentials the run fails loudly (the prototypes are not invented).

This is CPU/network only (no GPU). Examples::

    # Full run over the 1438 patches (needs GEMINI_API_KEY in .env.local):
    poetry run python -m scripts.build_phenology_italia \
        --dataset-root data/pastis_italia_2018 \
        --out data/features/phenology_class_prototypes_italia.parquet

    # Pilot smoke over the first 20 patches:
    poetry run python -m scripts.build_phenology_italia --max-patches 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import structlog

from ml.features.phenology_class_prototypes import (
    _DEFAULT_ITALIA_OUTPUT,
    _DEFAULT_ITALIA_ROOT,
    _N_TIME_BINS,
    generate_italia_class_prototypes,
)
from ml.features.phenology_description import _has_credentials

logger = structlog.get_logger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the Italian prototype runner."""
    p = argparse.ArgumentParser(
        description="Genera los prototipos fenologicos por clase italianos (US-079)."
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=_DEFAULT_ITALIA_ROOT,
        help="US-078 homologue dataset root (DATA_S2/ + ANNOTATIONS/ + class_mapping.json).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_ITALIA_OUTPUT,
        help="Output prototypes parquet (one row per present Italian class).",
    )
    p.add_argument(
        "--model",
        default="gemini-3.5-flash",
        help="LLM for the phenology descriptions (needs Gemini credentials).",
    )
    p.add_argument(
        "--n-time-bins",
        type=int,
        default=_N_TIME_BINS,
        help="DOY bins of the mean NDVI curve (matched to the PASTIS grid).",
    )
    p.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="Limit the NDVI scan to the first N patches (smoke on the pilot).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Parse the CLI, load credentials and generate the Italian prototypes.

    Args:
        argv: Optional explicit argument vector (for tests); ``None`` uses
            ``sys.argv``.

    Returns:
        Process exit code (0 on success).
    """
    # Load .env.local so GEMINI_API_KEY / GOOGLE_GENAI_USE_VERTEXAI are present for
    # the default (cloud) client. No-op when the file is absent.
    from ml.utils.notebook_setup import find_repo_root, load_env_local

    load_env_local(find_repo_root())

    args = _build_arg_parser().parse_args(argv)

    if not _has_credentials():
        # Honest barrier (Arthur's absolute rule): no fabricated descriptions. The
        # SHA256 cache still lets a re-run reuse already-generated descriptions, so
        # this only blocks the FIRST generation of an uncached curve.
        logger.error(
            "italia_prototypes_no_gemini_credentials",
            note="set GEMINI_API_KEY (or GOOGLE_GENAI_USE_VERTEXAI=true + "
            "GOOGLE_CLOUD_PROJECT) in .env.local. Descriptions are NOT fabricated; "
            "only fully-cached curves would succeed offline.",
        )
        # Continue anyway: generate_italia_class_prototypes will succeed iff every
        # curve is already cached, and raise loudly otherwise (no silent stub).
        logger.warning("italia_prototypes_proceeding_cache_only")

    out = generate_italia_class_prototypes(
        args.dataset_root,
        output_path=args.out,
        model=args.model,
        n_time_bins=args.n_time_bins,
        max_patches=args.max_patches,
    )
    logger.info("italia_prototypes_done", output=str(out))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
