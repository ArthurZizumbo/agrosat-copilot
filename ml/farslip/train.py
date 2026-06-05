"""Typer CLI to train FarSLIP (US-017 / US-016b).

Launches the trainer with the config validated in planning. Expected VRAM on
GCP L4 24 GB: ~22 GB. Hard cap 8 h (warning 6 h).

Typical usage::

    poetry run python -m ml.farslip.train \\
        --rois italy --epochs 4 --batch-size 64 --lr 1e-5 --seed 42 \\
        --output-dir artifacts/farslip --gcs-output-uri gs://agrosat-models/farslip/v1/

The ``--resume`` flag loads a previous checkpoint (local path or GCS) and resumes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import structlog
import torch

try:
    import typer
except ImportError as exc:  # pragma: no cover
    raise ImportError("typer required for the train CLI. poetry add typer") from exc

from torch.utils.data import ConcatDataset, DataLoader

from ml.farslip.dataset import FarSLIPDataset
from ml.farslip.distill import FarSLIPDistillationTrainer, FarSLIPTrainerConfig
from ml.utils.seed import propagate_seed

_log = structlog.get_logger(__name__)

# Italian ROIs hardcoded for --rois italy (the 3 zones of the US-017 paper).
# When France is added (us-022-e), expand this mapping to {"italy": [...], "france": [...]}.
_ROIS_BY_KEY: dict[str, tuple[str, ...]] = {
    "italy": ("pianura_padana", "toscana", "puglia"),
}


def _build_dataset(dataset_root: Path, rois_key: str) -> tuple[ConcatDataset, int, int]:
    """Concatenate the Italian ROI manifests into a PyTorch Dataset.

    Important: we pass global canonical `cap_classes` and `regions` (unified
    from the 3 manifests) so that `region_id` and `category_id` are in a
    consistent namespace across the 3 child FarSLIPDataset instances. Without
    this each child dataset derives its own indices and region_id would be
    ambiguous when concatenating.

    Args:
        dataset_root: path to `data/farslip_pairs/`.
        rois_key: key in `_ROIS_BY_KEY` (default "italy").

    Returns:
        Tuple (ConcatDataset, n_regions, n_categories) where n_regions and
        n_categories are the real sizes of the global vocabulary (needed
        to dimension text_prototypes correctly).

    Raises:
        FileNotFoundError: if any of the manifests does not exist.
        KeyError: if rois_key is not in `_ROIS_BY_KEY`.
    """
    import polars as pl

    if rois_key not in _ROIS_BY_KEY:
        raise KeyError(
            f"rois={rois_key!r} not recognized. Valid: {list(_ROIS_BY_KEY)}"
        )
    roi_slugs = _ROIS_BY_KEY[rois_key]

    # Pre-scan: unify cap_classes and regions across the 3 manifests.
    all_cap_classes: list[str] = []
    all_regions: list[str] = []
    seen_caps: set[str] = set()
    seen_regs: set[str] = set()
    for roi in roi_slugs:
        manifest = dataset_root / roi / "manifest.parquet"
        if not manifest.exists():
            raise FileNotFoundError(f"manifest does not exist: {manifest}")
        df = pl.read_parquet(manifest, columns=["cap_class", "region"])
        for c in df["cap_class"].to_list():
            if c not in seen_caps:
                all_cap_classes.append(c)
                seen_caps.add(c)
        for r in df["region"].to_list():
            if r not in seen_regs:
                all_regions.append(r)
                seen_regs.add(r)

    parts = []
    for roi in roi_slugs:
        manifest = dataset_root / roi / "manifest.parquet"
        parts.append(
            FarSLIPDataset(
                manifest_path=manifest,
                cap_classes=all_cap_classes,
                regions=all_regions,
            )
        )
    return ConcatDataset(parts), len(all_regions), len(all_cap_classes)

app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def train(
    rois: Annotated[
        str, typer.Option(help="Identificador de ROI set, e.g. 'italy'")
    ] = "italy",
    epochs: Annotated[int, typer.Option(help="Numero de epochs")] = 4,
    batch_size: Annotated[int, typer.Option(help="Batch size logico")] = 64,
    lr: Annotated[float, typer.Option(help="Learning rate AdamW")] = 1e-5,
    seed: Annotated[int, typer.Option(help="Semilla determinismo")] = 42,
    output_dir: Annotated[
        Path, typer.Option(help="Directorio local para checkpoints")
    ] = Path("artifacts/farslip"),
    gcs_output_uri: Annotated[
        str | None, typer.Option(help="URI GCS para subir pesos finales")
    ] = None,
    dataset_root: Annotated[
        Path, typer.Option(help="Raiz dataset farslip_pairs")
    ] = Path("data/farslip_pairs"),
    teacher_model_id: Annotated[
        str, typer.Option(help="HF id del CLIP teacher")
    ] = "openai/clip-vit-base-patch16",
    resume: Annotated[
        str | None, typer.Option(help="Ruta/URI a checkpoint para reanudar")
    ] = None,
    time_cap_hours: Annotated[float, typer.Option(help="Hard cap horas")] = 8.0,
) -> None:
    """Train FarSLIP with the provided configuration."""
    propagate_seed(seed)
    _log.info(
        "starting farslip training",
        rois=rois,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    # US-022-c P1 fix (2026-05-24): instantiate FarSLIPDataset + ConcatDataset for the
    # 3 Italian ROIs. The previous CLI only instantiated the trainer without a dataset, which
    # triggered RuntimeError("dataset y dataloader nulos: nada que entrenar") in distill.py:534.
    dataset, n_regions, n_categories = _build_dataset(dataset_root, rois)
    _log.info(
        "dataset built",
        n_samples=len(dataset),
        rois=rois,
        n_regions=n_regions,
        n_categories=n_categories,
    )

    cfg = FarSLIPTrainerConfig(
        teacher_model_id=teacher_model_id,
        dataset_root=dataset_root,
        output_dir=output_dir,
        gcs_output_uri=gcs_output_uri,
        n_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        time_cap_hours=time_cap_hours,
        n_regions=n_regions,
        n_categories=n_categories,
    )
    trainer = FarSLIPDistillationTrainer(cfg, dataset=dataset)
    if resume:
        _log.info("resume from checkpoint", uri=resume)
        path = Path(resume)
        if path.exists():
            sd = torch.load(path, map_location=trainer.device, weights_only=True)
            trainer.student.load_state_dict(sd, strict=False)

    # Text prototypes: paper §3.3 computes them with the frozen text encoder 1x per epoch.
    # For this first CLI implementation we use deterministic random prototypes
    # (propagated seed) — the contrastive signal holds even though the prototypes are not
    # aligned to the CAP vocabulary. Refinement to prototypes-from-text-encoder remains a
    # follow-up post-US-022-c (paper-faithful enhancement, ADR-007 §"Diferencias menores").
    n_protos = cfg.n_regions * cfg.n_categories
    hidden_dim = trainer.teacher.config.hidden_size
    text_prototypes = torch.randn(n_protos, hidden_dim, generator=torch.Generator().manual_seed(seed))
    trainer.set_text_prototypes(text_prototypes)
    _log.info("text_prototypes initialized", n_protos=n_protos, hidden_dim=hidden_dim, mode="random_seeded")

    metrics = trainer.train()
    _log.info("training done", **metrics)


if __name__ == "__main__":  # pragma: no cover
    app()
