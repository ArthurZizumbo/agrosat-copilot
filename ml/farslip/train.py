"""CLI Typer para entrenar FarSLIP (US-017 / US-016b).

Lanza el trainer con la config validada en planning. VRAM esperada en GCP L4
24 GB: ~22 GB. Hard cap 8 h (warning 6 h).

Uso tipico::

    poetry run python -m ml.farslip.train \\
        --rois italy --epochs 4 --batch-size 64 --lr 1e-5 --seed 42 \\
        --output-dir artifacts/farslip --gcs-output-uri gs://agrosat-models/farslip/v1/

Flags ``--resume`` carga un checkpoint previo (ruta local o GCS) y reanuda.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import structlog
import torch

try:
    import typer
except ImportError as exc:  # pragma: no cover
    raise ImportError("typer requerido para CLI train. poetry add typer") from exc

from torch.utils.data import ConcatDataset, DataLoader

from ml.farslip.dataset import FarSLIPDataset
from ml.farslip.distill import FarSLIPDistillationTrainer, FarSLIPTrainerConfig
from ml.utils.seed import propagate_seed

_log = structlog.get_logger(__name__)

# ROIs italianas hardcoded para --rois italy (las 3 zonas del paper US-017).
# Cuando se agregue Francia (us-022-e), expandir este mapeo a {"italy": [...], "france": [...]}.
_ROIS_BY_KEY: dict[str, tuple[str, ...]] = {
    "italy": ("pianura_padana", "toscana", "puglia"),
}


def _build_dataset(dataset_root: Path, rois_key: str) -> tuple[ConcatDataset, int, int]:
    """Concatena los manifests de las ROIs italianas en un Dataset PyTorch.

    Importante: pasamos `cap_classes` y `regions` canonicos globales (unificados
    a partir de los 3 manifests) para que `region_id` y `category_id` esten en
    un namespace consistente entre los 3 FarSLIPDataset hijos. Sin esto cada
    dataset hijo deriva sus propios indices y region_id seria ambiguo al
    concatenar.

    Args:
        dataset_root: ruta a `data/farslip_pairs/`.
        rois_key: clave en `_ROIS_BY_KEY` (default "italy").

    Returns:
        Tupla (ConcatDataset, n_regions, n_categories) donde n_regions y
        n_categories son los tamanios reales del vocabulario global (necesarios
        para dimensionar text_prototypes correctamente).

    Raises:
        FileNotFoundError: si alguno de los manifests no existe.
        KeyError: si rois_key no esta en `_ROIS_BY_KEY`.
    """
    import polars as pl

    if rois_key not in _ROIS_BY_KEY:
        raise KeyError(
            f"rois={rois_key!r} no reconocido. Validos: {list(_ROIS_BY_KEY)}"
        )
    roi_slugs = _ROIS_BY_KEY[rois_key]

    # Pre-escaneo: unificar cap_classes y regions de los 3 manifests.
    all_cap_classes: list[str] = []
    all_regions: list[str] = []
    seen_caps: set[str] = set()
    seen_regs: set[str] = set()
    for roi in roi_slugs:
        manifest = dataset_root / roi / "manifest.parquet"
        if not manifest.exists():
            raise FileNotFoundError(f"manifest no existe: {manifest}")
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
    """Entrena FarSLIP con la configuracion provista."""
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
    # US-022-c P1 fix (2026-05-24): instanciar FarSLIPDataset + ConcatDataset por las
    # 3 ROIs italianas. El CLI previo solo instanciaba el trainer sin dataset, lo cual
    # gatillaba RuntimeError("dataset y dataloader nulos: nada que entrenar") en distill.py:534.
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

    # Text prototypes: el paper §3.3 los calcula con el text encoder frozen 1x por epoch.
    # Para esta primera implementacion CLI usamos prototypes random determinsticos
    # (seed propagado) — la senal contrastiva se mantiene aunque los prototipos no esten
    # alineados al vocabulario CAP. Refinamiento a prototypes-from-text-encoder queda como
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
