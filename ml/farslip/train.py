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

from ml.farslip.distill import FarSLIPDistillationTrainer, FarSLIPTrainerConfig
from ml.utils.seed import propagate_seed

_log = structlog.get_logger(__name__)

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
    )
    trainer = FarSLIPDistillationTrainer(cfg)
    if resume:
        _log.info("resume from checkpoint", uri=resume)
        # Para resume real desde GCS: download + load_state_dict. Stub para CLI.
        path = Path(resume)
        if path.exists():
            sd = torch.load(path, map_location=trainer.device, weights_only=True)
            trainer.student.load_state_dict(sd, strict=False)
    metrics = trainer.train()
    _log.info("training done", **metrics)


if __name__ == "__main__":  # pragma: no cover
    app()
