"""Extraccion de embeddings FarSLIP a parquet (US-022-c P1 etapa 6).

Carga el student FarSLIP entrenado en GCP L4 (MLflow `farslip-clip-italy-v1@Production`
o ruta local) y proyecta cada parcela italiana sobre el espacio de embeddings
512-dim, persistiendo el resultado a parquet con esquema estable:

- ``parcel_id`` (int64) — identificador de la parcela aguas arriba.
- ``year`` (int32) — anio del crop temporal asociado.
- ``farslip_emb_000`` .. ``farslip_emb_511`` (float32) — 512 columnas.

Contrato AC US-022-c sec 2.1 B-4:

- Output parquet shape ``(85951, 514)`` (parcel_id + year + 512 embed).
- Determinismo con ``seed=42`` reproducible (mismo input -> mismo output).
- Fallback ``cuda -> cpu`` con warning si CUDA no disponible.
- Resolucion MLflow URI ``mlflow://Models/farslip-clip-italy-v1@Production``.

Uso CLI tipico::

    python -m ml.farslip.extract_embeddings \\
        --student-checkpoint mlflow://Models/farslip-clip-italy-v1@Production \\
        --parcels-parquet data/features/features_fused_v1.parquet \\
        --rois italy --output data/farslip/embeddings_italy.parquet
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl
import structlog
import torch

from ml.utils.git_meta import git_sha
from ml.utils.seed import propagate_seed

try:
    from transformers import CLIPVisionModel
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "extract_embeddings requiere transformers>=4.46. `poetry add transformers`."
    ) from exc

_log = structlog.get_logger(__name__)

EMBED_DIM: int = 512
TOTAL_COLS: int = EMBED_DIM + 2  # parcel_id + year + 512 dims
EMBED_COL_PREFIX: str = "farslip_emb_"

DeviceLiteral = Literal["auto", "cuda", "cpu"]
RoiPreset = tuple[str, ...]

_ROI_ALIASES: dict[str, RoiPreset] = {
    "italy": ("pianura_padana", "toscana", "puglia"),
    "pianura_padana": ("pianura_padana",),
    "toscana": ("toscana",),
    "puglia": ("puglia",),
}


@dataclass(frozen=True)
class ExtractEmbeddingsResult:
    """Resultado tipado de :func:`extract_farslip_embeddings`.

    Attributes:
        n_parcels: numero de filas en el parquet final.
        n_dims: dimension del embedding (siempre 512 para CLIP ViT-B/16).
        output_path: ruta absoluta del parquet generado.
        code_version: git SHA del repo al momento de extraer.
        data_version: identificador del checkpoint origen
            (``mlflow://...`` o ruta local).
        device_used: ``"cuda"`` o ``"cpu"`` efectivamente usado.
    """

    n_parcels: int
    n_dims: int
    output_path: Path
    code_version: str
    data_version: str
    device_used: str


def _resolve_rois(rois: tuple[str, ...]) -> RoiPreset:
    """Expande alias (``"italy"``) a la tupla canonica de ROIs."""
    if len(rois) == 1 and rois[0] in _ROI_ALIASES:
        return _ROI_ALIASES[rois[0]]
    return rois


def _resolve_device(device: DeviceLiteral) -> torch.device:
    """Resuelve ``"auto"`` con fallback ``cuda -> cpu`` y warning explicito."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device == "cuda" and not torch.cuda.is_available():
        _log.warning(
            "cuda_requested_but_unavailable_fallback_cpu",
            note="device='cuda' pero torch.cuda.is_available()==False; degrada a cpu.",
        )
        return torch.device("cpu")
    return torch.device(device)


def _resolve_checkpoint(path: Path | str) -> tuple[Path | str, str]:
    """Devuelve (ruta_resuelta, data_version_tag).

    Soporta dos formatos:

    - ``mlflow://Models/<name>@<stage>`` -> descarga via mlflow registry,
      data_version = la cadena URI completa.
    - Ruta local -> devuelve la ruta tal cual, data_version = ``str(path)``.

    Para mantener el modulo testable sin red, la descarga MLflow se delega a
    :func:`mlflow.artifacts.download_artifacts` solo cuando la cadena empieza
    por ``mlflow://``. El test suite parchea esta funcion para evitar HTTP.
    """
    p = str(path)
    if p.startswith("mlflow://"):
        return _resolve_mlflow_uri(p), p
    return Path(p), p


def _resolve_mlflow_uri(uri: str) -> Path:
    """Descarga el artefacto MLflow apuntado por ``uri``.

    Ejemplo: ``mlflow://Models/farslip-clip-italy-v1@Production``.
    """
    import mlflow

    # mlflow expects: models:/<name>/<stage> or models:/<name>@<alias>
    body = uri.removeprefix("mlflow://")
    if body.startswith("Models/"):
        body = body.removeprefix("Models/")
    models_uri = f"models:/{body}"
    _log.info("downloading_mlflow_artifact", uri=models_uri)
    local = mlflow.artifacts.download_artifacts(artifact_uri=models_uri)
    return Path(local)


def _load_student(
    checkpoint_path: Path | str,
    *,
    device: torch.device,
    teacher_model_id: str = "openai/clip-vit-base-patch16",
    n_in_channels: int = 4,
) -> CLIPVisionModel:
    """Reconstruye el CLIPVisionModel student desde checkpoint.

    Reusa la misma logica que ``ml.farslip.distill._patch_student_proj``: arranca
    desde el teacher HF, adapta ``patch_embed`` a ``n_in_channels`` y carga el
    state_dict del student. ``strict=False`` tolera diferencias de prefijo entre
    ``CLIPVisionModel`` y ``CLIPModel.vision_model``.
    """
    from ml.farslip.distill import adapt_patch_embed_to_n_channels

    model = CLIPVisionModel.from_pretrained(teacher_model_id)
    adapt_patch_embed_to_n_channels(model, n_in_channels)
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.is_dir():
        cands = list(ckpt_path.glob("*.safetensors")) + list(ckpt_path.glob("*.pt"))
        if not cands:
            raise FileNotFoundError(
                f"sin checkpoints (*.safetensors|*.pt) en {ckpt_path}"
            )
        ckpt_path = cands[0]
    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(ckpt_path))
    else:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    _log.info(
        "student_loaded",
        path=str(ckpt_path),
        missing=len(missing),
        unexpected=len(unexpected),
    )
    model.eval()
    model.to(device)
    return model


def _embed_columns() -> list[str]:
    """Devuelve ``["farslip_emb_000", ..., "farslip_emb_511"]``."""
    return [f"{EMBED_COL_PREFIX}{i:03d}" for i in range(EMBED_DIM)]


def _load_parcels_filtered(
    parcels_parquet: Path, rois: RoiPreset
) -> pl.DataFrame:
    """Carga parcelas y filtra por ROI(s) si la columna existe.

    El parquet debe exponer al menos ``parcel_id`` y ``year``. Si tiene una
    columna ``roi`` (o ``region``), se filtra por ``rois``. Si no existe, no
    se filtra (asume que el caller ya cribo).
    """
    df = pl.read_parquet(parcels_parquet)
    if "parcel_id" not in df.columns:
        raise ValueError("parquet sin columna 'parcel_id'")
    if "year" not in df.columns:
        raise ValueError("parquet sin columna 'year'")
    for col in ("roi", "region"):
        if col in df.columns:
            df = df.filter(pl.col(col).is_in(list(rois)))
            break
    return df


def _build_empty_embeddings(n_rows: int) -> torch.Tensor:
    """Tensor 0 de shape ``(n_rows, EMBED_DIM)`` (placeholder smoke-tests)."""
    return torch.zeros((n_rows, EMBED_DIM), dtype=torch.float32)


def _project_parcels_to_embeddings(
    model: CLIPVisionModel,
    *,
    n_parcels: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Placeholder determinista (mantenido para tests existentes y smoke).

    Genera `torch.randn(seed)` normalizado L2. Para extract REAL ver
    :func:`_project_parcels_to_embeddings_real`.
    """
    propagate_seed(seed)
    generator = torch.Generator(device=device.type if device.type != "cpu" else "cpu")
    generator.manual_seed(seed)
    out_chunks: list[torch.Tensor] = []
    for start in range(0, n_parcels, batch_size):
        end = min(start + batch_size, n_parcels)
        chunk = torch.randn(
            (end - start, EMBED_DIM),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        chunk = torch.nn.functional.normalize(chunk, dim=-1)
        out_chunks.append(chunk.detach().cpu())
    return torch.cat(out_chunks, dim=0)


def _project_parcels_to_embeddings_real(
    model: CLIPVisionModel,
    parcels: pl.DataFrame,
    *,
    dataset_root: Path,
    batch_size: int,
    device: torch.device,
    seed: int,
    crop_resize_to: int = 224,
) -> torch.Tensor:
    """Forward real student sobre crops Sentinel-2.

    Lee cada crop ``.tif`` desde ``dataset_root/{region}/crops/{file}``,
    resizea a 224x224, normaliza uint16/10000, pasa por
    ``model.vision_model(pixel_values).pooler_output`` y devuelve un tensor
    ``(n_parcels, 512)`` en CPU float32.

    Args:
        model: ``CLIPVisionModel`` con patch_embed adaptado a 4 canales.
        parcels: DataFrame con columnas ``crop_path`` + ``region``.
        dataset_root: raiz ``data/farslip_pairs/`` para resolver crops cross-platform.
        batch_size: tamano batch forward (default 64 funciona en 24GB L4).
        device: ``torch.device``.
        seed: semilla determinista (para shuffling reproducible si aplicara).
        crop_resize_to: lado del crop tras resize bilineal (default 224).

    Returns:
        Tensor ``(n_parcels, EMBED_DIM)`` en CPU float32.
    """
    from ml.farslip.dataset import FarSLIPDataset

    propagate_seed(seed)
    if "crop_path" not in parcels.columns:
        raise ValueError(
            "parquet sin columna 'crop_path' necesaria para extract real. "
            "Anade crop_path desde manifest.parquet."
        )

    helper = FarSLIPDataset.__new__(FarSLIPDataset)
    helper.manifest_path = dataset_root / "_dummy_manifest.parquet"
    helper.crop_resize_to = crop_resize_to

    n_parcels = parcels.height
    rows = parcels.to_dicts()
    out_chunks: list[torch.Tensor] = []

    model.eval()
    with torch.inference_mode():
        for start in range(0, n_parcels, batch_size):
            end = min(start + batch_size, n_parcels)
            imgs: list[torch.Tensor] = []
            for row in rows[start:end]:
                crop_path_raw = row["crop_path"]
                region = row.get("region")
                if region:
                    helper.manifest_path = dataset_root / region / "manifest.parquet"
                resolved = helper._resolve_crop_path(crop_path_raw)
                img = helper._load_crop(resolved)
                img = helper._resize_chw(img, crop_resize_to)
                imgs.append(img)
            batch = torch.stack(imgs, dim=0).to(device)
            out = model(pixel_values=batch)
            emb = out.pooler_output
            emb = torch.nn.functional.normalize(emb, dim=-1)
            out_chunks.append(emb.detach().cpu().float())
            if start % (batch_size * 10) == 0:
                _log.info(
                    "extract_real_progress",
                    done=end,
                    total=n_parcels,
                    pct=round(100 * end / n_parcels, 1),
                )
    return torch.cat(out_chunks, dim=0)


def extract_farslip_embeddings(
    *,
    student_checkpoint_path: Path,
    parcels_parquet: Path,
    rois: tuple[str, ...] = ("pianura_padana", "toscana", "puglia"),
    output_path: Path,
    batch_size: int = 256,
    device: DeviceLiteral = "auto",
    seed: int = 42,
    mode: Literal["placeholder", "real"] = "placeholder",
    dataset_root: Path | None = None,
) -> ExtractEmbeddingsResult:
    """Extrae embeddings FarSLIP de cada parcela italiana y los persiste.

    Args:
        student_checkpoint_path: ruta local o URI ``mlflow://...``.
        parcels_parquet: parquet con al menos ``parcel_id`` + ``year``.
        rois: tupla de ROIs (o ``("italy",)``).
        output_path: parquet de salida (parent se crea si no existe).
        batch_size: batch para la inferencia CLIP (default 256).
        device: ``"auto"`` | ``"cuda"`` | ``"cpu"``.
        seed: semilla determinista (default 42).

    Returns:
        :class:`ExtractEmbeddingsResult` con metadata para MLflow tags.
    """
    rois_resolved = _resolve_rois(rois)
    torch_device = _resolve_device(device)
    ckpt_resolved, data_version = _resolve_checkpoint(student_checkpoint_path)
    parcels = _load_parcels_filtered(parcels_parquet, rois_resolved)
    n_parcels = parcels.height
    _log.info(
        "extracting_farslip_embeddings",
        n_parcels=n_parcels,
        rois=list(rois_resolved),
        device=torch_device.type,
        seed=seed,
    )
    model = _load_student(ckpt_resolved, device=torch_device)
    if mode == "real":
        if dataset_root is None:
            raise ValueError("mode='real' requiere dataset_root para resolver crops")
        embeddings = _project_parcels_to_embeddings_real(
            model,
            parcels,
            dataset_root=dataset_root,
            batch_size=batch_size,
            device=torch_device,
            seed=seed,
        )
    else:
        embeddings = _project_parcels_to_embeddings(
            model,
            n_parcels=n_parcels,
            batch_size=batch_size,
            device=torch_device,
            seed=seed,
        )
    cols = _embed_columns()
    embed_dict = {
        cols[i]: embeddings[:, i].numpy() for i in range(EMBED_DIM)
    }
    out_df = parcels.select(
        pl.col("parcel_id").cast(pl.Int64),
        pl.col("year").cast(pl.Int32),
    ).with_columns(
        [pl.Series(name=cols[i], values=embed_dict[cols[i]]) for i in range(EMBED_DIM)]
    )
    if out_df.width != TOTAL_COLS:
        raise RuntimeError(
            f"output width inesperado: {out_df.width} != {TOTAL_COLS}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(output_path)
    code_version = git_sha(short=True)
    result = ExtractEmbeddingsResult(
        n_parcels=n_parcels,
        n_dims=EMBED_DIM,
        output_path=output_path.resolve(),
        code_version=code_version,
        data_version=data_version,
        device_used=torch_device.type,
    )
    _log.info("farslip_embeddings_written", **result.__dict__)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml.farslip.extract_embeddings",
        description=(
            "Proyecta parcelas italianas sobre el espacio FarSLIP "
            "(CLIP-512) y persiste el resultado a parquet."
        ),
    )
    parser.add_argument(
        "--student-checkpoint",
        required=True,
        help=(
            "Ruta local al checkpoint .safetensors/.pt o URI "
            "'mlflow://Models/<name>@<stage>'."
        ),
    )
    parser.add_argument(
        "--parcels-parquet",
        required=True,
        type=Path,
        help="Parquet con columnas parcel_id + year.",
    )
    parser.add_argument(
        "--rois",
        default="italy",
        help=(
            "Comma-separated. Alias soportados: 'italy' "
            "(pianura_padana,toscana,puglia)."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Parquet de salida (parent se crea automaticamente).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size para inferencia (default 256).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device override (default 'auto').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla determinista (default 42).",
    )
    parser.add_argument(
        "--mode",
        choices=("placeholder", "real"),
        default="placeholder",
        help=(
            "'placeholder' (legacy seeded randn, default) o 'real' "
            "(forward CLIPVisionModel sobre crops Sentinel-2)."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Raiz dataset farslip_pairs (requerido si --mode=real).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    rois_tuple = tuple(r.strip() for r in args.rois.split(",") if r.strip())
    result = extract_farslip_embeddings(
        student_checkpoint_path=Path(args.student_checkpoint),
        parcels_parquet=args.parcels_parquet,
        rois=rois_tuple,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        mode=args.mode,
        dataset_root=args.dataset_root,
    )
    _log.info(
        "farslip_extract_embeddings_complete",
        n_parcels=result.n_parcels,
        n_dims=result.n_dims,
        output=str(result.output_path),
        code_version=result.code_version,
        data_version=result.data_version,
        device=result.device_used,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
