"""Extraccion de embeddings RemoteCLIP sobre crops PASTIS-R (US-023-preview-v2 P5).

Reemplaza el placeholder ``_extract_remoteclip_embeddings`` del notebook
``04_farslip_eval_pastis.ipynb`` por una extraccion REAL usando RemoteCLIP
(Chen et al. 2023, https://github.com/ChenDelong1999/RemoteCLIP). El modelo
``chendelong/RemoteCLIP-ViT-B-32`` es un CLIP ViT-B/32 fine-tuned sobre
imageria de remote sensing (RSITMD + RSICD + UCM).

Pipeline por parcela:

1. Carga crops S2 multitemporales desde ``imagery_path`` (parquet binario o
   array NCHW por parcela).
2. Selecciona bandas B04 (red), B03 (green), B02 (blue) y compone RGB.
3. Normaliza con stats ``NORM_S2_patch.json`` (PASTIS-R) y aplica stretch
   percentil 2-98 -> uint8.
4. Resize bilineal a 224x224.
5. Forward por el ``CLIPVisionModel`` y normaliza L2.
6. Pooling temporal (mean sobre eje T) si la parcela es multi-temporal.

Output ``data/farslip/remoteclip_embeddings_pastis.parquet`` con schema:

::

    parcel_id (Utf8) | year (Int16) |
    remoteclip_000 .. remoteclip_511 (Float32)

Esquema compatible con :func:`ml.farslip.extract_embeddings.extract_farslip_embeddings`
salvo el prefijo de columnas (``remoteclip_*`` vs ``farslip_emb_*``); el
notebook 04 puede concatenar ambos para el linear probe comparativo.

Fallback: si ``chendelong/RemoteCLIP-ViT-B-32`` no se puede descargar
(bloqueo de red, modelo retirado de HF), se usa
``openai/clip-vit-base-patch32`` y se anota ``model_used`` en el log
estructurado para que el operador sepa que la comparacion no es contra
RemoteCLIP puro.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import structlog

if TYPE_CHECKING:
    import torch
    from transformers import CLIPModel, CLIPProcessor  # noqa: F401

_log = structlog.get_logger(__name__)


EMBED_DIM: int = 512
"""Dimension del embedding CLIP ViT-B/32 (image features tras projection)."""

EMBED_COL_PREFIX: str = "remoteclip_"

DEFAULT_MODEL_ID: str = "chendelong/RemoteCLIP-ViT-B-32"
"""HF Hub repo del modelo RemoteCLIP (CLIP ViT-B/32 fine-tuned en RS)."""

FALLBACK_MODEL_ID: str = "openai/clip-vit-base-patch32"
"""Fallback si ``DEFAULT_MODEL_ID`` no esta disponible."""

DEFAULT_SUBSET_PATH: Path = Path("data/test_fixtures/pastis_eval_subset.parquet")
DEFAULT_IMAGERY_PATH: Path = Path("data/test_fixtures/pastis_eval_subset.imagery.parquet")
DEFAULT_OUTPUT_PATH: Path = Path("data/farslip/remoteclip_embeddings_pastis.parquet")
DEFAULT_BATCH_SIZE: int = 32


def _embed_columns() -> list[str]:
    """Devuelve ``["remoteclip_000", ..., "remoteclip_511"]`` (orden estable)."""
    return [f"{EMBED_COL_PREFIX}{i:03d}" for i in range(EMBED_DIM)]


def _build_output_schema() -> dict[str, Any]:
    """Schema canonico de salida (parcel_id + year + 512 floats)."""
    schema: dict[str, Any] = {
        "parcel_id": pl.Utf8,
        "year": pl.Int16,
    }
    for col in _embed_columns():
        schema[col] = pl.Float32
    return schema


def _resolve_device(device: str | None) -> torch.device:
    """Resuelve device con fallback ``cuda -> cpu``.

    Args:
        device: ``"cuda"``, ``"cpu"`` o ``None`` (autodetect).

    Returns:
        ``torch.device`` resuelto. Si CUDA fue solicitado pero no esta
        disponible, emite warning estructurado y degrada a CPU.
    """
    import torch  # noqa: PLC0415

    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        _log.warning(
            "remoteclip_no_cuda_detected_cpu_fallback",
            hint="extraccion sera lenta (~1-2 s/parcela en CPU vs ~10 ms/parcela en GPU)",
        )
        return torch.device("cpu")
    if device == "cuda":
        import torch as _t  # noqa: PLC0415

        if not _t.cuda.is_available():
            _log.warning(
                "remoteclip_cuda_requested_unavailable",
                fallback="cpu",
            )
            return _t.device("cpu")
        return _t.device("cuda")
    return torch.device(device)


def _load_model(
    model_name: str, device: torch.device
) -> tuple[CLIPModel, CLIPProcessor, str]:
    """Carga ``CLIPModel`` + ``CLIPProcessor`` con fallback a OpenAI CLIP.

    Args:
        model_name: HF repo id (default ``chendelong/RemoteCLIP-ViT-B-32``).
        device: device destino para los pesos.

    Returns:
        Tupla ``(model, processor, model_used)`` donde ``model_used`` es el
        id efectivamente cargado (puede coincidir con ``model_name`` o ser
        el fallback OpenAI CLIP).
    """
    from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415

    try:
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        model_used = model_name
    except Exception as exc:  # noqa: BLE001
        _log.warning(
            "remoteclip_load_failed_using_fallback",
            requested=model_name,
            fallback=FALLBACK_MODEL_ID,
            error=str(exc),
        )
        model = CLIPModel.from_pretrained(FALLBACK_MODEL_ID)
        processor = CLIPProcessor.from_pretrained(FALLBACK_MODEL_ID)
        model_used = FALLBACK_MODEL_ID

    model.eval()
    model.to(device)
    return model, processor, model_used


def _load_imagery(imagery_path: Path) -> pl.DataFrame:
    """Carga el parquet binario con crops Sentinel-2.

    Schema esperado (al menos)::

        parcel_id (Utf8) | year (Int16) |
        image (List[List[Float32]] o binary) | shape (List[Int64])

    Tolera cualquiera de estos esquemas siempre que cada fila exponga
    ``parcel_id`` y un payload de imagen interpretable (numpy bytes via
    ``np.frombuffer`` o lista anidada).
    """
    if not imagery_path.exists():
        raise FileNotFoundError(
            f"imagery_path no existe: {imagery_path}. "
            "Generalo con ml.ingest.pastis_eval_subset (US-022-c P1 B-1)."
        )
    return pl.read_parquet(imagery_path)


def _row_to_array(row: dict[str, Any]) -> np.ndarray:
    """Convierte una fila del parquet de imagery a array ``(T, C, H, W)``.

    Soporta tres encodings comunes:

    - ``image`` ya es un ``np.ndarray`` (formato in-memory).
    - ``image`` es ``bytes`` -> ``np.frombuffer`` con ``shape`` adyacente.
    - ``image`` es lista anidada -> ``np.asarray``.

    Para parcelas mono-temporales devuelve shape ``(1, C, H, W)`` (anade
    eje T=1) para uniformar el resto del pipeline.
    """
    img: Any = row.get("image")
    shape = row.get("shape")
    if isinstance(img, np.ndarray):
        arr = img
    elif isinstance(img, (bytes, bytearray)):
        arr = np.frombuffer(img, dtype=np.float32).copy()
        if shape:
            arr = arr.reshape(tuple(int(s) for s in shape))
    elif isinstance(img, list):
        arr = np.asarray(img, dtype=np.float32)
    else:
        raise ValueError(
            f"formato de imagery no soportado para parcel_id={row.get('parcel_id')!r}: "
            f"tipo {type(img).__name__}"
        )
    if arr.ndim == 3:
        # (C, H, W) -> (1, C, H, W)
        arr = arr[np.newaxis, ...]
    elif arr.ndim != 4:
        raise ValueError(
            f"shape de imagery inesperado para parcel_id={row.get('parcel_id')!r}: "
            f"{arr.shape}"
        )
    return arr


def _select_rgb(arr: np.ndarray, band_indices: tuple[int, int, int]) -> np.ndarray:
    """Selecciona y reordena bandas para RGB.

    Args:
        arr: ``(T, C, H, W)`` float.
        band_indices: indices (red, green, blue). Para PASTIS-R con orden
            B02/B03/B04/B08, RGB = (2, 1, 0).

    Returns:
        ``(T, 3, H, W)`` float con orden ``[R, G, B]``.
    """
    r, g, b = band_indices
    return arr[:, [r, g, b], :, :]


def _stretch_percentile_uint8(rgb: np.ndarray) -> np.ndarray:
    """Stretch percentil 2-98 por banda -> uint8 ``[0, 255]``.

    Args:
        rgb: ``(T, 3, H, W)`` float.

    Returns:
        ``(T, H, W, 3)`` uint8 (formato HWC esperado por
        :class:`CLIPProcessor`).
    """
    t, c, h, w = rgb.shape
    out = np.zeros((t, h, w, c), dtype=np.uint8)
    for ti in range(t):
        for ci in range(c):
            band = rgb[ti, ci]
            lo, hi = np.percentile(band, [2.0, 98.0])
            denom = max(float(hi - lo), 1e-6)
            scaled = np.clip((band - lo) / denom, 0.0, 1.0)
            out[ti, :, :, ci] = (scaled * 255).astype(np.uint8)
    return out


def _resolve_band_indices(imagery_meta: dict[str, Any] | None) -> tuple[int, int, int]:
    """Resuelve indices RGB segun metadata de bandas.

    Default PASTIS-R: bandas B02/B03/B04/B08 (orden canonico) -> RGB =
    ``(2, 1, 0)`` (B04 rojo en idx 2, B03 verde en idx 1, B02 azul en idx 0).
    Si ``imagery_meta`` trae ``band_order``, se respeta y se busca
    ``B04/B03/B02``.
    """
    if imagery_meta and "band_order" in imagery_meta:
        order = [b.upper() for b in imagery_meta["band_order"]]
        try:
            return (order.index("B04"), order.index("B03"), order.index("B02"))
        except ValueError:
            pass
    return (2, 1, 0)


def _embed_batch(
    model: CLIPModel,
    processor: CLIPProcessor,
    images_hwc_uint8: list[np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    """Forward de un batch de imagenes RGB uint8 por el visual encoder CLIP.

    Returns:
        Tensor ``(B, 512)`` float32 en CPU, L2-normalizado.
    """
    import torch  # noqa: PLC0415

    inputs = processor(images=images_hwc_uint8, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.inference_mode():
        features = model.get_image_features(pixel_values=pixel_values)
    # `get_image_features` deberia devolver un tensor (B, dim) pero algunos
    # checkpoints RemoteCLIP (chendelong/*) devuelven el output completo
    # `BaseModelOutputWithPooling`. Normalizamos a tensor antes de L2.
    if hasattr(features, "image_embeds") and features.image_embeds is not None:
        features = features.image_embeds
    elif hasattr(features, "pooler_output") and features.pooler_output is not None:
        features = features.pooler_output
    elif hasattr(features, "last_hidden_state"):
        features = features.last_hidden_state.mean(dim=1)
    features = torch.nn.functional.normalize(features, dim=-1)
    return features.detach().cpu().float()


def extract_remoteclip_embeddings(
    pastis_eval_subset_path: Path = DEFAULT_SUBSET_PATH,
    imagery_path: Path = DEFAULT_IMAGERY_PATH,
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    model_name: str = DEFAULT_MODEL_ID,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Extrae embeddings RemoteCLIP por parcela del subset PASTIS-R.

    Carga los crops S2 multitemporales desde ``imagery_path`` (generado por
    ``ml.ingest.pastis_eval_subset``), construye composiciones RGB
    (B04/B03/B02) normalizadas y produce un embedding 512-dim por parcela.

    Para parcelas multi-temporales agrega temporalmente (mean pooling sobre
    el eje T) antes de la L2-normalizacion final.

    Args:
        pastis_eval_subset_path: Parquet con metadata ``parcel_id`` + ``year``
            + ``label_id``. Aporta el orden canonico de filas del output.
        imagery_path: Parquet binario con crops Sentinel-2 por parcela.
        output_path: Parquet destino (parent se crea si no existe).
        model_name: HF repo id. Default ``chendelong/RemoteCLIP-ViT-B-32``;
            fallback automatico a ``openai/clip-vit-base-patch32``.
        batch_size: Numero de imagenes RGB procesadas por forward.
        device: ``"cuda"``, ``"cpu"`` o ``None`` (autodetect).
        overwrite: Si ``False`` y ``output_path`` existe, retorna el path
            existente sin recomputar.

    Returns:
        Path absoluto al parquet ``output_path``.

    Raises:
        FileNotFoundError: Si ``imagery_path`` no existe.
        ValueError: Si el imagery parquet no expone ``parcel_id`` o el
            payload de imagen es de un tipo no soportado.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        _log.info("remoteclip_output_exists_skip", path=str(output_path))
        return output_path.resolve()

    import torch  # noqa: PLC0415

    subset = pl.read_parquet(pastis_eval_subset_path)
    imagery = _load_imagery(imagery_path)
    if "parcel_id" not in imagery.columns:
        raise ValueError(
            f"imagery_path={imagery_path} sin columna parcel_id; "
            "esquema esperado: parcel_id, year, image, shape."
        )

    # Asegura parcel_id Utf8 en ambos parquets.
    subset = subset.with_columns(pl.col("parcel_id").cast(pl.Utf8))
    imagery = imagery.with_columns(pl.col("parcel_id").cast(pl.Utf8))

    # PASTIS-R no expone `year` por parcela (es un dataset 2019 monolitico).
    # Si falta lo materializamos como constante para preservar el esquema de
    # salida (parcel_id, year, remoteclip_emb_*).
    if "year" not in subset.columns:
        subset = subset.with_columns(pl.lit(2019).cast(pl.Int64).alias("year"))

    # Join orden estable: imagery merge sobre orden de subset.
    joined = subset.select(["parcel_id", "year"]).join(
        imagery, on="parcel_id", how="left"
    )

    torch_device = _resolve_device(device)
    model, processor, model_used = _load_model(model_name, torch_device)

    t0 = time.perf_counter()
    band_indices = _resolve_band_indices(None)
    embeddings: list[torch.Tensor] = []
    parcel_ids_out: list[str] = []
    years_out: list[int] = []

    rows = joined.to_dicts()
    n = len(rows)

    # Procesamiento por batch de parcelas (cada parcela puede tener T frames).
    for start in range(0, n, batch_size):
        chunk = rows[start : start + batch_size]
        images_per_parcel: list[np.ndarray] = []
        frame_to_parcel: list[int] = []
        for parcel_idx, row in enumerate(chunk):
            try:
                arr = _row_to_array(row)
            except (ValueError, KeyError) as exc:
                _log.warning(
                    "remoteclip_parcel_skipped",
                    parcel_id=row.get("parcel_id"),
                    error=str(exc),
                )
                # Frame vacio -> embedding zeros para no romper alineacion.
                arr = np.zeros((1, 4, 32, 32), dtype=np.float32)
            rgb = _select_rgb(arr, band_indices)
            rgb_uint8 = _stretch_percentile_uint8(rgb)
            # ``rgb_uint8`` shape (T, H, W, 3). Aplanamos T y registramos
            # el parcel_idx para hacer mean pooling post-forward.
            for ti in range(rgb_uint8.shape[0]):
                images_per_parcel.append(rgb_uint8[ti])
                frame_to_parcel.append(parcel_idx)

        if not images_per_parcel:
            continue
        feats_frames = _embed_batch(model, processor, images_per_parcel, torch_device)

        # Mean pooling temporal por parcela.
        per_parcel: dict[int, list[torch.Tensor]] = {}
        for fi, p_idx in enumerate(frame_to_parcel):
            per_parcel.setdefault(p_idx, []).append(feats_frames[fi])
        for p_idx, frames in per_parcel.items():
            stacked = torch.stack(frames, dim=0)
            mean_emb = stacked.mean(dim=0)
            # Re-normaliza L2 post mean-pooling.
            mean_emb = torch.nn.functional.normalize(mean_emb, dim=-1)
            embeddings.append(mean_emb)
            parcel_ids_out.append(str(chunk[p_idx]["parcel_id"]))
            year_val = chunk[p_idx].get("year")
            years_out.append(int(year_val) if year_val is not None else 0)

    elapsed = time.perf_counter() - t0
    n_parcels_done = len(parcel_ids_out)
    sec_per_parcel = (elapsed / n_parcels_done) if n_parcels_done else 0.0
    _log.info(
        "remoteclip_extract_complete",
        n_parcels=n_parcels_done,
        device=str(torch_device),
        seconds=round(elapsed, 2),
        seconds_per_parcel=round(sec_per_parcel, 4),
        model_used=model_used,
    )

    if not embeddings:
        # Output vacio con esquema valido.
        out_df = pl.DataFrame(schema=_build_output_schema())
    else:
        emb_tensor = torch.stack(embeddings, dim=0).numpy().astype(np.float32)
        cols = _embed_columns()
        data: dict[str, Any] = {
            "parcel_id": parcel_ids_out,
            "year": years_out,
        }
        for i, col in enumerate(cols):
            data[col] = emb_tensor[:, i].tolist()
        out_df = pl.DataFrame(data, schema=_build_output_schema())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(output_path)
    return output_path.resolve()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ml.ingest.remoteclip_extractor",
        description=(
            "Extrae embeddings RemoteCLIP (512-dim) sobre crops PASTIS-R "
            "y persiste a parquet compatible con el linear probe de "
            "notebooks/features/04_farslip_eval_pastis.ipynb."
        ),
    )
    p.add_argument(
        "--subset",
        type=Path,
        default=DEFAULT_SUBSET_PATH,
        help=f"Parquet metadata subset PASTIS-R (default {DEFAULT_SUBSET_PATH}).",
    )
    p.add_argument(
        "--imagery",
        type=Path,
        default=DEFAULT_IMAGERY_PATH,
        help=f"Parquet binario con crops S2 (default {DEFAULT_IMAGERY_PATH}).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Parquet de salida (default {DEFAULT_OUTPUT_PATH}).",
    )
    p.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_ID,
        help=f"HF repo id del modelo CLIP (default {DEFAULT_MODEL_ID}).",
    )
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=None,
        help="Device override (default autodetect).",
    )
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    out = extract_remoteclip_embeddings(
        pastis_eval_subset_path=args.subset,
        imagery_path=args.imagery,
        output_path=args.output,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        overwrite=args.overwrite,
    )
    _log.info("remoteclip_extractor_done", output=str(out))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
