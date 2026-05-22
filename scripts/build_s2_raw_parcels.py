"""Genera el escenario (b) del baseline: Sentinel-2 crudo a nivel parcela.

Operativo de la US-022 (EPIC 4). Agrega las 10 bandas Sentinel-2 de
PASTIS-R (``PASTIS_S2_BANDS``) promediadas **temporal y espacialmente**
dentro de cada poligono de parcela, produciendo el escenario (b) de la
comparativa del baseline (``ml/eval/comparison.py``).

Para cada patch PASTIS-R:

1. Carga el tensor Sentinel-2 ``DATA_S2/S2_<patch_id>.npy`` de forma
   ``(T, 10, 128, 128)`` int16.
2. Carga la mascara instance ``ANNOTATIONS/TARGET_<patch_id>.npy`` canal 1,
   que asigna cada pixel a un ``instance_id`` (= una parcela).
3. Para cada ``instance_id`` presente en el patch, promedia las 10 bandas
   sobre los ``T`` instantes y sobre los pixeles de la mascara.

El ``parcel_id`` resultante es ``"<patch_id>_<instance_id>"``, identico a
la convencion de ``scripts/vectorize_pastis_parcels.py`` y de los parquets
de AlphaEarth y feature_selection — los tres escenarios alinean por
``parcel_id``.

Las parcelas a procesar (y sus ``class_id``, ``fold``) provienen del
GeoParquet ``data/processed/pastis_parcels_full.geoparquet`` (85.951
parcelas, generado por ``vectorize_pastis_parcels.py``); solo se agregan
las bandas de esas parcelas para garantizar el inner join con los otros
escenarios.

Uso::

    poetry run python scripts/build_s2_raw_parcels.py \\
        --pastis-root data/PASTIS-R \\
        --parcels data/processed/pastis_parcels_full.geoparquet \\
        --out data/cache/pastis/s2_raw_parcels_2019_85951.parquet \\
        --n-jobs -1
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl
import structlog
import typer
from joblib import Parallel, delayed

from ml.ingest.pastis_loader import PASTIS_S2_BANDS

logger = structlog.get_logger(__name__)
app = typer.Typer(add_completion=False, help=__doc__)

# Nombres de las columnas de banda media en el parquet de salida.
_BAND_MEAN_COLS: list[str] = [f"{band}_mean" for band in PASTIS_S2_BANDS]

# Schema canonico del parquet de salida (orden estable).
_OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    "parcel_id": pl.Utf8(),
    "patch_id": pl.Int64(),
    "instance_id": pl.Int64(),
    **{col: pl.Float64() for col in _BAND_MEAN_COLS},
    "class_id": pl.Int64(),
    "fold": pl.Int64(),
}


def aggregate_patch_bands(
    patch_id: int,
    s2_path: Path,
    instance_path: Path,
    parcels: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Agrega las 10 bandas S2 medias de las parcelas de un patch.

    Carga el tensor Sentinel-2 y la mascara instance del patch, y para
    cada parcela solicitada calcula la media de cada banda sobre los
    ``T`` instantes temporales y sobre los pixeles de su mascara.

    Args:
        patch_id: Identificador del patch PASTIS-R.
        s2_path: Ruta al tensor ``S2_<patch_id>.npy`` ``(T, 10, H, W)``.
        instance_path: Ruta al ``TARGET_<patch_id>.npy`` ``(3, H, W)``;
            el canal 1 es la mascara instance.
        parcels: Lista de dicts con ``parcel_id``, ``instance_id``,
            ``class_id`` y ``fold`` de las parcelas de este patch.

    Returns:
        Lista de dicts con ``parcel_id``, ``patch_id``, ``instance_id``,
        las 10 columnas ``<banda>_mean``, ``class_id`` y ``fold``. Vacia
        si los tensores no se pueden cargar o ninguna parcela tiene
        pixeles validos.
    """
    if not s2_path.exists() or not instance_path.exists():
        logger.warning(
            "patch_tensor_missing",
            patch_id=patch_id,
            s2=str(s2_path),
            instance=str(instance_path),
        )
        return []

    try:
        s2 = np.load(s2_path)
        target = np.load(instance_path)
    except (OSError, ValueError) as exc:
        logger.warning("patch_load_failed", patch_id=patch_id, error=str(exc))
        return []

    if s2.ndim != 4 or s2.shape[1] != len(PASTIS_S2_BANDS):
        logger.warning(
            "patch_s2_unexpected_shape", patch_id=patch_id, shape=tuple(s2.shape)
        )
        return []
    if target.ndim != 3 or target.shape[0] < 2:
        logger.warning(
            "patch_target_unexpected_shape",
            patch_id=patch_id,
            shape=tuple(target.shape),
        )
        return []

    instance = target[1]
    # Media temporal por banda: (T, 10, H, W) -> (10, H, W). float64 evita
    # overflow del int16 al sumar T instantes.
    band_means_thw = s2.astype(np.float64).mean(axis=0)

    records: list[dict[str, object]] = []
    for parcel in parcels:
        instance_id = int(parcel["instance_id"])  # type: ignore[arg-type]
        mask = instance == instance_id
        n_pixels = int(mask.sum())
        if n_pixels == 0:
            # La parcela del GeoParquet no tiene pixeles en la mascara
            # instance de este patch (raro: instance_id desfasado).
            continue
        # Media espacial de cada banda sobre los pixeles de la mascara.
        band_values = band_means_thw[:, mask].mean(axis=1)
        record: dict[str, object] = {
            "parcel_id": str(parcel["parcel_id"]),
            "patch_id": int(patch_id),
            "instance_id": instance_id,
            "class_id": int(parcel["class_id"]),  # type: ignore[arg-type]
            "fold": int(parcel["fold"]),  # type: ignore[arg-type]
        }
        for col, value in zip(_BAND_MEAN_COLS, band_values, strict=True):
            record[col] = float(value)
        records.append(record)

    return records


def _parcels_by_patch(
    parcels_path: Path,
) -> dict[int, list[dict[str, object]]]:
    """Agrupa las parcelas del GeoParquet por ``patch_id``.

    Lee solo las columnas escalares (sin la geometria) — la agregacion de
    bandas usa la mascara instance del raster, no el poligono vectorial.

    Args:
        parcels_path: Ruta al ``pastis_parcels_full.geoparquet``.

    Returns:
        Mapa ``{patch_id: [parcela, ...]}`` donde cada parcela es un dict
        con ``parcel_id``, ``instance_id``, ``class_id`` y ``fold``.

    Raises:
        FileNotFoundError: si el GeoParquet no existe.
    """
    if not parcels_path.exists():
        raise FileNotFoundError(
            f"GeoParquet de parcelas no encontrado en {parcels_path}. "
            "Genera las parcelas con `scripts/vectorize_pastis_parcels.py`."
        )

    # `pl.read_parquet` lee el GeoParquet ignorando la columna `geometry`
    # binaria; seleccionamos solo las columnas escalares necesarias.
    cols = ["parcel_id", "patch_id", "instance_id", "class_id", "fold"]
    parcels_df = pl.read_parquet(parcels_path, columns=cols)

    grouped: dict[int, list[dict[str, object]]] = {}
    for row in parcels_df.iter_rows(named=True):
        patch_id = int(row["patch_id"])
        grouped.setdefault(patch_id, []).append(
            {
                "parcel_id": str(row["parcel_id"]),
                "instance_id": int(row["instance_id"]),
                "class_id": int(row["class_id"]),
                "fold": int(row["fold"]),
            }
        )
    return grouped


@app.command()
def main(
    pastis_root: Path = typer.Option(  # noqa: B008
        Path("data/PASTIS-R"),
        "--pastis-root",
        help="Raiz del dataset PASTIS-R (con DATA_S2/ y ANNOTATIONS/).",
    ),
    parcels: Path = typer.Option(  # noqa: B008
        Path("data/processed/pastis_parcels_full.geoparquet"),
        "--parcels",
        help="GeoParquet de poligonos de parcela (vectorize_pastis_parcels.py).",
    ),
    out: Path = typer.Option(  # noqa: B008
        Path("data/cache/pastis/s2_raw_parcels_2019_85951.parquet"),
        "--out",
        help="Parquet de salida con las 10 bandas medias por parcela.",
    ),
    n_jobs: int = typer.Option(
        -1, "--n-jobs", help="Paralelismo joblib (-1 = todos los nucleos)."
    ),
    limit_patches: int = typer.Option(
        0,
        "--limit-patches",
        help="Si > 0, procesa solo los primeros N patches (smoke test).",
    ),
) -> None:
    """Genera el escenario (b) Sentinel-2 crudo a nivel parcela."""
    if not pastis_root.exists():
        logger.error("pastis_root_missing", path=str(pastis_root))
        raise typer.Exit(code=2)

    s2_dir = pastis_root / "DATA_S2"
    annotations_dir = pastis_root / "ANNOTATIONS"
    if not s2_dir.exists() or not annotations_dir.exists():
        logger.error(
            "pastis_subdirs_missing",
            data_s2=str(s2_dir),
            annotations=str(annotations_dir),
        )
        raise typer.Exit(code=2)

    logger.info("s2_raw_parcels_start", parcels=str(parcels), out=str(out))
    parcels_by_patch = _parcels_by_patch(parcels)
    patch_ids = sorted(parcels_by_patch)
    if limit_patches > 0:
        patch_ids = patch_ids[:limit_patches]
    n_parcels_total = sum(len(parcels_by_patch[p]) for p in patch_ids)
    logger.info(
        "patches_to_process",
        n_patches=len(patch_ids),
        n_parcels=n_parcels_total,
        n_jobs=n_jobs,
    )

    t0 = time.time()
    batches: list[list[dict[str, object]]] = Parallel(n_jobs=n_jobs)(
        delayed(aggregate_patch_bands)(
            patch_id,
            s2_dir / f"S2_{patch_id}.npy",
            annotations_dir / f"TARGET_{patch_id}.npy",
            parcels_by_patch[patch_id],
        )
        for patch_id in patch_ids
    )
    all_records: list[dict[str, object]] = [rec for batch in batches for rec in batch]
    elapsed = time.time() - t0
    logger.info(
        "s2_raw_aggregation_done",
        n_records=len(all_records),
        elapsed_s=round(elapsed, 1),
    )

    if not all_records:
        logger.error("no_records_extracted")
        raise typer.Exit(code=3)

    df = pl.DataFrame(all_records, schema=_OUTPUT_SCHEMA).sort("parcel_id")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)
    file_size_mb = out.stat().st_size / 1e6
    logger.info(
        "s2_raw_parcels_written",
        path=str(out),
        n_parcels=df.height,
        n_bands=len(_BAND_MEAN_COLS),
        file_size_mb=round(file_size_mb, 2),
    )

    fold_counts = (
        df.group_by("fold").len().sort("fold").to_dicts() if df.height else []
    )
    n_classes = df.get_column("class_id").n_unique()
    typer.echo(
        f"S2 raw parcels: {df.height} parcelas, {len(_BAND_MEAN_COLS)} bandas, "
        f"{n_classes} clases, folds={fold_counts}"
    )
    typer.echo(f"Output: {out}")


if __name__ == "__main__":
    app()
