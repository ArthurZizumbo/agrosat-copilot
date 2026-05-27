"""Construye un subset estratificado REAL de PASTIS-R para evaluacion FarSLIP/RemoteCLIP.

Este modulo es la fuente unica de verdad para el fixture
``data/test_fixtures/pastis_eval_subset.parquet`` consumido por
``notebooks/baseline/04_farslip_eval_pastis.ipynb`` y por los smoke tests
de los encoders FarSLIP / RemoteCLIP en EPIC 4.

Reglas:
    - NUNCA genera datos sinteticos. Si PASTIS-R no esta presente en disco
      (``data/PASTIS-R/metadata.geojson`` y ``DATA_S2/``) se lanza
      ``FileNotFoundError`` con la instruccion de descarga (DVC pull o link
      al dataset oficial INRAE).
    - Determinismo total: ``seed=42`` por defecto -> el MD5 del parquet
      debe ser estable run-to-run.
    - Polars 1.x para I/O parquet, sin pandas.
    - Logs estructurados via ``structlog``.

CLI:
    poetry run python -m ml.ingest.pastis_eval_subset \\
        --output data/test_fixtures/pastis_eval_subset.parquet \\
        --n-samples 1024 --seed 42 --stratify-by class
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import structlog

from ml.ingest.pastis_loader import (
    PASTIS_R_CLASSES,
    PASTIS_S2_BANDS,
    pastis_patch_coords,
    pastis_patch_index,
)

_log = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PASTIS_ROOT = _REPO_ROOT / "data" / "PASTIS-R"
_DEFAULT_OUTPUT = _REPO_ROOT / "data" / "test_fixtures" / "pastis_eval_subset.parquet"

_VOID_CLASS = 19
_BACKGROUND_CLASS = 0
_VALID_CLASS_RANGE = range(1, 19)  # 1..18 inclusive


_StratifyBy = Literal["class", "tile", "fold"]


# ---------------------------------------------------------------------------
# Errores
# ---------------------------------------------------------------------------


def _raise_missing_pastis(pastis_root: Path) -> None:
    """Lanza FileNotFoundError con instruccion de descarga.

    Args:
        pastis_root: Raiz esperada del dataset PASTIS-R.

    Raises:
        FileNotFoundError: Siempre. Incluye el path faltante y los comandos
            de descarga (DVC pull) o enlace al dataset oficial.
    """
    msg = (
        f"PASTIS-R no encontrado en {pastis_root}. "
        "Esperado: metadata.geojson + DATA_S2/ + ANNOTATIONS/. "
        "Para obtenerlo: `dvc pull data/PASTIS-R.dvc` "
        "o descarga manual desde "
        "https://zenodo.org/record/5735646 (PASTIS-R, INRAE, CC-BY-SA-4.0)."
    )
    raise FileNotFoundError(msg)


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def _validate_pastis_present(pastis_root: Path) -> None:
    """Valida que la estructura minima de PASTIS-R exista en disco.

    Args:
        pastis_root: Raiz esperada del dataset.

    Raises:
        FileNotFoundError: Si falta ``metadata.geojson`` o ``DATA_S2/``.
    """
    metadata = pastis_root / "metadata.geojson"
    data_s2 = pastis_root / "DATA_S2"
    if not metadata.exists() or not data_s2.exists() or not data_s2.is_dir():
        _raise_missing_pastis(pastis_root)


def _patch_majority_class(
    semantic: np.ndarray,
    exclude: tuple[int, ...] = (_BACKGROUND_CLASS, _VOID_CLASS),
) -> int:
    """Devuelve la clase mayoritaria (1..18) de un patch.

    Args:
        semantic: Mapa de clase 2D ``(H, W)``.
        exclude: Clases a excluir del conteo.

    Returns:
        int en 1..18 con la clase mayoritaria, o 0 si todo es background/void.
    """
    flat = semantic.ravel()
    mask = ~np.isin(flat, np.asarray(exclude, dtype=flat.dtype))
    filtered = flat[mask]
    if filtered.size == 0:
        return 0
    vals, counts = np.unique(filtered, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _load_target(pastis_root: Path, patch_id: str) -> np.ndarray | None:
    """Carga ``TARGET_<patch_id>.npy`` o ``None`` si no existe.

    Args:
        pastis_root: Raiz del dataset.
        patch_id: Identificador del patch.

    Returns:
        ndarray de shape ``(3, H, W)`` o ``None`` si no existe.
    """
    tgt = pastis_root / "ANNOTATIONS" / f"TARGET_{patch_id}.npy"
    if not tgt.exists():
        return None
    return np.load(tgt)


def _load_s2(pastis_root: Path, patch_id: str) -> np.ndarray | None:
    """Carga ``S2_<patch_id>.npy`` o ``None`` si no existe.

    Args:
        pastis_root: Raiz del dataset.
        patch_id: Identificador del patch.

    Returns:
        ndarray ``(T, 10, H, W)`` o ``None``.
    """
    s2 = pastis_root / "DATA_S2" / f"S2_{patch_id}.npy"
    if not s2.exists():
        return None
    return np.load(s2)


def _enumerate_parcels(
    pastis_root: Path,
    patch_ids: list[str],
) -> pl.DataFrame:
    """Enumera todas las parcelas ``(patch_id, instance_id, class_id, n_pixels)``.

    Una parcela = (patch_id, instance_id) unica derivada del canal 1
    (instancia) de ``TARGET_<patch_id>.npy``. La clase se toma de la moda
    del canal 0 (semantic) restringido a los pixeles de esa instancia.

    Args:
        pastis_root: Raiz PASTIS-R.
        patch_ids: Lista de patch_ids a escanear.

    Returns:
        DataFrame con columnas ``patch_id`` (Utf8), ``instance_id`` (Int64),
        ``class_id`` (Int64), ``n_pixels`` (Int64).
    """
    rows: list[dict[str, Any]] = []
    for pid in patch_ids:
        target = _load_target(pastis_root, pid)
        if target is None or target.ndim != 3 or target.shape[0] < 2:
            continue
        semantic = target[0]
        instance = target[1]
        inst_ids = np.unique(instance)
        for iid in inst_ids:
            iid_int = int(iid)
            if iid_int == 0:
                # 0 = sin instancia (background)
                continue
            mask = instance == iid
            n_pixels = int(np.count_nonzero(mask))
            if n_pixels == 0:
                continue
            cls_pixels = semantic[mask]
            cls_pixels = cls_pixels[
                ~np.isin(cls_pixels, np.asarray([_BACKGROUND_CLASS, _VOID_CLASS]))
            ]
            if cls_pixels.size == 0:
                continue
            vals, counts = np.unique(cls_pixels, return_counts=True)
            class_id = int(vals[int(np.argmax(counts))])
            if class_id not in _VALID_CLASS_RANGE:
                continue
            rows.append(
                {
                    "patch_id": pid,
                    "instance_id": iid_int,
                    "class_id": class_id,
                    "n_pixels": n_pixels,
                }
            )
    if not rows:
        return pl.DataFrame(
            schema={
                "patch_id": pl.Utf8,
                "instance_id": pl.Int64,
                "class_id": pl.Int64,
                "n_pixels": pl.Int64,
            }
        )
    return pl.DataFrame(rows)


def _stratified_sample(
    parcels: pl.DataFrame,
    n_samples: int,
    stratify_by: _StratifyBy,
    seed: int,
) -> pl.DataFrame:
    """Muestra ``n_samples`` parcelas estratificadas por la dimension indicada.

    Garantia (solo para ``stratify_by='class'``): cada clase con disponibilidad
    en ``parcels`` recibe al menos ``max(8, n_samples // 36)`` muestras (o todas
    las disponibles si hay menos).

    Args:
        parcels: DataFrame con columnas ``patch_id, instance_id, class_id,
            n_pixels, tile, fold``.
        n_samples: Tamano objetivo del subset.
        stratify_by: Dimension de estratificacion (``class``, ``tile``, ``fold``).
        seed: Semilla numpy para reproducibilidad.

    Returns:
        DataFrame muestreado con tamano <= ``n_samples`` (puede ser menor si
        no hay suficientes parcelas en el catalogo).
    """
    rng = np.random.default_rng(seed)
    col_map = {"class": "class_id", "tile": "tile", "fold": "fold"}
    strat_col = col_map[stratify_by]

    if parcels.is_empty():
        return parcels

    groups = parcels.group_by(strat_col).agg(pl.len().alias("_count"))
    n_groups = groups.height
    per_group_floor = max(1, n_samples // max(n_groups, 1))
    class_min = max(8, n_samples // 36) if stratify_by == "class" else per_group_floor

    selected_indices: list[int] = []
    parcels_with_idx = parcels.with_row_index(name="_row_idx")

    # Pase 1: garantizar minimo por grupo
    for grp_val in groups[strat_col].to_list():
        sub = parcels_with_idx.filter(pl.col(strat_col) == grp_val)
        sub_idx = sub["_row_idx"].to_list()
        target = min(class_min, len(sub_idx))
        choice = rng.choice(len(sub_idx), size=target, replace=False)
        selected_indices.extend(int(sub_idx[i]) for i in choice)

    # Pase 2: rellenar hasta n_samples con remainder distribuido proporcional
    remaining = n_samples - len(selected_indices)
    if remaining > 0:
        already = set(selected_indices)
        pool = parcels_with_idx.filter(~pl.col("_row_idx").is_in(list(already)))
        if not pool.is_empty():
            extra = min(remaining, pool.height)
            pool_idx = pool["_row_idx"].to_list()
            choice = rng.choice(len(pool_idx), size=extra, replace=False)
            selected_indices.extend(int(pool_idx[i]) for i in choice)
    elif remaining < 0:
        # Caso edge: el pase 1 ya excedio n_samples (n_groups * class_min > n_samples).
        # Truncamos manteniendo al menos 1 por grupo presente.
        kept: dict[Any, list[int]] = defaultdict(list)
        for idx in selected_indices:
            grp = parcels.row(idx, named=True)[strat_col]
            kept[grp].append(idx)
        # Round-robin hasta llenar n_samples
        new_selection: list[int] = []
        cursors = {k: 0 for k in kept}
        while len(new_selection) < n_samples:
            progressed = False
            for k in list(kept.keys()):
                c = cursors[k]
                if c < len(kept[k]):
                    new_selection.append(kept[k][c])
                    cursors[k] = c + 1
                    progressed = True
                    if len(new_selection) >= n_samples:
                        break
            if not progressed:
                break
        selected_indices = new_selection

    selected_indices = sorted(set(selected_indices))
    return parcels_with_idx.filter(pl.col("_row_idx").is_in(selected_indices)).drop(
        "_row_idx"
    )


def _build_imagery_blob(
    pastis_root: Path,
    subset: pl.DataFrame,
) -> pl.DataFrame:
    """Serializa crops S2 multitemporales SOLO de los pixeles de cada instancia.

    Para cada parcela en ``subset``, carga el ``S2_<patch_id>.npy`` y el
    ``TARGET_<patch_id>.npy`` correspondientes, calcula la mascara de la
    instancia y emite una fila long-format ``(parcel_id, t_index, band_NN)``
    con la media de los pixeles de esa instancia en esa banda y timestep.

    Promediar dentro de la instancia mantiene el parquet acotado
    (N parcelas * T * 10 bandas), suficiente para un eval de FarSLIP/RemoteCLIP
    en notebook.

    Args:
        pastis_root: Raiz PASTIS-R.
        subset: DataFrame con columnas ``parcel_id``, ``patch_id``,
            ``instance_id``.

    Returns:
        DataFrame con columnas ``parcel_id, t_index`` + ``band_B02..band_B12``
        (10 bandas, Float32). Vacio si ningun patch pudo leerse.
    """
    s2_cache: dict[str, np.ndarray] = {}
    target_cache: dict[str, np.ndarray] = {}

    rows: list[dict[str, Any]] = []
    for record in subset.iter_rows(named=True):
        pid = record["patch_id"]
        iid = int(record["instance_id"])
        parcel_id = record["parcel_id"]

        if pid not in s2_cache:
            s2_arr = _load_s2(pastis_root, pid)
            if s2_arr is None:
                continue
            s2_cache[pid] = s2_arr
        if pid not in target_cache:
            tgt = _load_target(pastis_root, pid)
            if tgt is None or tgt.ndim != 3 or tgt.shape[0] < 2:
                continue
            target_cache[pid] = tgt

        s2 = s2_cache[pid]
        instance = target_cache[pid][1]
        mask = instance == iid
        if not np.any(mask):
            continue

        T = s2.shape[0]
        for t in range(T):
            row: dict[str, Any] = {"parcel_id": parcel_id, "t_index": t}
            for b_idx, band_name in enumerate(PASTIS_S2_BANDS):
                vals = s2[t, b_idx][mask]
                row[f"band_{band_name}"] = float(vals.mean()) if vals.size else float("nan")
            rows.append(row)

    schema: dict[str, Any] = {"parcel_id": pl.Utf8, "t_index": pl.Int64}
    for band in PASTIS_S2_BANDS:
        schema[f"band_{band}"] = pl.Float32
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema)


def _md5_file(path: Path) -> str:
    """Devuelve el MD5 hex de un archivo.

    Args:
        path: Ruta al archivo.

    Returns:
        Hash MD5 en hex.
    """
    h = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------


def build_pastis_eval_subset(
    output_path: Path | str = _DEFAULT_OUTPUT,
    *,
    n_samples: int = 1024,
    seed: int = 42,
    pastis_root: Path | None = None,
    overwrite: bool = False,
    stratify_by: _StratifyBy = "class",
    save_imagery: bool = True,
) -> Path:
    """Construye un subset estratificado REAL de PASTIS-R para evaluacion FarSLIP/RemoteCLIP.

    NO genera datos sinteticos. Si PASTIS-R no esta en disco, lanza
    ``FileNotFoundError`` con instruccion de descarga.

    El subset se materializa como parquet con una fila por parcela
    ``(patch_id, instance_id)`` y las columnas:

    - ``parcel_id`` (Utf8, esquema ``{patch_id}_{instance_id}``)
    - ``patch_id`` (Int64)
    - ``instance_id`` (Int64)
    - ``class_id`` (Int64, 1..18)
    - ``class_name`` (Utf8, via ``PASTIS_R_CLASSES``)
    - ``tile`` (Utf8)
    - ``fold`` (Int64, 1..5)
    - ``lon`` / ``lat`` (Float64, EPSG:4326, centroide del patch)
    - ``n_pixels`` (Int64, size de la instancia)

    Si ``save_imagery=True``, ademas se materializa
    ``<output_path>.imagery.parquet`` con los crops S2 promediados por
    instancia (filas ``parcel_id, t_index, band_B02..band_B12``).

    Args:
        output_path: Ruta destino del parquet principal.
        n_samples: Numero objetivo de parcelas. Default 1024.
        seed: Semilla numpy para reproducibilidad. Default 42.
        pastis_root: Raiz del dataset. Default ``data/PASTIS-R/``.
        overwrite: Si False y el archivo ya existe, no regenera.
        stratify_by: Dimension de estratificacion (``class``, ``tile``, ``fold``).
        save_imagery: Si True, materializa el blob de imagery auxiliar.

    Returns:
        Ruta al parquet principal generado (o existente si ``overwrite=False``).

    Raises:
        FileNotFoundError: Si PASTIS-R no esta presente en disco.
        ValueError: Si tras enumerar instancias no hay parcelas validas.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and not overwrite:
        _log.info(
            "pastis_eval_subset.skip_existing",
            output=str(output),
            md5=_md5_file(output),
        )
        return output

    root = Path(pastis_root) if pastis_root is not None else _DEFAULT_PASTIS_ROOT
    _validate_pastis_present(root)

    index_df = pastis_patch_index(root / "metadata.geojson")
    if index_df.is_empty():
        raise ValueError(f"metadata.geojson de PASTIS-R sin features: {root}")

    coords_df = pastis_patch_coords(root / "metadata.geojson", target_crs="EPSG:4326")

    patch_ids = index_df["patch_id"].to_list()
    parcels_raw = _enumerate_parcels(root, patch_ids)
    if parcels_raw.is_empty():
        raise ValueError(
            f"No se encontraron instancias validas en {root / 'ANNOTATIONS'}."
        )

    # Enriquecer con tile, fold, lon, lat
    parcels_enriched = parcels_raw.join(
        index_df.rename({"TILE": "tile", "Fold": "fold"}),
        on="patch_id",
        how="left",
    )
    if not coords_df.is_empty():
        parcels_enriched = parcels_enriched.join(
            coords_df.select(["patch_id", "lon", "lat"]),
            on="patch_id",
            how="left",
        )
    else:
        parcels_enriched = parcels_enriched.with_columns(
            pl.lit(0.0).alias("lon"),
            pl.lit(0.0).alias("lat"),
        )

    sampled = _stratified_sample(
        parcels_enriched, n_samples=n_samples, stratify_by=stratify_by, seed=seed
    )

    # Construir parcel_id canonico Utf8 + class_name
    class_name_map = {int(k): v for k, v in PASTIS_R_CLASSES.items()}
    sampled = sampled.with_columns(
        (pl.col("patch_id").cast(pl.Utf8) + pl.lit("_") + pl.col("instance_id").cast(pl.Utf8))
        .alias("parcel_id"),
        pl.col("class_id")
        .cast(pl.Int64)
        .replace_strict(class_name_map, default="unknown")
        .alias("class_name"),
    )

    final = sampled.select(
        [
            pl.col("parcel_id").cast(pl.Utf8),
            pl.col("patch_id").cast(pl.Int64),
            pl.col("instance_id").cast(pl.Int64),
            pl.col("class_id").cast(pl.Int64),
            pl.col("class_name").cast(pl.Utf8),
            pl.col("tile").cast(pl.Utf8),
            pl.col("fold").cast(pl.Int64),
            pl.col("lon").cast(pl.Float64),
            pl.col("lat").cast(pl.Float64),
            pl.col("n_pixels").cast(pl.Int64),
        ]
    ).sort(["class_id", "patch_id", "instance_id"])

    final.write_parquet(output, compression="zstd")

    if save_imagery:
        imagery_path = output.with_suffix(output.suffix + ".imagery.parquet")
        imagery_df = _build_imagery_blob(root, final)
        imagery_df.write_parquet(imagery_path, compression="zstd")
        imagery_meta = {"path": str(imagery_path), "rows": imagery_df.height}
    else:
        imagery_meta = {"path": None, "rows": 0}

    class_counts: dict[int, int] = dict(Counter(final["class_id"].to_list()))
    _log.info(
        "pastis_eval_subset.built",
        output=str(output),
        n_parcels=final.height,
        n_parcels_per_class=class_counts,
        n_unique_tiles=final["tile"].n_unique(),
        n_unique_patches=final["patch_id"].n_unique(),
        md5=_md5_file(output),
        imagery=imagery_meta,
        stratify_by=stratify_by,
        seed=seed,
    )
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_cli() -> argparse.ArgumentParser:
    """Construye el ArgumentParser del CLI.

    Returns:
        Parser configurado.
    """
    parser = argparse.ArgumentParser(
        prog="python -m ml.ingest.pastis_eval_subset",
        description=(
            "Genera el subset REAL de PASTIS-R consumido por el notebook de "
            "evaluacion FarSLIP/RemoteCLIP (US-023). NO sintetico."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Ruta destino del parquet (default: data/test_fixtures/pastis_eval_subset.parquet).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=1024, help="Numero objetivo de parcelas."
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla numpy.")
    parser.add_argument(
        "--stratify-by",
        choices=("class", "tile", "fold"),
        default="class",
        help="Dimension de estratificacion.",
    )
    parser.add_argument(
        "--pastis-root",
        type=Path,
        default=None,
        help="Raiz PASTIS-R (default: data/PASTIS-R/).",
    )
    parser.add_argument(
        "--no-imagery",
        action="store_true",
        help="No materializar el blob auxiliar <output>.imagery.parquet.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobreescribir el parquet si ya existe.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entrypoint CLI.

    Args:
        argv: Argumentos opcionales (para tests). Si None, lee de sys.argv.

    Returns:
        0 si genero el subset correctamente; codigo de error en otro caso.
    """
    args = _build_cli().parse_args(argv)
    out = build_pastis_eval_subset(
        output_path=args.output,
        n_samples=args.n_samples,
        seed=args.seed,
        pastis_root=args.pastis_root,
        overwrite=args.overwrite,
        stratify_by=args.stratify_by,
        save_imagery=not args.no_imagery,
    )
    print(json.dumps({"output": str(out), "md5": _md5_file(out)}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_pastis_eval_subset", "main"]
