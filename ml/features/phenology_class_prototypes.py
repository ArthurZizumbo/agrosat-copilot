"""Prototipos fenologicos por clase para la rama semantica de TSViT.

Implementa el insumo del metodo de Wen et al. (2025), "Phenology
Description is All You Need!" (ISPRS J. Photogrammetry RS 228): en lugar de
una descripcion por parcela, se construye **una curva NDVI media por clase**
de cultivo y se genera con un LLM (Gemini 3.5 Flash) la descripcion
fenologica textual de cada clase. Estas 18 descripciones cubren el 100% de
las clases (a diferencia del subset por-parcela 60x18 que cubre ~0.72% de
los pixeles densos), por lo que son el insumo correcto para alinear por
contraste las features visuales de la segmentacion densa con el prototipo
semantico de la clase de cada pixel (paper Fig. 1, Tabla 2).

Flujo:
    1. ``compute_class_mean_ndvi_curves``: barre los patches PASTIS-R
       ``DATA_S2/S2_*.npy``, calcula NDVI por pixel, lo agrupa por la clase
       semantica del pixel (``ANNOTATIONS/TARGET_*.npy`` canal 0) y promedia
       sobre una rejilla temporal regular indexada por DOY (las fechas de
       adquisicion son irregulares por patch, de ``metadata.geojson``).
    2. ``generate_class_prototypes``: por cada una de las 18 clases, llama a
       :func:`ml.features.phenology_description.generate_phenology_description`
       con la curva media y el nombre de clase como ``crop_type_hint``, luego
       codifica el texto a un embedding con ``all-MiniLM-L6-v2`` (384-dim,
       el mismo encoder que el pheno_text por-parcela existente).

El output es ``data/features/phenology_class_prototypes_pastis.parquet`` con
18 filas ``class_id, class_name, ndvi_curve, description, emb_000..emb_383``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import structlog

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PASTIS_ROOT = _REPO_ROOT / "data" / "PASTIS-R"
_CLASS_MAP_PATH = _REPO_ROOT / "data" / "reference" / "pastis_class_mapping.json"
_DEFAULT_OUTPUT = (
    _REPO_ROOT / "data" / "features" / "phenology_class_prototypes_pastis.parquet"
)

#: Band indices in the PASTIS-R .npy files (standard 10-band S2 order:
#: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12). NDVI uses B4 (red) and B8 (NIR).
_BAND_B4 = 2
_BAND_B8 = 6

#: Number of regular temporal bins (DOY 1..365) over which the curve is
#: averaged. 37 matches the 10-day grid of the Wen paper.
_N_TIME_BINS = 37

#: Useful classes: 1..18 (excludes 0 Background and 19 Void). The prototype is
#: generated only for the 18 benchmark crops.
_CROP_CLASS_IDS: tuple[int, ...] = tuple(range(1, 19))

#: Text encoder -> 384-dim embedding. Same model as the existing per-parcel
#: pheno_text, to keep coherence of the semantic space.
_SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_EMB_DIM = 384


def load_class_names(path: Path = _CLASS_MAP_PATH) -> dict[int, str]:
    """Carga el mapa ``class_id -> nombre`` de las 18 clases PASTIS.

    Args:
        path: Ruta al ``pastis_class_mapping.json``.

    Returns:
        Diccionario ``{1: "Meadow", 2: "Soft winter wheat", ...}``.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    classes = data["classes"]
    out: dict[int, str] = {}
    for k, v in classes.items():
        name = v["name"] if isinstance(v, dict) else v
        out[int(k)] = name
    return out


def _patch_dates_doy(metadata_path: Path) -> dict[int, np.ndarray]:
    """Devuelve ``{patch_id: array de DOY (T,)}`` desde ``metadata.geojson``.

    Las fechas vienen como enteros ``YYYYMMDD`` en el campo ``dates-S2``
    (dict indexado por timestep). Se convierten a dia-del-anio (1..366).

    Args:
        metadata_path: Ruta a ``metadata.geojson``.

    Returns:
        Mapa de patch_id a vector de DOY alineado con el eje temporal del
        ``.npy`` correspondiente.

    Nota:
        Se parsea como JSON plano (``json.load``), NO con
        ``geopandas.read_file``: solo se necesitan las fechas
        (``properties.dates-S2``), no las geometrias Polygon. Cargar las
        2433 geometrias con geopandas es ~100x mas lento y puede colgar el
        proceso; el JSON crudo se lee en ~0.1s.
    """
    geojson = json.loads(metadata_path.read_text(encoding="utf-8"))
    out: dict[int, np.ndarray] = {}
    for feat in geojson.get("features", []):
        props = feat.get("properties", {})
        pid = int(props["ID_PATCH"])
        dates_raw = props["dates-S2"]
        if isinstance(dates_raw, str):
            dates_raw = json.loads(dates_raw)
        # Order by timestep index (keys "0".."T-1").
        ymd = [int(dates_raw[str(i)]) for i in range(len(dates_raw))]
        doy = np.array([_ymd_to_doy(v) for v in ymd], dtype=np.int32)
        out[pid] = doy
    return out


def _ymd_to_doy(ymd: int) -> int:
    """Convierte un entero ``YYYYMMDD`` a dia-del-anio (1..366)."""
    from datetime import date

    year = ymd // 10000
    month = (ymd % 10000) // 100
    day = ymd % 100
    return date(year, month, day).timetuple().tm_yday


def compute_class_mean_ndvi_curves(
    pastis_root: Path = _DEFAULT_PASTIS_ROOT,
    *,
    n_time_bins: int = _N_TIME_BINS,
    max_patches: int | None = None,
) -> dict[int, np.ndarray]:
    """Calcula la curva NDVI media por clase sobre una rejilla DOY regular.

    Para cada patch carga ``S2_<pid>.npy`` ``(T,10,H,W)`` y la mascara
    semantica ``TARGET_<pid>.npy`` canal 0 ``(H,W)``. Calcula NDVI por
    pixel-tiempo, acumula la suma y el conteo por clase en cada bin DOY, y al
    final divide para obtener la media. Las reflectancias int16 se escalan a
    [0,1] dividiendo por 10000 (escala S2 L2A).

    Args:
        pastis_root: Raiz del dataset PASTIS-R.
        n_time_bins: Numero de bins DOY regulares (1..365).
        max_patches: Si se indica, limita el barrido (para smoke/tests).

    Returns:
        ``{class_id: curva (n_time_bins,)}`` con NaN en bins sin observacion.
    """
    s2_dir = pastis_root / "DATA_S2"
    ann_dir = pastis_root / "ANNOTATIONS"
    dates_by_patch = _patch_dates_doy(pastis_root / "metadata.geojson")

    bin_edges = np.linspace(1, 366, n_time_bins + 1)
    # Per-class accumulators: sum and count in each temporal bin.
    sums = {c: np.zeros(n_time_bins, dtype=np.float64) for c in _CROP_CLASS_IDS}
    counts = {c: np.zeros(n_time_bins, dtype=np.int64) for c in _CROP_CLASS_IDS}

    s2_paths = sorted(s2_dir.glob("S2_*.npy"))
    if max_patches is not None:
        s2_paths = s2_paths[:max_patches]

    for s2_path in s2_paths:
        pid = int(s2_path.stem.split("_")[1])
        doy = dates_by_patch.get(pid)
        if doy is None:
            continue
        s2 = np.load(s2_path).astype(np.float32) / 10000.0  # (T,10,H,W)
        target = np.load(ann_dir / f"TARGET_{pid}.npy")[0]  # (H,W) semantic
        b4 = s2[:, _BAND_B4]  # (T,H,W)
        b8 = s2[:, _BAND_B8]
        denom = b8 + b4
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = np.where(denom > 1e-6, (b8 - b4) / denom, np.nan)  # (T,H,W)
        # NDVI valid in [-1, 1]; out-of-range values are artifacts of
        # clouds/shadows or a near-zero denominator (not masked in PASTIS).
        ndvi = np.where(np.abs(ndvi) <= 1.0, ndvi, np.nan)
        bin_idx = np.clip(np.digitize(doy, bin_edges) - 1, 0, n_time_bins - 1)
        for c in _CROP_CLASS_IDS:
            class_mask = target == c  # (H,W)
            if not class_mask.any():
                continue
            # Mean NDVI of the class at each timestep -> (T,)
            ndvi_class = ndvi[:, class_mask]  # (T, n_pix_class)
            per_t = np.nanmean(ndvi_class, axis=1)  # (T,)
            valid = np.isfinite(per_t)
            np.add.at(sums[c], bin_idx[valid], per_t[valid])
            np.add.at(counts[c], bin_idx[valid], 1)

    curves: dict[int, np.ndarray] = {}
    for c in _CROP_CLASS_IDS:
        with np.errstate(divide="ignore", invalid="ignore"):
            curve = np.where(counts[c] > 0, sums[c] / counts[c], np.nan)
        curves[c] = curve
    logger.info(
        "class_mean_ndvi_curves_computed",
        n_patches=len(s2_paths),
        n_classes=len(curves),
        n_time_bins=n_time_bins,
    )
    return curves


def _encode_descriptions(descriptions: Sequence[str]) -> np.ndarray:
    """Codifica una lista de descripciones a embeddings 384-dim L2-norm.

    Usa ``all-MiniLM-L6-v2`` (mismo encoder que el pheno_text por-parcela).

    Args:
        descriptions: Lista de textos.

    Returns:
        Matriz ``(len, 384)`` float32, L2-normalizada por fila.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(_SENTENCE_MODEL)
    emb = model.encode(
        list(descriptions),
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return emb.astype(np.float32)


def generate_class_prototypes(
    pastis_root: Path = _DEFAULT_PASTIS_ROOT,
    *,
    output_path: Path = _DEFAULT_OUTPUT,
    model: str = "gemini-3.5-flash",
    n_time_bins: int = _N_TIME_BINS,
    max_patches: int | None = None,
) -> Path:
    """Genera los 18 prototipos fenologicos por clase y los persiste.

    Pipeline completo: curva NDVI media por clase -> descripcion Gemini por
    clase (prompt 3-bloques Wen et al. Fig. 2, con el nombre de clase como
    ``crop_type_hint``) -> embedding 384-dim. Cobertura 100% de las 18
    clases.

    Args:
        pastis_root: Raiz PASTIS-R.
        output_path: Parquet de salida (18 filas).
        model: Modelo LLM para las descripciones.
        n_time_bins: Bins DOY de la curva media.
        max_patches: Limita el barrido NDVI (smoke/tests).

    Returns:
        ``Path`` del parquet escrito con columnas ``class_id, class_name,
        ndvi_curve (list), description, emb_000..emb_383``.
    """
    from ml.features.phenology_description import (
        generate_phenology_description,
    )

    class_names = load_class_names()
    curves = compute_class_mean_ndvi_curves(
        pastis_root, n_time_bins=n_time_bins, max_patches=max_patches
    )
    # Representative DOY of each bin (center), to pass to the generator.
    bin_edges = np.linspace(1, 366, n_time_bins + 1)
    bin_doy = ((bin_edges[:-1] + bin_edges[1:]) / 2).astype(np.int32)

    rows: list[dict[str, object]] = []
    descriptions: list[str] = []
    for c in _CROP_CLASS_IDS:
        curve = curves[c]
        name = class_names.get(c, f"class_{c}")
        desc = generate_phenology_description(
            ndvi_curve=curve,
            doy=bin_doy,
            parcel_id=f"class_{c}",
            crop_type_hint=name,
            model=model,
        )
        descriptions.append(desc)
        rows.append(
            {
                "class_id": c,
                "class_name": name,
                "ndvi_curve": curve.tolist(),
                "description": desc,
            }
        )
        logger.info("class_prototype_generated", class_id=c, class_name=name)

    embeddings = _encode_descriptions(descriptions)
    for i, row in enumerate(rows):
        for j in range(_EMB_DIM):
            row[f"emb_{j:03d}"] = float(embeddings[i, j])

    df = pl.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    logger.info(
        "class_prototypes_persisted",
        path=str(output_path),
        n_classes=df.height,
        emb_dim=_EMB_DIM,
    )
    return output_path


def load_class_prototype_embeddings(
    path: Path = _DEFAULT_OUTPUT,
) -> tuple[np.ndarray, list[int]]:
    """Carga la matriz de prototipos ``(18, 384)`` y sus class_ids.

    Helper para el entrenamiento de TSViT: el modelo indexa esta matriz por
    la clase de cada pixel para obtener el prototipo semantico objetivo de la
    alineacion contrastiva.

    Args:
        path: Parquet de prototipos.

    Returns:
        ``(prototypes (18,384) float32, class_ids ordenados)``.
    """
    df = pl.read_parquet(path)
    emb_cols = [f"emb_{j:03d}" for j in range(_EMB_DIM)]
    prototypes = df.select(emb_cols).to_numpy().astype(np.float32)
    class_ids = df["class_id"].to_list()
    return prototypes, class_ids


def _build_arg_parser():  # pragma: no cover - CLI thin wrapper.
    import argparse

    p = argparse.ArgumentParser(
        description="Genera los 18 prototipos fenologicos por clase (Wen 2025)."
    )
    p.add_argument("--pastis-root", type=Path, default=_DEFAULT_PASTIS_ROOT)
    p.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    p.add_argument("--model", default="gemini-3.5-flash")
    p.add_argument("--n-time-bins", type=int, default=_N_TIME_BINS)
    p.add_argument("--max-patches", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    args = _build_arg_parser().parse_args(argv)
    out = generate_class_prototypes(
        args.pastis_root,
        output_path=args.output,
        model=args.model,
        n_time_bins=args.n_time_bins,
        max_patches=args.max_patches,
    )
    logger.info("done", output=str(out))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
