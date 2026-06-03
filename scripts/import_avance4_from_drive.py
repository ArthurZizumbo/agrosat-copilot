"""Integra los artefactos de los modelos de segmentacion descargados del Drive
del equipo (unet, anysat, segformer, utae) al repo, en el formato que consume
``Avance4.Equipo17.ipynb``.

Fuente: una carpeta local con la estructura del Drive compartido (``reports/
segmentation/{metrics,figures}`` para unet/anysat, ``outputs/<modelo>/`` con
``results.json`` + figuras para segformer/utae).

Acciones:

- Copia tal cual los parquets ``model_comparison_avance4_{unet,anysat_fast}.parquet``
  y sus figuras (ya tienen el formato/nombres que espera el integrador).
- Convierte el ``results.json`` de segformer y utae a un parquet
  ``model_comparison_avance4_{modelo}.parquet`` con el esquema del integrador, y
  copia sus figuras renombrandolas al patron ``{key}_{modelo}.png``
  (``training_curves`` -> ``curves``, ``qualitative`` -> ``samples``).

El ``miou_grouped`` de segformer/utae se promedia de su ``hcat_group_iou`` (6
grupos HCAT), coherente con la columna *_grouped de la tabla.

Operativo permanente y parametrizable por ``--src``. No reentrena nada.

Uso::

    poetry run python scripts/import_avance4_from_drive.py \\
        --src "C:/Users/arthu/Downloads/avance_drive_pi"
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import polars as pl

# Integrator schema (mirror of Aaron's run_training).
_SCHEMA = {
    "model": pl.Utf8,
    "miou": pl.Float64, "f1_macro": pl.Float64, "pixel_accuracy": pl.Float64,
    "miou_grouped": pl.Float64, "f1_macro_grouped": pl.Float64,
    "pixel_accuracy_grouped": pl.Float64,
    "train_time_s": pl.Float64, "epochs": pl.Int64,
    "n_train": pl.Int64, "n_val": pl.Int64, "n_trainable_params": pl.Int64,
    "target_size": pl.Int64, "device": pl.Utf8,
}

#: Models with parquet+figures already in integrator format (just copy).
_READY = {
    "model_comparison_avance4_unet.parquet": "model_comparison_avance4_unet.parquet",
    "model_comparison_avance4_anysat_fast.parquet": "model_comparison_avance4_anysat_fast.parquet",
}

#: Models with only results.json in outputs/<dir>/ (to convert).
_FROM_RESULTS = {
    "segformer": "segformer_b0_pastis",
    "utae": "utae_pastis",
}

#: Mapping of the output figure name to the integrator pattern.
_FIG_MAP = {
    "training_curves.png": "curves",
    "per_class_iou.png": "per_class_iou",
    "qualitative.png": "samples",
}


def _row_from_results(model: str, res: dict) -> dict:
    """Construye una fila del integrador desde un results.json de segformer/utae.

    Args:
        model: Nombre canonico del modelo (``segformer`` / ``utae``).
        res: Contenido del ``results.json``.

    Returns:
        Dict con el esquema del integrador. ``miou_grouped`` se promedia del
        ``hcat_group_iou`` (mIoU sobre los 6 grupos HCAT).
    """
    grouped = res.get("hcat_group_iou") or {}
    miou_grouped = (
        round(sum(grouped.values()) / len(grouped), 4) if grouped else None
    )
    return {
        "model": model,
        "miou": res.get("test_miou"),
        "f1_macro": res.get("test_f1_macro"),
        "pixel_accuracy": res.get("test_pixel_accuracy"),
        "miou_grouped": miou_grouped,
        "f1_macro_grouped": None,
        "pixel_accuracy_grouped": None,
        "train_time_s": None,
        "epochs": res.get("epochs"),
        "n_train": None, "n_val": None, "n_trainable_params": None,
        "target_size": None, "device": None,
    }


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Carpeta local con la estructura del Drive.")
    parser.add_argument(
        "--dest", default=".", help="Raiz del repo (default: CWD)."
    )
    args = parser.parse_args(argv)

    src = Path(args.src)
    dest = Path(args.dest)
    src_seg = src / "reports" / "segmentation"
    src_metrics = src_seg / "metrics"
    src_figures = src_seg / "figures"

    dst_metrics = dest / "reports" / "segmentation" / "metrics"
    dst_figures = dest / "reports" / "segmentation" / "figures"
    dst_metrics.mkdir(parents=True, exist_ok=True)
    dst_figures.mkdir(parents=True, exist_ok=True)

    # 1. Parquets already ready (unet, anysat_fast).
    for src_name, dst_name in _READY.items():
        p = src_metrics / src_name
        if p.is_file():
            shutil.copy2(p, dst_metrics / dst_name)
            print(f"copiado parquet {dst_name}")

    # 2. Figures already with the correct name (unet, anysat_fast).
    if src_figures.is_dir():
        for fig in src_figures.glob("*.png"):
            shutil.copy2(fig, dst_figures / fig.name)
            print(f"copiada figura {fig.name}")

    # 3. segformer / utae: convert results.json -> parquet + renamed figures.
    for model, out_dir in _FROM_RESULTS.items():
        odir = src / "outputs" / out_dir
        rj = odir / "results.json"
        if not rj.is_file():
            print(f"AVISO: sin results.json para {model} ({rj}); se omite.")
            continue
        res = json.loads(rj.read_text(encoding="utf-8"))
        row = _row_from_results(model, res)
        df = pl.DataFrame([row], schema=_SCHEMA)
        out_parquet = dst_metrics / f"model_comparison_avance4_{model}.parquet"
        df.write_parquet(out_parquet)
        print(f"convertido {model}: results.json -> {out_parquet.name} (miou={row['miou']})")
        # Figures: rename to the integrator pattern.
        for orig, key in _FIG_MAP.items():
            fp = odir / orig
            if fp.is_file():
                shutil.copy2(fp, dst_figures / f"{key}_{model}.png")
                print(f"  figura {orig} -> {key}_{model}.png")

    print("\nIntegracion del Drive completa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
