"""Exporta los parquets comparativos de los modelos del equipo us-025 (DeepLabv3+
y TSViT) al formato que consume el integrador ``Avance4.Equipo17.ipynb``.

El notebook de Aaron consolida ``reports/segmentation/metrics/
model_comparison_avance4_<modelo>.parquet`` de cada integrante (celda de
consolidacion: ``glob('model_comparison_avance4_*.parquet')`` + ``concat`` +
``unique(subset=['model'])``). Este script genera los nuestros con el MISMO
esquema que escribe :func:`ml.train.train_segmentation.run_training` (Aaron), a
partir de las metricas reales medidas sobre el fold de validacion (fold 4, 482
parches) en ``5a_deeplabv3plus.ipynb`` y ``5b_tsvit.ipynb``.

Asi nuestros 2 modelos aparecen como "disponibles" en la consolidacion (no
"pendientes"), sin hardcodear filas en el notebook (data-driven, como el resto
del equipo). ``tsvit`` y ``tsvit-pheno`` van como dos filas en el mismo parquet
``_tsvit.parquet`` (la consolidacion las distingue por la columna ``model``).

Operativo permanente (reproducible), no un script de smoke/debug.

Uso::

    poetry run python scripts/export_avance4_metrics_us025.py
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

# Mirror schema of run_training (Aaron): columns consumed by the consolidation.
# The *_grouped metrics only apply to deeplabv3plus (HCAT-6 variant); for the
# temporal ones they stay None (the table convention leaves them empty).
#
# ``train_time_s`` is the real wall-clock of the training run measured on the
# local MLflow server (Docker Postgres :5010, experiment 7
# ``agrosat-segmentation``): ``end_time - start_time`` of the FINISHED run.
#   - deeplabv3plus (mobilenet, 18 classes, run 1c1e4f6f): 44676.4 s (~12.4 h).
#   - tsvit          (run 24f70756, batch 16, 30 epochs):  1894.4 s (~31.6 min, RTX 4070).
#   - tsvit-pheno    (run 0eef8a60, batch 16, 30 epochs):  1915.4 s (~31.9 min, RTX 4070).
_ROWS = [
    {
        "model": "deeplabv3plus",
        "miou": 0.2709, "f1_macro": 0.3864, "pixel_accuracy": 0.6743,
        "miou_grouped": 0.4682, "f1_macro_grouped": 0.6009,
        "pixel_accuracy_grouped": 0.8018,
        "train_time_s": 44676.4, "epochs": 15,
        "n_train": None, "n_val": 482, "n_trainable_params": None,
        "target_size": 128, "device": "cuda",
    },
    {
        "model": "tsvit",
        "miou": 0.6215, "f1_macro": 0.7473, "pixel_accuracy": 0.8724,
        "miou_grouped": None, "f1_macro_grouped": None,
        "pixel_accuracy_grouped": None,
        "train_time_s": 1894.4, "epochs": 30,
        "n_train": None, "n_val": 482, "n_trainable_params": None,
        "target_size": 128, "device": "cuda",
    },
    {
        "model": "tsvit-pheno",
        "miou": 0.6253, "f1_macro": 0.7500, "pixel_accuracy": 0.8759,
        "miou_grouped": None, "f1_macro_grouped": None,
        "pixel_accuracy_grouped": None,
        "train_time_s": 1915.4, "epochs": 30,
        "n_train": None, "n_val": 482, "n_trainable_params": None,
        "target_size": 128, "device": "cuda",
    },
]

# Explicit schema: prevents Polars from inferring Null on columns with only None
# (would break the vertical_relaxed of the consolidation against parquets with floats).
_SCHEMA = {
    "model": pl.Utf8,
    "miou": pl.Float64, "f1_macro": pl.Float64, "pixel_accuracy": pl.Float64,
    "miou_grouped": pl.Float64, "f1_macro_grouped": pl.Float64,
    "pixel_accuracy_grouped": pl.Float64,
    "train_time_s": pl.Float64, "epochs": pl.Int64,
    "n_train": pl.Int64, "n_val": pl.Int64, "n_trainable_params": pl.Int64,
    "target_size": pl.Int64, "device": pl.Utf8,
}


def main() -> int:
    """Escribe los parquets de deeplabv3plus y tsvit (incluye tsvit-pheno)."""
    out_dir = Path("reports/segmentation/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pl.DataFrame(_ROWS, schema=_SCHEMA)

    # deeplabv3plus -> its parquet; tsvit + tsvit-pheno -> the tsvit parquet
    # (the consolidation separates them by the `model` column).
    deeplab = df.filter(pl.col("model") == "deeplabv3plus")
    tsvit = df.filter(pl.col("model").is_in(["tsvit", "tsvit-pheno"]))

    p_deeplab = out_dir / "model_comparison_avance4_deeplabv3plus.parquet"
    p_tsvit = out_dir / "model_comparison_avance4_tsvit.parquet"
    deeplab.write_parquet(p_deeplab)
    tsvit.write_parquet(p_tsvit)

    print(f"escrito {p_deeplab} ({deeplab.height} fila)")
    print(f"escrito {p_tsvit} ({tsvit.height} filas: tsvit + tsvit-pheno)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
