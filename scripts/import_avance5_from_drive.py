"""Integra los artefactos del Avance 5 (modelo final mejorado) descargados del
Drive del equipo al repo, siguiendo el estandar de
``scripts/import_avance4_from_drive.py`` (Arthur):

- Artefactos PEQUENOS (parquets de metricas + figuras PNG) -> ``reports/best_model/``,
  forzados a GIT para que el entregable sea reproducible desde un clon limpio.
- CHECKPOINT pesado (.pt) -> ``checkpoints/best_model/<run>/`` para versionar con
  DVC, mismo patron que ``checkpoints/segmentation.dvc``: un directorio trackeado
  por un unico ``.dvc`` (no archivos sueltos).

Fuente (``--src``): una carpeta local con la estructura del Drive compartido, es
decir ``reports/best_model/{metrics,figures,checkpoints}`` tal como las escribe
``notebooks/best_model/Avance5.Equipo17.ipynb`` en Drive.

NO corre ``dvc add`` ni ``git`` por si mismo: copia a las carpetas estandar e
imprime los comandos exactos para que el equipo los ejecute (commits/push/dvc los
maneja el equipo). No reentrena nada.

Uso::

    poetry run python scripts/import_avance5_from_drive.py \\
        --src "G:/Mi unidad/Integrador"
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

#: Run/checkpoint por defecto que produce el notebook del Avance 5.
_RUN_NAME = "alt-tsvit-pheno-cw-aug-v1"


def _copy_glob(src_dir: Path, dst_dir: Path, pattern: str) -> int:
    """Copia ``src_dir/<pattern>`` a ``dst_dir`` (crea el destino).

    Args:
        src_dir: Carpeta origen.
        dst_dir: Carpeta destino (se crea si no existe).
        pattern: Glob de archivos a copiar (p.ej. ``"*.parquet"``).

    Returns:
        Numero de archivos copiados.
    """
    if not src_dir.is_dir():
        print(f"AVISO: no existe {src_dir}; se omite.")
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src_dir.glob(pattern)):
        if f.is_file():
            shutil.copy2(f, dst_dir / f.name)
            print(f"  copiado {f.name}")
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        required=True,
        help="Carpeta local con la estructura del Drive (contiene reports/best_model/).",
    )
    parser.add_argument("--dest", default=".", help="Raiz del repo (default: CWD).")
    parser.add_argument(
        "--run-name",
        default=_RUN_NAME,
        help=f"Run/checkpoint a importar (default: {_RUN_NAME}).",
    )
    args = parser.parse_args(argv)

    src = Path(args.src)
    dest = Path(args.dest)
    src_stage = src / "reports" / "best_model"

    # 1. Artefactos pequenos -> git en reports/best_model/.
    print("metricas (parquet) -> reports/best_model/metrics/")
    n_metrics = _copy_glob(
        src_stage / "metrics", dest / "reports" / "best_model" / "metrics", "*.parquet"
    )
    print("figuras (png) -> reports/best_model/figures/")
    n_figs = _copy_glob(
        src_stage / "figures", dest / "reports" / "best_model" / "figures", "*.png"
    )

    # 2. Checkpoint pesado -> checkpoints/best_model/<run>/ para DVC.
    src_ckpt = src_stage / "checkpoints" / args.run_name
    dst_ckpt = dest / "checkpoints" / "best_model" / args.run_name
    print(f"checkpoint -> checkpoints/best_model/{args.run_name}/")
    n_ckpt = _copy_glob(src_ckpt, dst_ckpt, "*.pt")
    if n_ckpt == 0:
        print("  (sin checkpoint: corre el entrenamiento con RUN_TRAINING=True primero)")

    print(f"\nResumen: {n_metrics} parquet, {n_figs} figuras, {n_ckpt} checkpoints.")

    # 3. Runbook: comandos para versionar (los ejecuta el equipo, no este script).
    print("\nSiguientes pasos (ejecutar a mano):")
    print("  # Checkpoint pesado -> DVC (mismo estandar que checkpoints/segmentation):")
    print("  dvc add checkpoints/best_model   # autostage=true ya agrega el .dvc a git")
    print("  dvc push                         # sube a gs://agrosat-dvc-remote")
    print("  # Parquets + figuras -> git (reports/ esta en .gitignore, por eso -f):")
    print("  git add -f reports/best_model/metrics reports/best_model/figures")
    print('  git commit -m "chore(E5): import avance5 best_model (parquets+figuras git, ckpt dvc)"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
