"""Builder de los notebooks de segmentacion densa (Avance 4), uno por modelo.

Genera un notebook independiente por arquitectura para poder correrlos en paralelo
en sesiones de Colab separadas:

- ``04d_segmentation_unet.ipynb``   -> U-Net ResNet-50
- ``04e_segmentation_anysat.ipynb`` -> AnySat congelado + cabeza lineal

Cada notebook es Colab-first (monta Drive, clona el repo, lee el dataset, entrena
con reanudacion por checkpoint) y deja sus artefactos en carpetas claras del Drive
compartido, para citarlos luego en el reporte:

    reports/segmentation/metrics/      parquet de metricas por modelo
    reports/segmentation/figures/      PNG de la matriz de confusion
    reports/segmentation/checkpoints/  modelo final + checkpoint reanudable

Uso::

    poetry run python scripts/build_segmentation_notebook.py --model unet
    poetry run python scripts/build_segmentation_notebook.py --model anysat

Operativo permanente (NO viola el anti-patron ``scripts/_*.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import nbformat as nbf
import typer

app = typer.Typer(add_completion=False, help=__doc__)

_OUT_BY_MODEL = {
    "unet": Path("notebooks/segmentation/04d_segmentation_unet.ipynb"),
    "anysat": Path("notebooks/segmentation/04e_segmentation_anysat.ipynb"),
}

# Celda de setup (identica en ambos notebooks): monta Drive, clona el repo y
# localiza el pyproject. Ajustar _branch si el codigo vive en otra rama.
_SETUP_CELL = (
    "# Setup del entorno. En Colab se monta Drive (donde vive el dataset) y se\n"
    "# instalan las dependencias que no vienen por defecto; en local no hace falta.\n"
    "import os, sys, subprocess\n"
    "from pathlib import Path\n\n"
    "_IN_COLAB = False\n"
    "shared_folder_path = ''\n"
    "try:\n"
    "    from google.colab import drive\n"
    "    drive.mount('/content/drive')\n"
    "    shared_folder_path = '/content/drive/MyDrive/Integrador/'\n"
    "    _IN_COLAB = True\n"
    "except ImportError:\n"
    "    pass\n\n"
    "# En Colab el repo no esta presente: se clona una vez en /content/agrosat-copilot.\n"
    "if _IN_COLAB:\n"
    "    from getpass import getpass\n"
    "    _repo_dir = '/content/agrosat-copilot'\n"
    "    _branch = 'users/abocanegra/unet-anysat'\n"
    "    _repo = 'github.com/ArthurZizumbo/agrosat-copilot.git'\n"
    "    if not Path(_repo_dir, 'pyproject.toml').is_file():\n"
    "        _rc = os.system(f'git clone --branch {_branch} --depth 1 "
    "https://{_repo} {_repo_dir}')\n"
    "        if _rc != 0:  # repo privado: pide token (no se guarda en el notebook)\n"
    "            _tok = getpass('GitHub token (repo privado): ')\n"
    "            os.system(f'git clone --branch {_branch} --depth 1 "
    "https://{_tok}@{_repo} {_repo_dir}')\n\n"
    "# El codigo no vive en Drive: se localiza el repo por su pyproject.toml.\n"
    "_search = [Path.cwd().resolve(), *Path.cwd().resolve().parents]\n"
    "if _IN_COLAB:\n"
    "    _search = [Path('/content/agrosat-copilot'), *_search]\n"
    "for _cand in _search:\n"
    "    if (_cand / 'pyproject.toml').is_file():\n"
    "        if str(_cand) not in sys.path:\n"
    "            sys.path.insert(0, str(_cand))\n"
    "        os.chdir(_cand)\n"
    "        break\n"
    "else:\n"
    "    raise RuntimeError('No se encontro el repo agrosat-copilot (pyproject.toml). '\n"
    "                       'Clonalo en /content/agrosat-copilot o sincronizalo desde VS Code.')\n\n"
    "if _IN_COLAB:\n"
    "    subprocess.run([sys.executable, '-m', 'pip', '-q', 'install',\n"
    "                    'segmentation-models-pytorch', 'structlog', 'typer', 'polars', "
    "'mlflow'], check=False)\n\n"
    "print('repo:', Path.cwd(), '| colab:', _IN_COLAB, '| drive:', shared_folder_path or '(local)')"
)

_COPY_CELL = (
    "# Copia del dataset de Drive al disco local, con barra de progreso.\n"
    "import shutil, time\n\n"
    "def copy_pastis_to_local(src_root, dst_root,\n"
    "                         subdirs=('DATA_S2', 'ANNOTATIONS'),\n"
    "                         files=('metadata.geojson', 'NORM_S2_patch.json')):\n"
    "    src_root, dst_root = Path(src_root), Path(dst_root)\n"
    "    dst_root.mkdir(parents=True, exist_ok=True)\n"
    "    todo = []\n"
    "    for sub in subdirs:\n"
    "        for f in sorted((src_root / sub).glob('*')):\n"
    "            if f.is_file():\n"
    "                todo.append((f, dst_root / sub / f.name))\n"
    "    for fname in files:\n"
    "        sp = src_root / fname\n"
    "        if sp.is_file():\n"
    "            todo.append((sp, dst_root / fname))\n"
    "    if not todo:\n"
    "        raise FileNotFoundError(f'No se hallaron DATA_S2/ANNOTATIONS en {src_root}')\n"
    "    total_bytes = sum(s.stat().st_size for s, _ in todo)\n"
    "    try:\n"
    "        from tqdm.auto import tqdm\n"
    "        bar = tqdm(total=total_bytes, unit='B', unit_scale=True, desc='Copiando PASTIS')\n"
    "    except Exception:\n"
    "        bar = None\n"
    "    t0 = time.time()\n"
    "    for i, (src, dst) in enumerate(todo, 1):\n"
    "        dst.parent.mkdir(parents=True, exist_ok=True)\n"
    "        # Salta el archivo si ya esta copiado con el mismo tamano.\n"
    "        if not (dst.exists() and dst.stat().st_size == src.stat().st_size):\n"
    "            shutil.copy2(src, dst)\n"
    "        if bar is not None:\n"
    "            bar.update(src.stat().st_size)\n"
    "        elif i % 200 == 0:\n"
    "            print(f'  {i}/{len(todo)} archivos...')\n"
    "    if bar is not None:\n"
    "        bar.close()\n"
    "    print(f'Listo: {len(todo)} archivos ({total_bytes / 1e9:.1f} GB) en "
    "{time.time() - t0:.0f}s -> {dst_root}')\n"
    "    return dst_root\n\n"
    "if _IN_COLAB and COPY_TO_LOCAL:\n"
    "    PASTIS_ROOT = copy_pastis_to_local(PASTIS_ROOT, '/content/PASTIS-R')\n"
    "    print('PASTIS_ROOT (local):', PASTIS_ROOT, '| exists:', PASTIS_ROOT.exists())\n"
    "else:\n"
    "    print('Lectura directa desde:', PASTIS_ROOT, '| exists:', PASTIS_ROOT.exists())"
)

_SPLIT_CELL = (
    "# Split en los folds oficiales de PASTIS (espacialmente disjuntos).\n"
    "from ml.ingest.pastis_dataset import pastis_fold_split\n\n"
    "split = pastis_fold_split(PASTIS_ROOT, train_folds=(1, 2, 3), val_folds=(4,), "
    "test_folds=(5,))\n"
    "print({k: len(v) for k, v in split.items()})"
)

_META = {
    "unet": {
        "title": "# Segmentación de cultivos con U-Net (ResNet-50) sobre PASTIS-R",
        "intro": (
            "Se entrena una U-Net de segmentación densa sobre las series Sentinel-2 de PASTIS-R "
            "y se evalúa su desempeño píxel a píxel. El encoder ResNet-50 viene preentrenado en "
            "ImageNet y se adapta a las diez bandas; como entrada se usa la mediana temporal de la "
            "serie y la salida es un mapa de clases a la resolución de la imagen.\n\n"
            "Este cuaderno corre de forma independiente (en paralelo con el de AnySat) y deja sus "
            "artefactos en carpetas del Drive compartido para el reporte: la tabla de métricas, la "
            "figura de la matriz de confusión y el modelo entrenado."
        ),
        "model_md": (
            "## Entrenamiento\n\n"
            "El entrenamiento guarda un checkpoint por época en Drive; si la sesión se reinicia, al "
            "volver a ejecutar esta celda se reanuda desde la última época completada en vez de "
            "empezar de cero."
        ),
        "batch": "16",
        "reduction": "median",
        "confusion_import": "from ml.models.segmentation import build_unet",
        "confusion_build": "lambda: build_unet(20, encoder_weights=None)",
    },
    "anysat": {
        "title": "# Segmentación de cultivos con AnySat (congelado) sobre PASTIS-R",
        "intro": (
            "Se entrena un segmentador basado en AnySat (Astruc et al., 2024), un modelo "
            "fundacional para datos de observación de la Tierra. AnySat se usa congelado, como "
            "extractor de características, y solo se entrena una cabeza lineal que las proyecta a "
            "las clases de cultivo; el entrenamiento es barato porque el grueso de los pesos no se "
            "actualiza.\n\n"
            "Este cuaderno corre de forma independiente (en paralelo con el de U-Net) y deja sus "
            "artefactos en carpetas del Drive compartido para el reporte: la tabla de métricas, la "
            "figura de la matriz de confusión y el modelo entrenado."
        ),
        "model_md": (
            "## Entrenamiento\n\n"
            "AnySat se descarga la primera vez desde su repositorio (torch.hub). El entrenamiento "
            "guarda un checkpoint por época en Drive; si la sesión se reinicia, al volver a "
            "ejecutar esta celda se reanuda desde la última época completada."
        ),
        "batch": "8",
        "reduction": "none",
        "confusion_import": "from ml.models.anysat_wrapper import AnySatSegmenter",
        "confusion_build": "lambda: AnySatSegmenter(20, target_size=TARGET_SIZE)",
    },
}


def _build_cells(model: str) -> list:
    """Construye las celdas del notebook para una arquitectura concreta."""
    md = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    meta = _META[model]
    cells = []

    cells.append(md(meta["title"] + "\n\n" + meta["intro"]))

    cells.append(
        md(
            "## Datos y métricas\n\n"
            "PASTIS-R entrega parches Sentinel-2 multitemporales de 128x128, que aquí se "
            "reescalan a 256. Las etiquetas tienen 20 clases: fondo, 18 tipos de cultivo y una "
            "clase void que se descarta en la pérdida y en las métricas. El split de "
            "entrenamiento y validación usa los folds oficiales del dataset, espacialmente "
            "disjuntos. Se reportan mIoU, F1-macro y exactitud a nivel de píxel en dos esquemas: "
            "las 18 clases planas y los 6 grupos agronómicos HCAT (cereales, oleaginosas, "
            "tubérculos, leguminosas, leñosos y otros), siendo este último el comparable con el "
            "baseline del avance anterior."
        )
    )

    cells.append(code(_SETUP_CELL))

    cells.append(
        code(
            "# Configuracion de la corrida.\n"
            "import torch\n\n"
            f"MODEL = '{model}'\n"
            "# El dataset vive en Drive; en local se usa la copia del repo.\n"
            "PASTIS_ROOT = Path((shared_folder_path + 'data/PASTIS-R') if shared_folder_path\n"
            "                   else 'data/PASTIS-R')\n"
            "# Carpetas de artefactos en Drive (claras para citarlas en el reporte):\n"
            "#   reports/segmentation/metrics      -> parquet de metricas por modelo\n"
            "#   reports/segmentation/figures      -> PNG de la matriz de confusion\n"
            "#   reports/segmentation/checkpoints  -> modelo final + checkpoint reanudable\n"
            "SEG_DIR = Path((shared_folder_path if shared_folder_path else '') + 'reports/segmentation')\n"
            "METRICS_DIR = SEG_DIR / 'metrics'\n"
            "FIGURES_DIR = SEG_DIR / 'figures'\n"
            "CHECKPOINT_DIR = SEG_DIR / 'checkpoints'\n"
            "for _d in (METRICS_DIR, FIGURES_DIR, CHECKPOINT_DIR):\n"
            "    _d.mkdir(parents=True, exist_ok=True)\n"
            "COMPARISON_PATH = METRICS_DIR / f'model_comparison_avance4_{MODEL}.parquet'\n"
            "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
            "TARGET_SIZE = 256\n"
            "SUBSET = 0            # 0 = todos; reducir (p.ej. 60) si la sesion es corta\n"
            "EPOCHS = 30\n"
            f"BATCH = {meta['batch']}      # pensado para L4 (24 GB); en T4 (16 GB) bajar a la mitad\n"
            "MLFLOW_URI = 'file:./mlruns'\n"
            "# Por defecto se lee directo de Drive (sin copiar). Si vas a entrenar muchas epocas y\n"
            "# preferis acelerar, pone COPY_TO_LOCAL=True (copia una vez al disco efimero).\n"
            "COPY_TO_LOCAL = False\n"
            "NUM_WORKERS = 4 if _IN_COLAB else 0\n\n"
            "print('modelo:', MODEL, '| device:', DEVICE, '| batch:', BATCH)\n"
            "print('PASTIS_ROOT:', PASTIS_ROOT, '| exists:', PASTIS_ROOT.exists())\n"
            "print('artefactos en:', SEG_DIR)"
        )
    )

    cells.append(
        md(
            "## Lectura del dataset\n\n"
            "Por defecto el dataset se lee directo desde Drive, sin copiar nada: así se evita la "
            "espera inicial y no se pierde trabajo si la sesión se reinicia. El loader abre cada "
            "parche con un solo acceso a disco (no relee el archivo de metadatos en cada paso) y el "
            "DataLoader usa varios procesos en paralelo. Si preferís acelerar, poné `COPY_TO_LOCAL = "
            "True` en la celda anterior para copiar una vez al disco local de la sesión."
        )
    )

    cells.append(code(_COPY_CELL))
    cells.append(code(_SPLIT_CELL))

    cells.append(md(meta["model_md"]))

    if model == "unet":
        train_code = (
            "# Entrenamiento de la U-Net.\n"
            "from ml.train.train_segmentation import run_training\n\n"
            "result = run_training(\n"
            "    model=MODEL, epochs=EPOCHS, batch_size=BATCH, target_size=TARGET_SIZE,\n"
            "    subset=SUBSET, device=DEVICE, root=PASTIS_ROOT, mlflow_uri=MLFLOW_URI,\n"
            "    comparison_path=COMPARISON_PATH, num_workers=NUM_WORKERS, output_dir=CHECKPOINT_DIR,\n"
            ")\n"
            "result"
        )
    else:
        train_code = (
            "# Carga de AnySat (torch.hub) y entrenamiento de la cabeza lineal.\n"
            "from ml.train.train_segmentation import run_training\n"
            "from ml.models.anysat_wrapper import load_anysat_encoder\n\n"
            "_ = load_anysat_encoder()  # descarga y valida los pesos antes de entrenar\n"
            "result = run_training(\n"
            "    model=MODEL, epochs=EPOCHS, batch_size=BATCH, target_size=TARGET_SIZE,\n"
            "    subset=SUBSET, device=DEVICE, root=PASTIS_ROOT, mlflow_uri=MLFLOW_URI,\n"
            "    comparison_path=COMPARISON_PATH, num_workers=NUM_WORKERS, output_dir=CHECKPOINT_DIR,\n"
            ")\n"
            "result"
        )
    cells.append(code(train_code))

    cells.append(
        md(
            "## Métricas\n\n"
            "Tabla de métricas de este modelo sobre el fold de validación, en los dos esquemas (18 "
            "clases y 6 grupos HCAT). Se guarda en `reports/segmentation/metrics/`; el notebook "
            "integrador la une con la del otro modelo para la comparativa final. Las columnas con "
            "sufijo `grouped` corresponden a los 6 grupos (el fondo no entra en esas métricas)."
        )
    )

    cells.append(
        code(
            "import polars as pl\n\n"
            "table = pl.read_parquet(COMPARISON_PATH)\n"
            "cols = ['model', 'miou_grouped', 'f1_macro_grouped', 'pixel_accuracy_grouped',\n"
            "        'miou', 'f1_macro', 'pixel_accuracy', 'train_time_s', 'epochs']\n"
            "table.select([c for c in cols if c in table.columns])"
        )
    )

    cells.append(
        md(
            "## Matriz de confusión\n\n"
            "Recall por clase a nivel de píxel sobre el fold de validación, sin contar la clase "
            "void. La figura se guarda en `reports/segmentation/figures/` para el reporte."
        )
    )

    cells.append(
        code(
            "# Matriz de confusion a nivel de pixel; se guarda como PNG en Drive.\n"
            "import torch\n"
            "from torch.utils.data import DataLoader\n"
            "from ml.ingest.pastis_dataset import PASTISDataset, load_norm_stats, PASTIS_IGNORE_INDEX\n"
            "from ml.ingest.pastis_loader import PASTIS_CLASS_MAP\n"
            "from ml.eval.dense_metrics import dense_confusion_figure\n"
            f"{meta['confusion_import']}\n\n"
            "def confusion_figure(model_name, reduction, build_fn, ckpt, max_patches=40):\n"
            "    norm = load_norm_stats(PASTIS_ROOT, folds=(1, 2, 3))\n"
            "    val_ids = split['val'][:max_patches]\n"
            "    ds = PASTISDataset(val_ids, root=PASTIS_ROOT, target_size=TARGET_SIZE,\n"
            "                       temporal_reduction=reduction, norm=norm)\n"
            "    loader = DataLoader(ds, batch_size=2)\n"
            "    model = build_fn().to(DEVICE)\n"
            "    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))\n"
            "    model.eval()\n"
            "    preds, tgts = [], []\n"
            "    with torch.no_grad():\n"
            "        for b in loader:\n"
            "            img = b['image'].to(DEVICE)\n"
            "            out = model(img) if model_name == 'unet' else model(img, b['dates'].to(DEVICE))\n"
            "            preds.append(out.argmax(1).cpu().reshape(-1))\n"
            "            tgts.append(b['semantic'].reshape(-1))\n"
            "    return dense_confusion_figure(torch.cat(preds), torch.cat(tgts),\n"
            "                                  class_names=PASTIS_CLASS_MAP, ignore_index=PASTIS_IGNORE_INDEX)\n\n"
            f"fig = confusion_figure(MODEL, '{meta['reduction']}', {meta['confusion_build']},\n"
            "                       result['checkpoint_path'])\n"
            "_fig_path = FIGURES_DIR / f'confusion_{MODEL}.png'\n"
            "fig.savefig(_fig_path, bbox_inches='tight', dpi=120)\n"
            "print('Figura guardada en:', _fig_path)\n"
            "fig"
        )
    )

    cells.append(
        md(
            "## Conclusiones\n\n"
            "Las métricas y la matriz de confusión quedan guardadas en `reports/segmentation/` "
            "(carpetas `metrics/` y `figures/`) y el modelo entrenado en `checkpoints/`. El "
            "notebook integrador `Avance4.Equipo17` reúne este modelo con el otro para la "
            "comparativa final, elige el de mejor desempeño y, si vale la pena, afina sus "
            "hiperparámetros con una búsqueda más fina como la del bloque siguiente."
        )
    )

    cells.append(
        code(
            "# Busqueda de hiperparametros con Optuna (opcional, si este modelo entra al top).\n"
            "#\n"
            "# import optuna\n"
            "# def objective(trial):\n"
            "#     lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)\n"
            "#     wd = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)\n"
            "#     res = run_training(model=MODEL, epochs=15, batch_size=BATCH, lr=lr, weight_decay=wd,\n"
            "#                        target_size=TARGET_SIZE, subset=SUBSET, device=DEVICE,\n"
            "#                        root=PASTIS_ROOT, mlflow_uri=MLFLOW_URI, resume=False)\n"
            "#     return res['miou_grouped']\n"
            "# study = optuna.create_study(direction='maximize', study_name=f'tune-{MODEL}')\n"
            "# study.optimize(objective, n_trials=30)\n"
            "# study.best_params"
        )
    )

    return cells


@app.command()
def main(
    model: Annotated[str, typer.Option(help="Arquitectura: 'unet' o 'anysat'.")] = "unet",
    out: Annotated[str, typer.Option(help="Ruta de salida (default segun modelo).")] = "",
) -> None:
    """Genera el notebook de segmentacion densa de una arquitectura.

    Args:
        model: ``unet`` o ``anysat``.
        out: Ruta destino del ``.ipynb`` (si vacia, se usa el default del modelo).
    """
    if model not in _OUT_BY_MODEL:
        raise typer.BadParameter("`--model` debe ser 'unet' o 'anysat'.")
    out_path = Path(out) if out else _OUT_BY_MODEL[model]
    nb = nbf.v4.new_notebook()
    nb["cells"] = _build_cells(model)
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    typer.echo(f"Notebook escrito: {out_path} ({len(nb['cells'])} celdas)")


if __name__ == "__main__":  # pragma: no cover - punto de entrada CLI
    app()
