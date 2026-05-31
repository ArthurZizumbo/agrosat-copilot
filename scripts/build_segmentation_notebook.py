"""Builder del notebook entregable de segmentacion densa (Avance 4, modelos #1 y #6).

Genera ``notebooks/segmentation/04d_segmentation_unet_anysat.ipynb`` de forma
programatica y reproducible (mismo patron que ``scripts/build_baseline_notebook.py``).
El notebook es Colab-first: entrena U-Net ResNet-50 (#1) y AnySat frozen (#6) sobre
PASTIS-R reusando ``ml.train.train_segmentation.run_training``, exporta la tabla
comparativa y las matrices de confusion.

Uso::

    poetry run python scripts/build_segmentation_notebook.py \\
        --out notebooks/segmentation/04d_segmentation_unet_anysat.ipynb

Operativo permanente (NO viola el anti-patron ``scripts/_*.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import nbformat as nbf
import typer

app = typer.Typer(add_completion=False, help=__doc__)

_DEFAULT_OUT = Path("notebooks/segmentation/04d_segmentation_unet_anysat.ipynb")


def _build_cells() -> list:
    """Construye la lista de celdas (markdown + code) del notebook."""
    md = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    cells = []

    cells.append(
        md(
            "# Segmentación semántica de cultivos sobre PASTIS-R\n\n"
            "Se entrenan dos modelos de segmentación densa sobre las series Sentinel-2 de "
            "PASTIS-R y se compara su desempeño píxel a píxel. El primero es una U-Net con "
            "encoder ResNet-50 sobre un composite temporal; el segundo toma AnySat como "
            "extractor congelado y entrena solo una cabeza lineal. La intención es ver cuánto "
            "rinde cada enfoque para asignar el tipo de cultivo a cada píxel de la parcela y "
            "quedarnos con el que mejor resultado dé.\n\n"
            "El cuaderno está pensado para correr en Colab: monta el dataset desde Drive, lo "
            "copia al disco local de la sesión para que las épocas lean rápido, entrena los dos "
            "modelos y deja una tabla con las métricas y los tiempos de cada uno."
        )
    )

    cells.append(
        md(
            "## Datos y métricas\n\n"
            "PASTIS-R entrega parches Sentinel-2 multitemporales de 128x128, que aquí se "
            "reescalan a 256. Las etiquetas tienen 20 clases: fondo, 18 tipos de cultivo y una "
            "clase void que se descarta tanto en la pérdida como en las métricas. El split de "
            "entrenamiento y validación usa los folds oficiales del dataset, que son "
            "espacialmente disjuntos, de modo que parcelas vecinas no queden a la vez en "
            "entrenamiento y validación. Para comparar los modelos se reportan tres métricas a "
            "nivel de píxel: mIoU, F1-macro y exactitud."
        )
    )

    cells.append(
        code(
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
            "# Ajusta _branch si tu codigo esta en otra rama.\n"
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
    )

    cells.append(
        code(
            "# Configuracion de la corrida.\n"
            "import torch\n\n"
            "# El dataset vive en Drive; en local se usa la copia del repo.\n"
            "PASTIS_ROOT = Path((shared_folder_path + 'data/PASTIS-R') if shared_folder_path\n"
            "                   else 'data/PASTIS-R')\n"
            "# Las metricas de cada modelo se guardan en este parquet para armar la comparativa.\n"
            "COMPARISON_PATH = Path((shared_folder_path if shared_folder_path else '')\n"
            "                       + 'reports/segmentation/model_comparison_avance4_segmentacion.parquet')\n"
            "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
            "TARGET_SIZE = 256\n"
            "SUBSET = 0            # 0 = todos; reducir (p.ej. 60) si la sesion es corta\n"
            "EPOCHS_UNET = 30\n"
            "EPOCHS_ANYSAT = 30\n"
            "MLFLOW_URI = 'file:./mlruns'\n\n"
            "print('PASTIS_ROOT:', PASTIS_ROOT, '| exists:', PASTIS_ROOT.exists())\n"
            "print('device:', DEVICE, '| target_size:', TARGET_SIZE, '| subset:', SUBSET)"
        )
    )

    cells.append(
        md(
            "## Copia del dataset al disco local\n\n"
            "Leer desde el Drive montado es lento porque cada archivo pasa por una capa de red. "
            "Copiar el dataset una vez al disco local de la sesión hace que las épocas lean mucho "
            "más rápido. Solo se copia lo que estos modelos usan: las imágenes Sentinel-2, las "
            "anotaciones y los dos archivos de metadatos. La copia es reanudable: si se corta o se "
            "vuelve a ejecutar la celda, retoma donde quedó. En local no hace nada."
        )
    )

    cells.append(
        code(
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
            "if _IN_COLAB:\n"
            "    PASTIS_ROOT = copy_pastis_to_local(PASTIS_ROOT, '/content/PASTIS-R')\n"
            "    print('PASTIS_ROOT (local):', PASTIS_ROOT, '| exists:', PASTIS_ROOT.exists())\n"
            "else:\n"
            "    print('Local: no se copia. PASTIS_ROOT =', PASTIS_ROOT)"
        )
    )

    cells.append(
        code(
            "# Split en los folds oficiales de PASTIS (espacialmente disjuntos).\n"
            "from ml.ingest.pastis_dataset import pastis_fold_split\n\n"
            "split = pastis_fold_split(PASTIS_ROOT, train_folds=(1, 2, 3), val_folds=(4,), "
            "test_folds=(5,))\n"
            "print({k: len(v) for k, v in split.items()})"
        )
    )

    cells.append(
        md(
            "## U-Net con encoder ResNet-50\n\n"
            "La primera arquitectura es una U-Net clásica. El encoder ResNet-50 viene preentrenado "
            "en ImageNet y se adapta a las diez bandas de Sentinel-2. Como entrada se usa la mediana "
            "temporal de la serie y la salida es un mapa de clases a la resolución de la imagen."
        )
    )

    cells.append(
        code(
            "# Entrenamiento de la U-Net.\n"
            "from ml.train.train_segmentation import run_training\n\n"
            "unet_result = run_training(\n"
            "    model='unet', epochs=EPOCHS_UNET, batch_size=8, target_size=TARGET_SIZE,\n"
            "    subset=SUBSET, device=DEVICE, root=PASTIS_ROOT, mlflow_uri=MLFLOW_URI,\n"
            "    comparison_path=COMPARISON_PATH,\n"
            ")\n"
            "unet_result"
        )
    )

    cells.append(
        md(
            "## AnySat congelado con cabeza lineal\n\n"
            "La segunda arquitectura parte de AnySat (Astruc et al., 2024), un modelo fundacional "
            "para datos de observación de la Tierra. Aquí se usa congelado, como extractor de "
            "características, y solo se entrena una cabeza lineal que las proyecta a las clases de "
            "cultivo. El entrenamiento resulta mucho más barato porque el grueso de los pesos no se "
            "actualiza. AnySat se descarga la primera vez desde su repositorio; si esa descarga "
            "falla, la celda lo avisa sin detener el resto del cuaderno."
        )
    )

    cells.append(
        code(
            "# Carga de AnySat y entrenamiento de la cabeza lineal.\n"
            "anysat_result = None\n"
            "try:\n"
            "    from ml.models.anysat_wrapper import load_anysat_encoder\n"
            "    _ = load_anysat_encoder()  # descarga y valida los pesos antes de entrenar\n"
            "    anysat_result = run_training(\n"
            "        model='anysat', epochs=EPOCHS_ANYSAT, batch_size=4, target_size=TARGET_SIZE,\n"
            "        subset=SUBSET, device=DEVICE, root=PASTIS_ROOT, mlflow_uri=MLFLOW_URI,\n"
            "        comparison_path=COMPARISON_PATH,\n"
            "    )\n"
            "except Exception as exc:\n"
            "    print('AnySat no disponible en esta corrida:', exc)\n"
            "    print('Revisa el acceso a torch.hub gastruc/anysat y reejecuta esta celda.')\n"
            "anysat_result"
        )
    )

    cells.append(
        md(
            "## Comparativa\n\n"
            "La tabla reúne las métricas de los dos modelos sobre el fold de validación, "
            "ordenadas por mIoU, junto con el tiempo de entrenamiento de cada uno."
        )
    )

    cells.append(
        code(
            "# --- Tabla comparativa ---\n"
            "import polars as pl\n\n"
            "table = pl.read_parquet(COMPARISON_PATH).sort('miou', descending=True)\n"
            "cols = ['model', 'miou', 'f1_macro', 'pixel_accuracy', 'train_time_s', 'epochs', "
            "'n_train', 'n_val']\n"
            "table.select([c for c in cols if c in table.columns])"
        )
    )

    cells.append(
        md(
            "## Matriz de confusión\n\n"
            "Recall por clase a nivel de píxel sobre el fold de validación, sin contar la clase "
            "void. Ayuda a ver qué cultivos se separan bien y cuáles se confunden entre sí."
        )
    )

    cells.append(
        code(
            "# Matriz de confusion a nivel de pixel.\n"
            "import torch\n"
            "from torch.utils.data import DataLoader\n"
            "from ml.ingest.pastis_dataset import PASTISDataset, load_norm_stats, PASTIS_IGNORE_INDEX\n"
            "from ml.ingest.pastis_loader import PASTIS_CLASS_MAP\n"
            "from ml.eval.dense_metrics import dense_confusion_figure\n"
            "from ml.models.segmentation import build_unet\n\n"
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
            "fig_unet = confusion_figure('unet', 'median', lambda: build_unet(20, encoder_weights=None),\n"
            "                            unet_result['checkpoint_path'])\n"
            "fig_unet"
        )
    )

    cells.append(
        md(
            "## Conclusiones\n\n"
            "La comparativa deja ver el contraste entre los dos enfoques. La U-Net trabaja sobre un "
            "resumen temporal de la serie y es la opción más directa; AnySat reutiliza un modelo ya "
            "entrenado y apenas ajusta una cabeza, con un costo de entrenamiento bastante menor. La "
            "matriz de confusión muestra dónde se concentra el error, que suele estar entre cultivos "
            "de la misma familia. Con estos resultados se elige el modelo que mejor desempeño dé y, "
            "si vale la pena, se afinan sus hiperparámetros con una búsqueda más fina como la del "
            "bloque siguiente."
        )
    )

    cells.append(
        code(
            "# Busqueda de hiperparametros con Optuna para el modelo elegido (opcional).\n"
            "# Descomentar para ajustar el que haya dado mejor resultado.\n"
            "#\n"
            "# import optuna\n"
            "# def objective(trial):\n"
            "#     lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)\n"
            "#     wd = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)\n"
            "#     bs = trial.suggest_categorical('batch_size', [4, 8])\n"
            "#     res = run_training(model='unet', epochs=15, batch_size=bs, lr=lr, weight_decay=wd,\n"
            "#                        target_size=TARGET_SIZE, subset=SUBSET, device=DEVICE,\n"
            "#                        root=PASTIS_ROOT, mlflow_uri=MLFLOW_URI)\n"
            "#     return res['miou']\n"
            "# study = optuna.create_study(direction='maximize', study_name='tune-unet')\n"
            "# study.optimize(objective, n_trials=30)\n"
            "# study.best_params"
        )
    )

    return cells


@app.command()
def main(
    out: Annotated[
        Path, typer.Option(help="Ruta del notebook de salida.")
    ] = _DEFAULT_OUT,
) -> None:
    """Genera el notebook de segmentacion densa del Avance 4.

    Args:
        out: Ruta destino del ``.ipynb``.
    """
    nb = nbf.v4.new_notebook()
    nb["cells"] = _build_cells()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    typer.echo(f"Notebook escrito: {out} ({len(nb['cells'])} celdas)")


if __name__ == "__main__":  # pragma: no cover - punto de entrada CLI
    app()
