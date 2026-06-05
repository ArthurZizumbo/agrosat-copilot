"""Builder of the ``04b_baseline_v2.ipynb`` notebook.

Generates a lightweight notebook that **reads** the artifacts persisted by the
real Baseline v2 run (3 models over the post-ablation winning set) and presents
them as a visual deliverable. The notebook does NOT re-train models: the real
training lives in MLflow + parquet under
``reports/baseline/model_comparison_v2/``.

Artifacts the notebook reads:
  - ``reports/baseline/model_comparison_v2/model_comparison_v2.parquet``
  - ``paper/figures/us-023-preview/model_comparison_v2.png``
  - MLflow runs (3) under ``mlruns/560033025078177743/``

Expected wall clock in papermill: <= 30 s (only parquet I/O + PNG render).

Usage:
    poetry run python scripts/build_baseline_v2_notebook.py
    poetry run python scripts/build_baseline_v2_notebook.py \\
        --out notebooks/baseline/04b_baseline_v2.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
import typer

app = typer.Typer(add_completion=False, help=__doc__)


def _md(source: str) -> nbf.NotebookNode:
    """Create a markdown cell."""
    return nbf.v4.new_markdown_cell(source)


def _code(source: str) -> nbf.NotebookNode:
    """Create a code cell."""
    return nbf.v4.new_code_cell(source)


# ---------------------------------------------------------------------------
# Notebook cells (reader-facing text with UTF-8 accents; ASCII identifiers).
# ---------------------------------------------------------------------------

CELLS: list[nbf.NotebookNode] = [
    _md(
        "# Baseline v2 — tres modelos sobre el conjunto ganador post-ablación\n"
        "\n"
        "Este cuaderno **lee** los artefactos reales generados por la corrida CUDA "
        "de los tres modelos del baseline (XGBoost + TempCNN + InceptionTime) "
        "sobre el conjunto de features ganador post-ablación: `no_geom`, 85 951 "
        "parcelas, 18 clases. El entrenamiento real está en MLflow + parquet; "
        "aquí únicamente se presentan los resultados.\n"
        "\n"
        "- **Hardware:** NVIDIA RTX 4070 8 GB (XGBoost en CPU, TempCNN e "
        "InceptionTime en GPU).\n"
        "- **Validación:** validación cruzada espacial de 5 pliegues con buffer "
        "de 1 km entre pliegues.\n"
        "- **Tiempo total real de entrenamiento:** 49,9 minutos.\n"
        "- **Reproducibilidad:** `scripts/run_baseline_v2_standalone.py` regenera "
        "los tres entrenamientos con la misma semilla.\n"
    ),
    _code(
        "from __future__ import annotations\n"
        "\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "import matplotlib\n"
        "\n"
        "matplotlib.use('Agg')  # headless backend\n"
        "import matplotlib.pyplot as plt  # noqa: E402,F401\n"
        "import polars as pl  # noqa: E402\n"
        "from IPython.display import Image, Markdown, display  # noqa: E402\n"
        "\n"
        "# Bootstrap sys.path para imports desde ml/*\n"
        "_REPO_BOOTSTRAP = Path.cwd().resolve()\n"
        "for _candidate in (_REPO_BOOTSTRAP, *_REPO_BOOTSTRAP.parents):\n"
        "    if (_candidate / 'pyproject.toml').is_file():\n"
        "        _REPO_BOOTSTRAP = _candidate\n"
        "        break\n"
        "if str(_REPO_BOOTSTRAP) not in sys.path:\n"
        "    sys.path.insert(0, str(_REPO_BOOTSTRAP))\n"
        "\n"
        "from ml.utils.notebook_setup import find_repo_root  # noqa: E402\n"
        "\n"
        "%load_ext autoreload\n"
        "%autoreload 2\n"
        "\n"
        "pl.Config.set_tbl_rows(20)\n"
        "pl.Config.set_tbl_cols(20)\n"
        "pl.Config.set_fmt_str_lengths(80)\n"
        "\n"
        "REPO = find_repo_root()\n"
        "display(Markdown(f'**Repositorio detectado:** `{REPO.name}`'))\n"
    ),
    # --- Section 1 ------------------------------------------------------
    _md(
        "## 1. Resultados de los tres modelos\n"
        "\n"
        "La tabla siguiente proviene de la corrida real persistida en "
        "`reports/baseline/model_comparison_v2/model_comparison_v2.parquet`. "
        "Cada fila reporta las métricas promediadas sobre los cinco pliegues de "
        "la **validación cruzada espacial**: cada pliegue agrupa parcelas por "
        "celda H3 + clustering KMeans, y se aplica un buffer de 1 km entre "
        "pliegues para que parcelas vecinas no aparezcan a la vez en "
        "entrenamiento y validación. Los números reflejan la capacidad de los "
        "modelos de generalizar a regiones no vistas — no una validación "
        "aleatoria optimista.\n"
        "\n"
        "Frente al baseline anterior, los tres modelos se reentrenaron sobre el "
        "conjunto `no_geom`, es decir, sin las tres columnas geométricas "
        "(`geom_*`) que introducían un atajo de identificación regional "
        "detectado durante la fase de saneamiento."
    ),
    _code(
        "df_v2 = pl.read_parquet(\n"
        "    REPO / 'reports/baseline/model_comparison_v2/model_comparison_v2.parquet'\n"
        ")\n"
        "display(\n"
        "    df_v2.select([\n"
        "        'model', 'f1_macro', 'f1_weighted', 'miou',\n"
        "        'accuracy', 'kappa', 'train_time_s', 'mlflow_run_id',\n"
        "    ])\n"
        ")\n"
    ),
    # --- Section 2 ------------------------------------------------------
    _md(
        "## 2. Comparación contra el baseline anterior\n"
        "\n"
        "La iteración previa del baseline reportó los mismos tres modelos sobre "
        "el conjunto `full` (con las columnas geométricas que se descartaron en "
        "la fase de saneamiento). La tabla a continuación mide cuánto cambió el "
        "F1-macro al pasar al conjunto sin geometría. Números pequeños "
        "confirman que la decisión de descartar las columnas geométricas no "
        "rompe los modelos — el valor de esta segunda corrida está en la "
        "trazabilidad reproducible (MLflow + DVC + validación cruzada espacial "
        "idéntica), no en una mejora numérica espectacular.\n"
        "\n"
        "| Modelo | F1-macro previo | F1-macro actual | Cambio |\n"
        "|--------|-----------------|-----------------|--------|\n"
        "| XGBoost | 0,4094 | 0,4094 | +0,0000 |\n"
        "| InceptionTime | 0,1865 | 0,1898 | +0,0033 |\n"
        "| TempCNN | 0,1430 | 0,1435 | +0,0005 |\n"
    ),
    _code(
        "# Numeros de la iteracion previa (baseline original, conjunto full).\n"
        "_previous = {\n"
        "    'xgboost': 0.4094,        # XGBoost CV espacial 5-fold, conjunto full\n"
        "    'inceptiontime': 0.1865,  # InceptionTime, conjunto full\n"
        "    'tempcnn': 0.1430,        # TempCNN, conjunto full\n"
        "}\n"
        "df_previous = pl.DataFrame({\n"
        "    'model': list(_previous.keys()),\n"
        "    'f1_macro_previo': list(_previous.values()),\n"
        "})\n"
        "df_deltas = (\n"
        "    df_v2.select(['model', 'f1_macro'])\n"
        "    .rename({'f1_macro': 'f1_macro_actual'})\n"
        "    .join(df_previous, on='model', how='left')\n"
        "    .with_columns(\n"
        "        (pl.col('f1_macro_actual') - pl.col('f1_macro_previo')).alias('cambio')\n"
        "    )\n"
        "    .select(['model', 'f1_macro_previo', 'f1_macro_actual', 'cambio'])\n"
        "    .sort('f1_macro_actual', descending=True)\n"
        ")\n"
        "display(df_deltas)\n"
    ),
    # --- Section 3 ------------------------------------------------------
    _md(
        "## 3. Visualización comparativa\n"
        "\n"
        "La figura `model_comparison_v2.png` resume las tres barras (XGBoost, "
        "InceptionTime, TempCNN) sobre el conjunto ganador. La barra más alta "
        "es la que se promueve como referencia tabular para la siguiente fase."
    ),
    _code(
        "_fig_path = REPO / 'paper/figures/us-023-preview/model_comparison_v2.png'\n"
        "display(Image(filename=str(_fig_path)))\n"
    ),
    # --- Section 4 ------------------------------------------------------
    _md(
        "## 4. Modelo ganador\n"
        "\n"
        "El criterio de selección es: **F1-macro principal**, con desempate por "
        "F1-weighted y, si el empate persiste, por mIoU. Sobre los números "
        "reales de la tabla anterior gana **XGBoost** con F1-macro = 0,4094. "
        "Los modelos temporales (TempCNN e InceptionTime) quedan a ~0,20 puntos "
        "por debajo y se reservan como aprendices base para la fase de modelos "
        "agregados."
    ),
    _code(
        "_winner = df_v2.sort('f1_macro', descending=True).row(0, named=True)\n"
        "display(Markdown(\n"
        "    f\"**Modelo ganador:** `{_winner['model']}` con \"\n"
        "    f\"F1-macro = `{_winner['f1_macro']:.4f}` \"\n"
        "    f\"(F1-weighted = `{_winner['f1_weighted']:.4f}`, mIoU = `{_winner['miou']:.4f}`).\"\n"
        "))\n"
    ),
    # --- Section 5 ------------------------------------------------------
    _md(
        "## 5. Trazabilidad en MLflow\n"
        "\n"
        "Cada modelo deja un registro en MLflow dentro del experimento "
        "`baseline-v2-us-023-preview`. Los registros incluyen las seis "
        "métricas, los hiperparámetros del divisor de pliegues "
        "(`k_folds=5`, `buffer_km=1.0`, `seed=42`), el tamaño del conjunto y "
        "los tags `data_version` y `code_version` necesarios para reabrir y "
        "reproducir la corrida."
    ),
    _code(
        "df_mlflow = pl.DataFrame([\n"
        "    {\n"
        "        'model': 'xgboost',\n"
        "        'run_id': 'ed898cea68524278bf3bdd9b6c703e6d',\n"
        "        'experiment': 'baseline-v2-us-023-preview',\n"
        "        'experiment_id': '560033025078177743',\n"
        "    },\n"
        "    {\n"
        "        'model': 'inceptiontime',\n"
        "        'run_id': '056178a2bd7d461b8ac79be820de7036',\n"
        "        'experiment': 'baseline-v2-us-023-preview',\n"
        "        'experiment_id': '560033025078177743',\n"
        "    },\n"
        "    {\n"
        "        'model': 'tempcnn',\n"
        "        'run_id': '3437f9b8756a4a58bf56c6982fd96c2e',\n"
        "        'experiment': 'baseline-v2-us-023-preview',\n"
        "        'experiment_id': '560033025078177743',\n"
        "    },\n"
        "])\n"
        "display(df_mlflow)\n"
    ),
    # --- Section 6 ------------------------------------------------------
    _md(
        "## 6. Conclusiones\n"
        "\n"
        "- **XGBoost confirma su rol como baseline tabular fuerte.** Sobre "
        "85 951 parcelas, 18 clases y validación cruzada espacial con cinco "
        "pliegues y buffer de 1 km, alcanza un F1-macro de 0,4094 — el mismo "
        "valor que ya teníamos, ahora con el conjunto saneado y trazabilidad "
        "completa.\n"
        "- **Los modelos temporales quedan a ~0,20 puntos por debajo.** "
        "TempCNN (0,1435) e InceptionTime (0,1898) no superan al modelo "
        "tabular sobre este conjunto. Es un resultado esperable para "
        "arquitecturas temporales entrenadas desde cero sin preentrenamiento, "
        "y los deja como candidatos válidos para ser combinados con XGBoost "
        "en modelos agregados, no como reemplazos.\n"
        "- **El salto al mínimo de F1-macro >= 0,60 no se cierra con bloques "
        "opcionales tabulares.** Las pruebas de FarSLIP, descripción "
        "fenológica textual y firma espectral produjeron cambios "
        "despreciables o negativos; el siguiente salto vendrá de los modelos "
        "densos (U-Net, U-TAE, TSViT, Swin-UNETR) y de combinar varios "
        "modelos.\n"
        "- **Reproducibilidad asegurada.** Los tres identificadores de "
        "ejecución listados arriba permiten reabrir el experimento en "
        "MLflow. El entrenamiento completo se puede regenerar con "
        "`scripts/run_baseline_v2_standalone.py`.\n"
        "\n"
        "### Lo que sigue\n"
        "\n"
        "Arrancar la fase de modelado denso sobre PASTIS-R con este baseline "
        "saneado como techo a batir. XGBoost queda como referencia tabular y "
        "los dos modelos temporales pasan al banco de aprendices base para "
        "los modelos agregados posteriores."
    ),
]


def build_notebook(out_path: Path) -> None:
    """Build the ``04b_baseline_v2.ipynb`` notebook and write it to disk.

    Args:
        out_path: Destination path of the ``.ipynb`` file.
    """
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out_path))


@app.command()
def main(
    out: Path = typer.Option(
        Path("notebooks/baseline/04b_baseline_v2.ipynb"),
        help="Ruta destino del notebook .ipynb.",
    ),
) -> None:
    """Rebuild ``notebooks/baseline/04b_baseline_v2.ipynb`` from scratch."""
    build_notebook(out)
    typer.echo(f"Notebook escrito en {out}")


if __name__ == "__main__":
    app()
