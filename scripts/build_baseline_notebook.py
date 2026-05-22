"""Constructor programatico de ``notebooks/04_baseline.ipynb`` (EPIC 4, US-019).

Genera el notebook del baseline RF/XGB celda a celda con ``nbformat.v4``,
ejecutable end-to-end con papermill y reproducible byte-a-byte. El notebook
es el entregable visual del Avance 3.

Secciones que produce US-019:
  - 1: Setup y carga del vector de features del EPIC 3.
  - 2: Justificacion del algoritmo (criterio "Algoritmo", 40 pts).
  - 6: Desempeno minimo vs umbral F1-macro >= 0.60 (criterio 10 pts).

Las secciones 3-5 (importance/SHAP, curvas) y 7-8 (comparativa) son
placeholders documentados que completan US-020, US-021 y US-022 — asi se
evita un merge conflict masivo sobre el .ipynb (decision D10).

Patron: ``scripts/build_us018_notebook.py``.

Uso:
    poetry run python scripts/build_baseline_notebook.py --out notebooks/04_baseline.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
import typer

app = typer.Typer(add_completion=False, help=__doc__)


def _md(source: str) -> nbf.NotebookNode:
    """Crea una celda markdown."""
    return nbf.v4.new_markdown_cell(source)


def _code(source: str) -> nbf.NotebookNode:
    """Crea una celda de codigo."""
    return nbf.v4.new_code_cell(source)


def _params_code(source: str) -> nbf.NotebookNode:
    """Crea la celda de parametros (tag ``parameters`` para papermill)."""
    cell = nbf.v4.new_code_cell(source)
    cell.metadata["tags"] = ["parameters"]
    return cell


# ---------------------------------------------------------------------------
# Celdas del notebook.
# ---------------------------------------------------------------------------

CELLS: list[nbf.NotebookNode] = [
    _md(
        "# 04 — Baseline RF/XGB sobre features combinados (EPIC 4)\n"
        "\n"
        "**Avance 3 — Baseline · CRISP-ML(Q) fase 3 Modeling**\n"
        "\n"
        "Este notebook entrena dos modelos tabulares (Random Forest y "
        "XGBoost) sobre el vector de features del EPIC 3 (AlphaEarth 64-dim "
        "+ indices espectrales + estadisticas temporales + SRTM + ERA5) y "
        "documenta su desempeno contra el umbral minimo del Avance 3.\n"
        "\n"
        "| Seccion | Contenido | US |\n"
        "|---------|-----------|-----|\n"
        "| 1 | Setup y carga del dataset | US-019 |\n"
        "| 2 | Justificacion del algoritmo (40 pts) | US-019 |\n"
        "| 3 | Importancia de features nativa | US-020 |\n"
        "| 4 | Analisis SHAP | US-020 |\n"
        "| 5 | Conclusiones de feature engineering | US-020 |\n"
        "| 5b | Curvas de aprendizaje y validacion | US-021 |\n"
        "| 6 | Desempeno minimo vs umbral 0.60 (10 pts) | US-019 |\n"
        "| 7 | Comparativa AlphaEarth vs Sentinel-2 crudo | US-022 |\n"
        "| 8 | Discusion y decisiones para EPIC 5 | US-022 |\n"
    ),
    # --- Seccion 1 -------------------------------------------------------
    _md(
        "## 1. Setup y carga del dataset\n"
        "\n"
        "El dataset de entrada es el subset PASTIS-R a nivel parcela "
        "generado en el EPIC 3 (US-018): 85.951 parcelas x 187 features "
        "espectro-temporales. Las etiquetas son las 20 clases de cultivo "
        "de PASTIS-R (se descartan las clases de fondo)."
    ),
    _params_code(
        "# Parametros papermill (celda con tag 'parameters'; sobreescribibles\n"
        "# en CI con valores reducidos via `papermill -p`).\n"
        "FEATURES_PATH = 'data/test_fixtures/feature_selection_parcels_subset.parquet'\n"
        "MAX_SAMPLES = 0  # 0 = dataset completo; >0 = submuestreo estratificado\n"
        "TUNE = True\n"
        "F1_THRESHOLD = 0.60\n"
    ),
    _code(
        "import warnings\n"
        "\n"
        "import matplotlib\n"
        "\n"
        "matplotlib.use('Agg')  # backend headless para papermill/CI\n"
        "import matplotlib.pyplot as plt\n"
        "import polars as pl\n"
        "\n"
        "warnings.filterwarnings('ignore')\n"
    ),
    _code(
        "from ml.train.baseline import _load_baseline_dataset, _prepare_dataframe\n"
        "\n"
        "df_raw = _load_baseline_dataset(FEATURES_PATH)\n"
        "df = _prepare_dataframe(df_raw)\n"
        "print(f'Parcelas: {df.height:,}  |  Columnas: {df.width}')\n"
        "df.head()"
    ),
    _code(
        "# Distribucion de clases — PASTIS-R tiene desbalance fuerte.\n"
        "class_counts = (\n"
        "    df.group_by('class_id').len().sort('len', descending=True)\n"
        ")\n"
        "class_counts"
    ),
    # --- Seccion 2 -------------------------------------------------------
    _md(
        "## 2. Justificacion del algoritmo\n"
        "\n"
        "Se eligen **Random Forest** y **XGBoost** como baseline tabular. "
        "Cuatro argumentos sustentan la decision:\n"
        "\n"
        "**(a) AlphaEarth ya codifica la informacion multisensor.** El "
        "embedding AlphaEarth Foundations de 64 dimensiones condensa "
        "informacion optica, radar y temporal aprendida por un Foundation "
        "Model entrenado sobre todo el archivo Sentinel. Sobre una "
        "representacion ya rica, un modelo tabular es un baseline "
        "suficiente y honesto — no se requiere una arquitectura profunda "
        "para establecer el lower bound (cf. Brown et al., 2025, "
        "*AlphaEarth Foundations*; EDA US-013).\n"
        "\n"
        "**(b) RF y XGBoost son interpretables.** Ambos exponen "
        "importancia de features nativa (Gini para RF, gain para XGBoost) "
        "y son compatibles con SHAP (TreeExplainer exacto). El criterio "
        "'Caracteristicas importantes' del Avance 3 (US-020) depende de "
        "esta interpretabilidad — un baseline opaco no permitiria "
        "auditar que features aportan (Lundberg & Lee, 2017, *SHAP*).\n"
        "\n"
        "**(c) Robustez a outliers y a la escala.** Los arboles "
        "particionan el espacio por umbrales y no asumen ninguna "
        "distribucion de las features; outliers residuales tras la "
        "winsorizacion del EPIC 3 no desplazan las fronteras de decision "
        "como lo harian en un modelo lineal o en una red sin "
        "normalizacion cuidadosa.\n"
        "\n"
        "**(d) Bajo costo computacional.** El problema (85k x 187, 20 "
        "clases) se entrena en minutos. XGBoost usa el GPU local cuando "
        "esta disponible y degrada a CPU de forma transparente; RF corre "
        "siempre en CPU multinucleo. El baseline es reproducible en la "
        "laptop de cualquier integrante del equipo y en CI sin reservar "
        "computo cloud, dejando el presupuesto H100 para EPIC 5/6."
    ),
    # --- Seccion 3-5 placeholders (US-020) -------------------------------
    _md(
        "## 3. Importancia de features nativa\n"
        "\n"
        "_Placeholder — completado por US-020 (Feature importance + SHAP)._"
    ),
    _md(
        "## 4. Analisis SHAP\n\n_Placeholder — completado por US-020 (Feature importance + SHAP)._"
    ),
    _md(
        "## 5. Conclusiones de feature engineering\n"
        "\n"
        "_Placeholder — completado por US-020 (Feature importance + SHAP)._"
    ),
    _md(
        "## 5b. Curvas de aprendizaje y validacion\n"
        "\n"
        "_Placeholder — completado por US-021 (Curvas de aprendizaje)._"
    ),
    # --- Seccion 6 -------------------------------------------------------
    _md(
        "## 6. Desempeno minimo\n"
        "\n"
        "El Avance 3 fija un umbral de **F1-macro >= 0.60** sobre "
        "PASTIS-R. Se entrenan RF y XGBoost con validacion cruzada "
        "**espacial** (H3 + KMeans + buffer 1 km, sin leakage entre "
        "parcelas vecinas) y se reporta la media CV de cada metrica.\n"
        "\n"
        "La rubrica evalua que el desempeno este **declarado y "
        "justificado**, no que el umbral se supere: si F1-macro < 0.60 "
        "se documentan las causas y las decisiones para EPIC 5 en la "
        "seccion 6.1."
    ),
    _code(
        "from ml.train.baseline import train_one_model, tune_baseline\n"
        "\n"
        "results = {}\n"
        "for kind in ('rf', 'xgb'):\n"
        "    if TUNE:\n"
        "        best_params = tune_baseline(df, model=kind)\n"
        "        results[kind] = train_one_model(\n"
        "            df, model=kind, hyperparams=best_params\n"
        "        )\n"
        "    else:\n"
        "        results[kind] = train_one_model(df, model=kind)\n"
        "    print(f'{kind.upper()}  entrenado.')"
    ),
    _code(
        "# Tabla resumen de las metricas CV-mean por modelo.\n"
        "summary = pl.DataFrame(\n"
        "    [\n"
        "        {\n"
        "            'modelo': kind.upper(),\n"
        "            **{m: round(v, 4) for m, v in res.metrics.items()},\n"
        "        }\n"
        "        for kind, res in results.items()\n"
        "    ]\n"
        ")\n"
        "summary"
    ),
    _code(
        "# Veredicto vs el umbral del Avance 3.\n"
        "best_kind = max(results, key=lambda k: results[k].metrics['f1_macro'])\n"
        "best_f1 = results[best_kind].metrics['f1_macro']\n"
        "passed = best_f1 >= F1_THRESHOLD\n"
        "print(f'Mejor modelo: {best_kind.upper()}  |  F1-macro = {best_f1:.4f}')\n"
        "print(f'Umbral Avance 3: {F1_THRESHOLD:.2f}  |  '\n"
        '      f\'{"ALCANZADO" if passed else "NO alcanzado — ver 6.1"}\')'
    ),
    _md(
        "### 6.1 Causas probables y decisiones para EPIC 5\n"
        "\n"
        "Si el F1-macro CV-mean queda por debajo de 0.60, las causas "
        "probables son:\n"
        "\n"
        "1. **Granularidad fina de PASTIS-R (20 clases).** Varias clases "
        "de cultivo son espectralmente similares; un modelo tabular sobre "
        "un embedding anual no captura la firma fenologica que las "
        "distingue.\n"
        "2. **Desbalance de clases.** Pese al balanceo "
        "(`class_weight='balanced'` en RF, `sample_weight` inverso a "
        "frecuencia en XGBoost), las clases minoritarias aportan pocas "
        "parcelas y el F1-macro las penaliza con fuerza.\n"
        "3. **Limite de un modelo tabular sobre un embedding generico.** "
        "AlphaEarth resume el ano en 64 dimensiones; pierde la dinamica "
        "temporal intra-anual que un modelo de series temporales si "
        "aprovecha.\n"
        "\n"
        "Decisiones concretas que EPIC 5 incorpora:\n"
        "\n"
        "- **U-TAE y TSViT** explotan la serie temporal Sentinel-2 "
        "completa (no el embedding resumido), capturando la fenologia "
        "que separa cultivos similares.\n"
        "- **Ensamble heterogeneo (EPIC 6)** combina el baseline tabular "
        "con los modelos temporales y un VLM, recuperando senal "
        "complementaria que ningun modelo aislado captura."
    ),
    # --- Seccion 7-8 placeholders (US-022) -------------------------------
    _md(
        "## 7. Comparativa AlphaEarth vs Sentinel-2 crudo\n"
        "\n"
        "_Placeholder — completado por US-022 (Notebook secuencial + "
        "comparativa)._"
    ),
    _md(
        "## 8. Discusion y decisiones para EPIC 5\n"
        "\n"
        "_Placeholder — completado por US-022 (Notebook secuencial + "
        "comparativa)._"
    ),
]


def build_notebook(out_path: Path) -> None:
    """Construye el notebook del baseline y lo escribe en ``out_path``.

    Args:
        out_path: Ruta destino del fichero ``.ipynb``.
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
    out: Path = typer.Option(  # noqa: B008
        Path("notebooks/04_baseline.ipynb"),
        help="Ruta destino del notebook .ipynb.",
    ),
) -> None:
    """Reconstruye ``notebooks/04_baseline.ipynb`` desde cero."""
    build_notebook(out)
    typer.echo(f"Notebook escrito en {out}")


if __name__ == "__main__":
    app()
