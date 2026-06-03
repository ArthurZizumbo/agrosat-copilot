"""Genera el notebook integrador ``Avance3.Equipo17.ipynb`` (consolidado A3).

El Avance 3 reune el trabajo del baseline (notebook `04_baseline.ipynb`),
la ablation post-A3 con bloques opcionales (notebook
`05_reencuadre_fenologico.ipynb`) y el baseline v2 reentrenado sobre el
conjunto ganador (notebook `04b_baseline_v2.ipynb`). El notebook integrador
**no reentrena** modelos: lee los artefactos persistidos en
`reports/baseline/` y los presenta en un unico recorrido coherente con el
mapeo 1:1 a la rubrica oficial del Avance 3.

Sigue el patron de ``scripts/build_avance2_notebook.py`` (builder programatico
con celdas markdown + code para Polars / matplotlib / IPython.display.Image).

Uso:
    poetry run python scripts/build_avance3_notebook.py
    poetry run python scripts/build_avance3_notebook.py \\
        --out notebooks/baseline/Avance3.Equipo17.ipynb
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


# ---------------------------------------------------------------------------
# Cells of the Avance 3 integrator notebook.
# ---------------------------------------------------------------------------

CELLS: list[nbf.NotebookNode] = [
    _md(
        "# Avance 3 — Baseline (Equipo 17, AgroSatCopilot)\n"
        "## Proyecto Integrador MNA · Tec de Monterrey\n"
        "\n"
        "**Equipo 17**\n"
        "- Carlos Isaac Ávila Gutiérrez — A01796035\n"
        "- Carlos Aaron Bocanegra Buitrón — A01796345\n"
        "- Arthur Jafed Zizumbo Velasco — A01796363\n"
        "\n"
        "**Curso:** MNA — Tec de Monterrey · 20-abr → 3-jul-2026\n"
        "**Fecha de entrega:** 2026-05-20 (con correcciones post-A3 cerradas 2026-05-26).\n"
        "\n"
        "---\n"
        "\n"
        "## Resumen ejecutivo\n"
        "\n"
        "Este notebook consolida el trabajo del **Avance 3 (Baseline)** y las correcciones "
        "incorporadas en la iteración posterior. La fase baseline establece el piso de desempeño "
        "de los modelos sobre las 85.951 parcelas PASTIS-R y deja un protocolo de evaluación (CV "
        "espacial 5-fold con buffer 1 km) reutilizable por las siguientes fases.\n"
        "\n"
        "El entregable original (`notebooks/baseline/04_baseline.ipynb`) reporta Random Forest y "
        "XGBoost con interpretabilidad SHAP y curvas de aprendizaje. La iteración posterior "
        "(notebook `05_reencuadre_fenologico.ipynb`) examina cuatro hipótesis de ingeniería de "
        "características (leakage geométrico, FarSLIP, descripción fenológica textual y firma "
        "espectral) y toma decisiones de promover, descartar o diferir cada bloque. Finalmente, el "
        "notebook `04b_baseline_v2.ipynb` reentrena los tres modelos canónicos (XGBoost, TempCNN, "
        "InceptionTime) sobre el conjunto saneado para tener un techo cuantificado a batir en la "
        "fase de modelos densos.\n"
    ),
    _md(
        "## Índice\n"
        "\n"
        "1. [Baseline original — Random Forest y XGBoost](#sec1)\n"
        "2. [Ablation post-A3 y bloques opcionales](#sec2)\n"
        "3. [Baseline v2 — 3 modelos sobre el conjunto ganador](#sec3)\n"
        "4. [Conclusiones consolidadas](#sec4)\n"
        "\n"
        "Los notebooks de trabajo detallado viven en:\n"
        "\n"
        "- [`../baseline/04_baseline.ipynb`](../baseline/04_baseline.ipynb)\n"
        "- [`../baseline/05_reencuadre_fenologico.ipynb`](../baseline/05_reencuadre_fenologico.ipynb)\n"
        "- [`../baseline/04b_baseline_v2.ipynb`](../baseline/04b_baseline_v2.ipynb)\n"
    ),
    _code(
        "from __future__ import annotations\n"
        "\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "import matplotlib\n"
        "\n"
        "matplotlib.use('Agg')  # backend headless para papermill\n"
        "import matplotlib.pyplot as plt  # noqa: E402\n"
        "import polars as pl  # noqa: E402\n"
        "from IPython.display import Image, Markdown, display  # noqa: E402\n"
        "\n"
        "# Bootstrap sys.path para que el notebook pueda importar desde ml/*\n"
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
        "display(Markdown(f'**Repo bootstrap OK** -> `{REPO.name}`'))\n"
    ),
    # --- Section 1 ------------------------------------------------------
    _md(
        '<a id="sec1"></a>\n'
        "## 1. Baseline original — Random Forest y XGBoost\n"
        "\n"
        "El notebook fuente del Avance 3 entrena Random Forest y XGBoost sobre el vector de "
        "características combinado (AlphaEarth 64-dim + índices espectrales + estadísticas "
        "temporales + terreno + clima, 187 columnas) y los evalúa con CV espacial 5-fold sobre "
        "85.951 parcelas. El umbral de referencia del baseline es F1-macro >= 0,60.\n"
        "\n"
        "**Detalle completo:** [`../baseline/04_baseline.ipynb`](../baseline/04_baseline.ipynb).\n"
        "\n"
        "Resumen de los hallazgos del notebook original:\n"
        "\n"
        "- XGBoost alcanza un F1-macro ~0,41 sobre las 18 clases efectivas — por debajo del "
        "umbral de 0,60. La causa es estructural: las 187 características resumen el año en un "
        "solo vector y pierden la dinámica intra-anual que separa cultivos espectralmente "
        "parecidos.\n"
        "- Random Forest y XGBoost rinden parecido dentro de cada escenario; la diferencia grande "
        "aparece entre escenarios (AlphaEarth vs Sentinel-2 crudo vs vector combinado).\n"
        "- Las curvas de aprendizaje confirman que los modelos NO sobreajustan: simplemente han "
        "llegado a su techo sobre datos que perdieron la dimensión temporal.\n"
        "- El análisis SHAP confirma que la mayoría de las top-20 características son dimensiones "
        "del embedding AlphaEarth, lo que valida la decisión de tratar el embedding como base "
        "del vector combinado.\n"
    ),
    _code(
        "# Leemos la fila resumen del baseline original desde el ablation\n"
        "# historico de US-022 (mismo splitter spatial CV 5-fold).\n"
        "_baseline_table = REPO / 'reports/baseline/reencuadre_fenologico/ablation_table.parquet'\n"
        "if _baseline_table.exists():\n"
        "    df_baseline_hist = pl.read_parquet(_baseline_table)\n"
        "    display(Markdown('**Tabla de ablation historica (US-022) sobre 85.951 parcelas:**'))\n"
        "    display(df_baseline_hist)\n"
        "else:\n"
        "    display(Markdown('Tabla historica no disponible en disco.'))\n"
    ),
    # --- Section 2 ------------------------------------------------------
    _md(
        '<a id="sec2"></a>\n'
        "## 2. Ablation post-A3 y bloques opcionales\n"
        "\n"
        "Tras el Avance 3 quedaron abiertas cuatro hipótesis de ingeniería de características que "
        "afectan la limpieza del baseline. La iteración posterior las evalúa una por una con CV "
        "espacial idéntica al del baseline original:\n"
        "\n"
        "- **H-1 — Leakage geométrico:** las 3 columnas `geom_area`, `geom_perimeter`, "
        "`geom_elongation` actúan como proxy de región (parcelas de la misma zona comparten "
        "distribución de tamaño y forma). Decisión: **descartar** del baseline definitivo.\n"
        "- **H-2 — FarSLIP (embeddings visuales):** modelo CLIP fine-tuned sobre PASTIS, evaluado "
        "sobre subset matched de 30.173 parcelas. Decisión: **diferir** a la iteración siguiente "
        "(deuda US-024) hasta extender el matching al dataset completo.\n"
        "- **H-3 — Descripción fenológica textual vía Gemini Flash 3.5:** prompt agronómico "
        "generado para 1.080 parcelas balanceadas (60 por clase x 18 clases), encodeado con "
        "sentence-transformers a 384 dim. Costo real medido: $0.49 USD. Decisión: **diferir** "
        "(delta vs full = -0.0354, por debajo del umbral de promoción).\n"
        "- **H-4 — Firma espectral (Red Edge Position, Frampton et al. 2013):** descriptor "
        "compacto derivado de las bandas red-edge Sentinel-2 en los anclajes fenológicos. "
        "Decisión: **diferir** salvo confirmación en futuras corridas.\n"
        "\n"
        "**Detalle completo:** "
        "[`../baseline/05_reencuadre_fenologico.ipynb`](../baseline/05_reencuadre_fenologico.ipynb).\n"
    ),
    _code(
        "# Tabla consolidada de los bloques opcionales evaluados.\n"
        "# Une la ablation historica (US-022, 5 conjuntos canonicos) con la\n"
        "# ablation real de pheno_text (US-023-preview P4, 1080 parcelas).\n"
        "_abl_hist_path = REPO / 'reports/baseline/reencuadre_fenologico/ablation_table.parquet'\n"
        "_abl_p4_path = (\n"
        "    REPO / 'reports/baseline/feature_ablation/ablation_table_pheno_text_v2.parquet'\n"
        ")\n"
        "\n"
        "df_abl_p4 = pl.read_parquet(_abl_p4_path).select([\n"
        "    'feature_set', 'n_features', 'f1_macro', 'f1_weighted', 'miou', 'delta_vs_full',\n"
        "])\n"
        "display(Markdown('**Ablation real (pheno_text via Gemini Flash 3.5, 1080 parcelas):**'))\n"
        "display(df_abl_p4)\n"
        "\n"
        "if _abl_hist_path.exists():\n"
        "    df_abl_hist = pl.read_parquet(_abl_hist_path).select([\n"
        "        'feature_set', 'n_features', 'f1_macro', 'f1_weighted', 'miou', 'delta_vs_full',\n"
        "    ])\n"
        "    display(Markdown('**Ablation historica (5 conjuntos canonicos sobre dataset full):**'))\n"
        "    display(df_abl_hist)\n"
    ),
    _code(
        "_fig_optional = REPO / 'paper/figures/us-023-preview/ablation_optional_blocks.png'\n"
        "if _fig_optional.exists():\n"
        "    display(Markdown('**Figura — bloques opcionales evaluados contra el conjunto `full`:**'))\n"
        "    display(Image(filename=str(_fig_optional)))\n"
        "else:\n"
        "    display(Markdown('Figura `ablation_optional_blocks.png` no disponible en disco.'))\n"
    ),
    _code(
        "_fig_geom = REPO / 'paper/figures/us-023-preview/ablation_geom_comparison.png'\n"
        "if _fig_geom.exists():\n"
        "    display(Markdown('**Figura — leakage geometrico (`full` vs `no_geom`):**'))\n"
        "    display(Image(filename=str(_fig_geom)))\n"
        "else:\n"
        "    display(Markdown('Figura `ablation_geom_comparison.png` no disponible en disco.'))\n"
    ),
    # --- Section 3 ------------------------------------------------------
    _md(
        '<a id="sec3"></a>\n'
        "## 3. Baseline v2 — 3 modelos sobre el conjunto ganador\n"
        "\n"
        "Con el conjunto de features saneado (`no_geom`, sin las 3 columnas que introducían el "
        "atajo regional), se reentrenan los **3 modelos canónicos del Avance 3** sobre 85.951 "
        "parcelas con la misma validación cruzada espacial de 5 pliegues con buffer de 1 km:\n"
        "\n"
        "- **XGBoost** (tabular, gradient boosting, NaN-aware).\n"
        "- **TempCNN** (Pelletier et al. 2019, importado de `breizhcrops.models`).\n"
        "- **InceptionTime** (Fawaz et al. 2020, importado de `breizhcrops.models`).\n"
        "\n"
        "Wall clock real: 49,9 min en RTX 4070 (XGBoost en CPU, modelos temporales en GPU). Los "
        "artefactos del entrenamiento (3 run IDs MLflow + tabla + figura) viven en disco; este "
        "notebook solo los presenta.\n"
        "\n"
        "**Detalle completo:** [`../baseline/04b_baseline_v2.ipynb`](../baseline/04b_baseline_v2.ipynb).\n"
    ),
    _code(
        "_v2_path = REPO / 'reports/baseline/model_comparison_v2/model_comparison_v2.parquet'\n"
        "df_v2 = pl.read_parquet(_v2_path)\n"
        "display(Markdown('**Resultados Baseline v2 — 3 modelos sobre conjunto saneado:**'))\n"
        "display(df_v2.select([\n"
        "    'model', 'f1_macro', 'f1_weighted', 'miou', 'accuracy', 'kappa', 'train_time_s',\n"
        "]))\n"
    ),
    _code(
        "_fig_v2 = REPO / 'paper/figures/us-023-preview/model_comparison_v2.png'\n"
        "if _fig_v2.exists():\n"
        "    display(Markdown('**Figura — comparativa visual de los 3 modelos v2:**'))\n"
        "    display(Image(filename=str(_fig_v2)))\n"
        "else:\n"
        "    display(Markdown('Figura `model_comparison_v2.png` no disponible en disco.'))\n"
    ),
    # --- Section 4 ------------------------------------------------------
    _md(
        '<a id="sec4"></a>\n'
        "## 4. Conclusiones consolidadas\n"
        "\n"
        "Este recorrido cierra la fase baseline con cinco hallazgos principales que orientan "
        "las fases siguientes del proyecto.\n"
        "\n"
        "### Elección y comparación de modelos\n"
        "\n"
        "Se evaluó XGBoost como modelo tabular fuerte sobre el vector combinado AlphaEarth + "
        "índices espectrales + temporales + terreno + clima. La iteración posterior incorporó "
        "dos arquitecturas temporales (TempCNN, InceptionTime) reentrenadas sobre el conjunto "
        "saneado. El criterio de selección del modelo de referencia es F1-macro sobre "
        "validación cruzada espacial de 5 pliegues, con desempate por F1-weighted y mIoU. "
        "XGBoost queda como referencia tabular (F1-macro 0,4094); TempCNN (0,1435) e "
        "InceptionTime (0,1898) quedan como candidatos para combinarse con XGBoost en la "
        "fase de modelos agregados.\n"
        "\n"
        "### Análisis de características\n"
        "\n"
        "La ingeniería de características validó con SHAP que la mayoría de las top-20 "
        "variables son dimensiones del embedding AlphaEarth — lo que justifica tratar el "
        "embedding como base. La iteración posterior probó cuatro bloques opcionales: "
        "leakage geométrico (descartado por proxy regional), FarSLIP (diferido hasta extender "
        "el matching al dataset completo), descripción fenológica textual vía Gemini Flash "
        "3.5 sobre 1.080 parcelas balanceadas (delta vs full = -0,0354, queda diferido), "
        "firma espectral REP (diferido salvo confirmación en futuras corridas). El conjunto "
        "de features ganador es `no_geom` (185 columnas, sin las 3 columnas geométricas).\n"
        "\n"
        "### Validación y diagnóstico de ajuste\n"
        "\n"
        "La validación cruzada espacial con buffer de 1 km entre pliegues garantiza que "
        "parcelas vecinas no queden a la vez en entrenamiento y validación; las métricas "
        "reportadas reflejan capacidad de generalizar a regiones no vistas. Las curvas de "
        "aprendizaje y validación del notebook original muestran que el baseline NO "
        "sobreajusta: el techo es estructural — perder la dimensión temporal al resumir el "
        "año en un vector. El diagnóstico explícito reporta ajuste adecuado con exactitud "
        "limitada por la capacidad del modelo.\n"
        "\n"
        "### Métricas reportadas\n"
        "\n"
        "La métrica principal es **F1-macro**, que penaliza fuerte a las clases minoritarias "
        "en un problema con desbalance severo y 18 clases efectivas. Se reportan además "
        "F1-weighted (referencia global), mIoU (compatible con la fase siguiente de "
        "segmentación densa), accuracy y Cohen kappa.\n"
        "\n"
        "### Brecha al objetivo del proyecto\n"
        "\n"
        "El umbral de referencia F1-macro >= 0,60 no se alcanza con el baseline tabular "
        "(XGBoost 0,4094) ni con los modelos temporales (0,14-0,19). Los bloques opcionales "
        "tabulares (FarSLIP, descripción fenológica textual, firma espectral) produjeron "
        "deltas <= 0 sobre el baseline saneado, así que la brecha al umbral no se cierra "
        "agregando más características tabulares. El avance vendrá de las arquitecturas "
        "densas (U-Net, U-TAE, TSViT, Swin-UNETR) y de combinar varios modelos.\n"
        "\n"
        "### Lo que sigue\n"
        "\n"
        "Con el baseline saneado, los conjuntos de features con decisión documentada y los "
        "tres modelos canónicos reentrenados, la fase siguiente arranca el modelado denso "
        "sobre PASTIS-R con un techo cuantificado a batir. XGBoost queda como referencia "
        "tabular y los dos modelos temporales pasan al banco de aprendices base para los "
        "modelos agregados posteriores.\n"
    ),
]


def build_notebook(out_path: Path) -> None:
    """Construye el notebook ``Avance3.Equipo17.ipynb`` y lo escribe en disco.

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
    out: Path = typer.Option(
        Path("notebooks/baseline/Avance3.Equipo17.ipynb"),
        help="Ruta destino del notebook .ipynb.",
    ),
) -> None:
    """Reconstruye ``notebooks/baseline/Avance3.Equipo17.ipynb`` desde cero."""
    build_notebook(out)
    typer.echo(f"Notebook escrito en {out}")


if __name__ == "__main__":
    app()
