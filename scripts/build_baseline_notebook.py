"""Constructor programatico de ``notebooks/04_baseline.ipynb`` (EPIC 4, US-019).

Genera el notebook del baseline RF/XGB celda a celda con ``nbformat.v4``,
ejecutable end-to-end con papermill y reproducible byte-a-byte. El notebook
es el entregable visual del Avance 3.

Secciones que produce US-019:
  - 1: Setup y carga del vector de features del EPIC 3.
  - 2: Justificacion del algoritmo (criterio "Algoritmo", 40 pts).
  - 6: Desempeno minimo vs umbral F1-macro >= 0.60 (criterio 10 pts).

Secciones que produce US-020 (criterio "Caracteristicas importantes", 20 pts):
  - 3: Importancia de features nativa (RF Gini / XGB gain, barplot top-20).
  - 4: Analisis SHAP (summary + dependence top-5 + waterfall + dominancia
    AlphaEarth).
  - 5: Conclusiones de feature engineering (cruce con el FE de US-018).

Seccion que produce US-021 (criterio "Sub/sobreajuste", 10 pts):
  - 5b: Curvas de aprendizaje (RF+XGB) + 3 curvas de validacion +
    diagnostico textual de sub/sobreajuste (`diagnose_fit`) + criterio de
    validacion cruzada espacial.

Las secciones 7-8 (comparativa) siguen siendo placeholders documentados
que completa US-022 — asi se evita un merge conflict masivo sobre el .ipynb
(decision D10).

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
    # --- Seccion 3 — Importancia de features nativa (US-020) -------------
    _md(
        "## 3. Importancia de features nativa\n"
        "\n"
        "Random Forest y XGBoost exponen una medida de importancia de "
        "features sin coste adicional: **Gini/MDI** para RF "
        "(`feature_importances_`) y **gain** para XGBoost "
        "(`Booster.get_score(importance_type='gain')`). Es el primer "
        "diagnostico de interpretabilidad — barato y directo — antes del "
        "analisis SHAP de la seccion 4.\n"
        "\n"
        "Se cargan los **modelos production** persistidos por US-019 "
        "(`artifacts/baseline_{rf,xgb}_v1.joblib`); si los artefactos no "
        "existen el notebook entrena RF/XGB in-notebook con los "
        "hiperparametros base (fallback D8 del plan US-020)."
    ),
    _code(
        "import joblib\n"
        "from pathlib import Path\n"
        "\n"
        "from ml.train.baseline import train_one_model\n"
        "\n"
        "REPORTS_DIR = Path('reports/baseline')\n"
        "REPORTS_DIR.mkdir(parents=True, exist_ok=True)\n"
        "ARTIFACTS = {'rf': Path('artifacts/baseline_rf_v1.joblib'),\n"
        "             'xgb': Path('artifacts/baseline_xgb_v1.joblib')}\n"
        "\n"
        "models = {}\n"
        "for kind, path in ARTIFACTS.items():\n"
        "    if path.exists():\n"
        "        payload = joblib.load(path)\n"
        "        models[kind] = {\n"
        "            'model': payload['model'],\n"
        "            'feature_cols': tuple(payload['feature_cols']),\n"
        "            'source': 'joblib US-019',\n"
        "        }\n"
        "    else:\n"
        "        res = train_one_model(df, model=kind)\n"
        "        models[kind] = {\n"
        "            'model': res.model,\n"
        "            'feature_cols': res.feature_cols,\n"
        "            'source': 'fallback in-notebook (D8)',\n"
        "        }\n"
        "    print(f\"{kind.upper()}: {models[kind]['source']}  |  \"\n"
        "          f\"{len(models[kind]['feature_cols'])} features\")"
    ),
    _code(
        "from ml.eval.interpretability import feature_importance_table\n"
        "\n"
        "importance = {}\n"
        "for kind, bundle in models.items():\n"
        "    table = feature_importance_table(\n"
        "        bundle['model'], kind, bundle['feature_cols']\n"
        "    )\n"
        "    importance[kind] = table\n"
        "    table.write_csv(REPORTS_DIR / f'feature_importance_{kind}.csv')\n"
        "importance['rf'].head(10)"
    ),
    _code(
        "# Barplot top-20 de la importancia nativa por modelo.\n"
        "for kind, table in importance.items():\n"
        "    top20 = table.head(20)\n"
        "    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)\n"
        "    ax.barh(top20['feature'].to_list()[::-1],\n"
        "            top20['importance'].to_list()[::-1],\n"
        "            color='#2c7fb8')\n"
        "    ax.set_xlabel('Importancia (' + ('Gini' if kind == 'rf' else 'gain') + ')')\n"
        "    ax.set_title(f'Importancia nativa top-20 — {kind.upper()}')\n"
        "    fig.tight_layout()\n"
        "    fig.savefig(REPORTS_DIR / f'importance_{kind}_top20.png',\n"
        "                dpi=200, bbox_inches='tight')\n"
        "    plt.show()"
    ),
    # --- Seccion 4 — Analisis SHAP (US-020) ------------------------------
    _md(
        "## 4. Analisis SHAP\n"
        "\n"
        "La importancia nativa de la seccion 3 ordena los features pero no "
        "explica *como* cada uno desplaza la prediccion. **SHAP** "
        "(Lundberg & Lee, 2017) descompone cada prediccion en "
        "contribuciones aditivas por feature con garantias teoricas de "
        "consistencia. Para modelos de arboles se usa `TreeExplainer` "
        "(algoritmo TreeSHAP exacto, CPU).\n"
        "\n"
        "**Detalles de implementacion** (decisiones del plan US-020):\n"
        "\n"
        "- **D6 — subsample**: SHAP corre sobre un subsample estratificado "
        "de ~3.000 parcelas, no sobre las ~85k del dataset; TreeSHAP es "
        "exacto pero O(samples x trees x depth).\n"
        "- **D3 — multiclase**: PASTIS-R tiene 18-20 clases; "
        "`compute_shap_values` normaliza la salida multiclase de SHAP "
        "(lista-por-clase / array 3D) a un tensor "
        "`(n_samples, n_features, n_classes)`.\n"
        "- **D4 — ranking global**: la importancia global SHAP es la media "
        "de `|SHAP|` sobre clases y muestras."
    ),
    _code(
        "from ml.eval.interpretability import (\n"
        "    compute_shap_values,\n"
        "    shap_summary_plot,\n"
        "    shap_dependence_plots,\n"
        "    shap_waterfall_plot,\n"
        ")\n"
        "\n"
        "SHAP_SAMPLE_SIZE = 3000\n"
        "shap_results = {}\n"
        "for kind, bundle in models.items():\n"
        "    shap_results[kind] = compute_shap_values(\n"
        "        bundle['model'], df, kind,\n"
        "        feature_cols=bundle['feature_cols'],\n"
        "        sample_size=SHAP_SAMPLE_SIZE,\n"
        "    )\n"
        "    print(f'{kind.upper()}: tensor SHAP '\n"
        "          f'{shap_results[kind].values.shape}')"
    ),
    _code(
        "# Summary plot (beeswarm/bar) de las top-20 features globales.\n"
        "for kind, result in shap_results.items():\n"
        "    fig = shap_summary_plot(result, df, top_n=20)\n"
        "    fig.savefig(REPORTS_DIR / f'shap_summary_{kind}.png',\n"
        "                dpi=200, bbox_inches='tight')\n"
        "    plt.show()"
    ),
    _code(
        "# Dependence plots de los 5 features mas importantes (RF).\n"
        "dependence = shap_dependence_plots(\n"
        "    shap_results['rf'], df, top_features=5\n"
        ")\n"
        "for idx, (feature_name, fig) in enumerate(dependence, start=1):\n"
        "    fig.savefig(\n"
        "        REPORTS_DIR / f'shap_dependence_{idx}_{feature_name}.png',\n"
        "        dpi=200, bbox_inches='tight',\n"
        "    )\n"
        "    plt.show()"
    ),
    _code(
        "# Waterfall de una prediccion ejemplo por modelo.\n"
        "for kind, result in shap_results.items():\n"
        "    fig = shap_waterfall_plot(result, row=0)\n"
        "    fig.savefig(REPORTS_DIR / f'shap_waterfall_{kind}.png',\n"
        "                dpi=200, bbox_inches='tight')\n"
        "    plt.show()"
    ),
    _md(
        "### 4.1 Dominancia de las dimensiones AlphaEarth\n"
        "\n"
        "Pregunta clave para el Paper Track: de las features mas "
        "influyentes segun SHAP, **¿cuantas son dimensiones del embedding "
        "AlphaEarth** (`dim_00..dim_63`) frente a indices espectrales, "
        "estadisticas temporales o bloques contextuales (S1/SRTM/ERA5)? "
        "`alphaearth_dominance_table` clasifica cada feature en su familia "
        "de origen y cuantifica la dominancia."
    ),
    _code(
        "from ml.eval.interpretability import alphaearth_dominance_table\n"
        "\n"
        "dominance = alphaearth_dominance_table(\n"
        "    shap_results['rf'].global_importance, top_n=20\n"
        ")\n"
        "dominance.write_csv(REPORTS_DIR / 'alphaearth_dominance.csv')\n"
        "dominance"
    ),
    _code(
        "# Conteo por familia y conclusion cuantificada (AC-4).\n"
        "family_counts = (\n"
        "    dominance.group_by('family').len()\n"
        "    .sort('len', descending=True)\n"
        ")\n"
        "n_alphaearth = int(\n"
        "    dominance.filter(pl.col('family') == 'alphaearth').height\n"
        ")\n"
        "top_ae = (\n"
        "    dominance.filter(pl.col('family') == 'alphaearth')['feature']\n"
        "    .to_list()[:3]\n"
        ")\n"
        "print(f'{n_alphaearth}/20 de las top features SHAP son '\n"
        "      f'dimensiones AlphaEarth.')\n"
        "if top_ae:\n"
        "    print(f'Lideran: ' + ', '.join(top_ae))\n"
        "family_counts"
    ),
    # --- Seccion 5 — Conclusiones de feature engineering (US-020) --------
    _md(
        "## 5. Conclusiones de feature engineering\n"
        "\n"
        "Esta seccion **valida o refuta** las decisiones de Feature "
        "Engineering del EPIC 3 (US-018) cruzando los rankings de "
        "interpretabilidad de este notebook con los artefactos de "
        "seleccion de features:\n"
        "\n"
        "- `reports/feature_selection/feature_importance_rf.csv` — "
        "importancia exploratoria de US-018.\n"
        "- `reports/feature_selection/anova_f_scores.csv` — F-scores "
        "univariados de la seleccion.\n"
        "- `reports/feature_selection/selected_features.json` — el "
        "conjunto que US-018 retuvo.\n"
        "\n"
        "El objetivo es responder tres preguntas: (a) ¿las top features "
        "SHAP coinciden con lo que US-018 selecciono?; (b) ¿algun feature "
        "descartado por US-018 aparece importante? (señal de refutacion); "
        "(c) ¿la dominancia AlphaEarth confirma la decision de usar el "
        "embedding como backbone?"
    ),
    _code(
        "# Cruce de las top SHAP con la seleccion de features de US-018.\n"
        "fs_dir = Path('reports/feature_selection')\n"
        "top_shap = set(\n"
        "    shap_results['rf'].global_importance.head(20)['feature'].to_list()\n"
        ")\n"
        "\n"
        "fs_importance_path = fs_dir / 'feature_importance_rf.csv'\n"
        "if fs_importance_path.exists():\n"
        "    fs_importance = pl.read_csv(fs_importance_path)\n"
        "    fs_top = set(fs_importance.head(20)['feature'].to_list())\n"
        "    overlap = top_shap & fs_top\n"
        "    print(f'Solapamiento top-20 SHAP vs top-20 US-018: '\n"
        "          f'{len(overlap)}/20 features.')\n"
        "    print('Comunes:', sorted(overlap))\n"
        "    print('Solo en SHAP (revisar FE):', sorted(top_shap - fs_top))\n"
        "else:\n"
        "    print('reports/feature_selection/feature_importance_rf.csv '\n"
        "          'no disponible — se omite el cruce cuantitativo.')"
    ),
    _md(
        "### 5.1 Hallazgos\n"
        "\n"
        "> _Esta celda se completa al ejecutar el notebook sobre los "
        "modelos production; los numeros concretos del cruce salen de la "
        "celda anterior._\n"
        "\n"
        "Hallazgos esperados (≥3, plan AC-5):\n"
        "\n"
        "1. **Convergencia importance nativa vs SHAP** — los features en "
        "el top de Gini/gain y los del top SHAP coinciden en su mayoria; "
        "discrepancias señalan features con efectos no lineales o "
        "interacciones que SHAP captura mejor que la importancia nativa.\n"
        "2. **Dominancia AlphaEarth** — la fraccion de dimensiones "
        "`dim_NN` en el top-20 SHAP (seccion 4.1) valida o matiza la "
        "decision irrevocable de usar AlphaEarth como backbone: si "
        "dominan, el embedding aporta la mayor parte de la senal; si no, "
        "los indices espectrales y la fenologia siguen siendo "
        "imprescindibles.\n"
        "3. **Validacion del FE de US-018** — si los features que US-018 "
        "selecciono coinciden con el top SHAP, la seleccion se valida; si "
        "un feature descartado aparece arriba, se documenta como señal de "
        "refutacion en `docs/product-backlog/us-020-fe-adjustments.md`.\n"
        "\n"
        "### 5.2 Recomendacion para el feature engineering\n"
        "\n"
        "Si el cruce de la seccion 5 confirma la seleccion de US-018, **no "
        "se requiere ajuste**: el FE del EPIC 3 queda validado por la "
        "interpretabilidad del baseline. Si el cruce refuta alguna "
        "decision (un feature relevante descartado, o ruido retenido en "
        "el top), la recomendacion concreta se registra en "
        "`docs/product-backlog/us-020-fe-adjustments.md` para que el EPIC "
        "5 la incorpore antes de entrenar las arquitecturas de "
        "segmentacion."
    ),
    # --- Seccion 5b (US-021) --------------------------------------------
    _md(
        "## 5b. Curvas de aprendizaje y validacion — diagnostico de "
        "sub/sobreajuste\n"
        "\n"
        "Esta seccion diagnostica si el baseline sub o sobreajusta. Se "
        "usan dos herramientas:\n"
        "\n"
        "- **Curva de aprendizaje**: accuracy de train y de validacion al "
        "crecer el numero de muestras de entrenamiento. Un gap grande "
        "train-val indica sobreajuste; ambas curvas bajas y juntas, "
        "subajuste.\n"
        "- **Curva de validacion**: accuracy frente a un hiperparametro "
        "critico (`max_depth` para RF, `n_estimators` y `learning_rate` "
        "para XGBoost), para localizar la zona de equilibrio.\n"
        "\n"
        "Toda la evaluacion usa el **mismo CV espacial 5-fold** (H3 + "
        "KMeans + buffer 1 km) del resto del notebook — los splits se "
        "materializan en una lista porque `learning_curve` reusa el `cv` "
        "una vez por cada tamano. El criterio de spatial CV esta "
        "documentado en `docs/spatial_cv_baseline.md`."
    ),
    _code(
        "from ml.eval.learning_curves import (\n"
        "    diagnose_fit,\n"
        "    plot_learning_curve,\n"
        "    plot_validation_curve,\n"
        ")\n"
        "from ml.train.baseline import _build_cv_splits\n"
        "\n"
        "# CV espacial materializado (lista de splits posicionales).\n"
        "cv_splits_5b = _build_cv_splits(\n"
        "    df, k_folds=5, buffer_km=1.0, random_state=42\n"
        ")\n"
        "print(f'CV espacial: {len(cv_splits_5b)} folds materializados')"
    ),
    _code(
        "# Curva de aprendizaje RF y XGB (accuracy train/val vs n muestras).\n"
        "from pathlib import Path\n"
        "\n"
        "from ml.train.baseline import build_estimator\n"
        "\n"
        "reports_dir = Path('reports/baseline')\n"
        "reports_dir.mkdir(parents=True, exist_ok=True)\n"
        "curve_train_sizes = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]\n"
        "learning_results = {}\n"
        "for kind in ('rf', 'xgb'):\n"
        "    estimator = build_estimator(kind, {})\n"
        "    lc_result, lc_fig = plot_learning_curve(\n"
        "        estimator, df, cv_splits_5b,\n"
        "        train_sizes=curve_train_sizes,\n"
        "        max_samples=MAX_SAMPLES,\n"
        "    )\n"
        "    learning_results[kind] = lc_result\n"
        "    lc_fig.suptitle(f'Curva de aprendizaje — {kind.upper()}')\n"
        "    lc_fig.savefig(\n"
        "        reports_dir / f'learning_curve_{kind}.png',\n"
        "        dpi=200, bbox_inches='tight',\n"
        "    )\n"
        "    plt.show()"
    ),
    _code(
        "# Diagnostico explicito de sub/sobreajuste por modelo.\n"
        "for kind, lc_result in learning_results.items():\n"
        "    diag = diagnose_fit(lc_result)\n"
        "    print(f'{kind.upper()}: veredicto={diag.verdict}  '\n"
        "          f'gap={diag.gap:.4f}  '\n"
        "          f'train_acc={diag.train_acc_max:.4f}  '\n"
        "          f'val_acc={diag.val_acc_max:.4f}')\n"
        "    print(f'  {diag.explanation}')"
    ),
    _code(
        "# Curva de validacion RF — max_depth.\n"
        "vc_rf, vc_rf_fig = plot_validation_curve(\n"
        "    build_estimator('rf', {}), df, 'max_depth',\n"
        "    [5, 10, 15, 20, 30, None], cv_splits_5b,\n"
        "    max_samples=MAX_SAMPLES,\n"
        ")\n"
        "vc_rf_fig.suptitle('Curva de validacion — RF max_depth')\n"
        "vc_rf_fig.savefig(\n"
        "    reports_dir / 'validation_curve_rf_max_depth.png',\n"
        "    dpi=200, bbox_inches='tight',\n"
        ")\n"
        "plt.show()"
    ),
    _code(
        "# Curva de validacion XGB — n_estimators.\n"
        "vc_xgb_ne, vc_xgb_ne_fig = plot_validation_curve(\n"
        "    build_estimator('xgb', {}), df, 'n_estimators',\n"
        "    [100, 200, 300, 400, 500], cv_splits_5b,\n"
        "    max_samples=MAX_SAMPLES,\n"
        ")\n"
        "vc_xgb_ne_fig.suptitle('Curva de validacion — XGB n_estimators')\n"
        "vc_xgb_ne_fig.savefig(\n"
        "    reports_dir / 'validation_curve_xgb_n_estimators.png',\n"
        "    dpi=200, bbox_inches='tight',\n"
        ")\n"
        "plt.show()"
    ),
    _md(
        "El diagnostico `diagnose_fit` reporta un veredicto explicito "
        "(`overfit` / `underfit` / `good_fit`) con el gap train-val "
        "numerico. El baseline tabular sobre embeddings genericos tiende "
        "a un accuracy modesto: si el veredicto es `good_fit` con "
        "accuracy de validacion baja, el limite es la **capacidad del "
        "modelo**, no el sobreajuste — justificacion directa de por que "
        "el EPIC 5 incorpora arquitecturas temporales (U-TAE, TSViT) con "
        "mayor capacidad."
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
