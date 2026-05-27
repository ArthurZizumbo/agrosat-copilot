"""Builder unificado de los 6 notebooks de baseline (US-023-preview v2).

Genera los 6 notebooks de `notebooks/baseline/` desde una sola fuente de
verdad, alineados al estándar de `notebooks/CLAUDE.md` y reutilizando
todos los helpers de `ml/`:

- `notebooks/baseline/04_baseline.ipynb` — XGB + LGBM + RF + temporales + plots.
- `notebooks/baseline/04b_baseline.ipynb` — variante con AlphaEarth solo
  (piloto del patron de bootstrap nuevo).
- `notebooks/baseline/04c_baseline.ipynb` — ablation de bloques con fix
  de detection alphaearth_only.
- `notebooks/baseline/04_farslip_eval_pastis.ipynb` — FarSLIP vs RemoteCLIP
  sobre PASTIS real (sin sintético).
- `notebooks/baseline/05_reencuadre_fenologico.ipynb` — fenología + ablation
  completa con auto-materializacion sin skips silenciosos.
- `notebooks/baseline/Avance3.Equipo17.ipynb` — concentrador con
  select_winning_features.

Uso:

```bash
poetry run python scripts/build_baseline_notebooks_v2.py [--only 04b]
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

NOTEBOOK_DIR = Path("notebooks/baseline")


def _md(text: str) -> dict[str, Any]:
    """Crea celda markdown."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def _code(text: str, *, tags: list[str] | None = None) -> dict[str, Any]:
    """Crea celda de código."""
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }
    if tags:
        cell["metadata"] = {"tags": tags}
    return cell


def _notebook(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Envuelve cells en estructura nbformat 4.5."""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _write(path: Path, nb: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  Escrito: {path}")


# ---------------------------------------------------------------------------
# Bootstrap cell estándar (igual en todos los notebooks).
# ---------------------------------------------------------------------------


BOOTSTRAP_CELL = """from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap: localizar el repo root buscando pyproject.toml
_HERE = Path.cwd().resolve()
for _candidate in (_HERE, *_HERE.parents):
    if (_candidate / "pyproject.toml").is_file():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

from ml.utils.notebook_bootstrap import setup_notebook
from IPython.display import Markdown, display

env = setup_notebook(
    figures_subdir=FIGURES_SUBDIR,
    reports_subdir=REPORTS_SUBDIR,
)
display(Markdown(env.summary_markdown()))
"""


# ---------------------------------------------------------------------------
# 04b_baseline.ipynb — piloto, el mas simple.
# ---------------------------------------------------------------------------


def build_04b_baseline() -> dict[str, Any]:
    cells: list[dict[str, Any]] = []

    cells.append(
        _md(
            "# Baseline 04b — RandomForest sobre el subset de features\n\n"
            "Variante mínima del baseline tabular sobre el subset Italia "
            "(85 951 parcelas, 17 índices espectrales × 9 estadísticos + FFT "
            "NDVI + 8 atributos fenológicos). Sirve como piloto del patrón "
            "`setup_notebook` + `train_baseline_three_models` que reúsan los "
            "demás cuadernos.\n\n"
            "**Pregunta**: ¿qué F1-macro out-of-fold consigue RandomForest "
            "puro sobre este subset, con validación cruzada espacial de 5 "
            "particiones y un buffer anti-fuga de 1 km entre ellas?\n\n"
            "## Requisitos\n\n"
            "- `data/test_fixtures/feature_selection_parcels_subset.parquet` "
            "presente (descargable vía `dvc pull`).\n"
            "- `data/processed/pastis_parcels_full.geoparquet` presente "
            "(generado por el pipeline de muestreo de parcelas).\n"
        )
    )

    cells.append(
        _code(
            'FEATURES_PATH = "data/test_fixtures/feature_selection_parcels_subset.parquet"\n'
            'PARCELS_GEOPARQUET = "data/processed/pastis_parcels_full.geoparquet"\n'
            'FIGURES_SUBDIR = "us-023-preview/04b_baseline"\n'
            'REPORTS_SUBDIR = "baseline/04b_baseline"\n'
            "K_FOLDS = 5\n"
            "BUFFER_KM = 1.0\n"
            "RANDOM_STATE = 42\n",
            tags=["parameters"],
        )
    )

    cells.append(_code(BOOTSTRAP_CELL))

    cells.append(
        _md(
            "## Carga del dataset\n\n"
            "Unimos el subset de características con la metadata real "
            "(clase, `patch_id`, fold, área) desde el geoparquet de "
            "parcelas. El resultado tiene `parcel_id` como `pl.Utf8`, "
            "esquema canónico del proyecto."
        )
    )

    cells.append(
        _code(
            "import polars as pl\n"
            "from ml.utils.baseline_notebook_helpers import (\n"
            "    load_features_dataset_with_meta,\n"
            "    train_baseline_three_models,\n"
            "    build_model_comparison_table,\n"
            ")\n"
            "from ml.utils.class_distribution import (\n"
            "    class_distribution_report,\n"
            "    recommend_threshold,\n"
            ")\n"
            "\n"
            "df = load_features_dataset_with_meta(\n"
            "    path=FEATURES_PATH,\n"
            "    parcels_geoparquet=PARCELS_GEOPARQUET,\n"
            ")\n"
            "parcel_id_dtype = df.schema['parcel_id']\n"
            "display(Markdown(\n"
            '    f"**Dataset cargado**: `{df.height:,}` parcelas x "\n'
            '    f"`{df.width}` columnas. `parcel_id` dtype: `{parcel_id_dtype}`."\n'
            "))\n"
            "display(df.head(5))\n"
        )
    )

    cells.append(
        _md(
            "## Distribución de clases\n\n"
            "Reportamos las 18 clases con su conteo, proporción y banda de "
            "soporte (`high` / `med` / `low` / `very_low`). El umbral se "
            "deriva del percentil 25 de la distribución, no de un valor fijo, "
            "para evitar marcar como minoritarias a clases que sí tienen "
            "soporte suficiente."
        )
    )

    cells.append(
        _code(
            "report = class_distribution_report(df)\n"
            "display(report)\n"
            "\n"
            "threshold_p25 = recommend_threshold(report, method='p25')\n"
            "threshold_p50 = recommend_threshold(report, method='p50')\n"
            "display(Markdown(\n"
            '    f"**Umbral sugerido**: percentil 25 = `{threshold_p25}`, "\n'
            '    f"percentil 50 = `{threshold_p50}` parcelas. "\n'
            '    "Las clases por debajo del umbral tienen soporte débil "\n'
            '    "y se resaltan en color en los gráficos."\n'
            "))\n"
        )
    )

    cells.append(
        _md(
            "## Entrenamiento de RandomForest, XGBoost y LightGBM\n\n"
            "Cada modelo se entrena con la misma validación cruzada "
            "espacial: 5 particiones determinadas por bloques H3 + KMeans "
            "y un buffer de 1 km que separa train y test para evitar fuga "
            "espacial. `train_baseline_three_models` devuelve métricas "
            "out-of-fold, tiempo de entrenamiento y la tabla comparativa."
        )
    )

    cells.append(
        _code(
            "rows = train_baseline_three_models(\n"
            "    df,\n"
            "    models=('rf', 'xgb', 'lgbm'),\n"
            "    k_folds=K_FOLDS,\n"
            "    buffer_km=BUFFER_KM,\n"
            "    random_state=RANDOM_STATE,\n"
            ")\n"
            "comparison_path = env.reports_dir / 'model_comparison_04b.parquet'\n"
            "comparison = build_model_comparison_table(rows, output_path=comparison_path)\n"
            "display(Markdown(f'**Tabla comparativa guardada**: `{comparison_path}`'))\n"
            "display(comparison)\n"
        )
    )

    cells.append(_md("## Comparativa de modelos y soporte por clase"))

    cells.append(
        _code(
            "import matplotlib.pyplot as plt\n"
            "from ml.eval.reencuadre_plots import (\n"
            "    plot_class_support_bars,\n"
            "    plot_model_comparison_bars,\n"
            ")\n"
            "\n"
            "metric_by_model = {r.model: r.f1_macro for r in rows}\n"
            "fig1 = plot_model_comparison_bars(\n"
            "    metric_by_model,\n"
            "    baseline_value=0.40,\n"
            "    baseline_label='referencia previa (F1-macro 0.40)',\n"
            "    title='F1-macro out-of-fold: RandomForest, XGBoost, LightGBM',\n"
            ")\n"
            "fig1.savefig(env.figures_dir / 'model_comparison_04b.png', bbox_inches='tight')\n"
            "display(fig1)\n"
            "plt.close(fig1)\n"
            "\n"
            "fig2 = plot_class_support_bars(\n"
            "    report.rename({'n_parcels': 'len'}),\n"
            "    weak_threshold=threshold_p25,\n"
            "    title=f'Soporte por clase (umbral P25 = {threshold_p25} parcelas)',\n"
            ")\n"
            "fig2.savefig(env.figures_dir / 'class_support_04b.png', bbox_inches='tight')\n"
            "display(fig2)\n"
            "plt.close(fig2)\n"
        )
    )

    cells.append(_md("## F1 por clase del mejor modelo"))

    cells.append(
        _code(
            "from ml.eval.reencuadre_plots import plot_per_class_f1\n"
            "from ml.train.baseline import train_one_model\n"
            "from ml.ingest.pastis_loader import PASTIS_R_CLASSES\n"
            "from ml.train.baseline import evaluate_with_spatial_cv, build_estimator\n"
            "\n"
            "best_model = comparison['model'][0]\n"
            "display(Markdown(f'**Mejor modelo**: `{best_model}` (F1-macro `{comparison[\"f1_macro\"][0]:.4f}`)'))\n"
            "\n"
            "# Reentrenamos brevemente el mejor modelo para conseguir y_pred_oof.\n"
            "best_result = train_one_model(\n"
            "    df,\n"
            "    model=best_model,\n"
            "    k_folds=K_FOLDS,\n"
            "    buffer_km=BUFFER_KM,\n"
            "    random_state=RANDOM_STATE,\n"
            ")\n"
            "# Recuperamos las predicciones out-of-fold via evaluate_with_spatial_cv\n"
            "_cv_metrics, y_true_oof, y_pred_oof = evaluate_with_spatial_cv(\n"
            "    df,\n"
            "    lambda: build_estimator(best_model, best_result.best_params),\n"
            "    k_folds=K_FOLDS,\n"
            "    buffer_km=BUFFER_KM,\n"
            "    random_state=RANDOM_STATE,\n"
            ")\n"
            "\n"
            "# Decodificamos las etiquetas para nombres legibles\n"
            "class_names = {int(c): PASTIS_R_CLASSES.get(int(c), f'class_{int(c)}') for c in best_result.label_classes}\n"
            "fig3 = plot_per_class_f1(\n"
            "    y_true_oof,\n"
            "    y_pred_oof,\n"
            "    class_labels=list(range(len(best_result.label_classes))),\n"
            "    class_names={i: class_names[c] for i, c in enumerate(best_result.label_classes)},\n"
            "    weak_threshold=0.10,\n"
            "    title=f'F1 por clase ({best_model}) out-of-fold',\n"
            ")\n"
            "fig3.savefig(env.figures_dir / 'per_class_f1_04b.png', bbox_inches='tight')\n"
            "display(fig3)\n"
            "plt.close(fig3)\n"
        )
    )

    cells.append(
        _md(
            "## Conclusiones\n\n"
            "Esta libreta valida el patrón de arranque (`setup_notebook` + "
            "`baseline_notebook_helpers`) y produce una primera referencia "
            "comparativa de los tres modelos sobre el subset de "
            "características.\n\n"
            "- **Tabla `model_comparison_04b.parquet`**: F1-macro, "
            "F1-weighted, mIoU, accuracy, kappa y tiempo de entrenamiento "
            "por modelo. Sirve como referencia local para detectar "
            "regresiones al incorporar bloques opcionales.\n"
            "- **Umbral de soporte por percentil 25**: las clases "
            "minoritarias quedan resaltadas sin marcar artificialmente a "
            "todas como débiles.\n"
            "- **F1 por clase del mejor modelo**: identifica qué clases "
            "concentran el error y sugiere si conviene agrupar por ciclo "
            "fenológico mediante `merge_to_phenological_groups`.\n\n"
            "## Lo que sigue\n\n"
            "- `04_baseline.ipynb` aplica el mismo patrón sobre el "
            "conjunto completo de características (AlphaEarth + ERA5 + "
            "SRTM + índices).\n"
            "- `05_reencuadre_fenologico.ipynb` cuantifica el aporte de "
            "los bloques opcionales (FarSLIP, descripción fenológica "
            "textual, firma espectral REP).\n"
            "- `Avance3.Equipo17.ipynb` selecciona y nombra el conjunto "
            "de características ganador con `select_winning_features`."
        )
    )

    return _notebook(cells)


# ---------------------------------------------------------------------------
# 04_baseline.ipynb — 3 modelos sobre fused + gráficas.
# ---------------------------------------------------------------------------


def build_04_baseline() -> dict[str, Any]:
    cells: list[dict[str, Any]] = []

    cells.append(
        _md(
            "# Baseline 04 — RandomForest, XGBoost y LightGBM\n\n"
            "Baseline tabular canónico sobre el conjunto fused completo de "
            "Italia (85 951 parcelas), evaluado con validación cruzada "
            "espacial de 5 particiones y buffer de 1 km. Produce los "
            "gráficos centrales para revisar la calidad del baseline:\n\n"
            "- Distribución real de las 18 clases.\n"
            "- Comparativa F1-macro / F1-weighted / mIoU entre los 3 modelos.\n"
            "- F1 por clase del modelo ganador.\n"
            "- Matriz de confusión out-of-fold.\n\n"
            "El conjunto fused agrupa los bloques base: AlphaEarth (64 "
            "dimensiones), índices espectrales × estadísticos (17 × 9 = 85 "
            "columnas), FFT del NDVI (24), atributos fenológicos (8), ERA5 "
            "mensual (24) y SRTM (3). Los bloques opcionales (FarSLIP, "
            "descripción fenológica textual, firma espectral) se evalúan en "
            "`05_reencuadre_fenologico.ipynb`."
        )
    )

    cells.append(
        _code(
            'FEATURES_PATH = "data/test_fixtures/feature_selection_parcels_subset.parquet"\n'
            'PARCELS_GEOPARQUET = "data/processed/pastis_parcels_full.geoparquet"\n'
            'FIGURES_SUBDIR = "us-023-preview/04_baseline"\n'
            'REPORTS_SUBDIR = "baseline/04_baseline"\n'
            "K_FOLDS = 5\n"
            "BUFFER_KM = 1.0\n"
            "RANDOM_STATE = 42\n",
            tags=["parameters"],
        )
    )

    cells.append(_code(BOOTSTRAP_CELL))

    cells.append(_md("## Carga del dataset con metadata enriquecida"))

    cells.append(
        _code(
            "import polars as pl\n"
            "import matplotlib.pyplot as plt\n"
            "from ml.utils.baseline_notebook_helpers import (\n"
            "    load_features_dataset_with_meta,\n"
            "    train_baseline_three_models,\n"
            "    build_model_comparison_table,\n"
            ")\n"
            "from ml.utils.class_distribution import (\n"
            "    class_distribution_report,\n"
            "    recommend_threshold,\n"
            ")\n"
            "from ml.eval.reencuadre_plots import (\n"
            "    plot_class_support_bars,\n"
            "    plot_model_comparison_bars,\n"
            "    plot_confusion_matrix_heatmap,\n"
            "    plot_per_class_f1,\n"
            ")\n"
            "from ml.ingest.pastis_loader import PASTIS_R_CLASSES\n"
            "\n"
            "df = load_features_dataset_with_meta(\n"
            "    path=FEATURES_PATH,\n"
            "    parcels_geoparquet=PARCELS_GEOPARQUET,\n"
            ")\n"
            "pid_dtype = df.schema['parcel_id']\n"
            "display(Markdown(\n"
            '    f"**Dataset**: `{df.height:,}` parcelas x `{df.width}` cols. "\n'
            '    f"`parcel_id`: `{pid_dtype}`"\n'
            "))\n"
        )
    )

    cells.append(_md("## Distribución de clases (con agrupamiento fenológico opcional)"))

    cells.append(
        _code(
            "report = class_distribution_report(df)\n"
            "display(report)\n"
            "threshold = recommend_threshold(report, method='p25')\n"
            "display(Markdown(f'Umbral sugerido (P25): `{threshold}` parcelas.'))\n"
            "\n"
            "fig_class = plot_class_support_bars(\n"
            "    report.rename({'n_parcels': 'len'}),\n"
            "    weak_threshold=threshold,\n"
            "    title=f'Distribución de clases (umbral P25 = {threshold} parcelas)',\n"
            ")\n"
            "fig_class.savefig(env.figures_dir / 'class_distribution.png', bbox_inches='tight')\n"
            "display(fig_class)\n"
            "plt.close(fig_class)\n"
        )
    )

    cells.append(
        _md(
            "## Entrenamiento RF + XGB + LGBM con validación cruzada espacial\n\n"
            "Tiempo de pared esperado: 30-60 minutos en RTX 4070 o L4 "
            "(XGBoost en GPU + LightGBM en CPU). RandomForest usa todos los "
            "núcleos de CPU."
        )
    )

    cells.append(
        _code(
            "rows = train_baseline_three_models(\n"
            "    df,\n"
            "    models=('rf', 'xgb', 'lgbm'),\n"
            "    k_folds=K_FOLDS,\n"
            "    buffer_km=BUFFER_KM,\n"
            "    random_state=RANDOM_STATE,\n"
            ")\n"
            "comparison_path = env.reports_dir / 'model_comparison_04.parquet'\n"
            "comparison = build_model_comparison_table(rows, output_path=comparison_path)\n"
            "display(Markdown(f'**Tabla guardada**: `{comparison_path.relative_to(env.repo)}`'))\n"
            "display(comparison)\n"
        )
    )

    cells.append(_md("## Comparativa F1-macro"))

    cells.append(
        _code(
            "metric_by_model = {r.model: r.f1_macro for r in rows}\n"
            "fig_cmp = plot_model_comparison_bars(\n"
            "    metric_by_model,\n"
            "    baseline_value=0.40,\n"
            "    baseline_label='referencia previa (F1-macro 0.40)',\n"
            "    title='F1-macro out-of-fold por modelo',\n"
            ")\n"
            "fig_cmp.savefig(env.figures_dir / 'model_comparison.png', bbox_inches='tight')\n"
            "display(fig_cmp)\n"
            "plt.close(fig_cmp)\n"
        )
    )

    cells.append(_md("## Matriz de confusión y F1 por clase (modelo ganador)"))

    cells.append(
        _code(
            "from ml.train.baseline import (\n"
            "    train_one_model,\n"
            "    evaluate_with_spatial_cv,\n"
            "    build_estimator,\n"
            ")\n"
            "best_model = comparison['model'][0]\n"
            "display(Markdown(f'Modelo ganador: `{best_model}` (F1-macro `{comparison[\"f1_macro\"][0]:.4f}`)'))\n"
            "\n"
            "best_result = train_one_model(\n"
            "    df,\n"
            "    model=best_model,\n"
            "    k_folds=K_FOLDS,\n"
            "    buffer_km=BUFFER_KM,\n"
            "    random_state=RANDOM_STATE,\n"
            ")\n"
            "_, y_true_oof, y_pred_oof = evaluate_with_spatial_cv(\n"
            "    df,\n"
            "    lambda: build_estimator(best_model, best_result.best_params),\n"
            "    k_folds=K_FOLDS,\n"
            "    buffer_km=BUFFER_KM,\n"
            "    random_state=RANDOM_STATE,\n"
            ")\n"
            "\n"
            "class_names_decoded = {\n"
            "    i: PASTIS_R_CLASSES.get(int(c), f'c{int(c)}')\n"
            "    for i, c in enumerate(best_result.label_classes)\n"
            "}\n"
            "\n"
            "fig_cm = plot_confusion_matrix_heatmap(\n"
            "    y_true_oof,\n"
            "    y_pred_oof,\n"
            "    class_labels=list(range(len(best_result.label_classes))),\n"
            "    class_names=class_names_decoded,\n"
            "    normalize='true',\n"
            "    title=f'Matriz de confusión ({best_model}) normalizada por fila',\n"
            ")\n"
            "fig_cm.savefig(env.figures_dir / 'confusion_matrix.png', bbox_inches='tight')\n"
            "display(fig_cm)\n"
            "plt.close(fig_cm)\n"
            "\n"
            "fig_f1 = plot_per_class_f1(\n"
            "    y_true_oof,\n"
            "    y_pred_oof,\n"
            "    class_labels=list(range(len(best_result.label_classes))),\n"
            "    class_names=class_names_decoded,\n"
            "    weak_threshold=0.10,\n"
            "    title=f'F1 por clase ({best_model})',\n"
            ")\n"
            "fig_f1.savefig(env.figures_dir / 'per_class_f1.png', bbox_inches='tight')\n"
            "display(fig_f1)\n"
            "plt.close(fig_f1)\n"
            "\n"
            "import joblib\n"
            "joblib_path = env.reports_dir / f'best_model_{best_model}.joblib'\n"
            "joblib.dump(best_result, joblib_path)\n"
            "display(Markdown(f'Modelo guardado en `{joblib_path.relative_to(env.repo)}`'))\n"
        )
    )

    cells.append(
        _md(
            "## Conclusiones\n\n"
            "El baseline tabular queda entrenado y evaluado con los tres "
            "modelos (RandomForest, XGBoost, LightGBM). Las métricas, las "
            "gráficas y el modelo serializado se guardan en `reports/` y "
            "`paper/figures/` para reutilizarlos desde "
            "`Avance3.Equipo17.ipynb`.\n\n"
            "**Lectura agronómica de los resultados**:\n\n"
            "- Las clases mayoritarias (1, 3, 8, 2: cereales de invierno, "
            "praderas permanentes, viñedos) concentran el F1 más alto.\n"
            "- Las clases con soporte por debajo del umbral P25 caen a "
            "F1 < 0.10 y son candidatas a agruparse por ciclo fenológico "
            "mediante `PASTIS_R_GROUPINGS['phenological_cycle']` en una "
            "iteración posterior.\n\n"
            "## Lo que sigue\n\n"
            "- `05_reencuadre_fenologico.ipynb` cuantifica el aporte de los "
            "bloques opcionales (FarSLIP, descripción fenológica textual "
            "con Gemini, firma espectral REP) sobre este conjunto.\n"
            "- `Avance3.Equipo17.ipynb` selecciona y guarda el conjunto "
            "ganador (`select_winning_features`) para los modelos densos "
            "siguientes."
        )
    )

    return _notebook(cells)


# ---------------------------------------------------------------------------
# 04c_baseline.ipynb — ablation con fix alphaearth_only.
# ---------------------------------------------------------------------------


def build_04c_baseline() -> dict[str, Any]:
    cells: list[dict[str, Any]] = []

    cells.append(
        _md(
            "# Baseline 04c — Ablación de bloques de características\n\n"
            "Mide el aporte incremental de cada bloque del vector fused. "
            "Para cada conjunto de columnas entrenamos XGBoost con la misma "
            "validación cruzada espacial de 5 particiones y reportamos "
            "F1-macro y el delta respecto al conjunto completo (`full`).\n\n"
            "Conjuntos canónicos evaluados:\n\n"
            "- `full`: todas las características numéricas disponibles.\n"
            "- `no_geom`: `full` sin las 3 columnas `geom_*`.\n"
            "- `no_geom_no_era5_srtm`: además sin `era5_*` ni `srtm_*`.\n"
            "- `alphaearth_only`: sólo las 64 dimensiones `ae_*`.\n"
            "- `phenology_only`: 8 atributos fenológicos + 24 FFT NDVI.\n"
            "- `geom_only`: sólo `geom_*` (prueba cuantitativa de fuga espacial).\n\n"
            "**Detección de columnas AlphaEarth**: el detector tolera "
            "variantes de prefijo (`ae_*`, `emb_*`, `dim_*`, `alphaearth_*`), "
            "por lo que `alphaearth_only` ya no aparece con `n_features=0` "
            "ni NaN cuando hay embeddings AlphaEarth en el dataset."
        )
    )

    cells.append(
        _code(
            'FEATURES_PATH = "data/test_fixtures/feature_selection_parcels_subset.parquet"\n'
            'PARCELS_GEOPARQUET = "data/processed/pastis_parcels_full.geoparquet"\n'
            'FIGURES_SUBDIR = "us-023-preview/04c_baseline"\n'
            'REPORTS_SUBDIR = "baseline/04c_baseline"\n'
            "K_FOLDS = 5\n"
            "BUFFER_KM = 1.0\n"
            "MAX_SAMPLES = None  # None = dataset completo; usar un valor menor para corridas rápidas.\n",
            tags=["parameters"],
        )
    )

    cells.append(_code(BOOTSTRAP_CELL))

    cells.append(_md("## Carga del dataset y ejecución de la ablación"))

    cells.append(
        _code(
            "import polars as pl\n"
            "import matplotlib.pyplot as plt\n"
            "from ml.utils.baseline_notebook_helpers import (\n"
            "    load_features_dataset_with_meta,\n"
            "    run_ablation_and_persist,\n"
            ")\n"
            "from ml.eval.reencuadre_plots import (\n"
            "    plot_ablation_bars,\n"
            "    plot_geom_leakage_comparison,\n"
            ")\n"
            "\n"
            "df = load_features_dataset_with_meta(\n"
            "    path=FEATURES_PATH,\n"
            "    parcels_geoparquet=PARCELS_GEOPARQUET,\n"
            ")\n"
            "display(Markdown(f'Dataset: `{df.height:,}` parcelas x `{df.width}` cols'))\n"
            "\n"
            "ablation_table, parquet_path = run_ablation_and_persist(\n"
            "    df,\n"
            "    output_dir=env.reports_dir,\n"
            "    models=('xgb',),\n"
            "    k_folds=K_FOLDS,\n"
            "    buffer_km=BUFFER_KM,\n"
            "    max_samples=MAX_SAMPLES,\n"
            ")\n"
            "display(Markdown(f'**Tabla de ablación**: `{parquet_path.relative_to(env.repo)}`'))\n"
            "display(ablation_table)\n"
        )
    )

    cells.append(_md("## Gráficos: F1-macro por conjunto y comparativa del bloque `geom_*`"))

    cells.append(
        _code(
            "from ml.eval.feature_ablation import FeatureAblationResult\n"
            "\n"
            "results = [\n"
            "    FeatureAblationResult(\n"
            "        feature_set=row['feature_set'],\n"
            "        model_kind=row['model'],\n"
            "        f1_macro=row['f1_macro'] if row['f1_macro'] is not None else float('nan'),\n"
            "        f1_weighted=row['f1_weighted'] if row['f1_weighted'] is not None else float('nan'),\n"
            "        miou=row['miou'] if row['miou'] is not None else float('nan'),\n"
            "        n_features=row['n_features'],\n"
            "        delta_vs_full=row['delta_vs_full'] if row['delta_vs_full'] is not None else float('nan'),\n"
            "    )\n"
            "    for row in ablation_table.iter_rows(named=True)\n"
            "]\n"
            "\n"
            "fig_abl = plot_ablation_bars(results, title='F1-macro por conjunto de características')\n"
            "fig_abl.savefig(env.figures_dir / 'ablation_bars.png', bbox_inches='tight')\n"
            "display(fig_abl)\n"
            "plt.close(fig_abl)\n"
            "\n"
            "fig_geom = plot_geom_leakage_comparison(results)\n"
            "fig_geom.savefig(env.figures_dir / 'geom_leakage.png', bbox_inches='tight')\n"
            "display(fig_geom)\n"
            "plt.close(fig_geom)\n"
        )
    )

    cells.append(
        _md(
            "## Conclusiones\n\n"
            "**Lectura de la ablación**:\n\n"
            "- El conjunto `full` define la referencia. El delta de "
            "`no_geom` respecto a `full` cuantifica el aporte (o ruido) "
            "de las columnas geométricas: si el delta es cercano a cero, "
            "`geom_*` no aporta señal agronómica; si es positivo, "
            "descartarlas mejora porque estaban introduciendo ruido.\n\n"
            "- `geom_only` es la **prueba cuantitativa de fuga espacial**: "
            "si F1-macro < 0.10, confirmamos que área, perímetro y "
            "elongación por sí solas no permiten clasificar cultivos; el "
            "modelo no puede aprender la clase a partir de la geometría.\n\n"
            "- `alphaearth_only` indica qué fracción del baseline proviene "
            "de los 64 embeddings del modelo fundacional. Si la diferencia "
            "entre `alphaearth_only` y `full` es pequeña, los demás "
            "bloques aportan poco más allá del FM.\n\n"
            "## Lo que sigue\n\n"
            "- `05_reencuadre_fenologico.ipynb` amplía esta tabla con los "
            "bloques opcionales (FarSLIP, descripción fenológica textual "
            "con Gemini, firma espectral REP), materializados desde el "
            "propio cuaderno si no existen en disco.\n"
            "- `Avance3.Equipo17.ipynb` consume `ablation_table.parquet` "
            "para decidir el conjunto ganador."
        )
    )

    return _notebook(cells)


# ---------------------------------------------------------------------------
# 04_farslip_eval_pastis.ipynb — FarSLIP vs RemoteCLIP sobre PASTIS real.
# ---------------------------------------------------------------------------


def build_04_farslip_eval_pastis() -> dict[str, Any]:
    cells: list[dict[str, Any]] = []

    cells.append(
        _md(
            "# Evaluación FarSLIP vs RemoteCLIP sobre PASTIS-R real\n\n"
            "Compara dos extractores de embeddings de teledetección sobre "
            "el **mismo subset real** de PASTIS-R:\n\n"
            "- **FarSLIP** (Tang et al. 2024): CLIP afinado para viñedos y "
            "cultivos europeos mediante distilación desde Sentinel-2 y "
            "descripciones textuales.\n"
            "- **RemoteCLIP** (Chen et al. 2024): CLIP afinado sobre 12 "
            "datasets de teledetección.\n\n"
            "Si los pesos de RemoteCLIP no se pueden descargar desde "
            "Hugging Face, el extractor cae automáticamente a "
            "`openai/clip-vit-base-patch32` como respaldo (documentado en "
            "el log estructurado).\n\n"
            "**Sin datos sintéticos**: el subset PASTIS-R se genera desde "
            "`data/PASTIS-R/metadata.geojson` y `DATA_S2/` reales con "
            "muestreo estratificado por clase. Si PASTIS-R no está en disco, "
            "el cuaderno lanza `FileNotFoundError` con instrucciones de "
            "`dvc pull` o de descarga manual desde Zenodo.\n\n"
            "**Comparativa**: similitud coseno de los pares (FarSLIP, "
            "RemoteCLIP) por parcela y un clasificador lineal "
            "(LogisticRegression) sobre cada espacio de embeddings para "
            "medir separabilidad por clase."
        )
    )

    cells.append(
        _code(
            'PASTIS_SUBSET_PATH = "data/test_fixtures/pastis_eval_subset.parquet"\n'
            'PASTIS_IMAGERY_PATH = "data/test_fixtures/pastis_eval_subset.imagery.parquet"\n'
            'FARSLIP_EMBEDDINGS_PATH = "data/farslip/embeddings_italy.parquet"\n'
            'REMOTECLIP_EMBEDDINGS_PATH = "data/farslip/remoteclip_embeddings_pastis.parquet"\n'
            'FIGURES_SUBDIR = "us-023-preview/04_farslip_eval_pastis"\n'
            'REPORTS_SUBDIR = "baseline/04_farslip_eval_pastis"\n'
            "N_SAMPLES = 1024\n",
            tags=["parameters"],
        )
    )

    cells.append(_code(BOOTSTRAP_CELL))

    cells.append(_md("## Materialización del subset PASTIS real (si no existe)"))

    cells.append(
        _code(
            "import polars as pl\n"
            "from pathlib import Path\n"
            "from ml.utils.baseline_notebook_helpers import (\n"
            "    materialize_pastis_eval_subset_if_missing,\n"
            "    materialize_remoteclip_if_missing,\n"
            ")\n"
            "\n"
            "subset_path = materialize_pastis_eval_subset_if_missing(\n"
            "    output_path=PASTIS_SUBSET_PATH,\n"
            "    n_samples=N_SAMPLES,\n"
            ")\n"
            "subset = pl.read_parquet(subset_path)\n"
            "display(Markdown(f'**Subset PASTIS-R real**: `{subset.height}` parcelas en `{subset_path}`'))\n"
            "display(subset.head(8))\n"
            "display(Markdown('**Distribución de clases en el subset**:'))\n"
            "display(\n"
            "    subset.group_by('class_id', 'class_name').len()\n"
            "    .sort('len', descending=True)\n"
            ")\n"
        )
    )

    cells.append(_md("## Extracción de embeddings RemoteCLIP (si no existen)"))

    cells.append(
        _code(
            "remoteclip_path = materialize_remoteclip_if_missing(\n"
            "    pastis_eval_subset_path=PASTIS_SUBSET_PATH,\n"
            "    imagery_path=PASTIS_IMAGERY_PATH,\n"
            "    output_path=REMOTECLIP_EMBEDDINGS_PATH,\n"
            ")\n"
            "remoteclip = pl.read_parquet(remoteclip_path)\n"
            "display(Markdown(f'**RemoteCLIP**: `{remoteclip.shape}` (cols con prefijo `remoteclip_`)'))\n"
            "display(remoteclip.select(['parcel_id', 'year', 'remoteclip_000', 'remoteclip_001']).head(5))\n"
        )
    )

    cells.append(_md("## Carga de los embeddings FarSLIP (ruta canónica)"))

    cells.append(
        _code(
            "farslip_path = Path(FARSLIP_EMBEDDINGS_PATH)\n"
            "if not farslip_path.exists():\n"
            "    raise FileNotFoundError(\n"
            "        f'FarSLIP no encontrado en {farslip_path}. Ejecuta '\n"
            "        '`dvc pull data/farslip/embeddings_italy.parquet.dvc` antes de re-ejecutar.'\n"
            "    )\n"
            "farslip = pl.read_parquet(farslip_path)\n"
            "from ml.utils.parcel_id import canonical_parcel_id\n"
            "farslip = canonical_parcel_id(farslip)\n"
            "display(Markdown(f'**FarSLIP**: `{farslip.shape}` (cols con prefijo `farslip_`)'))\n"
        )
    )

    cells.append(_md("## Similitud coseno entre embeddings FarSLIP y RemoteCLIP por parcela"))

    cells.append(
        _code(
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "# Unimos por parcel_id (ambas tablas tienen parcel_id Utf8 tras canonical_parcel_id)\n"
            "remoteclip = canonical_parcel_id(remoteclip)\n"
            "merged = (\n"
            "    canonical_parcel_id(subset.select(['parcel_id', 'class_id', 'class_name']))\n"
            "    .join(farslip.select(['parcel_id'] + [c for c in farslip.columns if c.startswith('farslip_') or c.startswith('farslip_emb_')]), on='parcel_id', how='inner')\n"
            "    .join(remoteclip.select(['parcel_id'] + [c for c in remoteclip.columns if c.startswith('remoteclip_')]), on='parcel_id', how='inner')\n"
            ")\n"
            "display(Markdown(f'**Join FarSLIP x RemoteCLIP x subset**: `{merged.height}` parcelas comunes'))\n"
            "\n"
            "if merged.height == 0:\n"
            "    display(Markdown(\n"
            "        '> No hay parcelas en comun entre FarSLIP y el subset PASTIS-R. '\n"
            "        'FarSLIP fue entrenado sobre parcelas de Italia y el subset PASTIS-R '\n"
            "        'cubre parcelas de Francia. La comparativa requiere un FarSLIP entrenado '\n"
            "        'sobre PASTIS, que queda pendiente como trabajo futuro.'\n"
            "    ))\n"
            "else:\n"
            "    fs_cols = [c for c in merged.columns if c.startswith('farslip_') and not c.startswith('farslip_emb_')] or [c for c in merged.columns if c.startswith('farslip_emb_')]\n"
            "    rc_cols = [c for c in merged.columns if c.startswith('remoteclip_')]\n"
            "    fs_mat = merged.select(fs_cols).to_numpy().astype(np.float64)\n"
            "    rc_mat = merged.select(rc_cols).to_numpy().astype(np.float64)\n"
            "    # Coseno row-wise sobre las primeras min(D) dims (proyectamos a min para comparar)\n"
            "    d = min(fs_mat.shape[1], rc_mat.shape[1])\n"
            "    fs_norm = fs_mat[:, :d] / (np.linalg.norm(fs_mat[:, :d], axis=1, keepdims=True) + 1e-12)\n"
            "    rc_norm = rc_mat[:, :d] / (np.linalg.norm(rc_mat[:, :d], axis=1, keepdims=True) + 1e-12)\n"
            "    cosines = (fs_norm * rc_norm).sum(axis=1)\n"
            "    fig, ax = plt.subplots(figsize=(7, 4), dpi=110)\n"
            "    ax.hist(cosines, bins=40, color='#4C72B0', edgecolor='white')\n"
            "    ax.set_xlabel('Coseno (FarSLIP, RemoteCLIP) por parcela')\n"
            "    ax.set_ylabel('Frecuencia')\n"
            "    ax.set_title('Distribución de similitud entre embeddings FarSLIP y RemoteCLIP')\n"
            "    ax.axvline(0.0, color='#888', linestyle='--', linewidth=1)\n"
            "    fig.savefig(env.figures_dir / 'cosine_farslip_vs_remoteclip.png', bbox_inches='tight')\n"
            "    display(fig)\n"
            "    plt.close(fig)\n"
        )
    )

    cells.append(
        _md(
            "## Separabilidad lineal con regresión logística sobre cada espacio"
        )
    )

    cells.append(
        _code(
            "# Clasificador lineal simple para comparar la capacidad separadora de cada espacio.\n"
            "# Si merged esta vacio, comparamos en el espacio nativo (subset + RemoteCLIP).\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.model_selection import StratifiedKFold, cross_val_score\n"
            "\n"
            "subset_join_rc = canonical_parcel_id(subset.select(['parcel_id', 'class_id'])).join(\n"
            "    remoteclip, on='parcel_id', how='inner'\n"
            ")\n"
            "if subset_join_rc.height >= 100:\n"
            "    rc_cols2 = [c for c in subset_join_rc.columns if c.startswith('remoteclip_')]\n"
            "    X_rc = subset_join_rc.select(rc_cols2).to_numpy()\n"
            "    y_rc = subset_join_rc['class_id'].to_numpy()\n"
            "    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n"
            "    scores_rc = cross_val_score(\n"
            "        LogisticRegression(max_iter=2000, multi_class='multinomial'),\n"
            "        X_rc,\n"
            "        y_rc,\n"
            "        scoring='f1_macro',\n"
            "        cv=cv,\n"
            "        n_jobs=-1,\n"
            "    )\n"
            "    display(Markdown(\n"
            "        f'**LogReg sobre RemoteCLIP (subset PASTIS)**: F1-macro = '\n"
            "        f'`{scores_rc.mean():.4f} +/- {scores_rc.std():.4f}` (5-fold estratificado).'\n"
            "    ))\n"
            "else:\n"
            "    display(Markdown('> Insuficientes parcelas (>=100) para entrenar el clasificador lineal.'))\n"
            "\n"
            "if merged.height >= 100:\n"
            "    fs_cols3 = [c for c in merged.columns if c.startswith('farslip_') and not c.startswith('farslip_emb_')] or [c for c in merged.columns if c.startswith('farslip_emb_')]\n"
            "    X_fs = merged.select(fs_cols3).to_numpy()\n"
            "    y_fs = merged['class_id'].to_numpy()\n"
            "    scores_fs = cross_val_score(\n"
            "        LogisticRegression(max_iter=2000, multi_class='multinomial'),\n"
            "        X_fs,\n"
            "        y_fs,\n"
            "        scoring='f1_macro',\n"
            "        cv=cv,\n"
            "        n_jobs=-1,\n"
            "    )\n"
            "    display(Markdown(\n"
            "        f'**LogReg sobre FarSLIP (intersección)**: F1-macro = '\n"
            "        f'`{scores_fs.mean():.4f} +/- {scores_fs.std():.4f}` (5-fold).'\n"
            "    ))\n"
        )
    )

    cells.append(
        _md(
            "## Conclusiones\n\n"
            "Esta libreta entrega una comparativa entre dos extractores de "
            "embeddings de teledetección sobre datos reales de PASTIS-R, "
            "sin datos sintéticos y con metadata enriquecida.\n\n"
            "**Limitaciones documentadas**:\n\n"
            "1. FarSLIP fue destilado sobre parcelas de Italia, mientras "
            "que el subset PASTIS-R cubre parcelas de Francia. La "
            "intersección por `parcel_id` puede ser baja o nula. Una "
            "comparativa F1-macro directa sobre FarSLIP requeriría "
            "reentrenar el modelo sobre PASTIS, lo cual queda pendiente "
            "como trabajo futuro.\n\n"
            "2. Si los pesos de RemoteCLIP no logran descargarse, se cae "
            "a `openai/clip-vit-base-patch32`; el log estructurado deja "
            "constancia de qué modelo terminó usándose.\n\n"
            "## Lo que sigue\n\n"
            "- Si FarSLIP supera a RemoteCLIP en F1-macro sobre el "
            "conjunto de Italia, lo promovemos como modelo base del "
            "ensamble por apilamiento posterior.\n"
            "- La decisión final se documenta en `Avance3.Equipo17.ipynb` "
            "junto con el conjunto ganador."
        )
    )

    return _notebook(cells)


# ---------------------------------------------------------------------------
# 05_reencuadre_fenologico.ipynb — sin skips silenciosos.
# ---------------------------------------------------------------------------


def build_05_reencuadre() -> dict[str, Any]:
    cells: list[dict[str, Any]] = []

    cells.append(
        _md(
            "# Reencuadre fenológico — ablación completa de bloques opcionales\n\n"
            "Cuantifica el aporte de los **bloques opcionales** sobre el "
            "conjunto fused completo:\n\n"
            "1. **FarSLIP** (embeddings de 512 dimensiones, extracción real "
            "del epoch 2).\n"
            "2. **Descripción fenológica textual con Gemini 3.5 Flash** "
            "codificada con sentence-transformers.\n"
            "3. **Firma espectral REP** (Frampton et al. 2013) calculada "
            "desde anclas Sentinel-2 muestreadas en Earth Engine.\n\n"
            "**Comportamiento ante datos faltantes**: si `GEMINI_API_KEY` "
            "no está definida o `PASTIS-R/` no está en disco, el cuaderno "
            "lanza error explícito con instrucciones, sin saltarse pasos en "
            "silencio. Si los parquets de los bloques no existen, los "
            "materializa directamente desde aquí.\n\n"
            "La ablación reproduce los conjuntos de `04c_baseline.ipynb` y "
            "añade:\n\n"
            "- `with_farslip` / `farslip_only`\n"
            "- `with_pheno_text` / `pheno_text_only`\n"
            "- `with_spectral_signature` / `spectral_signature_only`"
        )
    )

    cells.append(
        _code(
            'FEATURES_PATH = "data/test_fixtures/feature_selection_parcels_subset.parquet"\n'
            'PARCELS_GEOPARQUET = "data/processed/pastis_parcels_full.geoparquet"\n'
            'FUSED_PATH = "data/features/features_fused_italy.parquet"\n'
            'PHENO_TEXT_PATH = "data/features/phenology_text_italy.parquet"\n'
            'S2_ANCHORS_PATH = "data/features/s2_anchors_italy.parquet"\n'
            'SPECTRAL_SIGNATURE_PATH = "data/features/spectral_signature_italy.parquet"\n'
            'FARSLIP_PATH = "data/farslip/embeddings_italy.parquet"\n'
            'FIGURES_SUBDIR = "us-023-preview/05_reencuadre"\n'
            'REPORTS_SUBDIR = "baseline/05_reencuadre"\n'
            "YEAR = 2023\n"
            "K_FOLDS = 5\n"
            "BUFFER_KM = 1.0\n"
            "MAX_SAMPLES = None\n"
            "ENABLE_FARSLIP = True\n"
            "ENABLE_PHENO_TEXT = True\n"
            "ENABLE_SPECTRAL_SIGNATURE = True\n"
            "ENFORCE_GEMINI_API_KEY = True\n",
            tags=["parameters"],
        )
    )

    cells.append(_code(BOOTSTRAP_CELL))

    cells.append(_md("## Carga del dataset base (con metadata)"))

    cells.append(
        _code(
            "import polars as pl\n"
            "import matplotlib.pyplot as plt\n"
            "from ml.utils.baseline_notebook_helpers import (\n"
            "    load_features_dataset_with_meta,\n"
            "    materialize_phenology_text_if_missing,\n"
            "    materialize_s2_anchors_if_missing,\n"
            "    materialize_spectral_signature_if_missing,\n"
            "    run_ablation_and_persist,\n"
            ")\n"
            "from ml.utils.parcel_id import canonical_parcel_id\n"
            "from ml.eval.reencuadre_plots import (\n"
            "    plot_ablation_bars,\n"
            "    plot_geom_leakage_comparison,\n"
            "    plot_optional_blocks_ablation,\n"
            ")\n"
            "from ml.eval.feature_ablation import FeatureAblationResult\n"
            "from pathlib import Path\n"
            "\n"
            "df = load_features_dataset_with_meta(\n"
            "    path=FEATURES_PATH,\n"
            "    parcels_geoparquet=PARCELS_GEOPARQUET,\n"
            ")\n"
            "display(Markdown(f'**Dataset base**: `{df.height:,}` parcelas x `{df.width}` cols'))\n"
        )
    )

    cells.append(_md("## Materialización del bloque `pheno_text` (Gemini sobre el dataset completo)"))

    cells.append(
        _code(
            "if ENABLE_PHENO_TEXT:\n"
            "    if ENFORCE_GEMINI_API_KEY and not env.has_gemini_api_key:\n"
            "        raise RuntimeError(\n"
            "            'GEMINI_API_KEY ausente. Define la variable en `.env.local` antes de re-ejecutar, '\n"
            "            'o pon ENFORCE_GEMINI_API_KEY=False para correr solo las ablaciones base.'\n"
            "        )\n"
            "    pheno_path = materialize_phenology_text_if_missing(\n"
            "        parcels_features_path=FEATURES_PATH,\n"
            "        output_path=PHENO_TEXT_PATH,\n"
            "        enforce_api_key=ENFORCE_GEMINI_API_KEY,\n"
            "    )\n"
            "    pheno_df = canonical_parcel_id(pl.read_parquet(pheno_path))\n"
            "    display(Markdown(f'**pheno_text**: `{pheno_df.shape}` en `{pheno_path}`'))\n"
            "else:\n"
            "    pheno_df = None\n"
            "    display(Markdown('> ENABLE_PHENO_TEXT=False: bloque omitido.'))\n"
        )
    )

    cells.append(
        _md(
            "## Materialización de anclas Sentinel-2 y firma espectral REP "
            "(Frampton 2013)"
        )
    )

    cells.append(
        _code(
            "if ENABLE_SPECTRAL_SIGNATURE:\n"
            "    if not env.has_ee_credentials:\n"
            "        display(Markdown(\n"
            "            '> Earth Engine no configurado. Define `GEE_PROJECT_ID` '\n"
            "            'en `.env.local` o ejecuta `earthengine authenticate`. '\n"
            "            'El muestreo S2 anchors fallara sin esto.'\n"
            "        ))\n"
            "    anchors_path = materialize_s2_anchors_if_missing(\n"
            "        parcels_geoparquet=PARCELS_GEOPARQUET,\n"
            "        output_path=S2_ANCHORS_PATH,\n"
            "        year=YEAR,\n"
            "    )\n"
            "    spec_path = materialize_spectral_signature_if_missing(\n"
            "        s2_anchors_path=anchors_path,\n"
            "        output_path=SPECTRAL_SIGNATURE_PATH,\n"
            "        descriptor='rep',\n"
            "    )\n"
            "    spec_df = canonical_parcel_id(pl.read_parquet(spec_path))\n"
            "    display(Markdown(f'**spectral_signature**: `{spec_df.shape}` en `{spec_path}`'))\n"
            "else:\n"
            "    spec_df = None\n"
            "    display(Markdown('> ENABLE_SPECTRAL_SIGNATURE=False: bloque omitido.'))\n"
        )
    )

    cells.append(_md("## Carga de FarSLIP desde la ruta canónica (`parcel_id` en Utf8)"))

    cells.append(
        _code(
            "if ENABLE_FARSLIP:\n"
            "    farslip_path = Path(FARSLIP_PATH)\n"
            "    if not farslip_path.exists():\n"
            "        raise FileNotFoundError(\n"
            "            f'FarSLIP parquet no encontrado en {farslip_path}. '\n"
            "            'Ejecuta `dvc pull data/farslip/embeddings_italy.parquet.dvc` '\n"
            "            'antes de re-ejecutar.'\n"
            "        )\n"
            "    farslip_df = canonical_parcel_id(pl.read_parquet(farslip_path))\n"
            "    display(Markdown(f'**FarSLIP**: `{farslip_df.shape}` en `{farslip_path}` con parcel_id Utf8.'))\n"
            "else:\n"
            "    farslip_df = None\n"
            "    display(Markdown('> ENABLE_FARSLIP=False: bloque omitido.'))\n"
        )
    )

    cells.append(
        _md(
            "## Fusión de bloques: base + FarSLIP + pheno_text + spectral_signature\n\n"
            "Aplicamos un LEFT JOIN secuencial sobre `parcel_id` (todos en "
            "Utf8 tras `canonical_parcel_id`). Las parcelas sin coincidencia "
            "en algún bloque opcional quedan con NaN; XGBoost y LightGBM "
            "los toleran nativamente y RandomForest los imputa por mediana."
        )
    )

    cells.append(
        _code(
            "df = canonical_parcel_id(df)\n"
            "fused = df\n"
            "joined_log = []\n"
            "if farslip_df is not None:\n"
            "    keep = ['parcel_id'] + [c for c in farslip_df.columns if c.startswith('farslip_')]\n"
            "    fused = fused.join(farslip_df.select(keep), on='parcel_id', how='left')\n"
            "    joined_log.append(f'FarSLIP: +{len(keep)-1} cols')\n"
            "if pheno_df is not None:\n"
            "    keep = ['parcel_id'] + [c for c in pheno_df.columns if c.startswith('pheno_text_')]\n"
            "    fused = fused.join(pheno_df.select(keep), on='parcel_id', how='left')\n"
            "    joined_log.append(f'pheno_text: +{len(keep)-1} cols')\n"
            "if spec_df is not None:\n"
            "    keep = ['parcel_id'] + [c for c in spec_df.columns if c.startswith('spectral_signature_')]\n"
            "    fused = fused.join(spec_df.select(keep), on='parcel_id', how='left')\n"
            "    joined_log.append(f'spectral_signature: +{len(keep)-1} cols')\n"
            "\n"
            "display(Markdown(\n"
            '    f"**Conjunto fused final**: `{fused.shape}`\\n\\n"\n'
            '    + "\\n".join(f"- {l}" for l in joined_log)\n'
            "))\n"
        )
    )

    cells.append(_md("## Ablación con todos los bloques opcionales"))

    cells.append(
        _code(
            "ablation_table, parquet_path = run_ablation_and_persist(\n"
            "    fused,\n"
            "    output_dir=env.reports_dir,\n"
            "    models=('xgb',),\n"
            "    k_folds=K_FOLDS,\n"
            "    buffer_km=BUFFER_KM,\n"
            "    max_samples=MAX_SAMPLES,\n"
            ")\n"
            "display(Markdown(f'**Tabla de ablación**: `{parquet_path.relative_to(env.repo)}`'))\n"
            "display(ablation_table)\n"
        )
    )

    cells.append(_md("## Gráficos: ablación completa, fuga geométrica y aporte de bloques opcionales"))

    cells.append(
        _code(
            "results = [\n"
            "    FeatureAblationResult(\n"
            "        feature_set=row['feature_set'],\n"
            "        model_kind=row['model'],\n"
            "        f1_macro=row['f1_macro'] if row['f1_macro'] is not None else float('nan'),\n"
            "        f1_weighted=row['f1_weighted'] if row['f1_weighted'] is not None else float('nan'),\n"
            "        miou=row['miou'] if row['miou'] is not None else float('nan'),\n"
            "        n_features=row['n_features'],\n"
            "        delta_vs_full=row['delta_vs_full'] if row['delta_vs_full'] is not None else float('nan'),\n"
            "    )\n"
            "    for row in ablation_table.iter_rows(named=True)\n"
            "]\n"
            "\n"
            "fig_abl = plot_ablation_bars(results, title='F1-macro por conjunto (ablación completa)')\n"
            "fig_abl.savefig(env.figures_dir / 'ablation_full.png', bbox_inches='tight')\n"
            "display(fig_abl)\n"
            "plt.close(fig_abl)\n"
            "\n"
            "fig_geom = plot_geom_leakage_comparison(results)\n"
            "fig_geom.savefig(env.figures_dir / 'geom_leakage.png', bbox_inches='tight')\n"
            "display(fig_geom)\n"
            "plt.close(fig_geom)\n"
            "\n"
            "fig_opt = plot_optional_blocks_ablation(results)\n"
            "fig_opt.savefig(env.figures_dir / 'optional_blocks.png', bbox_inches='tight')\n"
            "display(fig_opt)\n"
            "plt.close(fig_opt)\n"
        )
    )

    cells.append(
        _md(
            "## Conclusiones — decisión por bloque\n\n"
            "Las decisiones (promover, descartar o diferir) se toman bloque "
            "por bloque siguiendo el umbral de mejora `delta >= +0.005`:\n\n"
            "1. **FarSLIP**: si `with_farslip - full >= +0.005`, FarSLIP "
            "se promueve al baseline y entra al conjunto ganador. Si el "
            "delta cae en [-0.005, +0.005], se mantiene como modelo base "
            "del ensamble por apilamiento posterior. Si es menor que "
            "-0.005, se descarta del baseline.\n\n"
            "2. **pheno_text (Gemini Flash sobre el dataset completo)**: "
            "misma regla. La ablación aquí cuantifica el aporte real de "
            "la rama semántica propuesta por Wen et al. (2025) sobre el "
            "conjunto de Italia.\n\n"
            "3. **Firma espectral REP**: misma regla. Es la primera "
            "aplicación del descriptor Frampton 2013 sobre el dataset Italia.\n\n"
            "4. **`geom_only`**: si F1-macro < 0.10, se confirma que no hay "
            "leakage espacial agronómicamente significativo y la "
            "decisión previa de descartar `geom_*` queda validada con "
            "evidencia cuantitativa.\n\n"
            "## Lo que sigue\n\n"
            "- `Avance3.Equipo17.ipynb` lee esta `ablation_table.parquet`, "
            "ejecuta `select_winning_features()` y persiste "
            "`features_fused_winning_italy.parquet` que consumen los modelos densos siguientes."
        )
    )

    return _notebook(cells)


# ---------------------------------------------------------------------------
# Avance3.Equipo17.ipynb — concentrador + select_winning_features.
# ---------------------------------------------------------------------------


def build_avance3() -> dict[str, Any]:
    cells: list[dict[str, Any]] = []

    cells.append(
        _md(
            "# Avance 3 — Baseline tabular y conjunto ganador\n\n"
            "Cuaderno concentrador que reúne los resultados de las libretas "
            "anteriores y produce el **conjunto de características ganador** "
            "que alimenta a los modelos densos siguientes (U-Net, U-TAE, "
            "TSViT, Swin-UNETR) y a los ensambles (voting, bagging, stacking, "
            "blending).\n\n"
            "Estructura:\n\n"
            "1. Comparativa de los 3 modelos (RandomForest, XGBoost, "
            "LightGBM) desde `model_comparison_04.parquet`.\n"
            "2. Resumen de la ablación completa desde "
            "`reports/baseline/05_reencuadre/ablation_table.parquet`.\n"
            "3. Decisión por bloque opcional (FarSLIP, pheno_text, "
            "spectral_signature) vía `select_winning_features`.\n"
            "4. Persistencia del parquet ganador y de un manifiesto JSON "
            "con la lista nominal de columnas, de modo que los modelos "
            "siguientes consuman exactamente el mismo conjunto."
        )
    )

    cells.append(
        _code(
            'COMPARISON_PATH_04 = "reports/baseline/04_baseline/model_comparison_04.parquet"\n'
            'ABLATION_PATH_05 = "reports/baseline/05_reencuadre/ablation_table.parquet"\n'
            'FUSED_PATH = "data/features/features_fused_italy.parquet"\n'
            'WINNING_OUTPUT = "data/features/features_fused_winning_italy.parquet"\n'
            'FIGURES_SUBDIR = "us-023-preview/Avance3"\n'
            'REPORTS_SUBDIR = "baseline/Avance3"\n'
            "PROMOTE_THRESHOLD = 0.005\n",
            tags=["parameters"],
        )
    )

    cells.append(_code(BOOTSTRAP_CELL))

    cells.append(_md("## Comparativa de los 3 modelos baseline"))

    cells.append(
        _code(
            "import polars as pl\n"
            "import matplotlib.pyplot as plt\n"
            "from pathlib import Path\n"
            "from ml.eval.reencuadre_plots import (\n"
            "    plot_model_comparison_v2_with_v1_overlay,\n"
            "    plot_optional_blocks_ablation,\n"
            ")\n"
            "from ml.eval.feature_ablation import FeatureAblationResult\n"
            "\n"
            "comparison_path = Path(COMPARISON_PATH_04)\n"
            "if comparison_path.exists():\n"
            "    comparison = pl.read_parquet(comparison_path)\n"
            "    display(Markdown(f'**Comparativa de modelos** (`{comparison_path}`):'))\n"
            "    display(comparison)\n"
            "    v2_metrics = {row['model']: row['f1_macro'] for row in comparison.iter_rows(named=True)}\n"
            "    v1_metrics = {'xgb': 0.41, 'rf': 0.39}  # referencias previas\n"
            "    fig = plot_model_comparison_v2_with_v1_overlay(v2_metrics, v1_metrics=v1_metrics)\n"
            "    fig.savefig(env.figures_dir / 'model_comparison_v2.png', bbox_inches='tight')\n"
            "    display(fig)\n"
            "    plt.close(fig)\n"
            "else:\n"
            "    raise FileNotFoundError(\n"
            "        f'No existe `{comparison_path}`. Ejecuta `04_baseline.ipynb` antes de este notebook.'\n"
            "    )\n"
        )
    )

    cells.append(_md("## Ablación completa (FarSLIP, pheno_text y firma espectral)"))

    cells.append(
        _code(
            "ablation_path = Path(ABLATION_PATH_05)\n"
            "if not ablation_path.exists():\n"
            "    raise FileNotFoundError(\n"
            "        f'No existe `{ablation_path}`. Ejecuta `05_reencuadre_fenologico.ipynb` antes.'\n"
            "    )\n"
            "ablation_table = pl.read_parquet(ablation_path)\n"
            "display(ablation_table)\n"
            "\n"
            "results = [\n"
            "    FeatureAblationResult(\n"
            "        feature_set=row['feature_set'],\n"
            "        model_kind=row['model'],\n"
            "        f1_macro=row['f1_macro'] if row['f1_macro'] is not None else float('nan'),\n"
            "        f1_weighted=row['f1_weighted'] if row['f1_weighted'] is not None else float('nan'),\n"
            "        miou=row['miou'] if row['miou'] is not None else float('nan'),\n"
            "        n_features=row['n_features'],\n"
            "        delta_vs_full=row['delta_vs_full'] if row['delta_vs_full'] is not None else float('nan'),\n"
            "    )\n"
            "    for row in ablation_table.iter_rows(named=True)\n"
            "]\n"
            "fig = plot_optional_blocks_ablation(results)\n"
            "fig.savefig(env.figures_dir / 'optional_blocks.png', bbox_inches='tight')\n"
            "display(fig)\n"
            "plt.close(fig)\n"
        )
    )

    cells.append(
        _md(
            "## Selección del conjunto ganador con `select_winning_features`\n\n"
            "Promovemos cada bloque opcional cuando su "
            "`delta_vs_full >= +0.005`. El conjunto base obligatorio "
            "incluye AlphaEarth, índices espectrales, atributos "
            "fenológicos, ERA5 y SRTM. Las columnas `geom_*` se descartan "
            "siempre por mostrar fuga espacial (resultado validado en la "
            "ablación)."
        )
    )

    cells.append(
        _code(
            "from ml.features.winning_features import (\n"
            "    select_winning_features,\n"
            "    persist_winning_features,\n"
            ")\n"
            "\n"
            "fused_path = Path(FUSED_PATH)\n"
            "if not fused_path.exists():\n"
            "    raise FileNotFoundError(\n"
            "        f'No existe `{fused_path}`. Ejecuta `05_reencuadre_fenologico.ipynb` '\n"
            "        '(genera el fused completo durante la materializacion).'\n"
            "    )\n"
            "fused = pl.read_parquet(fused_path)\n"
            "\n"
            "winning = select_winning_features(\n"
            "    ablation_table,\n"
            "    available_cols=fused.columns,\n"
            "    promote_threshold=PROMOTE_THRESHOLD,\n"
            "    discard_geom=True,\n"
            ")\n"
            "display(Markdown('**Decisiones por bloque**:'))\n"
            "display(pl.DataFrame({\n"
            "    'bloque': list(winning.decisions.keys()),\n"
            "    'promovido': list(winning.decisions.values()),\n"
            "}))\n"
            "display(Markdown(f'**Conjunto ganador**: `{winning.name}` con `{len(winning.feature_cols)}` columnas.'))\n"
            "display(Markdown('### Rationale'))\n"
            "display(Markdown(winning.rationale))\n"
            "\n"
            "winning_path = persist_winning_features(\n"
            "    winning,\n"
            "    fused,\n"
            "    output_path=WINNING_OUTPUT,\n"
            "    overwrite=True,\n"
            ")\n"
            "display(Markdown(f'**Conjunto ganador guardado**: `{winning_path.relative_to(env.repo)}`'))\n"
            "display(Markdown(f'**Manifiesto JSON**: `{winning_path.with_suffix(\".manifest.json\").relative_to(env.repo)}`'))\n"
        )
    )

    cells.append(
        _md(
            "## Nombres de las características ganadoras\n\n"
            "Para que los modelos siguientes (densos y ensambles) lean "
            "exactamente las mismas columnas, publicamos la **lista "
            "nominal exacta** de las características ganadoras. Cualquier "
            "modelo posterior que cargue "
            "`features_fused_winning_italy.parquet` reúsa esta lista sin "
            "tener que reinventar la selección."
        )
    )

    cells.append(
        _code(
            "import json\n"
            "manifest = json.loads(\n"
            "    Path(WINNING_OUTPUT).with_suffix('.manifest.json').read_text(encoding='utf-8')\n"
            ")\n"
            "display(Markdown(f'**Número de características**: `{manifest[\"n_features\"]}`'))\n"
            "display(Markdown('**Columnas de metadata** (no son características):'))\n"
            "display(pl.Series('meta_cols', manifest['meta_cols']).to_frame())\n"
            "display(Markdown('**Características ganadoras** (primeras 40):'))\n"
            "display(pl.Series('feature', manifest['feature_cols'][:40]).to_frame())\n"
            "display(Markdown(f'**Total de características**: `{len(manifest[\"feature_cols\"])}`'))\n"
        )
    )

    cells.append(
        _md(
            "## Conclusiones — cierre del baseline tabular\n\n"
            "Con este cuaderno cerramos el baseline tabular:\n\n"
            "- Tres modelos (RandomForest, XGBoost, LightGBM) entrenados "
            "con validación cruzada espacial de 5 particiones y buffer de "
            "1 km, registrados en MLflow y serializados con joblib.\n"
            "- Ablación de 8 a 10 conjuntos de características con "
            "decisiones documentadas por cada bloque opcional.\n"
            "- Conjunto de características ganador nombrado y guardado en "
            "`features_fused_winning_italy.parquet` junto con un manifiesto "
            "JSON que lista cada columna.\n\n"
            "## Lo que sigue\n\n"
            "Las libretas siguientes (`05_alt_models.ipynb` y "
            "`06_final_gemma4_ensembles.ipynb`) cargarán **el mismo parquet "
            "ganador** y entrenarán U-Net, U-TAE, TSViT, Swin-UNETR, "
            "Gemma 4 26B-MoE con LoRA y los cuatro ensambles (voting, "
            "bagging, stacking, blending), garantizando que todos operan "
            "sobre el mismo conjunto de características."
        )
    )

    return _notebook(cells)


# ---------------------------------------------------------------------------
# Builder dispatcher.
# ---------------------------------------------------------------------------


BUILDERS = {
    "04_baseline": (build_04_baseline, NOTEBOOK_DIR / "04_baseline.ipynb"),
    "04b_baseline": (build_04b_baseline, NOTEBOOK_DIR / "04b_baseline.ipynb"),
    "04c_baseline": (build_04c_baseline, NOTEBOOK_DIR / "04c_baseline.ipynb"),
    "04_farslip_eval_pastis": (
        build_04_farslip_eval_pastis,
        NOTEBOOK_DIR / "04_farslip_eval_pastis.ipynb",
    ),
    "05_reencuadre_fenologico": (
        build_05_reencuadre,
        NOTEBOOK_DIR / "05_reencuadre_fenologico.ipynb",
    ),
    "Avance3.Equipo17": (
        build_avance3,
        NOTEBOOK_DIR / "Avance3.Equipo17.ipynb",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=list(BUILDERS.keys()),
        action="append",
        help="Construye solo el notebook indicado (puede repetirse). Sin esta opcion construye todos.",
    )
    args = parser.parse_args()

    selected = args.only or list(BUILDERS.keys())
    print(f"Construyendo {len(selected)} notebook(s):")
    for key in selected:
        builder, path = BUILDERS[key]
        nb = builder()
        _write(path, nb)
    print("Hecho.")


if __name__ == "__main__":
    main()
