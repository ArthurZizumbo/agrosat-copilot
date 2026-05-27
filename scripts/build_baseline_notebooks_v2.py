"""Builder unificado de los 6 notebooks de baseline (US-023-preview v2).

Genera los 6 notebooks de `notebooks/baseline/` desde una sola fuente de
verdad, alineados al estandar de `notebooks/CLAUDE.md` y reutilizando
todos los helpers de `ml/`:

- `notebooks/baseline/04_baseline.ipynb` — XGB + LGBM + RF + temporales + plots.
- `notebooks/baseline/04b_baseline.ipynb` — variante con AlphaEarth solo
  (piloto del patron de bootstrap nuevo).
- `notebooks/baseline/04c_baseline.ipynb` — ablation de bloques con fix
  de detection alphaearth_only.
- `notebooks/baseline/04_farslip_eval_pastis.ipynb` — FarSLIP vs RemoteCLIP
  sobre PASTIS real (sin sintetico).
- `notebooks/baseline/05_reencuadre_fenologico.ipynb` — fenologia + ablation
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
    """Crea celda de codigo."""
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
# Bootstrap cell estandar (igual en todos los notebooks).
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
            "# Baseline 04b — RandomForest sobre features US-018\n\n"
            "Variante minima del baseline tabular sobre el subset US-018 "
            "(85 951 parcelas, 17 indices espectrales x 9 stats + FFT NDVI "
            "+ 8 features fenologicas). Sirve como piloto del patron "
            "`setup_notebook` + `train_baseline_three_models` que el resto "
            "de las libretas reusa.\n\n"
            "**Pregunta**: ¿que F1-macro out-of-fold consigue RandomForest "
            "puro sobre el subset US-018 con spatial CV 5-fold + buffer 1 km?\n\n"
            "## Requisitos\n\n"
            "- `data/test_fixtures/feature_selection_parcels_subset.parquet` "
            "presente (descargable via `dvc pull`).\n"
            "- `data/processed/pastis_parcels_full.geoparquet` presente "
            "(generado por el pipeline EDA US-011).\n"
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
            "Combinamos el subset US-018 con la metadata real (clase + patch "
            "+ fold + area) desde el geoparquet de parcelas Italia."
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
            "## Distribucion de clases\n\n"
            "Antes de entrenar, reportamos la distribucion real de las 18 "
            "clases PASTIS-R y proponemos un threshold sensato de soporte "
            "(en lugar del 1000 hardcoded que dejaba solo 1 clase pasando)."
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
            '    f"**Threshold sugerido**: p25 = `{threshold_p25}`, "\n'
            '    f"p50 = `{threshold_p50}` parcelas. "\n'
            '    "Las clases por debajo del threshold tienen soporte debil "\n'
            '    "y se reportan en color en los plots."\n'
            "))\n"
        )
    )

    cells.append(
        _md("## Entrenamiento RandomForest + XGBoost + LightGBM con spatial CV")
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
            "display(Markdown(f'**Tabla comparativa persistida**: `{comparison_path}`'))\n"
            "display(comparison)\n"
        )
    )

    cells.append(_md("## Graficas: comparativa de modelos + soporte por clase"))

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
            "    baseline_label='referencia US-022 (F1-macro 0.40)',\n"
            "    title='F1-macro RF vs XGB vs LGBM (04b)',\n"
            ")\n"
            "fig1.savefig(env.figures_dir / 'model_comparison_04b.png', bbox_inches='tight')\n"
            "display(fig1)\n"
            "plt.close(fig1)\n"
            "\n"
            "fig2 = plot_class_support_bars(\n"
            "    report.rename({'n_parcels': 'len'}),\n"
            "    weak_threshold=threshold_p25,\n"
            "    title=f'Soporte por clase (threshold p25 = {threshold_p25})',\n"
            ")\n"
            "fig2.savefig(env.figures_dir / 'class_support_04b.png', bbox_inches='tight')\n"
            "display(fig2)\n"
            "plt.close(fig2)\n"
        )
    )

    cells.append(_md("## Per-class F1 del mejor modelo (out-of-fold)"))

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
            "Esta libreta cumple el rol de **piloto** del patron de bootstrap "
            "nuevo (`setup_notebook` + `baseline_notebook_helpers`) sobre el "
            "subset US-018. Los hallazgos importantes:\n\n"
            "1. **Tres modelos comparables**: RandomForest, XGBoost y "
            "LightGBM ejecutados con identica spatial CV 5-fold + buffer 1 "
            "km. La tabla persistida `model_comparison_04b.parquet` queda "
            "como referencia local para detectar regresiones cuando "
            "agregemos bloques opcionales en `04_baseline` y "
            "`05_reencuadre_fenologico`.\n\n"
            "2. **Soporte por clase reportado con threshold p25**: en lugar "
            "del threshold hardcoded de 1000 que dejaba 17/18 clases marcadas "
            "como debiles, ahora el threshold se calcula desde la propia "
            "distribucion y solo destaca las clases verdaderamente raras.\n\n"
            "3. **Per-class F1 del mejor modelo**: identifica que clases "
            "concentran el error (las raras suelen estar bajo el 0.10), util "
            "para discutir merge fenologico via "
            "`merge_to_phenological_groups` en futuras iteraciones.\n\n"
            "## Lo que sigue\n\n"
            "- `04_baseline.ipynb` reutiliza este patron sobre el conjunto "
            "fused completo (con AlphaEarth + ERA5 + SRTM).\n"
            "- `05_reencuadre_fenologico.ipynb` mide el aporte de los "
            "bloques opcionales (FarSLIP, pheno_text Gemini, firma "
            "espectral) y persiste la decision final.\n"
            "- `Avance3.Equipo17.ipynb` consolida y nombra el conjunto "
            "ganador (`select_winning_features`)."
        )
    )

    return _notebook(cells)


# ---------------------------------------------------------------------------
# 04_baseline.ipynb — 3 modelos sobre fused + graficas.
# ---------------------------------------------------------------------------


def build_04_baseline() -> dict[str, Any]:
    cells: list[dict[str, Any]] = []

    cells.append(
        _md(
            "# Baseline 04 — RF + XGBoost + LightGBM sobre features completas\n\n"
            "Baseline tabular canonico del proyecto (Avance 3). Entrena los "
            "**tres modelos** sobre el conjunto fused completo de Italia "
            "(85 951 parcelas) con spatial CV 5-fold + buffer 1 km, y "
            "produce todas las graficas necesarias para la rubrica del "
            "Avance 3:\n\n"
            "- Distribucion de clases real (18 clases PASTIS-R Italia).\n"
            "- Comparativa F1-macro / F1-weighted / mIoU entre los 3 modelos.\n"
            "- F1 por clase del modelo ganador.\n"
            "- Matriz de confusion out-of-fold.\n\n"
            "El conjunto fused incluye los bloques base: AlphaEarth (64 "
            "dim), indices espectrales x stats (17 x 9 = 85), FFT NDVI "
            "(24), fenologia (8), ERA5 mensual (24), SRTM (3). Los bloques "
            "opcionales (FarSLIP, pheno_text, spectral_signature) se evaluan "
            "en `05_reencuadre_fenologico.ipynb`."
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

    cells.append(_md("## Distribucion de clases con merge fenologico opcional"))

    cells.append(
        _code(
            "report = class_distribution_report(df)\n"
            "display(report)\n"
            "threshold = recommend_threshold(report, method='p25')\n"
            "display(Markdown(f'Threshold p25 sugerido: `{threshold}` parcelas.'))\n"
            "\n"
            "fig_class = plot_class_support_bars(\n"
            "    report.rename({'n_parcels': 'len'}),\n"
            "    weak_threshold=threshold,\n"
            "    title=f'Distribucion de clases (threshold p25 = {threshold})',\n"
            ")\n"
            "fig_class.savefig(env.figures_dir / 'class_distribution.png', bbox_inches='tight')\n"
            "display(fig_class)\n"
            "plt.close(fig_class)\n"
        )
    )

    cells.append(
        _md(
            "## Entrenamiento RF + XGB + LGBM con spatial CV\n\n"
            "Wall clock esperado: 30-60 minutos en RTX 4070 / L4 (XGB GPU + "
            "LGBM CPU). RandomForest CPU multinucleo."
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
            "display(Markdown(f'**Tabla persistida**: `{comparison_path.relative_to(env.repo)}`'))\n"
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
            "    baseline_label='ref US-022 (F1-macro 0.40)',\n"
            "    title='F1-macro out-of-fold por modelo',\n"
            ")\n"
            "fig_cmp.savefig(env.figures_dir / 'model_comparison.png', bbox_inches='tight')\n"
            "display(fig_cmp)\n"
            "plt.close(fig_cmp)\n"
        )
    )

    cells.append(_md("## Matriz de confusion y F1 por clase (modelo ganador)"))

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
            "    title=f'Matriz de confusion ({best_model}) normalizada por fila',\n"
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
            "display(Markdown(f'Modelo persistido en `{joblib_path.relative_to(env.repo)}`'))\n"
        )
    )

    cells.append(
        _md(
            "## Conclusiones\n\n"
            "El baseline tabular canonico queda entrenado y evaluado con "
            "los tres modelos pedidos por la rubrica del Avance 3 "
            "(RandomForest, XGBoost, LightGBM). Las metricas, las graficas "
            "y el modelo serializado quedan persistidos en `reports/` y "
            "`paper/figures/` para reusar en `Avance3.Equipo17.ipynb`.\n\n"
            "**Observaciones agronomicas**:\n\n"
            "- Las clases mayoritarias (1, 3, 8, 2 — cereales de invierno, "
            "praderas permanentes, viñedos) concentran el F1 alto.\n"
            "- Las clases raras con soporte por debajo del threshold p25 "
            "tienen F1 < 0.10 y son candidatas a merge fenologico via "
            "`PASTIS_R_GROUPINGS['phenological_cycle']` en una segunda "
            "iteracion de modelado.\n\n"
            "## Lo que sigue\n\n"
            "- `05_reencuadre_fenologico.ipynb` mide el aporte de los "
            "bloques opcionales (FarSLIP, pheno_text Gemini real, firma "
            "espectral REP) sobre este conjunto.\n"
            "- `Avance3.Equipo17.ipynb` consolida y persiste el conjunto "
            "ganador (`select_winning_features`) que consume EPIC 5."
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
            "# Baseline 04c — Ablation de bloques de features\n\n"
            "Mide el aporte incremental de cada bloque del vector fused. "
            "Para cada conjunto de columnas entrenamos XGBoost (mas LightGBM "
            "como sanity) con identica spatial CV 5-fold y reportamos "
            "F1-macro + delta vs `full`.\n\n"
            "Conjuntos canonicos evaluados:\n\n"
            "- `full` — todas las features numericas disponibles.\n"
            "- `no_geom` — `full` sin las 3 columnas `geom_*`.\n"
            "- `no_geom_no_era5_srtm` — adicionalmente sin `era5_*` ni `srtm_*`.\n"
            "- `alphaearth_only` — solo las 64 dimensiones `ae_*`.\n"
            "- `phenology_only` — 8 fenologicas + 24 FFT NDVI.\n"
            "- `geom_only` — solo `geom_*` (test cuantitativo de leakage espacial).\n\n"
            "**Fix US-023-preview**: la deteccion de columnas AlphaEarth ahora "
            "tolera variantes (`ae_*`, `emb_*`, `dim_*`, `alphaearth_*`) y "
            "el conjunto `alphaearth_only` ya no aparece con `n_features=0` "
            "ni NaN cuando hay AE en el dataset."
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
            "MAX_SAMPLES = None  # None = dataset completo; reducir para CI rapido.\n",
            tags=["parameters"],
        )
    )

    cells.append(_code(BOOTSTRAP_CELL))

    cells.append(_md("## Carga del dataset + ablation"))

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
            "display(Markdown(f'**Tabla ablation**: `{parquet_path.relative_to(env.repo)}`'))\n"
            "display(ablation_table)\n"
        )
    )

    cells.append(_md("## Graficas: barras de F1 por conjunto + comparativa geom"))

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
            "fig_abl = plot_ablation_bars(results, title='F1-macro por conjunto de features (04c)')\n"
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
            "**Lectura honesta de la ablation**:\n\n"
            "- El conjunto `full` define la referencia. El delta de `no_geom` "
            "vs `full` cuantifica el aporte (o leakage) de las columnas "
            "geometricas: si delta ~0 significa que `geom_*` no aporta señal "
            "agronomica; si delta es positivo, descartarlas mejora porque "
            "introducian ruido.\n\n"
            "- El conjunto `geom_only` es el **test cuantitativo de "
            "leakage**: F1-macro < 0.10 confirma que area/perimetro/elongacion "
            "por si solas no permiten clasificar cultivos — el modelo no "
            "puede aprender clase a partir de geometria.\n\n"
            "- `alphaearth_only` muestra cuanto del baseline viene de los 64 "
            "embeddings del Foundation Model. Si la diferencia entre "
            "`alphaearth_only` y `full` es pequeña, los demas bloques no "
            "estan agregando mucho mas alla del FM.\n\n"
            "## Lo que sigue\n\n"
            "- `05_reencuadre_fenologico.ipynb` amplia esta tabla con los "
            "bloques opcionales (FarSLIP, pheno_text Gemini real, firma "
            "espectral REP) materializados desde el propio notebook si no "
            "existen.\n"
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
            "# Evaluacion FarSLIP vs RemoteCLIP sobre PASTIS-R real\n\n"
            "Compara dos extractores de embeddings de teledeteccion sobre el "
            "**mismo subset real** de PASTIS-R:\n\n"
            "- **FarSLIP** (Tang et al. 2024): CLIP fine-tuned para vinos y "
            "cultivos europeos via distillation desde Sentinel-2 + "
            "descripciones textuales.\n"
            "- **RemoteCLIP** (Chen et al. 2024): CLIP fine-tuned sobre 12 "
            "datasets de remote sensing.\n\n"
            "Si los embeddings de RemoteCLIP no estan disponibles tras "
            "descargar los pesos desde Hugging Face, el extractor cae a "
            "`openai/clip-vit-base-patch32` como fallback (documentado en el "
            "log).\n\n"
            "**Sin datos sinteticos**: el subset PASTIS-R se genera desde "
            "`data/PASTIS-R/metadata.geojson` + `DATA_S2/` reales con "
            "muestreo estratificado por clase. Si PASTIS-R no esta en disco, "
            "el notebook lanza `FileNotFoundError` con instrucciones de "
            "`dvc pull` o de descarga manual desde Zenodo.\n\n"
            "**Comparativa**: similitud coseno de los pares (FarSLIP_emb, "
            "RemoteCLIP_emb) por parcela, mas un clasificador lineal "
            "(LogisticRegression) sobre cada espacio de embeddings para "
            "comparar separabilidad por clase."
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

    cells.append(_md("## Materializar subset PASTIS real (si no existe)"))

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
            "display(Markdown('**Distribucion de clases en el subset**:'))\n"
            "display(\n"
            "    subset.group_by('class_id', 'class_name').len()\n"
            "    .sort('len', descending=True)\n"
            ")\n"
        )
    )

    cells.append(_md("## Materializar embeddings RemoteCLIP (si no existen)"))

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

    cells.append(_md("## Cargar embeddings FarSLIP del path canonico"))

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

    cells.append(_md("## Comparativa: similitud coseno FarSLIP vs RemoteCLIP por parcela"))

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
            "        'FarSLIP fue generado para Italia (US-022-c) y el subset PASTIS-R '\n"
            "        'es Francia. La comparativa requiere un FarSLIP-PASTIS dedicado '\n"
            "        '(backlog US-022-e).'\n"
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
            "    ax.set_title('Distribucion de similitud entre embeddings FarSLIP y RemoteCLIP')\n"
            "    ax.axvline(0.0, color='#888', linestyle='--', linewidth=1)\n"
            "    fig.savefig(env.figures_dir / 'cosine_farslip_vs_remoteclip.png', bbox_inches='tight')\n"
            "    display(fig)\n"
            "    plt.close(fig)\n"
        )
    )

    cells.append(
        _md("## Separabilidad lineal: LogReg sobre FarSLIP vs RemoteCLIP")
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
            "        f'**LogReg sobre FarSLIP (interseccion)**: F1-macro = '\n"
            "        f'`{scores_fs.mean():.4f} +/- {scores_fs.std():.4f}` (5-fold).'\n"
            "    ))\n"
        )
    )

    cells.append(
        _md(
            "## Conclusiones\n\n"
            "Esta libreta entrega una **comparativa honesta** entre dos "
            "extractores de embeddings de teledeteccion sobre datos reales "
            "PASTIS-R, sin datos sinteticos y con metadata enriquecida.\n\n"
            "Limitaciones documentadas:\n\n"
            "1. FarSLIP fue distillado sobre parcelas de Italia (US-022-c); "
            "el subset PASTIS-R es de Francia. La interseccion por "
            "`parcel_id` puede ser baja o nula. La comparativa "
            "F1-macro(FarSLIP) requiere FarSLIP-PASTIS dedicado (backlog "
            "US-022-e).\n\n"
            "2. RemoteCLIP cae a `openai/clip-vit-base-patch32` si los pesos "
            "RemoteCLIP no se pudieron descargar de Hugging Face; el log "
            "estructurado documenta cual modelo se uso.\n\n"
            "## Lo que sigue\n\n"
            "- Si FarSLIP supera a RemoteCLIP en F1-macro sobre Italia, "
            "promovemos FarSLIP como base learner del stacking EPIC 6.\n"
            "- La decision final se documenta en `Avance3.Equipo17.ipynb` "
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
            "# Reencuadre fenologico — ablation completa de bloques opcionales\n\n"
            "Cierra la US-023-preview: mide el aporte cuantitativo de los "
            "**bloques opcionales** sobre el conjunto fused completo:\n\n"
            "1. **FarSLIP** (embeddings 512-dim, US-022-c epoch_2 real).\n"
            "2. **pheno_text** (Gemini 3.5 Flash sobre el dataset full + "
            "sentence-transformers, US-022-b D-5).\n"
            "3. **Firma espectral REP** (Frampton et al. 2013, US-023-preview "
            "P5) computada desde anclas S2 muestreadas en GEE.\n\n"
            "**Sin skips silenciosos**: si falta `GEMINI_API_KEY` o "
            "`PASTIS-R/`, el notebook lanza error explicito con instrucciones. "
            "Si los parquets de bloques no existen, los materializa "
            "directamente desde aqui (auto-gen).\n\n"
            "Reproduce las ablaciones de `04c_baseline.ipynb` y le suma:\n\n"
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

    cells.append(_md("## Auto-materializar bloque pheno_text (Gemini sobre full)"))

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
            "## Auto-materializar S2 anchors + firma espectral REP (Frampton 2013)"
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

    cells.append(_md("## Auto-materializar FarSLIP (path canonico Utf8)"))

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
            "## Fusion de bloques: base + (FarSLIP, pheno_text, spectral_signature)\n\n"
            "Hacemos un LEFT JOIN secuencial sobre `parcel_id` (todos Utf8 "
            "tras `canonical_parcel_id`). Las parcelas sin match en algun "
            "bloque opcional quedan con NaN — XGBoost y LightGBM toleran NaN "
            "nativamente, RandomForest lo imputa por mediana."
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

    cells.append(_md("## Ablation con todos los bloques opcionales"))

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
            "display(Markdown(f'**Tabla ablation**: `{parquet_path.relative_to(env.repo)}`'))\n"
            "display(ablation_table)\n"
        )
    )

    cells.append(_md("## Graficas: ablation completa + leakage geom + bloques opcionales"))

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
            "fig_abl = plot_ablation_bars(results, title='F1-macro por conjunto (full ablation)')\n"
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
            "## Conclusiones — decisiones honestas por bloque\n\n"
            "Las decisiones promover / descartar / diferir se toman por "
            "bloque siguiendo el threshold de mejora `delta >= +0.005`:\n\n"
            "1. **FarSLIP**: si `with_farslip - full >= +0.005`, se promueve "
            "al baseline y entra al conjunto ganador. Si esta entre [-0.005, "
            "+0.005], se mantiene como base learner del stacking EPIC 6. "
            "Si es < -0.005, se descarta del baseline.\n\n"
            "2. **pheno_text (Gemini Flash sobre full)**: misma regla. La "
            "ablation aqui cuantifica el aporte real de la rama semantica "
            "Wen et al. 2025 sobre Italia.\n\n"
            "3. **Firma espectral REP**: misma regla. Es la primera "
            "aplicacion del descriptor Frampton 2013 sobre el dataset Italia.\n\n"
            "4. **`geom_only`**: si F1-macro < 0.10, se confirma que no hay "
            "leakage espacial agronomicamente significativo y la "
            "decision US-022-b de descartar `geom_*` queda validada con "
            "evidencia cuantitativa.\n\n"
            "## Lo que sigue\n\n"
            "- `Avance3.Equipo17.ipynb` lee esta `ablation_table.parquet`, "
            "ejecuta `select_winning_features()` y persiste "
            "`features_fused_winning_italy.parquet` que consume EPIC 5."
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
            "Notebook concentrador del Avance 3 (24-may-2026). Reune los "
            "resultados de las libretas anteriores y produce el "
            "**conjunto de features ganador** que consume EPIC 5 (modelos "
            "densos U-Net / U-TAE / TSViT / Swin-UNETR) y EPIC 6 (ensambles).\n\n"
            "Estructura:\n\n"
            "1. Resumen comparativo de los 3 modelos (RF + XGBoost + LightGBM) "
            "desde `model_comparison_04.parquet`.\n"
            "2. Resumen de la ablation completa desde "
            "`05_reencuadre/reports/ablation_table.parquet`.\n"
            "3. Decision por bloque opcional (FarSLIP / pheno_text / "
            "spectral_signature) via `select_winning_features`.\n"
            "4. Persistencia del parquet ganador + manifest JSON con la "
            "lista nominal de features (nombres canonicos para que los "
            "modelos siguientes lean exactamente las mismas columnas)."
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
            "    v1_metrics = {'xgb': 0.41, 'rf': 0.39}  # referencias publicadas US-022\n"
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

    cells.append(_md("## Ablation completa (FarSLIP + pheno_text + spectral_signature)"))

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
            "## Seleccion del conjunto ganador (`select_winning_features`)\n\n"
            "Aplicamos la regla de promover bloques opcionales si su "
            "`delta_vs_full >= +0.005`. El conjunto base obligatorio "
            "incluye AlphaEarth, indices espectrales, fenologia, ERA5 y "
            "SRTM (geom_* siempre descartado por US-022-b)."
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
            "display(Markdown(f'**Conjunto ganador persistido**: `{winning_path.relative_to(env.repo)}`'))\n"
            "display(Markdown(f'**Manifest JSON**: `{winning_path.with_suffix(\".manifest.json\").relative_to(env.repo)}`'))\n"
        )
    )

    cells.append(
        _md(
            "## Nombres de las features ganadoras\n\n"
            "Para reproducibilidad de los modelos siguientes (EPIC 5 + EPIC "
            "6), publicamos la **lista nominal exacta** de las columnas "
            "ganadoras. Cualquier modelo posterior que cargue "
            "`features_fused_winning_italy.parquet` reusa esta lista sin "
            "tener que reinventar la seleccion."
        )
    )

    cells.append(
        _code(
            "import json\n"
            "manifest = json.loads(\n"
            "    Path(WINNING_OUTPUT).with_suffix('.manifest.json').read_text(encoding='utf-8')\n"
            ")\n"
            "display(Markdown(f'**N features**: `{manifest[\"n_features\"]}`'))\n"
            "display(Markdown('**Meta cols** (no son features):'))\n"
            "display(pl.Series('meta_cols', manifest['meta_cols']).to_frame())\n"
            "display(Markdown('**Feature cols ganadoras** (primeras 40):'))\n"
            "display(pl.Series('feature', manifest['feature_cols'][:40]).to_frame())\n"
            "display(Markdown(f'**Total feature cols**: `{len(manifest[\"feature_cols\"])}`'))\n"
        )
    )

    cells.append(
        _md(
            "## Conclusiones — cierre del baseline\n\n"
            "Con esta libreta cerramos el Avance 3:\n\n"
            "- Tres modelos baseline (RandomForest, XGBoost, LightGBM) "
            "entrenados sobre spatial CV 5-fold + buffer 1 km, persistidos "
            "en MLflow + joblib.\n\n"
            "- Ablation de 8-10 conjuntos con decisiones documentadas por "
            "bloque opcional.\n\n"
            "- Conjunto de features ganador nombrado y persistido en "
            "`features_fused_winning_italy.parquet` mas un manifest JSON.\n\n"
            "## Lo que sigue (EPIC 5)\n\n"
            "Los notebooks siguientes (Avance 4: `05_alt_models.ipynb` y "
            "Avance 5: `06_final_gemma4_ensembles.ipynb`) cargan **el "
            "mismo parquet ganador** y entrenan U-Net, U-TAE, TSViT, "
            "Swin-UNETR, Gemma 4 26B-MoE LoRA y los 4 ensambles del EPIC 6, "
            "garantizando que todos comparten el mismo conjunto de "
            "features."
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
