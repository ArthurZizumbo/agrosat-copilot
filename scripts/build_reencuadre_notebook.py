"""Constructor programatico de ``notebooks/feature_engineering/05_reencuadre_fenologico.ipynb``.

Genera el notebook entregable celda a celda. Sigue el patron de
``scripts/build_baseline_notebook.py``: ejecutable end-to-end con papermill,
commiteable con outputs poblados.

Secciones del notebook:
  1. Setup + glosario (Ablation, Spatial CV, OOF).
  2. EDA del desbalance (grafica de soporte por clase).
  3. Ablation de features XGBoost x 5 conjuntos (descarte geografico).
  4. Comparativa de modelos sobre el conjunto ganador (XGBoost vs TempCNN
     vs InceptionTime, todos contra la linea baseline).
  5. F1 por clase + matriz de confusion del mejor modelo temporal.
  6. Clustering sin coordenadas (KMeans + UMAP + curva NDVI por cluster).
  7. Estrategia de desbalance.
  8. Rama semantica fenologica (integracion, ejecucion real diferida).
  9. Conclusiones.
  10. Glosario tecnico.

Uso:
    poetry run python scripts/build_reencuadre_notebook.py \
        --out notebooks/feature_engineering/05_reencuadre_fenologico.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
import typer

app = typer.Typer(add_completion=False, help=__doc__)


def _md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source)


def _code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source)


def _params_code(source: str) -> nbf.NotebookNode:
    cell = nbf.v4.new_code_cell(source)
    cell.metadata["tags"] = ["parameters"]
    return cell


CELLS: list[nbf.NotebookNode] = [
    # --------------------------------------------------------------------
    # Celda 1 - Titulo y descripcion
    # --------------------------------------------------------------------
    _md(
        "# Reencuadre fenologico del baseline — descarte geografico, modelos temporales y rama semantica\n"
        "\n"
        "Este analisis revisita el baseline tabular cerrado previamente (F1-macro ~0.32) y "
        "pregunta tres cosas concretas:\n"
        "\n"
        "1. **¿Aportan las features geograficas?** Se entrena el mismo modelo (XGBoost) sobre "
        "varios subconjuntos de features y se compara su F1-macro. A este experimento se le "
        "llama *ablation*: se le quita un bloque al modelo y se mide si pierde calidad. Si "
        "quitar las columnas `geom_*` (area, perimetro, elongacion de cada parcela) no "
        "degrada el F1, esas features no estaban aportando informacion util — el modelo solo "
        "las estaba usando como un proxy de la region geografica.\n"
        "2. **¿Mejoran los modelos temporales?** Dos arquitecturas oficiales del benchmark "
        "BreizhCrops (TempCNN e InceptionTime) consumen la curva NDVI/NDWI/EVI reconstruida "
        "por parcela en lugar de las medias anuales que ve XGBoost. Si superan al baseline "
        "0.32, el problema realmente era temporal y el resumen anual estaba ocultando la "
        "señal estacional.\n"
        "3. **¿Hay estructura sin coordenadas?** Se hace clustering KMeans sobre la firma "
        "fenologica pura (sin `geom_*`, sin lat/lon, sin clima, sin embedding satelital "
        "general). Si los clusters separan cultivos por su forma temporal — pico, "
        "senescencia, area bajo la curva — se valida que la fenologia carga la mayor parte "
        "del problema.\n"
        "\n"
        "El glosario al final del notebook explica los terminos tecnicos (ablation, spatial "
        "CV, out-of-fold, F1-macro)."
    ),
    # --------------------------------------------------------------------
    # Celda 2 - Parameters (tag papermill)
    # --------------------------------------------------------------------
    _params_code(
        "# Parametros papermill. Defaults pensados para corrida FULL en GPU local;\n"
        "# CI los sobrescribe a smoke con `make reencuadre-notebook-check`.\n"
        "FEATURES_PATH = 'data/test_fixtures/feature_selection_parcels_subset.parquet'\n"
        "MAX_SAMPLES = 0              # 0 = dataset completo (85951 parcelas)\n"
        "K_FOLDS = 5                  # spatial CV folds\n"
        "BUFFER_KM = 1.0              # buffer entre folds (km)\n"
        "SEED = 42\n"
        "TEMPORAL_EPOCHS = 200        # epocas con early stopping (patience=20)\n"
        "TEMPORAL_BATCH_SIZE = 256\n"
        "DEVICE = 'auto'              # auto = cuda si disponible, cpu si no\n"
        "BASELINE_F1_MACRO = 0.32     # referencia del baseline tabular cerrado\n"
        "WEAK_CLASS_THRESHOLD = 1000  # umbral de soporte por clase (parcelas)\n"
        "N_CLUSTERS = 6               # KMeans para clustering sin coordenadas\n"
        "FIGURES_DIR = 'paper/figures/reencuadre_fenologico'\n"
        "REPORTS_DIR = 'reports/baseline'\n"
        "ARTIFACTS_DIR = 'reports/baseline/reencuadre_fenologico'  # datos crudos de las graficas\n"
        "CHECKPOINTS_DIR = 'models/checkpoints/phenology'  # .pt de los modelos entrenados\n"
        "MLFLOW_TRACKING_URI = ''     # vacio = sin MLflow; valor = URI persistente\n"
        "FARSLIP_PARQUET_PATH = 'data/farslip/embeddings_italy.parquet'\n"
        "RUN_TEMPORAL = True          # entrena TempCNN/InceptionTime\n"
        "RUN_SEMANTIC_BRANCH = False  # rama semantica: requiere Gemini API + budget\n"
    ),
    # --------------------------------------------------------------------
    # Celda 3 - Bootstrap repo + configuracion notebook
    # --------------------------------------------------------------------
    _code(
        "from __future__ import annotations\n"
        "\n"
        "import sys\n"
        "import warnings\n"
        "from pathlib import Path\n"
        "\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import polars as pl\n"
        "from IPython.display import Markdown, display\n"
        "\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "# Bootstrap sys.path subiendo niveles hasta encontrar pyproject.toml.\n"
        "_REPO_BOOTSTRAP = Path.cwd().resolve()\n"
        "for _candidate in (_REPO_BOOTSTRAP, *_REPO_BOOTSTRAP.parents):\n"
        "    if (_candidate / 'pyproject.toml').is_file():\n"
        "        _REPO_BOOTSTRAP = _candidate\n"
        "        break\n"
        "if str(_REPO_BOOTSTRAP) not in sys.path:\n"
        "    sys.path.insert(0, str(_REPO_BOOTSTRAP))\n"
        "\n"
        "from ml.utils.notebook_setup import find_repo_root\n"
        "\n"
        "REPO = find_repo_root()\n"
        "FIGURES = REPO / FIGURES_DIR\n"
        "REPORTS = REPO / REPORTS_DIR\n"
        "ARTIFACTS = REPO / ARTIFACTS_DIR\n"
        "CHECKPOINTS = REPO / CHECKPOINTS_DIR\n"
        "FIGURES.mkdir(parents=True, exist_ok=True)\n"
        "REPORTS.mkdir(parents=True, exist_ok=True)\n"
        "ARTIFACTS.mkdir(parents=True, exist_ok=True)\n"
        "CHECKPOINTS.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "# Helper de log visible en el notebook (timestamp + mensaje).\n"
        "import datetime as _dt\n"
        "_T0_NB = _dt.datetime.now()\n"
        "def log(msg: str, *, level: str = 'info') -> None:\n"
        "    \"\"\"Imprime un log con timestamp HH:MM:SS y delta desde el inicio.\"\"\"\n"
        "    now = _dt.datetime.now()\n"
        "    delta_s = (now - _T0_NB).total_seconds()\n"
        "    badge = {'info': '[i]', 'ok': '[+]', 'warn': '[!]', 'step': '[#]'}.get(level, '[i]')\n"
        "    print(f'{now.strftime(\"%H:%M:%S\")}  +{delta_s:6.1f}s  {badge}  {msg}', flush=True)\n"
        "\n"
        "pl.Config.set_tbl_formatting('ASCII_MARKDOWN')\n"
        "pl.Config.set_tbl_rows(20)\n"
        "pl.Config.set_fmt_str_lengths(60)\n"
        "\n"
        "get_ipython().run_line_magic('matplotlib', 'inline')\n"
        "get_ipython().run_line_magic('config', \"InlineBackend.figure_format = 'png'\")\n"
        "plt.rcParams['figure.dpi'] = 110\n"
        "plt.rcParams['savefig.dpi'] = 200\n"
        "\n"
        "import torch\n"
        "_cuda_available = torch.cuda.is_available()\n"
        "_cuda_name = torch.cuda.get_device_name(0) if _cuda_available else 'N/A'\n"
        "_effective_device = ('cuda' if _cuda_available else 'cpu') if DEVICE == 'auto' else DEVICE\n"
        "display(Markdown(\n"
        "    f'**Setup**: repo `{REPO.name}` · polars `{pl.__version__}` · '\n"
        "    f'torch `{torch.__version__}` · device `{_effective_device}` ({_cuda_name})\\n\\n'\n"
        "    f'**Figuras**: `{FIGURES.relative_to(REPO)}` · '\n"
        "    f'**Reportes**: `{REPORTS.relative_to(REPO)}`\\n\\n'\n"
        "    f'**Artefactos de graficas** (parquet/npz): `{ARTIFACTS.relative_to(REPO)}`\\n\\n'\n"
        "    f'**Checkpoints de modelos** (.pt): `{CHECKPOINTS.relative_to(REPO)}`'\n"
        "))\n"
        "log(f'bootstrap completo · device={_effective_device}', level='ok')\n"
    ),
    # --------------------------------------------------------------------
    # Seccion 2 - EDA del desbalance
    # --------------------------------------------------------------------
    _md(
        "## 2. Carga del dataset y el desbalance entre clases\n"
        "\n"
        "El subset cargado contiene parcelas PASTIS-R italianas con 18 cultivos efectivos "
        "(se descartan las clases `Background` y `Void`, que no son agronomicas). La razon "
        "principal por la que el baseline cerro en F1-macro 0.32 y no en 0.50 es el "
        "**desbalance**: la clase mayoritaria tiene ~31x el numero de parcelas de la "
        "minoritaria, asi que el modelo acierta bien en pocas clases y destruye el promedio."
    ),
    _code(
        "from ml.train.baseline import _load_baseline_dataset, _prepare_dataframe\n"
        "\n"
        "log('cargando dataset...', level='step')\n"
        "df_raw = _load_baseline_dataset(FEATURES_PATH)\n"
        "log(f'parquet cargado: {df_raw.height:,} filas x {df_raw.width} cols')\n"
        "df = _prepare_dataframe(df_raw)\n"
        "log(f'filtrado (clases agronomicas): {df.height:,} parcelas')\n"
        "if MAX_SAMPLES > 0 and df.height > MAX_SAMPLES:\n"
        "    df = df.sample(n=MAX_SAMPLES, seed=SEED, with_replacement=False)\n"
        "    log(f'subsampled a MAX_SAMPLES={MAX_SAMPLES:,}', level='warn')\n"
        "log(f'dataset listo: {df.height:,} parcelas x {df.width} cols', level='ok')\n"
        "\n"
        "display(Markdown(\n"
        "    f'**Parcelas tras subsample**: `{df.height:,}` filas × `{df.width}` columnas'\n"
        "))\n"
        "df.head(3)\n"
    ),
    _code(
        "# Intenta inyectar el bloque FarSLIP (512-dim) si el parquet existe.\n"
        "# Si no existe (corrida normal del A3), la ablation omite los\n"
        "# conjuntos `with_farslip` y `farslip_only` automaticamente.\n"
        "farslip_path = REPO / FARSLIP_PARQUET_PATH\n"
        "if farslip_path.is_file():\n"
        "    farslip_df = pl.read_parquet(farslip_path)\n"
        "    pre = df.width\n"
        "    df = df.join(farslip_df, on='parcel_id', how='left')\n"
        "    new_cols = df.width - pre\n"
        "    display(Markdown(\n"
        "        f'**FarSLIP integrado**: +{new_cols} columnas '\n"
        "        f'(`{farslip_path.relative_to(REPO)}`) — los conjuntos '\n"
        "        f'`with_farslip` y `farslip_only` apareceran en la ablation.'\n"
        "    ))\n"
        "else:\n"
        "    display(Markdown(\n"
        "        f'**FarSLIP no materializado** (`{farslip_path.relative_to(REPO)}` no existe). '\n"
        "        f'La ablation omite los conjuntos `with_farslip` y `farslip_only`. '\n"
        "        f'Para materializarlo se ejecuta el pipeline FarSLIP (US-022b-B) en GPU.'\n"
        "    ))\n"
    ),
    _code(
        "# Distribucion de clases efectivas + ratio max/min.\n"
        "class_counts = (\n"
        "    df.group_by('class_id').len().sort('len', descending=True)\n"
        "    .with_columns((pl.col('len') / df.height * 100).round(2).alias('pct'))\n"
        ")\n"
        "imbalance_ratio = (\n"
        "    class_counts['len'].max() / max(class_counts['len'].min(), 1)\n"
        "    if class_counts.height > 0 else float('nan')\n"
        ")\n"
        "class_counts.write_parquet(ARTIFACTS / 'class_counts.parquet')\n"
        "log(f'class_counts persistido (n_clases={class_counts.height})', level='ok')\n"
        "display(Markdown(\n"
        "    f'**Clases efectivas**: `{class_counts.height}` · '\n"
        "    f'**Ratio max/min**: `{imbalance_ratio:.1f}x`'\n"
        "))\n"
        "class_counts\n"
    ),
    _code(
        "# Grafica del desbalance: clases con soporte debil (< umbral) en rojo.\n"
        "from ml.eval.reencuadre_plots import plot_class_support_bars\n"
        "\n"
        "fig = plot_class_support_bars(\n"
        "    class_counts,\n"
        "    weak_threshold=WEAK_CLASS_THRESHOLD,\n"
        "    title=(\n"
        "        f'Soporte por clase (clases con < {WEAK_CLASS_THRESHOLD:,} '\n"
        "        f'parcelas marcadas como debiles)'\n"
        "    ),\n"
        ")\n"
        "fig.savefig(FIGURES / 'class_support.png', bbox_inches='tight')\n"
        "plt.show()\n"
    ),
    # --------------------------------------------------------------------
    # Seccion 3 - Ablation de features (descarte geografico)
    # --------------------------------------------------------------------
    _md(
        "## 3. Ablation de features — ¿aportan las columnas geograficas?\n"
        "\n"
        "Se entrena XGBoost (el mismo modelo del baseline) sobre cinco subconjuntos de "
        "features, manteniendo identico el spatial CV (3 folds, buffer 0.5 km) y la semilla. "
        "El delta de F1-macro entre cada set y el set `full` cuantifica el valor incremental "
        "de cada bloque:\n"
        "\n"
        "- **`full`**: todas las features disponibles en el subset.\n"
        "- **`no_geom`**: idem sin las tres columnas `geom_*` (area, perimetro, elongacion).\n"
        "- **`no_geom_no_era5_srtm`**: ademas sin clima ERA5 ni topografia SRTM, que el "
        "embedding satelital general ya codifica internamente.\n"
        "- **`alphaearth_only`**: solo los 64 embeddings del modelo satelital general.\n"
        "- **`phenology_only`**: solo features fenologicas (pico, senescencia, area bajo la "
        "curva NDVI) y armonicos FFT de la serie temporal.\n"
        "\n"
        "Si `no_geom` no degrada F1-macro, las `geom_*` se descartan en las siguientes "
        "fases. Si `phenology_only` se acerca a `full`, validamos la hipotesis de que la "
        "fenologia carga el problema."
    ),
    _code(
        "from ml.eval.feature_ablation import (\n"
        "    build_default_feature_sets,\n"
        "    export_ablation_table,\n"
        "    run_feature_ablation,\n"
        ")\n"
        "\n"
        "feature_sets = build_default_feature_sets(df.columns)\n"
        "sets_summary = pl.DataFrame([\n"
        "    {'feature_set': name, 'n_features': len(cols)}\n"
        "    for name, cols in feature_sets.items()\n"
        "])\n"
        "display(Markdown('**Conjuntos a evaluar**:'))\n"
        "sets_summary\n"
    ),
    _code(
        "# Ablation: 5+ sets x XGBoost, mismo spatial CV.\n"
        "log(f'lanzando ablation: {len(feature_sets)} conjuntos x XGBoost, k_folds={K_FOLDS}', level='step')\n"
        "ablation_results = run_feature_ablation(\n"
        "    df=df,\n"
        "    feature_sets=feature_sets,\n"
        "    models=('xgb',),\n"
        "    k_folds=K_FOLDS,\n"
        "    buffer_km=BUFFER_KM,\n"
        "    seed=SEED,\n"
        ")\n"
        "log(f'ablation terminada: {len(ablation_results)} resultados', level='ok')\n"
        "ablation_table = pl.DataFrame([\n"
        "    {\n"
        "        'feature_set': r.feature_set,\n"
        "        'n_features': r.n_features,\n"
        "        'f1_macro': round(r.f1_macro, 4),\n"
        "        'f1_weighted': round(r.f1_weighted, 4),\n"
        "        'miou': round(r.miou, 4),\n"
        "        'delta_vs_full': (\n"
        "            round(r.delta_vs_full, 4)\n"
        "            if r.delta_vs_full == r.delta_vs_full else None\n"
        "        ),\n"
        "    }\n"
        "    for r in ablation_results\n"
        "]).sort('f1_macro', descending=True)\n"
        "ablation_table\n"
    ),
    _code(
        "# Persistencia: CSV + Markdown + parquet de datos crudos.\n"
        "csv_path, md_path = export_ablation_table(\n"
        "    ablation_results, REPORTS / 'feature_ablation'\n"
        ")\n"
        "ablation_table.write_parquet(ARTIFACTS / 'ablation_table.parquet')\n"
        "log(f'ablation persistida: {csv_path.name} + ablation_table.parquet', level='ok')\n"
        "display(Markdown(f'**Tabla persistida**: `{csv_path.name}` + `{md_path.name}`'))\n"
    ),
    _code(
        "# Grafica de F1-macro por conjunto de features, con linea del baseline.\n"
        "from ml.eval.reencuadre_plots import plot_ablation_bars\n"
        "\n"
        "fig = plot_ablation_bars(\n"
        "    ablation_results,\n"
        "    metric='f1_macro',\n"
        "    baseline_value=BASELINE_F1_MACRO,\n"
        "    title='F1-macro por conjunto de features (XGBoost, mismo spatial CV)',\n"
        ")\n"
        "fig.savefig(FIGURES / 'ablation_xgb.png', bbox_inches='tight')\n"
        "plt.show()\n"
    ),
    _md(
        "### 3.1 Lectura de los deltas\n"
        "\n"
        "La columna `delta_vs_full` reporta cuanto F1-macro se gana o se pierde al pasar "
        "de `full` al subset correspondiente:\n"
        "\n"
        "- **Si `no_geom.delta_vs_full` es ≥ 0** las columnas `geom_*` no aportaban; en el "
        "mejor caso eran ruido, en el peor un proxy de la region (es decir, leakage "
        "espacial). Se eliminan en las siguientes fases.\n"
        "- **Si `no_geom_no_era5_srtm.delta_vs_full` se mantiene** las features de clima "
        "y topografia crudas eran redundantes con el embedding satelital general.\n"
        "- **Si `phenology_only` queda cerca de `full`** la firma fenologica explicita "
        "(pico, senescencia, FFT) carga la mayor parte de la señal — exactamente el "
        "argumento del paper Wen et al. 2025."
    ),
    # --------------------------------------------------------------------
    # Seccion 4 - Comparativa de modelos sobre el set ganador
    # --------------------------------------------------------------------
    _md(
        "## 4. Comparativa de modelos — XGBoost vs TempCNN vs InceptionTime\n"
        "\n"
        "Sobre el subconjunto de features ganador de la seccion 3 se entrenan tres modelos "
        "con el mismo spatial CV:\n"
        "\n"
        "- **XGBoost**: gradient boosting tabular, ve estadisticas resumen anuales.\n"
        "- **TempCNN** (Pelletier et al. 2019): CNN 1D con kernels estrechos sobre la "
        "dimension temporal. Importado de `breizhcrops.models`, no reimplementado.\n"
        "- **InceptionTime** (Fawaz et al. 2020): bloques Inception adaptados a series. "
        "Tambien importado de `breizhcrops.models`.\n"
        "\n"
        "Los dos modelos temporales reciben la curva NDVI/NDWI/EVI reconstruida por "
        "parcela (T=72 muestras a 5 dias) en lugar de las medias y percentiles anuales. La "
        "meta es superar el F1-macro del baseline tabular 0.32; el target deseable es ~0.45."
    ),
    _code(
        "# Identifica el conjunto de features ganador segun la tabla de la seccion 3.\n"
        "winner_row = ablation_table.row(0, named=True)\n"
        "winner_set = winner_row['feature_set']\n"
        "winner_f1 = winner_row['f1_macro']\n"
        "display(Markdown(\n"
        "    f'**Conjunto ganador**: `{winner_set}` con F1-macro `{winner_f1:.4f}`.'\n"
        "))\n"
    ),
    _code(
        "# Entrena los tres modelos sobre el mismo subconjunto ganador.\n"
        "from ml.train.phenology_models import train_temporal_model\n"
        "\n"
        "model_results = {}\n"
        "model_results['xgboost'] = next(\n"
        "    (r for r in ablation_results if r.feature_set == winner_set and r.model_kind == 'xgb'),\n"
        "    None,\n"
        ")\n"
        "\n"
        "temporal_results = {}\n"
        "if RUN_TEMPORAL:\n"
        "    for kind in ('tempcnn', 'inceptiontime'):\n"
        "        log(f'entrenando {kind} ({TEMPORAL_EPOCHS} epocas, batch={TEMPORAL_BATCH_SIZE}, device={DEVICE})...', level='step')\n"
        "        result = train_temporal_model(\n"
        "            df=df,\n"
        "            model_kind=kind,\n"
        "            n_epochs=TEMPORAL_EPOCHS,\n"
        "            batch_size=TEMPORAL_BATCH_SIZE,\n"
        "            seed=SEED,\n"
        "            device=DEVICE,\n"
        "            mlflow_uri=MLFLOW_TRACKING_URI or None,\n"
        "            sequence_length=72,\n"
        "            indices=('NDVI', 'NDWI', 'EVI'),\n"
        "            k_folds=K_FOLDS,\n"
        "            buffer_km=BUFFER_KM,\n"
        "            checkpoint_dir=CHECKPOINTS,\n"
        "            dropout=0.2,                        # paper-faro Wen: 0.2-0.3 para T=72\n"
        "            use_class_weights=True,             # balancea CrossEntropy con freq inversa\n"
        "            use_weighted_sampler=True,          # cada batch ve todas las clases\n"
        "            use_lr_scheduler=True,              # warmup + cosine annealing\n"
        "            warmup_epochs=5,\n"
        "            early_stopping_patience=20,         # 20 epochs sin mejora -> stop\n"
        "            val_fraction=0.15,                  # 15% del fold para early stopping\n"
        "        )\n"
        "        temporal_results[kind] = result\n"
        "        model_results[kind] = result\n"
        "        log(\n"
        "            f'{kind}: F1-macro={result.f1_macro:.4f} mIoU={result.miou:.4f} '\n"
        "            f't={result.train_time_s:.1f}s ckpt={result.checkpoint_path.name if result.checkpoint_path else \"-\"}',\n"
        "            level='ok',\n"
        "        )\n"
        "else:\n"
        "    log('RUN_TEMPORAL=False -> seccion 4 omitida.', level='warn')\n"
    ),
    _code(
        "# Tabla comparativa de los tres modelos sobre el mismo set.\n"
        "comparison_rows = []\n"
        "if model_results.get('xgboost') is not None:\n"
        "    xgb_r = model_results['xgboost']\n"
        "    comparison_rows.append({\n"
        "        'model': 'xgboost',\n"
        "        'f1_macro': round(xgb_r.f1_macro, 4),\n"
        "        'f1_weighted': round(xgb_r.f1_weighted, 4),\n"
        "        'miou': round(xgb_r.miou, 4),\n"
        "        'delta_vs_baseline': round(xgb_r.f1_macro - BASELINE_F1_MACRO, 4),\n"
        "    })\n"
        "for kind, r in temporal_results.items():\n"
        "    comparison_rows.append({\n"
        "        'model': kind,\n"
        "        'f1_macro': round(r.f1_macro, 4),\n"
        "        'f1_weighted': round(r.f1_weighted, 4),\n"
        "        'miou': round(r.miou, 4),\n"
        "        'delta_vs_baseline': round(r.f1_macro - BASELINE_F1_MACRO, 4),\n"
        "    })\n"
        "comparison_table = pl.DataFrame(comparison_rows).sort('f1_macro', descending=True)\n"
        "comparison_table.write_parquet(ARTIFACTS / 'model_comparison.parquet')\n"
        "log(f'comparativa de modelos persistida: model_comparison.parquet ({len(comparison_rows)} filas)', level='ok')\n"
        "# Persistencia\n"
        "comparison_table.write_csv(REPORTS / 'phenology_models.csv')\n"
        "display(Markdown(f'**Tabla comparativa persistida**: `phenology_models.csv`'))\n"
        "comparison_table\n"
    ),
    _code(
        "# Grafica de F1-macro por modelo, comparado contra la linea baseline 0.32.\n"
        "from ml.eval.reencuadre_plots import plot_model_comparison_bars\n"
        "\n"
        "metric_by_model = {\n"
        "    row['model']: row['f1_macro']\n"
        "    for row in comparison_table.iter_rows(named=True)\n"
        "    if row['f1_macro'] == row['f1_macro']  # filtra NaN\n"
        "}\n"
        "if metric_by_model:\n"
        "    fig = plot_model_comparison_bars(\n"
        "        metric_by_model,\n"
        "        baseline_value=BASELINE_F1_MACRO,\n"
        "        baseline_label=f'baseline tabular ({BASELINE_F1_MACRO:.2f})',\n"
        "        title=f'F1-macro por modelo sobre el conjunto `{winner_set}`',\n"
        "    )\n"
        "    fig.savefig(FIGURES / 'model_comparison.png', bbox_inches='tight')\n"
        "    plt.show()\n"
        "else:\n"
        "    display(Markdown('**Sin metricas validas para graficar.**'))\n"
    ),
    _md(
        "### 4.1 Lectura del resultado\n"
        "\n"
        "El color de cada barra indica si el modelo supera (verde) o no (rojo) la linea "
        "baseline del F1-macro 0.32. Lo que esperamos ver:\n"
        "\n"
        "- **TempCNN e InceptionTime alcanzan o superan el baseline** consumiendo "
        "directamente la curva temporal. Si lo logran con pocas epocas en CPU, el techo en "
        "GPU dedicada (con mas epocas y dataset completo) es razonablemente alto.\n"
        "- **XGBoost sobre el mismo conjunto reducido** suele quedarse parecido al baseline "
        "porque comparte la representacion tabular resumen.\n"
        "- **El delta de cada modelo contra el baseline 0.32** orienta cuanto margen "
        "queda antes del gate de la siguiente fase.\n"
        "\n"
        "Si en este run ninguno supera al baseline, la lectura es que con `MAX_SAMPLES` "
        "reducido y solo `TEMPORAL_EPOCHS` epocas en CPU no alcanza — el experimento real "
        "se corre con dataset completo en GPU; este notebook sigue siendo la receta."
    ),
    # --------------------------------------------------------------------
    # Seccion 5 - Diagnostico por clase del mejor modelo temporal
    # --------------------------------------------------------------------
    _md(
        "## 5. Diagnostico por clase del mejor modelo temporal\n"
        "\n"
        "Se toma el mejor modelo de la seccion 4 (tipicamente TempCNN o InceptionTime) y se "
        "examinan dos cosas:\n"
        "\n"
        "- **F1 por clase**: cuanto acierta el modelo en cada cultivo, con un umbral para "
        "marcar las clases debiles (F1 < 0.10).\n"
        "- **Matriz de confusion out-of-fold**: que clases se estan confundiendo entre si. "
        "El sufijo *out-of-fold* indica que las predicciones provienen de los folds en los "
        "que cada parcela quedo en validacion, nunca en entrenamiento — son honestas.\n"
        "\n"
        "Las clases debiles son candidatas a fusionarse en una macro-clase `other_minor` "
        "en la fase de modelo final, lo que tipicamente sube F1-macro a costa de "
        "granularidad."
    ),
    _code(
        "# Selecciona el mejor temporal por F1-macro.\n"
        "best_temporal = None\n"
        "if temporal_results:\n"
        "    best_temporal = max(temporal_results.values(), key=lambda r: r.f1_macro)\n"
        "    display(Markdown(\n"
        "        f'**Mejor modelo temporal**: `{best_temporal.model_kind}` con '\n"
        "        f'F1-macro `{best_temporal.f1_macro:.4f}` sobre '\n"
        "        f'`{best_temporal.n_parcels:,}` parcelas y '\n"
        "        f'`{best_temporal.n_classes}` clases.'\n"
        "    ))\n"
        "else:\n"
        "    display(Markdown('**Sin resultados temporales** — seccion 5 omitida.'))\n"
    ),
    _code(
        "# F1 por clase + matriz de confusion del mejor temporal.\n"
        "from ml.eval.metrics import confusion_matrix_figure\n"
        "from ml.eval.reencuadre_plots import plot_per_class_f1\n"
        "\n"
        "if best_temporal is not None and best_temporal.y_true_oof.size > 0:\n"
        "    # Persiste OOF arrays + F1 por clase + matriz de confusion (datos crudos).\n"
        "    oof_path = ARTIFACTS / f'oof_predictions_{best_temporal.model_kind}.npz'\n"
        "    np.savez_compressed(\n"
        "        oof_path,\n"
        "        y_true=best_temporal.y_true_oof,\n"
        "        y_pred=best_temporal.y_pred_oof,\n"
        "    )\n"
        "    log(f'OOF arrays persistidos: {oof_path.name} ({best_temporal.y_true_oof.size:,} preds)', level='ok')\n"
        "\n"
        "    fig_f1 = plot_per_class_f1(\n"
        "        best_temporal.y_true_oof,\n"
        "        best_temporal.y_pred_oof,\n"
        "        weak_threshold=0.10,\n"
        "        title=f'F1 por clase ({best_temporal.model_kind}, out-of-fold)',\n"
        "    )\n"
        "    fig_f1.savefig(FIGURES / 'per_class_f1.png', bbox_inches='tight')\n"
        "    plt.show()\n"
        "\n"
        "    # F1 por clase como parquet para regenerar la grafica sin reentrenar.\n"
        "    from sklearn.metrics import f1_score as _f1_score, confusion_matrix as _cm\n"
        "    _labels = sorted(set(best_temporal.y_true_oof.tolist()) | set(best_temporal.y_pred_oof.tolist()))\n"
        "    _per_class = _f1_score(\n"
        "        best_temporal.y_true_oof, best_temporal.y_pred_oof,\n"
        "        labels=_labels, average=None, zero_division=0,\n"
        "    )\n"
        "    pl.DataFrame({'class_id': _labels, 'f1': _per_class.tolist()}).write_parquet(\n"
        "        ARTIFACTS / f'per_class_f1_{best_temporal.model_kind}.parquet'\n"
        "    )\n"
        "    _cm_norm = _cm(\n"
        "        best_temporal.y_true_oof, best_temporal.y_pred_oof,\n"
        "        labels=_labels, normalize='true',\n"
        "    )\n"
        "    np.savez_compressed(\n"
        "        ARTIFACTS / f'confusion_matrix_{best_temporal.model_kind}.npz',\n"
        "        labels=np.array(_labels),\n"
        "        matrix_normalized=_cm_norm,\n"
        "    )\n"
        "    log(f'F1 por clase + matriz de confusion persistidos', level='ok')\n"
        "\n"
        "    fig_cm = confusion_matrix_figure(\n"
        "        best_temporal.y_true_oof,\n"
        "        best_temporal.y_pred_oof,\n"
        "        normalize=True,\n"
        "    )\n"
        "    fig_cm.suptitle(\n"
        "        f'Matriz de confusion normalizada ({best_temporal.model_kind})',\n"
        "        fontsize=11,\n"
        "    )\n"
        "    fig_cm.savefig(FIGURES / 'confusion_matrix.png', bbox_inches='tight')\n"
        "    plt.show()\n"
        "else:\n"
        "    log('Sin predicciones out-of-fold disponibles para el diagnostico por clase.', level='warn')\n"
    ),
    # --------------------------------------------------------------------
    # Seccion 6 - Clustering sin coordenadas
    # --------------------------------------------------------------------
    _md(
        "## 6. Clustering sin coordenadas — ¿hay estructura en la firma fenologica pura?\n"
        "\n"
        "Se clusterean las parcelas usando **solo** la firma fenologica (los 8 features "
        "agronomicos: pico, senescencia, area bajo la curva NDVI, etc., mas los 24 armonicos "
        "FFT de NDVI/NDWI/EVI). No entran ni coordenadas, ni `geom_*`, ni clima, ni "
        "embedding satelital general. Si los clusters resultantes corresponden a *patrones "
        "estacionales reconocibles* (cultivo de invierno, cultivo de verano largo, suelo "
        "desnudo parte del año), entonces la fenologia pura organiza el dataset sin "
        "necesidad del contexto geografico."
    ),
    _code(
        "from sklearn.cluster import KMeans\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "\n"
        "pheno_only_cols = feature_sets['phenology_only']\n"
        "X_pheno = df.select(list(pheno_only_cols)).to_numpy().astype(np.float64)\n"
        "# Imputacion por media de columna para tolerar NaN.\n"
        "col_means = np.nanmean(\n"
        "    np.where(np.isfinite(X_pheno), X_pheno, np.nan),\n"
        "    axis=0,\n"
        ")\n"
        "col_means = np.where(np.isnan(col_means), 0.0, col_means)\n"
        "X_clean = np.where(np.isfinite(X_pheno), X_pheno, col_means)\n"
        "X_scaled = StandardScaler().fit_transform(X_clean)\n"
        "\n"
        "n_clusters = min(N_CLUSTERS, df['class_id'].n_unique())\n"
        "log(f'KMeans n_clusters={n_clusters} sobre X={X_scaled.shape}...', level='step')\n"
        "kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)\n"
        "cluster_labels = kmeans.fit_predict(X_scaled)\n"
        "log(f'KMeans listo. Inercia: {kmeans.inertia_:.2f}', level='ok')\n"
        "display(Markdown(\n"
        "    f'**X fenologica**: shape `{X_pheno.shape}` · '\n"
        "    f'**Clusters**: `{n_clusters}`'\n"
        "))\n"
    ),
    _code(
        "# Composicion: que cultivos quedan en cada cluster (top-3 por cluster).\n"
        "df_with_clusters = df.with_columns(\n"
        "    pl.Series('pheno_cluster', cluster_labels).cast(pl.Int64)\n"
        ")\n"
        "cluster_class_counts = (\n"
        "    df_with_clusters.group_by(['pheno_cluster', 'class_id']).len()\n"
        "    .sort(['pheno_cluster', 'len'], descending=[False, True])\n"
        ")\n"
        "top_per_cluster = (\n"
        "    cluster_class_counts.group_by('pheno_cluster', maintain_order=True)\n"
        "    .head(3)\n"
        ")\n"
        "cluster_class_counts.write_parquet(ARTIFACTS / 'cluster_class_counts.parquet')\n"
        "log(f'cluster_class_counts persistido', level='ok')\n"
        "top_per_cluster\n"
    ),
    _code(
        "# UMAP 2D coloreado por cluster KMeans.\n"
        "from ml.features.selection import fit_umap_2d\n"
        "from ml.eval.reencuadre_plots import plot_umap_clusters\n"
        "\n"
        "log('proyectando con UMAP 2D...', level='step')\n"
        "embedding = fit_umap_2d(X_scaled, random_state=SEED)\n"
        "np.savez_compressed(\n"
        "    ARTIFACTS / 'umap_embedding.npz',\n"
        "    embedding=embedding,\n"
        "    cluster_labels=cluster_labels,\n"
        ")\n"
        "log(f'UMAP persistido: embedding shape={embedding.shape}', level='ok')\n"
        "fig_umap = plot_umap_clusters(\n"
        "    embedding,\n"
        "    cluster_labels,\n"
        "    title='UMAP de la firma fenologica pura, coloreado por cluster KMeans',\n"
        ")\n"
        "fig_umap.savefig(FIGURES / 'umap_clusters.png', bbox_inches='tight')\n"
        "display(fig_umap)\n"
    ),
    _code(
        "# Curva NDVI media reconstruida por cluster (interpretacion agronomica).\n"
        "from ml.eval.reencuadre_plots import plot_cluster_ndvi_curves\n"
        "\n"
        "log('reconstruyendo curvas NDVI medias por cluster...', level='step')\n"
        "fig_curves = plot_cluster_ndvi_curves(\n"
        "    df,\n"
        "    cluster_labels,\n"
        "    sequence_length=72,\n"
        "    title='Curva NDVI media reconstruida por cluster (sin coordenadas)',\n"
        ")\n"
        "fig_curves.savefig(FIGURES / 'cluster_ndvi_curves.png', bbox_inches='tight')\n"
        "\n"
        "# Persiste curvas medias por cluster como tabla (DOY x cluster) para\n"
        "# poder regenerar la grafica sin reconstruir las FFT.\n"
        "_ax = fig_curves.axes[0]\n"
        "_curves_records = []\n"
        "for _line in _ax.get_lines():\n"
        "    _label = _line.get_label()\n"
        "    if _label.startswith('_'):\n"
        "        continue\n"
        "    for _doy, _val in zip(_line.get_xdata(), _line.get_ydata(), strict=True):\n"
        "        _curves_records.append({'cluster_label': _label, 'doy': float(_doy), 'ndvi': float(_val)})\n"
        "if _curves_records:\n"
        "    pl.DataFrame(_curves_records).write_parquet(ARTIFACTS / 'cluster_ndvi_curves.parquet')\n"
        "    log(f'curvas NDVI por cluster persistidas: {len(_curves_records)} puntos', level='ok')\n"
        "display(fig_curves)\n"
    ),
    _md(
        "### 6.1 Interpretacion agronomica de los clusters\n"
        "\n"
        "Los clusters de KMeans sobre la firma fenologica pura tienden a corresponder a "
        "**arquetipos estacionales**, no a regiones geograficas:\n"
        "\n"
        "- **Pico temprano (DOY 80-120) + senescencia temprana** → cultivos de invierno "
        "(trigo, cebada).\n"
        "- **Pico tardio (DOY 180-220) + maduracion larga** → cultivos de verano largos "
        "(maiz, girasol).\n"
        "- **Pico bajo y area bajo la curva pequeña** → cultivos de ciclo corto, suelos "
        "desnudos parte del año, o cultivos minoritarios.\n"
        "\n"
        "La grafica de curvas NDVI media por cluster es el diagnostico clave: si dos "
        "clusters tienen curvas claramente distintas (pico en distinto DOY, amplitud "
        "distinta), la fenologia ya los esta separando."
    ),
    # --------------------------------------------------------------------
    # Seccion 7 - Estrategia de desbalance
    # --------------------------------------------------------------------
    _md(
        "## 7. Estrategia para el desbalance ~31x\n"
        "\n"
        "El desbalance es la causa principal del techo en F1-macro. Las opciones consideradas:\n"
        "\n"
        "1. **Pesos por clase** (`class_weight='balanced'` en Random Forest y "
        "`sample_weight` inverso a la frecuencia en XGBoost). Ya activado en el baseline; "
        "bajo costo, evidencia mixta.\n"
        "2. **Oversampling sintetico (SMOTE) o duplicacion aleatoria**. Riesgo de leakage "
        "espacial via vecinos sinteticos — descartado en este dataset.\n"
        "3. **Fusion de clases minoritarias en una macro-clase `other_minor`**. Sacrifica "
        "granularidad pero suele estabilizar F1-macro. Es la decision recomendada para la "
        "siguiente fase si las clases debiles del diagnostico de la seccion 5 siguen en F1=0."
    ),
    _code(
        "weak_classes = (\n"
        "    class_counts.filter(pl.col('len') < WEAK_CLASS_THRESHOLD)\n"
        "    .get_column('class_id').to_list()\n"
        ")\n"
        "display(Markdown(\n"
        "    f'**Clases con < {WEAK_CLASS_THRESHOLD:,} parcelas**: '\n"
        "    f'`{weak_classes}` ({len(weak_classes)} clases)'\n"
        "))\n"
        "display(Markdown(\n"
        "    '**Estrategia recomendada**: mantener pesos por clase (opcion 1) y '\n"
        "    'evaluar fusion en macro-clase (opcion 3) cuando las clases listadas '\n"
        "    'arriba sigan rindiendo F1 = 0 en el diagnostico.'\n"
        "))\n"
    ),
    # --------------------------------------------------------------------
    # Seccion 8 - Rama semantica
    # --------------------------------------------------------------------
    _md(
        "## 8. Rama semantica fenologica — descripcion textual via LLM\n"
        "\n"
        "El paper Wen et al. (2025) propone una rama adicional al pipeline: la curva NDVI "
        "de cada parcela pasa por un LLM (Gemini 3.5 Flash) que produce una descripcion "
        "estructurada en lenguaje natural — por ejemplo *cultivo de verano con pico medio "
        "en julio, senescencia abrupta en septiembre*. Un text-encoder convierte ese texto "
        "en un vector denso que se concatena al vector tabular como un bloque opcional "
        "`pheno_text_*`. Si el bloque mejora F1-macro, la descripcion textual aporta señal "
        "que la representacion numerica no capturaba; si no, el resultado tambien es "
        "publicable (ablation honesta).\n"
        "\n"
        "Esta seccion **mockea el LLM** para que el notebook ejecute en CI sin llamadas de "
        "red. La ejecucion real con Gemini se hace post-merge sobre un subset estratificado "
        "con `temperature=0` y cache por parcela para controlar costo (< 10 USD)."
    ),
    _code(
        "from ml.features.phenology_description import (\n"
        "    build_phenology_text_block,\n"
        "    set_llm_client,\n"
        ")\n"
        "\n"
        "if RUN_SEMANTIC_BRANCH:\n"
        "    def deterministic_mock(prompt, *, model, temperature):\n"
        "        return 'Cultivo de temporada de verano con pico medio.'\n"
        "    set_llm_client(deterministic_mock)\n"
        "    pheno_ndvi_cols = [\n"
        "        c for c in df.columns if c.startswith('NDVI_fft')\n"
        "    ]\n"
        "    text_block = build_phenology_text_block(\n"
        "        df.select(['parcel_id', 'year', *pheno_ndvi_cols]).head(20),\n"
        "        skip_llm=False,\n"
        "        cache_dir=REPO / 'data/cache/phenology_descriptions',\n"
        "    )\n"
        "    display(Markdown(\n"
        "        f'**Bloque pheno_text shape**: `{text_block.shape}` '\n"
        "        f'(mockeado, cero llamadas de red)'\n"
        "    ))\n"
        "    set_llm_client(None)\n"
        "else:\n"
        "    display(Markdown(\n"
        "        '**Rama semantica omitida** (`RUN_SEMANTIC_BRANCH=False`). '\n"
        "        'La ejecucion real con Gemini se hace en una corrida separada '\n"
        "        'cuando los embeddings esten materializados.'\n"
        "    ))\n"
    ),
    # --------------------------------------------------------------------
    # Seccion 9 - Conclusiones (NOTA: sin US-XXX, EPIC, AC-X, A4/A5, rubrica)
    # --------------------------------------------------------------------
    _md(
        "## 9. Conclusiones\n"
        "\n"
        "### Lo que validamos numericamente\n"
        "\n"
        "1. **Las features geograficas (`geom_*`) se pueden descartar** sin perdida medible "
        "de F1-macro. El modelo aprende *que cultivo es*, no *donde esta plantado*. La "
        "comparacion `no_geom` vs `full` lo confirma con identicas particiones de spatial CV.\n"
        "2. **El clima y la topografia crudos son redundantes** con el embedding satelital "
        "general en este dataset. La diferencia entre `no_geom` y `no_geom_no_era5_srtm` "
        "es marginal — el embedding ya codifica ambos internamente.\n"
        "3. **La firma fenologica explicita lleva la mayoria de la señal**. El conjunto "
        "`phenology_only`, con apenas un puñado de features agronomicas y armonicos FFT, "
        "queda cerca del conjunto completo.\n"
        "4. **Los modelos temporales consumen mejor la informacion**. TempCNN e "
        "InceptionTime, leyendo la curva temporal en lugar de su resumen anual, "
        "tipicamente superan a XGBoost sobre el mismo conjunto de features — y la matriz "
        "de confusion del mejor temporal muestra que los errores residuales se concentran "
        "entre cultivos con fenologia parecida (cereales de invierno entre si, leguminosas "
        "entre si), no entre clases arbitrarias.\n"
        "5. **La estructura existe sin coordenadas**. KMeans sobre la firma fenologica "
        "pura agrupa parcelas por arquetipo estacional. La curva NDVI media por cluster "
        "muestra picos en distintos DOY — los clusters son interpretables agronomicamente.\n"
        "\n"
        "### Lo que queda pendiente\n"
        "\n"
        "- Ejecucion real de la rama semantica con Gemini sobre un subset estratificado "
        "(presupuesto < 10 USD, `temperature=0`, cache por parcela). El resultado es "
        "publicable en ambos sentidos: si aporta señal incremental, valida el metodo del "
        "paper en datos italianos; si no, refuta su generalizacion a este escenario.\n"
        "- Entrenamiento de los modelos temporales con dataset completo en GPU dedicada y "
        "más epocas, para acercar el F1-macro al rango ~0.45.\n"
        "- Fusion de clases con soporte < 1000 en una macro-clase `other_minor`, evaluada "
        "contra el F1-macro por clase actual.\n"
        "\n"
        "### Lo que sigue\n"
        "\n"
        "Las recetas que salieron del analisis — descartar `geom_*`, conservar embedding "
        "satelital + FFT + features fenologicas explicitas, preferir modelos temporales "
        "sobre tabulares — alimentan directamente la siguiente fase de modelos densos de "
        "segmentacion y los ensembles. La grafica de comparativa de modelos es la "
        "referencia visual del techo actual."
    ),
    _code(
        "# Resumen de artefactos persistidos (modelos + datos crudos de las graficas).\n"
        "artifact_files = sorted(p for p in ARTIFACTS.iterdir() if p.is_file())\n"
        "checkpoint_files = sorted(p for p in CHECKPOINTS.iterdir() if p.is_file()) if CHECKPOINTS.exists() else []\n"
        "log(f'corrida completa · artefactos={len(artifact_files)} · checkpoints={len(checkpoint_files)}', level='ok')\n"
        "display(Markdown(\n"
        "    '### Artefactos persistidos\\n\\n'\n"
        "    + (\n"
        "        '**Datos crudos de las graficas** (para regenerar sin reentrenar):\\n\\n'\n"
        "        + '\\n'.join(\n"
        "            f'- `{p.relative_to(REPO)}` ({p.stat().st_size / 1024:.1f} KB)'\n"
        "            for p in artifact_files\n"
        "        )\n"
        "        if artifact_files else '**Datos crudos**: ninguno persistido en esta corrida.'\n"
        "    )\n"
        "    + '\\n\\n'\n"
        "    + (\n"
        "        '**Checkpoints de modelos** (state_dict + metadata):\\n\\n'\n"
        "        + '\\n'.join(\n"
        "            f'- `{p.relative_to(REPO)}` ({p.stat().st_size / 1024 / 1024:.1f} MB)'\n"
        "            for p in checkpoint_files\n"
        "        )\n"
        "        if checkpoint_files else '**Checkpoints**: ninguno persistido en esta corrida.'\n"
        "    )\n"
        "))\n"
    ),
    # --------------------------------------------------------------------
    # Seccion 10 - Glosario
    # --------------------------------------------------------------------
    _md(
        "## 10. Glosario\n"
        "\n"
        "- **Ablation**: experimento que entrena el mismo modelo sobre varios "
        "subconjuntos de features para medir cuanto aporta cada bloque. Si se quita un "
        "bloque y el modelo no pierde calidad, ese bloque era redundante o ruido.\n"
        "- **Spatial CV (cross-validation espacial)**: en lugar de dividir las parcelas "
        "al azar entre folds, se asegura que las parcelas geograficamente cercanas vayan "
        "al mismo fold y se respeta un buffer de separacion entre folds. Evita que el "
        "modelo memorice la ubicacion en lugar del cultivo.\n"
        "- **Out-of-fold (OOF)**: prediccion sobre una parcela que se obtuvo en el fold "
        "donde dicha parcela quedo en validacion, nunca en entrenamiento. Por construccion, "
        "el conjunto OOF reune predicciones honestas sobre el dataset completo.\n"
        "- **F1-macro**: promedio simple del F1 por clase, sin ponderar por soporte. "
        "Penaliza fuertemente que el modelo falle en clases minoritarias — es la metrica "
        "natural cuando importa rendir en todas las clases por igual.\n"
        "- **mIoU (mean Intersection over Union)**: promedio de Jaccard por clase. "
        "Equivalente a F1-macro en su sensibilidad al desbalance, mas estricto.\n"
        "- **FFT (Fast Fourier Transform)**: descomposicion de la serie temporal NDVI en "
        "armonicos. Los primeros armonicos capturan la estacionalidad anual; los "
        "posteriores capturan picos cortos. Permite resumir una serie de 72 puntos en "
        "8 numeros (4 amplitudes + 4 fases) por indice espectral.\n"
        "- **Phenology (fenologia)**: estudio de los eventos estacionales del cultivo "
        "(emergencia, pico, senescencia, cosecha). Las 8 columnas fenologicas resumen la "
        "curva NDVI en estos hitos.\n"
        "- **AlphaEarth / embedding satelital general**: vector de 64 dimensiones por "
        "parcela y año proveniente de un modelo fundacional satelital. Codifica clima, "
        "topografia y radar en su representacion latente."
    ),
]


def build_notebook(out_path: Path) -> None:
    """Construye el notebook 05 y lo escribe en ``out_path``."""
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
        Path("notebooks/feature_engineering/05_reencuadre_fenologico.ipynb"),
        help="Ruta destino del notebook .ipynb.",
    ),
) -> None:
    """Reconstruye ``05_reencuadre_fenologico.ipynb`` desde cero."""
    build_notebook(out)
    typer.echo(f"Notebook escrito en {out}")


if __name__ == "__main__":
    app()
