"""Entrenamiento del modelo de referencia tabular (Avance 3, EPIC 4).

Modulo canonico de la fase **Modeling** del CRISP-ML(Q) para el baseline:
entrena tres clasificadores sobre features tabulares (embeddings AlphaEarth
o estadisticas espectro-temporales PASTIS-R) con validacion cruzada
**espacial** (folds disjuntos, NO random split):

- :class:`~sklearn.dummy.DummyClassifier` — piso del azar (estrategia
  ``stratified``). Cualquier modelo util debe superarlo con holgura.
- :class:`~sklearn.ensemble.RandomForestClassifier` — bagging de arboles,
  importance Gini nativa, robusto a escala y a features irrelevantes.
- ``xgboost.XGBClassifier`` — gradient boosting, importance por ganancia,
  estado del arte en clasificacion tabular.

Caracteristicas
---------------
- **Spatial CV** con la columna ``fold`` de los datos (folds oficiales
  PASTIS-R 1-5) o, en su defecto, ``ml.features.spatial_split`` — nunca
  ``KFold`` aleatorio (evita leakage espacial entre vecinos).
- **GridSearchCV ligero** sobre una rejilla pequena (decision: el subset es
  reducido, una rejilla amplia sobreajustaria el tuning).
- **MLflow con file store local**: ``mlflow.set_tracking_uri`` apunta a un
  directorio ``mlruns/`` del repo. Jamas requiere un servidor en marcha.
  Los runs registran ``params``, ``metrics`` y los tags obligatorios
  ``data_version`` y ``code_version``.
- Polars in / numpy en el borde sklearn.

Uso CLI (Typer)::

    poetry run python -m ml.train.train_baseline \\
        --data-path data/test_fixtures/feature_selection_subset.parquet \\
        --target-col class_id --fold-col fold

Referencias
-----------
- Breiman, L. (2001). *Random Forests*. Machine Learning 45(1), 5-32.
- Chen, T., Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*.
  KDD 2016.
- Russwurm et al. (2020). *BreizhCrops* — Random Forest es el baseline
  oficial de crop mapping multitemporal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import structlog
import typer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from ml.eval.metrics import compute_classification_metrics, summarize_cv_metrics

logger = structlog.get_logger(__name__)

__all__ = [
    "BaselineConfig",
    "BaselineResult",
    "build_alphaearth_pastis_table",
    "filter_rare_classes",
    "load_tabular_dataset",
    "spatial_cv_evaluate",
    "train_baselines",
]

app = typer.Typer(add_completion=False, help="Entrena el baseline tabular Avance 3.")

# Columnas que nunca participan como features.
_INDEX_COLS: tuple[str, ...] = ("parcel_id", "year", "px_id", "patch_id", "lon", "lat")

# Rejillas GridSearchCV ligeras: deliberadamente pequenas porque el subset
# es reducido y una rejilla amplia sobreajustaria el propio tuning.
_RF_GRID: dict[str, list[Any]] = {
    "n_estimators": [200, 400],
    "max_depth": [None, 12],
}
_XGB_GRID: dict[str, list[Any]] = {
    "n_estimators": [200, 400],
    "max_depth": [4, 6],
}


# ---------------------------------------------------------------------------
# Dataclasses de configuracion y salida
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineConfig:
    """Configuracion del entrenamiento del baseline.

    Attributes:
        target_col: Columna objetivo (clase entera).
        fold_col: Columna de fold espacial (1..k). Si no existe se construye
            un fold round-robin determinista.
        feature_prefixes: Prefijos de columna a usar como features. Si esta
            vacio, se usan todas las columnas numericas no indice.
        random_state: Semilla global.
        n_jobs: Paralelismo de los estimadores.
        grid_search: Si ``True`` ejecuta GridSearchCV ligero antes del CV.
        mlflow_experiment: Nombre del experimento MLflow.
    """

    target_col: str = "class_id"
    fold_col: str = "fold"
    feature_prefixes: tuple[str, ...] = ()
    random_state: int = 42
    n_jobs: int = -1
    grid_search: bool = True
    mlflow_experiment: str = "avance3_baseline"


@dataclass
class BaselineResult:
    """Resultado del entrenamiento de un modelo del baseline.

    Attributes:
        model_name: Identificador del modelo (``dummy`` / ``random_forest``
            / ``xgboost``).
        fold_metrics: Lista de diccionarios de metricas por fold.
        summary: DataFrame Polars con media/std por metrica.
        best_params: Hiperparametros elegidos (vacio para Dummy).
        oof_true: Etiquetas verdaderas out-of-fold concatenadas.
        oof_pred: Predicciones out-of-fold concatenadas.
        fitted_model: Estimador re-ajustado sobre TODO el dataset (para
            importance / SHAP / persistencia).
        train_seconds: Tiempo total de entrenamiento en segundos.
    """

    model_name: str
    fold_metrics: list[dict[str, float]] = field(default_factory=list)
    summary: pl.DataFrame = field(default_factory=pl.DataFrame)
    best_params: dict[str, Any] = field(default_factory=dict)
    oof_true: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    oof_pred: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    fitted_model: Any = None
    train_seconds: float = 0.0

    def f1_macro(self) -> float:
        """Devuelve el F1-macro medio out-of-fold del modelo.

        Returns:
            F1-macro promedio sobre folds (0.0 si no hay metricas).
        """
        if self.summary.height == 0:
            return 0.0
        row = self.summary.filter(pl.col("metric") == "f1_macro")
        if row.height == 0:
            return 0.0
        return float(row.get_column("mean").item())


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------


def _resolve_feature_cols(
    df: pl.DataFrame,
    *,
    target_col: str,
    fold_col: str,
    feature_prefixes: tuple[str, ...],
) -> list[str]:
    """Resuelve las columnas que actuan como features.

    Args:
        df: DataFrame de entrada.
        target_col: Columna objetivo a excluir.
        fold_col: Columna de fold a excluir.
        feature_prefixes: Si no esta vacio, solo las columnas con esos
            prefijos se consideran features.

    Returns:
        Lista ordenada de nombres de columna feature numericas.
    """
    excluded = set(_INDEX_COLS) | {target_col, fold_col}
    numeric = {
        c
        for c, dt in df.schema.items()
        if dt.is_numeric() and c not in excluded
    }
    if feature_prefixes:
        cols = [
            c
            for c in df.columns
            if c in numeric and any(c.startswith(p) for p in feature_prefixes)
        ]
    else:
        cols = [c for c in df.columns if c in numeric]
    return cols


def load_tabular_dataset(
    data_path: Path,
    config: BaselineConfig,
) -> tuple[pl.DataFrame, pl.Series, np.ndarray, list[str]]:
    """Carga un parquet tabular y separa features / target / folds.

    Args:
        data_path: Ruta al parquet (subset PASTIS o frame AlphaEarth).
        config: Configuracion con nombres de columna objetivo y fold.

    Returns:
        Tupla ``(X, y, folds, feature_cols)``:

        - ``X``: DataFrame Polars solo con las columnas feature.
        - ``y``: Serie Polars Int64 con la clase.
        - ``folds``: Vector ``np.ndarray`` Int64 de folds espaciales.
        - ``feature_cols``: Lista de nombres de feature.

    Raises:
        FileNotFoundError: Si ``data_path`` no existe.
        ValueError: Si falta la columna objetivo.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset tabular no encontrado: {data_path}")
    df = pl.read_parquet(data_path)
    if config.target_col not in df.columns:
        raise ValueError(
            f"Columna objetivo {config.target_col!r} ausente. Disponibles: {df.columns}"
        )

    feature_cols = _resolve_feature_cols(
        df,
        target_col=config.target_col,
        fold_col=config.fold_col,
        feature_prefixes=config.feature_prefixes,
    )
    if not feature_cols:
        raise ValueError("No se hallaron columnas feature numericas en el dataset.")

    y = df.get_column(config.target_col).cast(pl.Int64)
    if config.fold_col in df.columns:
        folds = df.get_column(config.fold_col).cast(pl.Int64).to_numpy()
    else:
        # Fold round-robin determinista (5 folds) como fallback documentado.
        folds = (np.arange(df.height) % 5 + 1).astype(np.int64)
        logger.warning(
            "fold_col_missing_fallback_roundrobin",
            fold_col=config.fold_col,
            n_folds=5,
        )
    X = df.select(feature_cols)
    logger.info(
        "tabular_dataset_loaded",
        path=str(data_path),
        n_samples=X.height,
        n_features=len(feature_cols),
        n_classes=int(y.n_unique()),
        unique_folds=sorted(np.unique(folds).tolist()),
    )
    return X, y, folds, feature_cols


def filter_rare_classes(
    df: pl.DataFrame,
    *,
    target_col: str = "class_id",
    min_count: int = 30,
) -> tuple[pl.DataFrame, list[int]]:
    """Descarta filas de clases con soporte menor a ``min_count``.

    Las clases extremadamente raras (p. ej. < 30 muestras) no pueden
    estimarse de forma fiable bajo CV espacial: desaparecen de algunos
    folds y arrastran el F1-macro a cero por construccion. Filtrarlas hace
    la metrica representativa de las clases efectivamente aprendibles.

    Args:
        df: DataFrame con la columna objetivo.
        target_col: Nombre de la columna de clase.
        min_count: Soporte minimo para conservar una clase.

    Returns:
        Tupla ``(df_filtrado, clases_descartadas)``.
    """
    counts = df.group_by(target_col).len()
    keep = counts.filter(pl.col("len") >= min_count).get_column(target_col).to_list()
    dropped = sorted(
        set(counts.get_column(target_col).to_list()) - set(keep)
    )
    df_out = df.filter(pl.col(target_col).is_in(keep))
    logger.info(
        "rare_classes_filtered",
        min_count=min_count,
        n_kept_classes=len(keep),
        n_dropped_classes=len(dropped),
        n_rows_in=df.height,
        n_rows_out=df_out.height,
    )
    return df_out, [int(c) for c in dropped]


def build_alphaearth_pastis_table(
    ae_parquet: Path,
    pastis_root: Path,
    *,
    n_folds: int = 5,
    min_class_count: int = 30,
    seed: int = 42,
) -> tuple[pl.DataFrame, str]:
    """Construye la tabla de entrenamiento AlphaEarth + etiquetas PASTIS.

    Une los embeddings AlphaEarth de 64 dimensiones (cacheados desde Earth
    Engine) con la clase PASTIS-R por pixel via
    :func:`ml.ingest.pastis_loader.pastis_pixel_labels` (split de ``px_id``
    por ``_``: el primer token es el ``patch_id``). El fold espacial se
    deriva del ``patch_id`` (parcelas del mismo patch nunca caen en folds
    distintos), respetando el principio de CV espacial sin leakage.

    Si el join queda vacio (patches no coinciden, PASTIS-R ausente) se
    genera un fixture sintetico determinista (``seed``) con la misma firma
    de columnas, de modo que el notebook completa la ejecucion.

    Args:
        ae_parquet: Ruta al parquet de embeddings AlphaEarth (``px_id`` +
            ``dim_00..dim_63``).
        pastis_root: Raiz del dataset PASTIS-R.
        n_folds: Numero de folds espaciales a derivar de los patches.
        min_class_count: Soporte minimo por clase (ver
            :func:`filter_rare_classes`).
        seed: Semilla del fallback sintetico.

    Returns:
        Tupla ``(df, mode)`` donde ``df`` tiene ``px_id, patch_id,
        class_id, fold`` + ``dim_00..dim_63`` y ``mode`` es ``"real"`` o
        ``"synthetic"``.
    """
    from ml.ingest.pastis_loader import pastis_pixel_labels

    mode = "synthetic"
    df: pl.DataFrame | None = None

    if ae_parquet.exists():
        ae = pl.read_parquet(ae_parquet)
        patches = sorted({p.split("_")[0] for p in ae.get_column("px_id").to_list()})
        label_frames: list[pl.DataFrame] = []
        for pid in patches:
            labels = pastis_pixel_labels(pid, root=pastis_root)
            if labels.height > 0:
                label_frames.append(labels.select(["px_id", "patch_id", "class_id"]))
        if label_frames:
            labels_df = pl.concat(label_frames, how="vertical")
            joined = ae.join(labels_df, on="px_id", how="inner")
            if joined.height > 0:
                df = joined
                mode = "real"

    if df is None:
        # Fixture sintetico determinista: 8 clases separables sobre el
        # embedding de 64 dims, 12 patches, ~3600 pixeles.
        rng = np.random.default_rng(seed)
        n_patches = 12
        n_classes = 8
        per_patch = 300
        rows: dict[str, list[Any]] = {
            "px_id": [],
            "patch_id": [],
            "class_id": [],
        }
        dim_data: dict[str, list[float]] = {f"dim_{j:02d}": [] for j in range(64)}
        class_centers = rng.normal(0.0, 1.0, size=(n_classes, 64))
        for p in range(n_patches):
            pid = str(20000 + p)
            for i in range(per_patch):
                cls = int(rng.integers(1, n_classes + 1))
                vec = class_centers[cls - 1] + rng.normal(0.0, 0.6, size=64)
                rows["px_id"].append(f"{pid}_{p * per_patch + i}")
                rows["patch_id"].append(pid)
                rows["class_id"].append(cls)
                for j in range(64):
                    dim_data[f"dim_{j:02d}"].append(float(vec[j]))
        df = pl.DataFrame({**rows, **dim_data})

    df = df.with_columns(pl.col("class_id").cast(pl.Int64))
    df, _ = filter_rare_classes(df, min_count=min_class_count)

    # Fold espacial por patch: patches ordenados, round-robin sobre n_folds.
    patch_ids = sorted(df.get_column("patch_id").unique().to_list())
    fold_map = {p: (i % n_folds) + 1 for i, p in enumerate(patch_ids)}
    df = df.with_columns(
        pl.col("patch_id").replace_strict(fold_map).cast(pl.Int64).alias("fold")
    )
    logger.info(
        "alphaearth_pastis_table_built",
        mode=mode,
        n_rows=df.height,
        n_classes=int(df.get_column("class_id").n_unique()),
        n_patches=len(patch_ids),
        n_folds=n_folds,
    )
    return df, mode


# ---------------------------------------------------------------------------
# Factory de estimadores
# ---------------------------------------------------------------------------


class XGBLabelSafeClassifier(ClassifierMixin, BaseEstimator):
    """Envoltura de ``XGBClassifier`` que re-mapea etiquetas internamente.

    ``XGBClassifier`` exige etiquetas densas y contiguas ``0..N-1``. Cuando
    un fold de CV no contiene todas las clases, el subconjunto deja de ser
    contiguo y ``fit`` falla. Esta envoltura aprende el mapeo a un rango
    contiguo en cada ``fit`` y lo revierte en ``predict``, de modo que el
    estimador encaja en :class:`~sklearn.model_selection.GridSearchCV` y en
    cualquier split espacial sin precondiciones sobre las etiquetas.

    Hereda de :class:`~sklearn.base.BaseEstimator` para que ``clone`` y la
    introspeccion de ``GridSearchCV`` funcionen sin friccion.

    Attributes:
        n_estimators: Numero de arboles del boosting.
        max_depth: Profundidad maxima de cada arbol.
        learning_rate: Tasa de aprendizaje.
        tree_method: Algoritmo de construccion de arboles.
        random_state: Semilla.
        n_jobs: Paralelismo.
        verbosity: Nivel de verbosidad de XGBoost.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        tree_method: str = "hist",
        random_state: int = 42,
        n_jobs: int = -1,
        verbosity: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.tree_method = tree_method
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self._model: Any = None
        self._to_original: dict[int, int] = {}

    def _xgb_params(self) -> dict[str, Any]:
        """Devuelve los hiperparametros a pasar al ``XGBClassifier``."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "tree_method": self.tree_method,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbosity": self.verbosity,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> XGBLabelSafeClassifier:
        """Ajusta el modelo re-mapeando ``y`` a un rango contiguo.

        Expone los atributos ajustados ``classes_`` y ``n_features_in_`` que
        sklearn exige a todo clasificador: ``GridSearchCV`` y los scorers
        acceden a ``estimator.classes_`` via ``_get_response_values``, y sin
        el atributo cada candidato del grid cae en ``error_score`` (de ahi el
        ``best_score`` degenerado a 0.0).

        Args:
            X: Matriz de features.
            y: Vector de etiquetas (no necesita ser contiguo).

        Returns:
            La instancia ajustada.
        """
        import xgboost as xgb  # type: ignore[import-untyped]

        y_arr = np.asarray(y, dtype=np.int64)
        y_dense, _, to_original = _remap_labels(y_arr)
        self._to_original = to_original
        # `_remap_labels` asigna el indice denso en orden ascendente de las
        # etiquetas originales: `classes_` (ordenado) comparte ese orden con
        # las columnas de `predict_proba` del XGBClassifier interno.
        self.classes_ = np.unique(y_arr)
        self.n_features_in_ = X.shape[1]
        self._model = xgb.XGBClassifier(**self._xgb_params())
        self._model.fit(X, y_dense)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice etiquetas en el espacio original (revierte el re-mapeo).

        Args:
            X: Matriz de features.

        Returns:
            Vector de etiquetas en el espacio original.
        """
        pred_dense = self._model.predict(X)
        return np.array(
            [self._to_original.get(int(p), -1) for p in pred_dense], dtype=np.int64
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilidades por clase con columnas alineadas a ``classes_``.

        El orden de columnas coincide con ``classes_`` porque el re-mapeo
        denso de :func:`_remap_labels` preserva el orden ascendente de las
        etiquetas originales.

        Args:
            X: Matriz de features.

        Returns:
            Matriz ``(n_samples, n_classes)`` de probabilidades.
        """
        return np.asarray(self._model.predict_proba(X), dtype=np.float64)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Importancia por ganancia del ``XGBClassifier`` subyacente."""
        return np.asarray(self._model.feature_importances_, dtype=np.float64)


def _make_estimator(model_name: str, config: BaselineConfig) -> Any:
    """Construye un estimador sklearn-compatible por nombre.

    Args:
        model_name: ``dummy`` / ``random_forest`` / ``xgboost``.
        config: Configuracion del baseline.

    Returns:
        Estimador sin ajustar.

    Raises:
        ValueError: Si ``model_name`` no es reconocido.
    """
    if model_name == "dummy":
        return DummyClassifier(strategy="stratified", random_state=config.random_state)
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=config.n_jobs,
            random_state=config.random_state,
        )
    if model_name == "xgboost":
        return XGBLabelSafeClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            random_state=config.random_state,
            n_jobs=config.n_jobs,
            verbosity=0,
        )
    raise ValueError(f"model_name desconocido: {model_name!r}")


def _impute_mean(matrix: np.ndarray) -> np.ndarray:
    """Imputa NaN con la media de columna (sklearn no acepta NaN).

    Args:
        matrix: Matriz ``(n_samples, n_features)``.

    Returns:
        Matriz sin NaN (columnas all-NaN imputadas con 0.0).
    """
    if matrix.size == 0:
        return matrix
    col_means = np.nanmean(matrix, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(matrix)
    if nan_mask.any():
        matrix = matrix.copy()
        matrix[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return matrix


def _remap_labels(y: np.ndarray) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
    """Re-mapea etiquetas a un rango contiguo 0..N-1 (requisito XGBoost).

    Args:
        y: Vector de etiquetas originales.

    Returns:
        Tupla ``(y_dense, to_dense, to_original)`` con el vector denso y los
        dos diccionarios de traduccion.
    """
    unique = sorted(int(v) for v in np.unique(y))
    to_dense = {lab: i for i, lab in enumerate(unique)}
    to_original = {i: lab for lab, i in to_dense.items()}
    y_dense = np.array([to_dense[int(v)] for v in y], dtype=np.int64)
    return y_dense, to_dense, to_original


# ---------------------------------------------------------------------------
# Validacion cruzada espacial
# ---------------------------------------------------------------------------


def spatial_cv_evaluate(
    model_name: str,
    X: pl.DataFrame,
    y: pl.Series,
    folds: np.ndarray,
    config: BaselineConfig,
) -> BaselineResult:
    """Evalua un modelo con validacion cruzada espacial (folds disjuntos).

    Para cada fold ``k``: entrena con ``fold != k`` y predice ``fold == k``.
    Las predicciones out-of-fold se concatenan para construir una matriz de
    confusion global. XGBoost requiere etiquetas densas: el re-mapeo se hace
    internamente y se revierte antes de calcular metricas.

    Args:
        model_name: ``dummy`` / ``random_forest`` / ``xgboost``.
        X: DataFrame Polars solo con features.
        y: Serie Polars con la clase.
        folds: Vector de folds espaciales (1..k).
        config: Configuracion del baseline.

    Returns:
        :class:`BaselineResult` con metricas por fold, resumen, predicciones
        out-of-fold y el modelo re-ajustado sobre todo el dataset.
    """
    started = time.perf_counter()
    matrix = _impute_mean(X.to_numpy().astype(np.float64))
    y_arr = y.to_numpy().astype(np.int64)
    folds_arr = np.asarray(folds, dtype=np.int64)
    unique_folds = sorted(int(f) for f in np.unique(folds_arr))

    fold_metrics: list[dict[str, float]] = []
    oof_true_parts: list[np.ndarray] = []
    oof_pred_parts: list[np.ndarray] = []

    for k in unique_folds:
        test_mask = folds_arr == k
        train_mask = ~test_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        estimator = _make_estimator(model_name, config)
        estimator.fit(matrix[train_mask], y_arr[train_mask])
        y_pred = np.asarray(estimator.predict(matrix[test_mask]), dtype=np.int64)
        y_test = y_arr[test_mask]
        fold_metrics.append(compute_classification_metrics(y_test, y_pred))
        oof_true_parts.append(y_test)
        oof_pred_parts.append(y_pred)

    summary = summarize_cv_metrics(fold_metrics)
    oof_true = (
        np.concatenate(oof_true_parts) if oof_true_parts else np.empty(0, dtype=np.int64)
    )
    oof_pred = (
        np.concatenate(oof_pred_parts) if oof_pred_parts else np.empty(0, dtype=np.int64)
    )

    # Re-ajuste sobre todo el dataset (para importance / SHAP / persistencia).
    fitted = _make_estimator(model_name, config)
    fitted.fit(matrix, y_arr)

    elapsed = time.perf_counter() - started
    result = BaselineResult(
        model_name=model_name,
        fold_metrics=fold_metrics,
        summary=summary,
        oof_true=oof_true,
        oof_pred=oof_pred,
        fitted_model=fitted,
        train_seconds=elapsed,
    )
    logger.info(
        "spatial_cv_done",
        model=model_name,
        n_folds=len(fold_metrics),
        f1_macro=round(result.f1_macro(), 4),
        seconds=round(elapsed, 2),
    )
    return result


# ---------------------------------------------------------------------------
# GridSearchCV ligero
# ---------------------------------------------------------------------------


def _grid_search_light(
    model_name: str,
    X: pl.DataFrame,
    y: pl.Series,
    folds: np.ndarray,
    config: BaselineConfig,
) -> dict[str, Any]:
    """Ejecuta GridSearchCV ligero usando los folds espaciales como CV.

    Args:
        model_name: ``random_forest`` o ``xgboost`` (Dummy no se tunea).
        X: DataFrame Polars con features.
        y: Serie Polars con la clase.
        folds: Vector de folds espaciales (define los splits del CV).
        config: Configuracion del baseline.

    Returns:
        Diccionario de mejores hiperparametros (vacio si no aplica o falla).
    """
    if model_name == "dummy":
        return {}
    grid = _RF_GRID if model_name == "random_forest" else _XGB_GRID
    matrix = _impute_mean(X.to_numpy().astype(np.float64))
    y_arr = y.to_numpy().astype(np.int64)
    folds_arr = np.asarray(folds, dtype=np.int64)
    unique_folds = sorted(int(f) for f in np.unique(folds_arr))
    if len(unique_folds) < 2:
        return {}

    # Splits espaciales explicitos: (train_idx, test_idx) por fold.
    splits = [
        (np.where(folds_arr != k)[0], np.where(folds_arr == k)[0])
        for k in unique_folds
    ]
    estimator = _make_estimator(model_name, config)
    try:
        search = GridSearchCV(
            estimator,
            grid,
            scoring="f1_macro",
            cv=splits,
            n_jobs=config.n_jobs,
            refit=False,
            error_score=0.0,
        )
        search.fit(matrix, y_arr)
        logger.info(
            "grid_search_done",
            model=model_name,
            best_params=search.best_params_,
            best_score=round(float(search.best_score_), 4),
        )
        return dict(search.best_params_)
    except Exception as exc:  # noqa: BLE001
        logger.warning("grid_search_failed", model=model_name, error=str(exc))
        return {}


# ---------------------------------------------------------------------------
# MLflow (file store local)
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    """Devuelve el SHA corto del commit actual (o ``unknown``).

    Returns:
        SHA git de 7 caracteres o ``unknown`` si no es un repo git.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _log_to_mlflow(
    results: dict[str, BaselineResult],
    *,
    config: BaselineConfig,
    data_path: Path,
    mlruns_dir: Path,
) -> None:
    """Registra los resultados del baseline en MLflow (file store local).

    No requiere servidor: ``mlflow.set_tracking_uri`` apunta a un directorio
    ``mlruns/`` del repo. Cada modelo es un run con sus params, metrics y los
    tags obligatorios ``data_version`` (hash del parquet) y ``code_version``
    (git SHA).

    Args:
        results: Diccionario ``{model_name: BaselineResult}``.
        config: Configuracion del baseline.
        data_path: Ruta del dataset (para calcular ``data_version``).
        mlruns_dir: Directorio del file store MLflow.
    """
    import hashlib

    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow_not_installed_skipping_tracking")
        return

    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(mlruns_dir.resolve().as_uri())
    mlflow.set_experiment(config.mlflow_experiment)

    data_version = "missing"
    if data_path.exists():
        data_version = hashlib.sha256(data_path.read_bytes()).hexdigest()[:12]
    code_version = _git_sha()

    for name, res in results.items():
        with mlflow.start_run(run_name=f"baseline_{name}"):
            mlflow.set_tag("data_version", data_version)
            mlflow.set_tag("code_version", code_version)
            mlflow.set_tag("model_family", name)
            mlflow.set_tag("cv_strategy", "spatial_kfold")
            mlflow.log_param("dataset", data_path.name)
            mlflow.log_param("target_col", config.target_col)
            mlflow.log_param("fold_col", config.fold_col)
            for pk, pv in res.best_params.items():
                mlflow.log_param(f"best_{pk}", pv)
            if res.summary.height > 0:
                for row in res.summary.iter_rows(named=True):
                    metric = str(row["metric"])
                    mlflow.log_metric(f"{metric}_mean", float(row["mean"]))
                    mlflow.log_metric(f"{metric}_std", float(row["std"]))
            mlflow.log_metric("train_seconds", res.train_seconds)
    logger.info(
        "mlflow_logged",
        experiment=config.mlflow_experiment,
        n_runs=len(results),
        data_version=data_version,
        code_version=code_version,
    )


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------


def train_baselines(
    X: pl.DataFrame,
    y: pl.Series,
    folds: np.ndarray,
    *,
    config: BaselineConfig | None = None,
    models: tuple[str, ...] = ("dummy", "random_forest", "xgboost"),
    data_path: Path | None = None,
    mlruns_dir: Path | None = None,
    log_mlflow: bool = True,
) -> dict[str, BaselineResult]:
    """Entrena los modelos del baseline con spatial CV y registra en MLflow.

    Args:
        X: DataFrame Polars solo con features.
        y: Serie Polars con la clase.
        folds: Vector de folds espaciales.
        config: Configuracion. Si ``None`` usa :class:`BaselineConfig` default.
        models: Tupla de modelos a entrenar.
        data_path: Ruta del dataset (para ``data_version`` en MLflow).
        mlruns_dir: Directorio del file store MLflow. Si ``None`` usa
            ``mlruns/`` junto al CWD.
        log_mlflow: Si ``True`` registra los runs en MLflow.

    Returns:
        Diccionario ``{model_name: BaselineResult}``.
    """
    cfg = config or BaselineConfig()
    results: dict[str, BaselineResult] = {}

    for name in models:
        if cfg.grid_search and name != "dummy":
            best = _grid_search_light(name, X, y, folds, cfg)
        else:
            best = {}
        res = spatial_cv_evaluate(name, X, y, folds, cfg)
        res.best_params = best
        results[name] = res

    if log_mlflow:
        _log_to_mlflow(
            results,
            config=cfg,
            data_path=data_path or Path("unknown.parquet"),
            mlruns_dir=mlruns_dir or (Path.cwd() / "mlruns"),
        )
    return results


# ---------------------------------------------------------------------------
# CLI Typer
# ---------------------------------------------------------------------------


@app.command()
def main(
    data_path: Path = typer.Option(
        Path("data/test_fixtures/feature_selection_subset.parquet"),
        "--data-path",
        help="Ruta al parquet tabular de entrada.",
    ),
    target_col: str = typer.Option("class_id", "--target-col", help="Columna objetivo."),
    fold_col: str = typer.Option("fold", "--fold-col", help="Columna de fold espacial."),
    feature_prefix: list[str] = typer.Option(
        [], "--feature-prefix", help="Prefijo(s) de columna feature (vacio = todas)."
    ),
    grid_search: bool = typer.Option(
        True, "--grid-search/--no-grid-search", help="Ejecutar GridSearchCV ligero."
    ),
    mlflow_tracking: bool = typer.Option(
        True, "--mlflow/--no-mlflow", help="Registrar runs en MLflow file store."
    ),
    mlruns_dir: Path = typer.Option(
        Path("mlruns"), "--mlruns-dir", help="Directorio del file store MLflow."
    ),
) -> None:
    """Entrena el baseline tabular (Dummy + RandomForest + XGBoost).

    Imprime el resumen de metricas por modelo. El veredicto de viabilidad se
    interpreta en el notebook ``notebooks/baseline/Avance3.Equipo17.ipynb``.
    """
    config = BaselineConfig(
        target_col=target_col,
        fold_col=fold_col,
        feature_prefixes=tuple(feature_prefix),
        grid_search=grid_search,
    )
    X, y, folds, feature_cols = load_tabular_dataset(data_path, config)
    logger.info(
        "baseline_cli_start",
        n_samples=X.height,
        n_features=len(feature_cols),
    )
    results = train_baselines(
        X,
        y,
        folds,
        config=config,
        data_path=data_path,
        mlruns_dir=mlruns_dir,
        log_mlflow=mlflow_tracking,
    )
    for name, res in results.items():
        f1 = res.f1_macro()
        logger.info(
            "baseline_cli_result",
            model=name,
            f1_macro=round(f1, 4),
            n_folds=len(res.fold_metrics),
            seconds=round(res.train_seconds, 2),
        )


if __name__ == "__main__":
    app()
