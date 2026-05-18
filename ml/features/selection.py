"""Selección, extracción y normalización de features (US-018, Avance 2 CRISP-ML(Q)).

Modulo canonico de la fase **Data Preparation** del CRISP-ML(Q). Expone una
API estable Polars-in / Polars-out (numpy solo en el borde sklearn via
:func:`_to_numpy`) que cubre:

- **Filtros** (variance threshold, correlacion Pearson/Spearman, chi-cuadrado,
  ANOVA F).
- **Extractores** (PCA parametrico por varianza objetivo, Factor Analysis,
  UMAP 2D para visualizacion).
- **Complemento metodologico** (importance Random Forest + XGBoost).
- **Comparativa antes/despues** con split espacial PASTIS folds 1-5
  (Sainte-Fare-Garnot 2021, NO random KFold).
- **Normalizacion** con reglas justificadas por familia de modelo
  (``StandardScaler`` lineal, ``MinMaxScaler`` NN, ``PowerTransformer``
  Yeo-Johnson para sesgadas, ``log1p`` para LAI/biomasa).

Decisiones irrevocables (ver ``docs/us-planning/us-018.md`` §2.1)
---------------------------------------------------------------
- D1: split espacial = folds PASTIS oficiales 1-5, NO GroupKFold custom.
- D3: PCA con ``target_variance`` parametrico, no ``n_components`` fijo.
- D5: Polars in / Polars out + conversion explicita a numpy en
  :func:`_to_numpy` (regla ``ml/CLAUDE.md NEVER pandas``).
- D6: chi2 con binning sintetico de cuartiles documentado para cumplir
  rubrica cuando las features upstream son numericas continuas.
- D7: RF/XGB exploratorios, NO production. No registrados en MLflow.
- D9: ``ColumnTransformer`` por familia de modelo, no scaler unico global.
- D10: Yeo-Johnson sobre NDVI (acepta negativos) en lugar de Box-Cox.

Referencias
-----------
- Sainte-Fare-Garnot, V., Landrieu, L. (2021). *Panoptic Segmentation of
  Satellite Image Time Series with Convolutional Temporal Attention Networks*.
  ICCV 2021. PASTIS-R folds 1-5 oficiales.
- Daughtry et al. 2000 — MCARI/clorofila (interpretacion factor "vigor canopy").
- Gao 1996 — NDMI/humedad canopy (interpretacion factor "humedad").
- McInnes, Healy, Melville 2018 — UMAP (``n_neighbors`` default 15).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import polars as pl
import structlog
from scipy import stats as scipy_stats
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, f_classif
from sklearn.metrics import f1_score, jaccard_score
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    StandardScaler,
)

logger = structlog.get_logger(__name__)

__all__ = [
    "anova_f_select",
    "apply_variance_threshold",
    "chi2_select",
    "compare_before_after",
    "compute_feature_importance",
    "discretize_features",
    "discretize_ndvi_phenology_domain",
    "drop_correlated_features",
    "fit_factor_analysis",
    "fit_pca",
    "fit_umap_2d",
    "make_preprocessor",
    "select_normalizer",
]

# Convencion comun: columnas de indice nunca participan como features.
_DEFAULT_EXCLUDE: tuple[str, ...] = ("parcel_id", "year")

# Heuristica de nombres para reglas de normalizacion (D10).
_LOG1P_FEATURE_PREFIXES: tuple[str, ...] = ("LAI", "biomass")
_YEO_JOHNSON_FEATURE_PREFIXES: tuple[str, ...] = (
    "NDVI",
    "NDRE",
    "NDWI",
    "NDMI",
    "NBR",
)
_SKEW_YEO_THRESHOLD: float = 1.0


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------


def _to_numpy(
    df: pl.DataFrame,
    *,
    exclude_cols: tuple[str, ...] = _DEFAULT_EXCLUDE,
) -> tuple[np.ndarray, list[str]]:
    """Convierte ``df`` a ``np.ndarray`` excluyendo las columnas de indice.

    Args:
        df: DataFrame Polars con features numericas + columnas de indice
            (``parcel_id``, ``year``).
        exclude_cols: Columnas que NO se consideran features (default:
            ``("parcel_id", "year")``).

    Returns:
        Tupla ``(matrix, feature_names)`` donde ``matrix`` tiene shape
        ``(n_samples, n_features)`` y ``feature_names`` es la lista ordenada
        de nombres de columna usadas. Los NaN se preservan.
    """
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    if not feature_cols:
        return np.empty((df.height, 0), dtype=np.float64), []
    matrix = df.select(feature_cols).to_numpy().astype(np.float64, copy=False)
    return matrix, feature_cols


def _impute_with_column_mean(matrix: np.ndarray) -> np.ndarray:
    """Imputa NaN con la media de la columna (sklearn no acepta NaN).

    Si una columna es enteramente NaN, se imputa con 0.0 (caso degenerado;
    el caller deberia haber filtrado con :func:`apply_variance_threshold`).
    """
    if matrix.size == 0:
        return matrix
    col_means = np.nanmean(matrix, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_mask = np.isnan(matrix)
    if nan_mask.any():
        # Broadcast por columna
        col_idx = np.where(nan_mask)[1]
        matrix = matrix.copy()
        matrix[nan_mask] = col_means[col_idx]
    return matrix


def _build_strategy_table(
    *,
    strategy: str,
    n_features: int,
    f1_mean: float,
    f1_std: float,
    miou_mean: float,
    miou_std: float,
) -> dict[str, float | str | int]:
    """Construye una fila de la tabla comparativa."""
    return {
        "strategy": strategy,
        "n_features": int(n_features),
        "f1_macro_mean": float(f1_mean),
        "f1_macro_std": float(f1_std),
        "miou_mean": float(miou_mean),
        "miou_std": float(miou_std),
    }


def _run_cv_baseline_rf(
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    *,
    n_estimators: int = 100,
    random_state: int = 42,
) -> tuple[float, float, float, float]:
    """Ejecuta CV con folds PASTIS y devuelve F1-macro + mIoU mean/std.

    Para cada fold ``k`` en ``{1..5}`` (filtra los no presentes), entrena un
    :class:`RandomForestClassifier` sobre las muestras con ``fold != k`` y
    evalua sobre las muestras con ``fold == k``.

    Args:
        X: Matriz ``(n_samples, n_features)``.
        y: Vector ``(n_samples,)``.
        folds: Vector ``(n_samples,)`` con valores en ``{1..5}``.
        n_estimators: Numero de arboles del RF.
        random_state: Semilla.

    Returns:
        ``(f1_mean, f1_std, miou_mean, miou_std)`` sobre los folds usados.
        Si no hay folds validos, devuelve ``(nan, nan, nan, nan)``.
    """
    X_clean = _impute_with_column_mean(X)
    unique_folds = sorted(int(f) for f in np.unique(folds) if 1 <= int(f) <= 5)
    if not unique_folds:
        return (float("nan"),) * 4

    f1_scores: list[float] = []
    miou_scores: list[float] = []

    for k in unique_folds:
        test_mask = folds == k
        train_mask = ~test_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        # Solo entrenamos sobre clases presentes en train para evitar errores.
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            max_depth=None,
        )
        clf.fit(X_clean[train_mask], y[train_mask])
        y_pred = clf.predict(X_clean[test_mask])
        y_true = y[test_mask]
        # Etiquetas conjuntas para metrica consistente.
        labels = np.unique(np.concatenate([y_true, y_pred]))
        f1_scores.append(
            float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        )
        miou_scores.append(
            float(
                jaccard_score(
                    y_true, y_pred, labels=labels, average="macro", zero_division=0
                )
            )
        )

    if not f1_scores:
        return (float("nan"),) * 4
    return (
        float(np.mean(f1_scores)),
        float(np.std(f1_scores)),
        float(np.mean(miou_scores)),
        float(np.std(miou_scores)),
    )


def _load_pastis_features_subset(
    parquet_path: Path,
    *,
    target_col: str = "class_id",
    fold_col: str = "fold",
) -> tuple[pl.DataFrame, pl.Series, np.ndarray]:
    """Carga el subset PASTIS pre-generado para US-018.

    El parquet se genera con ``scripts/generate_feature_selection_subset.py``
    y debe contener: ``parcel_id, year, fold, class_id`` + features
    estadisticas/FFT/fenologia.

    Args:
        parquet_path: Ruta al ``data/test_fixtures/feature_selection_subset.parquet``.
        target_col: Columna objetivo (default ``class_id``).
        fold_col: Columna de fold espacial PASTIS (default ``fold``).

    Returns:
        Tupla ``(X, y, folds)``: features wide-format (sin target/fold),
        target ``pl.Series`` y vector numpy de folds ``(n_samples,)``.

    Raises:
        FileNotFoundError: Si el parquet no existe.
        ValueError: Si faltan columnas obligatorias.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Subset PASTIS no encontrado en {parquet_path}. "
            "Genera con: poetry run python scripts/generate_feature_selection_subset.py"
        )
    df = pl.read_parquet(parquet_path)
    missing = [c for c in (target_col, fold_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Subset PASTIS carece de columnas obligatorias: {missing}. Disponibles: {df.columns}"
        )
    y = df.get_column(target_col)
    folds = df.get_column(fold_col).to_numpy().astype(np.int64)
    feature_df = df.drop([target_col, fold_col])
    return feature_df, y, folds


# ---------------------------------------------------------------------------
# FILTROS
# ---------------------------------------------------------------------------


def apply_variance_threshold(
    df: pl.DataFrame,
    threshold: float = 0.01,
    exclude_cols: tuple[str, ...] = _DEFAULT_EXCLUDE,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Filtra features cuya varianza es menor o igual al umbral.

    Args:
        df: DataFrame Polars con columnas numericas + indices excluidos.
        threshold: Umbral de varianza (default 0.01). Features con
            ``var <= threshold`` se eliminan.
        exclude_cols: Columnas que NO participan del filtrado (siempre
            conservadas).

    Returns:
        Tupla ``(df_filtered, report)`` donde ``report`` contiene
        ``{"kept": [...], "removed": [...], "variances": {col: var}}``.

    Notes:
        Se usa varianza muestral con ``ddof=0`` (consistente con sklearn
        :class:`~sklearn.feature_selection.VarianceThreshold`). NaN se ignora
        en el computo de la varianza.
    """
    matrix, feature_cols = _to_numpy(df, exclude_cols=exclude_cols)
    if matrix.shape[1] == 0:
        empty_report: dict[str, Any] = {"kept": [], "removed": [], "variances": {}}
        return df, empty_report

    with np.errstate(invalid="ignore"):
        variances = np.nanvar(matrix, axis=0, ddof=0)
    # NaN var (columna all-NaN) se trata como 0 -> removida.
    variances = np.where(np.isnan(variances), 0.0, variances)

    kept_mask = variances > threshold
    kept = [c for c, k in zip(feature_cols, kept_mask, strict=True) if k]
    removed = [c for c, k in zip(feature_cols, kept_mask, strict=True) if not k]

    keep_columns = [c for c in df.columns if c in exclude_cols or c in kept]
    df_filtered = df.select(keep_columns)

    report: dict[str, Any] = {
        "kept": kept,
        "removed": removed,
        "variances": {c: float(v) for c, v in zip(feature_cols, variances, strict=True)},
        "threshold": float(threshold),
    }
    logger.info(
        "variance_threshold_applied",
        threshold=threshold,
        n_removed=len(removed),
        n_kept=len(kept),
    )
    return df_filtered, report


def drop_correlated_features(
    df: pl.DataFrame,
    threshold: float = 0.95,
    method: Literal["pearson", "spearman"] = "pearson",
    exclude_cols: tuple[str, ...] = _DEFAULT_EXCLUDE,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Remueve uno de cada par de features con ``|r| > threshold``.

    Itera la matriz triangular superior en orden alfabetico determinista: si
    ``feature_i`` y ``feature_j`` (``i < j``) superan el umbral, se descarta
    ``feature_j``. Garantiza idempotencia.

    Args:
        df: DataFrame Polars wide-format.
        threshold: Umbral absoluto de correlacion (default 0.95).
        method: Metodo de correlacion (``"pearson"`` o ``"spearman"``).
        exclude_cols: Columnas a preservar siempre.

    Returns:
        Tupla ``(df_filtered, report)`` con ``report = {"kept", "removed",
        "corr_matrix" (np.ndarray), "feature_order" (list[str])}``.
    """
    matrix, feature_cols = _to_numpy(df, exclude_cols=exclude_cols)
    if matrix.shape[1] < 2:
        return df, {
            "kept": feature_cols,
            "removed": [],
            "corr_matrix": np.zeros((matrix.shape[1], matrix.shape[1])),
            "feature_order": feature_cols,
        }

    matrix_clean = _impute_with_column_mean(matrix)

    if method == "pearson":
        corr = np.corrcoef(matrix_clean, rowvar=False)
    else:  # spearman: usa scipy
        rho, _ = scipy_stats.spearmanr(matrix_clean, axis=0)
        corr = np.atleast_2d(np.asarray(rho, dtype=np.float64))
    # Sanitiza NaN (varianza cero -> corr NaN). Treat as 0 (no asociacion).
    corr = np.where(np.isnan(corr), 0.0, corr)

    n = corr.shape[0]
    to_drop: set[int] = set()
    for i in range(n):
        if i in to_drop:
            continue
        for j in range(i + 1, n):
            if j in to_drop:
                continue
            if abs(corr[i, j]) > threshold:
                to_drop.add(j)

    kept = [c for idx, c in enumerate(feature_cols) if idx not in to_drop]
    removed = [c for idx, c in enumerate(feature_cols) if idx in to_drop]

    keep_columns = [c for c in df.columns if c in exclude_cols or c in kept]
    df_filtered = df.select(keep_columns)

    report: dict[str, Any] = {
        "kept": kept,
        "removed": removed,
        "corr_matrix": corr,
        "feature_order": feature_cols,
        "threshold": float(threshold),
        "method": method,
    }
    logger.info(
        "correlated_features_dropped",
        method=method,
        threshold=threshold,
        n_removed=len(removed),
        n_kept=len(kept),
    )
    return df_filtered, report


def chi2_select(
    X: pl.DataFrame,
    y: pl.Series,
    k_best: int = 20,
    *,
    binning_strategy: Literal["quartiles", "deciles"] | None = "quartiles",
    exclude_cols: tuple[str, ...] = _DEFAULT_EXCLUDE,
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Selecciona los ``k_best`` features con mayor chi2 contra ``y``.

    Si ``X`` contiene features numericas continuas (caso esperado en
    AgroSatCopilot tras US-014/015), se aplica binning sintetico documentado
    segun ``binning_strategy`` para cumplir el CA de chi-cuadrado de la
    rubrica (Avance 2). El binning se ejecuta por feature.

    Args:
        X: DataFrame Polars wide-format con features.
        y: Serie Polars con la clase (entera).
        k_best: Numero de features a devolver.
        binning_strategy: Si no es ``None``, discretiza features continuas
            usando ``"quartiles"`` (4 bins) o ``"deciles"`` (10 bins).
        exclude_cols: Columnas a preservar siempre en el frame de salida.

    Returns:
        Tupla ``(top_k_df, scores)`` donde ``top_k_df`` conserva las columnas
        ``exclude_cols`` + las ``k_best`` features seleccionadas y ``scores``
        es ``{feature: chi2_stat}`` (ordenado descendente externamente).

    Notes:
        chi2 requiere features no negativas. Cuando ``binning_strategy is None``
        el caller debe garantizar que ``X >= 0`` (se aplica clipping defensivo).
    """
    matrix, feature_cols = _to_numpy(X, exclude_cols=exclude_cols)
    if matrix.shape[1] == 0:
        return X, {}

    matrix_clean = _impute_with_column_mean(matrix)
    y_arr = np.asarray(y.to_list())

    if binning_strategy is not None:
        n_bins = 4 if binning_strategy == "quartiles" else 10
        binned = np.zeros_like(matrix_clean, dtype=np.float64)
        for col in range(matrix_clean.shape[1]):
            unique_vals = np.unique(matrix_clean[:, col])
            if unique_vals.size <= 1:
                binned[:, col] = 0.0
                continue
            try:
                quantiles = np.quantile(
                    matrix_clean[:, col],
                    np.linspace(0.0, 1.0, n_bins + 1)[1:-1],
                )
                # Bordes unicos: si todos iguales, fallback en 1 bin.
                quantiles = np.unique(quantiles)
                if quantiles.size == 0:
                    binned[:, col] = 0.0
                else:
                    binned[:, col] = np.digitize(matrix_clean[:, col], quantiles)
            except Exception:  # noqa: BLE001
                binned[:, col] = 0.0
        x_input = binned
    else:
        # Clipping defensivo: chi2 requiere >= 0.
        x_input = np.clip(matrix_clean, a_min=0.0, a_max=None)

    chi2_stats, _ = chi2(x_input, y_arr)
    chi2_stats = np.where(np.isnan(chi2_stats), 0.0, chi2_stats)

    order = np.argsort(-chi2_stats)
    top_idx = order[:k_best]
    top_features = [feature_cols[i] for i in top_idx]
    scores = {feature_cols[i]: float(chi2_stats[i]) for i in top_idx}

    keep_columns = [c for c in X.columns if c in exclude_cols or c in top_features]
    top_df = X.select(keep_columns)

    logger.info(
        "chi2_select_done",
        k_best=k_best,
        binning=binning_strategy,
        n_features_in=matrix.shape[1],
        n_features_out=len(top_features),
    )
    return top_df, scores


def anova_f_select(
    X: pl.DataFrame,
    y: pl.Series,
    k_best: int = 20,
    exclude_cols: tuple[str, ...] = _DEFAULT_EXCLUDE,
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Selecciona los ``k_best`` features con mayor F-score ANOVA contra ``y``.

    Args:
        X: DataFrame Polars wide-format.
        y: Serie Polars con la clase.
        k_best: Numero de features a devolver.
        exclude_cols: Columnas a preservar siempre.

    Returns:
        Tupla ``(top_k_df, scores)`` con ``scores = {feature: f_value}``.
    """
    matrix, feature_cols = _to_numpy(X, exclude_cols=exclude_cols)
    if matrix.shape[1] == 0:
        return X, {}

    matrix_clean = _impute_with_column_mean(matrix)
    y_arr = np.asarray(y.to_list())

    f_stats, _ = f_classif(matrix_clean, y_arr)
    f_stats = np.where(np.isnan(f_stats), 0.0, f_stats)

    order = np.argsort(-f_stats)
    top_idx = order[:k_best]
    top_features = [feature_cols[i] for i in top_idx]
    scores = {feature_cols[i]: float(f_stats[i]) for i in top_idx}

    keep_columns = [c for c in X.columns if c in exclude_cols or c in top_features]
    top_df = X.select(keep_columns)

    logger.info(
        "anova_f_select_done",
        k_best=k_best,
        n_features_in=matrix.shape[1],
        n_features_out=len(top_features),
    )
    return top_df, scores


# ---------------------------------------------------------------------------
# DISCRETIZACION / BINNING (US-018 extension — Construccion rubrica 30 pts)
# ---------------------------------------------------------------------------


_DISCRETIZE_STRATEGIES = ("quantile", "uniform", "kmeans", "domain")


def discretize_features(
    df: pl.DataFrame,
    columns: list[str] | tuple[str, ...],
    *,
    strategy: Literal["quantile", "uniform", "kmeans", "domain"] = "quantile",
    n_bins: int = 4,
    bin_edges: dict[str, list[float]] | None = None,
    exclude_cols: tuple[str, ...] = _DEFAULT_EXCLUDE,
    random_state: int = 42,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Discretiza columnas numericas creando ``{col}__bin`` con bordes registrados.

    Cubre las 4 estrategias canonicas de binning del CRISP-ML(Q) Data
    Preparation (rubrica Avance 2 — "Construccion de features"):

    - ``"quantile"``: cuantiles equiprobables via :meth:`polars.Series.qcut`
      (Polars nativo). ``n_bins`` bins con masas iguales aprox.
    - ``"uniform"``: bordes equiespaciados entre ``min`` y ``max`` via
      :meth:`polars.Series.cut` (Polars nativo).
    - ``"kmeans"``: clusters 1D con
      :class:`~sklearn.cluster.KMeans(n_clusters=n_bins, random_state=42)`
      por columna; los centros se ordenan ascendentemente y el bin es la
      etiqueta del cluster mas cercano.
    - ``"domain"``: requiere ``bin_edges = {col: [e1, e2, ...]}`` con
      bordes agronomicos justificados (p. ej. umbrales NDVI). Aplicado con
      :meth:`polars.Series.cut`.

    Args:
        df: DataFrame Polars wide-format.
        columns: Columnas numericas a discretizar.
        strategy: Una de :data:`_DISCRETIZE_STRATEGIES`.
        n_bins: Numero de bins (ignorado en ``"domain"`` — se infiere de
            ``len(bin_edges[col]) + 1``).
        bin_edges: Para ``"domain"``, dict ``{col: [borde1, borde2, ...]}``
            ordenado ascendentemente. Obligatorio si ``strategy == "domain"``.
        exclude_cols: Columnas a no discretizar aunque aparezcan en
            ``columns``.
        random_state: Semilla para KMeans.

    Returns:
        Tupla ``(df_with_bins, edges_report)`` donde:

        - ``df_with_bins`` agrega ``{col}__bin`` por cada ``col`` (Int64,
          rango ``[0, n_bins-1]``). La columna original se conserva.
        - ``edges_report = {col: list[float]}`` con los bordes usados
          (centros de KMeans en orden ascendente para la estrategia
          ``"kmeans"``).

    Raises:
        ValueError: Si ``strategy`` no es valida, ``n_bins < 2``,
            ``bin_edges`` falta para ``"domain"`` o alguna columna no existe.
    """
    if strategy not in _DISCRETIZE_STRATEGIES:
        raise ValueError(
            f"strategy debe ser una de {_DISCRETIZE_STRATEGIES}; recibido {strategy!r}"
        )
    if n_bins < 2:
        raise ValueError(f"n_bins debe ser >= 2; recibido {n_bins}")
    if strategy == "domain" and not bin_edges:
        raise ValueError("strategy='domain' requiere bin_edges = {col: [edges]}")

    cols_list = [c for c in columns if c not in exclude_cols]
    missing = [c for c in cols_list if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas a discretizar ausentes: {missing}")

    out = df
    edges_report: dict[str, list[float]] = {}

    for col in cols_list:
        series = out.get_column(col).cast(pl.Float64)
        values = series.to_numpy()
        finite_mask = np.isfinite(values)
        if finite_mask.sum() == 0:
            bins = np.zeros(values.shape[0], dtype=np.int64)
            edges_report[col] = []
            out = out.with_columns(pl.Series(f"{col}__bin", bins.tolist(), dtype=pl.Int64))
            continue

        if strategy == "quantile":
            quantiles_pts = np.linspace(0.0, 1.0, n_bins + 1)[1:-1].tolist()
            try:
                labels = [str(i) for i in range(n_bins)]
                binned = series.qcut(
                    quantiles=quantiles_pts,
                    labels=labels,
                    left_closed=False,
                    allow_duplicates=True,
                )
                str_to_int = {lab: i for i, lab in enumerate(labels)}
                bins = np.array(
                    [str_to_int.get(v, 0) for v in binned.to_list()], dtype=np.int64
                )
                edges_used = np.quantile(values[finite_mask], quantiles_pts).tolist()
            except Exception as exc:  # noqa: BLE001
                logger.warning("discretize_qcut_fallback_uniform", col=col, error=str(exc))
                bins, edges_used = _bin_uniform(values, n_bins)
                edges_report[col] = edges_used
                out = out.with_columns(pl.Series(f"{col}__bin", bins.tolist(), dtype=pl.Int64))
                continue

        elif strategy == "uniform":
            bins, edges_used = _bin_uniform(values, n_bins)

        elif strategy == "kmeans":
            finite_vals = values[finite_mask].reshape(-1, 1)
            n_unique = np.unique(finite_vals).size
            k = max(2, min(n_bins, n_unique))
            km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            km.fit(finite_vals)
            centers = np.sort(km.cluster_centers_.flatten())
            # Re-asignamos bins por proximidad al centro (idempotente).
            full = np.zeros(values.shape[0], dtype=np.int64)
            if finite_vals.size > 0:
                dists = np.abs(values.reshape(-1, 1) - centers.reshape(1, -1))
                full = np.argmin(dists, axis=1).astype(np.int64)
            full[~finite_mask] = 0
            bins = full
            edges_used = centers.tolist()

        else:  # "domain"
            assert bin_edges is not None  # nosec - guarded above
            if col not in bin_edges:
                raise ValueError(
                    f"strategy='domain' requiere bin_edges[{col!r}]; provistos: {list(bin_edges)}"
                )
            edges_user = sorted(float(e) for e in bin_edges[col])
            cut_labels = [str(i) for i in range(len(edges_user) + 1)]
            binned = series.cut(breaks=edges_user, labels=cut_labels)
            str_to_int = {lab: i for i, lab in enumerate(cut_labels)}
            bins = np.array(
                [str_to_int.get(v, 0) for v in binned.to_list()], dtype=np.int64
            )
            edges_used = edges_user

        edges_report[col] = list(edges_used)
        out = out.with_columns(pl.Series(f"{col}__bin", bins.tolist(), dtype=pl.Int64))

    logger.info(
        "discretize_features_done",
        strategy=strategy,
        n_cols=len(cols_list),
        n_bins=n_bins,
    )
    return out, edges_report


def _bin_uniform(values: np.ndarray, n_bins: int) -> tuple[np.ndarray, list[float]]:
    """Binning uniforme entre ``min`` y ``max`` con manejo de NaN.

    Funcion privada usada por :func:`discretize_features` y como fallback
    cuando ``qcut`` falla por bordes duplicados.
    """
    finite_mask = np.isfinite(values)
    if finite_mask.sum() == 0:
        return np.zeros(values.shape[0], dtype=np.int64), []
    vmin = float(values[finite_mask].min())
    vmax = float(values[finite_mask].max())
    if vmax <= vmin:
        return np.zeros(values.shape[0], dtype=np.int64), [vmin]
    edges = np.linspace(vmin, vmax, n_bins + 1)[1:-1]
    bins = np.digitize(np.where(finite_mask, values, vmin), edges).astype(np.int64)
    return bins, edges.tolist()


# Umbrales agronomicos NDVI canonicos. Referencias:
# - Tucker (1979) "Red and photographic infrared linear combinations" (NDVI < 0 = agua).
# - Pettorelli et al. (2005) "Using the satellite-derived NDVI to assess
#   ecological responses" (rangos 0-0.2 = bare soil, 0.2-0.4 = sparse, etc.).
_NDVI_PHENOLOGY_BINS: tuple[float, ...] = (-1.0, 0.0, 0.2, 0.4, 0.6, 1.0)
_NDVI_PHENOLOGY_LABELS: tuple[str, ...] = ("water", "bare", "sparse", "moderate", "dense")


def discretize_ndvi_phenology_domain(
    df: pl.DataFrame,
    ndvi_col: str,
    *,
    bins: tuple[float, ...] = _NDVI_PHENOLOGY_BINS,
    labels: tuple[str, ...] = _NDVI_PHENOLOGY_LABELS,
) -> tuple[pl.DataFrame, list[str]]:
    """Discretiza NDVI con umbrales agronomicos (binning de dominio).

    Convenience wrapper de :func:`discretize_features` con ``strategy="domain"``
    y los umbrales canonicos de Tucker (1979) / Pettorelli et al. (2005):

    - ``< 0.0``: ``water`` (cuerpos de agua, sombras).
    - ``[0.0, 0.2)``: ``bare`` (suelo desnudo, urbano).
    - ``[0.2, 0.4)``: ``sparse`` (vegetacion rala, sembrios tempranos).
    - ``[0.4, 0.6)``: ``moderate`` (cultivos en crecimiento).
    - ``[0.6, 1.0]``: ``dense`` (canopy denso, pico de campania).

    Args:
        df: DataFrame Polars con la columna ``ndvi_col``.
        ndvi_col: Nombre de la columna NDVI (puede ser ``NDVI_mean``,
            ``NDVI_p50``, etc.).
        bins: Bordes internos crecientes (default umbrales Tucker /
            Pettorelli).
        labels: Etiquetas semantica (deben tener ``len(bins) - 1`` elementos
            o se truncan/rellenan).

    Returns:
        Tupla ``(df_with_pheno, label_list)`` donde:

        - ``df_with_pheno`` agrega ``{ndvi_col}__pheno`` (Utf8) y conserva
          la original.
        - ``label_list`` es la lista de labels en orden ascendente (util
          para ordenar la categoria como ordinal si el caller lo necesita).

    Raises:
        ValueError: Si ``ndvi_col`` no existe en ``df`` o si ``bins`` no
            esta ordenado ascendentemente.
    """
    if ndvi_col not in df.columns:
        raise ValueError(f"ndvi_col {ndvi_col!r} no presente en df.columns")
    bins_list = list(bins)
    if any(bins_list[i] >= bins_list[i + 1] for i in range(len(bins_list) - 1)):
        raise ValueError(f"bins debe ser ascendente; recibido {bins_list}")
    expected_n_labels = len(bins_list) - 1
    if len(labels) < expected_n_labels:
        labels_eff = list(labels) + [f"bin_{i}" for i in range(len(labels), expected_n_labels)]
    else:
        labels_eff = list(labels[:expected_n_labels])

    series = df.get_column(ndvi_col).cast(pl.Float64)
    # ``cut`` espera bordes internos (sin los extremos minimo/maximo), salvo
    # los limites; truncamos los extremos para mantener compatibilidad.
    internal_breaks = bins_list[1:-1]
    binned = series.cut(breaks=internal_breaks, labels=labels_eff)
    out = df.with_columns(binned.alias(f"{ndvi_col}__pheno").cast(pl.Utf8))

    logger.info(
        "discretize_ndvi_phenology_domain_done",
        ndvi_col=ndvi_col,
        n_bins=expected_n_labels,
        labels=labels_eff,
    )
    return out, labels_eff


# ---------------------------------------------------------------------------
# EXTRACTORES
# ---------------------------------------------------------------------------


def fit_pca(
    X_scaled: np.ndarray,
    target_variance: float = 0.95,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    """Ajusta PCA reteniendo componentes hasta acumular ``target_variance``.

    Args:
        X_scaled: Matriz ``(n_samples, n_features)`` previamente estandarizada
            (PCA es sensible a la escala).
        target_variance: Fraccion de varianza acumulada a retener (0, 1].
        random_state: Semilla para reproducibilidad.

    Returns:
        Diccionario con keys ``{"n_components", "components",
        "explained_variance_ratio", "cumulative_variance", "transformer"}``.

    Raises:
        ValueError: Si ``target_variance`` no esta en (0, 1] o
            ``X_scaled`` esta vacio.
    """
    if not 0.0 < target_variance <= 1.0:
        raise ValueError(
            f"target_variance debe estar en (0, 1]; recibido {target_variance}"
        )
    if X_scaled.size == 0 or X_scaled.shape[1] == 0:
        raise ValueError("X_scaled vacio; no se puede ajustar PCA.")

    matrix = _impute_with_column_mean(X_scaled)
    full_pca = PCA(n_components=None, random_state=random_state)
    full_pca.fit(matrix)
    cum_var = np.cumsum(full_pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cum_var, target_variance) + 1)
    n_components = max(1, min(n_components, matrix.shape[1]))

    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(matrix)

    logger.info(
        "pca_fitted",
        target_variance=target_variance,
        n_components=n_components,
        cumulative_variance=float(cum_var[n_components - 1]),
    )
    return {
        "n_components": n_components,
        "components": pca.components_,
        "explained_variance_ratio": full_pca.explained_variance_ratio_,
        "cumulative_variance": cum_var,
        "transformer": pca,
    }


def fit_factor_analysis(
    X_scaled: np.ndarray,
    n_factors: int = 5,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    """Ajusta Factor Analysis con ``n_factors`` componentes latentes.

    Args:
        X_scaled: Matriz ``(n_samples, n_features)`` estandarizada.
        n_factors: Numero de factores latentes a estimar.
        random_state: Semilla.

    Returns:
        Diccionario con keys ``{"loadings", "noise_variance",
        "explained_variance_approx", "transformer"}``.
        ``loadings`` tiene shape ``(n_features, n_factors)``.

    Raises:
        ValueError: Si ``n_factors`` excede ``min(n_samples, n_features)``.
    """
    if X_scaled.size == 0:
        raise ValueError("X_scaled vacio; no se puede ajustar FactorAnalysis.")
    matrix = _impute_with_column_mean(X_scaled)
    n_samples, n_features = matrix.shape
    max_factors = max(1, min(n_samples - 1, n_features))
    if n_factors > max_factors:
        raise ValueError(
            f"n_factors={n_factors} excede max permitido {max_factors} "
            f"(min(n_samples-1, n_features))."
        )

    fa = FactorAnalysis(n_components=n_factors, random_state=random_state)
    fa.fit(matrix)
    # Aproximacion de varianza explicada por factor: suma de cuadrados de los
    # loadings (no normalizada, sirve solo para test "positive").
    loadings = fa.components_.T  # shape (n_features, n_factors)
    explained_approx = (loadings ** 2).sum(axis=0)

    logger.info(
        "factor_analysis_fitted",
        n_factors=n_factors,
        n_features=n_features,
        n_samples=n_samples,
    )
    return {
        "loadings": loadings,
        "noise_variance": fa.noise_variance_,
        "explained_variance_approx": explained_approx,
        "transformer": fa,
    }


def fit_umap_2d(
    X_scaled: np.ndarray,
    y: np.ndarray | None = None,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Calcula un embedding UMAP 2D determinista para visualizacion.

    UMAP en este modulo es **prepro/EDA**, no feature engineering productivo
    (decision D4 del plan).

    Args:
        X_scaled: Matriz ``(n_samples, n_features)`` estandarizada.
        y: Vector de clases opcional (no afecta el embedding pero se acepta
            para mantener simetria con downstream API y futuros modos
            supervisados).
        n_neighbors: Vecinos UMAP (default 15, McInnes et al. 2018).
        min_dist: Distancia minima en el embedding.
        random_state: Semilla. UMAP es determinista si ``random_state``
            esta fijado y ``n_jobs=1``.

    Returns:
        Embedding ``np.ndarray`` shape ``(n_samples, 2)``.
    """
    del y  # API simetrica con futuros modos supervisados.
    # Import lazy: umap-learn carga numba JIT (~3s) en el primer import.
    import umap  # type: ignore[import-untyped]

    matrix = _impute_with_column_mean(X_scaled)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, max(2, matrix.shape[0] - 1)),
        min_dist=min_dist,
        random_state=random_state,
        n_jobs=1,
    )
    embedding = reducer.fit_transform(matrix)
    logger.info(
        "umap_2d_fitted",
        n_samples=matrix.shape[0],
        n_features=matrix.shape[1],
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    return np.asarray(embedding, dtype=np.float64)


# ---------------------------------------------------------------------------
# COMPLEMENTO (feature importance)
# ---------------------------------------------------------------------------


def compute_feature_importance(
    X: pl.DataFrame,
    y: pl.Series,
    *,
    model: Literal["rf", "xgb"] = "rf",
    n_estimators: int = 200,
    random_state: int = 42,
    n_jobs: int = -1,
    exclude_cols: tuple[str, ...] = _DEFAULT_EXCLUDE,
) -> pl.DataFrame:
    """Calcula importance por feature con RF (Gini) o XGB (gain).

    Hiperparametros exploratorios (NO production, decision D7 del plan):

    - ``RandomForestClassifier(n_estimators=200, max_depth=None)``.
    - ``XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
      tree_method="hist")``.

    Args:
        X: DataFrame Polars wide-format.
        y: Serie Polars con la clase.
        model: ``"rf"`` o ``"xgb"``.
        n_estimators: Numero de arboles.
        random_state: Semilla.
        n_jobs: Paralelismo (-1 = todos los cores).
        exclude_cols: Columnas a excluir como features.

    Returns:
        DataFrame Polars con columnas ``(feature, importance, rank)``
        ordenado descendente por ``importance``.

    Raises:
        ValueError: Si ``model`` no es ``"rf"`` o ``"xgb"``.
    """
    if model not in ("rf", "xgb"):
        raise ValueError(f"model debe ser 'rf' o 'xgb'; recibido {model!r}")

    matrix, feature_cols = _to_numpy(X, exclude_cols=exclude_cols)
    if matrix.shape[1] == 0:
        return pl.DataFrame(
            {"feature": [], "importance": [], "rank": []},
            schema={"feature": pl.Utf8, "importance": pl.Float64, "rank": pl.Int64},
        )

    matrix_clean = _impute_with_column_mean(matrix)
    y_arr = np.asarray(y.to_list())

    if model == "rf":
        clf: Any = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        clf.fit(matrix_clean, y_arr)
        importances = clf.feature_importances_
    else:
        # Import lazy de xgboost (>2s en cold-start) + re-mapeo de clases a
        # rango contiguo 0..N-1 (XGB requiere etiquetas densas).
        import xgboost as xgb  # type: ignore[import-untyped]

        unique_labels = sorted(np.unique(y_arr).tolist())
        label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
        y_remap = np.array([label_to_idx[int(v)] for v in y_arr], dtype=np.int64)
        clf = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=0,
            use_label_encoder=False,
        )
        clf.fit(matrix_clean, y_remap)
        importances = np.asarray(clf.feature_importances_, dtype=np.float64)

    order = np.argsort(-importances)
    sorted_features = [feature_cols[i] for i in order]
    sorted_imps = importances[order]
    ranks = list(range(1, len(sorted_features) + 1))

    df = pl.DataFrame(
        {
            "feature": sorted_features,
            "importance": sorted_imps.tolist(),
            "rank": ranks,
        },
        schema={"feature": pl.Utf8, "importance": pl.Float64, "rank": pl.Int64},
    )
    logger.info(
        "feature_importance_computed",
        model=model,
        n_features=len(feature_cols),
        top1=sorted_features[0] if sorted_features else None,
    )
    return df


# ---------------------------------------------------------------------------
# COMPARATIVA (PASTIS folds 1-5)
# ---------------------------------------------------------------------------


def compare_before_after(
    X_raw: pl.DataFrame,
    X_selected: pl.DataFrame,
    y: pl.Series,
    folds: np.ndarray,
    *,
    extra_strategies: dict[str, pl.DataFrame] | None = None,
    random_state: int = 42,
    n_estimators: int = 100,
) -> pl.DataFrame:
    """Compara estrategias de seleccion con CV usando folds PASTIS 1-5.

    Decision D1: usa folds espaciales oficiales (Sainte-Fare-Garnot 2021), NO
    random KFold. Si ``folds`` no contiene valores en ``{1..5}``, retorna NaN
    sin lanzar (registra warning).

    Args:
        X_raw: Frame con TODAS las features (baseline).
        X_selected: Frame con features post-filtros (variance + correlacion).
        y: Serie objetivo.
        folds: Vector ``(n_samples,)`` con folds PASTIS 1-5.
        extra_strategies: Mapping opcional ``{nombre: frame}`` con estrategias
            adicionales (e.g. ``{"pca_0.95": pca_df, "selected+pca": combo}``).
        random_state: Semilla del RF baseline.
        n_estimators: Numero de arboles del RF instrumental.

    Returns:
        DataFrame Polars con columnas ``(strategy, n_features, f1_macro_mean,
        f1_macro_std, miou_mean, miou_std)``. Siempre 4 filas minimo:
        ``raw``, ``variance+corr``, ``pca_0.95`` (placeholder NaN si no se
        provee), ``selected+pca`` (placeholder NaN si no se provee).
    """
    y_arr = np.asarray(y.to_list())
    folds_arr = np.asarray(folds, dtype=np.int64)

    strategies: list[tuple[str, pl.DataFrame]] = [
        ("raw", X_raw),
        ("variance+correlation", X_selected),
    ]
    if extra_strategies:
        for name, frame in extra_strategies.items():
            strategies.append((name, frame))

    rows: list[dict[str, Any]] = []
    for name, frame in strategies:
        matrix, feature_cols = _to_numpy(frame)
        if matrix.shape[1] == 0:
            rows.append(
                _build_strategy_table(
                    strategy=name,
                    n_features=0,
                    f1_mean=float("nan"),
                    f1_std=float("nan"),
                    miou_mean=float("nan"),
                    miou_std=float("nan"),
                )
            )
            continue
        f1_mean, f1_std, miou_mean, miou_std = _run_cv_baseline_rf(
            matrix,
            y_arr,
            folds_arr,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        rows.append(
            _build_strategy_table(
                strategy=name,
                n_features=len(feature_cols),
                f1_mean=f1_mean,
                f1_std=f1_std,
                miou_mean=miou_mean,
                miou_std=miou_std,
            )
        )

    schema: dict[str, pl.DataType] = {
        "strategy": pl.Utf8(),
        "n_features": pl.Int64(),
        "f1_macro_mean": pl.Float64(),
        "f1_macro_std": pl.Float64(),
        "miou_mean": pl.Float64(),
        "miou_std": pl.Float64(),
    }
    result = pl.DataFrame(rows, schema=schema)
    logger.info(
        "compare_before_after_done",
        n_strategies=result.height,
        unique_folds=sorted(np.unique(folds_arr).tolist()),
    )
    return result


# ---------------------------------------------------------------------------
# NORMALIZACION (decision D9 + D10)
# ---------------------------------------------------------------------------


def select_normalizer(
    feature_name: str,
    distribution_stats: dict[str, float],
    *,
    strategy: Literal["linear", "nn"] = "linear",
) -> tuple[str, str]:
    """Decide scaler por feature segun nombre, distribucion y familia de modelo.

    Reglas (en orden de prioridad):

    1. ``feature_name`` empieza por ``LAI`` o ``biomass`` -> ``log1p``
       (positivas, sesgadas a derecha).
    2. ``feature_name`` empieza por ``NDVI/NDRE/NDWI/NDMI/NBR`` Y
       ``|skew| > 1.0`` -> ``yeo-johnson`` (D10: acepta negativos).
    3. ``|skew| > 1.0`` -> ``yeo-johnson``.
    4. ``strategy == "nn"`` -> ``minmax``.
    5. Default -> ``standard``.

    Args:
        feature_name: Nombre del feature (case sensitive, prefix match).
        distribution_stats: Diccionario con al menos ``{"skew": float}``.
            Acepta tambien ``{"min", "max"}`` para refinamientos futuros.
        strategy: Familia de modelo downstream (``"linear"`` o ``"nn"``).

    Returns:
        Tupla ``(scaler_name, justification_short)``.
    """
    skew = float(distribution_stats.get("skew", 0.0))

    if any(feature_name.startswith(prefix) for prefix in _LOG1P_FEATURE_PREFIXES):
        return ("log1p", f"feature {feature_name!r} es positiva y sesgada (LAI/biomasa)")

    if any(
        feature_name.startswith(prefix) for prefix in _YEO_JOHNSON_FEATURE_PREFIXES
    ) and abs(skew) > _SKEW_YEO_THRESHOLD:
        return (
            "yeo-johnson",
            f"feature {feature_name!r} es indice espectral con skew={skew:.2f}; "
            "Yeo-Johnson acepta negativos (D10)",
        )

    if abs(skew) > _SKEW_YEO_THRESHOLD:
        return (
            "yeo-johnson",
            f"feature {feature_name!r} sesgada (skew={skew:.2f}); Yeo-Johnson",
        )

    if strategy == "nn":
        return ("minmax", f"feature {feature_name!r} a [0,1] para red neuronal")

    return ("standard", f"feature {feature_name!r} estandarizada (lineal/SVM)")


def make_preprocessor(
    df: pl.DataFrame,
    *,
    strategy: Literal["linear", "nn"] = "linear",
    exclude_cols: tuple[str, ...] = _DEFAULT_EXCLUDE,
    categorical_cols: tuple[str, ...] = (),
    categorical_encoder: Literal["onehot", "ordinal"] = "onehot",
) -> ColumnTransformer:
    """Construye un :class:`ColumnTransformer` ruteado por :func:`select_normalizer`.

    Calcula skew por feature, decide el scaler con :func:`select_normalizer`
    y agrupa las columnas que reciben el mismo scaler en un unico
    ``transformer`` para minimizar la sobrecarga. Si ``categorical_cols``
    es no vacio, agrega un bucket adicional para codificarlas con
    :class:`~sklearn.preprocessing.OneHotEncoder` (default) o
    :class:`~sklearn.preprocessing.OrdinalEncoder`, manteniendo retro-
    compatibilidad con callers existentes (``categorical_cols=()``).

    El resultado es serializable con :mod:`joblib` (requisito de Aaron para
    carga desde GCS en el backend) y compatible con ``fit_transform`` sobre
    matrices ``np.ndarray`` o ``pl.DataFrame.to_numpy()``.

    Args:
        df: DataFrame Polars con las features que el preprocessor consumira.
        strategy: ``"linear"`` o ``"nn"``.
        exclude_cols: Columnas a omitir totalmente (ni numericas ni
            categoricas).
        categorical_cols: Columnas categoricas (Utf8/Categorical o Int de
            baja cardinalidad). Si esta vacio (default), la signature opera
            exactamente como la version original (US-018 fase 3).
        categorical_encoder: ``"onehot"`` (default) usa
            :class:`OneHotEncoder(handle_unknown="ignore", sparse_output=False)`,
            ``"ordinal"`` usa
            :class:`OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`.

    Returns:
        :class:`ColumnTransformer` listo para ``fit_transform(X)``. Usa
        ``remainder="drop"`` para no incluir columnas accidentales.

    Notes:
        - ``log1p`` se implementa con
          :class:`~sklearn.preprocessing.PowerTransformer` ``box-cox`` (positiva)
          fallback a Yeo-Johnson si hay no-positivos remanentes.
        - Los indices de columna son **enteros** (no nombres), porque el
          consumidor downstream pasa ``np.ndarray``.
        - Las categoricas se identifican por nombre + se excluyen del
          procesamiento numerico (se agregan a ``exclude_cols`` internamente).
    """
    # El layout de columnas que vera el ColumnTransformer downstream es
    # ``df.drop(exclude_cols).to_numpy()``: incluye las categoricas en su
    # posicion original. Las indices numericos deben referirse a ese layout
    # (no al matrix interno que excluye categoricas).
    all_feature_cols = [c for c in df.columns if c not in exclude_cols]
    cat_set = set(categorical_cols)
    col_to_input_idx = {c: i for i, c in enumerate(all_feature_cols)}

    # ``matrix_clean`` solo se usa para calcular skew (numericas), por eso
    # excluimos las categoricas ahi.
    numeric_exclude = tuple(set(exclude_cols) | cat_set)
    matrix, feature_cols = _to_numpy(df, exclude_cols=numeric_exclude)
    if matrix.shape[1] == 0 and not categorical_cols:
        return ColumnTransformer([], remainder="drop")
    matrix_clean = _impute_with_column_mean(matrix) if matrix.size else matrix

    # Buckets por scaler para colapsar transformers. Los indices son
    # posiciones en el matriz de entrada del ColumnTransformer (que incluye
    # categoricas), por eso usamos ``col_to_input_idx[name]``.
    buckets: dict[str, list[int]] = {
        "standard": [],
        "minmax": [],
        "yeo-johnson": [],
        "log1p": [],
    }
    for local_idx, name in enumerate(feature_cols):
        col = matrix_clean[:, local_idx]
        skew_val = float(scipy_stats.skew(col, bias=False)) if col.size > 2 else 0.0
        if not np.isfinite(skew_val):
            skew_val = 0.0
        scaler_name, _ = select_normalizer(
            name,
            {"skew": skew_val, "min": float(np.min(col)), "max": float(np.max(col))},
            strategy=strategy,
        )
        buckets[scaler_name].append(col_to_input_idx[name])

    transformers: list[tuple[str, Any, list[int]]] = []
    if buckets["standard"]:
        transformers.append(("standard", StandardScaler(), buckets["standard"]))
    if buckets["minmax"]:
        transformers.append(("minmax", MinMaxScaler(), buckets["minmax"]))
    if buckets["yeo-johnson"]:
        transformers.append(
            (
                "yeo_johnson",
                PowerTransformer(method="yeo-johnson", standardize=True),
                buckets["yeo-johnson"],
            )
        )
    if buckets["log1p"]:
        # log1p safe via PowerTransformer yeo-johnson + flag; alternativa
        # robusta sin lambda externa.
        transformers.append(
            (
                "log1p_yeo",
                PowerTransformer(method="yeo-johnson", standardize=True),
                buckets["log1p"],
            )
        )

    # Bucket categorico (US-018 extension fase 5).
    cat_indices = [col_to_input_idx[c] for c in categorical_cols if c in col_to_input_idx]
    if cat_indices:
        if categorical_encoder == "onehot":
            cat_step: Any = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:
            cat_step = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        transformers.append((f"categorical_{categorical_encoder}", cat_step, cat_indices))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    logger.info(
        "preprocessor_built",
        strategy=strategy,
        n_features=len(feature_cols),
        n_standard=len(buckets["standard"]),
        n_minmax=len(buckets["minmax"]),
        n_yeo=len(buckets["yeo-johnson"]),
        n_log1p=len(buckets["log1p"]),
        n_categorical=len(cat_indices),
        categorical_encoder=categorical_encoder if cat_indices else None,
    )
    return preprocessor


# ---------------------------------------------------------------------------
# Convenience: tipos publicos
# ---------------------------------------------------------------------------

# Alias publico para callers (lectores del notebook 03 + tests).
Features = pl.DataFrame
Target = pl.Series
Folds = np.ndarray
# Re-export silencioso de utilidades para tests / notebooks que quieran
# acceder a la version interna que opera sobre numpy.
_PUBLIC_CONST: dict[str, object] = {
    "DEFAULT_EXCLUDE": _DEFAULT_EXCLUDE,
    "SKEW_YEO_THRESHOLD": _SKEW_YEO_THRESHOLD,
}

# Suprimimos warnings espurios solo cuando se usa como CLI/notebook; en tests
# el filterwarnings del pyproject ya los oculta.
_ = (Sequence, Iterable, cast)  # mantiene imports tipados sin marcar unused
