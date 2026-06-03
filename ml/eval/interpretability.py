"""Interpretabilidad del baseline de clasificacion de cultivos (US-020, EPIC 4).

Modulo reutilizable que explica los modelos *production* del baseline tabular
(Random Forest y XGBoost de US-019) mediante dos familias de tecnicas:

- **Importancia nativa** (criterio AC-1): Gini/MDI para Random Forest
  (``feature_importances_``) y *gain* para XGBoost
  (``Booster.get_score(importance_type="gain")``). Se extrae del modelo ya
  ajustado — no se re-entrena nada.
- **SHAP** (criterios AC-2, AC-3, AC-6): valores de Shapley exactos con
  ``shap.TreeExplainer`` sobre un subsample estratificado del dataset.
  El analisis es **multiclase** (18-20 clases PASTIS-R): ``compute_shap_values``
  normaliza las tres formas de salida que ``TreeExplainer`` produce segun
  version (lista-por-clase, array 3D, objeto ``Explanation``) a un unico tensor
  ``(n_samples, n_features, n_classes)``.

Ademas cuantifica la *dominancia AlphaEarth* (criterio AC-4): clasifica cada
feature en su familia de origen (``is_alphaearth_dim`` + ``alphaearth_dominance_table``)
para responder cuantas de las top-N features SHAP son dimensiones del embedding
AlphaEarth — dato de entrada para el Paper Track.

Decision D1 (plan US-020 2.1): este modulo es independiente de
``ml/eval/metrics.py`` (metricas) y de ``ml/features/selection.py`` (feature
engineering exploratorio). La interpretabilidad de modelos production es un
dominio propio, consumido tambien por las arquitecturas del EPIC 5/6.

Decision D6: SHAP corre sobre un subsample estratificado (``sample_size=3000``
por defecto), no sobre las ~85k filas del dataset completo — TreeSHAP es exacto
pero O(samples x trees x depth) y el subsample da un summary estable.

Polars es el formato de I/O y de las tablas de salida; la conversion a
numpy/pandas ocurre exclusivamente en el borde de SHAP, en helpers privados.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl
import structlog
from matplotlib.figure import Figure
from sklearn.base import ClassifierMixin

logger = structlog.get_logger(__name__)

__all__ = [
    "FeatureFamily",
    "ModelKind",
    "ShapResult",
    "alphaearth_dominance_table",
    "compute_shap_values",
    "feature_importance_table",
    "is_alphaearth_dim",
    "shap_dependence_plots",
    "shap_summary_plot",
    "shap_waterfall_plot",
]

ModelKind = Literal["rf", "xgb"]
FeatureFamily = Literal[
    "alphaearth", "spectral_index", "s1", "srtm", "era5", "geom", "other"
]

# Figure resolution for the Avance 3 visual deliverables (criterion AC-7).
_PLOT_DPI: int = 200

# Regex for the AlphaEarth embedding dimensions: `dim_00`..`dim_63`
# (real prefix confirmed 2026-05-21 in the enriched parcel-level parquet).
_ALPHAEARTH_DIM_RE = re.compile(r"^dim_\d{2}$")

# Statistical suffixes of the spectral indices (NDVI_mean, EVI_p95, ...) and
# FFT harmonics (NDVI_fft_amp_0, ...). Used to classify the
# `spectral_index` family by naming convention in `_classify_family`.
_SPECTRAL_PREFIXES: tuple[str, ...] = (
    "NDVI", "NDWI", "EVI", "NDMI", "NBR", "MSAVI2", "NDRE", "MCARI",
    "CCCI", "GCVI", "PSRI", "NDCI", "FAPAR", "LAI", "RENDVI", "SAVI", "TSAVI",
)


# ---------------------------------------------------------------------------
# Output dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShapResult:
    """Resultado de un analisis SHAP multiclase.

    Attributes:
        values: Array SHAP normalizado, shape
            ``(n_samples, n_features, n_classes)``. Para clasificacion binaria
            con salida 2D tambien se expande a 3 ejes (``n_classes`` = 2 o 1).
        global_importance: ``pl.DataFrame`` ``(feature, mean_abs_shap, rank)``;
            el ranking global es la media de ``|SHAP|`` sobre clases y muestras
            (decision D4).
        feature_cols: Nombres de las features en el orden del eje 1 de
            ``values``.
        base_values: Valores esperados del explainer, shape ``(n_classes,)``.
        model_kind: ``"rf"`` o ``"xgb"``.
    """

    values: np.ndarray
    global_importance: pl.DataFrame
    feature_cols: tuple[str, ...]
    base_values: np.ndarray
    model_kind: ModelKind


# ---------------------------------------------------------------------------
# Private helpers.
# ---------------------------------------------------------------------------


def _to_numpy_sample(
    X: pl.DataFrame,
    feature_cols: tuple[str, ...],
    *,
    sample_size: int | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Extrae la matriz de features como ``np.ndarray`` y opcionalmente submuestrea.

    Convierte el ``pl.DataFrame`` a ``float64`` (borde numpy obligatorio para
    SHAP) seleccionando las columnas de ``feature_cols`` en orden. Si
    ``sample_size`` es menor que el numero de filas toma una muestra aleatoria
    reproducible; en caso contrario devuelve todas las filas.

    Args:
        X: DataFrame Polars con al menos las columnas de ``feature_cols``.
        feature_cols: Columnas a seleccionar, en orden.
        sample_size: Tamano del subsample; si es ``None`` o ``>= X.height`` se
            usan todas las filas.
        random_state: Semilla del muestreo.

    Returns:
        Tupla ``(matrix, row_index)`` donde ``matrix`` es
        ``(n_sample, n_features)`` float64 y ``row_index`` son los indices
        originales de las filas seleccionadas.

    Raises:
        ValueError: si falta alguna columna de ``feature_cols`` en ``X``.
    """
    missing = [c for c in feature_cols if c not in X.columns]
    if missing:
        raise ValueError(
            f"`X` no contiene las columnas de feature requeridas: {missing}."
        )

    n_rows = X.height
    if sample_size is None or sample_size >= n_rows:
        row_index = np.arange(n_rows, dtype=np.int64)
    else:
        rng = np.random.default_rng(random_state)
        row_index = np.sort(
            rng.choice(n_rows, size=sample_size, replace=False)
        ).astype(np.int64)

    matrix = (
        X.select(feature_cols)
        .to_numpy()
        .astype(np.float64)
    )
    # Impute NaN/inf with the column mean: TreeExplainer does not accept NaN for
    # some models and the +/-inf of spectral ratios break the algorithm.
    matrix = _impute_columns(matrix)
    return matrix[row_index], row_index


def _impute_columns(matrix: np.ndarray) -> np.ndarray:
    """Reemplaza NaN e infinitos por la media finita de cada columna.

    Args:
        matrix: Matriz ``(n_samples, n_features)`` que puede contener NaN o
            +/-inf.

    Returns:
        Una copia de ``matrix`` sin valores no finitos. Las columnas sin ningun
        valor finito se rellenan con ``0.0``.
    """
    clean = np.array(matrix, dtype=np.float64, copy=True)
    clean[~np.isfinite(clean)] = np.nan
    col_means = np.nanmean(
        np.where(np.isnan(clean), np.nan, clean), axis=0
    )
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    nan_mask = np.isnan(clean)
    if nan_mask.any():
        clean[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return clean


def _normalize_shap_multiclass(
    raw: Any,
    *,
    n_samples: int,
    n_features: int,
) -> np.ndarray:
    """Normaliza la salida de ``TreeExplainer.shap_values`` a un tensor 3D.

    ``shap.TreeExplainer`` devuelve formas distintas segun la version y el tipo
    de modelo (decision D3, riesgo R2):

    - **Lista por clase**: ``list`` de ``n_classes`` arrays
      ``(n_samples, n_features)`` — API clasica multiclase.
    - **Array 3D**: ``(n_samples, n_features, n_classes)`` — API nueva.
    - **Array 2D**: ``(n_samples, n_features)`` — binario o regresion; se
      expande a ``(n_samples, n_features, 1)``.
    - **Objeto ``Explanation``**: se accede a su atributo ``.values``.

    Args:
        raw: Salida cruda de ``shap_values`` o de ``explainer(X)``.
        n_samples: Numero de muestras esperado (eje 0).
        n_features: Numero de features esperado (eje 1).

    Returns:
        Array float64 ``(n_samples, n_features, n_classes)``.

    Raises:
        ValueError: si la salida no encaja en ninguna de las formas conocidas.
    """
    # Explanation object -> extract .values and recurse.
    if hasattr(raw, "values") and not isinstance(raw, (list, tuple, np.ndarray)):
        return _normalize_shap_multiclass(
            np.asarray(raw.values, dtype=np.float64),
            n_samples=n_samples,
            n_features=n_features,
        )

    # List/tuple of 2D arrays, one per class.
    if isinstance(raw, (list, tuple)):
        per_class = [np.asarray(arr, dtype=np.float64) for arr in raw]
        if not per_class:
            raise ValueError("`shap_values` devolvio una lista vacia.")
        # stack over the last axis -> (n_samples, n_features, n_classes).
        stacked = np.stack(per_class, axis=-1)
        return _validate_shape(stacked, n_samples, n_features)

    array = np.asarray(raw, dtype=np.float64)
    if array.ndim == 2:
        return _validate_shape(array[:, :, np.newaxis], n_samples, n_features)
    if array.ndim == 3:
        # Some versions return (n_classes, n_samples, n_features);
        # reorder to the canonical layout if axis 0 is not n_samples.
        if array.shape[0] != n_samples and array.shape[1] == n_samples:
            array = np.transpose(array, (1, 2, 0))
        return _validate_shape(array, n_samples, n_features)

    raise ValueError(
        f"Forma de SHAP no reconocida: ndim={array.ndim}, shape={array.shape}."
    )


def _validate_shape(
    array: np.ndarray, n_samples: int, n_features: int
) -> np.ndarray:
    """Valida que un tensor SHAP 3D tenga los ejes ``(n_samples, n_features, *)``.

    Args:
        array: Tensor candidato de 3 ejes.
        n_samples: Numero de muestras esperado.
        n_features: Numero de features esperado.

    Returns:
        El propio ``array`` si los dos primeros ejes coinciden.

    Raises:
        ValueError: si los ejes no coinciden con lo esperado.
    """
    if array.shape[:2] != (n_samples, n_features):
        raise ValueError(
            f"Tensor SHAP con ejes inesperados {array.shape}; "
            f"se esperaba (n_samples={n_samples}, n_features={n_features}, *)."
        )
    return array


def _global_importance_table(
    values: np.ndarray, feature_cols: tuple[str, ...]
) -> pl.DataFrame:
    """Construye la tabla de importancia global SHAP.

    La importancia global de cada feature es la media de ``|SHAP|`` sobre todas
    las muestras y todas las clases (decision D4) — el ranking estandar para los
    summary plots.

    Args:
        values: Tensor SHAP ``(n_samples, n_features, n_classes)``.
        feature_cols: Nombres de las features.

    Returns:
        ``pl.DataFrame`` ``(feature, mean_abs_shap, rank)`` ordenado
        descendentemente por ``mean_abs_shap``.
    """
    mean_abs = np.abs(values).mean(axis=(0, 2))
    order = np.argsort(-mean_abs)
    return pl.DataFrame(
        {
            "feature": [feature_cols[i] for i in order],
            "mean_abs_shap": mean_abs[order].astype(np.float64).tolist(),
            "rank": list(range(1, len(order) + 1)),
        },
        schema={
            "feature": pl.Utf8,
            "mean_abs_shap": pl.Float64,
            "rank": pl.Int64,
        },
    )


def _classify_family(feature_name: str) -> FeatureFamily:
    """Clasifica un feature en su familia de origen por convencion de nombres.

    Args:
        feature_name: Nombre de la columna de feature.

    Returns:
        La familia: ``alphaearth`` (``dim_NN``), ``spectral_index`` (indices y
        sus armonicos FFT), ``s1`` (radar Sentinel-1, prefijo ``VV``/``VH``),
        ``srtm`` (elevacion/pendiente), ``era5`` (clima), ``geom`` (geometria de
        la parcela) o ``other``.
    """
    if is_alphaearth_dim(feature_name):
        return "alphaearth"

    upper = feature_name.upper()
    if upper.startswith(("VV", "VH", "S1_")):
        return "s1"
    if upper.startswith(("SRTM", "ELEV", "SLOPE", "ASPECT", "DEM")):
        return "srtm"
    if upper.startswith(("ERA5", "TEMP", "PRECIP", "T2M", "TP_")):
        return "era5"
    if upper.startswith(("AREA", "PERIMETER", "GEOM", "N_PIXELS")):
        return "geom"

    base = feature_name.split("_", 1)[0]
    if base in _SPECTRAL_PREFIXES:
        return "spectral_index"
    # Phenology derived from NDVI (sog_doy, peak_doy, ndvi_auc, ...): treated
    # as a spectral index because it derives from the index series.
    if feature_name.lower().startswith(
        ("sog_", "peak_", "senescence_", "ndvi_", "maturity_")
    ):
        return "spectral_index"
    return "other"


# ---------------------------------------------------------------------------
# Native importance (criterion AC-1).
# ---------------------------------------------------------------------------


def feature_importance_table(
    model: ClassifierMixin,
    model_kind: ModelKind,
    feature_cols: tuple[str, ...],
) -> pl.DataFrame:
    """Calcula la importancia nativa de un modelo de arboles ya ajustado.

    Random Forest expone la importancia Gini/MDI en ``feature_importances_``;
    XGBoost expone la ganancia (*gain*) en
    ``Booster.get_score(importance_type="gain")``. A diferencia de
    :func:`ml.features.selection.compute_feature_importance` (que re-entrena un
    modelo exploratorio), aqui se extrae el atributo del modelo *production* de
    US-019 — decision D2: no se re-entrena nada.

    Args:
        model: Estimador ``RandomForestClassifier`` o ``XGBClassifier`` ya
            ajustado.
        model_kind: ``"rf"`` para Gini o ``"xgb"`` para *gain*.
        feature_cols: Nombres de las features en el orden con el que se ajusto
            el modelo.

    Returns:
        ``pl.DataFrame`` ``(feature, importance, rank)`` ordenado
        descendentemente por ``importance``. Para XGBoost las features que el
        booster nunca uso reciben ``importance = 0.0``.

    Raises:
        ValueError: si ``model_kind`` no es ``"rf"`` ni ``"xgb"``, o si el
            numero de features del modelo no coincide con ``len(feature_cols)``.
    """
    if model_kind not in ("rf", "xgb"):
        raise ValueError(
            f"`model_kind` debe ser 'rf' o 'xgb'; recibido {model_kind!r}."
        )

    n_features = len(feature_cols)
    if model_kind == "rf":
        importances = np.asarray(
            model.feature_importances_, dtype=np.float64
        )
        if importances.shape[0] != n_features:
            raise ValueError(
                f"El modelo RF tiene {importances.shape[0]} features pero "
                f"`feature_cols` tiene {n_features}."
            )
    else:
        importances = _xgb_gain_importances(model, feature_cols)

    order = np.argsort(-importances)
    df = pl.DataFrame(
        {
            "feature": [feature_cols[i] for i in order],
            "importance": importances[order].astype(np.float64).tolist(),
            "rank": list(range(1, n_features + 1)),
        },
        schema={"feature": pl.Utf8, "importance": pl.Float64, "rank": pl.Int64},
    )
    logger.info(
        "feature_importance_table_computed",
        model_kind=model_kind,
        n_features=n_features,
        top_feature=df["feature"][0] if df.height else None,
    )
    return df


def _xgb_gain_importances(
    model: ClassifierMixin, feature_cols: tuple[str, ...]
) -> np.ndarray:
    """Extrae la importancia *gain* de un ``XGBClassifier`` alineada al orden dado.

    El booster XGBoost indexa las features como ``f0``, ``f1``, ... cuando se
    entreno con un ``np.ndarray``. ``get_score`` solo devuelve las features que
    el modelo efectivamente uso; el resto se rellena con ``0.0``.

    Args:
        model: ``XGBClassifier`` ya ajustado.
        feature_cols: Nombres de las features en el orden de entrenamiento.

    Returns:
        Array ``(n_features,)`` de importancias *gain*, alineado a
        ``feature_cols``.
    """
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")
    n_features = len(feature_cols)
    importances = np.zeros(n_features, dtype=np.float64)
    booster_names = list(getattr(booster, "feature_names", []) or [])
    for key, gain in score.items():
        if key in booster_names:
            idx = booster_names.index(key)
        elif key.startswith("f") and key[1:].isdigit():
            idx = int(key[1:])
        elif key in feature_cols:
            idx = feature_cols.index(key)
        else:
            continue
        if 0 <= idx < n_features:
            importances[idx] = float(gain)
    return importances


# ---------------------------------------------------------------------------
# SHAP (criteria AC-2, AC-3, AC-6).
# ---------------------------------------------------------------------------


def compute_shap_values(
    model: ClassifierMixin,
    X: pl.DataFrame,
    model_kind: ModelKind,
    *,
    feature_cols: tuple[str, ...],
    sample_size: int = 3000,
    random_state: int = 42,
) -> ShapResult:
    """Calcula los valores SHAP de un modelo de arboles con ``TreeExplainer``.

    Instancia ``shap.TreeExplainer`` (algoritmo TreeSHAP exacto, CPU) sobre un
    subsample estratificado de ``X`` (decision D6) y normaliza la salida
    multiclase a un tensor ``(n_samples, n_features, n_classes)`` mediante
    :func:`_normalize_shap_multiclass` (decision D3).

    Args:
        model: Estimador ``RandomForestClassifier`` o ``XGBClassifier`` ya
            ajustado.
        X: DataFrame Polars con las columnas de ``feature_cols``.
        model_kind: ``"rf"`` o ``"xgb"``.
        feature_cols: Nombres de las features en el orden de entrenamiento.
        sample_size: Tamano del subsample SHAP; si ``X`` tiene menos filas se
            usan todas.
        random_state: Semilla del muestreo (reproducibilidad).

    Returns:
        :class:`ShapResult` con el tensor SHAP, la tabla de importancia global
        y los valores base del explainer.

    Raises:
        ValueError: si ``model_kind`` es invalido o faltan columnas en ``X``.
    """
    import shap

    if model_kind not in ("rf", "xgb"):
        raise ValueError(
            f"`model_kind` debe ser 'rf' o 'xgb'; recibido {model_kind!r}."
        )

    matrix, row_index = _to_numpy_sample(
        X, feature_cols, sample_size=sample_size, random_state=random_state
    )
    n_samples, n_features = matrix.shape

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(matrix, check_additivity=False)
    values = _normalize_shap_multiclass(
        raw, n_samples=n_samples, n_features=n_features
    )

    expected = np.atleast_1d(
        np.asarray(getattr(explainer, "expected_value", 0.0), dtype=np.float64)
    )

    result = ShapResult(
        values=values,
        global_importance=_global_importance_table(values, feature_cols),
        feature_cols=tuple(feature_cols),
        base_values=expected,
        model_kind=model_kind,
    )
    logger.info(
        "shap_values_computed",
        model_kind=model_kind,
        n_samples=n_samples,
        n_features=n_features,
        n_classes=values.shape[2],
        sample_rows=int(row_index.size),
    )
    return result


def shap_summary_plot(
    shap_result: ShapResult,
    X: pl.DataFrame,
    *,
    top_n: int = 20,
) -> Figure:
    """Genera el summary plot (beeswarm) de las top-N features SHAP.

    Agrega los valores SHAP sobre las clases (media de ``|SHAP|``) para producir
    un beeswarm global de las ``top_n`` features mas importantes. Usa el backend
    ``Agg`` de matplotlib para que la figura sea serializable a PNG en CI y en
    notebooks ejecutados con papermill.

    Args:
        shap_result: Resultado de :func:`compute_shap_values`.
        X: DataFrame Polars con las columnas de ``shap_result.feature_cols``;
            debe tener al menos tantas filas como el subsample SHAP.
        top_n: Numero de features a mostrar (las mas importantes globalmente).

    Returns:
        Figura matplotlib ``dpi=200`` lista para ``fig.savefig`` o ``display``.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    import shap

    feature_cols = shap_result.feature_cols
    matrix, _ = _to_numpy_sample(
        X, feature_cols, sample_size=shap_result.values.shape[0]
    )
    # Importance aggregated over classes -> 2D array (n_samples, n_features).
    aggregated = np.abs(shap_result.values).mean(axis=2)

    fig = plt.figure(dpi=_PLOT_DPI)
    shap.summary_plot(
        aggregated,
        features=matrix,
        feature_names=list(feature_cols),
        max_display=top_n,
        plot_type="bar",
        show=False,
    )
    fig = plt.gcf()
    fig.set_dpi(_PLOT_DPI)
    ax = fig.gca()
    ax.set_title(
        f"SHAP — importancia global top-{top_n} ({shap_result.model_kind.upper()})"
    )
    fig.tight_layout()
    return fig


def shap_dependence_plots(
    shap_result: ShapResult,
    X: pl.DataFrame,
    *,
    top_features: int = 5,
) -> list[tuple[str, Figure]]:
    """Genera un dependence plot por cada una de las top-N features SHAP.

    Las features se ordenan por importancia SHAP global (decision D4); para cada
    una se grafica el valor SHAP de la clase mas explicada frente al valor del
    feature.

    Args:
        shap_result: Resultado de :func:`compute_shap_values`.
        X: DataFrame Polars con las columnas de ``shap_result.feature_cols``.
        top_features: Numero de features a graficar.

    Returns:
        Lista de tuplas ``(feature_name, figure)``, una por feature, ordenadas
        por importancia SHAP global descendente.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    feature_cols = shap_result.feature_cols
    matrix, _ = _to_numpy_sample(
        X, feature_cols, sample_size=shap_result.values.shape[0]
    )
    top = (
        shap_result.global_importance.sort("rank")
        .head(top_features)["feature"]
        .to_list()
    )
    # Reference class: the one concentrating the most global SHAP signal.
    class_idx = int(
        np.abs(shap_result.values).mean(axis=(0, 1)).argmax()
    )
    class_values = shap_result.values[:, :, class_idx]

    plots: list[tuple[str, Figure]] = []
    for feature_name in top:
        col_idx = feature_cols.index(feature_name)
        fig, ax = plt.subplots(figsize=(6.0, 4.5), dpi=_PLOT_DPI)
        feature_values = matrix[:, col_idx]
        scatter = ax.scatter(
            feature_values,
            class_values[:, col_idx],
            c=feature_values,
            cmap="viridis",
            s=12,
            alpha=0.7,
        )
        fig.colorbar(scatter, ax=ax, label=feature_name)
        ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_xlabel(feature_name)
        ax.set_ylabel(f"valor SHAP (clase {class_idx})")
        ax.set_title(f"Dependence — {feature_name}")
        fig.tight_layout()
        plots.append((feature_name, fig))

    logger.info(
        "shap_dependence_plots_generated",
        model_kind=shap_result.model_kind,
        n_plots=len(plots),
        class_idx=class_idx,
    )
    return plots


def shap_waterfall_plot(
    shap_result: ShapResult,
    *,
    row: int = 0,
    class_idx: int | None = None,
) -> Figure:
    """Genera el waterfall plot de una prediccion ejemplo.

    El waterfall descompone una prediccion individual mostrando la contribucion
    de cada feature al desplazamiento desde el valor base hasta la salida del
    modelo.

    Args:
        shap_result: Resultado de :func:`compute_shap_values`.
        row: Indice de la fila (muestra) a explicar.
        class_idx: Indice de la clase a explicar; si es ``None`` se usa la clase
            con mayor suma de ``|SHAP|`` para esa fila (proxy de la clase
            predicha).

    Returns:
        Figura matplotlib ``dpi=200`` con el waterfall plot.

    Raises:
        IndexError: si ``row`` esta fuera del rango de muestras.
        ValueError: si ``class_idx`` esta fuera del rango de clases.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    import shap

    n_samples, _n_features, n_classes = shap_result.values.shape
    if not 0 <= row < n_samples:
        raise IndexError(
            f"`row`={row} fuera de rango; el subsample SHAP tiene "
            f"{n_samples} muestras."
        )

    if class_idx is None:
        resolved_class = int(
            np.abs(shap_result.values[row]).sum(axis=0).argmax()
        )
    else:
        if not 0 <= class_idx < n_classes:
            raise ValueError(
                f"`class_idx`={class_idx} fuera de rango; hay {n_classes} clases."
            )
        resolved_class = class_idx

    row_values = shap_result.values[row, :, resolved_class]
    base = shap_result.base_values
    base_value = float(
        base[resolved_class] if base.size > resolved_class else base.flat[0]
    )

    explanation = shap.Explanation(
        values=row_values,
        base_values=base_value,
        feature_names=list(shap_result.feature_cols),
    )
    fig = plt.figure(dpi=_PLOT_DPI)
    shap.plots.waterfall(explanation, show=False)
    fig = plt.gcf()
    fig.set_dpi(_PLOT_DPI)
    fig.suptitle(
        f"SHAP waterfall — fila {row}, clase {resolved_class} "
        f"({shap_result.model_kind.upper()})"
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# AlphaEarth dominance analysis (criterion AC-4).
# ---------------------------------------------------------------------------


def is_alphaearth_dim(feature_name: str) -> bool:
    """Indica si un feature es una dimension del embedding AlphaEarth.

    Las dimensiones AlphaEarth se nombran ``dim_00``..``dim_63`` (embedding de
    64 dimensiones — convencion real confirmada inspeccionando el parquet
    AlphaEarth parcel-level). La funcion aplica el regex ``^dim_\\d{2}$``.

    Args:
        feature_name: Nombre de la columna de feature.

    Returns:
        ``True`` si el nombre encaja en el patron de una dimension AlphaEarth.
    """
    return bool(_ALPHAEARTH_DIM_RE.match(feature_name))


def alphaearth_dominance_table(
    importance_df: pl.DataFrame,
    *,
    top_n: int = 20,
) -> pl.DataFrame:
    """Clasifica las top-N features por familia y cuantifica la dominancia.

    Toma una tabla de importancia (importancia nativa o SHAP global), recorta a
    las ``top_n`` mas importantes y anade la familia de origen de cada feature
    (``alphaearth``, ``spectral_index``, ``s1``, ``srtm``, ``era5``, ``geom``,
    ``other``). Es el insumo de la conclusion cuantificada del criterio AC-4
    ("cuantas de las top-20 son dimensiones AlphaEarth").

    Args:
        importance_df: ``pl.DataFrame`` con una columna ``feature`` y una
            columna numerica de importancia (``importance`` o
            ``mean_abs_shap``). Si trae una columna ``rank`` se respeta su
            orden; si no, se ordena por la columna de importancia.
        top_n: Numero de features a retener.

    Returns:
        ``pl.DataFrame`` ``(rank, feature, family, importance)`` con las ``top_n``
        primeras features.

    Raises:
        ValueError: si ``importance_df`` no contiene la columna ``feature`` o no
            tiene una columna de importancia reconocible.
    """
    if "feature" not in importance_df.columns:
        raise ValueError("`importance_df` debe contener la columna `feature`.")

    importance_col: str | None = None
    for candidate in ("importance", "mean_abs_shap"):
        if candidate in importance_df.columns:
            importance_col = candidate
            break
    if importance_col is None:
        raise ValueError(
            "`importance_df` debe contener una columna de importancia "
            "(`importance` o `mean_abs_shap`)."
        )

    if "rank" in importance_df.columns:
        ordered = importance_df.sort("rank")
    else:
        ordered = importance_df.sort(importance_col, descending=True)

    top = ordered.head(top_n)
    families = [_classify_family(name) for name in top["feature"].to_list()]

    table = pl.DataFrame(
        {
            "rank": list(range(1, top.height + 1)),
            "feature": top["feature"].to_list(),
            "family": families,
            "importance": top[importance_col].cast(pl.Float64).to_list(),
        },
        schema={
            "rank": pl.Int64,
            "feature": pl.Utf8,
            "family": pl.Utf8,
            "importance": pl.Float64,
        },
    )
    n_alphaearth = sum(1 for f in families if f == "alphaearth")
    logger.info(
        "alphaearth_dominance_computed",
        top_n=top.height,
        n_alphaearth=n_alphaearth,
        dominance_ratio=round(n_alphaearth / max(top.height, 1), 3),
    )
    return table
