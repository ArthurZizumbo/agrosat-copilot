"""Curvas de aprendizaje/validacion y diagnostico de sub/sobreajuste (US-021, EPIC 4).

Modulo reutilizable consumido por el baseline tabular (RF/XGB de US-019) y, mas
adelante, por las arquitecturas del EPIC 5/6 para diagnosticar el sub/sobreajuste
de cualquier estimador sklearn. Expone tres familias de funciones:

- **Curva de aprendizaje** (criterio AC-1): :func:`plot_learning_curve` envuelve
  :func:`sklearn.model_selection.learning_curve` para trazar accuracy de train y
  validacion frente al numero de muestras de entrenamiento. Las curvas
  RE-ENTRENAN estimadores frescos por cada punto (decision D3) — no cargan los
  joblib de produccion de US-019.
- **Curva de validacion** (criterio AC-2): :func:`plot_validation_curve` envuelve
  :func:`sklearn.model_selection.validation_curve` para trazar accuracy frente a
  un hiperparametro critico (``max_depth``, ``n_estimators``, ``learning_rate``).
- **Diagnostico** (criterio AC-4): :func:`diagnose_fit` deriva un veredicto
  ``overfit``/``underfit``/``good_fit`` del resultado de una curva de aprendizaje
  con umbrales parametricos. Funcion pura — no re-entrena (decision D8).

Decision D2 (plan US-021 2.1): el ``cv`` que reciben ``learning_curve`` y
``validation_curve`` debe ser una **lista materializada** de tuplas
``(train_idx, test_idx)`` — ``learning_curve`` reusa el ``cv`` una vez por cada
``train_size``; un generador se agota tras el primer uso y los demas tamanos
quedan sin folds. :func:`_materialize_cv_splits` garantiza la materializacion.

Decision D4: la metrica de las curvas es ``accuracy`` (el criterio de aceptacion
lo pide literal); F1-macro es la metrica principal del baseline (US-019) pero el
CA de US-021 especifica accuracy para las curvas.

Polars es el formato de I/O; la conversion a numpy ocurre exclusivamente en el
borde de sklearn, en el helper privado :func:`_to_numpy_xy`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
import structlog
from matplotlib.figure import Figure
from sklearn.base import ClassifierMixin
from sklearn.model_selection import learning_curve, validation_curve

# `ml.train.baseline` se importa de forma diferida dentro de `_to_numpy_xy`
# para romper el ciclo de imports: `baseline` importa de `ml.eval.metrics`,
# y `ml.eval.__init__` re-exporta este modulo — un import a nivel de modulo
# dispararia un circular import al cargar el paquete `ml.eval`.

logger = structlog.get_logger(__name__)

__all__ = [
    "FitDiagnosis",
    "FitVerdict",
    "LearningCurveResult",
    "ValidationCurveResult",
    "diagnose_fit",
    "plot_learning_curve",
    "plot_validation_curve",
]

FitVerdict = Literal["overfit", "underfit", "good_fit"]

# Resolucion de figuras de los entregables visuales del Avance 3 (criterio AC-8).
_PLOT_DPI: int = 200

# Fracciones de muestras por defecto para la curva de aprendizaje (decision D6:
# fracciones, no conteos absolutos — se adaptan a cualquier tamano de dataset).
_DEFAULT_TRAIN_SIZES: tuple[float, ...] = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0)


# ---------------------------------------------------------------------------
# Dataclasses de salida.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LearningCurveResult:
    """Resultado de una curva de aprendizaje.

    Attributes:
        train_sizes_abs: Numero absoluto de muestras de entrenamiento por
            punto de la curva, vector ``(n,)``.
        train_scores_mean: Accuracy media de train por tamano, vector ``(n,)``.
        train_scores_std: Desviacion estandar de la accuracy de train por
            tamano, vector ``(n,)``.
        val_scores_mean: Accuracy media de validacion por tamano, vector
            ``(n,)``.
        val_scores_std: Desviacion estandar de la accuracy de validacion por
            tamano, vector ``(n,)``.
        scoring: Metrica usada en la curva (``"accuracy"`` por defecto).
    """

    train_sizes_abs: np.ndarray
    train_scores_mean: np.ndarray
    train_scores_std: np.ndarray
    val_scores_mean: np.ndarray
    val_scores_std: np.ndarray
    scoring: str


@dataclass(frozen=True)
class ValidationCurveResult:
    """Resultado de una curva de validacion sobre un hiperparametro.

    Attributes:
        param_name: Nombre del hiperparametro variado.
        param_range: Valores evaluados, en el mismo orden de las curvas.
        train_scores_mean: Accuracy media de train por valor, vector ``(n,)``.
        train_scores_std: Desviacion estandar de la accuracy de train, vector
            ``(n,)``.
        val_scores_mean: Accuracy media de validacion por valor, vector
            ``(n,)``.
        val_scores_std: Desviacion estandar de la accuracy de validacion,
            vector ``(n,)``.
    """

    param_name: str
    param_range: list
    train_scores_mean: np.ndarray
    train_scores_std: np.ndarray
    val_scores_mean: np.ndarray
    val_scores_std: np.ndarray


@dataclass(frozen=True)
class FitDiagnosis:
    """Diagnostico de sub/sobreajuste derivado de una curva de aprendizaje.

    Attributes:
        verdict: ``"overfit"``, ``"underfit"`` o ``"good_fit"``.
        gap: ``accuracy_train - accuracy_val`` en el tamano maximo de la
            curva de aprendizaje.
        train_acc_max: Accuracy de train en el tamano maximo.
        val_acc_max: Accuracy de validacion en el tamano maximo.
        explanation: Texto en espanol que justifica el veredicto con los
            numeros concretos.
    """

    verdict: FitVerdict
    gap: float
    train_acc_max: float
    val_acc_max: float
    explanation: str


# ---------------------------------------------------------------------------
# Helpers privados.
# ---------------------------------------------------------------------------


def _materialize_cv_splits(
    cv_splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materializa los splits espaciales en una lista reutilizable.

    Decision D2 (riesgo R2, el bug mas probable): ``learning_curve`` reusa el
    ``cv`` una vez por cada ``train_size``; un generador se agota tras el primer
    uso y los tamanos restantes quedan sin folds. Esta funcion fuerza una
    ``list`` de tuplas ``(train_idx, test_idx)`` de arrays ``np.int64`` —
    reutilizable cuantas veces sklearn la consuma.

    Args:
        cv_splits: Secuencia (lista o generador) de tuplas
            ``(train_idx, test_idx)`` de indices posicionales, tipicamente la
            salida de ``ml.train.baseline._build_cv_splits``.

    Returns:
        Lista materializada de tuplas ``(train_idx, test_idx)`` con los indices
        convertidos a arrays ``np.int64``.

    Raises:
        ValueError: si ``cv_splits`` esta vacio o si algun split no tiene
            muestras en train o en test.
    """
    materialized: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        train_arr = np.asarray(train_idx, dtype=np.int64)
        test_arr = np.asarray(test_idx, dtype=np.int64)
        if train_arr.size == 0 or test_arr.size == 0:
            raise ValueError(
                f"El split espacial {fold_idx} no tiene muestras en train "
                f"({train_arr.size}) o en test ({test_arr.size})."
            )
        materialized.append((train_arr, test_arr))
    if not materialized:
        raise ValueError(
            "`cv_splits` esta vacio; las curvas requieren al menos un split."
        )
    return materialized


def _to_numpy_xy(
    df: pl.DataFrame,
    *,
    max_samples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convierte el DataFrame de features a la matriz numpy del borde sklearn.

    Reutiliza los helpers del baseline (``_feature_columns``, ``_feature_matrix``,
    ``_impute``, ``_encode_labels``) para que la matriz X, las etiquetas y la
    imputacion sean identicas a las que usa US-019. La conversion a numpy ocurre
    solo aqui — el resto del modulo opera sobre Polars.

    Cuando ``max_samples > 0`` y el dataset es mas grande, devuelve un subsample
    estratificado por clase (decision D7) para acelerar dev/CI; tambien devuelve
    los indices posicionales conservados para que el caller realinee el ``cv``.

    Args:
        df: DataFrame Polars de features ya preparado.
        max_samples: Cota superior de muestras; ``0`` desactiva el subsample.
        random_state: Semilla determinista del subsample estratificado.

    Returns:
        Tupla ``(matrix, y_encoded, kept_idx)`` donde ``matrix`` es la matriz
        de features imputada ``(n, n_features)``, ``y_encoded`` las etiquetas
        contiguas ``(n,)`` y ``kept_idx`` los indices posicionales del ``df``
        original conservados (todos si no hubo subsample).
    """
    # Import diferido: rompe el ciclo `baseline` <-> `ml.eval` (ver cabecera).
    from ml.train.baseline import (
        _encode_labels,
        _feature_columns,
        _feature_matrix,
        _impute,
    )

    feature_cols = _feature_columns(df)
    _encoder, y_all = _encode_labels(df)
    matrix_all = _impute(_feature_matrix(df, feature_cols))
    n_rows = df.height
    kept_idx = np.arange(n_rows, dtype=np.int64)

    if max_samples <= 0 or max_samples >= n_rows:
        return matrix_all, y_all, kept_idx

    # Subsample estratificado por clase: conserva la proporcion de cada clase.
    rng = np.random.default_rng(random_state)
    fraction = max_samples / n_rows
    selected: list[np.ndarray] = []
    for cls in np.unique(y_all):
        cls_idx = np.where(y_all == cls)[0]
        # Al menos una muestra por clase para no perder ninguna etiqueta.
        n_take = max(1, round(cls_idx.size * fraction))
        n_take = min(n_take, cls_idx.size)
        selected.append(rng.choice(cls_idx, size=n_take, replace=False))
    kept_idx = np.sort(np.concatenate(selected)).astype(np.int64)
    logger.info(
        "learning_curve_subsampled",
        n_original=n_rows,
        n_kept=int(kept_idx.size),
        max_samples=max_samples,
    )
    return matrix_all[kept_idx], y_all[kept_idx], kept_idx


def _remap_cv_splits(
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    kept_idx: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Realinea los splits espaciales tras un subsample de muestras.

    Cuando :func:`_to_numpy_xy` submuestrea el dataset, los indices posicionales
    de ``cv_splits`` ya no apuntan a las filas correctas de la matriz reducida.
    Esta funcion traduce cada indice original a su nueva posicion en el subset y
    descarta los indices que el subsample dejo fuera.

    Args:
        cv_splits: Lista materializada de splits sobre el dataset completo.
        kept_idx: Indices posicionales conservados por el subsample, ordenados.

    Returns:
        Lista de splits ``(train_idx, test_idx)`` con indices posicionales del
        dataset reducido; se descartan los folds que se quedan sin train o test.
    """
    # `position[i]` = nueva posicion del indice original `i`, o -1 si se descarto.
    max_original = int(kept_idx.max()) + 1 if kept_idx.size else 0
    position = np.full(max_original, -1, dtype=np.int64)
    position[kept_idx] = np.arange(kept_idx.size, dtype=np.int64)

    remapped: list[tuple[np.ndarray, np.ndarray]] = []
    for train_idx, test_idx in cv_splits:
        train_in = train_idx[train_idx < max_original]
        test_in = test_idx[test_idx < max_original]
        new_train = position[train_in]
        new_test = position[test_in]
        new_train = new_train[new_train >= 0]
        new_test = new_test[new_test >= 0]
        if new_train.size == 0 or new_test.size == 0:
            continue
        remapped.append((new_train, new_test))
    if not remapped:
        raise ValueError(
            "El subsample dejo sin muestras a todos los folds espaciales; "
            "aumenta `max_samples`."
        )
    return remapped


def _curve_figure(
    x_values: Sequence,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    val_mean: np.ndarray,
    val_std: np.ndarray,
    *,
    x_label: str,
    title: str,
) -> Figure:
    """Construye una figura de curva con banda +/-std sombreada.

    Patron de US-019 (``ml/eval/metrics.py``): backend ``Agg`` no interactivo
    para que la figura sea serializable a PNG en CI y en notebooks ejecutados
    con papermill. La banda sombreada (``fill_between``) cubre +/-1 desviacion
    estandar alrededor de la media de cada curva (criterio AC-8).

    Args:
        x_values: Valores del eje X (tamanos de muestra o valores del
            hiperparametro). Se rotulan como categorias para soportar valores
            no numericos como ``None`` (riesgo R4).
        train_mean: Accuracy media de train por punto, vector ``(n,)``.
        train_std: Desviacion estandar de train por punto, vector ``(n,)``.
        val_mean: Accuracy media de validacion por punto, vector ``(n,)``.
        val_std: Desviacion estandar de validacion por punto, vector ``(n,)``.
        x_label: Etiqueta del eje X.
        title: Titulo de la figura.

    Returns:
        Figura matplotlib lista para ``fig.savefig(...)`` o ``display``.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    # Posiciones equiespaciadas: soporta valores no numericos (e.g. `None` en
    # `max_depth`) sin romper el eje (riesgo R4).
    positions = np.arange(len(x_values))
    tick_labels = [str(v) for v in x_values]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=_PLOT_DPI)
    ax.plot(positions, train_mean, marker="o", color="#2c7fb8", label="Train")
    ax.fill_between(
        positions,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color="#2c7fb8",
    )
    ax.plot(positions, val_mean, marker="s", color="#d95f0e", label="Validacion")
    ax.fill_between(
        positions,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.15,
        color="#d95f0e",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# API publica.
# ---------------------------------------------------------------------------


def plot_learning_curve(
    estimator: ClassifierMixin,
    df: pl.DataFrame,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    train_sizes: list[float] | None = None,
    scoring: str = "accuracy",
    max_samples: int = 0,
    random_state: int = 42,
) -> tuple[LearningCurveResult, Figure]:
    """Traza la curva de aprendizaje de un estimador con validacion cruzada espacial.

    Envuelve :func:`sklearn.model_selection.learning_curve` para medir como
    evoluciona la accuracy de train y de validacion al crecer el numero de
    muestras de entrenamiento. ``learning_curve`` re-entrena el estimador
    ``len(train_sizes) * len(cv_splits)`` veces — el ``estimator`` debe estar
    sin ajustar (decision D3).

    ``cv_splits`` DEBE ser una lista materializada de tuplas
    ``(train_idx, test_idx)`` (no un generador): la funcion reusa el ``cv`` por
    cada ``train_size`` y un generador se agotaria (decision D2). Se materializa
    de nuevo internamente por seguridad.

    Args:
        estimator: Estimador sklearn/xgboost sin ajustar (factory de US-019 via
            ``ml.train.baseline.build_estimator``).
        df: DataFrame Polars de features ya preparado (con ``parcel_id``,
            ``class_id`` y columnas de feature numericas).
        cv_splits: Lista materializada de splits espaciales
            ``(train_idx, test_idx)`` de indices posicionales.
        train_sizes: Fracciones del train por punto de la curva; si es ``None``
            se usan ``(0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0)`` (decision D6).
        scoring: Metrica de la curva; ``"accuracy"`` por defecto (decision D4).
        max_samples: Cota superior de muestras para subsample estratificado en
            dev/CI; ``0`` (default) usa el dataset completo (decision D7).
        random_state: Semilla determinista del subsample.

    Returns:
        Tupla ``(LearningCurveResult, Figure)`` con los scores agregados por
        tamano y la figura con la banda +/-std.

    Raises:
        ValueError: si ``cv_splits`` esta vacio o si ``df`` carece de columnas
            obligatorias.
    """
    sizes = list(train_sizes) if train_sizes is not None else list(_DEFAULT_TRAIN_SIZES)
    splits = _materialize_cv_splits(cv_splits)
    matrix, y_encoded, kept_idx = _to_numpy_xy(
        df, max_samples=max_samples, random_state=random_state
    )
    if kept_idx.size != df.height:
        splits = _remap_cv_splits(splits, kept_idx)

    logger.info(
        "learning_curve_start",
        n_samples=int(matrix.shape[0]),
        n_features=int(matrix.shape[1]),
        n_train_sizes=len(sizes),
        n_folds=len(splits),
        scoring=scoring,
    )
    train_sizes_abs, train_scores, val_scores = learning_curve(
        estimator,
        matrix,
        y_encoded,
        train_sizes=np.asarray(sizes, dtype=np.float64),
        cv=splits,
        scoring=scoring,
        n_jobs=None,
        shuffle=False,
        random_state=random_state,
    )
    result = LearningCurveResult(
        train_sizes_abs=np.asarray(train_sizes_abs, dtype=np.int64),
        train_scores_mean=train_scores.mean(axis=1),
        train_scores_std=train_scores.std(axis=1),
        val_scores_mean=val_scores.mean(axis=1),
        val_scores_std=val_scores.std(axis=1),
        scoring=scoring,
    )
    figure = _curve_figure(
        result.train_sizes_abs.tolist(),
        result.train_scores_mean,
        result.train_scores_std,
        result.val_scores_mean,
        result.val_scores_std,
        x_label="Muestras de entrenamiento",
        title=f"Curva de aprendizaje ({scoring})",
    )
    logger.info(
        "learning_curve_done",
        train_acc_max=round(float(result.train_scores_mean[-1]), 4),
        val_acc_max=round(float(result.val_scores_mean[-1]), 4),
    )
    return result, figure


def plot_validation_curve(
    estimator: ClassifierMixin,
    df: pl.DataFrame,
    param_name: str,
    param_range: list,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    scoring: str = "accuracy",
    max_samples: int = 0,
    random_state: int = 42,
) -> tuple[ValidationCurveResult, Figure]:
    """Traza la curva de validacion de un hiperparametro con validacion cruzada espacial.

    Envuelve :func:`sklearn.model_selection.validation_curve` para medir como
    cambia la accuracy de train y de validacion al variar un hiperparametro
    critico (``max_depth``, ``n_estimators``, ``learning_rate``). ``validation_curve``
    re-instancia el estimador por cada valor del rango — el ``estimator`` debe
    estar sin ajustar.

    El eje X soporta valores no numericos como ``None`` (riesgo R4): el ``param_range``
    se preserva tal cual en el resultado y la figura lo rotula como categoria
    (``"None"``).

    Args:
        estimator: Estimador sklearn/xgboost sin ajustar.
        df: DataFrame Polars de features ya preparado.
        param_name: Nombre del hiperparametro a variar (e.g. ``"max_depth"``).
        param_range: Valores del hiperparametro a evaluar; puede contener
            ``None`` (e.g. ``max_depth`` sin tope).
        cv_splits: Lista materializada de splits espaciales.
        scoring: Metrica de la curva; ``"accuracy"`` por defecto (decision D4).
        max_samples: Cota superior de muestras para subsample en dev/CI; ``0``
            usa el dataset completo (decision D7).
        random_state: Semilla determinista del subsample.

    Returns:
        Tupla ``(ValidationCurveResult, Figure)`` con los scores agregados por
        valor del hiperparametro y la figura con la banda +/-std.

    Raises:
        ValueError: si ``cv_splits`` esta vacio, si ``param_range`` esta vacio
            o si ``df`` carece de columnas obligatorias.
    """
    if not param_range:
        raise ValueError("`param_range` no puede estar vacio.")
    splits = _materialize_cv_splits(cv_splits)
    matrix, y_encoded, kept_idx = _to_numpy_xy(
        df, max_samples=max_samples, random_state=random_state
    )
    if kept_idx.size != df.height:
        splits = _remap_cv_splits(splits, kept_idx)

    logger.info(
        "validation_curve_start",
        param_name=param_name,
        n_values=len(param_range),
        n_samples=int(matrix.shape[0]),
        n_folds=len(splits),
        scoring=scoring,
    )
    train_scores, val_scores = validation_curve(
        estimator,
        matrix,
        y_encoded,
        param_name=param_name,
        param_range=param_range,
        cv=splits,
        scoring=scoring,
        n_jobs=None,
    )
    result = ValidationCurveResult(
        param_name=param_name,
        param_range=list(param_range),
        train_scores_mean=train_scores.mean(axis=1),
        train_scores_std=train_scores.std(axis=1),
        val_scores_mean=val_scores.mean(axis=1),
        val_scores_std=val_scores.std(axis=1),
    )
    figure = _curve_figure(
        result.param_range,
        result.train_scores_mean,
        result.train_scores_std,
        result.val_scores_mean,
        result.val_scores_std,
        x_label=param_name,
        title=f"Curva de validacion — {param_name} ({scoring})",
    )
    logger.info(
        "validation_curve_done",
        param_name=param_name,
        best_val_acc=round(float(result.val_scores_mean.max()), 4),
    )
    return result, figure


def diagnose_fit(
    result: LearningCurveResult,
    *,
    gap_threshold: float = 0.10,
    low_acc_threshold: float = 0.65,
) -> FitDiagnosis:
    """Diagnostica sub/sobreajuste a partir de una curva de aprendizaje.

    Funcion pura (decision D8): deriva el veredicto de los scores ya calculados
    en ``LearningCurveResult`` evaluados en el tamano maximo de la curva — no
    re-entrena ningun modelo.

    Reglas (criterio AC-4):

    - ``gap > gap_threshold``                                  -> ``"overfit"``
    - ``train_acc < low_acc AND val_acc < low_acc``            -> ``"underfit"``
    - resto                                                    -> ``"good_fit"``

    El sobreajuste se evalua primero: un modelo con un gap grande es overfit
    incluso si ambas accuracies son modestas. El subajuste solo aplica cuando
    el modelo no logra ajustar ni siquiera el train.

    Args:
        result: Resultado de :func:`plot_learning_curve`.
        gap_threshold: Umbral del gap train-val por encima del cual hay
            sobreajuste; ``0.10`` por defecto (criterio AC-4).
        low_acc_threshold: Umbral por debajo del cual ambas accuracies se
            consideran bajas (subajuste); ``0.65`` por defecto.

    Returns:
        Un :class:`FitDiagnosis` con el veredicto, el gap, las accuracies en el
        tamano maximo y una explicacion textual.

    Raises:
        ValueError: si la curva no tiene puntos.
    """
    if result.train_scores_mean.size == 0:
        raise ValueError("La curva de aprendizaje no tiene puntos para diagnosticar.")

    train_acc = float(result.train_scores_mean[-1])
    val_acc = float(result.val_scores_mean[-1])
    gap = train_acc - val_acc

    if gap > gap_threshold:
        verdict: FitVerdict = "overfit"
        explanation = (
            f"Sobreajuste: el gap train-val es {gap:.3f} (> {gap_threshold:.2f}). "
            f"El modelo memoriza el train (accuracy {train_acc:.3f}) pero "
            f"generaliza peor en validacion (accuracy {val_acc:.3f})."
        )
    elif train_acc < low_acc_threshold and val_acc < low_acc_threshold:
        verdict = "underfit"
        explanation = (
            f"Subajuste: train ({train_acc:.3f}) y validacion ({val_acc:.3f}) "
            f"estan ambos por debajo de {low_acc_threshold:.2f}. El modelo no "
            f"captura la senal ni en el conjunto de entrenamiento; falta "
            f"capacidad o las features no son suficientemente informativas."
        )
    else:
        verdict = "good_fit"
        explanation = (
            f"Buen ajuste: el gap train-val es {gap:.3f} (<= {gap_threshold:.2f}) "
            f"y la accuracy de validacion ({val_acc:.3f}) no es baja. El modelo "
            f"generaliza de forma consistente con su desempeno en train "
            f"({train_acc:.3f})."
        )

    logger.info(
        "fit_diagnosed",
        verdict=verdict,
        gap=round(gap, 4),
        train_acc_max=round(train_acc, 4),
        val_acc_max=round(val_acc, 4),
    )
    return FitDiagnosis(
        verdict=verdict,
        gap=gap,
        train_acc_max=train_acc,
        val_acc_max=val_acc,
        explanation=explanation,
    )
