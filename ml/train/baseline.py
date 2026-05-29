"""Baseline tabular de clasificacion de cultivos: Random Forest + XGBoost (US-019).

Libreria del EPIC 4 (Avance 3). Entrena dos modelos tabulares sobre el
vector de features combinado del EPIC 3 (AlphaEarth + indices espectrales
+ estadisticas temporales + SRTM + ERA5) con evaluacion por validacion
cruzada **espacial** y tuning ligero opcional.

Decisiones canonicas (plan ``docs/us-planning/us-019.md`` 2.1):

- **D1**: el CV es espacial via :func:`ml.features.spatial_split.build_spatial_kfold`
  (H3 + KMeans + buffer 1 km). CERO ``KFold``/``train_test_split`` aleatorio.
- **D2**: solo ``RandomForestClassifier`` + ``xgboost.XGBClassifier``.
- **D3**: ``tree_method="hist"``; XGBoost usa ``device="cuda"`` si hay una
  GPU NVIDIA disponible y degrada automaticamente a CPU si no (CI sin GPU,
  laptop sin CUDA). RandomForest es siempre CPU (sklearn no tiene backend
  GPU). El problema (85 k x 187) corre en minutos en cualquiera de los dos.
- **D5**: balanceo de clases (``class_weight="balanced"`` para RF,
  ``sample_weight`` inverso a frecuencia para XGB).
- **D12**: ``LabelEncoder`` sobre ``class_id`` persistido en el resultado;
  XGB ``multi:softprob`` exige etiquetas contiguas ``[0, n_classes)``.

El dataset de features no incluye geometria de parcela; el centroide
espacial se deriva del ``patch_id`` PASTIS-R via
``data/PASTIS-R/metadata.geojson`` (geometria por patch en EPSG:2154).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import structlog
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from ml.eval.metrics import compute_baseline_metrics
from ml.features.scaler import fit_scaler_on_train
from ml.features.spatial_split import FoldAssignment, build_spatial_kfold

logger = structlog.get_logger(__name__)

__all__ = [
    "BaselineResult",
    "ModelKind",
    "build_estimator",
    "evaluate_with_spatial_cv",
    "train_one_model",
    "tune_baseline",
]

ModelKind = Literal["rf", "xgb", "lgbm"]

# Columnas de metadata que NO son features (se excluyen de la matriz X).
_META_COLS: tuple[str, ...] = (
    "parcel_id",
    "year",
    "patch_id",
    "instance_id",
    "class_id",
    "class_name",
    "fold",
    "n_pixels",
    "area_m2",
    "geometry",
)

# Sufijos de columnas que indican un join sin coalesce previo y nunca son
# features. Defensa en profundidad sobre `_META_COLS`: el bug US-023-preview-v2
# (patch_id_right importance=0.27 en XGB) entraba aqui via Polars left join.
_META_SUFFIXES: tuple[str, ...] = ("_right", "_left", "_x", "_y")

# Clases PASTIS-R no agronomicas a descartar (Background, Void label).
_DROP_CLASS_IDS: tuple[int, ...] = (0, 19)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FEATURES_PATH = (
    _REPO_ROOT / "data" / "test_fixtures" / "feature_selection_parcels_subset.parquet"
)
_PASTIS_METADATA_PATH = _REPO_ROOT / "data" / "PASTIS-R" / "metadata.geojson"

# Hiperparametros base documentados (criterio AC-1).
# `max_depth` y `min_samples_leaf` acotados (no None / no 1): un RF sin poda
# sobre 85k parcelas crece hasta hojas puras -> modelo de ~700 MB inmanejable
# para el Model Registry y con sobreajuste severo. La poda (depth 20,
# min_samples_leaf 10, 150 arboles) deja el modelo en ~100-150 MB, logueable,
# sin perder F1 material (desviacion justificada del plan US-019; ver handoff).
_RF_BASE_PARAMS: dict[str, object] = {
    "n_estimators": 150,
    "max_depth": 20,
    "min_samples_leaf": 10,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 42,
}
_XGB_BASE_PARAMS: dict[str, object] = {
    "n_estimators": 400,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "objective": "multi:softprob",
    "random_state": 42,
}
# LightGBM (3er modelo del baseline tabular). Hiperparametros alineados con XGB
# para una comparacion justa: misma profundidad efectiva (`num_leaves=63 ~ 2^6`
# con `max_depth=-1`), mismo `learning_rate=0.05`, mismo subsample/colsample.
# `class_weight="balanced"` reemplaza el `sample_weight` manual que XGB requiere
# (LGBM si expone el parametro nativamente, decision D5). LGBM acepta NaN sin
# imputacion previa pero seguimos el mismo `_impute_with` por consistencia.
# Nota: la rueda PyPI de `lightgbm` no incluye build con CUDA; se queda en CPU.
_LGBM_BASE_PARAMS: dict[str, object] = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "class_weight": "balanced",
    "objective": "multiclass",
    "n_jobs": -1,
    "random_state": 42,
    "verbose": -1,
}


def resolve_xgb_device() -> str:
    """Resuelve el device de XGBoost segun la disponibilidad de GPU NVIDIA.

    Detecta una GPU CUDA via ``nvidia-smi``. Si esta presente devuelve
    ``"cuda"`` (XGBoost 3.x usa ``tree_method="hist"`` + ``device="cuda"``
    para el entrenamiento acelerado); si no, degrada a ``"cpu"`` para que
    el baseline corra en CI y en laptops sin CUDA (decision D3).

    Returns:
        ``"cuda"`` si hay GPU NVIDIA detectable, ``"cpu"`` en caso contrario.
    """
    import shutil
    import subprocess

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return "cpu"
    try:
        result = subprocess.run(  # noqa: S603 — ruta resuelta con shutil.which
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "cpu"
    if result.returncode == 0 and result.stdout.strip():
        logger.info("xgb_device_resolved", device="cuda", gpu=result.stdout.strip())
        return "cuda"
    return "cpu"

# Grids de tuning ligero (criterio AC-4): 8 combinaciones por modelo.
# `max_depth` y `min_samples_leaf` acotados: evita el RF de ~700 MB y el
# sobreajuste de arboles sin poda (ver _RF_BASE_PARAMS).
_RF_PARAM_GRID: dict[str, list] = {
    "n_estimators": [100, 150],
    "max_depth": [15, 20],
    "min_samples_leaf": [10, 20],
}
_XGB_PARAM_GRID: dict[str, list] = {
    "n_estimators": [300, 400],
    "max_depth": [6, 8],
    "learning_rate": [0.05, 0.1],
}
# LightGBM: 8 combinaciones (2 x 2 x 2). `num_leaves` acotado a [31, 63] para no
# crecer arboles que dupliquen el modelo en memoria (mismo criterio que RF).
_LGBM_PARAM_GRID: dict[str, list] = {
    "n_estimators": [300, 400],
    "num_leaves": [31, 63],
    "learning_rate": [0.05, 0.1],
}

_METRIC_KEYS: tuple[str, ...] = (
    "f1_macro",
    "f1_weighted",
    "miou",
    "accuracy",
    "cohen_kappa",
)


# ---------------------------------------------------------------------------
# Dataclass de salida.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineResult:
    """Resultado de entrenar un baseline tabular.

    Attributes:
        model: Estimador sklearn/xgboost ya ajustado sobre el dataset
            completo (etiquetas codificadas con ``LabelEncoder``).
        model_kind: ``"rf"`` o ``"xgb"``.
        metrics: Las cinco metricas de :func:`compute_baseline_metrics`
            calculadas sobre las predicciones out-of-fold del CV espacial.
        cv_metrics: Mapa ``{metrica: (media, desviacion)}`` sobre los
            folds del CV espacial.
        feature_cols: Nombres de las columnas usadas como features, en el
            mismo orden de las columnas de la matriz X.
        best_params: Hiperparametros usados (de ``GridSearchCV`` si hubo
            tuning, o los base si no).
        label_classes: Clases originales en el orden del ``LabelEncoder``;
            ``label_classes[i]`` es la clase real de la etiqueta ``i``.
        label_encoder: El ``LabelEncoder`` ajustado, para decodificar
            predicciones en downstream (US-020, inferencia).
    """

    model: ClassifierMixin
    model_kind: ModelKind
    metrics: dict[str, float]
    cv_metrics: dict[str, tuple[float, float]]
    feature_cols: tuple[str, ...]
    best_params: dict[str, object]
    label_classes: tuple[int, ...]
    label_encoder: LabelEncoder


# ---------------------------------------------------------------------------
# Construccion de estimadores.
# ---------------------------------------------------------------------------


def build_estimator(model: ModelKind, hyperparams: dict[str, object]) -> ClassifierMixin:
    """Instancia un estimador RF, XGB o LGBM con los hiperparametros dados.

    Args:
        model: ``"rf"`` para :class:`RandomForestClassifier`, ``"xgb"``
            para :class:`xgboost.XGBClassifier` o ``"lgbm"`` para
            :class:`lightgbm.LGBMClassifier`.
        hyperparams: Diccionario de hiperparametros del constructor.

    Returns:
        El estimador instanciado (sin ajustar).

    Raises:
        ValueError: si ``model`` no es ``"rf"``, ``"xgb"`` ni ``"lgbm"``.
    """
    if model == "rf":
        return RandomForestClassifier(**hyperparams)
    if model == "xgb":
        # Inyecta el device (cuda/cpu) si el caller no lo fijo; permite
        # acelerar en GPU local sin romper CI sin CUDA (decision D3).
        xgb_params = dict(hyperparams)
        xgb_params.setdefault("device", resolve_xgb_device())
        return XGBClassifier(**xgb_params)
    if model == "lgbm":
        # LGBM se queda en CPU: la rueda PyPI no trae build CUDA. Para GPU
        # haria falta `pip install lightgbm --config-settings=cmake.define...`
        # con `device_type="gpu"`, fuera del alcance del baseline.
        return LGBMClassifier(**hyperparams)  # type: ignore[arg-type]
    raise ValueError(f"`model` debe ser 'rf', 'xgb' o 'lgbm'; recibido {model!r}.")


# ---------------------------------------------------------------------------
# API publica.
# ---------------------------------------------------------------------------


def train_one_model(
    df: pl.DataFrame,
    *,
    model: ModelKind,
    hyperparams: dict[str, object] | None = None,
    k_folds: int = 5,
    buffer_km: float = 1.0,
    random_state: int = 42,
) -> BaselineResult:
    """Entrena un baseline (RF o XGB) con evaluacion por CV espacial.

    Carga las features, construye los folds espaciales con
    :func:`build_spatial_kfold`, evalua con scaler anti-leakage por fold,
    obtiene predicciones out-of-fold y ajusta el modelo final sobre todo
    el dataset.

    Args:
        df: DataFrame Polars de features (debe contener ``parcel_id``,
            ``class_id`` y al menos una columna de feature numerica).
        model: ``"rf"`` o ``"xgb"``.
        hyperparams: Hiperparametros del estimador; si es ``None`` se usan
            los valores base documentados (``_RF_BASE_PARAMS`` /
            ``_XGB_BASE_PARAMS``).
        k_folds: Numero de folds del CV espacial (default 5).
        buffer_km: Buffer anti-leakage en km entre folds (default 1.0).
        random_state: Semilla determinista.

    Returns:
        Un :class:`BaselineResult` con el modelo ajustado, las metricas
        out-of-fold, las metricas por fold y la metadata de features.

    Raises:
        ValueError: si ``df`` carece de columnas obligatorias o si tras
            descartar las clases no agronomicas no quedan muestras.
    """
    clean_df = _prepare_dataframe(df)
    feature_cols = _feature_columns(clean_df)
    encoder, y_encoded = _encode_labels(clean_df)

    params = dict(hyperparams) if hyperparams is not None else _base_params(model)
    if model == "xgb":
        params.setdefault("num_class", len(encoder.classes_))

    def factory() -> ClassifierMixin:
        return build_estimator(model, params)

    cv_metrics, y_true_oof, y_pred_oof = evaluate_with_spatial_cv(
        clean_df,
        factory,
        k_folds=k_folds,
        buffer_km=buffer_km,
        random_state=random_state,
    )
    oof_metrics = compute_baseline_metrics(
        y_true_oof,
        y_pred_oof,
        labels=list(range(len(encoder.classes_))),
    )

    # Ajuste final sobre el dataset completo (modelo de produccion). Los
    # arboles (RF/XGB) son invariantes a escala monotona, asi que el modelo
    # final opera sobre features imputadas crudas (sin StandardScaler); el
    # CV si escala porque `fit_scaler_on_train` es el patron del repo.
    matrix = _impute(_feature_matrix(clean_df, feature_cols))
    final_model = build_estimator(model, params)
    # XGB no expone `class_weight`: inyectamos `sample_weight` inverso a
    # frecuencia (decision D5). LGBM con `class_weight="balanced"` ya lo
    # maneja nativamente; si el caller lo quita, caemos a sample_weight.
    sample_weight: np.ndarray | None
    if model == "xgb":
        sample_weight = _sample_weights(y_encoded)
    elif model == "lgbm" and "class_weight" not in params:
        sample_weight = _sample_weights(y_encoded)
    else:
        sample_weight = None
    if sample_weight is not None:
        final_model.fit(matrix, y_encoded, sample_weight=sample_weight)
    else:
        final_model.fit(matrix, y_encoded)

    logger.info(
        "baseline_trained",
        model=model,
        n_samples=clean_df.height,
        n_features=len(feature_cols),
        n_classes=len(encoder.classes_),
        f1_macro_oof=oof_metrics["f1_macro"],
    )
    return BaselineResult(
        model=final_model,
        model_kind=model,
        metrics=oof_metrics,
        cv_metrics=cv_metrics,
        feature_cols=feature_cols,
        best_params=params,
        label_classes=tuple(int(c) for c in encoder.classes_),
        label_encoder=encoder,
    )


def tune_baseline(
    df: pl.DataFrame,
    *,
    model: ModelKind,
    param_grid: dict[str, list] | None = None,
    k_folds: int = 5,
    buffer_km: float = 1.0,
    scoring: str = "f1_macro",
    random_state: int = 42,
) -> dict[str, object]:
    """Tuning ligero de hiperparametros via ``GridSearchCV`` sobre CV espacial.

    El parametro ``cv`` de :class:`GridSearchCV` recibe la **lista de
    splits espaciales** ``(train_idx, test_idx)`` (no un entero), de modo
    que el tuning respeta la particion geografica y no introduce leakage.

    Args:
        df: DataFrame Polars de features.
        model: ``"rf"`` o ``"xgb"``.
        param_grid: Grilla de hiperparametros; si es ``None`` se usan las
            grillas ligeras documentadas (8 combinaciones por modelo).
        k_folds: Numero de folds del CV espacial (default 5).
        buffer_km: Buffer anti-leakage en km (default 1.0).
        scoring: Metrica de seleccion de ``GridSearchCV`` (default
            ``"f1_macro"``).
        random_state: Semilla determinista.

    Returns:
        El diccionario ``best_params_`` del ``GridSearchCV``.

    Raises:
        ValueError: si ``df`` carece de columnas obligatorias.
    """
    clean_df = _prepare_dataframe(df)
    feature_cols = _feature_columns(clean_df)
    encoder, y_encoded = _encode_labels(clean_df)
    matrix = _impute(_feature_matrix(clean_df, feature_cols))

    grid = param_grid if param_grid is not None else _default_grid(model)
    cv_splits = _build_cv_splits(
        clean_df, k_folds=k_folds, buffer_km=buffer_km, random_state=random_state
    )

    base_params = _base_params(model)
    if model == "xgb":
        base_params.setdefault("num_class", len(encoder.classes_))
    # Quitamos del estimador base las claves que la grilla va a sobrescribir.
    for key in grid:
        base_params.pop(key, None)
    estimator = build_estimator(model, base_params)

    n_combos = 1
    for values in grid.values():
        n_combos *= len(values)
    # Con XGB en GPU el GridSearchCV usa n_jobs=1: una sola GPU no puede
    # atender varios fits en paralelo y N workers compitiendo por ella
    # causan thrashing (cada uno crea su contexto CUDA). El boosting de XGB
    # ya paraleliza internamente en la GPU. RF (CPU) si usa todos los nucleos.
    xgb_on_gpu = model == "xgb" and resolve_xgb_device() == "cuda"
    search_n_jobs = 1 if xgb_on_gpu else -1
    logger.info(
        "baseline_tuning_start",
        model=model,
        n_combos=n_combos,
        n_folds=len(cv_splits),
        n_fits=n_combos * len(cv_splits),
        search_n_jobs=search_n_jobs,
    )
    search = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        scoring=scoring,
        cv=cv_splits,
        n_jobs=search_n_jobs,
        refit=True,
        verbose=2,
    )
    search.fit(matrix, y_encoded)
    logger.info(
        "baseline_tuned",
        model=model,
        n_combos=len(search.cv_results_["params"]),
        best_score=float(search.best_score_),
        best_params=search.best_params_,
    )
    return dict(search.best_params_)


def evaluate_with_spatial_cv(
    df: pl.DataFrame,
    model_factory: Callable[[], ClassifierMixin],
    *,
    k_folds: int = 5,
    buffer_km: float = 1.0,
    random_state: int = 42,
) -> tuple[dict[str, tuple[float, float]], np.ndarray, np.ndarray]:
    """Evalua un estimador con validacion cruzada espacial anti-leakage.

    Por cada fold espacial ajusta un :class:`StandardScaler` solo sobre el
    train (anti-leakage, via :func:`fit_scaler_on_train`), entrena un
    estimador fresco y predice sobre el test del fold. Agrega la media y
    desviacion de las cinco metricas y devuelve tambien las predicciones
    out-of-fold concatenadas.

    Args:
        df: DataFrame Polars de features ya preparado (ver
            :func:`_prepare_dataframe`).
        model_factory: Callable sin argumentos que devuelve un estimador
            nuevo sin ajustar (se invoca una vez por fold).
        k_folds: Numero de folds del CV espacial (default 5).
        buffer_km: Buffer anti-leakage en km (default 1.0).
        random_state: Semilla determinista.

    Returns:
        Tupla ``(cv_metrics, y_true_oof, y_pred_oof)`` donde ``cv_metrics``
        es ``{metrica: (media, std)}``, ``y_true_oof`` son las etiquetas
        verdaderas codificadas concatenadas por fold y ``y_pred_oof`` las
        predicciones correspondientes.
    """
    feature_cols = _feature_columns(df)
    encoder, y_encoded = _encode_labels(df)
    matrix = _feature_matrix(df, feature_cols)
    n_classes = len(encoder.classes_)

    cv_splits = _build_cv_splits(
        df, k_folds=k_folds, buffer_km=buffer_km, random_state=random_state
    )

    per_fold: list[dict[str, float]] = []
    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []

    logger.info("spatial_cv_start", n_folds=len(cv_splits), n_classes=n_classes)
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        if train_idx.size == 0 or test_idx.size == 0:
            logger.warning("spatial_cv_fold_skipped", fold=fold_idx)
            continue
        logger.info(
            "spatial_cv_fold_start",
            fold=f"{fold_idx + 1}/{len(cv_splits)}",
            n_train=int(train_idx.size),
            n_test=int(test_idx.size),
        )

        scaler, scaler_cols = _fit_fold_scaler(
            df, feature_cols=feature_cols, train_idx=train_idx, fold_idx=fold_idx
        )
        # `fit_scaler_on_train` puede descartar columnas all-NaN: alineamos la
        # matriz a las columnas que el scaler conoce antes de `transform`.
        col_idx = np.array(
            [feature_cols.index(c) for c in scaler_cols], dtype=np.int64
        )
        raw_train = matrix[np.ix_(train_idx, col_idx)]
        raw_test = matrix[np.ix_(test_idx, col_idx)]
        # Imputacion anti-leakage: las medianas se calculan solo sobre train.
        train_medians = _column_medians(raw_train)
        x_train = scaler.transform(_impute_with(raw_train, train_medians))
        x_test = scaler.transform(_impute_with(raw_test, train_medians))
        y_train = y_encoded[train_idx]
        y_test = y_encoded[test_idx]

        estimator = model_factory()
        if _is_xgb(estimator):
            estimator.fit(x_train, y_train, sample_weight=_sample_weights(y_train))
        elif _is_lgbm(estimator) and getattr(estimator, "class_weight", None) is None:
            # LGBM sin `class_weight="balanced"` recibe el sample_weight inverso
            # a frecuencia para alineacion con XGB (decision D5).
            estimator.fit(x_train, y_train, sample_weight=_sample_weights(y_train))
        else:
            estimator.fit(x_train, y_train)
        y_pred = estimator.predict(x_test)

        fold_metrics = compute_baseline_metrics(
            y_test, y_pred, labels=list(range(n_classes))
        )
        per_fold.append(fold_metrics)
        y_true_chunks.append(y_test)
        y_pred_chunks.append(np.asarray(y_pred))
        logger.info(
            "spatial_cv_fold_done",
            fold=f"{fold_idx + 1}/{len(cv_splits)}",
            f1_macro=round(fold_metrics["f1_macro"], 4),
        )

    cv_metrics = _aggregate_fold_metrics(per_fold)
    y_true_oof = (
        np.concatenate(y_true_chunks) if y_true_chunks else np.array([], dtype=np.int64)
    )
    y_pred_oof = (
        np.concatenate(y_pred_chunks) if y_pred_chunks else np.array([], dtype=np.int64)
    )
    return cv_metrics, y_true_oof, y_pred_oof


# ---------------------------------------------------------------------------
# Helpers privados — carga y limpieza.
# ---------------------------------------------------------------------------


def _load_baseline_dataset(features_path: Path | str | None = None) -> pl.DataFrame:
    """Carga el parquet de features del baseline desde disco.

    Args:
        features_path: Ruta al parquet; si es ``None`` usa el subset
            canonico de US-018 (``feature_selection_parcels_subset.parquet``).

    Returns:
        El DataFrame Polars crudo (sin limpiar).

    Raises:
        FileNotFoundError: si el parquet no existe.
    """
    path = Path(features_path) if features_path is not None else _DEFAULT_FEATURES_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset de features no encontrado en {path}. "
            "Genera el subset con `make feature-selection-subset` o ejecuta "
            "el pipeline de extraccion del EPIC 3."
        )
    return pl.read_parquet(path)


def _prepare_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Valida y limpia el DataFrame de features para el baseline.

    Descarta las clases no agronomicas PASTIS-R (0 Background, 19 Void) y
    elimina filas sin ``class_id``.

    Args:
        df: DataFrame Polars crudo de features.

    Returns:
        El DataFrame filtrado, listo para entrenar.

    Raises:
        ValueError: si faltan ``parcel_id`` o ``class_id``, o si tras el
            filtrado no quedan filas.
    """
    for col in ("parcel_id", "class_id"):
        if col not in df.columns:
            raise ValueError(f"`df` debe contener la columna obligatoria `{col}`.")

    clean = df.filter(
        pl.col("class_id").is_not_null()
        & ~pl.col("class_id").is_in(list(_DROP_CLASS_IDS))
    )
    if clean.height == 0:
        raise ValueError(
            "Tras descartar las clases no agronomicas el DataFrame quedo vacio."
        )

    # El dataset real trae +/-inf en algunas pendientes/ratios espectrales;
    # los normalizamos a null para que el scaler (que solo trata NaN) y la
    # imputacion downstream los manejen de forma uniforme.
    float_cols = [c for c in clean.columns if clean.schema[c] in (pl.Float32, pl.Float64)]
    if float_cols:
        clean = clean.with_columns(
            pl.when(pl.col(c).is_infinite())
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in float_cols
        )
    return clean


def _feature_columns(df: pl.DataFrame) -> tuple[str, ...]:
    """Devuelve las columnas numericas usables como features.

    Excluye la metadata (``_META_COLS``) y cualquier columna no numerica.

    Args:
        df: DataFrame Polars ya preparado.

    Returns:
        Tupla ordenada de nombres de columnas de feature.

    Raises:
        ValueError: si no queda ninguna columna de feature.
    """
    cols = [
        c
        for c in df.columns
        if c not in _META_COLS
        and not c.endswith(_META_SUFFIXES)
        and df.schema[c].is_numeric()
    ]
    if not cols:
        raise ValueError("No se encontraron columnas numericas de feature en `df`.")
    return tuple(cols)


def _feature_matrix(df: pl.DataFrame, feature_cols: tuple[str, ...]) -> np.ndarray:
    """Extrae la matriz de features como ``np.ndarray`` float64.

    Args:
        df: DataFrame Polars ya preparado.
        feature_cols: Columnas a seleccionar, en orden.

    Returns:
        Matriz ``(n_samples, n_features)`` de dtype float64.
    """
    return df.select(feature_cols).to_numpy().astype(np.float64)


def _encode_labels(df: pl.DataFrame) -> tuple[LabelEncoder, np.ndarray]:
    """Codifica ``class_id`` a etiquetas contiguas ``[0, n_classes)``.

    PASTIS-R no tiene class_ids contiguos tras descartar 0 y 19; XGBoost
    ``multi:softprob`` exige etiquetas contiguas (decision D12).

    Args:
        df: DataFrame Polars ya preparado.

    Returns:
        Tupla ``(encoder, y_encoded)`` con el ``LabelEncoder`` ajustado y
        el vector de etiquetas codificadas.
    """
    raw = df.get_column("class_id").to_numpy().astype(np.int64)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(raw)
    return encoder, y_encoded.astype(np.int64)


def _base_params(model: ModelKind) -> dict[str, object]:
    """Devuelve una copia de los hiperparametros base del modelo dado."""
    if model == "rf":
        return dict(_RF_BASE_PARAMS)
    if model == "xgb":
        return dict(_XGB_BASE_PARAMS)
    if model == "lgbm":
        return dict(_LGBM_BASE_PARAMS)
    raise ValueError(f"`model` debe ser 'rf', 'xgb' o 'lgbm'; recibido {model!r}.")


def _default_grid(model: ModelKind) -> dict[str, list]:
    """Devuelve una copia de la grilla de tuning ligero del modelo dado."""
    if model == "rf":
        grid = _RF_PARAM_GRID
    elif model == "xgb":
        grid = _XGB_PARAM_GRID
    elif model == "lgbm":
        grid = _LGBM_PARAM_GRID
    else:
        raise ValueError(
            f"`model` debe ser 'rf', 'xgb' o 'lgbm'; recibido {model!r}."
        )
    return {k: list(v) for k, v in grid.items()}


def _sample_weights(y_encoded: np.ndarray) -> np.ndarray:
    """Calcula pesos por muestra inversamente proporcionales a la frecuencia.

    Reproduce el efecto de ``class_weight="balanced"`` para XGBoost, que no
    expone ese parametro (decision D5).

    Args:
        y_encoded: Vector de etiquetas codificadas.

    Returns:
        Vector de pesos ``(n_samples,)`` float64.
    """
    classes, counts = np.unique(y_encoded, return_counts=True)
    n_samples = y_encoded.size
    n_classes = classes.size
    weight_per_class = {
        int(c): n_samples / (n_classes * cnt) for c, cnt in zip(classes, counts, strict=True)
    }
    return np.array([weight_per_class[int(c)] for c in y_encoded], dtype=np.float64)


def _is_xgb(estimator: ClassifierMixin) -> bool:
    """Indica si ``estimator`` es un ``XGBClassifier``."""
    return isinstance(estimator, XGBClassifier)


def _is_lgbm(estimator: ClassifierMixin) -> bool:
    """Indica si ``estimator`` es un ``LGBMClassifier``."""
    return isinstance(estimator, LGBMClassifier)


def _column_medians(matrix: np.ndarray) -> np.ndarray:
    """Calcula la mediana de cada columna ignorando NaN e infinitos.

    Args:
        matrix: Matriz ``(n_samples, n_features)`` que puede contener NaN
            o ``+/-inf`` (el dataset real trae ``inf`` en algunas
            pendientes/ratios espectrales).

    Returns:
        Vector ``(n_features,)`` de medianas; ``0.0`` para columnas
        enteramente no-finitas.
    """
    finite = np.where(np.isfinite(matrix), matrix, np.nan)
    medians = np.nanmedian(finite, axis=0)
    return np.where(np.isnan(medians), 0.0, medians)


def _impute_with(matrix: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """Imputa valores no finitos usando un vector de medianas precomputado.

    Trata ``NaN`` y ``+/-inf`` por igual: sklearn no acepta ninguno.

    Args:
        matrix: Matriz ``(n_samples, n_features)`` que puede contener NaN
            o infinitos.
        medians: Vector ``(n_features,)`` de valores de imputacion (las
            medianas del split train, para evitar leakage hacia test).

    Returns:
        Una copia de la matriz con todos los valores finitos.
    """
    out = np.array(matrix, dtype=np.float64, copy=True)
    non_finite = ~np.isfinite(out)
    if not non_finite.any():
        return out
    bad_idx = np.where(non_finite)
    out[bad_idx] = np.take(medians, bad_idx[1])
    return out


def _impute(matrix: np.ndarray) -> np.ndarray:
    """Imputa NaN por la mediana de cada columna del propio ``matrix``.

    Atajo para el ajuste final sobre el dataset completo, donde no aplica
    la separacion train/test.

    Args:
        matrix: Matriz ``(n_samples, n_features)`` que puede contener NaN.

    Returns:
        Una copia de la matriz sin NaN.
    """
    return _impute_with(matrix, _column_medians(matrix))


# ---------------------------------------------------------------------------
# Helpers privados — CV espacial.
# ---------------------------------------------------------------------------


_SPATIAL_FOLDS_CACHE_DIR = Path("data/test_fixtures")


def _spatial_folds_cache_path(
    n_rows: int, k_folds: int, buffer_km: float, random_state: int
) -> Path:
    """Ruta del parquet de caché de los splits espaciales.

    La clave incluye el numero de filas, ``k``, el buffer y la semilla:
    cualquier cambio invalida el caché y fuerza recomputar.
    """
    buffer_tag = f"{buffer_km:g}".replace(".", "p")
    name = (
        f"baseline_spatial_folds_n{n_rows}_k{k_folds}"
        f"_b{buffer_tag}_s{random_state}.parquet"
    )
    return _SPATIAL_FOLDS_CACHE_DIR / name


def _load_cached_cv_splits(path: Path) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Lee los splits espaciales cacheados, o ``None`` si no existen."""
    if not path.exists():
        return None
    try:
        cached = pl.read_parquet(path)
    except (OSError, pl.exceptions.PolarsError) as exc:  # pragma: no cover
        logger.warning("spatial_folds_cache_unreadable", path=str(path), error=str(exc))
        return None
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in sorted(cached["fold"].unique().to_list()):
        fold_df = cached.filter(pl.col("fold") == fold_idx)
        train_idx = fold_df.filter(pl.col("split") == "train")["idx"].to_numpy()
        test_idx = fold_df.filter(pl.col("split") == "test")["idx"].to_numpy()
        splits.append(
            (train_idx.astype(np.int64), test_idx.astype(np.int64))
        )
    logger.info("spatial_folds_cache_hit", path=str(path), n_folds=len(splits))
    return splits


def _save_cached_cv_splits(
    path: Path, splits: list[tuple[np.ndarray, np.ndarray]]
) -> None:
    """Persiste los splits espaciales a parquet para futuras corridas."""
    rows: list[dict[str, object]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        rows.extend({"fold": fold_idx, "split": "train", "idx": int(i)} for i in train_idx)
        rows.extend({"fold": fold_idx, "split": "test", "idx": int(i)} for i in test_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)
    logger.info("spatial_folds_cache_saved", path=str(path), n_folds=len(splits))


def _build_cv_splits(
    df: pl.DataFrame,
    *,
    k_folds: int,
    buffer_km: float,
    random_state: int,
    use_cache: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Convierte los folds espaciales en splits posicionales ``(train, test)``.

    Construye un :class:`geopandas.GeoDataFrame` con un ``parcel_id`` entero
    sintetico (posicion en el DataFrame) y la geometria del centroide del
    patch PASTIS-R, llama a :func:`build_spatial_kfold` y traduce los
    ``parcel_id`` de cada :class:`FoldAssignment` a indices posicionales.

    ``build_spatial_kfold`` es O(N^2) por el buffer anti-leakage; sobre 85k
    parcelas tarda minutos. Por eso los splits se cachean en un parquet
    (clave: n_filas + k + buffer + seed) y se reusan en corridas
    posteriores (handoff US-019 R3).

    Args:
        df: DataFrame Polars ya preparado.
        k_folds: Numero de folds.
        buffer_km: Buffer anti-leakage en km.
        random_state: Semilla determinista.
        use_cache: Si ``True`` (default) lee/escribe el caché de splits.

    Returns:
        Lista de tuplas ``(train_idx, test_idx)`` de arrays de indices
        posicionales, una por fold con muestras en ambos lados.
    """
    cache_path = _spatial_folds_cache_path(
        df.height, k_folds, buffer_km, random_state
    )
    if use_cache:
        cached = _load_cached_cv_splits(cache_path)
        if cached is not None:
            return cached

    logger.info(
        "spatial_folds_building",
        n_rows=df.height,
        k_folds=k_folds,
        buffer_km=buffer_km,
        note="O(N^2) — puede tardar minutos en datasets grandes",
    )
    parcels_gdf = _build_parcels_geodataframe(df)
    folds: list[FoldAssignment] = build_spatial_kfold(
        parcels_gdf,
        k=k_folds,
        buffer_km=buffer_km,
        random_state=random_state,
    )

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    n_rows = df.height
    all_idx = np.arange(n_rows, dtype=np.int64)
    for fold in folds:
        # train_ids del FoldAssignment ya equivalen a indices posicionales
        # porque el GeoDataFrame usa la posicion como `parcel_id` sintetico.
        train_pool = np.array(
            sorted(fold.train_ids) + sorted(fold.val_ids), dtype=np.int64
        )
        test_idx = np.array(sorted(fold.test_ids), dtype=np.int64)
        if train_pool.size == 0 or test_idx.size == 0:
            continue
        # Filtramos por seguridad contra ids fuera de rango.
        train_idx = train_pool[np.isin(train_pool, all_idx)]
        test_idx = test_idx[np.isin(test_idx, all_idx)]
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        splits.append((train_idx, test_idx))

    if not splits:
        raise ValueError(
            "El CV espacial no produjo ningun fold con train y test no vacios. "
            "Revisa el numero de parcelas o reduce `k_folds`."
        )
    if use_cache:
        _save_cached_cv_splits(cache_path, splits)
    return splits


def _build_parcels_geodataframe(df: pl.DataFrame):  # type: ignore[no-untyped-def]
    """Construye el GeoDataFrame de parcelas para el CV espacial.

    El dataset de features no incluye geometria; el centroide se deriva del
    ``patch_id`` PASTIS-R via ``data/PASTIS-R/metadata.geojson``. Cada
    parcela recibe un ``parcel_id`` sintetico igual a su posicion en el
    DataFrame para poder traducir folds a indices.

    Args:
        df: DataFrame Polars ya preparado.

    Returns:
        Un ``GeoDataFrame`` en EPSG:4326 con ``parcel_id`` (posicion) y
        ``geometry`` (centroide del patch, o jitter determinista si la
        metadata no esta disponible).
    """
    import geopandas as gpd
    from shapely.geometry import Point

    n_rows = df.height
    positions = np.arange(n_rows, dtype=np.int64)

    patch_centroids = _load_patch_centroids()
    if patch_centroids is not None and "patch_id" in df.columns:
        patch_ids = df.get_column("patch_id").to_numpy()
        coords = np.array(
            [patch_centroids.get(int(p), (np.nan, np.nan)) for p in patch_ids],
            dtype=np.float64,
        )
    else:
        coords = np.full((n_rows, 2), np.nan, dtype=np.float64)

    # Fallback determinista: si falta la metadata o algun patch, distribuye
    # los centroides en una rejilla pseudo-aleatoria estable por patch_id.
    missing = np.isnan(coords).any(axis=1)
    if missing.any():
        logger.warning(
            "spatial_cv_centroid_fallback",
            n_missing=int(missing.sum()),
            note="metadata.geojson ausente o incompleto; rejilla determinista por patch.",
        )
        key = (
            df.get_column("patch_id").to_numpy()
            if "patch_id" in df.columns
            else positions
        )
        rng = np.random.default_rng(20240519)
        # Centroides en una caja sobre Francia continental (PASTIS-R).
        grid = rng.uniform(low=[-1.0, 43.0], high=[7.0, 49.0], size=(n_rows, 2))
        # Asegura que parcelas del mismo patch compartan centroide.
        unique_keys, inverse = np.unique(key, return_inverse=True)
        per_key = rng.uniform(
            low=[-1.0, 43.0], high=[7.0, 49.0], size=(unique_keys.size, 2)
        )
        grid = per_key[inverse]
        coords[missing] = grid[missing]

    geometry = [Point(float(lon), float(lat)) for lon, lat in coords]
    return gpd.GeoDataFrame(
        {"parcel_id": positions},
        geometry=geometry,
        crs="EPSG:4326",
    )


def _load_patch_centroids() -> dict[int, tuple[float, float]] | None:
    """Carga los centroides de patch PASTIS-R desde ``metadata.geojson``.

    Returns:
        Mapa ``{patch_id: (lon, lat)}`` en EPSG:4326, o ``None`` si la
        metadata no esta disponible en disco.
    """
    if not _PASTIS_METADATA_PATH.exists():
        return None
    try:
        import geopandas as gpd

        meta = gpd.read_file(_PASTIS_METADATA_PATH)
        # Centroide en CRS proyectado (3857) para evitar el UserWarning de
        # geopandas sobre operaciones geometricas en CRS geografico; luego
        # se reproyecta a 4326 (lat/lng) que es lo que espera el consumidor.
        centroids = meta.geometry.to_crs("EPSG:3857").centroid.to_crs("EPSG:4326")
        id_col = "ID_PATCH" if "ID_PATCH" in meta.columns else meta.columns[0]
        return {
            int(pid): (float(geom.x), float(geom.y))
            for pid, geom in zip(meta[id_col], centroids, strict=True)
        }
    except (OSError, ValueError, KeyError) as exc:  # pragma: no cover
        logger.warning("pastis_metadata_load_failed", error=str(exc))
        return None


def _fit_fold_scaler(
    df: pl.DataFrame,
    *,
    feature_cols: tuple[str, ...],
    train_idx: np.ndarray,
    fold_idx: int,
):  # type: ignore[no-untyped-def]
    """Ajusta un :class:`StandardScaler` solo sobre el train del fold.

    Reutiliza :func:`fit_scaler_on_train` (anti-leakage); el scaler se
    persiste en un archivo temporal por fold que se descarta.

    Args:
        df: DataFrame Polars ya preparado.
        feature_cols: Columnas de feature.
        train_idx: Indices posicionales del train del fold.
        fold_idx: Indice del fold (para nombrar el archivo temporal).

    Returns:
        Tupla ``(scaler, scaler_cols)`` con el ``StandardScaler`` ajustado
        y la tupla de columnas que efectivamente conoce (puede ser un
        subconjunto de ``feature_cols`` si hubo columnas all-NaN).
    """
    import tempfile

    # `fit_scaler_on_train` filtra por `parcel_id`; sustituimos esa columna
    # por la posicion de fila para alinear con los `train_idx` posicionales.
    positional = (
        df.drop("parcel_id")
        .with_row_index(name="parcel_id")
        .with_columns(pl.col("parcel_id").cast(pl.Int64))
    )
    train_ids = tuple(int(i) for i in train_idx)
    with tempfile.TemporaryDirectory() as tmp:
        scaler_path = Path(tmp) / f"fold_{fold_idx}_scaler.joblib"
        scaler = fit_scaler_on_train(
            positional,
            train_ids,
            feature_cols,
            scaler_path=scaler_path,
        )
    meta = getattr(scaler, "_agrosat_meta", {})
    scaler_cols = tuple(meta.get("feature_cols", feature_cols))
    return scaler, scaler_cols


def _aggregate_fold_metrics(
    per_fold: list[dict[str, float]],
) -> dict[str, tuple[float, float]]:
    """Agrega las metricas por fold en ``{metrica: (media, std)}``.

    Args:
        per_fold: Lista de diccionarios de metricas, uno por fold.

    Returns:
        Mapa ``{metrica: (media, desviacion)}`` sobre las cinco metricas;
        ``(nan, nan)`` para cada metrica si no hubo folds validos.
    """
    if not per_fold:
        return {key: (float("nan"), float("nan")) for key in _METRIC_KEYS}
    return {
        key: (
            float(np.mean([fold[key] for fold in per_fold])),
            float(np.std([fold[key] for fold in per_fold])),
        )
        for key in _METRIC_KEYS
    }
