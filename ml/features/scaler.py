"""Persistencia del :class:`StandardScaler` ajustado sobre el split train (US-016).

El scaler se ajusta solo con los ``parcel_id`` del split train (Fold-0 por
convención) para evitar leakage hacia val/test. Persiste con :mod:`joblib`
(no ``pickle`` desnudo) — formato firmado, soportado por scikit-learn y
compatible con DVC tracking.

El frame ``StandardScaler`` recibido aquí siempre debe ser el output de
:func:`ml.features.fusion.build_fused_features`. Las columnas categóricas
(``srtm_aspect_dominant``) se excluyen automáticamente si están presentes
en ``feature_cols``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import polars as pl
import structlog
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger(__name__)

__all__ = [
    "fit_scaler_on_train",
    "load_scaler",
]


# ---------------------------------------------------------------------------
# API pública.
# ---------------------------------------------------------------------------


def fit_scaler_on_train(
    df: pl.DataFrame,
    train_ids: tuple[int, ...],
    feature_cols: tuple[str, ...],
    *,
    scaler_path: Path | str,
    version: str = "v1",
    val_ids: tuple[int, ...] | None = None,
    test_ids: tuple[int, ...] | None = None,
) -> StandardScaler:
    """Ajusta un :class:`StandardScaler` sobre el subset train y lo persiste.

    Args:
        df: DataFrame fusionado (output de
            :func:`ml.features.fusion.build_fused_features`).
        train_ids: ``parcel_id`` del split train. Debe ser disjunto de
            ``val_ids`` y ``test_ids`` (validado si se proporcionan).
        feature_cols: Nombres de columnas numéricas a estandarizar.
            Las columnas categóricas presentes (ej.
            ``srtm_aspect_dominant``) se filtran silenciosamente con un
            log.warning.
        scaler_path: Destino joblib. Convención
            ``artifacts/scaler_{version}.pkl``. El directorio padre se
            crea si no existe.
        version: Tag de versión inyectado en los metadatos del scaler para
            trazabilidad downstream.
        val_ids: ``parcel_id`` del split val (opcional, para validar
            ausencia de leakage).
        test_ids: ``parcel_id`` del split test (opcional, mismo propósito).

    Returns:
        El :class:`StandardScaler` ajustado. Atributos adicionales en
        ``_agrosat_meta``: ``{"version", "feature_cols", "n_train"}``.

    Raises:
        ValueError: si ``train_ids`` interseca con ``val_ids``/``test_ids``,
            si ``feature_cols`` está vacío tras filtrar categóricas, o si
            ``df`` no contiene ``parcel_id``.
    """
    if "parcel_id" not in df.columns:
        raise ValueError("`df` debe contener la columna `parcel_id`.")

    if val_ids is not None or test_ids is not None:
        _validate_no_leakage(train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)

    numeric_cols = _filter_numeric(df=df, feature_cols=feature_cols)
    if not numeric_cols:
        raise ValueError(
            "No quedan columnas numéricas tras filtrar categóricas; revisa `feature_cols`."
        )

    train_set = set(int(x) for x in train_ids)
    train_df = df.filter(pl.col("parcel_id").is_in(list(train_set))).select(numeric_cols)
    if train_df.height == 0:
        raise ValueError(
            "Tras filtrar por `train_ids` el frame está vacío. ¿IDs en otro fold?"
        )

    matrix = train_df.to_numpy()
    # Filtramos columnas all-NaN antes del fit para evitar `RuntimeWarning: Mean
    # of empty slice` en np.nanmean + `invalid value encountered in divide` en
    # sklearn. Ocurre cuando el frame proviene del modo demo sin GEE (todas las
    # cols de bloques no inyectados son null).
    all_nan_mask = np.all(np.isnan(matrix), axis=0)
    if all_nan_mask.any():
        dropped_all_nan = [c for c, drop in zip(numeric_cols, all_nan_mask, strict=True) if drop]
        logger.warning(
            "scaler_dropped_all_nan_columns",
            n_dropped=len(dropped_all_nan),
            examples=dropped_all_nan[:5],
            note="Frame sin GEE poblado; el scaler ignora estas columnas.",
        )
        numeric_cols = [c for c, drop in zip(numeric_cols, all_nan_mask, strict=True) if not drop]
        if not numeric_cols:
            raise ValueError(
                "Todas las columnas numéricas eran all-NaN. ¿Frame sin GEE poblado?"
            )
        matrix = matrix[:, ~all_nan_mask]
    # Reemplazamos NaN remanentes por la media de la columna (StandardScaler
    # no acepta NaN). Ya garantizamos que ninguna columna es all-NaN, así
    # nanmean no emite warnings.
    col_means = np.nanmean(matrix, axis=0)
    inds = np.where(np.isnan(matrix))
    matrix[inds] = np.take(col_means, inds[1])

    scaler = StandardScaler()
    scaler.fit(matrix)

    # Inyecta metadata para downstream + auditoría.
    scaler._agrosat_meta = {  # type: ignore[attr-defined]
        "version": version,
        "feature_cols": tuple(numeric_cols),
        "n_train": int(train_df.height),
    }

    out_path = Path(scaler_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out_path)
    logger.info(
        "scaler_persisted",
        path=str(out_path),
        version=version,
        n_features=len(numeric_cols),
        n_train=int(train_df.height),
    )
    return scaler


def load_scaler(path: Path | str) -> StandardScaler:
    """Carga un :class:`StandardScaler` persistido con :func:`fit_scaler_on_train`.

    Args:
        path: Ruta al fichero joblib (``artifacts/scaler_v1.pkl`` por
            convención).

    Returns:
        Scaler con metadata ``_agrosat_meta`` si fue ajustado por este
        módulo, o el scaler raw si fue producido por otro pipeline.

    Raises:
        FileNotFoundError: si ``path`` no existe.
        ValueError: si el archivo no es un :class:`StandardScaler` válido.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Scaler no encontrado en {p}.")
    obj = joblib.load(p)
    if not isinstance(obj, StandardScaler):
        raise ValueError(
            f"El archivo en {p} no es un StandardScaler (tipo={type(obj).__name__})."
        )
    return cast(StandardScaler, obj)


# ---------------------------------------------------------------------------
# Helpers privados.
# ---------------------------------------------------------------------------


def _validate_no_leakage(
    *,
    train_ids: tuple[int, ...],
    val_ids: tuple[int, ...] | None,
    test_ids: tuple[int, ...] | None,
) -> None:
    """Lanza :class:`ValueError` si ``train_ids`` interseca val/test."""
    train_set = set(int(x) for x in train_ids)
    for label, ids in (("val", val_ids), ("test", test_ids)):
        if not ids:
            continue
        overlap = train_set.intersection(int(x) for x in ids)
        if overlap:
            raise ValueError(
                f"Leakage detectado: train_ids interseca {label}_ids en "
                f"{len(overlap)} parcel_ids (ej. {sorted(overlap)[:3]}...)."
            )


def _filter_numeric(*, df: pl.DataFrame, feature_cols: tuple[str, ...]) -> list[str]:
    """Filtra `feature_cols` quedándose solo con las numéricas presentes en df."""
    keep: list[str] = []
    dropped: list[str] = []
    for col in feature_cols:
        if col not in df.columns:
            dropped.append(col)
            continue
        dtype: Any = df.schema[col]
        # Polars dtype helpers: numeric == Float* | Int* | UInt*.
        if dtype.is_numeric():
            keep.append(col)
        else:
            dropped.append(col)
    if dropped:
        logger.warning(
            "scaler_dropped_non_numeric",
            n_dropped=len(dropped),
            examples=dropped[:5],
        )
    return keep
