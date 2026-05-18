"""Codificacion de variables categoricas y derivacion de atributos (US-018 ext, Avance 2).

Modulo complementario a :mod:`ml.features.selection` que cierra el bloque
"Construccion de features" del Avance 2 (rubrica 30 pts), cubriendo lo que el
notebook WIP del equipo (``notebooks/feature_engineering/02a_fe_sentinel2.ipynb``)
exploraba interactivamente con ``pandas.get_dummies`` y ``KBinsDiscretizer``.

Aqui la API publica es **Polars-first** (regla ``ml/CLAUDE.md NEVER pandas``):
todas las funciones reciben y devuelven :class:`polars.DataFrame` /
:class:`polars.Series`. ``numpy`` solo aparece de forma interna para los
calculos numericos.

API publica
-----------
- :func:`derive_crop_group_from_class_id` — colapsa las 20 clases PASTIS-R en
  ~5 grupos agronomicos reusando ``PASTIS_R_GROUPINGS["agronomic_group"]``
  del loader oficial (US-001).
- :func:`derive_season_from_doy` — convierte day-of-year a etiqueta de
  estacion (``winter/spring/summer/autumn``); util para sembrar
  :func:`encode_ordinal` cuando el feature de entrada es ``peak_doy`` u otro
  derivado fenologico.
- :func:`encode_onehot` — wrapper sobre :meth:`polars.DataFrame.to_dummies`
  con report de cardinalidad y soporte de ``drop_first``.
- :func:`encode_ordinal` — aplica un mapping explicito por columna
  (``dict[col, dict[valor, int]]``); valores desconocidos -> ``-1`` con
  warning estructurado.
- :func:`encode_target_mean` — target encoding bayesiano con smoothing
  (Galli 2022 cap. 3) para categoricas de alta cardinalidad sin explotar el
  ancho del DataFrame.

Decisiones clave
----------------
- Polars in / Polars out (ningun ``pandas`` import). Para one-hot se usa
  ``df.to_dummies(columns=..., separator="__")``; para discretizacion la
  cuenta queda en :func:`ml.features.selection.discretize_features`.
- ``exclude_cols`` excluye siempre ``parcel_id`` y ``year`` por convencion
  del proyecto (no son features candidatas a codificar).
- Cuando ``drop_first=True`` en one-hot, se elimina la primera categoria
  alfabeticamente (k -> k-1 columnas), siguiendo la convencion de modelos
  lineales para evitar colinealidad de la matriz indicadora completa.

Referencias
-----------
- Galli, S. (2022). *Python Feature Engineering Cookbook* (2nd ed.), cap. 3
  "Encoding Categorical Variables". Smoothing bayesiano para target encoding.
- Sainte-Fare-Garnot, V., Landrieu, L. (2021). *PASTIS dataset documentation*
  — 20 clases agrupadas en ``agronomic_group`` (cereals, root_crops,
  oilseeds_legumes, permanent_long_cycle, special_crops).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "derive_crop_group_from_class_id",
    "derive_season_from_doy",
    "encode_onehot",
    "encode_ordinal",
    "encode_target_mean",
]


# Convencion compartida con :mod:`ml.features.selection`.
_DEFAULT_EXCLUDE: tuple[str, ...] = ("parcel_id", "year")

# Mapping default para estaciones del hemisferio norte (mes -> estacion).
# Sigue la convencion meteorologica: winter = DEC/JAN/FEB (mes 12, 1, 2).
_SEASON_NORTH_BY_MONTH: dict[int, str] = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}

# Hemisferio sur: estaciones invertidas (winter = JUN/JUL/AUG).
_SEASON_SOUTH_BY_MONTH: dict[int, str] = {
    m: {
        "winter": "summer",
        "summer": "winter",
        "spring": "autumn",
        "autumn": "spring",
    }[s]
    for m, s in _SEASON_NORTH_BY_MONTH.items()
}

# Fallback default cuando ``PASTIS_R_GROUPINGS["agronomic_group"]`` no esta
# disponible (e.g. el JSON de referencia no se ha desplegado en el entorno).
# Coincide con la agrupacion agronomica publicada en el JSON oficial
# (Sainte-Fare-Garnot 2021, Figura 2).
_DEFAULT_CROP_GROUP_MAP: dict[int, str] = {
    0: "background",
    1: "permanent_long_cycle",
    2: "cereals",
    3: "cereals",
    4: "cereals",
    5: "oilseeds_legumes",
    6: "cereals",
    7: "oilseeds_legumes",
    8: "permanent_long_cycle",
    9: "root_crops",
    10: "cereals",
    11: "cereals",
    12: "special_crops",
    13: "root_crops",
    14: "oilseeds_legumes",
    15: "oilseeds_legumes",
    16: "permanent_long_cycle",
    17: "cereals",
    18: "special_crops",
    19: "void",
}


def _filter_exclude(columns: list[str], exclude_cols: tuple[str, ...]) -> list[str]:
    """Devuelve ``columns`` sin las que aparezcan en ``exclude_cols``."""
    return [c for c in columns if c not in exclude_cols]


# ---------------------------------------------------------------------------
# Helpers de derivacion (insumos tipicos del notebook WIP de Isaac)
# ---------------------------------------------------------------------------


def derive_season_from_doy(
    doy_series: pl.Series,
    *,
    hemisphere: Literal["north", "south"] = "north",
) -> pl.Series:
    """Convierte day-of-year (1..366) a etiqueta de estacion.

    Util para sembrar :func:`encode_ordinal` cuando el insumo es un feature
    fenologico tipo ``peak_doy`` o ``sog_doy`` (provenientes de
    ``ml.features.temporal_features``).

    Args:
        doy_series: Serie Polars con valores en ``[1, 366]``. Acepta floats;
            se redondea hacia abajo. NaN se mapea a ``"unknown"``.
        hemisphere: ``"north"`` (default) o ``"south"`` para invertir
            estaciones en el hemisferio sur.

    Returns:
        Serie ``pl.Series`` Utf8 con valores en
        ``{"winter", "spring", "summer", "autumn", "unknown"}`` y mismo
        ``name`` que ``doy_series`` con sufijo ``__season``.

    Notes:
        Aproxima ``month = ceil(doy / 30.5)`` con clamp a ``[1, 12]``. La
        precision es suficiente para estacionalidad agronomica (la diferencia
        de 1-2 dias en los bordes de mes no cambia la estacion).
    """
    season_map = _SEASON_NORTH_BY_MONTH if hemisphere == "north" else _SEASON_SOUTH_BY_MONTH
    raw = doy_series.cast(pl.Float64).to_numpy()
    months = np.clip(np.ceil(np.where(np.isnan(raw), 0.0, raw) / 30.5).astype(np.int64), 0, 12)
    out: list[str] = []
    for doy_val, mo in zip(raw.tolist(), months.tolist(), strict=True):
        if doy_val is None or (isinstance(doy_val, float) and np.isnan(doy_val)) or mo == 0:
            out.append("unknown")
        else:
            out.append(season_map.get(int(mo), "unknown"))
    name_suffix = f"{doy_series.name}__season" if doy_series.name else "season"
    return pl.Series(name_suffix, out, dtype=pl.Utf8)


def derive_crop_group_from_class_id(
    class_id_series: pl.Series,
    *,
    mapping: dict[int, str] | None = None,
) -> pl.Series:
    """Colapsa las 20 clases PASTIS-R en grupos agronomicos.

    Cuando ``mapping`` es ``None``, intenta cargar
    ``PASTIS_R_GROUPINGS["agronomic_group"]`` desde
    :mod:`ml.ingest.pastis_loader`; si la agrupacion no esta disponible
    (entorno sin ``data/reference/pastis_class_mapping.json``), cae al
    diccionario inline :data:`_DEFAULT_CROP_GROUP_MAP` documentado en el
    docstring del modulo.

    Args:
        class_id_series: Serie ``pl.Series`` Int con valores en ``[0, 19]``.
        mapping: Override opcional ``{class_id: nombre_grupo}``. Valores
            fuera del mapping se etiquetan como ``"unknown"``.

    Returns:
        Serie ``pl.Series`` Utf8 con grupo agronomico por fila; ``name``
        del input + sufijo ``__group`` (o ``"crop_group"`` si el input no
        tiene ``name``).
    """
    if mapping is None:
        try:
            from ml.ingest.pastis_loader import PASTIS_R_GROUPINGS

            mapping = PASTIS_R_GROUPINGS.get("agronomic_group") or _DEFAULT_CROP_GROUP_MAP
            source = "pastis_loader.PASTIS_R_GROUPINGS[agronomic_group]"
        except Exception as exc:  # noqa: BLE001
            logger.warning("pastis_loader_import_failed", error=str(exc))
            mapping = _DEFAULT_CROP_GROUP_MAP
            source = "default_inline_map"
    else:
        source = "caller_provided"

    raw = class_id_series.to_list()
    out = [mapping.get(int(v), "unknown") if v is not None else "unknown" for v in raw]
    out_name = f"{class_id_series.name}__group" if class_id_series.name else "crop_group"
    logger.info(
        "crop_group_derived",
        source=source,
        n=len(out),
        n_groups=len(set(out)),
    )
    return pl.Series(out_name, out, dtype=pl.Utf8)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


def encode_ordinal(
    df: pl.DataFrame,
    mapping: dict[str, dict[Any, int]],
    *,
    exclude_cols: tuple[str, ...] = _DEFAULT_EXCLUDE,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Aplica un mapping ordinal explicito por columna.

    Args:
        df: DataFrame Polars wide-format.
        mapping: ``{col_name: {valor_original: int_ordinal}}``. Cada columna
            se reemplaza por su version codificada manteniendo el mismo
            nombre. Valores no presentes en el mapping -> ``-1`` (con
            warning estructurado por columna afectada).
        exclude_cols: Columnas que NO se codifican aunque aparezcan en
            ``mapping`` (defensa contra ``parcel_id``/``year``).

    Returns:
        Tupla ``(df_encoded, report)`` donde ``report`` contiene
        ``{col: {"mapping": dict, "unknown_count": int}}`` por columna
        procesada.

    Raises:
        ValueError: Si alguna columna de ``mapping`` no existe en ``df``.
    """
    missing = [c for c in mapping if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columnas en mapping ausentes del DataFrame: {missing}. "
            f"Disponibles: {df.columns}"
        )

    out = df
    report: dict[str, Any] = {}
    for col, value_map in mapping.items():
        if col in exclude_cols:
            logger.warning("encode_ordinal_skip_excluded", col=col)
            continue
        original = out.get_column(col).to_list()
        encoded = [int(value_map.get(v, -1)) for v in original]
        unknown_count = sum(1 for e in encoded if e == -1)
        if unknown_count > 0:
            logger.warning(
                "encode_ordinal_unknown_values",
                col=col,
                unknown_count=unknown_count,
                total=len(encoded),
            )
        out = out.with_columns(pl.Series(col, encoded, dtype=pl.Int64))
        report[col] = {"mapping": dict(value_map), "unknown_count": unknown_count}

    logger.info(
        "encode_ordinal_done",
        cols_encoded=list(report.keys()),
        n_rows=df.height,
    )
    return out, report


def encode_onehot(
    df: pl.DataFrame,
    columns: list[str] | tuple[str, ...],
    *,
    drop_first: bool = False,
    exclude_cols: tuple[str, ...] = _DEFAULT_EXCLUDE,
) -> tuple[pl.DataFrame, dict[str, list[str]]]:
    """Codifica columnas categoricas con :meth:`polars.DataFrame.to_dummies`.

    Wrapper Polars-nativo sobre la API one-hot del propio Polars. **NO usa
    pandas** (regla ``ml/CLAUDE.md NEVER pandas``).

    Args:
        df: DataFrame Polars wide-format.
        columns: Lista/tupla de columnas a codificar. Las que aparezcan en
            ``exclude_cols`` se filtran defensivamente.
        drop_first: Si ``True``, elimina la primera categoria (orden
            alfabetico) de cada columna codificada. Reduce ``k`` columnas a
            ``k - 1`` para evitar colinealidad en modelos lineales.
        exclude_cols: Columnas a preservar sin codificar.

    Returns:
        Tupla ``(df_wide, report)`` donde:

        - ``df_wide`` reemplaza cada ``col`` por columnas
          ``{col}__{categoria}`` (separator fijo ``"__"``).
        - ``report = {col_original: [nuevas_columnas]}`` para trazabilidad.

    Raises:
        ValueError: Si alguna columna de ``columns`` no existe en ``df``.
    """
    cols_list = [c for c in columns if c not in exclude_cols]
    missing = [c for c in cols_list if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columnas a codificar ausentes del DataFrame: {missing}. "
            f"Disponibles: {df.columns}"
        )
    if not cols_list:
        return df, {}

    pre_cols = set(df.columns)
    encoded = df.to_dummies(columns=list(cols_list), separator="__")

    report: dict[str, list[str]] = {}
    for col in cols_list:
        new_cols_all = sorted(c for c in encoded.columns if c.startswith(f"{col}__"))
        if drop_first and new_cols_all:
            dropped = new_cols_all[0]
            encoded = encoded.drop(dropped)
            new_cols = new_cols_all[1:]
        else:
            new_cols = new_cols_all
        report[col] = new_cols

    new_total = len(encoded.columns) - (len(pre_cols) - len(cols_list))
    logger.info(
        "encode_onehot_done",
        cols_encoded=list(report.keys()),
        n_new_columns=new_total,
        drop_first=drop_first,
    )
    return encoded, report


def encode_target_mean(
    df: pl.DataFrame,
    target_col: str,
    columns: list[str] | tuple[str, ...],
    *,
    smoothing: float = 10.0,
    exclude_cols: tuple[str, ...] = _DEFAULT_EXCLUDE,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Aplica bayesian target mean encoding con smoothing (Galli 2022).

    Para cada categoria ``c`` en una columna a codificar:

    .. math::

        \\hat{y}_c = \\frac{n_c \\cdot \\bar{y}_c + m \\cdot \\bar{y}}{n_c + m}

    donde ``n_c`` es el numero de muestras de la categoria, ``mean_c`` es
    la media de ``target_col`` para esa categoria, ``mean_global`` es la
    media global del target y ``m`` es el smoothing (``smoothing > 0``
    desplaza categorias raras hacia la media global, evitando overfitting).

    Args:
        df: DataFrame Polars con la columna ``target_col`` presente.
        target_col: Nombre de la columna objetivo (numerica). En
            clasificacion multiclase, conviene binarizar antes (e.g.
            ``one-vs-rest`` por clase) o usar la media del entero como
            proxy ordinal de severidad.
        columns: Categoricas a codificar. ``parcel_id``/``year`` excluidas
            por defecto.
        smoothing: Factor ``m`` del smoothing bayesiano. ``m=0`` -> media
            por categoria pura; ``m -> inf`` -> media global. Galli 2022
            recomienda ``m in [5, 20]`` para datasets de tamano medio.
        exclude_cols: Columnas a no codificar.

    Returns:
        Tupla ``(df_encoded, report)`` donde:

        - ``df_encoded`` agrega columnas ``{col}_target_enc`` (mantiene la
          original para no perder informacion).
        - ``report`` contiene:
          ``{"global_mean": float, "per_column": {col: {cat: encoded_value}}}``.

    Raises:
        ValueError: Si ``target_col`` no existe en ``df`` o no es numerico.
    """
    if target_col not in df.columns:
        raise ValueError(f"target_col {target_col!r} no presente en df.columns")
    if not df.get_column(target_col).dtype.is_numeric():
        raise ValueError(
            f"target_col {target_col!r} debe ser numerico (es {df.get_column(target_col).dtype})"
        )

    cols_list = [c for c in columns if c not in exclude_cols]
    missing = [c for c in cols_list if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columnas a target-encodear ausentes del DataFrame: {missing}."
        )

    target_arr = df.get_column(target_col).cast(pl.Float64).to_numpy()
    global_mean = float(np.nanmean(target_arr)) if target_arr.size else 0.0
    if not np.isfinite(global_mean):
        global_mean = 0.0

    out = df
    per_column: dict[str, dict[Any, float]] = {}

    for col in cols_list:
        per_cat = (
            df.group_by(col)
            .agg(
                pl.len().alias("__n"),
                pl.col(target_col).cast(pl.Float64).mean().alias("__mean"),
            )
            .to_dict(as_series=False)
        )
        cat_to_enc: dict[Any, float] = {}
        for cat, n, mean_c in zip(per_cat[col], per_cat["__n"], per_cat["__mean"], strict=True):
            if mean_c is None or (isinstance(mean_c, float) and not np.isfinite(mean_c)):
                encoded_val = global_mean
            else:
                encoded_val = (n * mean_c + smoothing * global_mean) / (n + smoothing)
            cat_to_enc[cat] = float(encoded_val)
        original = out.get_column(col).to_list()
        encoded_series = pl.Series(
            f"{col}_target_enc",
            [cat_to_enc.get(v, global_mean) for v in original],
            dtype=pl.Float64,
        )
        out = out.with_columns(encoded_series)
        per_column[col] = cat_to_enc

    report: dict[str, Any] = {
        "global_mean": global_mean,
        "per_column": per_column,
        "smoothing": float(smoothing),
        "target_col": target_col,
    }
    logger.info(
        "encode_target_mean_done",
        cols_encoded=list(per_column.keys()),
        smoothing=smoothing,
        global_mean=global_mean,
    )
    return out, report
