"""Esquema canonico de ``parcel_id`` para el proyecto AgroSatCopilot.

Contexto y motivacion
---------------------

El identificador de parcela ``parcel_id`` viaja por multiples capas del
proyecto: GeoDataFrames de ingesta, parquets de embeddings (FarSLIP,
DINOv3, AlphaEarth), frames Polars de fusion (``ml.features.fusion``) y
caches de spatial CV. Historicamente convivieron dos representaciones:

1. ``Int64`` — heredada de PASTIS (``parcel_id`` numerico desde el GeoJSON
   original) y consumida por los samplers GEE.
2. ``Utf8`` — emergente desde los baselines que construyen el id como
   ``"{patch_id}_{i}"`` para identificar pixeles dentro de patches.

Cuando un ``LEFT JOIN`` mezcla ambos esquemas el resultado es silenciosa
mente vacio en Polars 1.x (el join produce NaN en todas las columnas del
lado derecho). El bug se manifesto en ``05_reencuadre_fenologico.ipynb``
con el FarSLIP omitido pese a existir el parquet canonico.

Esquema canonico
----------------

A partir de US-023-preview v2 el esquema oficial es::

    parcel_id: pl.Utf8  (siempre, en todo el proyecto)

Todo bloque que se incorpore via ``LEFT JOIN`` debe castear su columna
``parcel_id`` a ``pl.Utf8`` antes del join. La utilidad
:func:`canonical_parcel_id` aplica el cast de forma idempotente y sin
perder precision (no usa notacion cientifica para enteros grandes).
"""

from __future__ import annotations

import polars as pl
from polars.datatypes import DataType

__all__ = ["canonical_parcel_id"]


def _is_numeric_dtype(dtype: DataType) -> bool:
    """Devuelve ``True`` si ``dtype`` es Int/UInt/Float castable a Utf8 directo.

    En Polars 1.x ``cast(pl.Utf8)`` sobre enteros produce representacion
    decimal sin notacion cientifica; sobre floats puede producir notacion
    cientifica para valores muy grandes — en la practica los ``parcel_id``
    son enteros.
    """
    return dtype.is_integer() or dtype.is_float()


def canonical_parcel_id(df: pl.DataFrame, col: str = "parcel_id") -> pl.DataFrame:
    """Normaliza la columna ``parcel_id`` al esquema canonico ``pl.Utf8``.

    La funcion es idempotente: si ``col`` ya es ``pl.Utf8``, devuelve el
    DataFrame sin cambios. Para columnas numericas (Int8..Int64, UInt*,
    Float32/Float64) aplica un ``cast(pl.Utf8)`` que preserva el valor
    decimal sin notacion cientifica para enteros razonables.

    Args:
        df: DataFrame de Polars cuya columna ``col`` se va a normalizar.
        col: Nombre de la columna a normalizar. Default ``"parcel_id"``.

    Returns:
        Un nuevo ``pl.DataFrame`` con ``col`` en dtype ``pl.Utf8``. Si
        ``col`` ya era ``Utf8``, se devuelve ``df`` tal cual (sin clonado
        defensivo — Polars maneja copy-on-write internamente).

    Raises:
        KeyError: si ``col`` no existe en ``df.columns``. Se levanta un
            error explicito para evitar bugs silenciosos cuando el caller
            confunde el nombre de la columna.
        TypeError: si ``col`` no es numerica ni ``Utf8`` (ej. ``Datetime``,
            ``List``, ``Struct``). Estos tipos requieren conversion
            explicita por el caller.

    Examples:
        Sirve para alinear el esquema antes de un ``LEFT JOIN``::

            >>> base = pl.DataFrame({"parcel_id": ["p1", "p2"]})
            >>> rhs = pl.DataFrame({"parcel_id": [1, 2], "x": [10.0, 20.0]})
            >>> rhs = canonical_parcel_id(rhs)
            >>> base.join(rhs, on="parcel_id", how="left")  # ahora ambos Utf8
    """
    if col not in df.columns:
        raise KeyError(
            f"Columna '{col}' no esta presente en el DataFrame. "
            f"Columnas disponibles: {df.columns}"
        )
    dtype = df.schema[col]
    if dtype == pl.Utf8:
        return df
    if _is_numeric_dtype(dtype):
        return df.with_columns(pl.col(col).cast(pl.Utf8).alias(col))
    raise TypeError(
        f"Columna '{col}' tiene dtype no soportado para casteo canonico: "
        f"{dtype}. Tipos aceptados: pl.Utf8 o un dtype numerico "
        f"(Int*, UInt*, Float32, Float64)."
    )
