"""Tests de ``ml.utils.parcel_id.canonical_parcel_id`` (US-023-preview v2).

Garantizan que el esquema canonico ``parcel_id: pl.Utf8`` se aplica de
forma idempotente sobre los tres origenes practicos del proyecto:

- columnas ya ``Utf8`` (no-op),
- columnas ``Int64`` heredadas de PASTIS / FarSLIP v2,
- columnas ``Float`` (defensivo, no esperado en produccion pero soportado).

Tambien verifica que la ausencia de la columna levanta ``KeyError`` con
mensaje explicito (evita bugs silenciosos de typo en el nombre).
"""

from __future__ import annotations

import polars as pl
import pytest

from ml.utils.parcel_id import canonical_parcel_id


def test_canonical_parcel_id_noop_when_already_utf8() -> None:
    """Si la columna ya es Utf8, el DataFrame se devuelve sin alterar."""
    df = pl.DataFrame(
        {"parcel_id": ["10000_0", "10000_1", "10001_2"], "x": [1, 2, 3]},
        schema={"parcel_id": pl.Utf8, "x": pl.Int64},
    )
    out = canonical_parcel_id(df)
    assert out.schema["parcel_id"] == pl.Utf8
    assert out.get_column("parcel_id").to_list() == ["10000_0", "10000_1", "10001_2"]


def test_canonical_parcel_id_casts_int64_to_utf8() -> None:
    """Int64 -> Utf8 sin notacion cientifica y preservando el valor decimal."""
    df = pl.DataFrame(
        {"parcel_id": [1, 2, 99_999_999_999], "x": [10.0, 20.0, 30.0]},
        schema={"parcel_id": pl.Int64, "x": pl.Float64},
    )
    out = canonical_parcel_id(df)
    assert out.schema["parcel_id"] == pl.Utf8
    assert out.get_column("parcel_id").to_list() == ["1", "2", "99999999999"]


def test_canonical_parcel_id_casts_float_to_utf8() -> None:
    """Float* -> Utf8 (path defensivo). El valor se preserva razonable."""
    df = pl.DataFrame(
        {"parcel_id": [1.0, 42.0, 100.0]},
        schema={"parcel_id": pl.Float64},
    )
    out = canonical_parcel_id(df)
    assert out.schema["parcel_id"] == pl.Utf8
    # Polars representa floats integrales como "1.0", "42.0", "100.0".
    values = out.get_column("parcel_id").to_list()
    assert all(v.startswith(("1", "4", "1")) for v in values)
    assert len(values) == 3


def test_canonical_parcel_id_missing_column_raises_keyerror() -> None:
    """Columna inexistente => KeyError con mensaje explicativo."""
    df = pl.DataFrame({"other": [1, 2, 3]})
    with pytest.raises(KeyError, match="parcel_id"):
        canonical_parcel_id(df)


def test_canonical_parcel_id_unsupported_dtype_raises_typeerror() -> None:
    """Dtype no soportado (ej. lista) => TypeError descriptivo."""
    df = pl.DataFrame(
        {"parcel_id": [[1, 2], [3, 4]]},
        schema={"parcel_id": pl.List(pl.Int64)},
    )
    with pytest.raises(TypeError, match="dtype"):
        canonical_parcel_id(df)


def test_canonical_parcel_id_custom_col_name() -> None:
    """Acepta un nombre de columna distinto al default."""
    df = pl.DataFrame({"pid": [1, 2, 3]}, schema={"pid": pl.Int32})
    out = canonical_parcel_id(df, col="pid")
    assert out.schema["pid"] == pl.Utf8
    assert out.get_column("pid").to_list() == ["1", "2", "3"]
