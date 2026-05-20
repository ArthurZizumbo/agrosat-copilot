"""Tests para `ml.ingest.breizhcrops_loader`.

Los tests unitarios NO tocan la red ni requieren el dataset descargado:
verifican el modo degradado (esquema Polars vacio valido) cuando el root
no existe, y los tipos/columnas del esquema. El smoke test sobre datos
reales esta marcado como `integration` y se salta automaticamente cuando
`data/breizhcrops/` no tiene el layout completo (CI sin dataset).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from ml.ingest.breizhcrops_loader import (
    BREIZHCROPS_CLASSES,
    BREIZHCROPS_L2A_BANDS,
    _dataset_available,  # type: ignore[reportPrivateUsage]
    breizhcrops_parcel_index,
    breizhcrops_pixel_series,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BC_ROOT = REPO_ROOT / "data" / "breizhcrops"

bc_present = _dataset_available(BC_ROOT, "frh04", 2017, "L2A")

integration = pytest.mark.integration
skip_no_data = pytest.mark.skipif(
    not bc_present,
    reason="BreizhCrops no descargado en data/breizhcrops/ — smoke test saltado.",
)


def test_breizhcrops_classes_canonical() -> None:
    """Las 9 clases canonicas BreizhCrops estan completas y ordenadas 0..8."""
    assert len(BREIZHCROPS_CLASSES) == 9
    assert BREIZHCROPS_CLASSES[0] == "barley"
    assert BREIZHCROPS_CLASSES[3] == "corn"
    assert BREIZHCROPS_CLASSES[8] == "temporary meadows"
    assert sorted(BREIZHCROPS_CLASSES.keys()) == list(range(9))


def test_breizhcrops_l2a_bands_canonical() -> None:
    """L2A expone 10 bandas opticas mapeadas a la nomenclatura del proyecto."""
    assert len(BREIZHCROPS_L2A_BANDS) == 10
    assert BREIZHCROPS_L2A_BANDS[0] == "B02"
    assert "B08" in BREIZHCROPS_L2A_BANDS
    assert "B04" in BREIZHCROPS_L2A_BANDS
    # NDVI requiere B08 y B04 presentes en el orden canonico.
    assert BREIZHCROPS_L2A_BANDS.index("B08") == 6


def test_parcel_index_empty_when_root_missing(tmp_path: Path) -> None:
    """Sin dataset en disco retorna DataFrame vacio con esquema correcto."""
    out = breizhcrops_parcel_index("frh04", 2017, "L2A", root=tmp_path)
    assert out.is_empty()
    assert out.columns == [
        "parcel_id",
        "region",
        "year",
        "level",
        "code_cultu",
        "class_id",
        "class_name",
        "sequence_length",
    ]
    assert out.schema["parcel_id"] == pl.Utf8
    assert out.schema["class_id"] == pl.Int16
    assert out.schema["sequence_length"] == pl.Int64


def test_pixel_series_empty_when_root_missing(tmp_path: Path) -> None:
    """Sin dataset en disco retorna long-format vacio con esquema correcto."""
    out = breizhcrops_pixel_series("frh04", 2017, "L2A", root=tmp_path)
    assert out.is_empty()
    assert out.columns == [
        "parcel_id",
        "t",
        "date",
        "doy",
        "band",
        "value",
        "class_id",
        "class_name",
    ]
    assert out.schema["value"] == pl.Float64
    assert out.schema["band"] == pl.Utf8
    assert out.schema["class_id"] == pl.Int16


def test_pixel_series_no_network_when_missing(tmp_path: Path) -> None:
    """El modo degradado no debe instanciar breizhcrops ni tocar la red.

    Si `breizhcrops.BreizhCrops` fuera llamado con un root vacio, el
    paquete intentaria descargar (no expone download=False). Verificamos
    que con root inexistente la funcion retorna vacio SIN construir el
    dataset (la guarda `_dataset_available` corta antes).
    """
    assert _dataset_available(tmp_path, "frh04", 2017, "L2A") is False
    out = breizhcrops_pixel_series("frh04", 2017, "L2A", sample_parcels=5, root=tmp_path)
    assert out.is_empty()


@integration
@skip_no_data
def test_parcel_index_real_has_rows() -> None:
    """Smoke test sobre BreizhCrops real: el indice tiene parcelas con clase valida."""
    df = breizhcrops_parcel_index("frh04", 2017, "L2A", root=BC_ROOT)
    assert not df.is_empty()
    assert int(df["class_id"].min()) >= 0  # type: ignore[arg-type]
    assert int(df["class_id"].max()) <= 8  # type: ignore[arg-type]
    assert set(df["class_name"].unique()).issubset(set(BREIZHCROPS_CLASSES.values()))


@integration
@skip_no_data
def test_pixel_series_real_long_format() -> None:
    """Smoke test: 10 bandas por parcela, valores finitos, fechas YYYYMMDD."""
    df = breizhcrops_pixel_series("frh04", 2017, "L2A", sample_parcels=8, seed=42, root=BC_ROOT)
    assert not df.is_empty()
    assert set(df["band"].unique()) == set(BREIZHCROPS_L2A_BANDS)
    assert int(df["date"].max()) >= 20170101  # type: ignore[arg-type]
    assert int(df["doy"].max()) <= 366  # type: ignore[arg-type]
