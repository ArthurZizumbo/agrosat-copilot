"""Tests para `ml.ingest.pastis_loader`.

El smoke test sobre datos reales está marcado como `integration` y se
salta automáticamente cuando `data/PASTIS-R/DATA_S2/` no está presente
(CI sin dataset descargado). En local con PASTIS-R descomprimido los
tests verifican shapes y tipos del patch real `S2_10000.npy`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ml.ingest.pastis_loader import (
    PASTIS_S2_BANDS,
    iter_pastis_patches,
    load_pastis_patch,
    pastis_to_polars,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PASTIS_ROOT = REPO_ROOT / "data" / "PASTIS-R"
PASTIS_S2_DIR = PASTIS_ROOT / "DATA_S2"

pastis_present = PASTIS_S2_DIR.exists() and any(PASTIS_S2_DIR.glob("S2_*.npy"))

integration = pytest.mark.integration
skip_no_data = pytest.mark.skipif(
    not pastis_present,
    reason="PASTIS-R no descargado en data/PASTIS-R/DATA_S2/ — smoke test saltado.",
)


def _first_available_patch_id() -> str:
    """Retorna el primer patch_id disponible en disco (ej. '10000')."""
    sample = next(PASTIS_S2_DIR.glob("S2_*.npy"))
    return sample.stem.replace("S2_", "")


def test_pastis_s2_bands_canonical() -> None:
    """El orden canónico de bandas PASTIS expone 10 bandas (sin atmosféricas B01/B09/B10)."""
    assert len(PASTIS_S2_BANDS) == 10
    assert PASTIS_S2_BANDS[0] == "B02"
    assert "B01" not in PASTIS_S2_BANDS
    assert "B09" not in PASTIS_S2_BANDS
    assert "B10" not in PASTIS_S2_BANDS


def test_load_pastis_patch_missing_raises(tmp_path: Path) -> None:
    """Carga de patch inexistente lanza FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_pastis_patch("999999", root=tmp_path)


def test_pastis_to_polars_empty_when_root_missing(tmp_path: Path) -> None:
    """Sin patches en disco retorna DataFrame vacío con esquema correcto."""
    out = pastis_to_polars(["nonexistent"], root=tmp_path)
    assert out.is_empty()
    assert "patch_id" in out.columns
    assert "band" in out.columns
    assert "value" in out.columns


def test_iter_pastis_patches_skips_missing(tmp_path: Path) -> None:
    """`iter_pastis_patches` debe ignorar patches inexistentes sin levantar."""
    patches = list(iter_pastis_patches(["nope1", "nope2"], root=tmp_path))
    assert patches == []


@integration
@skip_no_data
def test_load_pastis_patch_real_shape_and_dtype() -> None:
    """Smoke test sobre patch real PASTIS-R: shape (T,10,128,128) int16."""
    pid = _first_available_patch_id()
    patch = load_pastis_patch(pid, root=PASTIS_ROOT, load_annotations=True)
    s2 = patch["s2"]
    assert isinstance(s2, np.ndarray)
    assert s2.ndim == 4
    T, C, H, W = s2.shape
    assert C == 10
    assert H == 128
    assert W == 128
    assert T >= 30  # T ∈ [38, 61] según planning, mínimo razonable
    assert s2.dtype == np.int16
    assert patch["patch_id"] == pid
    # `dates_s2` debería ser lista de ints YYYYMMDD ordenada (si metadata existe).
    if patch["dates_s2"]:
        assert all(isinstance(d, int) for d in patch["dates_s2"])
        assert patch["dates_s2"] == sorted(patch["dates_s2"])
        assert len(patch["dates_s2"]) == T


@integration
@skip_no_data
def test_load_pastis_patch_annotations_shape() -> None:
    """`semantic` debe ser (128,128) uint8 con valores en [0,19]."""
    pid = _first_available_patch_id()
    patch = load_pastis_patch(pid, root=PASTIS_ROOT, load_annotations=True)
    sem = patch["semantic"]
    if sem is None:
        pytest.skip("ANNOTATIONS/ no descargado.")
    assert sem.shape == (128, 128)
    assert sem.dtype == np.uint8
    assert int(sem.min()) >= 0
    assert int(sem.max()) <= 19


@integration
@skip_no_data
def test_pastis_to_polars_long_format_real() -> None:
    """Patch real convertido a Polars long-format produce columnas esperadas."""
    pid = _first_available_patch_id()
    df = pastis_to_polars(
        [pid],
        bands=["B04", "B08"],
        root=PASTIS_ROOT,
        include_labels=True,
        include_dates=True,
        pixel_stride=32,  # 4x4 = 16 px por banda x t para mantener test rápido
    )
    assert not df.is_empty()
    expected_cols = {"patch_id", "t", "date", "y", "x", "band", "value", "class_id", "fold"}
    assert expected_cols.issubset(set(df.columns))
    assert set(df["band"].unique().to_list()) == {"B04", "B08"}


def test_class_mapping_json_present_if_data_present() -> None:
    """Si `data/reference/pastis_class_mapping.json` existe, debe tener 20 clases."""
    path = REPO_ROOT / "data" / "reference" / "pastis_class_mapping.json"
    if not path.exists():
        pytest.skip("pastis_class_mapping.json no presente en este checkout.")
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    classes = raw.get("classes", {})
    assert len(classes) == 20
