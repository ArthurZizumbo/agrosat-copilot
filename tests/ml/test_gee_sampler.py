"""Tests para `ml.ingest.gee_sampler`.

Earth Engine NUNCA se llama de forma real desde tests — todo se mockea
con `unittest.mock`. Verifica:
- Esquema correcto del DataFrame retornado en éxito.
- Fallback degradado (DataFrame vacío con esquema correcto) cuando EE falla.
- Uso de cache parquet local para evitar re-llamadas.
- `init_ee` levanta ImportError limpio si `ee` no está instalado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from ml.ingest import gee_sampler
from ml.ingest.gee_sampler import (
    DEFAULT_S2_BANDS,
    _cache_key,
    init_ee,
    sample_s2_roi,
)


def _make_fake_ee_module(getinfo_payload: dict[str, Any]) -> MagicMock:
    """Construye un mock de módulo `ee` compatible con la chain usada en sample_s2_roi.

    La chain es:
        ee.ImageCollection(...).filterBounds(...).filterDate(...)
                              .filter(...).select(...).median().sample(...).getInfo()
    """
    fake_ee = MagicMock(name="ee")
    fake_sample = MagicMock(name="ee.sample_result")
    fake_sample.getInfo.return_value = getinfo_payload

    fake_median = MagicMock(name="ee.median")
    fake_median.sample.return_value = fake_sample

    fake_collection = MagicMock(name="ee.ImageCollection_chain")
    fake_collection.filterBounds.return_value = fake_collection
    fake_collection.filterDate.return_value = fake_collection
    fake_collection.filter.return_value = fake_collection
    fake_collection.select.return_value = fake_collection
    fake_collection.median.return_value = fake_median

    fake_ee.ImageCollection.return_value = fake_collection
    fake_ee.Filter.lt.return_value = MagicMock(name="ee.Filter.lt")
    fake_ee.Initialize = MagicMock()
    fake_ee.Authenticate = MagicMock()
    return fake_ee


@pytest.fixture
def fake_ee_success() -> MagicMock:
    """Mock EE con respuesta sintética de `sampleRegions` (2 features x 2 bandas)."""
    payload = {
        "features": [
            {
                "properties": {"B4": 1200.0, "B8": 3400.0, "B3": 900.0},
                "geometry": {"coordinates": [9.5, 45.2]},
            },
            {
                "properties": {"B4": 1500.0, "B8": 3700.0, "B3": 1000.0},
                "geometry": {"coordinates": [9.6, 45.3]},
            },
        ]
    }
    return _make_fake_ee_module(payload)


def test_sample_s2_roi_returns_dataframe_with_expected_schema(
    fake_ee_success: MagicMock, tmp_path: Path
) -> None:
    """Mock EE devuelve features -> DataFrame con columnas roi, date, band, value, lon, lat."""
    with patch.object(gee_sampler, "ee", fake_ee_success):
        out = sample_s2_roi(
            roi=MagicMock(name="ee.Geometry"),
            start_date="2024-04-01",
            end_date="2024-04-30",
            bands=["B3", "B4", "B8"],
            n_pixels=10,
            cache_path=tmp_path,
            roi_name="pianura_padana",
        )
    assert isinstance(out, pl.DataFrame)
    assert not out.is_empty()
    expected_cols = {"roi", "date", "band", "value", "lon", "lat"}
    assert expected_cols.issubset(set(out.columns))
    # 2 features x 3 bandas = 6 filas
    assert out.height == 6
    assert set(out["band"].unique().to_list()) == {"B3", "B4", "B8"}
    assert out["roi"].unique().to_list() == ["pianura_padana"]


def test_sample_s2_roi_caches_parquet(fake_ee_success: MagicMock, tmp_path: Path) -> None:
    """Segunda llamada con mismos parámetros debe leer del cache (no re-llamar a EE)."""
    with patch.object(gee_sampler, "ee", fake_ee_success):
        first = sample_s2_roi(
            roi=MagicMock(),
            start_date="2024-04-01",
            end_date="2024-04-30",
            bands=["B3", "B4"],
            n_pixels=5,
            cache_path=tmp_path,
            roi_name="toscana",
        )
        assert (
            tmp_path / _cache_key("toscana", "2024-04-01", "2024-04-30", 5)
        ).exists()

        # Reseteamos contadores y volvemos a llamar — EE no debe invocarse.
        fake_ee_success.ImageCollection.reset_mock()
        second = sample_s2_roi(
            roi=MagicMock(),
            start_date="2024-04-01",
            end_date="2024-04-30",
            bands=["B3", "B4"],
            n_pixels=5,
            cache_path=tmp_path,
            roi_name="toscana",
        )
    assert first.equals(second)
    fake_ee_success.ImageCollection.assert_not_called()


def test_sample_s2_roi_degraded_fallback_on_ee_exception(tmp_path: Path) -> None:
    """Si EE lanza excepción (quota/auth/red), retorna DataFrame vacío con esquema correcto."""
    fake_ee = MagicMock(name="ee")
    fake_ee.ImageCollection.side_effect = RuntimeError("simulated quota exceeded")
    fake_ee.Filter.lt.return_value = MagicMock()
    with patch.object(gee_sampler, "ee", fake_ee):
        out = sample_s2_roi(
            roi=MagicMock(),
            start_date="2024-04-01",
            end_date="2024-04-30",
            n_pixels=100,
            cache_path=tmp_path,
            roi_name="apulia",
        )
    assert isinstance(out, pl.DataFrame)
    assert out.is_empty()
    # Esquema preservado para no romper downstream Polars ops.
    expected_cols = {"roi", "date", "band", "value", "lon", "lat"}
    assert expected_cols.issubset(set(out.columns))
    # No se escribió cache cuando hubo fallback.
    assert not any(tmp_path.glob("*.parquet"))


def test_sample_s2_roi_returns_empty_when_ee_unavailable(tmp_path: Path) -> None:
    """Si `ee` es None (earthengine-api no instalado), retorna DataFrame vacío sin levantar."""
    with patch.object(gee_sampler, "ee", None):
        out = sample_s2_roi(
            roi=None,
            start_date="2024-04-01",
            end_date="2024-04-30",
            cache_path=tmp_path,
            roi_name="no_ee",
        )
    assert out.is_empty()
    expected_cols = {"roi", "date", "band", "value", "lon", "lat"}
    assert expected_cols.issubset(set(out.columns))


def test_init_ee_raises_importerror_when_ee_missing() -> None:
    """`init_ee` debe levantar ImportError claro si `ee` no está disponible."""
    with patch.object(gee_sampler, "ee", None):
        with pytest.raises(ImportError, match="earthengine-api"):
            init_ee()


def test_init_ee_calls_initialize_with_project() -> None:
    """`init_ee` sin service account llama a `ee.Initialize(project=...)`."""
    fake_ee = MagicMock(name="ee")
    fake_ee.Initialize = MagicMock()
    fake_ee.Authenticate = MagicMock()
    with patch.object(gee_sampler, "ee", fake_ee):
        init_ee(project="my-gcp-project")
    fake_ee.Initialize.assert_called_once_with(project="my-gcp-project")
    fake_ee.Authenticate.assert_not_called()


def test_init_ee_falls_back_to_authenticate_on_initialize_error() -> None:
    """Si `ee.Initialize` falla, debe disparar `ee.Authenticate` y reintentar Initialize."""
    fake_ee = MagicMock(name="ee")
    fake_ee.Initialize = MagicMock(side_effect=[RuntimeError("no creds"), None])
    fake_ee.Authenticate = MagicMock()
    with patch.object(gee_sampler, "ee", fake_ee):
        init_ee(project="proj")
    fake_ee.Authenticate.assert_called_once()
    assert fake_ee.Initialize.call_count == 2


def test_cache_key_is_reproducible() -> None:
    """`_cache_key` debe ser determinístico dada la misma tupla de inputs."""
    a = _cache_key("toscana", "2024-04-01", "2024-04-30", 100)
    b = _cache_key("toscana", "2024-04-01", "2024-04-30", 100)
    assert a == b
    assert a.endswith(".parquet")


def test_default_s2_bands_canonical() -> None:
    """Las bandas EE por defecto cubren las 10 bandas no atmosféricas."""
    assert len(DEFAULT_S2_BANDS) == 10
    assert "B8A" in DEFAULT_S2_BANDS
    assert "B1" not in DEFAULT_S2_BANDS  # atmosférica excluida
