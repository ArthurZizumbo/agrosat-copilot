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
    ALPHAEARTH_DIM_COLS,
    DEFAULT_S2_BANDS,
    DYNAMIC_WORLD_CLASSES,
    _cache_key,
    fetch_s2_ndvi_rgb_for_parcel,
    init_ee,
    sample_alphaearth_at_coords,
    sample_dynamic_world_at,
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
        assert (tmp_path / _cache_key("toscana", "2024-04-01", "2024-04-30", 5)).exists()

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


def test_init_ee_propagates_error_by_default() -> None:
    """Con `interactive_auth=False` (default), un fallo de Initialize debe propagarse
    sin disparar `ee.Authenticate` (evita bloqueos en papermill / CI)."""
    fake_ee = MagicMock(name="ee")
    fake_ee.Initialize = MagicMock(side_effect=RuntimeError("no creds"))
    fake_ee.Authenticate = MagicMock()
    with patch.object(gee_sampler, "ee", fake_ee):
        with pytest.raises(RuntimeError, match="no creds"):
            init_ee(project="proj")
    fake_ee.Authenticate.assert_not_called()
    fake_ee.Initialize.assert_called_once_with(project="proj")


def test_init_ee_falls_back_to_authenticate_when_interactive_flag() -> None:
    """Con `interactive_auth=True`, un fallo de Initialize dispara Authenticate
    y reintenta Initialize."""
    fake_ee = MagicMock(name="ee")
    fake_ee.Initialize = MagicMock(side_effect=[RuntimeError("no creds"), None])
    fake_ee.Authenticate = MagicMock()
    with patch.object(gee_sampler, "ee", fake_ee):
        init_ee(project="proj", interactive_auth=True)
    fake_ee.Authenticate.assert_called_once()
    assert fake_ee.Initialize.call_count == 2


def test_init_ee_skips_service_account_if_file_missing(tmp_path: Path) -> None:
    """Si `service_account_json` apunta a un archivo inexistente, debe caer a ADC
    en vez de fallar inmediatamente (placeholders del .env.local)."""
    fake_ee = MagicMock(name="ee")
    fake_ee.Initialize = MagicMock()
    fake_ee.ServiceAccountCredentials = MagicMock()
    missing = tmp_path / "nope.json"
    with patch.object(gee_sampler, "ee", fake_ee):
        init_ee(service_account_json=missing, project="proj")
    fake_ee.ServiceAccountCredentials.assert_not_called()
    fake_ee.Initialize.assert_called_once_with(project="proj")


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


# ============================================================
# Tests para funciones añadidas en US-011 (AlphaEarth / DW / S2)
# ============================================================


def _build_alphaearth_reduce_regions_ee(px_ids: list[str], n_dims: int = 64) -> MagicMock:
    """Mock del modulo `ee` para `sample_alphaearth_at_coords`.

    Simula la chain `ee.ImageCollection(...).filterDate(...).mosaic().select(...)
    .reduceRegions(collection=fc, reducer=...).getInfo()`.

    Args:
        px_ids: Lista de `px_id` esperados como features de salida con dims sinteticas.
        n_dims: Cantidad de dimensiones AlphaEarth a sintetizar (default 64).

    Returns:
        Mock del modulo `ee` listo para usar via monkeypatch / patch.object.
    """
    fake = MagicMock(name="ee")

    payload_features: list[dict[str, Any]] = []
    for i, pid in enumerate(px_ids):
        props: dict[str, Any] = {"px_id": pid}
        for j in range(n_dims):
            props[f"A{j:02d}"] = float(i * 0.1 + j * 0.01)
        payload_features.append({"properties": props})

    fake_sampled = MagicMock(name="ee.ReduceRegionsResult")
    fake_sampled.getInfo.return_value = {"features": payload_features}

    fake_image = MagicMock(name="ee.Image")
    fake_image.reduceRegions.return_value = fake_sampled
    fake_image.select.return_value = fake_image

    fake_collection = MagicMock(name="ee.ImageCollection_chain")
    fake_collection.filterDate.return_value = fake_collection
    # AlphaEarth: usar mosaic() (no first()) para unir todos los tiles del anio.
    fake_collection.mosaic.return_value = fake_image

    fake.ImageCollection.return_value = fake_collection
    # ee.Geometry.Point, ee.Feature, ee.FeatureCollection, ee.Reducer.first se
    # construyen para cada coord. Devolvemos mocks pasivos.
    fake.Geometry.Point = MagicMock(name="ee.Geometry.Point")
    fake.Feature = MagicMock(name="ee.Feature")
    fake.FeatureCollection = MagicMock(name="ee.FeatureCollection")
    fake.Reducer.first = MagicMock(name="ee.Reducer.first")
    return fake


def test_sample_alphaearth_at_coords_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mock EE devuelve 3 features con `dim_00..dim_63` sinteticas.

    Verifica esquema, height, dims completas y escritura de cache parquet.
    """
    coords = pl.DataFrame(
        {
            "px_id": ["p1", "p2", "p3"],
            "lon": [-3.5, -2.5, -1.5],
            "lat": [47.8, 48.0, 48.2],
        }
    )
    fake_ee = _build_alphaearth_reduce_regions_ee(px_ids=["p1", "p2", "p3"])
    monkeypatch.setattr(gee_sampler, "ee", fake_ee)

    out = sample_alphaearth_at_coords(
        coords, year=2019, cache_path=tmp_path, cache_key="francia_test"
    )

    assert out.height == 3
    for col in ALPHAEARTH_DIM_COLS:
        assert col in out.columns
    assert set(out["px_id"].to_list()) == {"p1", "p2", "p3"}
    assert set(out["year"].unique().to_list()) == {2019}
    # Cache parquet escrito
    cache_files = list(tmp_path.glob("alphaearth_at_francia_test_2019_*.parquet"))
    assert cache_files, "Cache parquet no escrito"


def test_sample_alphaearth_at_coords_cache_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Segunda llamada lee del cache sin re-invocar EE."""
    coords = pl.DataFrame({"px_id": ["p1", "p2"], "lon": [-3.0, -2.0], "lat": [47.5, 48.0]})
    fake_ee = _build_alphaearth_reduce_regions_ee(px_ids=["p1", "p2"])
    monkeypatch.setattr(gee_sampler, "ee", fake_ee)

    first = sample_alphaearth_at_coords(
        coords, year=2019, cache_path=tmp_path, cache_key="cache_test"
    )
    fake_ee.ImageCollection.reset_mock()
    second = sample_alphaearth_at_coords(
        coords, year=2019, cache_path=tmp_path, cache_key="cache_test"
    )
    assert first.equals(second)
    fake_ee.ImageCollection.assert_not_called()


def test_sample_alphaearth_at_coords_empty_coords(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """DataFrame de coords vacio retorna DataFrame vacio con esquema correcto."""
    coords = pl.DataFrame(schema={"px_id": pl.Utf8, "lon": pl.Float64, "lat": pl.Float64})
    fake_ee = _build_alphaearth_reduce_regions_ee(px_ids=[])
    monkeypatch.setattr(gee_sampler, "ee", fake_ee)
    out = sample_alphaearth_at_coords(coords, year=2019, cache_path=tmp_path, cache_key="vacio")
    assert out.is_empty()
    assert "dim_00" in out.columns
    assert "dim_63" in out.columns


def test_sample_alphaearth_at_coords_degraded_on_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Si EE lanza excepcion retorna DataFrame vacio con esquema valido."""
    coords = pl.DataFrame({"px_id": ["p1"], "lon": [-3.0], "lat": [47.5]})
    fake_ee = MagicMock(name="ee")
    fake_ee.Geometry.Point.side_effect = RuntimeError("simulated EE failure")
    monkeypatch.setattr(gee_sampler, "ee", fake_ee)
    out = sample_alphaearth_at_coords(coords, year=2019, cache_path=tmp_path, cache_key="fail")
    assert out.is_empty()
    assert "dim_00" in out.columns


def _build_dynamic_world_ee(features_payload: list[dict[str, Any]]) -> MagicMock:
    """Mock minimo de `ee` para `sample_dynamic_world_at`.

    Args:
        features_payload: Lista de dicts a retornar como `features` de getInfo.

    Returns:
        Mock del modulo `ee` ya cableado.
    """
    fake = MagicMock(name="ee")

    fake_sampled = MagicMock(name="ee.ReduceRegionsResult")
    fake_sampled.getInfo.return_value = {"features": features_payload}

    fake_mode = MagicMock(name="ee.Image.mode")
    fake_mode.reduceRegions.return_value = fake_sampled

    fake_collection = MagicMock()
    fake_collection.filterDate.return_value = fake_collection
    fake_collection.select.return_value = fake_collection
    fake_collection.mode.return_value = fake_mode

    fake.ImageCollection.return_value = fake_collection
    fake.Geometry.Point = MagicMock()
    fake.Feature = MagicMock()
    fake.FeatureCollection = MagicMock()
    fake.Reducer.first = MagicMock()
    return fake


def test_sample_dynamic_world_at_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Mock DW retorna 3 clases distintas. Verifica mapeo `dw_class_name`."""
    coords = pl.DataFrame(
        {"px_id": ["a", "b", "c"], "lon": [9.5, 9.6, 9.7], "lat": [45.2, 45.3, 45.4]}
    )
    payload = [
        {"properties": {"px_id": "a", "first": 4, "confidence": 0.9}},
        {"properties": {"px_id": "b", "first": 1, "confidence": 0.85}},
        {"properties": {"px_id": "c", "first": 7, "confidence": 0.7}},
    ]
    fake_ee = _build_dynamic_world_ee(payload)
    monkeypatch.setattr(gee_sampler, "ee", fake_ee)
    out = sample_dynamic_world_at(coords, year=2024, cache_path=tmp_path, cache_key="italia_dw")
    assert out.height == 3
    assert set(out["dw_class_id"].to_list()) == {4, 1, 7}
    # `crops` (4), `trees` (1), `bare` (7) deben aparecer
    names = set(out["dw_class_name"].to_list())
    assert {"crops", "trees", "bare"}.issubset(names)
    # Esquema completo y cache escrito
    cache_files = list(tmp_path.glob("dw_at_italia_dw_2024_*.parquet"))
    assert cache_files


def test_sample_dynamic_world_at_cache_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Segunda llamada lee del cache parquet sin re-invocar EE."""
    coords = pl.DataFrame({"px_id": ["a"], "lon": [9.5], "lat": [45.2]})
    payload = [{"properties": {"px_id": "a", "first": 4, "confidence": 0.9}}]
    fake_ee = _build_dynamic_world_ee(payload)
    monkeypatch.setattr(gee_sampler, "ee", fake_ee)
    first = sample_dynamic_world_at(coords, year=2024, cache_path=tmp_path, cache_key="round")
    fake_ee.ImageCollection.reset_mock()
    second = sample_dynamic_world_at(coords, year=2024, cache_path=tmp_path, cache_key="round")
    assert first.equals(second)
    fake_ee.ImageCollection.assert_not_called()


def test_sample_dynamic_world_at_unknown_class_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Clase fuera del catalogo (-1 / 99) debe mapearse a `unknown`."""
    coords = pl.DataFrame({"px_id": ["a", "b"], "lon": [9.5, 9.6], "lat": [45.2, 45.3]})
    payload = [
        {"properties": {"px_id": "a", "first": 99, "confidence": 0.5}},
        {"properties": {"px_id": "b", "first": None}},
    ]
    fake_ee = _build_dynamic_world_ee(payload)
    monkeypatch.setattr(gee_sampler, "ee", fake_ee)
    out = sample_dynamic_world_at(coords, year=2024, cache_path=tmp_path, cache_key="unk")
    # Sin escribir cache: la lista contiene la clase pero mapeada
    names = out["dw_class_name"].to_list()
    assert "unknown" in names


def test_sample_dynamic_world_at_degraded_on_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Excepcion en EE -> filas marcadas con `class_id=-1` / `unknown`.

    Diseno: el sampler procesa coords en batches. Si un batch falla, anota
    sus puntos como `unknown` para preservar cardinalidad en el join downstream
    (un DataFrame vacio rompe el join y deja todas las filas null).
    """
    coords = pl.DataFrame({"px_id": ["a"], "lon": [9.5], "lat": [45.2]})
    fake_ee = MagicMock()
    fake_ee.Geometry.Point.side_effect = RuntimeError("EE down")
    monkeypatch.setattr(gee_sampler, "ee", fake_ee)
    out = sample_dynamic_world_at(coords, year=2024, cache_path=tmp_path, cache_key="x")
    assert out.height == 1
    assert out["dw_class_id"].to_list() == [-1]
    assert out["dw_class_name"].to_list() == ["unknown"]


def test_sample_dynamic_world_at_batches_large_coords(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Con >batch_size coords, `reduceRegions` se llama varias veces (1 por batch).

    Regresion guard: antes la funcion enviaba 6000 puntos en una sola request
    server-side, GEE colgaba el compute graph y retornaba payload vacio ->
    todas las filas downstream caian a `dw_class_name=null`.
    """
    n = 1200  # > batch_size default (500) -> 3 batches
    coords = pl.DataFrame(
        {
            "px_id": [f"p_{i}" for i in range(n)],
            "lon": [9.5 + (i % 10) * 0.01 for i in range(n)],
            "lat": [45.2 + (i % 10) * 0.01 for i in range(n)],
        }
    )
    # `side_effect` con tres respuestas: el mock devuelve un payload distinto
    # por cada batch (mismos px_id que el slice correspondiente).
    payloads = []
    for start in range(0, n, 500):
        chunk_ids = [f"p_{i}" for i in range(start, min(start + 500, n))]
        payloads.append(
            {"features": [{"properties": {"px_id": pid, "first": 4}} for pid in chunk_ids]}
        )

    fake_ee = _build_dynamic_world_ee([])
    reduce_regions_mock = (
        fake_ee.ImageCollection.return_value.select.return_value.mode.return_value.reduceRegions
    )
    reduce_regions_mock.return_value.getInfo.side_effect = payloads
    monkeypatch.setattr(gee_sampler, "ee", fake_ee)

    out = sample_dynamic_world_at(
        coords, year=2024, cache_path=tmp_path, cache_key="batched", batch_size=500
    )
    assert out.height == n
    assert set(out["dw_class_name"].unique().to_list()) == {"crops"}
    # Tres batches -> tres llamadas a getInfo
    assert (
        fake_ee.ImageCollection.return_value.select.return_value.mode.return_value.reduceRegions.return_value.getInfo.call_count
        == 3
    )


def test_dynamic_world_classes_canonical() -> None:
    """El catalogo de 9 clases LULC esta completo y bien nombrado."""
    assert len(DYNAMIC_WORLD_CLASSES) == 9
    assert DYNAMIC_WORLD_CLASSES[4] == "crops"
    assert DYNAMIC_WORLD_CLASSES[0] == "water"
    assert DYNAMIC_WORLD_CLASSES[8] == "snow_and_ice"


def _build_fetch_s2_ee(b2: list, b3: list, b4: list, b8: list) -> MagicMock:
    """Mock minimo de `ee` para `fetch_s2_ndvi_rgb_for_parcel`.

    Simula la chain `ImageCollection(...).filterBounds(...).filterDate(...)
    .filter(...).select(...)` para extraer la proyeccion de referencia via
    `.first().select("B4").projection()` y luego `.median().reproject(...)
    .sampleRectangle(...).getInfo()`.
    """
    fake = MagicMock(name="ee")
    fake_rect = MagicMock(name="ee.SampleRectangle")
    fake_rect.getInfo.return_value = {"properties": {"B2": b2, "B3": b3, "B4": b4, "B8": b8}}
    fake_reprojected = MagicMock(name="ee.Image.reprojected")
    fake_reprojected.sampleRectangle.return_value = fake_rect
    fake_median = MagicMock(name="ee.Image.median")
    fake_median.reproject.return_value = fake_reprojected
    # Chain para ref_proj = collection.first().select("B4").projection()
    fake_first = MagicMock(name="ee.Image.first")
    fake_first.select.return_value = fake_first
    fake_first.projection.return_value = MagicMock(name="ee.Projection")
    fake_collection = MagicMock()
    fake_collection.filterBounds.return_value = fake_collection
    fake_collection.filterDate.return_value = fake_collection
    fake_collection.filter.return_value = fake_collection
    fake_collection.select.return_value = fake_collection
    fake_collection.median.return_value = fake_median
    fake_collection.first.return_value = fake_first
    fake.ImageCollection.return_value = fake_collection
    fake.Filter.lt.return_value = MagicMock()
    return fake


def test_fetch_s2_ndvi_rgb_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock S2 entrega 4 bandas 2x2; NDVI y RGB se calculan con shape correcta."""
    b2 = [[500.0, 600.0], [550.0, 650.0]]
    b3 = [[700.0, 800.0], [750.0, 850.0]]
    b4 = [[1000.0, 1100.0], [1050.0, 1150.0]]
    b8 = [[3000.0, 3100.0], [3050.0, 3150.0]]
    fake_ee = _build_fetch_s2_ee(b2, b3, b4, b8)
    monkeypatch.setattr(gee_sampler, "ee", fake_ee)
    out = fetch_s2_ndvi_rgb_for_parcel(MagicMock(), "2024-06-15")
    assert out["rgb"].shape == (2, 2, 3)
    assert out["ndvi"].shape == (2, 2)
    # NDVI esperado positivo para vegetacion (B8 >> B4)
    assert (out["ndvi"] > 0.0).all()
    # RGB normalizado al rango [0, 1]
    assert out["rgb"].min() >= 0.0
    assert out["rgb"].max() <= 1.0
    assert out["date_used"] == "2024-06-15"


def test_fetch_s2_ndvi_rgb_empty_bands_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Si EE retorna arrays vacios, la funcion devuelve los arrays vacios sin error."""
    fake_ee = _build_fetch_s2_ee([], [], [], [])
    monkeypatch.setattr(gee_sampler, "ee", fake_ee)
    out = fetch_s2_ndvi_rgb_for_parcel(MagicMock(), "2024-06-15")
    assert out["rgb"].size == 0
    assert out["ndvi"].size == 0


def test_fetch_s2_ndvi_rgb_handles_ee_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Excepcion en EE -> dict vacio con keys garantizadas."""
    fake_ee = MagicMock()
    fake_ee.ImageCollection.side_effect = RuntimeError("quota")
    fake_ee.Filter.lt.return_value = MagicMock()
    monkeypatch.setattr(gee_sampler, "ee", fake_ee)
    out = fetch_s2_ndvi_rgb_for_parcel(MagicMock(), "2024-06-15")
    assert out["rgb"].size == 0
    assert out["ndvi"].size == 0
    assert out["date_used"] == ""
