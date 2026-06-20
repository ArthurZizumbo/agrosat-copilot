"""Tests for the qualitative Mexico demo (US-077).

Mocks the ``ee`` module and the Gemini client (``set_llm_client``); ZERO real
calls in CI. The meta-test ``test_no_classifier_no_f1`` enforces the hard rule
that the module imports no classification metric over Mexico.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from ml.features import phenology_description
from ml.transfer import mexico_demo as mx

# ---------------------------------------------------------------------------
# Fake ``ee`` builders (shape-only, no scientific data).
# ---------------------------------------------------------------------------


class _FakeReduceResult:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def getInfo(self) -> dict[str, Any]:
        return self._payload


class _FakeImage:
    def __init__(self, reduce_payload: dict[str, Any]) -> None:
        self._reduce_payload = reduce_payload

    def select(self, *_args: Any, **_kwargs: Any) -> _FakeImage:
        return self

    def reduceRegion(self, **_kwargs: Any) -> _FakeReduceResult:
        return _FakeReduceResult(self._reduce_payload)


class _FakeImageCollection:
    """Records the chained calls and yields a mosaic image / mapped features."""

    def __init__(self, reduce_payload: dict[str, Any], feature_payload: dict[str, Any]) -> None:
        self._reduce_payload = reduce_payload
        self._feature_payload = feature_payload

    def filterDate(self, *_a: Any, **_k: Any) -> _FakeImageCollection:
        return self

    def filterBounds(self, *_a: Any, **_k: Any) -> _FakeImageCollection:
        return self

    def filter(self, *_a: Any, **_k: Any) -> _FakeImageCollection:
        return self

    def map(self, _fn: Any) -> _FakeImageCollection:
        return _FakeMappedCollection(self._feature_payload)

    def mosaic(self) -> _FakeImage:
        return _FakeImage(self._reduce_payload)


class _FakeMappedCollection:
    def __init__(self, feature_payload: dict[str, Any]) -> None:
        self._feature_payload = feature_payload

    def map(self, _fn: Any) -> _FakeMappedCollection:
        return self

    def getInfo(self) -> dict[str, Any]:
        return self._feature_payload


class _FakeGeometryPoint:
    def __init__(self) -> None:
        self.buffered_with: int | None = None

    def buffer(self, buffer_m: int) -> _FakeGeometryPoint:
        self.buffered_with = buffer_m
        return self


class _FakeGeometry:
    def __init__(self) -> None:
        self.last_point: _FakeGeometryPoint | None = None
        self.point_coords: list[float] | None = None

    def Point(self, coords: list[float]) -> _FakeGeometryPoint:
        self.point_coords = coords
        self.last_point = _FakeGeometryPoint()
        return self.last_point


class _FakeReducer:
    @staticmethod
    def mean() -> str:
        return "mean"


class _FakeFilter:
    @staticmethod
    def lt(*_a: Any, **_k: Any) -> str:
        return "lt"


class _FakeFeature:
    def __init__(self, *_a: Any, **_k: Any) -> None:
        pass


class _FakeEE:
    def __init__(self, reduce_payload: dict[str, Any], feature_payload: dict[str, Any]) -> None:
        self._reduce_payload = reduce_payload
        self._feature_payload = feature_payload
        self.Geometry = _FakeGeometry()
        self.Reducer = _FakeReducer()
        self.Filter = _FakeFilter()
        self.Feature = _FakeFeature

    def ImageCollection(self, _name: str) -> _FakeImageCollection:
        return _FakeImageCollection(self._reduce_payload, self._feature_payload)


@pytest.fixture
def aoi() -> mx.MexicoAOI:
    return mx.DEFAULT_AOIS[0]


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_default_aois_coords_real() -> None:
    by_crop = {a.crop: a for a in mx.DEFAULT_AOIS}
    assert "aguacate" in by_crop and "guayaba" in by_crop
    avocado = by_crop["aguacate"]
    guava = by_crop["guayaba"]
    assert (avocado.lon, avocado.lat) == pytest.approx((-102.05, 19.41))
    assert (guava.lon, guava.lat) == pytest.approx((-102.72, 21.85))
    assert all(a.buffer_m > 0 for a in mx.DEFAULT_AOIS)


def test_aoi_geometry_builds_buffer(monkeypatch: pytest.MonkeyPatch, aoi: mx.MexicoAOI) -> None:
    fake = _FakeEE({}, {})
    monkeypatch.setattr(mx, "ee", fake)
    geom = mx.aoi_geometry(aoi)
    assert fake.Geometry.point_coords == [aoi.lon, aoi.lat]
    assert isinstance(geom, _FakeGeometryPoint)
    assert geom.buffered_with == aoi.buffer_m


def test_extract_alphaearth_zonal_shape(
    monkeypatch: pytest.MonkeyPatch, aoi: mx.MexicoAOI, tmp_path: Path
) -> None:
    payload = {f"A{i:02d}": float(i) / 100.0 for i in range(mx.ALPHAEARTH_N_DIMS)}
    monkeypatch.setattr(mx, "ee", _FakeEE(payload, {}))
    vec = mx.extract_alphaearth_zonal(aoi, 2023, cache_dir=tmp_path)
    assert vec.shape == (64,)
    assert vec.dtype == np.float64


def test_extract_alphaearth_zonal_degraded(
    monkeypatch: pytest.MonkeyPatch, aoi: mx.MexicoAOI, tmp_path: Path
) -> None:
    # Missing bands -> degraded empty array with valid dtype.
    monkeypatch.setattr(mx, "ee", _FakeEE({"A00": 0.1}, {}))
    vec = mx.extract_alphaearth_zonal(aoi, 2023, cache_dir=tmp_path)
    assert vec.shape == (0,)
    assert vec.dtype == np.float64


def test_extract_alphaearth_zonal_no_ee(
    monkeypatch: pytest.MonkeyPatch, aoi: mx.MexicoAOI, tmp_path: Path
) -> None:
    monkeypatch.setattr(mx, "ee", None)
    vec = mx.extract_alphaearth_zonal(aoi, 2023, cache_dir=tmp_path)
    assert vec.shape == (0,)
    assert vec.dtype == np.float64


def test_extract_s2_ndvi_series_schema(
    monkeypatch: pytest.MonkeyPatch, aoi: mx.MexicoAOI, tmp_path: Path
) -> None:
    feature_payload = {
        "features": [
            {"properties": {"date": "2023-01-04", "ndvi": 0.2}},
            {"properties": {"date": "2023-06-15", "ndvi": 0.75}},
            {"properties": {"date": "2023-11-20", "ndvi": 0.68}},
        ]
    }
    monkeypatch.setattr(mx, "ee", _FakeEE({}, feature_payload))
    frame = mx.extract_s2_ndvi_series(aoi, 2023, cache_dir=tmp_path)
    assert frame.columns == ["date", "doy", "ndvi"]
    assert frame.height == 3
    assert frame["doy"].dtype == pl.Int64
    assert frame["ndvi"].min() >= -1.0 and frame["ndvi"].max() <= 1.0
    # sorted by date and doy derived correctly (Jan 4 -> doy 4).
    assert frame.row(0) == ("2023-01-04", 4, 0.2)


def test_extract_s2_ndvi_series_degraded(
    monkeypatch: pytest.MonkeyPatch, aoi: mx.MexicoAOI, tmp_path: Path
) -> None:
    monkeypatch.setattr(mx, "ee", _FakeEE({}, {"features": []}))
    frame = mx.extract_s2_ndvi_series(aoi, 2023, cache_dir=tmp_path)
    assert frame.is_empty()
    assert frame.columns == ["date", "doy", "ndvi"]


def test_describe_phenology_delegates(
    monkeypatch: pytest.MonkeyPatch, aoi: mx.MexicoAOI, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    def _stub(prompt: str, *, model: str, temperature: float) -> str:
        seen["model"] = model
        seen["temperature"] = temperature
        seen["prompt"] = prompt
        return "Descripcion fenologica de prueba."

    phenology_description.set_llm_client(_stub)
    try:
        series = pl.DataFrame(
            {
                "date": ["2023-01-04", "2023-06-15"],
                "doy": [4, 166],
                "ndvi": [0.6, 0.7],
            }
        )
        out = mx.describe_phenology(series, aoi, cache_dir=tmp_path)
        assert out == "Descripcion fenologica de prueba."
        assert seen["temperature"] == 0.0
        # crop_type_hint must reach the prompt (block 2).
        assert aoi.crop_type_hint in seen["prompt"]
    finally:
        phenology_description.set_llm_client(None)


def test_describe_phenology_empty_raises(aoi: mx.MexicoAOI) -> None:
    series = pl.DataFrame(schema={"date": pl.Utf8, "doy": pl.Int64, "ndvi": pl.Float64})
    with pytest.raises(ValueError, match="empty"):
        mx.describe_phenology(series, aoi)


def test_hcat_framing_is_perennial() -> None:
    framing = mx.hcat_perennial_framing()
    assert framing  # non-empty
    assert all(v.startswith("PERMANENT_WOODY|") for v in framing.values())
    groups = {v.split("|")[1] for v in framing.values()}
    assert "orchard" in groups
    assert "vineyard" in groups


def test_no_classifier_no_f1() -> None:
    """The module must NOT import any classification metric over Mexico."""
    source = Path(mx.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
    forbidden = {"f1_score", "accuracy_score", "classification_report", "classify"}
    assert not (imported & forbidden), f"forbidden import found: {imported & forbidden}"
    # Defensive: no bare f1/accuracy tokens used as calls in the source.
    for token in ("f1_score", "accuracy_score", "build_estimator"):
        assert token not in source, f"forbidden token {token!r} present in module"
