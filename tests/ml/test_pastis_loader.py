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
    pastis_patch_coords,
    pastis_patch_index,
    pastis_pixel_labels,
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


# ============================================================
# Tests para funciones añadidas en US-011
# ============================================================


def _write_pastis_metadata_geojson(
    out_path: Path,
    patch_id: str = "10000",
    bbox: tuple[float, float, float, float] = (
        391500.0,
        6955430.0,
        391500.0 + 1280.0,
        6955430.0 + 1280.0,
    ),
    fold: int = 1,
    tile: str = "T30UXV",
) -> None:
    """Genera un `metadata.geojson` minimo con 1 feature MultiPolygon en EPSG:2154.

    Args:
        out_path: Ruta destino del archivo.
        patch_id: Identificador del patch.
        bbox: Bbox `(xmin, ymin, xmax, ymax)` en EPSG:2154 (Lambert-93).
        fold: Numero de fold PASTIS (1..5).
        tile: Tile S2 asociado.
    """
    xmin, ymin, xmax, ymax = bbox
    ring = [
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
        [xmin, ymin],
    ]
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "id": patch_id,
                "type": "Feature",
                "properties": {
                    "ID_PATCH": int(patch_id),
                    "TILE": tile,
                    "Fold": fold,
                },
                "geometry": {"type": "MultiPolygon", "coordinates": [[ring]]},
            }
        ],
    }
    out_path.write_text(json.dumps(geojson), encoding="utf-8")


def test_pastis_patch_coords_missing_file_returns_empty(tmp_path: Path) -> None:
    """Sin metadata.geojson retorna DataFrame vacio con esquema canonico."""
    out = pastis_patch_coords(tmp_path / "no_existe.geojson")
    assert out.is_empty()
    assert {"patch_id", "lon", "lat", "tile", "fold"}.issubset(set(out.columns))


def test_pastis_patch_coords_reprojection_2154_to_4326(tmp_path: Path) -> None:
    """Reproyecta centroide de patch arbitrario EPSG:2154 -> EPSG:4326.

    Se construye un MultiPolygon en bbox conocido `[391500, 6955430,
    391500+1280, 6955430+1280]` EPSG:2154 (Lambert-93). El centroide
    `(392140, 6956070)` debe mapear aprox a `lon~-3.32, lat~47.83`
    en EPSG:4326 (territorio breton, Francia). Tolerancia 1e-3 grados.
    """
    pytest.importorskip("pyproj")
    from pyproj import Transformer

    path = tmp_path / "metadata.geojson"
    _write_pastis_metadata_geojson(path)

    # Coord esperada: transformar el centroide del bbox conocido
    xmin, ymin = 391500.0, 6955430.0
    xmax, ymax = xmin + 1280.0, ymin + 1280.0
    cx_exp, cy_exp = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    lon_exp, lat_exp = transformer.transform(cx_exp, cy_exp)

    out = pastis_patch_coords(path)
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["patch_id"] == "10000"
    assert row["tile"] == "T30UXV"
    assert row["fold"] == 1
    # Tolerancia 1e-3 grados
    assert abs(row["lon"] - lon_exp) < 1e-3
    assert abs(row["lat"] - lat_exp) < 1e-3
    # Sanity check: territorio frances continental
    assert -6.0 < row["lon"] < 8.0
    assert 41.0 < row["lat"] < 51.5


def test_pastis_patch_coords_skips_non_multipolygon(tmp_path: Path) -> None:
    """Features cuya geometria no es MultiPolygon se ignoran sin error."""
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "id": "p_bad",
                "type": "Feature",
                "properties": {"ID_PATCH": 1, "TILE": "X", "Fold": 1},
                "geometry": {"type": "Point", "coordinates": [391500.0, 6955430.0]},
            }
        ],
    }
    path = tmp_path / "metadata.geojson"
    path.write_text(json.dumps(geojson), encoding="utf-8")
    out = pastis_patch_coords(path)
    assert out.is_empty()


def test_pastis_patch_coords_empty_feature_collection(tmp_path: Path) -> None:
    """FeatureCollection sin features retorna DataFrame vacio."""
    path = tmp_path / "metadata.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    out = pastis_patch_coords(path)
    assert out.is_empty()
    assert {"patch_id", "lon", "lat", "tile", "fold"}.issubset(set(out.columns))


def test_pastis_pixel_labels_missing_target_returns_empty(tmp_path: Path) -> None:
    """TARGET_*.npy inexistente retorna DataFrame vacio con esquema."""
    out = pastis_pixel_labels("999999", root=tmp_path)
    assert out.is_empty()
    expected = {"px_id", "patch_id", "lon", "lat", "class_id", "class_name"}
    assert expected.issubset(set(out.columns))


def test_pastis_pixel_labels_filters_default_classes(tmp_path: Path) -> None:
    """Por defecto excluye clase 0 (background) y 19 (void).

    Fixture: TARGET_10000.npy con canal 0 conteniendo solo clases {0, 5, 19}.
    Tras filtrar `(0, 19)` solo debe quedar la clase 5.
    """
    ann_dir = tmp_path / "ANNOTATIONS"
    ann_dir.mkdir()
    sem = np.zeros((128, 128), dtype=np.uint8)
    # 1/3 con clase 5, resto mezclado entre 0 y 19
    sem[:, :42] = 5
    sem[:, 42:85] = 0
    sem[:, 85:] = 19
    target = np.stack([sem, np.zeros_like(sem), np.zeros_like(sem)], axis=0)
    np.save(ann_dir / "TARGET_10000.npy", target)

    # metadata.geojson valido para georreferenciar
    _write_pastis_metadata_geojson(tmp_path / "metadata.geojson")

    out = pastis_pixel_labels("10000", root=tmp_path)
    assert not out.is_empty()
    cls = out["class_id"].to_numpy()
    # Filtrado por defecto -> solo clase 5
    assert (cls == 5).all()
    # class_name resuelta (no "unknown") porque 5 esta en el mapping canonico
    assert "unknown" not in set(out["class_name"].to_list())


def test_pastis_pixel_labels_custom_exclude(tmp_path: Path) -> None:
    """`exclude_classes=()` desactiva el filtro -> incluye 0 y 19."""
    ann_dir = tmp_path / "ANNOTATIONS"
    ann_dir.mkdir()
    sem = np.zeros((128, 128), dtype=np.uint8)
    sem[:64, :] = 0
    sem[64:, :] = 19
    target = np.stack([sem, np.zeros_like(sem), np.zeros_like(sem)], axis=0)
    np.save(ann_dir / "TARGET_42.npy", target)
    _write_pastis_metadata_geojson(tmp_path / "metadata.geojson", patch_id="42")

    out = pastis_pixel_labels("42", root=tmp_path, exclude_classes=())
    assert not out.is_empty()
    classes = set(out["class_id"].to_list())
    assert 0 in classes and 19 in classes


def test_pastis_pixel_labels_sample_per_patch(tmp_path: Path) -> None:
    """`sample_per_patch=N` muestrea exactamente N pixeles aleatorios y es reproducible."""
    ann_dir = tmp_path / "ANNOTATIONS"
    ann_dir.mkdir()
    rng = np.random.default_rng(123)
    sem = rng.integers(low=1, high=19, size=(128, 128), dtype=np.uint8)
    target = np.stack([sem, np.zeros_like(sem), np.zeros_like(sem)], axis=0)
    np.save(ann_dir / "TARGET_77.npy", target)
    _write_pastis_metadata_geojson(tmp_path / "metadata.geojson", patch_id="77")

    n = 250
    out1 = pastis_pixel_labels("77", root=tmp_path, sample_per_patch=n, seed=7)
    out2 = pastis_pixel_labels("77", root=tmp_path, sample_per_patch=n, seed=7)
    assert out1.height == n
    assert out2.height == n
    # Reproducible con misma seed
    assert out1["px_id"].to_list() == out2["px_id"].to_list()


def test_pastis_pixel_labels_sample_larger_than_population(tmp_path: Path) -> None:
    """Si `sample_per_patch > N_pixels_filtrados` retorna todos disponibles."""
    ann_dir = tmp_path / "ANNOTATIONS"
    ann_dir.mkdir()
    sem = np.zeros((128, 128), dtype=np.uint8)
    sem[0, 0] = 7
    sem[0, 1] = 8
    sem[0, 2] = 9
    # Resto = 0 (filtrado por default)
    target = np.stack([sem, np.zeros_like(sem), np.zeros_like(sem)], axis=0)
    np.save(ann_dir / "TARGET_55.npy", target)
    _write_pastis_metadata_geojson(tmp_path / "metadata.geojson", patch_id="55")

    out = pastis_pixel_labels("55", root=tmp_path, sample_per_patch=1000)
    # Solo hay 3 pixeles validos tras filtrar 0; sample > N -> devuelve los 3
    assert out.height == 3
    assert set(out["class_id"].to_list()) == {7, 8, 9}


def test_pastis_pixel_labels_coords_within_bbox_4326(tmp_path: Path) -> None:
    """Coordenadas (lon, lat) generadas caen dentro del bbox reproyectado a 4326."""
    pytest.importorskip("pyproj")
    from pyproj import Transformer

    ann_dir = tmp_path / "ANNOTATIONS"
    ann_dir.mkdir()
    sem = np.full((128, 128), fill_value=5, dtype=np.uint8)
    target = np.stack([sem, np.zeros_like(sem), np.zeros_like(sem)], axis=0)
    np.save(ann_dir / "TARGET_10000.npy", target)

    _write_pastis_metadata_geojson(tmp_path / "metadata.geojson")

    out = pastis_pixel_labels("10000", root=tmp_path, sample_per_patch=200)
    assert not out.is_empty()

    transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    lon1, lat1 = transformer.transform(391500.0, 6955430.0)
    lon2, lat2 = transformer.transform(391500.0 + 1280.0, 6955430.0 + 1280.0)
    lon_min, lon_max = min(lon1, lon2), max(lon1, lon2)
    lat_min, lat_max = min(lat1, lat2), max(lat1, lat2)
    tol = 1e-3
    lons = out["lon"].to_numpy()
    lats = out["lat"].to_numpy()
    assert (lons >= lon_min - tol).all() and (lons <= lon_max + tol).all()
    assert (lats >= lat_min - tol).all() and (lats <= lat_max + tol).all()


def test_pastis_patch_index_missing_file_returns_empty(tmp_path: Path) -> None:
    """Sin metadata.geojson retorna DataFrame vacio con schema correcto."""
    out = pastis_patch_index(tmp_path / "missing.geojson")
    assert out.is_empty()
    assert set(out.columns) == {"patch_id", "TILE", "Fold"}


def test_pastis_patch_index_parses_geojson(tmp_path: Path) -> None:
    """Lee correctamente patch_id, TILE y Fold del metadata."""
    meta = tmp_path / "metadata.geojson"
    _write_pastis_metadata_geojson(meta, patch_id="10000", tile="T30UXV", fold=3)

    out = pastis_patch_index(meta)
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["patch_id"] == "10000"
    assert row["TILE"] == "T30UXV"
    assert row["Fold"] == 3


def test_pastis_patch_index_schema(tmp_path: Path) -> None:
    """El schema retornado es estable: patch_id str, TILE str, Fold int64."""
    meta = tmp_path / "metadata.geojson"
    _write_pastis_metadata_geojson(meta)
    out = pastis_patch_index(meta)
    assert str(out.schema["patch_id"]) == "String"
    assert str(out.schema["TILE"]) == "String"
    assert str(out.schema["Fold"]) == "Int64"


def test_pastis_pixel_labels_all_excluded_returns_empty(tmp_path: Path) -> None:
    """Si todos los pixeles caen en clases excluidas retorna DataFrame vacio."""
    ann_dir = tmp_path / "ANNOTATIONS"
    ann_dir.mkdir()
    # 100% background
    sem = np.zeros((128, 128), dtype=np.uint8)
    target = np.stack([sem, np.zeros_like(sem), np.zeros_like(sem)], axis=0)
    np.save(ann_dir / "TARGET_91.npy", target)
    _write_pastis_metadata_geojson(tmp_path / "metadata.geojson", patch_id="91")
    out = pastis_pixel_labels("91", root=tmp_path)
    assert out.is_empty()
    assert "class_name" in out.columns
